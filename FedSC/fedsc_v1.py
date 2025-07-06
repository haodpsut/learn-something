#!/usr/bin/env python
"""fedsc_scenarios.py – v4 FULL (bug‑fix NID2 empty client)
===========================================================
Self‑contained script that (1) partitions datasets into five scenarios from the
FedSC paper and (2) trains a minimal FedSC implementation on a single machine.

Changes in v4
-------------
* **Fix NID2**: ensure no client receives zero samples (previous bug caused
  ValueError in DataLoader).
* Minor: if any client is still empty during training, skip with warning rather
  than crashing.

Usage quick‑start
-----------------
```bash
pip install torch torchvision tqdm
# Example NID2 (10 clients)
python fedsc_scenarios.py --scenario nid2 --dataset cifar10 --num_clients 10 \
       --rounds 80 --local_epochs 5 --batch_size 64 --train
```
For full paper experiments, see the `run_all.sh` in repo root.
"""
from __future__ import annotations
import argparse, json, random, math, os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# PARTITION HELPERS
# --------------------------------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def dirichlet_split(class_indices: List[List[int]], num_clients: int, alpha: float) -> Dict[int, List[int]]:
    """Return mapping client_id -> list[index] using Dirichlet distribution."""
    client_dict = {cid: [] for cid in range(num_clients)}
    for cls_idx in class_indices:
        if len(cls_idx) == 0:
            continue
        proportions = torch.distributions.dirichlet.Dirichlet(torch.full((num_clients,), alpha)).sample().tolist()
        counts = [int(p * len(cls_idx)) for p in proportions]
        # adjust to match total
        diff = len(cls_idx) - sum(counts)
        for _ in range(diff):
            counts[random.randrange(num_clients)] += 1
        random.shuffle(cls_idx)
        offset = 0
        for cid, c in enumerate(counts):
            client_dict[cid].extend(cls_idx[offset:offset + c])
            offset += c
    return client_dict


def partition_iid(dataset, num_clients):
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    split = [idxs[i::num_clients] for i in range(num_clients)]
    return {cid: split[cid] for cid in range(num_clients)}


def partition_nid1(dataset, num_clients, alpha):
    # Dirichlet across *all* classes
    class_indices = [[] for _ in range(len(dataset.classes))]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    return dirichlet_split(class_indices, num_clients, alpha)


def partition_nid2(dataset, num_clients, subset_classes_per_client=2, alpha=0.3):
    """Extreme label‑shift:
    * one *rich* client sees all classes
    * the rest (num_clients-1) each see exactly `subset_classes_per_client` classes
    """
    num_classes = len(dataset.classes)
    rich_client = 0
    restricted_clients = [c for c in range(num_clients) if c != rich_client]
    # assign a *distinct* subset to each restricted client (wrap around if needed)
    class_cycle = list(range(num_classes)) * math.ceil(len(restricted_clients) * subset_classes_per_client / num_classes)
    random.shuffle(class_cycle)
    client_allowed: Dict[int, set] = {rich_client: set(range(num_classes))}
    for i, cid in enumerate(restricted_clients):
        start = i * subset_classes_per_client
        client_allowed[cid] = set(class_cycle[start:start + subset_classes_per_client])
    # per‑class buckets
    class_buckets = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_buckets[label].append(idx)
    # allocate via Dirichlet among clients *that have this class* (+ rich)
    mapping = {cid: [] for cid in range(num_clients)}
    for c in range(num_classes):
        clients_for_c = [cid for cid in range(num_clients) if c in client_allowed[cid]]
        k = len(clients_for_c)
        if k == 0:
            continue  # should not happen
        props = torch.distributions.dirichlet.Dirichlet(torch.full((k,), alpha)).sample().tolist()
        counts = [max(1, int(p * len(class_buckets[c]))) for p in props]  # guarantee ≥1
        diff = len(class_buckets[c]) - sum(counts)
        for _ in range(abs(diff)):
            counts[_ % k] += 1 if diff > 0 else -1
        random.shuffle(class_buckets[c])
        offset = 0
        for cid, cnt in zip(clients_for_c, counts):
            mapping[cid].extend(class_buckets[c][offset:offset + cnt])
            offset += cnt
    # ensure no client empty
    empty = [cid for cid, lst in mapping.items() if len(lst) == 0]
    while empty:
        donor = max(mapping, key=lambda x: len(mapping[x]))
        take = mapping[donor].pop()
        cid_empty = empty.pop()
        mapping[cid_empty].append(take)
    return mapping


def partition_longtail(dataset, num_clients, rho, alpha):
    num_classes = len(dataset.classes)
    # create long‑tail class sizes
    img_per_cls = []
    max_samples = len(dataset) // num_classes
    for j in range(num_classes):
        img_per_cls.append(int(max_samples * (rho ** (-j / (num_classes - 1)))))
    # trim dataset according to long‑tail
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < img_per_cls[label]:
            class_indices[label].append(idx)
    return dirichlet_split(class_indices, num_clients, alpha)


def partition_fewshot(dataset, num_clients, k_shot):
    class_indices = [[] for _ in range(len(dataset.classes))]
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < k_shot * num_clients:
            class_indices[label].append(idx)
    mapping = {cid: [] for cid in range(num_clients)}
    for c, idxs in enumerate(class_indices):
        random.shuffle(idxs)
        for cid in range(num_clients):
            mapping[cid].append(idxs[cid])  # 1 image per class => k_shot=1
    return mapping


def save_partition_json(mapping: Dict[int, List[int]], out_path: Path, dataset):
    data = {}
    for cid, idxs in mapping.items():
        labels = [dataset[i][1] for i in idxs]
        data[f"client_{cid}"] = {"indices": idxs, "labels": labels}
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"[INFO] Partition mapping saved to {out_path}")

# --------------------------------------------------------------------------------------
# MODEL & LOSS
# --------------------------------------------------------------------------------------

class ResNet18Embed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])  # remove FC
        self.embed_dim = 512
        self.proj = nn.Linear(512, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)
    def forward(self, x):
        f = self.features(x).flatten(1)
        z = F.normalize(self.proj(f), dim=1)
        logits = self.classifier(z)
        return logits, z


def rpcl_loss(z, y, proto, tau):
    """Relational Prototypes Contrastive Loss (simple version)."""
    sim = torch.mm(z, proto.t()) / tau        # [B, C]
    return F.cross_entropy(sim, y)


def cpdr_loss(z, proto):
    return ((z - proto[y]).pow(2).mean())

# --------------------------------------------------------------------------------------
# TRAINER
# --------------------------------------------------------------------------------------

def aggregate_prototypes(proto_list, count_list):
    proto_sum = torch.stack(proto_list).sum(0)   # [C, d]
    count_sum = torch.stack(count_list).sum(0)   # [C, 1]
    global_proto = proto_sum / torch.clamp(count_sum, min=1e-8)
    empty_mask = (count_sum.squeeze(-1) == 0)
    global_proto[empty_mask] = 0
    return global_proto


def train_fedsc(args, part_path: Path, num_classes):
    print("[INFO] ⏳ Starting FedSC training…")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mapping = json.load(open(part_path))
    # dataset objects
    ds_train = load_dataset(args.dataset, train=True)
    ds_test  = load_dataset(args.dataset, train=False)
    # lok up per client loader
    client_loaders = {}
    for cid in range(args.num_clients):
        idxs = mapping[f"client_{cid}"]["indices"]
        subset = Subset(ds_train, idxs)
        if len(subset) == 0:
            print(f"[WARN] client {cid} has 0 samples – skipped")
            continue
        client_loaders[cid] = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=len(subset)>=args.batch_size)
    active_clients = list(client_loaders.keys())
    model = ResNet18Embed(num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # global prototypes init
    global_proto = torch.zeros(num_classes, model.embed_dim, device=device)
    for rnd in range(1, args.rounds + 1):
        model.train()
        proto_list, count_list = [], []
        for cid in active_clients:
            for epoch in range(args.local_epochs):
                for x, y in client_loaders[cid]:
                    x, y = x.to(device), y.to(device)
                    logits, z = model(x)
                    ce = F.cross_entropy(logits, y)
                    rp = rpcl_loss(z, y, global_proto, args.tau)
                    loss = ce + args.lambda_rpcl * rp
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            # collect client prototype
            with torch.no_grad():
                vec = torch.zeros(num_classes, model.embed_dim, device=device)
                cnt = torch.zeros(num_classes, 1, device=device)
                for x, y in client_loaders[cid]:
                    x, y = x.to(device), y.to(device)
                    _, z = model(x)
                    for cls in y.unique():
                        m = y == cls
                        vec[cls] += z[m].mean(0)
                        cnt[cls] += 1
                proto_list.append(vec)
                count_list.append(cnt)
        # aggregate
        global_proto = aggregate_prototypes(proto_list, count_list)
        # evaluation
        if rnd % args.eval_interval == 0 or rnd == args.rounds:
            acc = eval_model(model, ds_test, device)
            print(f"Round {rnd:3}/{args.rounds} | Global Acc = {acc:.2f}%")
    # save
    out = f"fedsc_{args.dataset}_{args.scenario}.pt"
    torch.save(model.state_dict(), out)
    print(f"[✓] Model saved to {out}")


def eval_model(model, ds_test, device):
    model.eval()
    loader = DataLoader(ds_test, batch_size=256, num_workers=2)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logit, _ = model(x)
            pred = logit.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# --------------------------------------------------------------------------------------
# DATA UTIL
# --------------------------------------------------------------------------------------

def load_dataset(name: str, train: bool):
    if name == "cifar10":
        return datasets.CIFAR10("data", train=train, download=True, transform=TRANSFORM)
    if name == "cifar100":
        return datasets.CIFAR100("data", train=train, download=True, transform=TRANSFORM)
    if name == "fc100":
        from torchvision.datasets import CIFAR100 as FC100  # placeholder
        return FC100("data", train=train, download=True, transform=TRANSFORM)
    raise ValueError(name)

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=["iid","nid1","nid2","longtail","fewshot"])
    p.add_argument("--dataset", required=True, choices=["cifar10","cifar100","fc100"])
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--rho", type=int, default=200)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--train", action="store_true")
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lambda_rpcl", type=float, default=1.0)
    p.add_argument("--lambda_cpdr", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval_interval", type=int, default=5)
    args = p.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    # build partition
    ds_train = load_dataset(args.dataset, train=True)
    if args.scenario == "iid":
        mapping = partition_iid(ds_train, args.num_clients)
    elif args.scenario == "nid1":
        mapping = partition_nid1(ds_train, args.num_clients, args.alpha)
    elif args.scenario == "nid2":
        mapping = partition_nid2(ds_train, args.num_clients)
    elif args.scenario == "longtail":
        mapping = partition_longtail(ds_train, args.num_clients, args.rho, args.alpha)
    elif args.scenario == "fewshot":
        mapping = partition_fewshot(ds_train, args.num_clients, args.k_shot)
    else:
        raise ValueError()

    out_json = Path(f"partitions_{args.scenario}_{args.dataset}.json")
    save_partition_json(mapping, out_json, ds_train)

    if not args.train:
        print("[INFO] Partitioning complete – no training requested.")
        return

    num_classes = len(ds_train.classes)
    train_fedsc(args, out_json, num_classes)

if __name__ == "__main__":
    main()
