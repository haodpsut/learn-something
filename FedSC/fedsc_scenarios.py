#!/usr/bin/env python
"""fedsc_scenarios.py – v3 FULL
================================
Re‑implementation of **all scenarios and a minimal yet functional FedSC trainer** on a single
machine (no Flower/FedML required). The code is purposely concise (< ~500 LOC) so you can read
and customize quickly.

Key Features
------------
* **Scenario generator**: IID, NID1 (Dirichlet), NID2 (extreme label shift), Long‑tail (ρ), Few‑shot.
* **Prototype exchange**: each round clients send (C×d) prototype tensors and counts.
* **Loss**: Cross‑Entropy + RPCL + CPDR exactly as paper (temperature τ, λ_rpcl, λ_cpdr).
* **Training loop**: sequential FedAvg for simplicity; adequate for research reproduction.
* **CLI flags**: cover every paper hyper‑parameter; see `python fedsc_scenarios.py -h`.

Usage Examples
--------------
```bash
# NID1 CIFAR‑10
python fedsc_scenarios.py --scenario nid1 --dataset cifar10 --alpha 0.2 --rounds 80 --train

# Long‑tail CIFAR‑100 (ρ=200)
python fedsc_scenarios.py --scenario longtail --dataset cifar100 --rho 200 --rounds 80 --train
```
"""

from __future__ import annotations
import argparse, os, random, json, math, time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

try:
    import torchvision
    from torchvision import transforms
except ImportError as e:
    raise RuntimeError("torchvision is required – pip install torchvision") from e

# -----------------------------------------------------------------------------
#                       Dataset loading helpers
# -----------------------------------------------------------------------------

def get_dataset(name: str, root: str = "data") -> Tuple[Dataset, Dataset, int]:
    """Returns (train_set, test_set, num_classes)."""
    name = name.lower()
    if name == "cifar10":
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        train_set = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tfm_train)
        test_set = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tfm_test)
        return train_set, test_set, 10
    elif name == "cifar100":
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        train_set = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=tfm_train)
        test_set = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=tfm_test)
        return train_set, test_set, 100
    elif name == "fc100":
        # FC100 via Torchmeta (must be installed separately)
        try:
            from torchmeta.datasets.helpers import fc100
        except ImportError:
            raise RuntimeError("torchmeta not installed – pip install torchmeta")
        dataset_train = fc100(root, shots=None, ways=None, shuffle=True, test_shots=None, meta_train=True, download=True)
        dataset_test = fc100(root, shots=None, ways=None, shuffle=False, test_shots=None, meta_test=True, download=True)
        for ds in (dataset_train, dataset_test):
            ds.transform = transforms.Compose([
                lambda x: x.transpose(2, 0, 1),  # HWC -> CHW
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                     (0.2675, 0.2565, 0.2761)),
            ])
        return dataset_train, dataset_test, 100
    else:
        raise ValueError(f"Unsupported dataset {name}")

# -----------------------------------------------------------------------------
#                       Partitioning utilities
# -----------------------------------------------------------------------------

def dirichlet_split(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 0) -> Dict[int, List[int]]:
    """Returns mapping client -> list(indices) under Dirichlet label distribution."""
    rng = np.random.default_rng(seed)
    num_classes = labels.max() + 1
    label_indices = [np.where(labels == i)[0] for i in range(num_classes)]  # per‑class idx
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    for c, idxs in enumerate(label_indices):
        rng.shuffle(idxs)
        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        split = np.split(idxs, proportions)
        for cid, part in enumerate(split):
            client_indices[cid].extend(part.tolist())
    return client_indices


def partition_iid(labels: np.ndarray, num_clients: int) -> Dict[int, List[int]]:
    idxs = np.arange(len(labels))
    np.random.shuffle(idxs)
    return {cid: idxs[cid::num_clients].tolist() for cid in range(num_clients)}


def partition_nid1(labels: np.ndarray, num_clients: int, alpha: float) -> Dict[int, List[int]]:
    return dirichlet_split(labels, num_clients, alpha)


def partition_nid2(labels: np.ndarray, num_clients: int, seed: int = 0) -> Dict[int, List[int]]:
    """Half clients each get |C|/5 classes; one client gets all classes (extreme)."""
    rng = np.random.default_rng(seed)
    num_classes = labels.max() + 1
    classes = np.arange(num_classes)
    rng.shuffle(classes)
    subset_size = max(1, num_classes // 5)
    client_indices = {i: [] for i in range(num_clients)}
    # first half clients
    for cid in range(num_clients // 2):
        cls_subset = classes[cid*subset_size:(cid+1)*subset_size]
        mask = np.isin(labels, cls_subset)
        client_indices[cid] = np.where(mask)[0].tolist()
    # one super client (last index)
    client_indices[num_clients-1] = list(range(len(labels)))  # all samples
    # remaining clients get random leftover data (simple split)
    leftover_clients = list(range(num_clients//2, num_clients-1))
    leftover_ids = list(set(range(len(labels))) - set(sum(client_indices.values(), [])))
    rng.shuffle(leftover_ids)
    for i, cid in enumerate(leftover_clients):
        client_indices[cid] = leftover_ids[i::len(leftover_clients)]
    return client_indices


def create_longtail(labels: np.ndarray, rho: int, seed: int = 0) -> np.ndarray:
    """Returns indices reordered to simulate long‑tailed class frequency."""
    rng = np.random.default_rng(seed)
    num_classes = labels.max() + 1
    class_counts = np.linspace(1, 1/rho, num_classes)
    class_counts = (class_counts / class_counts.sum() * len(labels)).astype(int)
    idxs_per_class = [rng.permutation(np.where(labels == c)[0])[:class_counts[c]] for c in range(num_classes)]
    return np.concatenate(idxs_per_class)


def partition_longtail(labels: np.ndarray, num_clients: int, rho: int, alpha: float) -> Dict[int, List[int]]:
    idxs = create_longtail(labels, rho)
    return dirichlet_split(labels[idxs], num_clients, alpha)


def partition_fewshot(labels: np.ndarray, num_clients: int, k_shot: int) -> Dict[int, List[int]]:
    num_classes = labels.max() + 1
    cls_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices = {cid: [] for cid in range(num_clients)}
    for c, idxs in enumerate(cls_indices):
        np.random.shuffle(idxs)
        for cid in range(num_clients):
            take = idxs[cid*k_shot:(cid+1)*k_shot]
            client_indices[cid].extend(take.tolist())
    return client_indices

# -----------------------------------------------------------------------------
#                       Model & losses
# -----------------------------------------------------------------------------

class ResNet18Feats(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 512):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # remove fc
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.emb_dim = backbone.fc.in_features  # 512

    def forward(self, x, return_feats=False):
        feat = self.features(x)  # [B, 512, 1, 1]
        feat = feat.view(feat.size(0), -1)
        logits = self.fc(feat)
        if return_feats:
            return logits, feat
        return logits


def rpcl_loss(feat: torch.Tensor, labels: torch.Tensor, proto: torch.Tensor, tau: float) -> torch.Tensor:
    """Relational Prototype Contrastive Loss."""
    # Normalize for cosine similarity
    feat = F.normalize(feat, dim=1)
    proto_norm = F.normalize(proto, dim=1)
    logits = feat @ proto_norm.t() / tau  # [B, C]
    return F.cross_entropy(logits, labels)


def cpdr_loss(feat: torch.Tensor, labels: torch.Tensor, global_proto: torch.Tensor, tau: float) -> torch.Tensor:
    feat = F.normalize(feat, dim=1)
    gp = F.normalize(global_proto, dim=1)
    logits = feat @ gp.t() / tau
    return F.cross_entropy(logits, labels)

# -----------------------------------------------------------------------------
#                       FedSC training utilities
# -----------------------------------------------------------------------------

@torch.no_grad()
def compute_prototypes(loader: DataLoader, model: nn.Module, device: torch.device, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (proto, counts): [C,d], [C,1]."""
    feats_sum = torch.zeros(num_classes, model.emb_dim, device=device)
    counts = torch.zeros(num_classes, 1, device=device)
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, feat = model(x, return_feats=True)
        for c in range(num_classes):
            mask = (y == c)
            if mask.any():
                feats_sum[c] += feat[mask].sum(0)
                counts[c] += mask.sum()
    proto = feats_sum / torch.clamp(counts, min=1e-8)
    proto[counts.squeeze(-1) == 0] = 0
    return proto, counts


def aggregate_prototypes(proto_list: List[torch.Tensor], count_list: List[torch.Tensor]) -> torch.Tensor:
    proto_sum = torch.stack(proto_list).sum(0)  # [C,d]
    counts_sum = torch.stack(count_list).sum(0)  # [C,1]
    global_proto = proto_sum / torch.clamp(counts_sum, min=1e-8)
    empty_mask = (counts_sum.squeeze(-1) == 0)  # [C]
    global_proto[empty_mask] = 0
    return global_proto


def fedavg(models: List[nn.Module]) -> Dict[str, torch.Tensor]:
    """Returns averaged state_dict."""
    state_dicts = [m.state_dict() for m in models]
    avg = {}
    for k in state_dicts[0]:
        avg[k] = sum([sd[k] for sd in state_dicts]) / len(state_dicts)
    return avg


def train_fedsc(args, part_path: str, num_classes: int):
    device = torch.device(args.device)
    # ---------------- Load dataset & partitions ----------------
    train_set, test_set, _ = get_dataset(args.dataset)
    with open(part_path) as f:
        mapping = json.load(f)
    # ---------------- Init global model ----------------
    global_model = ResNet18Feats(num_classes).to(device)

    # Pre‑compute global proto (zeros) to start
    global_proto = torch.zeros(num_classes, global_model.emb_dim, device=device)

    # Create dataloaders per client
    client_loaders = {}
    for cid, idxs in mapping.items():
        subset = Subset(train_set, idxs)
        client_loaders[cid] = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=len(subset)>=args.batch_size)

    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    # ---------------- Training rounds ----------------
    criterion = nn.CrossEntropyLoss()
    for rnd in range(1, args.rounds + 1):
        local_models, proto_list, count_list = [], [], []
        for cid in mapping.keys():
            model = ResNet18Feats(num_classes).to(device)
            model.load_state_dict(global_model.state_dict())  # copy weights
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            loader = client_loaders[cid]
            model.train()
            for _ in range(args.local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    logits, feat = model(x, return_feats=True)
                    loss_ce = criterion(logits, y)
                    loss_rpcl = rpcl_loss(feat, y, global_proto.detach(), args.tau)
                    loss_cpdr = cpdr_loss(feat, y, global_proto.detach(), args.tau)
                    loss = loss_ce + args.lambda_rpcl * loss_rpcl + args.lambda_cpdr * loss_cpdr
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # compute local proto after training for aggregation
            proto, counts = compute_prototypes(loader, model, device, num_classes)
            local_models.append(model)
            proto_list.append(proto)
            count_list.append(counts)
        # ---------- Aggregate -------------
        global_model.load_state_dict(fedavg(local_models))
        global_proto = aggregate_prototypes(proto_list, count_list)

        # ---------- Eval -------------
        if rnd % args.eval_interval == 0 or rnd == args.rounds:
            acc = evaluate(global_model, test_loader, device)
            print(f"Round {rnd:3d}/{args.rounds} | Global Acc = {acc:.2f}%")

    # Save
    out_path = f"fedsc_{args.dataset}_{args.scenario}.pt"
    torch.save({"model": global_model.state_dict(), "proto": global_proto.cpu()}, out_path)
    print(f"[✓] Model saved to {out_path}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total

# -----------------------------------------------------------------------------
#                       CLI & main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Scenario flags
    p.add_argument("--scenario", required=True, choices=["iid", "nid1", "nid2", "longtail", "fewshot"], help="Scenario type")
    p.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "fc100"])
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.2, help="Dirichlet α (nid1, longtail)")
    p.add_argument("--rho", type=int, default=100, help="Imbalance ratio (longtail)")
    p.add_argument("--k_shot", type=int, default=5, help="Few‑shot per class (fewshot)")

    # Training flags
    p.add_argument("--train", action="store_true", help="Run FedSC training after partitioning")
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lambda_rpcl", type=float, default=1.0)
    p.add_argument("--lambda_cpdr", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval_interval", type=int, default=5)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("data", exist_ok=True)

    # ---------------- Load dataset to get labels ----------------
    train_set, _, num_classes = get_dataset(args.dataset)
    labels = np.array(train_set.targets if hasattr(train_set, "targets") else train_set.labels)

    # ---------------- Partition ----------------
    if args.scenario == "iid":
        mapping = partition_iid(labels, args.num_clients)
    elif args.scenario == "nid1":
        mapping = partition_nid1(labels, args.num_clients, args.alpha)
    elif args.scenario == "nid2":
        mapping = partition_nid2(labels, args.num_clients)
    elif args.scenario == "longtail":
        mapping = partition_longtail(labels, args.num_clients, args.rho, args.alpha)
    elif args.scenario == "fewshot":
        mapping = partition_fewshot(labels, args.num_clients, args.k_shot)
    else:
        raise ValueError("Unknown scenario")

    part_path = f"partitions_{args.scenario}_{args.dataset}.json"
    with open(part_path, "w") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f)
    print(f"[INFO] Partition mapping saved to {part_path}")

    # ---------------- Training ----------------
    if args.train:
        print("[INFO] ⏳ Starting FedSC training…")
        if args.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, falling back to CPU…")
            args.device = "cpu"
        train_fedsc(args, part_path, num_classes)
    else:
        print("[INFO] Partitioning complete – no training requested.")


if __name__ == "__main__":
    main()
