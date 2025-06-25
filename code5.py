"""python
#!/usr/bin/env python
"""

from __future__ import annotations
import argparse, shutil, urllib.request, tarfile, json, subprocess, sys, os
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# ---------------------------------------------------------------------
# -------------------------  I/O utilities  ---------------------------
# ---------------------------------------------------------------------

def download_chunks(base_url: str, dst_dir: Path, num_chunks: int) -> List[Path]:
    """Download *num_chunks* sequential files `{base_url}{00..}` into *dst_dir*."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: List[Path] = []
    for idx in range(num_chunks):
        fname = f"{Path(base_url).name}{idx:02d}"
        url   = f"{base_url}{idx:02d}"
        out   = dst_dir / fname
        chunk_paths.append(out)
        if out.exists():
            print(f"[download] skip existing {out.name}")
            continue
        print(f"[download] {url} → {out}")
        urllib.request.urlretrieve(url, out)
    return chunk_paths


def combine_chunks(chunk_paths: List[Path], combined_tar: Path) -> None:
    """Concatenate chunk_paths → *combined_tar* (binary cat)."""
    if combined_tar.exists():
        print(f"[combine] skip – already exists {combined_tar}")
        return
    print(f"[combine] writing {combined_tar} …")
    with combined_tar.open("wb") as w:
        for p in chunk_paths:
            with p.open("rb") as r:
                shutil.copyfileobj(r, w)
    print("[combine] done.")


def extract_tar(tar_path: Path, extract_to: Path) -> None:
    """Extract a .tar.gz archive (tar_path) into *extract_to*."""
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"[extract] skip – dir not empty: {extract_to}")
        return
    print(f"[extract] extracting {tar_path} …")
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=extract_to)
    print("[extract] done.")

# ---------------------------------------------------------------------
# ---------------------  JSON → PyG conversion  -----------------------
# ---------------------------------------------------------------------

def _json_to_data(j: Dict) -> Data:
    """Best‑effort parse a MalNet JSON record → torch_geometric.data.Data"""
    # node features
    if "node_features" in j:
        x = torch.tensor(j["node_features"], dtype=torch.float)
    elif "x" in j:
        x = torch.tensor(j["x"], dtype=torch.float)
    else:
        raise KeyError("Missing node features key in JSON")

    # edges
    if "edge_index" in j.get("graph", {}):
        edge = j["graph"]["edge_index"]
    elif "edges" in j.get("graph", {}):
        edge = j["graph"]["edges"]
    elif "edge_index" in j:
        edge = j["edge_index"]
    else:
        raise KeyError("Missing edge index key in JSON")
    edge_index = torch.tensor(edge, dtype=torch.long).t().contiguous()

    # labels – mandatory family, optional type
    y_family = torch.tensor([j.get("family") or j.get("y_family")], dtype=torch.long)
    y_type   = j.get("type") or j.get("y_type")
    data = Data(x=x, edge_index=edge_index, y_family=y_family)
    if y_type is not None:
        data.y_type = torch.tensor([y_type], dtype=torch.long)
    return data


def json_dir_to_pt(json_dir: Path, pt_out: Path) -> None:
    """Parse all *.json* inside *json_dir* → list[Data], torch.save to *pt_out*."""
    graphs: List[Data] = []
    json_files = sorted(json_dir.rglob("*.json"))
    for jpath in json_files:
        with jpath.open() as f:
            j = json.load(f)
        try:
            graphs.append(_json_to_data(j))
        except Exception as e:
            print(f"[warn] skip {jpath.name}: {e}")
    torch.save(graphs, pt_out)
    print(f"[PT] saved {len(graphs)} graphs → {pt_out}")

# ---------------------------------------------------------------------
# ---------------------------  Dataset  --------------------------------
# ---------------------------------------------------------------------
class MalwareDataset(InMemoryDataset):
    """Load *processed/data.pt* if exists else expects caller to have run prepare."""
    def __init__(self, root: str):
        self._processed_file = Path(root)/"processed"/"data.pt"
        super().__init__(root)
        self.data, self.slices = torch.load(self._processed_file.as_posix())

    @property
    def raw_file_names(self):
        # Not used – we rely on external prepare step
        return []

    @property
    def processed_file_names(self):
        return [self._processed_file.name]

    def download(self):
        pass  # done by prepare step

    def process(self):
        pass  # done by prepare step

# ---------------------------------------------------------------------
# -------------------------------  HGAT  -------------------------------
# ---------------------------------------------------------------------
class HierarchicalReadout(nn.Module):
    def __init__(self, in_dim: int, type_emb_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.type_emb = nn.Embedding(256, type_emb_dim)  # adjust if >256 types
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + type_emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x, batch, type_idx):
        # global mean pool for node embeddings
        g = global_mean_pool(x, batch)
        t = self.type_emb(type_idx.squeeze()) if type_idx is not None else torch.zeros_like(g)
        z = torch.cat([g, t], dim=1)
        return self.mlp(z)


class HGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, heads: int = 4, num_classes: int = 100):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads))
        for _ in range(num_layers-1):
            self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads))
        self.readout = HierarchicalReadout(hidden_dim*heads, type_emb_dim=32, hidden=hidden_dim, out_dim=num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        logits = self.readout(x, batch, getattr(data, "y_type", None))
        return logits

# ---------------------------------------------------------------------
# ----------------------------  Training  -----------------------------
# ---------------------------------------------------------------------

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y_family.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1)
            correct += (pred == data.y_family.squeeze()).sum().item()
    return correct / len(loader.dataset)

# ---------------------------------------------------------------------
# --------------------------  CLI pipeline  ---------------------------
# ---------------------------------------------------------------------

def prepare_stage(args):
    raw_dir   = Path(args.data_root)/"raw"
    chunks    = download_chunks(args.base_url, raw_dir, args.num_chunks)
    tar_path  = raw_dir/"malnet_sample.tar.gz"
    combine_chunks(chunks, tar_path)

    extract_dir = raw_dir/"extracted"
    extract_tar(tar_path, extract_dir)

    # Convert JSON → .pt (only within extracted/graphs or similar)
    # Heuristic: search for directory containing many .json files (>100)
    json_root = None
    for d in extract_dir.rglob("*"):
        if d.is_dir():
            cnt = len(list(d.glob("*.json")))
            if cnt > 100:
                json_root = d; break
    if json_root is None:
        # Fallback – maybe JSONs at top level
        json_root = extract_dir

    processed_dir = Path(args.data_root)/"processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pt_file = processed_dir/"data.pt"

    json_dir_to_pt(json_root, pt_file)

    # Build slices for InMemoryDataset
    graphs: List[Data] = torch.load(pt_file)
    data, slices = InMemoryDataset.collate(graphs)
    torch.save((data, slices), pt_file)
    print(f"[prepare] finished ✓ {pt_file.relative_to(Path.cwd())}")


def train_stage(args):
    dataset = MalwareDataset(args.data_root)
    num_feats = dataset.num_features
    num_classes = int(dataset.data.y_family.max().item()+1)

    torch.manual_seed(42)
    perm = torch.randperm(len(dataset))
    train_size = int(0.8*len(dataset))
    train_idx, val_idx = perm[:train_size], perm[train_size:]
    train_ds = dataset[train_idx]
    val_ds   = dataset[val_idx]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HGAT(num_feats, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                 heads=args.heads, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.
    for epoch in range(1, args.epochs+1):
        loss = train(model, train_loader, optimizer, criterion, device)
        acc  = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}: loss={loss:.4f} | val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
    print(f"Best val acc: {best_acc:.4f}")

# ---------------------------------------------------------------------
# ------------------------------  Main  -------------------------------
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HGAT MalNet pipeline (sample chunks)")
    p.add_argument("--data_root", type=str, default="data/malnet_sample", help="root dir for dataset")
    p.add_argument("--base_url", type=str, default="http://malnet.cc.gatech.edu/graph-data/full-data-as-1GB/malnet-graph", help="base URL without index")
    p.add_argument("--num_chunks", type=int, default=2, help="#chunks to download (1GB each)")
    p.add_argument("--prepare_only", action="store_true", help="only run prepare stage and exit")
    # training hyper‑params
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not (Path(args.data_root)/"processed"/"data.pt").exists():
        prepare_stage(args)
        if args.prepare_only:
            sys.exit(0)
    if args.prepare_only:
        print("[warn] --prepare_only specified but processed file already exists; nothing to do.")
        sys.exit(0)
    train_stage(args)
