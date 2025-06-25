#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HGAT pipeline for MalNet-Graph (split-chunk friendly)
====================================================
• Chuẩn bị dữ liệu từ n chunk:   --prepare_only
• Huấn luyện demo HGAT:         --epochs N

Ví dụ (2 chunk – prepare):
python hgat.py \
  --base_url  http://malnet.cc.gatech.edu/graph-data/full-data-as-1GB/malnet-graph \
  --num_chunks 2 \
  --data_root  data/malnet_sample \
  --prepare_only \
  --allow_partial_extract
"""

from __future__ import annotations
import os, sys, argparse, urllib.request, gzip, tarfile, subprocess
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm import tqdm

# ----------------------------------------------------------------------
# Download & combine split-chunks
# ----------------------------------------------------------------------
def download_chunks(base_url: str, num_chunks: int, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_chunks):
        dst = raw_dir / f"malnet-graph{i:02d}"
        if dst.exists():
            print(f"[download] skip existing {dst.name}")
            continue
        url = f"{base_url}{i:02d}"
        print(f"[download] {url} → {dst}")
        urllib.request.urlretrieve(url, dst)

def combine_chunks(raw_dir: Path, tar_name: str) -> Path:
    tar_path = raw_dir / tar_name
    if tar_path.exists():
        print(f"[combine] {tar_name} already exists – skip")
        return tar_path
    print(f"[combine] writing {tar_name} …")
    with tar_path.open("wb") as w:
        for fp in sorted(raw_dir.glob("malnet-graph??")):
            w.write(fp.read_bytes())
    print("[combine] done.")
    return tar_path

# ----------------------------------------------------------------------
# Partial extraction (works with truncated archives)
# ----------------------------------------------------------------------
def detect_gzip(path: Path) -> bool:
    with path.open("rb") as f:
        return f.read(2) == b"\x1f\x8b"


def extract_tar_partial(tar_path: Path, extract_to: Path):
    """Stream-extract .tar or .tar.gz even khi bị cắt, bỏ qua lỗi cuối file."""
    extract_to.mkdir(parents=True, exist_ok=True)

    is_gz = tar_path.read_bytes()[:2] == b"\x1f\x8b"
    mode  = "r|gz" if is_gz else "r|"
    fileobj = gzip.open(tar_path, "rb") if is_gz else tar_path.open("rb")
    print(f"[extract] streaming ({'gzip' if is_gz else 'plain tar'}) …")

    n = 0
    try:
        with tarfile.open(fileobj=fileobj, mode=mode) as tf:
            for ti in tf:
                try:
                    tf.extract(ti, path=extract_to)
                    n += 1
                except (tarfile.ExtractError, OSError):
                    continue
    except EOFError:
        print("  [warn] reached unexpected EOF – archive truncated.")
    finally:
        fileobj.close()
    print(f"[extract] extracted {n:,} members (partial archive)")


# ----------------------------------------------------------------------
# Build processed/data.pt from .edgelist
# ----------------------------------------------------------------------
def build_pt(extract_dir: Path, processed_pt: Path):
    print("[prepare] building Data objects …")
    edge_files = list(extract_dir.rglob("*.edgelist"))
    if not edge_files:
        print("  [ERR] không tìm thấy .edgelist – kiểm tra lại chunk hoặc đường dẫn!")
        sys.exit(1)

    data_list, fam2id, type2id = [], {}, {}
    for fp in tqdm(edge_files):
        # …/malnet-graphs/<type>/<family>/<sha>.edgelist
        gtype, gfamily = fp.parts[-3], fp.parts[-2]
        tid = type2id.setdefault(gtype, len(type2id))
        fid = fam2id.setdefault(gfamily, len(fam2id))

        edges = np.loadtxt(fp, dtype=int)
        if edges.size == 0:
            continue
        edges = edges.reshape(-1, 2)
        edge_index = torch.tensor(edges.T, dtype=torch.long)

        num_nodes = int(edge_index.max()) + 1
        deg = torch.bincount(edge_index.reshape(-1),
                             minlength=num_nodes).unsqueeze(1).float()

        data_list.append(Data(x=deg,
                              edge_index=edge_index,
                              y_family=torch.tensor([fid]),
                              y_type=torch.tensor([tid])))

    processed_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, processed_pt)
    print(f"[prepare] saved {len(data_list):,} graphs → {processed_pt}")

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class MalwareDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):        # not used – placeholder
        return ["dummy"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        pass

# ----------------------------------------------------------------------
# HGAT demo model
# ----------------------------------------------------------------------
class HGAT(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=64, heads=4, layers=2):
        super().__init__()
        self.gats = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            self.gats.append(GATConv(last, hidden, heads=heads, dropout=0.2))
            last = hidden * heads
        self.cls = nn.Linear(last, num_classes)

    def forward(self, x, edge_index, batch):
        for gat in self.gats:
            x = torch.relu(gat(x, edge_index))
        hg = global_mean_pool(x, batch)
        return self.cls(hg)

# ----------------------------------------------------------------------
# Train / test helpers
# ----------------------------------------------------------------------
def train_one(model, loader, opt, device):
    model.train(); total = 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = nn.CrossEntropyLoss()(out, data.y_family.squeeze())
        loss.backward(); opt.step()
        total += loss.item() * data.num_graphs
    return total / len(loader.dataset)

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval(); correct = total = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(1)
        correct += (pred == data.y_family.squeeze()).sum().item()
        total += data.num_graphs
    return correct / total

# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", required=True)
    p.add_argument("--num_chunks", type=int, required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--allow_partial_extract", action="store_true")
    p.add_argument("--prepare_only", action="store_true")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()

def main():
    args = parse_args()
    root = Path(args.data_root)
    raw_dir = root / "raw"
    processed_pt = root / "processed" / "data.pt"

    # -------------------- PREPARE --------------------
    if args.prepare_only or not processed_pt.exists():
        download_chunks(args.base_url, args.num_chunks, raw_dir)
        tar_path = combine_chunks(raw_dir, "malnet.tar")
        extract_dir = root / "extracted"
        if args.allow_partial_extract:
            extract_tar_partial(tar_path, extract_dir)
        else:
            extract_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["tar", "-xf", tar_path, "-C", extract_dir], check=True)
        build_pt(extract_dir, processed_pt)
        if args.prepare_only:
            return

    # -------------------- TRAIN ----------------------
    dataset = MalwareDataset(root.as_posix())
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(val_ds,   batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HGAT(1, num_classes=int(dataset.data.y_family.max()+1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs+1):
        loss = train_one(model, tl, opt, device)
        acc  = eval_acc(model,  vl, device)
        print(f"[E{epoch:02d}] loss={loss:.4f} | val_acc={acc:.3f}")

if __name__ == "__main__":
    main()
