#!/usr/bin/env python
"""
code_Ling.py – …
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────── #
import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (accuracy_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.exceptions import UndefinedMetricWarning

from transformers import (AutoTokenizer, AutoModel,
                          logging as hf_logging)

# ── SUPPRESS MOST WARNINGS ─────────────────────────────────────────────────── #
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch_geometric.sampler.neighbor_sampler")
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()      # ẩn warning transformers

# ───────────────────────────────────────────────────────────────────────────── #



# ───────────────────────────── UTILITIES ───────────────────────────────────── #

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_embeddings(texts: List[str],
                       model_name: str = "distilbert-base-uncased",
                       batch_size: int = 32,
                       device: str = "cpu") -> torch.Tensor:
    """
    Trả về ma trận (N, 768) của CLS-embedding DistilBERT.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(batch,
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_tensors="pt").to(device)
            outputs = model(**encoded)          # last_hidden_state
            cls_vec = outputs.last_hidden_state[:, 0, :]   # CLS token
            embeddings.append(cls_vec.cpu())

    return torch.cat(embeddings, dim=0)          # (N, 768)

def build_knn_graph(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Trả về edge_index (2, E) từ đồ thị K-NN cosine-similarity (có hướng).
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1,      # +1 để bỏ chính nó
                            metric="cosine",
                            algorithm="auto").fit(x)
    distances, indices = nbrs.kneighbors(x)        # (N, k+1)

    edges_src = []
    edges_dst = []

    num_nodes = x.size(0)
    for src in range(num_nodes):
        for j in range(1, k + 1):                  # skip j = 0 (self)
            dst = indices[src, j]
            edges_src.append(src)
            edges_dst.append(dst)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index


# ──────────────────────────────  MODEL ─────────────────────────────────────── #

class KANGuard(nn.Module):
    """
    GraphSAGE encoder → KAN-style tiny MLP → 1-d logit *mỗi node*.
    (Không dùng global pooling để phân loại từng email/nút.)
    """
    def __init__(self,
                 in_dim: int = 768,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 kan_q: int = 4,
                 dropout: float = 0.2):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # KAN-like: mỗi chiều hidden qua MLP nhỏ, gộp lại
        self.kan = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * kan_q),
            nn.GELU(),
            nn.Linear(hidden_dim * kan_q, hidden_dim),
            nn.GELU(),
        )

        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)

        h = self.kan(h)
        logits = self.classifier(h).squeeze(-1)   # shape (N_nodes,)
        return logits


# ────────────────────────── TRAIN / EVAL LOOP ─────────────────────────────── #

def build_loss(args, device):
    if args.class_weight is not None:
        w = torch.tensor([args.class_weight], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=w)
    return nn.BCEWithLogitsLoss()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index)
        prob  = torch.sigmoid(out)
        pred  = (prob > 0.5).long()

        y_true.append(batch.y.cpu())
        y_pred.append(pred.cpu())
        y_prob.append(prob.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    y_prob = torch.cat(y_prob).numpy()

    acc    = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1     = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:  # chỉ một lớp
        roc = 0.5

    return acc, recall, f1, roc


# ────────────────────────────── MAIN ──────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=5,
                        help="K của đồ thị K-NN")
    parser.add_argument("--class_weight", type=float, default=None,
                        help="Trọng số lớp dương (spam) cho BCEWithLogits")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    set_seed()

    # 1) Nạp & trích embedding
    df = pd.read_csv(args.csv_path)
    assert {"subject", "body", "label"}.issubset(df.columns)

    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    texts  = df["text"].tolist()
    labels = df["label"].astype(int).values

    x = compute_embeddings(texts, device=device)        # (N, 768)
    edge_index = build_knn_graph(x, k=args.k)

    # 2) Tạo đối tượng Data (PyG)
    data = Data(x=x,
                y=torch.tensor(labels, dtype=torch.long),
                edge_index=edge_index)

    # 3) Chia train / val theo node
    idx = np.arange(data.num_nodes)
    train_idx, val_idx = train_test_split(idx, test_size=0.2, stratify=labels,
                                          random_state=42)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx]     = True

    # 4) Loader
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=True,
        input_nodes=data.train_mask)

    val_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=False,
        input_nodes=data.val_mask)

    # 5) Model, optimizer, loss
    model = KANGuard().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = build_loss(args, device)

    best_f1   = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    # 6) Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out   = model(batch.x, batch.edge_index)
            loss  = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_nodes

        epoch_loss /= len(train_idx)

        acc, rec, f1, roc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | loss={epoch_loss:.4f} | "
              f"acc={acc:.3f} | recall={rec:.3f} | f1={f1:.3f} | roc={roc:.3f}")

        # Lưu model tốt nhất theo F1
        if f1 > best_f1:
            best_f1 = f1
            ckpt_path = f"checkpoints/best_model_epoch{epoch:02d}_f1{f1:.3f}.pt"
            torch.save(model.state_dict(), ckpt_path)

    print(f"Done. Best F1 = {best_f1:.3f}")

# ───────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    main()
