import sys
import csv
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, Module
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import networkx as nx

class GraphSAGE(Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.dropout = Dropout(0.2)

    def forward(self, x, edge_index):
        x = self.dropout(ReLU()(self.conv1(x, edge_index)))
        x = self.conv2(x, edge_index)
        return x.view(-1)

def evaluate(model, loader, device):
    model.eval()
    ys, yh, yp = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            prob = torch.sigmoid(out)
            pred = (prob > 0.5).int()
            ys += batch.y.tolist()
            yh += pred.tolist()
            yp += prob.tolist()

    acc = accuracy_score(ys, yh)
    rec = recall_score(ys, yh, pos_label=1, zero_division=0)
    f1 = f1_score(ys, yh, pos_label=1, zero_division=0)
    roc = roc_auc_score(ys, yp)
    return acc, rec, f1, roc

def main():
    sys.setrecursionlimit(100000)
    csv.field_size_limit(sys.maxsize)

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    # === 1. Load data ===
    df = pd.read_csv(args.csv_path, engine='python')
    df = df.dropna(subset=["subject", "body", "label"])
    df["label"] = df["label"].astype(str).str.extract(r"(\d+)").astype(int)

    texts = (df["subject"] + " " + df["body"]).tolist()
    labels = df["label"].tolist()

    # === 2. Embedding ===
    print("Embedding:", end=" ")
    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    X = model_embed.encode(tqdm(texts, desc="Embedding"))
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # === 3. Build kNN Graph ===
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=args.k + 1).fit(X)
    neigh = knn.kneighbors(X, return_distance=False)
    edge_index = []
    for i, neighbors in enumerate(neigh):
        for j in neighbors[1:]:  # skip self-loop
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index).t().contiguous()

    # === 4. PyG Data ===
    data = Data(x=X, edge_index=edge_index, y=y)
    data.num_nodes = X.shape[0]

    idx = np.arange(data.num_nodes)
    y_np = y.cpu().numpy()
    idx_train, idx_val = train_test_split(idx, test_size=0.2, stratify=y_np, random_state=42)

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[idx_train] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[idx_val] = True

    tr_loader = NeighborLoader(data, num_neighbors=[15, 15], batch_size=args.batch_size,
                               input_nodes=data.train_mask)
    va_loader = NeighborLoader(data, num_neighbors=[15, 15], batch_size=args.batch_size,
                               input_nodes=data.val_mask)

    # === 5. Train ===
    dev = torch.device(args.device)
    model = GraphSAGE(X.size(1), 64).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_f1 = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tr_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        acc, rec, f1, roc = evaluate(model, va_loader, dev)
        print(f"Epoch {epoch:02d} | loss={np.mean(losses):.4f} | acc={acc:.3f} | recall={rec:.3f} | f1={f1:.3f} | roc={roc:.3f}")
        best_f1 = max(best_f1, f1)

    print(f"Done. Best F1 = {best_f1:.3f}")

if __name__ == "__main__":
    main()
