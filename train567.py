import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import argparse
import warnings
import os
warnings.filterwarnings("ignore")  # Suppress warnings

# ===== 1. Define KAN Layer =====
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(KANLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.sin(self.fc(x))

# ===== 2. Define KANGuard =====
class KANGuard(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(KANGuard, self).__init__()
        self.kan = KANLayer(in_features, hidden_dim)
        self.gcn = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_features)

    def forward(self, x, edge_index):
        x = self.kan(x)
        x = F.relu(self.gcn(x, edge_index))
        return self.classifier(x).squeeze()

# ===== 3. Load Data =====
def load_data(embedding_path, label_path, k=10):
    # Load labels first to get number of nodes
    labels = torch.load(label_path).long()
    n = len(labels)

    # Load embeddings
    if embedding_path.endswith('.npy'):
        # raw memmap without header: compute dims via file size
        file_size = os.path.getsize(embedding_path)
        dtype_size = np.dtype('float32').itemsize
        total_dim = file_size // (n * dtype_size)
        embs = np.memmap(embedding_path, dtype='float32', mode='r', shape=(n, total_dim))
        embeddings = torch.from_numpy(np.asarray(embs)).float()
    else:
        embeddings = torch.load(embedding_path).float()

    # Build k-NN graph on embeddings
    emb_np = embeddings.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(emb_np)
    _, indices = nbrs.kneighbors(emb_np)

    # Create bidirectional edge_index
    edges = []
    for i, neigh in enumerate(indices):
        for j in neigh[1:]:
            edges.append([i, j])
            edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=embeddings, edge_index=edge_index, y=labels)

# ===== 4. Train and Evaluate =====
def run(data, hidden_dim=64, epochs=30, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = KANGuard(data.num_node_features, hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Split indices
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_idx = perm[:int(0.7 * num_nodes)]
    val_idx   = perm[int(0.7 * num_nodes):int(0.85 * num_nodes)]
    test_idx  = perm[int(0.85 * num_nodes):]

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_idx], data.y[train_idx].float())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = torch.sigmoid(out[val_idx])
            val_pred = (val_out > 0.5).long()
        val_acc = accuracy_score(data.y[val_idx].cpu(), val_pred.cpu())
        val_f1  = f1_score(data.y[val_idx].cpu(), val_pred.cpu())
        print(f"[Epoch {epoch:02d}] Loss: {loss.item():.4f}  Val Acc: {val_acc:.4f}  Val F1: {val_f1:.4f}")

    model.eval()
    with torch.no_grad():
        test_out = torch.sigmoid(model(data.x, data.edge_index)[test_idx])
        test_pred = (test_out > 0.5).long()
    acc = accuracy_score(data.y[test_idx].cpu(), test_pred.cpu())
    f1  = f1_score(data.y[test_idx].cpu(), test_pred.cpu())
    roc = roc_auc_score(data.y[test_idx].cpu(), test_out.cpu())
    print(f"\n[Test Results] Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC AUC: {roc:.4f}")

# ===== 5. Entry Point =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, default="embeddings.npy",
                        help="Path to embeddings file (.npy raw or .pt)")
    parser.add_argument('--label_path', type=str, default="labels.pt",
                        help="Path to labels (.pt)")
    parser.add_argument('--epochs', type=int, default=30, help="Training epochs")
    parser.add_argument('--k', type=int, default=10, help="k for k-NN graph")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden layer size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    print("Loading data...")
    data = load_data(args.embedding_path, args.label_path, k=args.k)

    print("Training KANGuard...")
    run(data, hidden_dim=args.hidden_dim, epochs=args.epochs, lr=args.lr)
