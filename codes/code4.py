import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import argparse
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings
# ===== 1. Define KAN Layer =====
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(KANLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return torch.sin(self.fc(x))  # Simple nonlinear basis

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
    embeddings = torch.load(embedding_path)
    labels = torch.load(label_path)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    edge_index = []
    for i in range(len(embeddings)):
        for j in indices[i][1:]:
            edge_index.append([i, j])
            edge_index.append([j, i])
    edge_index = torch.tensor(edge_index).t().contiguous()

    return Data(x=embeddings, edge_index=edge_index, y=labels)

# ===== 4. Train and Evaluate =====
def run(data, hidden_dim=64, epochs=30, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
    model = KANGuard(data.num_node_features, hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    train_idx = perm[:int(0.7*num_nodes)]
    val_idx = perm[int(0.7*num_nodes):int(0.85*num_nodes)]
    test_idx = perm[int(0.85*num_nodes):]

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_idx], data.y[train_idx].float())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = torch.sigmoid(out[val_idx])
            val_pred = (val_out > 0.5).int()
            val_f1 = f1_score(data.y[val_idx].cpu(), val_pred.cpu())
            val_acc = accuracy_score(data.y[val_idx].cpu(), val_pred.cpu())
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_out = torch.sigmoid(model(data.x, data.edge_index)[test_idx])
        pred = (test_out > 0.5).int()
        acc = accuracy_score(data.y[test_idx].cpu(), pred.cpu())
        f1 = f1_score(data.y[test_idx].cpu(), pred.cpu())
        roc = roc_auc_score(data.y[test_idx].cpu(), test_out.cpu())
    print(f"\n[Test Results] Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC AUC: {roc:.4f}")

# ===== 5. Entry Point =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, default="trec_embeddings.pt")
    parser.add_argument('--label_path', type=str, default="trec_labels.pt")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    print("Loading data...")
    data = load_data(args.embedding_path, args.label_path, args.k)

    print("Training model...")
    run(data, epochs=args.epochs)
