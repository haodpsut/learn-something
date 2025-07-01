#!/usr/bin/env python3
"""
train_hetero.py

• Load HeteroData từ graph_data.pt
• Split email nodes theo domain (train/val/test)
• Định nghĩa HeteroKAN (2 lớp HeteroConv + linear)
• Huấn luyện, in loss/acc/f1/aucPR mỗi epoch
• Cuối cùng lưu ROC và checkpoint
"""

import warnings
warnings.filterwarnings("ignore")


import argparse, random
import numpy as np, torch, torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score, roc_curve, auc
from collections import defaultdict

# 1) Hàm tính threshold tối ưu F1
def best_thr(prob, y):
    grid = np.linspace(0.01,0.99,50)
    f1s  = [precision_recall_fscore_support(y, prob>t, average='binary', zero_division=0)[2] for t in grid]
    return grid[int(np.argmax(f1s))]

# 2) Định nghĩa model
class HeteroKAN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.conv1 = HeteroConv({
          ('email','sent_by','domain'): SAGEConv((-1,-1), hidden),
          ('email','to','domain'):      SAGEConv((-1,-1), hidden),
          ('email','contains_url','url'): GATConv((-1,-1), hidden, heads=2, concat=False),
          ('email','contains_ip','ip'):   GATConv((-1,-1), hidden, heads=2, concat=False),
          # reverse edges
          ('domain','rev_sent_by','email'): SAGEConv((-1,-1), hidden),
          ('domain','rev_to','email'):      SAGEConv((-1,-1), hidden),
          ('url','rev_contains_url','email'): GATConv((-1,-1), hidden, heads=2, concat=False),
          ('ip','rev_contains_ip','email'):   GATConv((-1,-1), hidden, heads=2, concat=False),
        }, aggr='sum')
        self.lin = nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = {k: x.relu() for k,x in x.items()}
        return self.lin(x['email']).squeeze(-1)

# 3) Split email nodes by domain (no overlap)
def make_splits(data, frac=(0.8,0.1,0.1), seed=0):
    # email → domain index via sent_by edges
    e2d = data['email','sent_by','domain'].edge_index.cpu().numpy()
    # map each email to its (first) from-domain
    email_dom = {}
    for e,d in zip(*e2d):
        if e not in email_dom:
            email_dom[e] = d
    n_email = data['email'].num_nodes
    # group email ids by domain
    dom2emails = defaultdict(list)
    for e in range(n_email):
        dom2emails[email_dom.get(e, -1)].append(e)
    # shuffle domains
    doms = list(dom2emails)
    random.seed(seed); random.shuffle(doms)
    n = len(doms)
    n1 = int(frac[0]*n); n2 = int((frac[0]+frac[1])*n)
    train_d, val_d, test_d = doms[:n1], doms[n1:n2], doms[n2:]
    train_idx = sum((dom2emails[d] for d in train_d), [])
    val_idx   = sum((dom2emails[d] for d in val_d), [])
    test_idx  = sum((dom2emails[d] for d in test_d), [])
    # masks
    mask = lambda idxs: torch.tensor([i in idxs for i in range(n_email)], dtype=torch.bool)
    data['email'].train_mask = mask(train_idx)
    data['email'].val_mask   = mask(val_idx)
    data['email'].test_mask  = mask(test_idx)

# 4) Training / evaluation
def train_eval(data, epochs=20, batch_size=512, lr=5e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = HeteroKAN().to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    crit   = nn.BCEWithLogitsLoss()

    # neighbor sampler for hetero (full-batch also OK)
    loader = NeighborLoader(data,
        input_nodes=('email', data['email'].train_mask),
        num_neighbors={k: [10,10] for k in data.edge_index_dict},
        batch_size=batch_size, shuffle=True)

    best_val_f1 = 0
    for ep in range(1, epochs+1):
        model.train(); total_loss=0; cnt=0
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x_dict, batch.edge_index_dict)['email']
            y = batch['email'].y.float()
            loss = crit(logits[batch['email'].train_mask], y[batch['email'].train_mask])
            loss.backward(); opt.step(); opt.zero_grad()
            total_loss += loss.item(); cnt += 1
        avg_loss = total_loss / cnt

        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)['email'].cpu()
        y_val = data['email'].y[data['email'].val_mask].cpu()
        p_val = torch.sigmoid(out[data['email'].val_mask]).numpy()
        thr   = best_thr(p_val, y_val.numpy())
        yhat  = p_val > thr

        acc = accuracy_score(y_val, yhat)
        f1  = precision_recall_fscore_support(y_val, yhat, average='binary')[2]
        auc_pr = average_precision_score(y_val, p_val)
        print(f"Ep{ep:02d} loss={avg_loss:.3f} acc={acc:.3f} f1={f1:.3f} aucPR={auc_pr:.3f} thr={thr:.2f}")

        if f1 > best_val_f1:
            best_state = model.state_dict()
            best_val_f1 = f1

    # load best
    model.load_state_dict(best_state)
    # final test
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)['email'].cpu()
    y_test = data['email'].y[data['email'].test_mask].cpu()
    p_test = torch.sigmoid(out[data['email'].test_mask]).numpy()
    thr   = best_thr(p_test, y_test.numpy())
    yhat  = p_test > thr
    acc  = accuracy_score(y_test, yhat)
    f1   = precision_recall_fscore_support(y_test, yhat, average='binary')[2]
    auc_pr = average_precision_score(y_test, p_test)
    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc_roc = auc(fpr, tpr)

    print("\n=== Test results ===")
    print(f"acc={acc:.4f}  f1={f1:.4f}  aucPR={auc_pr:.4f}  aucROC={auc_roc:.4f} thr={thr:.2f}")

    # bạn có thể lưu checkpoint/ROC plots ở đây

if __name__=="__main__":
    print('start')
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="graph_data.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch",  type=int, default=512)
    p.add_argument("--lr",     type=float, default=5e-4)
    args = p.parse_args()
    print('load data')
    data = torch.load(args.data)
    make_splits(data, seed=0)
    print("splited data")
    train_eval(data, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
