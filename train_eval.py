#!/usr/bin/env python3
"""
train_eval.py  —  KANGuard & baseline (full-batch, CUDA)

  • KANGuard        • GraphSAGE   • 2-layer GCN
  • MLP             • RandomForest

Each epoch prints: loss | acc_val | f1_val | aucPR | thr_opt.
ROC PNG + CSV và checkpoint lưu trong plots/ và checkpoints/.
"""

import os, warnings, csv, random, argparse
warnings.filterwarnings("ignore")

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             average_precision_score, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt, pandas as pd, numpy as np

# --------- device & seed ----------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

# --------- helpers ----------------------------------------------------------
def best_thr(prob_t, y_t):
    p, y = prob_t.detach().cpu().numpy(), y_t.detach().cpu().numpy()
    grid = np.linspace(0.01, 0.90, 61)          # bước 0.015
    f1s  = [precision_recall_fscore_support(y, p>t, average='binary',
                                            zero_division=0)[2] for t in grid]
    return float(grid[int(np.argmax(f1s))])

def metrics(prob_t, y_t, thr):
    p, y = prob_t.detach().cpu().numpy(), y_t.detach().cpu().numpy()
    yhat  = p > thr
    acc   = accuracy_score(y, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, yhat, average='binary', zero_division=0)
    auc_pr = average_precision_score(y, p)
    fpr, tpr, _ = roc_curve(y, p); auc_roc = auc(fpr, tpr)
    return acc, prec, rec, f1, auc_pr, (fpr, tpr, auc_roc)

def save_roc(fpr, tpr, auc_roc, name):
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_roc:.3f}")
    plt.plot([0,1],[0,1],"--",lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – {name}"); plt.legend()
    plt.savefig(f"plots/roc_{name}.png", dpi=150); plt.close()
    with open(f"plots/roc_{name}.csv","w",newline="") as fp:
        csv.writer(fp).writerows([["fpr","tpr"], *zip(fpr,tpr)])

# --------- models -----------------------------------------------------------
class KAN(nn.Module):
    def __init__(self, d, Q):
        super().__init__()
        self.a = nn.Parameter(torch.randn(Q))
        self.b = nn.Parameter(torch.randn(Q))
        self.c = nn.Parameter(torch.zeros(Q))
        self.psi = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(nn.Linear(1,8), nn.ReLU(), nn.Linear(8,1))
                for _ in range(d)])
            for _ in range(Q)])
    def forward(self, h):
        return torch.stack(
            [self.a[q]*torch.tanh(self.b[q]*sum(fun(h[:,i:i+1]).squeeze(-1)
                                                for i,fun in enumerate(self.psi[q]))
                                   + self.c[q])
             for q in range(len(self.a))], -1).sum(-1)

class KANGuard(nn.Module):
    def __init__(self, d_in, h=128):
        super().__init__()
        self.c1 = SAGEConv(d_in, h)
        self.c2 = SAGEConv(h, h)
        self.norm = nn.LayerNorm(h)
        self.kan  = KAN(h, h//2)                 # Q = 64
    def forward(self, d):
        x = F.relu(self.c1(d.x, d.edge_index))
        x = self.c2(x, d.edge_index)
        x = self.norm(x)
        return self.kan(x)

class SAGEOnly(nn.Module):
    def __init__(self, d, h=128):
        super().__init__(); self.c1=SAGEConv(d,h); self.c2=SAGEConv(h,h); self.out=nn.Linear(h,1)
    def forward(self,d):
        x=F.relu(self.c1(d.x,d.edge_index)); x=self.c2(x,d.edge_index); return self.out(x).squeeze(-1)

class TwoGCN(nn.Module):
    def __init__(self,d,h=128):
        super().__init__(); self.c1=GCNConv(d,h); self.c2=GCNConv(h,h); self.out=nn.Linear(h,1)
    def forward(self,d):
        x=F.relu(self.c1(d.x,d.edge_index)); x=self.c2(x,d.edge_index); return self.out(x).squeeze(-1)

def make_mlp(d): return nn.Sequential(nn.Linear(d,256), nn.ReLU(), nn.Linear(256,1))

# --------- train full-batch --------------------------------------------------
def train_full(model, data, split, name, epochs, pos_w):
    tr, val, _ = split
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt  = torch.optim.AdamW(model.parameters(), lr=5e-5)  # LR nhỏ ổn định
    best_f1, pat = 0., 0
    for ep in range(1, epochs+1):
        model.train()
        logit = model(data.to(DEVICE))[tr]
        loss  = crit(logit, data.y[tr].float().to(DEVICE))
        loss.backward(); opt.step(); opt.zero_grad()
        # ----- validation -----
        model.eval(); 
        with torch.no_grad():
            log_val = model(data.to(DEVICE))[val].cpu()
        thr = best_thr(torch.sigmoid(log_val), data.y[val])
        acc,_,_,f1,auc_pr,_ = metrics(torch.sigmoid(log_val), data.y[val], thr)
        print(f"{name:<10} Ep{ep:02d} loss={loss.item():.3f} acc={acc:.3f} f1={f1:.3f} aucPR={auc_pr:.3f} thr={thr:.2f}")
        if f1 > best_f1: best_state, best_f1, pat = model.state_dict(), f1, 0
        else: pat += 1
        if pat == 10: break
    model.load_state_dict(best_state)

def eval_full(model, data, split, name):
    _, val, test = split
    model.eval(); 
    with torch.no_grad():
        log_val  = model(data.to(DEVICE))[val].cpu()
        thr      = best_thr(torch.sigmoid(log_val), data.y[val])
        log_test = model(data.to(DEVICE))[test].cpu()
    acc,pre,rec,f1,auc_pr,(fpr,tpr,auc_roc) = metrics(torch.sigmoid(log_test),
                                                      data.y[test], thr)
    save_roc(fpr,tpr,auc_roc,name)
    return acc,pre,rec,f1,auc_pr,auc_roc

# --------- main -------------------------------------------------------------
def main(a):
    os.makedirs("checkpoints", exist_ok=True)
    data = torch.load(a.data)
    d_in = data.x.size(1); split = data.split
    pos   = (data.y==1).sum(); neg = (data.y==0).sum()
    pos_w = ((neg/pos)*1.0).float().to(DEVICE)       # nhẹ hơn

    results = []

    def run(model_cls, name):
        model = model_cls(d_in).to(DEVICE)
        train_full(model, data, split, name, a.epochs, pos_w)
        torch.save(model.state_dict(), f"checkpoints/{name}.pt")
        results.append((name,)+eval_full(model, data, split, name))

    run(KANGuard,  "KANGuard")
    run(SAGEOnly,  "GraphSAGE")
    run(TwoGCN,    "GCN")

    # ---- MLP ----
    tr,val,test = split
    Xtr,ytr=data.x[tr],data.y[tr]; Xval,yval=data.x[val],data.y[val]
    Xte,yte=data.x[test],data.y[test]
    mlp = make_mlp(d_in).to(DEVICE)
    crit= nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(mlp.parameters(), lr=8e-4)
    best_f1,pat = 0.,0
    for ep in range(1, a.epochs+1):
        mlp.train(); opt.zero_grad()
        logit = mlp(Xtr.to(DEVICE)).squeeze(-1)
        loss  = crit(logit, ytr.float().to(DEVICE))
        loss.backward(); opt.step()
        mlp.eval(); 
        with torch.no_grad():
            log_val = mlp(Xval.to(DEVICE)).squeeze(-1).cpu()
        thr = best_thr(torch.sigmoid(log_val), yval)
        acc,_,_,f1,auc_pr,_ = metrics(torch.sigmoid(log_val), yval, thr)
        print(f"MLP        Ep{ep:02d} loss={loss.item():.3f} acc={acc:.3f} f1={f1:.3f} aucPR={auc_pr:.3f} thr={thr:.2f}")
        if f1>best_f1: best_state,best_f1,pat=mlp.state_dict(),f1,0
        else: pat+=1
        if pat==10: break
    mlp.load_state_dict(best_state); torch.save(best_state,"checkpoints/MLP.pt")
    pr=torch.sigmoid(mlp(Xte.to(DEVICE)).squeeze(-1).cpu())
    acc,pre,rec,f1,auc_pr,(fpr,tpr,auc_roc)=metrics(pr, yte.cpu(), thr)
    save_roc(fpr,tpr,auc_roc,"MLP")
    results.append(("MLP",acc,pre,rec,f1,auc_pr,auc_roc))

    # ---- RandomForest ----
    rf=RandomForestClassifier(n_estimators=400,n_jobs=-1,random_state=0)
    rf.fit(Xtr.cpu().numpy(), ytr.cpu().numpy())
    prob=torch.tensor(rf.predict_proba(Xte.cpu().numpy())[:,1])
    thr=best_thr(prob, yte.cpu())
    acc,pre,rec,f1,auc_pr,(fpr,tpr,auc_roc)=metrics(prob, yte.cpu(), thr)
    save_roc(fpr,tpr,auc_roc,"RandomForest")
    results.append(("RandomForest",acc,pre,rec,f1,auc_pr,auc_roc))

    # -------- summary --------
    df=pd.DataFrame(results, columns=["Model","Acc","Prec","Rec","F1","AUC_PR","AUC_ROC"])
    print("\n========= SUMMARY =========")
    print(df.to_string(index=False,float_format="{:.4f}".format))

# --------- CLI --------------------------------------------------------------
if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",   default="graph_data.pt")
    p.add_argument("--epochs", type=int, default=60)     # 60 epoch cho full-batch
    p.add_argument("--batch",  type=int, default=1024)   # batch chỉ dùng cho MLP
    main(p.parse_args())
