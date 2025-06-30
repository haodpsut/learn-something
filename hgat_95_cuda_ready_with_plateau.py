#!/usr/bin/env python3
# hgat_95_cuda_ready_with_plateau.py – HGAT with CUDA, augmentations, Optuna, ensemble, and early‐stop on plateau

import argparse, math, random
import torch, optuna, numpy as np
from torch import nn
from torch_geometric.datasets import MalNetTiny
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, dropout_adj
from torch_geometric.nn import (
    GATv2Conv, global_mean_pool, global_max_pool, global_add_pool, LayerNorm
)
from torch.utils.checkpoint import checkpoint
from contextlib import nullcontext

# ---------------- augmentations ----------------
def augment_graph(g, edge_drop_p, feat_mask_p, device):
    g = g.to(device)
    edge_index, _ = dropout_adj(
        g.edge_index, p=edge_drop_p,
        num_nodes=g.num_nodes, training=True
    )
    g.edge_index = edge_index
    mask = (torch.rand_like(g.x) > feat_mask_p).float()
    g.x = g.x * mask
    return g

# ---------------- model ----------------
class HGAT(nn.Module):
    def __init__(self, in_dim, n_fam, n_type, hidden, heads, layers, type_emb, dropout):
        super().__init__()
        self.type_emb = nn.Embedding(n_type, type_emb)
        self.convs, self.lns, self.res = nn.ModuleList(), nn.ModuleList(), []
        last = in_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(last, hidden, heads=heads, dropout=dropout))
            self.lns.append(LayerNorm(hidden * heads))
            self.res.append(last == hidden * heads)
            last = hidden * heads
        self.out_dim = last * 3 + type_emb
        self.cls_fam  = nn.Linear(self.out_dim, n_fam)
        self.cls_type = nn.Linear(self.out_dim, n_type)

    def _block(self, conv, ln, skip, h, edge):
        z = torch.relu(conv(h, edge))
        z = ln(z)
        return z + h if skip else z

    def forward(self, x, edge, batch, y_type):
        h = x
        for conv, ln, skip in zip(self.convs, self.lns, self.res):
            h = checkpoint(self._block, conv, ln, skip, h, edge, use_reentrant=False)
        m1 = global_mean_pool(h, batch)
        m2 = global_max_pool(h, batch)
        m3 = global_add_pool(h, batch)
        g = torch.cat([m1, m2, m3, self.type_emb(y_type)], dim=1)
        return self.cls_fam(g), self.cls_type(g)

# ---------------- dataset sanitize ----------------
def sanitize(ds):
    out = []
    for g in ds:
        if g.num_nodes == 0: continue
        if not hasattr(g, "y_family"):
            g.y_family = getattr(g, "y", torch.zeros(1, dtype=torch.long))
        if not hasattr(g, "y_type"):
            g.y_type = getattr(g, "type",
                               getattr(g, "type_id", torch.zeros(1, dtype=torch.long)))
        g.x = torch.cat([
            degree(g.edge_index[1], g.num_nodes).unsqueeze(1),
            degree(g.edge_index[0], g.num_nodes).unsqueeze(1),
            torch.full((g.num_nodes, 1), math.log(g.num_nodes + 1e-6))
        ], 1).float()
        out.append(g)
    return out

# ---------------- training / evaluation ----------------
def train_one_epoch(model, loader, optimizer, device, alpha, edge_p, feat_p):
    model.train()
    total_loss = 0.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for g in loader:
        g = augment_graph(g, edge_p, feat_p, device)
        optimizer.zero_grad()
        yf, yt = model(g.x, g.edge_index, g.batch, g.y_type.squeeze())
        loss = criterion(yf, g.y_family.squeeze()) + alpha * criterion(yt, g.y_type.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    corr = tot = 0
    for g in loader:
        g = g.to(device)
        yf, _ = model(g.x, g.edge_index, g.batch, g.y_type.squeeze())
        corr += (yf.argmax(1) == g.y_family.squeeze()).sum().item()
        tot += g.num_graphs
    return corr / tot

# ---------------- Optuna objective with plateau early-stop ----------------
def objective(trial):
    # hyperparams
    hidden    = trial.suggest_categorical("hidden", [128,256,512])
    heads     = trial.suggest_categorical("heads", [4,8,16])
    layers    = trial.suggest_int("layers", 4, 10)
    lr        = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    edge_p    = trial.suggest_float("edge_p", 0.0, 0.1)
    feat_p    = trial.suggest_float("feat_p", 0.0, 0.1)
    alpha     = trial.suggest_float("alpha", 0.1, 1.0)
    type_emb  = trial.suggest_categorical("type_emb", [8,16,32])
    dropout   = trial.suggest_float("dropout", 0.1, 0.5)
    wd        = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)

    device = torch.device("cuda")
    train_ds = sanitize(MalNetTiny("data/malnet_tiny", split="train"))
    val_ds   = sanitize(MalNetTiny("data/malnet_tiny", split="val"))
    tl = DataLoader(train_ds, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)
    vl = DataLoader(val_ds, batch_size=16, pin_memory=True, num_workers=4)

    model = HGAT(3, MalNetTiny("data/malnet_tiny").num_classes,
                 len({g.y_type.item() for g in train_ds}),
                 hidden, heads, layers, type_emb, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=lr*0.1
    )

    best_val = 0.0
    wait = 0
    patience = 10  # early-stop if no improvement in 10 epochs

    for epoch in range(1, 101):
        train_one_epoch(model, tl, optimizer, device, alpha, edge_p, feat_p)
        val_acc = evaluate(model, vl, device)
        scheduler.step()

        # Print progress
        print(f"[Trial {trial.number:03d}] Epoch {epoch:03d} → val_acc = {val_acc:.4f}")

        # Update best/wait counters
        if val_acc > best_val:
            best_val = val_acc
            wait = 0
        else:
            wait += 1

        # Early-stop on plateau
        if wait >= patience:
            print(f"No improvement in {patience} epochs → early stopping trial {trial.number}")
            break

        # Optuna pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val

# ---------------- Ensemble ----------------
def run_ensemble(params, seeds=[0,1,2,3,4]):
    device = torch.device("cuda")
    train_ds = sanitize(MalNetTiny("data/malnet_tiny", split="train"))
    val_ds   = sanitize(MalNetTiny("data/malnet_tiny", split="val"))
    tl = DataLoader(train_ds, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)
    vl = DataLoader(val_ds, batch_size=16, pin_memory=True, num_workers=4)

    all_logits = []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = HGAT(3, MalNetTiny("data/malnet_tiny").num_classes,
                     len({g.y_type.item() for g in train_ds}),
                     params["hidden"], params["heads"],
                     params["layers"], params["type_emb"], params["dropout"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=params["lr"]*0.1
        )

        # train each member
        for epoch in range(1, 51):
            train_one_epoch(model, tl, optimizer, device,
                            params["alpha"], params["edge_p"], params["feat_p"])
            scheduler.step()

        # collect logits
        logits = []
        for g in vl:
            g = g.to(device)
            yf, _ = model(g.x, g.edge_index, g.batch, g.y_type.squeeze())
            logits.append(yf.softmax(1).cpu())
        all_logits.append(torch.cat(logits, dim=0))

    ensemble = torch.stack(all_logits).mean(0)
    y_true = torch.cat([g.y_family for g in val_ds], dim=0)
    acc = (ensemble.argmax(1) == y_true).float().mean().item()
    print("Ensemble ACC:", acc)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print("Best params:", study.best_params, "Val ACC:", study.best_value)
    run_ensemble(study.best_params) 
