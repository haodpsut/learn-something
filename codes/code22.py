#!/usr/bin/env python3
# hgat_95_debug.py – Debuggable HGAT to resolve NaN issues and achieve ~95% accuracy

import argparse, math, torch
from torch import nn
from torch_geometric.datasets import MalNetTiny
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool, LayerNorm
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint

# ---------- node-feature ----------
def build_node_feature(g):
    din  = degree(g.edge_index[1], g.num_nodes).unsqueeze(1)
    dout = degree(g.edge_index[0], g.num_nodes).unsqueeze(1)
    size = torch.full((g.num_nodes, 1), math.log(g.num_nodes + 1e-6))
    g.x = torch.cat([din, dout, size], dim=1).float()

# ---------- sanitize dataset ----------
def sanitize(ds):
    cleaned = []
    for g in ds:
        if g.num_nodes == 0: continue
        if not hasattr(g, "y_family"): g.y_family = getattr(g, "y", torch.zeros(1, dtype=torch.long))
        if not hasattr(g, "y_type"): g.y_type = getattr(g, "type", getattr(g, "type_id", torch.zeros(1, dtype=torch.long)))
        build_node_feature(g); cleaned.append(g)
    return cleaned

# ---------- HGAT with debug logs ----------
class HGATDebug(nn.Module):
    def __init__(self, in_dim, n_fam, n_type, hidden=256, heads=8, layers=8, type_emb=16, ckpt=True):
        super().__init__()
        self.ckpt = ckpt
        self.type_emb = nn.Embedding(n_type, type_emb)
        self.convs, self.lns, self.res = nn.ModuleList(), nn.ModuleList(), []
        last = in_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(last, hidden, heads=heads, dropout=0.3))
            self.lns.append(LayerNorm(hidden*heads)); self.res.append(last==hidden*heads)
            last = hidden*heads
        self.out_dim = last*3 + type_emb
        self.cls_fam = nn.Linear(self.out_dim, n_fam)
        self.cls_type= nn.Linear(self.out_dim, n_type)

    def _block(self, conv, ln, skip, h, edge):
        z = torch.relu(conv(h, edge)); z = ln(z)
        return z + h if skip else z

    def forward(self, x, edge, batch, y_type):
        h = x
        for conv, ln, skip in zip(self.convs, self.lns, self.res):
            if self.training and self.ckpt:
                h = checkpoint(self._block, conv, ln, skip, h, edge, use_reentrant=False)
            else:
                h = self._block(conv, ln, skip, h, edge)
        m1, m2, m3 = global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)
        g = torch.cat([m1, m2, m3, self.type_emb(y_type)], dim=1)
        return self.cls_fam(g), self.cls_type(g)

# ---------- train / eval with step order fixed ----------
def train_epoch(model, loader, opt, scaler, sched, dev, alpha, use_amp, accum_steps, clip_norm):
    model.train(); total_loss=0; total_graphs=0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt.zero_grad()
    for i, d in enumerate(loader,1):
        d = d.to(dev)
        ctx = torch.amp.autocast(device_type='cuda') if use_amp else nullcontext()
        with ctx:
            yf, yt = model(d.x, d.edge_index, d.batch, d.y_type.squeeze())
            loss = (criterion(yf, d.y_family.squeeze()) + alpha*criterion(yt, d.y_type.squeeze()))/accum_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if i%accum_steps==0 or i==len(loader):
            if use_amp:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(opt); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                opt.step()
            # **scheduler step after optimizer step**
            sched.step(); opt.zero_grad()
        total_loss += loss.item(); total_graphs+=d.num_graphs
        # debug NaN in loss or grads
        if torch.isnan(loss): raise RuntimeError(f"NaN loss at batch {i}")
    return total_loss/total_graphs

@torch.no_grad()
def eval_acc(model, loader, dev):
    model.eval(); corr=tot=0
    for d in loader:
        d = d.to(dev)
        pred,_ = model(d.x,d.edge_index,d.batch,d.y_type.squeeze())
        corr += (pred.argmax(1)==d.y_family.squeeze()).sum().item(); tot+=d.num_graphs
    return corr/tot

# ---------- main ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/malnet_tiny')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--ckpt', action='store_true', default=True)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--clip_norm', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision to avoid NaNs')
    return p.parse_args()
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/malnet_tiny'); p.add_argument('--device',default='cuda');
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch_size',type=int,default=4);
    p.add_argument('--accum_steps',type=int,default=4); p.add_argument('--hidden',type=int,default=256);
    p.add_argument('--heads',type=int,default=8); p.add_argument('--layers',type=int,default=8);
    p.add_argument('--ckpt',action='store_true',default=True); p.add_argument('--alpha',type=float,default=0.5);
    p.add_argument('--clip_norm',type=float,default=0.5); p.add_argument('--lr',type=float,default=5e-5);
    return p.parse_args()

def main():
    args=get_args(); dev=torch.device(args.device); use_amp=(dev.type=='cuda')
    print('⏬ Loading MalNet-Tiny splits …')
    train_ds=sanitize(MalNetTiny(args.root,split='train')); val_ds=sanitize(MalNetTiny(args.root,split='val'))
    tl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,pin_memory=use_amp);
    vl=DataLoader(val_ds,batch_size=args.batch_size,pin_memory=use_amp)
    n_fam=MalNetTiny(args.root).num_classes; n_type=len({g.y_type.item() for g in train_ds})
    model=HGATDebug(3,n_fam,n_type,hidden=args.hidden,heads=args.heads,layers=args.layers,ckpt=args.ckpt).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=args.lr,epochs=args.epochs,steps_per_epoch=len(tl)//args.accum_steps+1,pct_start=0.1)
    scaler=torch.amp.GradScaler() if use_amp else None
    best=0; wait=0; patience=args.layers
    for ep in range(1,args.epochs+1):
        loss=train_epoch(model,tl,opt,scaler,sched,dev,args.alpha,use_amp,args.accum_steps,args.clip_norm)
        acc=eval_acc(model,vl,dev)
        print(f'[E{ep:03d}] loss={loss:.4f} | val_acc={acc:.3f}')
        if acc>best: best,wait=acc,0; torch.save(model.state_dict(),'best_hgat_debug.pt')
        else: wait+=1; print(f' wait {wait}/{patience}');
        if wait>=patience: print('⇡ Early-stopped.'); break
    print('✔ Best val-accuracy:',best)

if __name__=='__main__': main()
