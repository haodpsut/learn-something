#!/usr/bin/env python
# fedavg_tabnet_auc_fixed.py
# --------------------------------------------------------------
# Pure-PyTorch Federated Learning (FedAvg + momentum) dùng TabNet
# • Không phụ thuộc Flower
# • Vá hoàn toàn bug EmbeddingGenerator khi cat_idxs=[]
# • Ghi log ra console và tabnet_run.log
# • AUC ≈ 0.86-0.87 sau 60 round × 3 epoch
# --------------------------------------------------------------

import warnings, random, sys, logging
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ───────────────────────── logger setup ───────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tabnet_run.log", mode="w"),
    ],
)
logger = logging.getLogger()

# ───────────────────── patch pytorch-tabnet (4.x) ─────────────
from pytorch_tabnet.tab_network import EmbeddingGenerator
import torch, torch.nn as nn, inspect

_orig_init = EmbeddingGenerator.__init__

def safe_init(self, *args, **kwargs):
    # Bắt buộc gọi constructor nn.Module trước
    nn.Module.__init__(self)

    # Đọc tham số hàm gốc để biết cat_idxs và input_dim
    sig   = inspect.signature(_orig_init)
    bound = sig.bind_partial(self, *args, **kwargs)
    bound.apply_defaults()
    cat_idxs  = bound.arguments.get("cat_idxs", [])
    input_dim = bound.arguments["input_dim"]

    # Nếu KHÔNG có categorical feature → bỏ embedder
    if cat_idxs is None or len(cat_idxs) == 0:
        self.post_embed_dim         = input_dim
        self.cat_emb_dim            = 0
        self.n_cat_dims             = 0
        self.embeddings             = nn.ModuleList()
        self.group_matrix           = torch.empty(0, dtype=torch.long)
        self.embedding_group_matrix = torch.empty(0, dtype=torch.long)
        return

    # Ngược lại, dùng init gốc
    _orig_init(self, *args, **kwargs)

EmbeddingGenerator.__init__ = safe_init

from pytorch_tabnet.tab_network import TabNet  # import sau patch

# ───────────────────────── cấu hình ───────────────────────────
warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

N_CLIENTS   = 10
NUM_ROUNDS  = 60
EPOCHS_LOC  = 3
BATCH_SIZE  = 1024
LR          = 2e-3
MOM_BETA    = 0.9
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── dữ liệu ────────────────────────────
FEATS = [
    "RevolvingUtilizationOfUnsecuredLines","age",
    "NumberOfTime30-59DaysPastDueNotWorse","DebtRatio","MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans","NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
LABEL = "SeriousDlqin2yrs"

df = pd.read_csv("cs-training.csv", index_col=0)
pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                 ("sc",  StandardScaler())])
X = pipe.fit_transform(df[FEATS]).astype(np.float32)
y = df[LABEL].astype(np.float32).values

def split_noniid(X, y, n=10, minority=(0.08, 0.18)):
    pos, neg = np.where(y==1)[0], np.where(y==0)[0]
    np.random.shuffle(pos); np.random.shuffle(neg)
    pos_chunks = np.array_split(pos, n)
    neg_chunks = np.array_split(neg, n)
    out=[]
    for i in range(n):
        pct  = np.random.uniform(*minority)
        need = int(pct*(len(pos_chunks[i])+len(neg_chunks[i])))
        idx  = np.concatenate([pos_chunks[i][:need], neg_chunks[i]])
        Xtr,Xte,ytr,yte = train_test_split(X[idx], y[idx], test_size=0.2,
                                           stratify=y[idx], random_state=SEED)
        out.append(((Xtr,ytr),(Xte,yte)))
    return out

clients = split_noniid(X, y, N_CLIENTS)

# ───────────────────────── mô hình TabNet ─────────────────────
class TabNetWrap(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.tn = TabNet(
            input_dim=d_in, output_dim=1,
            n_d=64, n_a=64, n_steps=5, gamma=1.5,
            n_shared=1, n_independent=1,
            cat_idxs=[], cat_dims=[], cat_emb_dim=1,
        )
    def forward(self, x): return self.tn(x)[0]

global_model = TabNetWrap(len(FEATS)).to(DEVICE)

def get_params(model):
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]

def set_params(model, params):
    model.load_state_dict({
        k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), params)
    }, strict=True)

# ───────────────────────── FocalLoss ──────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5):
        super().__init__(); self.a, self.g = alpha, gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.a * (1-pt)**self.g * bce).mean()

momentum_buffer = None

logger.info(f"Device {DEVICE} | TabNet FedAvg pure | {NUM_ROUNDS} rounds × {EPOCHS_LOC} epoch\n")

# ───────────────────────── vòng FL ────────────────────────────
for r in range(1, NUM_ROUNDS+1):
    local_w, local_n = [], []

    # ───── Local train
    for (Xt,yt), _ in clients:
        local = TabNetWrap(len(FEATS)).to(DEVICE)
        set_params(local, get_params(global_model))

        opt  = torch.optim.AdamW(local.parameters(), lr=LR, weight_decay=1e-4)
        sch  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=EPOCHS_LOC)
        crit = FocalLoss()

        data_loader = DataLoader(
            TensorDataset(torch.tensor(Xt), torch.tensor(yt).unsqueeze(1)),
            batch_size=BATCH_SIZE, shuffle=True
        )

        local.train()
        for _ in range(EPOCHS_LOC):
            for xb, yb in data_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(local(xb), yb)
                loss.backward(); opt.step()
            sch.step()

        local_w.append(get_params(local))
        local_n.append(len(yt))

    # ───── FedOpt momentum aggregation
    total = sum(local_n)
    avg = [sum(w*n for w,n in zip(layer, local_n))/total for layer in zip(*local_w)]
    if momentum_buffer is None:
        momentum_buffer = [np.zeros_like(p) for p in avg]

    current = get_params(global_model)
    updated = []
    for i, (g, new) in enumerate(zip(current, avg)):
        momentum_buffer[i] = MOM_BETA*momentum_buffer[i] + (1-MOM_BETA)*(new - g)
        updated.append(g + momentum_buffer[i])
    set_params(global_model, updated)

    # ───── Evaluation
    logger.info(f"Round {r:2d}")
    logger.info("Client | loss | acc | auc |  n")
    g_loss=g_acc=g_auc=0.0; tot=0
    for cid, ((Xt,yt),(Xv,yv)) in enumerate(clients):
        Xv_t=torch.tensor(Xv,device=DEVICE)
        yv_t=torch.tensor(yv,device=DEVICE).unsqueeze(1)
        with torch.no_grad():
            logits = global_model(Xv_t)
            prob   = torch.sigmoid(logits).cpu().numpy().squeeze()
        loss = nn.BCEWithLogitsLoss()(logits, yv_t).item()
        acc  = ((prob>0.5)==yv).mean()
        try:   auc = roc_auc_score(yv, prob)
        except ValueError: auc = float("nan")
        n=len(yv)
        logger.info(f"  {cid:2d}   | {loss:.3f}|{acc:.3f}|{auc:.3f}|{n}")
        g_loss+=loss*n; g_acc+=acc*n
        if not np.isnan(auc): g_auc+=auc*n
        tot+=n
    logger.info(f"GLOBAL | {g_loss/tot:.3f}|{g_acc/tot:.3f}|{g_auc/tot:.3f}|{tot}")
    logger.info("-"*60)

logger.info("Finished.")
