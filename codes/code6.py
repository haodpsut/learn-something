#!/usr/bin/env python3
# build_pt_only.py  –  tạo processed/data.pt từ thư mục extracted đã có sẵn

from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

ROOT = Path("data/malnet_sample")       # chỉnh nếu dùng đường dẫn khác
EXTRACT_DIR = ROOT / "extracted" / "malnet-graphs"
OUT_PT      = ROOT / "processed" / "data.pt"

edge_files = list(EXTRACT_DIR.rglob("*.edgelist"))
if not edge_files:
    raise SystemExit(f"❌ Không tìm thấy *.edgelist trong {EXTRACT_DIR}")

print(f"🔍 Found {len(edge_files):,} edgelist files – converting…")

fam2id, type2id, data_list = {}, {}, []
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

OUT_PT.parent.mkdir(parents=True, exist_ok=True)
torch.save(data_list, OUT_PT)
print(f"✅ Saved {len(data_list):,} graphs → {OUT_PT}")
