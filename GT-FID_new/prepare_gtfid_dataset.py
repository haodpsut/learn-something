import os
import torch
from torch_geometric.data import Data
from typing import List
from tqdm import tqdm

# Hyperparams
MAX_SEQ_LEN = 300
VOCAB_SIZE = 400

def read_syscall_sequence(filepath: str) -> List[int]:
    with open(filepath, "r") as f:
        return list(map(int, f.read().strip().split()))

def build_graph_from_sequence(seq: List[int]) -> Data:
    if len(seq) > MAX_SEQ_LEN:
        seq = seq[:MAX_SEQ_LEN]
    seq_len = len(seq)
    padded_seq = seq + [0] * (MAX_SEQ_LEN - len(seq))

    # Node map
    unique_nodes = sorted(set(seq))
    node_to_idx = {sid: i for i, sid in enumerate(unique_nodes)}
    x = torch.zeros(len(unique_nodes), VOCAB_SIZE)
    for sid in unique_nodes:
        if sid < VOCAB_SIZE:
            x[node_to_idx[sid], sid] = 1.0

    # Edge index
    edges = []
    for i in range(len(seq) - 1):
        s, t = seq[i], seq[i+1]
        if s in node_to_idx and t in node_to_idx:
            edges.append([node_to_idx[s], node_to_idx[t]])
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous() if edges else torch.empty((2,0), dtype=torch.long)

    return Data(x=x,
                edge_index=edge_index,
                sequence=torch.tensor(padded_seq, dtype=torch.long),
                seq_len=torch.tensor(seq_len, dtype=torch.long))

def load_all_graphs(root: str, normal_dir: str, attack_dir: str, val_ratio=0.2):
    all_graphs, all_labels = [], []

    # Load normal (label 0)
    print("Loading normal samples...")
    normal_path = os.path.join(root, normal_dir)
    for fname in tqdm(os.listdir(normal_path)):
        fpath = os.path.join(normal_path, fname)
        if os.path.isfile(fpath):
            seq = read_syscall_sequence(fpath)
            g = build_graph_from_sequence(seq)
            all_graphs.append(g)
            all_labels.append(0)

    # Load attack (label 1)
    print("Loading attack samples...")
    attack_root = os.path.join(root, attack_dir)
    for subfolder in os.listdir(attack_root):
        subpath = os.path.join(attack_root, subfolder)
        if not os.path.isdir(subpath): continue
        for fname in os.listdir(subpath):
            fpath = os.path.join(subpath, fname)
            if os.path.isfile(fpath):
                seq = read_syscall_sequence(fpath)
                g = build_graph_from_sequence(seq)
                all_graphs.append(g)
                all_labels.append(1)

    # Shuffle and split
    indices = torch.randperm(len(all_graphs))
    val_split = int(len(indices) * val_ratio)
    val_idx, train_idx = indices[:val_split], indices[val_split:]

    train_data = [all_graphs[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_data = [all_graphs[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    return train_data, train_labels, val_data, val_labels

if __name__ == "__main__":
    DATA_ROOT = ""
    train_graphs, train_y, val_graphs, val_y = load_all_graphs(
        root=DATA_ROOT,
        normal_dir="Training_Data_Master",
        attack_dir="Attack_Data_Master"
    )

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    # Save to file (optional)
    torch.save((train_graphs, train_y, val_graphs, val_y), "gtfid_graph_data.pt")
