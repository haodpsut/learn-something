import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import random
import networkx as nx

# Đường dẫn đến dữ liệu
NORMAL_DIR = './Training_Data_Master'
ATTACK_DIR = './Attack_Data_Master'

# Gán nhãn: 0 = normal, 1 = attack
def load_sequences_from_folder(folder_path, label):
    data_list = []
    for fname in tqdm(sorted(os.listdir(folder_path))):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r') as f:
                tokens = list(map(int, f.read().strip().split()))
                if len(tokens) < 2:
                    continue
                # sequence features
                seq = torch.tensor(tokens, dtype=torch.long)
                seq_len = len(seq)

                # tạo node set (unique syscalls)
                nodes = sorted(set(seq.tolist()))
                node_map = {v: i for i, v in enumerate(nodes)}
                node_feats = torch.tensor(nodes, dtype=torch.long).unsqueeze(1)

                # tạo edge index
                edge_index = []
                for i in range(seq_len - 1):
                    src = node_map[seq[i].item()]
                    dst = node_map[seq[i + 1].item()]
                    edge_index.append([src, dst])
                if not edge_index:
                    continue
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                # tạo graph data object
                data = Data(
                    x=node_feats,
                    edge_index=edge_index,
                    seq=seq,
                    seq_len=torch.tensor([seq_len]),
                    y=torch.tensor([label], dtype=torch.long)
                )
                data_list.append(data)
    return data_list

def main():
    print("Loading normal samples...")
    normal_data = load_sequences_from_folder(NORMAL_DIR, label=0)

    print("Loading attack samples...")
    attack_data = []
    for subfolder in sorted(os.listdir(ATTACK_DIR)):
        subpath = os.path.join(ATTACK_DIR, subfolder)
        if os.path.isdir(subpath):
            attack_data.extend(load_sequences_from_folder(subpath, label=1))

    # Gộp lại rồi shuffle
    all_data = normal_data + attack_data
    random.shuffle(all_data)

    # Chia train/val
    split = int(0.8 * len(all_data))
    train_data = all_data[:split]
    val_data = all_data[split:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    torch.save({'train': train_data, 'val': val_data}, 'gtfid_graph_data.pt')

if __name__ == '__main__':
    main()
