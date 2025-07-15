import os
import json
import pickle
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data



def build_vocab(dataset_dir):
    counter = Counter()
    for folder in ['Training_Data_Master', 'Validation_Data_Master', 'Attack_Data_Master']:
        folder_path = os.path.join(dataset_dir, folder)
        for root, _, files in os.walk(folder_path):
            for f in files:
                try:
                    with open(os.path.join(root, f), encoding='utf-8', errors='ignore') as file:
                        tokens = file.read().split()
                        counter.update(tokens)
                except Exception as e:
                    print(f"[WARN] Skipping file {f} due to error: {e}")
    tok2idx = {tok: idx+1 for idx, tok in enumerate(counter)}  # Start from 1
    tok2idx['<PAD>'] = 0
    return tok2idx

def load_data(dataset_dir, tok2idx):
    sequences, labels = [], []
    label_dict = {"Normal": 0}
    label_counter = 1
    
    def read_sequence(file_path, tok2idx):
        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                return [tok2idx[tok] for tok in f.read().split() if tok in tok2idx]
        except Exception as e:
            print(f"[WARN] Error reading {file_path}: {e}")
            return []


    # Normal data
    for folder in ['Training_Data_Master', 'Validation_Data_Master']:
        for file in os.listdir(os.path.join(dataset_dir, folder)):
            path = os.path.join(dataset_dir, folder, file)
            seq = read_sequence(path, tok2idx)
            sequences.append(seq)
            labels.append(0)  # Normal

    # Attack data
    attack_dir = os.path.join(dataset_dir, 'Attack_Data_Master')
    for attack_type in os.listdir(attack_dir):
        attack_path = os.path.join(attack_dir, attack_type)
        if not os.path.isdir(attack_path): continue
        if attack_type not in label_dict:
            label_dict[attack_type] = label_counter
            label_counter += 1
        for file in os.listdir(attack_path):
            path = os.path.join(attack_path, file)
            seq = read_sequence(path, tok2idx)
            sequences.append(seq)
            labels.append(label_dict[attack_type])

    return sequences, labels, label_dict

def build_transition_graph(seq, vocab_size):
    edges = []
    for i in range(len(seq) - 1):
        u, v = seq[i], seq[i+1]
        edges.append((u, v))
    if not edges:
        edges.append((seq[0], seq[0]))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    unique_nodes = sorted(set(seq))
    x = torch.nn.functional.one_hot(torch.tensor(unique_nodes), num_classes=vocab_size).float()
    return Data(x=x, edge_index=edge_index)

def save_processed_data(save_dir, sequences, labels, tok2idx, label_dict):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'sequences.pkl'), 'wb') as f:
        pickle.dump(sequences, f)
    with open(os.path.join(save_dir, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)
    with open(os.path.join(save_dir, 'vocab.json'), 'w') as f:
        json.dump(tok2idx, f)
    with open(os.path.join(save_dir, 'label_dict.json'), 'w') as f:
        json.dump(label_dict, f)

def main():
    dataset_path = "."  # 👈 sửa đường dẫn tại đây
    save_path = "preprocessed_adfa"

    print("[INFO] Building vocabulary...")
    tok2idx = build_vocab(dataset_path)
    vocab_size = len(tok2idx)
    print(f"[INFO] Vocabulary size: {vocab_size}")

    print("[INFO] Loading and encoding data...")
    sequences, labels, label_dict = load_data(dataset_path, tok2idx)
    print(f"[INFO] Loaded {len(sequences)} samples. Classes: {label_dict}")

    print("[INFO] Saving to disk...")
    save_processed_data(save_path, sequences, labels, tok2idx, label_dict)
    print(f"[INFO] Data saved to: {save_path}")

if __name__ == "__main__":
    main()
