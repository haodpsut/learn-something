from __future__ import annotations
import argparse
import json
import os
import random
from typing import List, Tuple
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (pad_sequence, pack_padded_sequence,
                                pad_packed_sequence)
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch_geometric.data import Data as GeoData, Batch as GeoBatch
from torch_geometric.nn import GCNConv, global_mean_pool

mp.set_sharing_strategy("file_system")

# ------------------------------------
# Dataset
# ------------------------------------
class SequenceGraphSample:
    def __init__(self, seq: List[int], label: int):
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.label = int(label)
        if len(seq) > 1:
            src = torch.tensor(seq[:-1], dtype=torch.long)
            dst = torch.tensor(seq[1:], dtype=torch.long)
            self.edge_index = torch.stack([src, dst], dim=0)
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)

class SyscallDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab_path: str, seq_field: str = "sequence"):
        self.samples: List[SequenceGraphSample] = []
        with open(jsonl_path) as f:
            for ln, raw in enumerate(f, 1):
                j = json.loads(raw)
                seq = j.get(seq_field) or j.get("sequence") or j.get("seq")
                if seq is None:
                    raise KeyError(f"Missing '{seq_field}' in line {ln}")
                self.samples.append(SequenceGraphSample(seq, j["label"]))
        with open(vocab_path) as f:
            self.vocab = json.load(f)
        self.num_classes = len({s.label for s in self.samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ------------------------------------
# Collate fn
# ------------------------------------
PAD_IDX = 0

def collate(batch: List[SequenceGraphSample]):
    seqs = [b.seq for b in batch]
    lens = torch.tensor([len(s) for s in seqs])
    seq_pad = pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)
    graphs = []
    for i, b in enumerate(batch):
        g = GeoData(x=b.seq.unsqueeze(1).float(), edge_index=b.edge_index)
        g.y = torch.tensor([b.label])
        graphs.append(g)
    g_batch = GeoBatch.from_data_list(graphs)
    labels = torch.tensor([b.label for b in batch])
    return seq_pad, lens, g_batch, labels

# ------------------------------------
# Model
# ------------------------------------
class GTBlock(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128,
                 lstm_hidden: int = 256, g_hidden: int = 256, fusion_dim: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(emb_dim, lstm_hidden // 2, batch_first=True,
                            bidirectional=True)
        self.temp_proj = nn.Linear(lstm_hidden, fusion_dim)
        self.temp_bn = nn.BatchNorm1d(fusion_dim)
        self.gcn1 = GCNConv(1, g_hidden)
        self.gcn2 = GCNConv(g_hidden, g_hidden)
        self.graph_proj = nn.Linear(g_hidden, fusion_dim)
        self.graph_bn = nn.BatchNorm1d(fusion_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, seq_pad: torch.Tensor, lengths: torch.Tensor, g_batch: GeoBatch):
        x = self.emb(seq_pad)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        tx = self.dropout(self.temp_bn(self.temp_proj(h)))
        gx = F.relu(self.gcn1(g_batch.x, g_batch.edge_index))
        gx = F.relu(self.gcn2(gx, g_batch.edge_index))
        gx = global_mean_pool(gx, g_batch.batch)
        gx = self.dropout(self.graph_bn(self.graph_proj(gx)))
        return F.relu(gx + tx)

class GTFID(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.block = GTBlock(vocab_size)
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, seq_pad, lens, g_batch):
        h = self.block(seq_pad, lens, g_batch)
        h = self.dropout(h)
        return self.classifier(h)

# ------------------------------------
# Train / Eval helpers
# ------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, correct, tot = 0.0, 0, 0
    for seqs, lens, g_batch, labels in loader:
        seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
        g_batch = g_batch.to(device)
        logits = model(seqs, lens, g_batch)
        loss = criterion(logits, labels)
        tot_loss += loss.item() * len(labels)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        tot += len(labels)
    return tot_loss / tot, correct / tot

def train_epoch(model, loader, optim, criterion, device):
    model.train()
    tot_loss, correct, tot = 0.0, 0, 0
    for seqs, lens, g_batch, labels in loader:
        seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
        g_batch = g_batch.to(device)
        optim.zero_grad()
        logits = model(seqs, lens, g_batch)
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()
        tot_loss += loss.item() * len(labels)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        tot += len(labels)
    return tot_loss / tot, correct / tot

# ------------------------------------
# CLI
# ------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Graph-Temporal Fusion (GT-FID) trainer")
    p.add_argument("--data", required=True, help="JSONL with sequences/labels")
    p.add_argument("--vocab", required=True, help="Path to vocab.json")
    p.add_argument("--seq_field", default="sequence",
                   help="Key name for sequence array in JSONL (default 'sequence')")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()

# ------------------------------------
# Main
# ------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = SyscallDataset(args.data, args.vocab, args.seq_field)
    labels = [s.label for s in ds.samples]
    lengths = [len(s.seq) for s in ds.samples]
    print(f"Dataset size: {len(ds)}, Num classes: {len(set(labels))}, Vocab size: {len(ds.vocab)}")
    print(f"Class distribution: {Counter(labels)}")
    print(f"Sequence lengths: min={min(lengths) if lengths else 0}, max={max(lengths) if lengths else 0}, avg={sum(lengths)/len(lengths) if lengths else 0:.2f}")

    if len(ds) == 0:
        raise ValueError("Dataset is empty. Check the input file or preprocessing steps.")

    indices = list(range(len(ds)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    train_ds = torch.utils.data.Subset(ds, train_indices)
    val_ds = torch.utils.data.Subset(ds, val_indices)

    val_labels = [ds.samples[i].label for i in val_ds.indices]
    val_lengths = [len(ds.samples[i].seq) for i in val_ds.indices]
    print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}")
    print(f"Validation class distribution: {Counter(val_labels)}")
    print(f"Validation sequence lengths: min={min(val_lengths) if val_lengths else 0}, max={max(val_lengths) if val_lengths else 0}, avg={sum(val_lengths)/len(val_lengths) if val_lengths else 0:.2f}")

    labels = [ds.samples[i].label for i in train_ds.indices]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[ds.samples[i].label] for i in train_ds.indices]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                             collate_fn=collate, num_workers=args.num_workers,
                             persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            persistent_workers=args.num_workers > 0)

    model = GTFID(vocab_size=len(ds.vocab), num_classes=len(set(labels))).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    class_counts = [Counter(labels)[i] for i in range(len(set(labels)))]
    weights = [1.0 / c for c in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=3)

    best_acc = 0.0
    patience = 10
    counter = 0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {ep:03d}: train-loss {tr_loss:.4f} acc {tr_acc:.3f} | val-loss {val_loss:.4f} acc {val_acc:.3f}")
        scheduler.step(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "gt_fid_best.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {ep}")
                break
    print(f"Best validation accuracy: {best_acc:.3f}")

if __name__ == "__main__":
    main()