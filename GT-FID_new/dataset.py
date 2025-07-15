import torch
from torch.utils.data import Dataset
import pickle
import json
import os

class ADFALDDataset(Dataset):
    def __init__(self, data_dir, max_len=100):
        with open(os.path.join(data_dir, 'sequences.pkl'), 'rb') as f:
            self.sequences = pickle.load(f)
        with open(os.path.join(data_dir, 'labels.pkl'), 'rb') as f:
            self.labels = pickle.load(f)

        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Pad or truncate to fixed length
        if len(seq) < self.max_len:
            seq = seq + [0] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]

        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)
