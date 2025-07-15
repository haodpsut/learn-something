# file: dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class PAMAP2Dataset(Dataset):
    def __init__(self, path='pamap2_fused.npz'):
        data = np.load(path)
        self.X = torch.tensor(data['X'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
