import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# Fix seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Constants
MAX_LEN = 150
PAD_ID = 0
VOCAB_SIZE = 400
BATCH_SIZE = 64
EPOCHS = 30
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 2

# ====== Dataset ======
class SystemCallDataset(Dataset):
    def __init__(self, data_list, label_list, max_len=MAX_LEN):
        self.samples = []
        for seq in data_list:
            if len(seq) < max_len:
                seq = seq + [PAD_ID] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            self.samples.append(seq)
        self.labels = label_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.LongTensor(self.samples[idx]), torch.LongTensor([self.labels[idx]])

def load_sequences_from_folder(folder_path, label):
    all_data, all_labels = [], []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    seq = list(map(int, f.read().strip().split()))
                    all_data.append(seq)
                    all_labels.append(label)
    return all_data, all_labels


def build_dataset():
    normal_path = './Training_Data_Master'
    attack_path = './Attack_Data_Master'
    normal_seqs, normal_labels = load_sequences_from_folder(normal_path, 0)
    attack_seqs, attack_labels = load_sequences_from_folder(attack_path, 1)
    all_seqs = normal_seqs + attack_seqs
    all_labels = normal_labels + attack_labels
    combined = list(zip(all_seqs, all_labels))
    random.shuffle(combined)
    all_seqs, all_labels = zip(*combined)
    return list(all_seqs), list(all_labels)


def split_dataset(dataset):
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])


# ====== GRU Model ======
class GRUBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# ====== Transformer Model ======
class TransformerBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # mean pooling
        return self.fc(x)


def train(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device).squeeze()
            optimizer.zero_grad()
            out = model(seq)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            _, pred = out.max(1)
            correct += (pred == label).sum().item()
            total += label.size(0)
        acc = correct / total
        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}] Train Acc: {acc:.4f} | Val Acc: {val_acc:.4f}")
    return model

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for seq, label in loader:
            seq, label = seq.to(device), label.to(device).squeeze()
            out = model(seq)
            _, pred = out.max(1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    return correct / total


# ====== Main ======
def run_baselines():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    sequences, labels = build_dataset()
    dataset = SystemCallDataset(sequences, labels)
    train_set, val_set, test_set = split_dataset(dataset)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    print("\n=== GRU Baseline ===")
    model_gru = GRUBaseline()
    trained_gru = train(model_gru, train_loader, val_loader, device)
    test_acc_gru = evaluate(trained_gru, test_loader, device)
    print(f"[TEST] GRU Accuracy: {test_acc_gru:.4f}")

    print("\n=== Transformer Baseline ===")
    model_trans = TransformerBaseline()
    trained_trans = train(model_trans, train_loader, val_loader, device)
    test_acc_trans = evaluate(trained_trans, test_loader, device)
    print(f"[TEST] Transformer Accuracy: {test_acc_trans:.4f}")


if __name__ == "__main__":
    run_baselines()
