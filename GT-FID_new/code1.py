import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    sequences, labels = zip(*batch)
    # sequences là tuple các tensor kích thước khác nhau (dài khác nhau)
    # pad_sequence sẽ padding 0 để về max độ dài batch
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, labels



# ===== Dataset =====
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ===== Models =====
class GRUBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        emb = self.embedding(x)              # (B, L, E)
        _, h = self.gru(emb)                # h: (1, B, H)
        h = h.squeeze(0)                    # (B, H)
        out = self.fc(h)                    # (B, 2)
        return out

class CNNLSTMBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, cnn_channels=64, lstm_hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d = nn.Conv1d(embed_dim, cnn_channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 2)
    
    def forward(self, x):
        emb = self.embedding(x)            # (B, L, E)
        emb = emb.permute(0, 2, 1)        # (B, E, L) for Conv1d
        c = self.conv1d(emb)              # (B, C, L)
        c = c.permute(0, 2, 1)            # (B, L, C)
        _, (h, _) = self.lstm(c)          # h: (1, B, H)
        h = h.squeeze(0)                  # (B, H)
        out = self.fc(h)                  # (B, 2)
        return out

class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2, max_seq_len=100, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Khởi tạo position embedding với đúng max_seq_len:
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))  # max_seq_len = 100
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # shape: [batch_size, seq_len, embed_dim]
        # Cộng position embedding đúng slice theo seq_len:
        emb = emb + self.pos_embedding[:, :emb.size(1), :]
        out = self.transformer_encoder(emb)
        out = out.mean(dim=1)
        out = self.classifier(out)
        return out



# ===== Training & Evaluation =====
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.detach().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

# ===== Main =====
def main():
    # Paths (thay đổi cho phù hợp)
    data_dir = "preprocessed_adfa"
    seq_path = os.path.join(data_dir, "sequences.pkl")
    label_path = os.path.join(data_dir, "labels.pkl")
    vocab_path = os.path.join(data_dir, "vocab.json")

    # Load data
    with open(seq_path, "rb") as f:
        sequences = pickle.load(f)  # list of sequences (list of token ids)
    with open(label_path, "rb") as f:
        labels_list = pickle.load(f)  # list[int]

    # Chuyển multi-class -> binary: Normal=0, Attack=1
    binary_labels = [0 if lbl == 0 else 1 for lbl in labels_list]
    binary_labels = np.array(binary_labels)

    # Lọc những class có ít hơn 2 samples nếu cần ở bước này (đã làm ở split_data.py)

    # Lọc dữ liệu nếu cần (ví dụ chọn các samples dài 100 tokens)
    # Ở đây giả sử sequences có độ dài 100, hoặc truncate/pad rồi

    # Chia train/val/test stratify theo nhãn binary
    seq_trainval, seq_test, lab_trainval, lab_test = train_test_split(
        sequences, binary_labels, test_size=0.15, random_state=42, stratify=binary_labels)

    seq_train, seq_val, lab_train, lab_val = train_test_split(
        seq_trainval, lab_trainval, test_size=0.15, random_state=42, stratify=lab_trainval)

    print(f"Train: {len(seq_train)}, Val: {len(seq_val)}, Test: {len(seq_test)}")

    # Tạo Dataset và DataLoader
    batch_size = 64
    train_dataset = SequenceDataset(seq_train, lab_train)
    val_dataset = SequenceDataset(seq_val, lab_val)
    test_dataset = SequenceDataset(seq_test, lab_test)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)


    # Load vocab size
    import json
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab) + 1  # +1 for padding idx if any

    # Chọn model: GRU, CNN-LSTM, Transformer
    model_name = "Transformer"  # Thay đổi sang "CNNLSTM" hoặc "Transformer" nếu muốn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if model_name == "GRU":
        model = GRUBaseline(vocab_size)
    elif model_name == "CNNLSTM":
        model = CNNLSTMBaseline(vocab_size)
    elif model_name == "Transformer":
        #model = TransformerBaseline(vocab_size)
        model = TransformerBaseline(vocab_size=len(vocab), max_seq_len=100).to(device)
    else:
        raise ValueError("Unknown model")

    model = model.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 20
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_{model_name}.pt")
            print(f"[INFO] Model saved at epoch {epoch}")

    # Test final
    model.load_state_dict(torch.load(f"best_model_{model_name}.pt"))
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print(f"[TEST] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
