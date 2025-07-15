import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, max_seq_len=100):
    sequences, labels = zip(*batch)
    sequences_truncated = [seq[:max_seq_len] if len(seq) > max_seq_len else seq for seq in sequences]
    # Nếu seq đã tensor, chỉ clone, nếu list thì chuyển tensor
    sequences_tensors = []
    for seq in sequences_truncated:
        if isinstance(seq, torch.Tensor):
            sequences_tensors.append(seq.clone().detach())
        else:
            sequences_tensors.append(torch.tensor(seq))
    sequences_padded = pad_sequence(sequences_tensors, batch_first=True, padding_value=0)

    if sequences_padded.size(1) < max_seq_len:
        pad_size = max_seq_len - sequences_padded.size(1)
        padding = torch.zeros(sequences_padded.size(0), pad_size, dtype=torch.long)
        sequences_padded = torch.cat([sequences_padded, padding], dim=1)

    labels = torch.tensor(labels, dtype=torch.long)
    return sequences_padded, labels



# Dataset với truncate ngay khi lấy sample
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, max_seq_len=100):
        self.sequences = sequences
        self.labels = labels
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        x = torch.tensor(seq, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Model Transformer
class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2, max_seq_len=100, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))  # position embedding
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        emb = emb + self.pos_embedding[:, :emb.size(1), :]
        out = self.transformer_encoder(emb)  # [batch, seq_len, embed_dim]
        out = out.mean(dim=1)  # mean pooling
        out = self.classifier(out)
        return out


# Train & eval
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


# Main
def main():
    data_dir = "preprocessed_adfa"
    seq_path = os.path.join(data_dir, "sequences.pkl")
    label_path = os.path.join(data_dir, "labels.pkl")
    vocab_path = os.path.join(data_dir, "vocab.json")

    with open(seq_path, "rb") as f:
        sequences = pickle.load(f)
    with open(label_path, "rb") as f:
        labels_list = pickle.load(f)
    import json
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # Chuyển multi-class sang binary
    binary_labels = [0 if lbl == 0 else 1 for lbl in labels_list]
    binary_labels = np.array(binary_labels)

    # Chia train/val/test stratify theo nhãn binary
    seq_trainval, seq_test, lab_trainval, lab_test = train_test_split(
        sequences, binary_labels, test_size=0.15, random_state=42, stratify=binary_labels)

    seq_train, seq_val, lab_train, lab_val = train_test_split(
        seq_trainval, lab_trainval, test_size=0.15, random_state=42, stratify=lab_trainval)

    print(f"Train: {len(seq_train)}, Val: {len(seq_val)}, Test: {len(seq_test)}")

    max_seq_len = 100
    batch_size = 64
    train_dataset = SequenceDataset(seq_train, lab_train, max_seq_len=max_seq_len)
    val_dataset = SequenceDataset(seq_val, lab_val, max_seq_len=max_seq_len)
    test_dataset = SequenceDataset(seq_test, lab_test, max_seq_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda b: collate_fn(b, max_seq_len=max_seq_len))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda b: collate_fn(b, max_seq_len=max_seq_len))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda b: collate_fn(b, max_seq_len=max_seq_len))

    vocab_size = len(vocab) + 1  # +1 for padding idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = TransformerBaseline(vocab_size, max_seq_len=max_seq_len)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"           Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_Transformer.pt")
            print(f"[INFO] Model saved at epoch {epoch}")

    model.load_state_dict(torch.load(f"best_model_Transformer.pt"))
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    print(f"[TEST] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
