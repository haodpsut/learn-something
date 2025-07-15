import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- BiLSTM + Attention Model ---
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [B, T, 2H]
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # [B, T, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [B, 2H]
        return self.classifier(context)

# --- Training utilities ---
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# --- Data loading ---
from sklearn.feature_selection import VarianceThreshold

def load_data(path='data_pamap2.npz', seq_len=113):
    data = np.load(path)
    X = data['X']
    y = data['y']
    print(f"Loaded dataset with shape X={X.shape}, y={y.shape}")

    # Label normalization
    original_labels = np.unique(y)
    label_map = {l: i for i, l in enumerate(original_labels)}
    y = np.array([label_map[l] for l in y])
    print("Original labels:", original_labels)
    print("Label map:", label_map)

    # Remove near-constant columns BEFORE scaling
    selector = VarianceThreshold(threshold=1e-6)
    X = selector.fit_transform(X)
    print(f"Removed near-constant columns, new shape: {X.shape}")

    # Standardize
    X = X.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if np.isnan(X).any():
        print("❌ Error: NaNs remain in scaled data. Stopping.")
        exit(1)

    # Reshape
    assert X.shape[1] % seq_len == 0, f"Cannot reshape X.shape[1]={X.shape[1]} into ({seq_len}, -1)"
    feature_dim = X.shape[1] // seq_len
    X = X.reshape(-1, seq_len, feature_dim)

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), len(label_map)


# --- Main training ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    (X_train, X_val, y_train, y_val), num_classes = load_data()
    X_train, X_val = torch.tensor(X_train), torch.tensor(X_val)
    y_train, y_val = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)

    input_dim = X_train.shape[2]
    model = AttentionLSTM(input_dim, hidden_dim=128, num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, 31):
        train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_attention_lstm.pt')
            print(f"  🔥 Saved best model at epoch {epoch}")

if __name__ == '__main__':
    main()
