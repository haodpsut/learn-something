import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ========== Dataset ==========
class PAMAP2Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========== Model ==========
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ========== Training ==========
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# ========== Main ==========
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = np.load('data_pamap2.npz')
    X, y = data['X'], data['y']
    print(f"Loaded dataset with shape X={X.shape}, y={y.shape}")
    print(f"Original unique labels: {np.unique(y)}")

    # Map labels
    unique_labels = np.unique(y)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y])
    print(f"Mapped labels: {np.unique(y)}")
    print(f"Label mapping: {label_mapping}")

    # Remove near-constant features (avoid NaN from zero variance)
    X_var = np.var(X, axis=0)
    X = X[:, X_var > 1e-8]
    print(f"Removed near-constant columns, new shape: {X.shape}")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # DataLoader
    train_dataset = PAMAP2Dataset(X_train, y_train)
    val_dataset = PAMAP2Dataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    input_dim = X.shape[1]
    num_classes = len(label_mapping)
    model = MLP(input_dim, num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    best_val_acc = 0
    for epoch in range(1, 31):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  🔥 Saved new best model at epoch {epoch}")

if __name__ == '__main__':
    main()
