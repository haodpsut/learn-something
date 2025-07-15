import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

# ========== FL helper functions ==========
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
        total_samples += X_batch.size(0)
    return total_loss / total_samples, total_correct / total_samples

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total_samples += X_batch.size(0)
    return total_loss / total_samples, total_correct / total_samples

def average_weights(models):
    """FedAvg: average the parameters of models."""
    avg_model = {}
    for k in models[0].state_dict().keys():
        avg_model[k] = torch.mean(torch.stack([m.state_dict()[k].float() for m in models]), dim=0)
    for m in models:
        m.load_state_dict(avg_model)
    return models[0]

# ========== Main FL training ==========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load("data_pamap2_federated.npz")
    X = data["X"]          # (N, D)
    y_full = data["y"]     # (N, 3) last col is client_id

    print(f"Loaded X shape: {X.shape}, y shape: {y_full.shape}")

    # Labels are assumed in y_full[:, 0], y_full[:,1] are other label info - adapt if needed
    labels = y_full[:, 0].astype(int)
    client_ids = y_full[:, -1].astype(int)

    unique_clients = np.unique(client_ids)
    print(f"Clients found: {unique_clients}")

    input_dim = X.shape[1]
    num_classes = len(np.unique(labels))

    # Create data per client
    client_data = {}
    for c in unique_clients:
        idx = np.where(client_ids == c)[0]
        client_data[c] = {
            "X": X[idx],
            "y": labels[idx]
        }
        print(f"Client {c}: {len(idx)} samples")

    # Initialize global model
    global_model = MLP(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # FL parameters
    num_rounds = 10
    local_epochs = 1
    batch_size = 64
    lr = 1e-3

    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Federated Round {rnd} ---")
        local_models = []
        local_losses = []
        local_accuracies = []

        # For each client, train locally
        for c in unique_clients:
            model = MLP(input_dim, num_classes).to(device)
            model.load_state_dict(global_model.state_dict())  # sync global weights

            optimizer = optim.Adam(model.parameters(), lr=lr)

            dataset = PAMAP2Dataset(client_data[c]["X"], client_data[c]["y"])
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(local_epochs):
                loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)

            print(f"Client {c} local training loss: {loss:.4f}, acc: {acc:.4f}")
            local_models.append(model)
            local_losses.append(loss)
            local_accuracies.append(acc)

        # FedAvg aggregation
        global_model = average_weights(local_models)

        # Evaluate global model on all data combined (optional)
        all_dataset = PAMAP2Dataset(X, labels)
        all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size)
        val_loss, val_acc = evaluate(global_model, all_loader, criterion, device)
        print(f"Global model evaluation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Save final global model
    torch.save(global_model.state_dict(), "fl_global_model.pth")
    print("Training finished, global model saved.")

if __name__ == "__main__":
    main()
