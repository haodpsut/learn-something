import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler

# --- Config ---
DATA_PATH = 'data_pamap2_federated_top8_filtered.npz'
NUM_CLIENTS = 7  # đúng với dữ liệu bạn có
BATCH_SIZE = 32  # giảm batch size để cập nhật thường xuyên hơn
LOCAL_EPOCHS = 5  # tăng local epochs
ROUNDS = 20  # tăng số vòng federated learning
LR = 0.005  # tăng learning rate một chút
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load dữ liệu ---
data = np.load(DATA_PATH)
X = data['X']  # (N, 100, 52)
y = data['y']  # (N,)
client_ids = data['client_ids']  # (N,)

print(f"Loaded X shape: {X.shape}, y shape: {y.shape}, clients shape: {client_ids.shape}")

# --- Chuẩn hóa dữ liệu (normalization) theo feature trên toàn bộ dataset ---
# reshape để scaler dễ làm việc: (N*100, 52)
X_reshaped = X.reshape(-1, X.shape[2])
scaler = StandardScaler()
X_reshaped = scaler.fit_transform(X_reshaped)
X = X_reshaped.reshape(X.shape[0], X.shape[1], X.shape[2])

# --- LSTM Model ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=52, hidden_dim=128, num_layers=2, num_classes=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (hn, _) = self.lstm(x)  # hn: (num_layers, batch, hidden_dim)
        hn = hn[-1]  # take last layer hidden state (batch, hidden_dim)
        out = self.classifier(hn)
        return out

# --- Helper functions ---
def get_dataloader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_local(model, dataloader, epochs=LOCAL_EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"  Local epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")

def average_weights(w_list, sizes):
    avg_w = OrderedDict()
    total_size = sum(sizes)
    for k in w_list[0].keys():
        avg_w[k] = sum(w_list[i][k].float() * sizes[i] / total_size for i in range(len(w_list)))
    return avg_w

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total

# --- Chuẩn bị dữ liệu từng client ---
clients_data = {}
for client in np.unique(client_ids):
    idxs = np.where(client_ids == client)[0]
    clients_data[client] = (X[idxs], y[idxs])

# --- Tạo dataloader test (lấy 10% dữ liệu random) ---
np.random.seed(42)
all_indices = np.arange(len(X))
np.random.shuffle(all_indices)
split_idx = int(0.9 * len(all_indices))
train_idx = all_indices[:split_idx]
test_idx = all_indices[split_idx:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
test_loader = get_dataloader(X_test, y_test, shuffle=False)

# --- Khởi tạo global model ---
global_model = LSTMClassifier().to(DEVICE)
global_weights = global_model.state_dict()

# --- Bắt đầu vòng FL ---
for r in range(1, ROUNDS + 1):
    print(f"\n--- Round {r} ---")
    local_weights = []
    local_sizes = []
    for client in clients_data.keys():
        print(f" Training client {client}...")
        local_model = LSTMClassifier().to(DEVICE)
        local_model.load_state_dict(global_weights)
        Xc, yc = clients_data[client]
        loader = get_dataloader(Xc, yc)
        train_local(local_model, loader)
        local_weights.append(local_model.state_dict())
        local_sizes.append(len(Xc))

    # Trung bình weights theo số lượng mẫu
    avg_weights = average_weights(local_weights, local_sizes)

    global_model.load_state_dict(avg_weights)
    global_weights = global_model.state_dict()

    acc = evaluate(global_model, test_loader)
    print(f"Test accuracy after round {r}: {acc*100:.2f}%")

print("Training finished.")
