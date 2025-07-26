import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import matplotlib.pyplot as plt
import os
import time # Thêm thư viện time để đo thời gian

# --- Thiết lập môi trường ---
torch.set_default_dtype(torch.float32)

print("Bắt đầu Bài toán MNIST 10 lớp (Phiên bản chạy nhanh)...")

# --- Thiết lập ---
if not os.path.exists("figures"):
    os.makedirs("figures")
n_qubits = 10
batch_size = 32 # Giảm batch size để phù hợp với lượng data ít hơn
# CẬP NHẬT: Giảm số lượng mẫu
n_train_samples = 800
n_test_samples = 200
# CẬP NHẬT: Giảm số epoch
epochs = 50

# --- Tải và Xử lý Dữ liệu ---
train_dataset_full = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset_full = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

train_subset = Subset(train_dataset_full, range(n_train_samples))
test_subset = Subset(test_dataset_full, range(n_test_samples))

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

print(f"Đã chuẩn bị xong dữ liệu: {len(train_loader.dataset)} mẫu huấn luyện, {len(test_loader.dataset)} mẫu kiểm thử.")

# --- Mạch Lượng tử (10 qubits) ---
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# --- Mô hình Hybrid CNN-QNN (10 lớp) ---
class HybridCNNQNN(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(256, 120), nn.ReLU(),
            nn.Linear(120, n_qubits)
        )
        n_layers = 2
        weight_shape = (n_layers, n_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.01)

    def forward(self, x):
        features = self.cnn_feature_extractor(x)
        scaled_features = torch.pi * torch.sigmoid(features)
        q_out_list = quantum_circuit(scaled_features.double(), self.q_weights.double())
        return torch.stack(q_out_list, dim=1).float()

# --- Huấn luyện và Đánh giá ---
model = HybridCNNQNN(n_qubits=n_qubits)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

loss_history, accuracy_history = [], []

print("\nBắt đầu huấn luyện mô hình CNN-QNN 10 lớp...")
start_time = time.time() # Bắt đầu đếm giờ

for epoch in range(epochs):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, dim=1)
            total += batch_Y.size(0)
            correct += (predicted == batch_Y).sum().item()
            
    avg_accuracy = 100 * correct / total
    accuracy_history.append(avg_accuracy)
    
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.2f}%, Time: {epoch_time:.2f}s")

total_time = time.time() - start_time
print(f"\nHuấn luyện hoàn tất! Tổng thời gian: {total_time:.2f}s")

# --- Trực quan hóa ---
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color=color)
ax1.plot(loss_history, color=color, marker='o'); ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', linewidth=0.5)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(accuracy_history, color=color, marker='x'); ax2.tick_params(axis='y', labelcolor=color)
fig.suptitle('Training (CNN-QNN, 10-Class, Quick Test)', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
figure_path = "figures/cnn_qnn_mnist_10_class_quick.png"
plt.savefig(figure_path)
print(f"\nBiểu đồ huấn luyện đã được lưu tại: {figure_path}")
plt.close()