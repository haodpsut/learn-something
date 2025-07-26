import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

print("Bắt đầu bài toán QNN cho phân loại XOR...")

# Tạo thư mục để lưu ảnh nếu chưa có
if not os.path.exists("figures"):
    os.makedirs("figures")

# Bước 1: Chuẩn bị dữ liệu XOR
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
Y = Y.view(-1, 1)

# Bước 2: Thiết kế Mạch Lượng tử
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    Mạch lượng tử bao gồm 3 phần:
    1. Mã hóa dữ liệu (Embedding)
    2. Tầng lượng tử biến phân (Variational Layers)
    3. Phép đo (Measurement)
    """
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# Bước 3: Xây dựng Mô hình Lượng tử với PyTorch
class QuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        n_layers = 2
        q_params = 0.01 * torch.randn(n_layers, n_qubits)
        self.q_weights = nn.Parameter(q_params)

    def forward(self, x):
        results = []
        for i in range(x.shape[0]):
            q_out = quantum_circuit(x[i], self.q_weights)
            results.append(q_out)
        
        # Chuyển đổi output [-1, 1] về [0, 1]
        return (torch.stack(results).view(-1, 1) + 1) / 2

# Bước 4: Huấn luyện Mô hình
model = QuantumNet()
optimizer = optim.Adam(model.parameters(), lr=0.3)
loss_fn = nn.BCELoss()

epochs = 100
loss_history = []

print("\nBắt đầu huấn luyện...")
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = loss_fn(predictions, Y)
    loss_history.append(loss.item())
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Huấn luyện hoàn tất!")

# Bước 5: Đánh giá kết quả
print("\nĐánh giá mô hình sau khi huấn luyện:")
with torch.no_grad():
    predictions_final = model(X)
    predicted_labels = torch.round(predictions_final)

    for i in range(len(X)):
        print(f"Input: {X[i].tolist()}, Target: {int(Y[i].item())}, Predicted: {int(predicted_labels[i].item())}, Probability: {predictions_final[i].item():.3f}")

# Bước 6: Trực quan hóa và lưu loss
figure_path = "figures/qnn_xor_loss.png"
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("Training Loss History")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.grid(True)
# Thay vì plt.show(), ta dùng plt.savefig()
plt.savefig(figure_path)
print(f"\nBiểu đồ loss đã được lưu tại: {figure_path}")
plt.close() # Đóng figure để giải phóng bộ nhớ