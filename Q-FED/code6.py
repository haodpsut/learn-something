import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib.pyplot as plt

# ====== Step 1: Dữ liệu XOR ======
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float64)

Y = torch.tensor([0, 1, 1, 0], dtype=torch.float64)


# Vẽ và lưu dữ liệu XOR
colors = ['red' if label == 0 else 'blue' for label in Y]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.title("XOR Dataset")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.savefig("xor_dataset.png")
plt.close()
print("✔️ Đã lưu hình ảnh dữ liệu XOR vào xor_dataset.png")

# ====== Step 2: Quantum circuit (QNode) ======
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# ====== Step 3: QNN Model ======
class QNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Số layers = 1, mỗi qubit có 1 tham số
        self.q_weights = nn.Parameter(torch.randn((1, n_qubits)))

    def forward(self, x):
        # Duyệt từng mẫu và tính output quantum
        outputs = [quantum_circuit(xi, self.q_weights) for xi in x]
        return torch.stack(outputs)

# ====== Step 4: Huấn luyện ======
model = QNN()
optimizer = optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = loss_fn(outputs, Y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

# ====== Step 5: Đánh giá ======
with torch.no_grad():
    raw_preds = model(X).squeeze()
    preds = (raw_preds > 0).int()

    print("\nDự đoán:")
    for i in range(len(X)):
        print(f"  Input: {X[i].tolist()}, Pred: {int(preds[i])}, True: {int(Y[i])}")
    
    acc = (preds == Y.int()).sum().item() / len(Y)
    print(f"\n🎯 Độ chính xác: {acc * 100:.2f}%")
