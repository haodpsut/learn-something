# t3_fixed.py – PennyLane + PyTorch XOR, khởi tạo đúng chuẩn

import math
import pennylane as qml
import torch
from torch import nn

###############################
# 1) Dữ liệu XOR
###############################
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)
Y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

###############################
# 2) Thiết bị lượng tử
###############################
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)           # "lightning.gpu" nếu có CUDA

###############################
# 3) QNode
###############################
@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs * math.pi, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

###############################
# 4) TorchLayer với init_method “đúng chuẩn”
###############################
weight_shapes = {"weights": (6, n_qubits, 3)}   # 6 layer, 3 tham số/qubit

def uniform_0_2pi(tensor):
    """Ghi tensor bằng giá trị U(0, 2π) rồi trả về tensor đó"""
    return tensor.uniform_(0.0, 2 * math.pi)

qlayer = qml.qnn.TorchLayer(
    circuit,
    weight_shapes,
    init_method=uniform_0_2pi
)

###############################
# 5) Mạng lai
###############################
class HybridXOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = qlayer
        self.clf = nn.Linear(2, 1)

    def forward(self, x):
        x = self.q(x)            # [-1, 1]
        return self.clf(x)       # logits

torch.manual_seed(0)             # tái lập
model = HybridXOR()

###############################
# 6) Huấn luyện
###############################
opt  = torch.optim.Adam(model.parameters(), lr=0.2)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(800):
    opt.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, Y)
    loss.backward()
    opt.step()

    if (epoch + 1) % 80 == 0:
        with torch.no_grad():
            probs = torch.sigmoid(logits).squeeze().tolist()
        print(f"Epoch {epoch+1:4d} | Loss {loss.item():.4f} | Probs {probs}")

###############################
# 7) Kiểm thử
###############################
with torch.no_grad():
    probs = torch.sigmoid(model(X)).squeeze()
    preds = torch.round(probs)

print("\nDự đoán cuối cùng:", preds.tolist())
print("Xác suất :", probs.tolist())
