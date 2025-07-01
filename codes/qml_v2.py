import math, torch, pennylane as qml
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

#################### 1) DATA ####################
pre = transforms.Compose([
    transforms.ToTensor(),                # 1×28×28 in [0,1]
    transforms.Resize((4, 4)),            # 1×4×4
    transforms.Lambda(lambda x: x.view(-1))   # 16-D vector
])

train_raw = datasets.MNIST("data", train=True,  download=True, transform=pre)
test_raw  = datasets.MNIST("data", train=False, download=True, transform=pre)

train_idx = [i for i, (_, y) in enumerate(train_raw) if y in (0, 1)]
test_idx  = [i for i, (_, y) in enumerate(test_raw)  if y in (0, 1)]

train_loader = DataLoader(Subset(train_raw, train_idx), batch_size=32, shuffle=True)
test_loader  = DataLoader(Subset(test_raw,  test_idx),  batch_size=32)

#################### 2) QUANTUM CIRCUIT ####################
n_qubits, n_layers = 4, 6
dev = qml.device("default.qubit", wires=n_qubits)         # dùng "lightning.gpu" nếu có CUDA

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def circuit(inputs, weights):
    # 16-chiều  →  4-qubit (2**4 = 16)
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),
                           normalize=True, pad_with=0.0)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(0))]                    # LIST → shape (batch, 1)

weight_shapes = {"weights": (n_layers, n_qubits, 3)}

def init_0_2pi(t):           # Pennylane sẽ gọi và truyền tensor cần ghi
    return t.uniform_(0.0, 2 * math.pi)

qlayer = qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_0_2pi)

#################### 3) HYBRID MODEL ####################
class QMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.q  = qlayer           # (batch, 1)
        self.fc = nn.Linear(1, 1)  # logit

    def forward(self, x):
        x = self.q(x)              # shape (batch, 1)
        return self.fc(x).squeeze()  # → (batch,)

torch.manual_seed(0)
model = QMNIST()
opt   = torch.optim.Adam(model.parameters(), lr=0.02)
lossf = nn.BCEWithLogitsLoss()

#################### 4) TRAIN ####################
for epoch in range(8):            # tăng thêm epoch nếu muốn
    model.train()
    for xb, yb in train_loader:
        opt.zero_grad()
        logits = model(xb)            # (batch,)
        loss   = lossf(logits, yb.float())
        loss.backward()
        opt.step()
    print(f"Epoch {epoch+1}: loss {loss.item():.4f}")

#################### 5) TEST ####################
model.eval()
correct = total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = (torch.sigmoid(model(xb)) > 0.5).int()
        correct += (preds == yb).sum().item()
        total   += yb.size(0)

print(f"Test accuracy: {100*correct/total:.2f}%")
