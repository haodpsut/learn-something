import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from neal import SimulatedAnnealingSampler

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
num_clients = 10
k = 5  # number of clients to select per round

# --- CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # (batch, 16, 24, 24)
        x = F.max_pool2d(x, 2)          # (batch, 16, 12, 12)
        x = F.relu(self.conv2(x))       # (batch, 32, 8, 8)
        x = F.max_pool2d(x, 2)          # (batch, 32, 4, 4)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- Data Loader ---
def get_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# --- Simulate Client Delay ---
def simulate_clients(num_clients=10):
    np.random.seed(42)
    bandwidths = np.random.uniform(1, 10, num_clients)  # Mbps
    model_size = 1  # MB
    delays = model_size * 8 / bandwidths  # seconds
    return bandwidths, delays

# --- QUBO Builder ---
def build_qubo(delays, k, lam=10):
    N = len(delays)
    Q = {}
    for i in range(N):
        Q[(i, i)] = delays[i] + lam * (1 - 2 * k + 1)  # xi^2 = xi
        for j in range(i + 1, N):
            Q[(i, j)] = 2 * lam
    return Q

# --- Solve QUBO using Simulated Annealing ---
def solve_qubo(Q, num_reads=50):
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=100)
    
    # Get best solution
    best_sample = response.first.sample
    selected = np.array([best_sample[i] for i in sorted(best_sample)])
    return selected

# --- Federated Training Round ---
def fl_round(global_model, clients, client_data, selected, lr=0.01, local_epochs=1):
    global_weights = global_model.state_dict()
    selected_weights = []

    for i in range(len(clients)):
        if selected[i] == 1:
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_weights)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
            loader = torch.utils.data.DataLoader(client_data[i], batch_size=32, shuffle=True)

            local_model.train()
            for epoch in range(local_epochs):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = F.cross_entropy(local_model(x), y)
                    loss.backward()
                    optimizer.step()
            selected_weights.append(local_model.state_dict())

    # Aggregate weights
    for key in global_weights:
        global_weights[key] = torch.stack([w[key] for w in selected_weights], 0).mean(0)
    global_model.load_state_dict(global_weights)
    return global_model

# --- Evaluation ---
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# === Main Execution ===
if __name__ == "__main__":
    train_dataset, test_dataset = get_data()
    client_data = torch.utils.data.random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    # Simulate bandwidth and delay
    bandwidths, delays = simulate_clients(num_clients)
    print("Bandwidths (Mbps):", np.round(bandwidths, 2))
    print("Delays (s):", np.round(delays, 3))

    # Build and solve QUBO
    Q = build_qubo(delays, k)
    selected = solve_qubo(Q)
    print("Selected clients:", selected)
    print("Total delay of selection:", np.round(np.dot(delays, selected), 3))

    # Train Federated Learning
    global_model = SimpleCNN().to(device)
    for round_num in range(5):
        global_model = fl_round(global_model, range(num_clients), client_data, selected)
        acc = evaluate(global_model, test_loader)
        print(f"[Round {round_num+1}] Test Accuracy: {acc:.4f}")
