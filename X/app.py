# app.py

print("--- EXECUTING SCRIPT WITH STABLE SIMULATION PATTERN (v8) ---")

from collections import OrderedDict
import warnings
import os
from typing import List, Tuple, Dict, Optional
from strategy import CEFedCSStrategy 

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# from flwr.common import Parameters # No longer needed for type hint
from utils import load_data_cifar10, get_device

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Hyperparameters
NUM_CLIENTS = 100
BATCH_SIZE = 32
NUM_ROUNDS = 50
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS = 5

# 2. Model Definition
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 3. Train/Test Functions
# Sửa hàm train
def train(net, trainloader, epochs, device):
    """Train the model and return the average loss of the last epoch."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    epoch_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
    return epoch_loss / num_batches # Trả về loss trung bình

def test(net, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy

# 4. Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, device):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)


    # Sửa hàm fit trong class FlowerClient
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Nhận lại loss từ hàm train
        last_epoch_loss = train(self.net, self.trainloader, epochs=LOCAL_EPOCHS, device=self.device)
        # Gửi loss về server
        metrics = {"loss": last_epoch_loss}
        return self.get_parameters(config={}), len(self.trainloader.dataset), metrics
    


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# 5. Helper Functions for Simulation
def get_evaluate_fn(testloader, device):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict) -> Optional[Tuple[float, Dict]]:
        net = Net().to(device)
        # THE FIX IS HERE: `parameters` is a list of NumPy arrays, not a Parameters object
        params_dict = zip(net.state_dict().keys(), [torch.from_numpy(p) for p in parameters])
        state_dict = OrderedDict(params_dict)
        net.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = test(net, testloader, device)
        print(f"Server-side evaluation round {server_round} / accuracy {accuracy} / loss {loss}")
        return loss, {"accuracy": accuracy}
    return evaluate

def client_fn_factory(trainloaders, testloader, device):
    """Return a function that creates a new client for a given CID."""
    def client_fn(cid: str) -> fl.client.Client:
        net = Net().to(device)
        trainloader = trainloaders[int(cid)]
        return FlowerClient(net, trainloader, testloader, device).to_client()
    return client_fn

# 6. Main Execution Block
if __name__ == "__main__":
    DEVICE = get_device()
    print(f"Running on device: {DEVICE}")

    # Load data
    trainloaders, testloader = load_data_cifar10(NUM_CLIENTS, BATCH_SIZE)

    # Create the client factory function
    client_fn = client_fn_factory(trainloaders, testloader, DEVICE)

    # Define the strategy
    
    strategy = CEFedCSStrategy(
        fraction_fit=1.0, # Chúng ta sẽ tự chọn client, nên để fraction_fit=1.0
        fraction_evaluate=0.0,
        min_fit_clients=CLIENTS_PER_ROUND, # Số client sẽ chọn
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(testloader, DEVICE),
    )



    # Start simulation
    print("Starting FedAvg baseline simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_gpus": 1.0 if DEVICE.type == "cuda" else 0.0},
    )

    # 7. Plot results
    print("Simulation finished. Plotting results...")
    
    rounds = [int(r) for r, metrics in history.metrics_centralized["accuracy"]]
    accuracies = [float(acc) for r, acc in history.metrics_centralized["accuracy"]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-')
    plt.title("FedAvg Baseline: Accuracy vs. Communication Rounds (Non-IID CIFAR-10)")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Model Accuracy")
    plt.grid(True)
    plt.xticks(range(0, NUM_ROUNDS + 1, 5))
    plt.ylim(0, 1)

    if not os.path.exists("figures"):
        os.makedirs("figures")
    figure_path = "figures/baseline_fedavg_accuracy.png"
    plt.savefig(figure_path, dpi=600, bbox_inches='tight')
    print(f"Results saved to {figure_path}")