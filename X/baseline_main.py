# baseline_main.py

from collections import OrderedDict
import warnings
import os

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import load_data_cifar10, get_device

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Define Hyperparameters
NUM_CLIENTS = 100
BATCH_SIZE = 32
NUM_ROUNDS = 50
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS = 5

# 2. Define Model (a simple CNN for CIFAR-10)
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

# 3. Define Train and Test functions
def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader, device):
    """Validate the model on the test set."""
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

# 4. Define the Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, testloader, device):
        self.cid = cid
        self.net = net.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print(f"[Client {self.cid}] Training...")
        train(self.net, self.trainloader, epochs=LOCAL_EPOCHS, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

# 5. Main Simulation Logic
if __name__ == "__main__":
    # Get device
    DEVICE = get_device()
    print(f"Running on device: {DEVICE}")
    
    # Load data
    trainloaders, testloader = load_data_cifar10(NUM_CLIENTS, BATCH_SIZE)
    
    # Function to create a client
    def client_fn(cid: str) -> FlowerClient:
        net = Net()
        trainloader = trainloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, testloader, DEVICE)

    # Define the strategy (FedAvg)
    # This is what we will customize later!
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CLIENTS_PER_ROUND / NUM_CLIENTS, # Fraction of clients to use for training
        fraction_evaluate=0.1, # Use 10% of clients for evaluation
        min_fit_clients=CLIENTS_PER_ROUND, # Minimum number of clients to train
        min_available_clients=NUM_CLIENTS, # Wait until all clients are available
        evaluate_fn=None, # We use client-side evaluation
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

    # 6. Process and plot results
    print("Simulation finished. Plotting results...")
    
    # Extract accuracy from history
    accuracies = [result[1] for _, result in history.metrics_distributed["accuracy"]]
    rounds = range(1, len(accuracies) + 1)
    
    # Create figures directory if it doesn't exist
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-')
    plt.title("FedAvg Baseline: Accuracy vs. Communication Rounds")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Model Accuracy")
    plt.grid(True)
    plt.xticks(rounds[::5]) # Show ticks every 5 rounds
    plt.ylim(0, 1)

    # Save the figure with high resolution
    figure_path = "figures/baseline_fedavg_accuracy.png"
    plt.savefig(figure_path, dpi=600)
    print(f"Results saved to {figure_path}")