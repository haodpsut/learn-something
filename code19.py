import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

NUM_CLIENTS = 3
ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LR = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dirichlet non-i.i.d. split
def dirichlet_split(dataset, num_clients, alpha=1.0):
    labels = np.array(dataset.targets)
    class_indices = [np.where(labels == i)[0] for i in range(10)]
    client_indices = [[] for _ in range(num_clients)]
    for idxs in class_indices:
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())
    return client_indices

# Data
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
client_indices = dirichlet_split(full_train, NUM_CLIENTS, alpha=1.0)
client_loaders = [DataLoader(Subset(full_train, idx), batch_size=BATCH_SIZE, shuffle=True) for idx in client_indices]

test_loader = DataLoader(datasets.CIFAR10('./data', train=False, download=True, transform=test_transform),
                         batch_size=1000, shuffle=False)

# PN và NPN
class PN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(num_features))
        self.var = nn.Parameter(torch.ones(num_features))
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
    def forward(self, x):
        mu = self.mu.view(1,-1,1,1)
        var = self.var.view(1,-1,1,1)
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        return self.scale.view(1,-1,1,1)*x_norm + self.shift.view(1,-1,1,1)

class NPN(PN):
    def __init__(self, num_features, eps=1e-5, sigma=0.2):
        super().__init__(num_features, eps)
        self.sigma = sigma
    def forward(self, x):
        noise_mu = (1 + torch.randn_like(self.mu)*self.sigma).view(1,-1,1,1)
        noise_var = (1 + torch.randn_like(self.var)*self.sigma).view(1,-1,1,1)
        mu = self.mu.view(1,-1,1,1)*noise_mu
        var = self.var.view(1,-1,1,1)*noise_var
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        return self.scale.view(1,-1,1,1)*x_norm + self.shift.view(1,-1,1,1)

# Model with configurable norm layer
def get_model(norm_layer):
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.norm1 = norm_layer(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.norm2 = norm_layer(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.norm3 = norm_layer(128)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(128*4*4, 10)
        def forward(self, x):
            x = torch.relu(self.norm1(self.conv1(x)))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.norm2(self.conv2(x)))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.norm3(self.conv3(x)))
            x = torch.max_pool2d(x, 2)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    return CNN().to(DEVICE)

# Federated training
def federated_train(norm_layer):
    global_model = get_model(norm_layer)
    loss_fn = nn.CrossEntropyLoss()
    for rnd in range(ROUNDS):
        local_states = []
        for loader in client_loaders:
            model = get_model(norm_layer)
            model.load_state_dict(global_model.state_dict())
            opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            model.train()
            for _ in range(LOCAL_EPOCHS):
                for x, y in loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    opt.step()
            local_states.append(model.state_dict())
        # Aggregation
        new_state = global_model.state_dict()
        for k in new_state:
            new_state[k] = sum(st[k] for st in local_states) / NUM_CLIENTS
        global_model.load_state_dict(new_state)
    # Evaluation
    global_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = global_model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# Run
acc_bn = federated_train(lambda c: nn.BatchNorm2d(c))
acc_pn = federated_train(lambda c: PN(c))
acc_npn = federated_train(lambda c: NPN(c))

print(f"BN Accuracy  : {acc_bn:.4f}")
print(f"PN Accuracy  : {acc_pn:.4f}")
print(f"NPN Accuracy : {acc_npn:.4f}")
