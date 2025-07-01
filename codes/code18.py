import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# ---- Configurations ----
NUM_CLIENTS = 5
CLIENT_SPLIT_SIZE = 1000
META_BATCH_SIZE = 3      # number of clients sampled per meta-update
INNER_STEPS = 1          # number of local adaptation steps
INNER_LR = 0.01
META_LR = 0.001
ROUNDS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Dataset and Client Split ----
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
indices = np.arange(len(full_dataset))
np.random.shuffle(indices)
client_indices = np.array_split(indices[:NUM_CLIENTS*CLIENT_SPLIT_SIZE], NUM_CLIENTS)

clients_loaders = []
for idx in client_indices:
    subset = Subset(full_dataset, idx)
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    clients_loaders.append(loader)

# ---- Model Definition ----
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32*5*5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

# ---- Meta-Learning + Federated ----
# Initialize global model
global_model = SimpleCNN().to(DEVICE)
meta_optimizer = optim.Adam(global_model.parameters(), lr=META_LR)
criterion = nn.CrossEntropyLoss()

for rnd in range(1, ROUNDS+1):
    meta_optimizer.zero_grad()
    # Sample a meta-batch of clients
    sampled_clients = np.random.choice(range(NUM_CLIENTS), META_BATCH_SIZE, replace=False)

    for client_id in sampled_clients:
        # Clone global model for local adaptation
        local_model = SimpleCNN().to(DEVICE)
        local_model.load_state_dict(global_model.state_dict())
        local_optimizer = optim.SGD(local_model.parameters(), lr=INNER_LR)

        # Get two batches: support (inner) and query (meta)
        data_iter = iter(clients_loaders[client_id])
        support_x, support_y = next(data_iter)
        query_x, query_y     = next(data_iter)
        support_x, support_y = support_x.to(DEVICE), support_y.to(DEVICE)
        query_x, query_y     = query_x.to(DEVICE), query_y.to(DEVICE)

        # Inner loop adaptation
        local_optimizer.zero_grad()
        support_preds = local_model(support_x)
        support_loss = criterion(support_preds, support_y)
        support_loss.backward()
        local_optimizer.step()

        # Compute loss on query set
        query_preds = local_model(query_x)
        query_loss = criterion(query_preds, query_y)

        # Accumulate meta-gradient
        query_loss.backward()

    # Meta update: apply accumulated gradients to global model
    meta_optimizer.step()

    print(f"Round {rnd}/{ROUNDS} meta-updated.")

# Save global model
torch.save(global_model.state_dict(), 'global_maml_fl.pth')
print('Meta-FL Training Complete.')
