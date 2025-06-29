import torch
import numpy as np

# Simple GA for discrete-phase RIS sum-rate maximization
# Single-user SISO via RIS only (no direct link)
# Objective: maximize |h2^T diag(e^{j phi}) h1|^2

# System parameters
dtype = torch.cfloat
data_device = 'cpu'

# Number of RIS elements
N = 16
# Discrete phase levels
L = 8  # e.g., 8-level
phase_set = torch.exp(1j * 2 * np.pi * torch.arange(L) / L).to(dtype)

# Random channels
h1 = (torch.randn(N, device=data_device) + 1j * torch.randn(N, device=data_device)) / np.sqrt(2)
h2 = (torch.randn(N, device=data_device) + 1j * torch.randn(N, device=data_device)) / np.sqrt(2)

# Genetic Algorithm parameters
pop_size = 50
num_generations = 100
p_mutation = 0.1
p_crossover = 0.8

def initialize_population():
    # Each individual: tensor of shape (N,) with phase indices [0, L)
    return torch.randint(0, L, (pop_size, N), device=data_device)

def fitness(pop):
    # pop: (pop_size, N)
    # map indices to phase values
    phases = phase_set[pop]  # (pop_size, N)
    # compute combined channel
    # batch compute inner product
    h1_rep = h1.unsqueeze(0).expand(pop_size, -1)
    h2_rep = h2.unsqueeze(0).expand(pop_size, -1)
    # elementwise multi then sum
    combined = (h2_rep.conj() * phases * h1_rep).sum(dim=1)
    # fitness = squared magnitude
    return combined.abs()**2  # (pop_size,)

def select(pop, fit):
    # roulette wheel selection
    prob = fit / fit.sum()
    idx = torch.multinomial(prob, pop_size, replacement=True)
    return pop[idx]

def crossover(pop):
    new_pop = pop.clone()
    for i in range(0, pop_size, 2):
        if torch.rand(1).item() < p_crossover and i+1 < pop_size:
            point = torch.randint(1, N, (1,)).item()
            new_pop[i, point:], new_pop[i+1, point:] = pop[i+1, point:].clone(), pop[i, point:].clone()
    return new_pop

def mutate(pop):
    mask = torch.rand(pop.shape, device=data_device) < p_mutation
    random_genes = torch.randint(0, L, pop.shape, device=data_device)
    pop[mask] = random_genes[mask]
    return pop

# Main GA loop
pop = initialize_population()
best_fit = 0
best_ind = None

for gen in range(num_generations):
    fit = fitness(pop)
    max_fit, idx = torch.max(fit, dim=0)
    if max_fit > best_fit:
        best_fit = max_fit
        best_ind = pop[idx].clone()
    # selection
    pop = select(pop, fit)
    # crossover & mutation
    pop = crossover(pop)
    pop = mutate(pop)
    if (gen+1) % 10 == 0:
        print(f"Gen {gen+1}: Best fitness = {best_fit.item():.4f}")

# Output best phase
best_phases = phase_set[best_ind]
print("Optimal phase shifts:", best_phases)
print("Max channel gain:", best_fit.item())
