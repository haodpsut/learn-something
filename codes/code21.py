import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simple DDPG for continuous-phase RIS sum-rate maximization
# Single-user SISO via RIS only (no direct link)
# Action: continuous phase shifts in [-pi, pi]
# Objective: maximize |h2^H diag(e^{j phi}) h1|^2

# Environment definition
gamma = 0.99

class RISEnv:
    def __init__(self, N=8):
        self.N = N
        self.dtype = torch.cfloat

    def reset(self):
        # sample random channels
        self.h1 = (torch.randn(self.N) + 1j*torch.randn(self.N)) / np.sqrt(2)
        self.h2 = (torch.randn(self.N) + 1j*torch.randn(self.N)) / np.sqrt(2)
        # state: real-imag concatenated
        state = torch.cat([torch.view_as_real(self.h1), torch.view_as_real(self.h2)], dim=1).flatten()
        return state

    def step(self, action):
        # action: tensor (N,) continuous in [-pi, pi]
        phi = action
        phases = torch.exp(1j * phi)
        combined = (self.h2.conj() * phases * self.h1).sum()
        reward = (combined.abs()**2).item()
        # no next state dynamics; use new random channel each step
        next_state = self.reset()
        done = False
        return next_state, reward, done, {}

# Actor network\ class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()  # output in [-1,1]
        )

    def forward(self, state):
        return self.net(state)

# Critic network\ class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# Replay buffer
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return torch.stack(state), torch.stack(action), torch.tensor(reward), torch.stack(next_state)

    def __len__(self):
        return len(self.buffer)

# DDPG Agent setup
def train():
    env = RISEnv(N=8)
    state_dim = 4 * env.N  # real and imag for h1,h2
    action_dim = env.N

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    target_actor = Actor(state_dim, action_dim)
    target_critic = Critic(state_dim, action_dim)

    # copy weights\    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

    buffer = ReplayBuffer()
    batch_size = 64
    tau = 0.005

    # Training loop
    for episode in range(200):
        state = env.reset().float()
        for step in range(50):
            # select action
            with torch.no_grad():
                raw_action = actor(state)
                action = raw_action * np.pi  # scale to [-pi,pi]

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state.float())
            state = next_state.float()

            if len(buffer) > batch_size:
                s, a, r, ns = buffer.sample(batch_size)
                # Critic update
                with torch.no_grad():
                    target_actions = target_actor(ns) * np.pi
                    q_next = target_critic(ns, target_actions)
                    q_target = r.unsqueeze(1) + gamma * q_next
                q_current = critic(s, a)
                critic_loss = nn.MSELoss()(q_current, q_target)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                # Actor update
                pred_action = actor(s) * np.pi
                actor_loss = -critic(s, pred_action).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # Soft update targets
                for param, target in zip(actor.parameters(), target_actor.parameters()):
                    target.data.copy_(tau * param.data + (1 - tau) * target.data)
                for param, target in zip(critic.parameters(), target_critic.parameters()):
                    target.data.copy_(tau * param.data + (1 - tau) * target.data)

        if episode % 10 == 0:
            print(f"Episode {episode}: Last reward {reward:.4f}")

    # Output final actor for inference
    torch.save(actor.state_dict(), 'ris_actor.pth')
    print("Training complete. Actor saved to ris_actor.pth")

if __name__ == '__main__':
    train()
