import torch
import torch.nn as nn

class ReplayBuffer:
    ...

# as outlined in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
class DQN(nn.Module):
    def __init__(self, num_actions):
        self.model = nn.Sequential([
            nn.Conv2d(1, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        ])

    def forward(self, x):
        out = model(x)
        return out
