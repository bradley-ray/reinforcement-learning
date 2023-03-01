import torch
import torch.nn as nn

import numpy as np

class ReplayBuffer:
    def __init__(self):
        self.__buf = []

    def add(self, transition):
        self.__buf.append(transition)

    def sample(self, size):
        idxs = np.random.choice(len(self.__buf), size, replace=False)
        return [self.__buf[i] for i in idxs]

# as outlined in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(1, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        ])

    def forward(self, x):
        out = self.model(x)
        return out
