import torch
import torch.nn as nn
from torch.utils.data.dataset import IterableDataset
from torch.distributions.categorical import Categorical

def get_reward(rewards):
    n = len(rewards)
    r2g = np.zeros_like(rewards)
    for i in reversed(range(n)):
        r2g[i] = rewards[i] + (r2g[i+1] if i+1 < n else 0)

def get_advantage():
    ...

def get_mse(pred, tgt):
    return (pred-tgt)**2

class SimpleNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim):
        super().__init__()
        self.ff1 = nn.Linear(obs_dim, hid_dim)
        self.ff2 = nn.Linear(hid_dim, act_dim)

    def forward(self, obs):
        out = self.ff1(obs).relu()
        out = self.ff2(out)
        return out

class Agent:
    def __init__(self, env, hid_dim):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = int(env.action_space.n)
        self.policy = SimpleNetwork(obs_dim, act_dim, hid_dim)
        self.value = SimpleNetwork(obs_dim, act_dim, hid_dim)

    def step(self):
        ...

    def train_step(self, eps, policy_opt, value_opt):

        r = ...
        advangtage = get_advantage()

        policy_opt.zero_grad()
        policy_loss = (r*advantage, r.clip(1-eps, 1+eps)*advatnage).min()
        policy_loss.backward()
        policy_opt.step()

        value_opt.zero_grad()
        value_loss = get_mse(values, rewards)
        value_loss.backward()
        value_opt.step()
