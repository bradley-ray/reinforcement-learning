import torch
import torch.nn as nn
import torchvision.transforms as T

import cv2
import numpy as np
import os

def transforms(env, device='cpu'):
    obs = cv2.resize(env.render(), (110,84))
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = torch.from_numpy(obs).float().reshape((1, 84, 110)).to(device)
    return obs

class DQNSimple(nn.Module):
    def __init__(self, obs_dim, act_dim, h_dim):
        super().__init__()
        self.ff1 = nn.Linear(obs_dim, h_dim)
        self.ff2 = nn.Linear(h_dim, act_dim)

    def forward(self, obs):
        out = self.ff1(obs).relu()
        out = self.ff2(out)
        return out

class DQNPixel(nn.Module):
    def __init__(self, act_dim, h_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.ff1 = nn.Linear(3456, h_dim)
        self.ff2 = nn.Linear(h_dim, act_dim)

    def forward(self, obs):
        out = self.conv1(obs).relu()
        out = self.conv2(out).relu().flatten(1)
        out = self.ff1(out).relu()
        out = self.ff2(out)
        return out

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device='cpu'):
        self.device = device
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.obs_buf = torch.zeros((self.capacity, *obs_dim), requires_grad=False).float().to(self.device)
        self.act_buf = torch.zeros((self.capacity, 1), requires_grad=False).long().to(self.device)
        self.rew_buf = torch.zeros((self.capacity, 1), requires_grad=False).float().to(self.device)
        self.done_buf = torch.zeros((self.capacity, 1), requires_grad=False).float().to(self.device)
        self.obs_n_buf = torch.zeros((self.capacity, *obs_dim), requires_grad=False).float().to(self.device)

    def append(self, exp):
        self.obs_buf[self.idx] = exp[0]
        self.act_buf[self.idx] = exp[1]
        self.rew_buf[self.idx] = exp[2]
        self.done_buf[self.idx] = exp[3]
        self.obs_n_buf[self.idx] = exp[4]

        self.size = min(self.size+1, self.capacity)
        self.idx = (self.idx+1) % self.capacity

    def sample(self, bs):
        assert(self.size >= bs)

        idxs = np.random.choice(self.size, bs, replace=False)
        return (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rew_buf[idxs],
            self.done_buf[idxs],
            self.obs_n_buf[idxs],
        )


class Agent:
    def __init__(self, env, model, *args, device='cpu'):
        self.device = device
        self.pixels = model != DQNSimple
        self.env = env
        if self.pixels:
            self.memory = ReplayBuffer(10_000, (1, 84, 110), int(env.action_space.n), device=self.device)
        else:
            self.memory = ReplayBuffer(10_000, env.observation_space.shape, int(env.action_space.n), device=self.device)
        self.q = model(*args).to(self.device)

    def reset(self):
        obs, _ = self.env.reset()
        self.obs = torch.from_numpy(obs).float().detach().to(self.device)
        if self.pixels:
            self.obs = transforms(self.env)


    def fill_memory(self, num):
        self.reset()
        for _ in range(num):
            _, done = self.step(1)
            if done: self.reset()

    def get_action(self, eps):
        action = int(self.env.action_space.sample())
        if np.random.random() > eps:
            q_vals = self.q(self.obs.unsqueeze(0)).detach()
            action = q_vals.argmax(-1).item()
        return action

    def step(self, eps):
        action = self.get_action(eps)
        obs_n, reward, term, trunc, _ = self.env.step(action)
        obs_n = torch.from_numpy(obs_n).float().detach().to(self.device)
        if self.pixels:
            obs_n = transforms(self.env)
        done = term or trunc
        exp = (self.obs, action, reward, done, obs_n)
        self.memory.append(exp)
        self.obs = obs_n
        return reward, done
    
    def fit(self, eps_fn, gamma, opt,  bs, step, max_steps):
        eps_reward = 0
        eps_loss = 0
        steps = 0
        self.q.train()
        for t in range(max_steps):
            eps = eps_fn(step + t)
            reward, eps_done = self.step(eps)
            eps_reward += reward

            opt.zero_grad()

            (obs, action, reward, done, obs_n) = self.memory.sample(bs)
            q_vals = self.q(obs_n).max(-1, keepdim=True).values
            tgt = reward + (1-done)*gamma*q_vals
            pred = self.q(obs)[torch.arange(bs), action.reshape(-1)].reshape((-1,1))
            assert(tgt.shape == pred.shape), f'tgt:{tgt.shape} != pred:{pred.shape}'
            mse_loss = (tgt-pred).pow(2).mean()
            mse_loss.backward()
            opt.step()
            eps_loss += mse_loss.item()
            if eps_done:
                steps = t
                break

        return eps_reward, eps_loss, steps

    @torch.no_grad()
    def play(self, max_steps):
        replay = []
        self.q.eval()
        self.reset()
        done = False
        steps = 0
        eps_reward = 0
        while not done and steps < max_steps:
            render = self.env.render()
            replay.append(render)
            action = self.get_action(-1)
            self.obs, reward, term, trunc, _ = self.env.step(action)
            self.obs = torch.from_numpy(self.obs).float().to(self.device)
            if self.pixels:
                self.obs = transforms(self.env)
            done = term or trunc
            steps += 1
            eps_reward += reward
        render = self.env.render()
        replay.append(render)
        return replay, eps_reward
