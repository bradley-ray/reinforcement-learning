#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from PIL import Image
import numpy as np
import gymnasium as gym

def env_to_img(arr):
  to_img = T.ToPILImage()
  arr = torch.from_numpy(arr).permute(2,0,1)
  img = to_img(arr)
  return img

def save_replay(loc, name, buf, dur):
  buf = list(map(env_to_img, buf))
  buf[0].save(f'{loc}/{name}.gif', format='GIF', append_images=buf,
          save_all=True, duration=dur, loop=0)


class DQNSimple(nn.Module):
    def __init__(self, obs_dim, act_dim, h_dim):
        super().__init__()
        self.ff1 = nn.Linear(obs_dim, h_dim)
        self.ff2 = nn.Linear(h_dim, act_dim)

    def forward(self, obs):
        out = self.ff1(obs).relu()
        out = self.ff2(out)
        return out

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.obs_buf = torch.zeros((self.capacity, *obs_dim), requires_grad=False).float()
        self.act_buf = torch.zeros((self.capacity, 1), requires_grad=False).long()
        self.rew_buf = torch.zeros((self.capacity, 1), requires_grad=False).float()
        self.done_buf = torch.zeros((self.capacity, 1), requires_grad=False).float()
        self.obs_n_buf = torch.zeros((self.capacity, *obs_dim), requires_grad=False).float()

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
    def __init__(self, env, model, *args):
        self.env = env
        self.memory = ReplayBuffer(10_000, env.observation_space.shape, int(env.action_space.n))
        self.q = model(*args)

    def reset(self):
        obs, _ = self.env.reset()
        self.obs = torch.from_numpy(obs).float().detach()

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
        obs_n = torch.from_numpy(obs_n).float().detach()
        done = term or trunc
        exp = (self.obs, action, reward, done, obs_n)
        self.memory.append(exp)
        self.obs = obs_n
        return reward, done
    
    def fit(self, eps_fn, gamma, opt,  bs, timesteps):
        eps_reward = 0
        eps_loss = 0
        steps = 0
        for t in range(timesteps):
            eps = eps_fn(t)
            reward, eps_done = self.step(eps)
            eps_reward += reward

            opt.zero_grad()

            (obs, action, reward, done, obs_n) = self.memory.sample(bs)
            q_vals = self.q(obs_n).max(-1, keepdim=True).values
            tgt = reward + (1-done)*gamma*q_vals
            pred = self.q(obs)[torch.arange(bs), action.reshape(-1)].reshape(-1,1)
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
        while not done and steps < max_steps:
            render = agent.env.render()
            replay.append(render)
            action = agent.get_action(-1)
            self.obs, _, term, trunc, _ = agent.env.step(action)
            self.obs = torch.from_numpy(self.obs)
            done = term or trunc
            steps += 1
        render = agent.env.render()
        replay.append(render)
        return replay




if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode='rgb_array')
    agent = Agent(env, DQNSimple, env.observation_space.shape[0], int(env.action_space.n), 128)
    opt = optim.Adam(agent.q.parameters(), lr=1e-3)
    eps = lambda _: 0.1
    gamma = 0.99
    bs = 32

    agent.fill_memory(100)
    total_reward = 0
    num_eps = 500
    for n in range(num_eps):
        agent.reset()
        reward, loss, steps = agent.fit(eps, gamma, opt, bs, 10_000)
        total_reward += reward
        print(f'Episode {n} - reward: {reward} - avg reward: {total_reward/(n+1)} - loss: {loss}')
    replay = agent.play(10_000)
    save_replay('.', 'dqn-cartpole', replay, 30)
    print(f'saved replay: {len(replay)} frames')
    env.close()
