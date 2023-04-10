import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from collections import namedtuple


def avg(arr):
    return sum(arr) / len(arr)

def simple_network(obs_dim, act_dim, hid_dim):
    return nn.Sequential(*[
        nn.Linear(*obs_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, act_dim)
    ])

class ExpBuff:
    def __init__(self, obs_dim, act_dim, capacity):
        self.start = 0
        self.idx = 0
        self.size = 0
        self.capacity = capacity

        self.obs = torch.zeros((self.capacity, *obs_dim), requires_grad=False).float()
        self.r2g = torch.zeros((self.capacity, 1), requires_grad=False).float()
        self.adv = torch.zeros((self.capacity, 1), requires_grad=False).float()
        self.logp = torch.zeros((self.capacity, 1), requires_grad=False).float()

    def store(self, obs, logp):
        self.obs[self.idx] = obs
        self.logp[self.idx] = logp
        self.idx+=1
        self.size+=1

    def store_finished(self, r2g, adv):
        self.r2g[self.start:self.size] = r2g[:, None]
        self.adv[self.start:self.size] = adv[:, None]
        self.start = self.idx

    def reset(self):
        self.idx = 0
        self.size = 0
        self.start = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        Experience = namedtuple('Experience', ['obs', 'r2g', 'adv', 'logp'])
        return Experience(
            self.obs[idx],
            self.r2g[idx],
            self.adv[idx],
            self.logp[idx]
        )

class Agent:
    def __init__(self, env, capacity):
        self.env = env
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.n.item()
        hid_dim = 256

        self.actor = simple_network(obs_dim, act_dim, hid_dim)
        self.critic = simple_network(obs_dim, act_dim, hid_dim)
        self.experience = ExpBuff(obs_dim, act_dim, capacity)

        self.obs, _ = self.env.reset()
        self.obs = torch.from_numpy(self.obs).detach()

    def step(self):
        logits = self.actor(self.obs)
        dist = Categorical(logits=logits)

        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.critic(self.obs)[action]

        return action, value, logp

    def update_actor(self, opt, hparams):
        exp = DataLoader(self.experience, batch_size=hparams['bs'], shuffle=True)

        def get_loss(data, hparams):
            logits = self.actor(data.obs)
            dist = Categorical(logits=logits)

            action = dist.sample()
            logp = dist.log_prob(action)[:,None]
            logp_old = data.logp
            
            advantage = data.adv
            # normalize advantage
            mean = advantage.mean()
            std = advantage.std()
            advantage = (advantage - mean) / (std+1e-8)

            ratio = torch.exp(logp - logp_old)
            assert ratio.shape == (hparams['bs'], 1)
            clip_advantage = torch.clamp(ratio, 1-hparams['eps'], 1+hparams['eps'])*advantage

            loss = -(torch.min(ratio*advantage, clip_advantage)).mean()
            return loss

        losses = []
        for _ in range(hparams['actor_epochs']):
            for data in exp:
                loss = get_loss(data, hparams)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())

        return losses

    def update_critic(self, opt, hparams):
        exp = DataLoader(self.experience, batch_size=hparams['bs'], shuffle=True)

        def get_loss(data, hparams):
            diff = (self.critic(data.obs) - data.r2g).pow(2)
            mse = diff.mean()
            return mse

        losses = []
        for _ in range(hparams['critic_epochs']):
            for data in exp:
                opt.zero_grad()
                loss = get_loss(data, hparams)
                loss.backward()
                opt.step()
                losses.append(loss.item())

        return losses

    def fit(self, hparams, actor_opt, critic_opt):
        for epoch in range(hparams['epochs']):
            self.experience.reset()
            self.obs, _ = self.env.reset()
            self.obs = torch.from_numpy(self.obs).detach()

            epoch_rews = []
            ep_rews = []
            ep_vals = []

            for t in range(hparams['steps']):
                with torch.no_grad():
                    action, value, logp = self.step()

                    action = action.item()
                    value = value.item()
                    logp = logp.item()

                next_obs, reward, term, trunc, _  = self.env.step(action)
                done = term or trunc or (t == hparams['steps']-1)

                self.experience.store(self.obs, logp)

                ep_rews.append(reward)
                ep_vals.append(value)

                self.obs = torch.from_numpy(next_obs).detach()
                if done:
                    # store estimation of reward if epsiode cut off early
                    final = 0
                    if trunc or (t==(hparams['steps']-1)):
                        with torch.no_grad():
                            _, value, _ = self.step()
                            final = value.item()

                    ep_rews.append(final)
                    ep_vals.append(final)

                    # reward to go
                    n = len(ep_rews)
                    r2g = torch.zeros(n, requires_grad=False)
                    for i in reversed(range(n)):
                        r2g[i] = ep_rews[i] + hparams['gamma']*(0 if i+1 >= n else r2g[i+1])

                    # generalized advantage estimation
                    n = len(ep_rews)
                    gamma = hparams['gamma']
                    lam = hparams['lambda']
                    diff = [ep_rews[i] + gamma*ep_vals[i+1] - ep_vals[i] for i in range(n-1)]
                    advantage = torch.zeros(n-1, requires_grad=False)
                    for i in reversed(range(n-1)):
                        advantage[i] = diff[i] + gamma*lam*(0 if i+1 >= n-1 else advantage[i+1])

                    # store r2g and advantage
                    self.experience.store_finished(r2g[:-1], advantage)

                    # end of episode
                    epoch_rews.append(sum(ep_rews))

                    # reset
                    ep_rews = []
                    ep_vals = []
                    self.obs, _ = self.env.reset()
                    self.obs = torch.from_numpy(self.obs).detach()

            # update actor-critic
            actor_loss = self.update_actor(actor_opt, hparams)
            critic_loss = self.update_critic(critic_opt, hparams)

            print(f'Epoch: {epoch} - Avg Reward: {avg(epoch_rews):.4f} - Policy Loss: {avg(actor_loss):.4f} - Critic Loss: {avg(critic_loss):.4f}')


