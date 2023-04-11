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

def calculate_reward_to_go(rewards, gamma):
    n = len(rewards)
    r2g = torch.zeros(n, requires_grad=False)
    for i in reversed(range(n)):
        r2g[i] = rewards[i] + gamma*(0 if i+1 >= n else r2g[i+1])

    return r2g

def calculate_advantage(rewards, values, gamma, lam):
    # generalized advantage estimation
    n = len(rewards)
    diff = [rewards[i] + gamma*values[i+1] - values[i] for i in range(n-1)]
    # diff = torch.tensor(rewards)[:-1] + gamma*torch.tensor(values)[1:] - torch.tensor(values)[:-1]
    advantage = torch.zeros(n-1, requires_grad=False)
    for i in reversed(range(n-1)):
        advantage[i] = diff[i] + gamma*lam*(0 if i+1 >= n-1 else advantage[i+1])
    
    return advantage

class ExpBuff:
    def __init__(self, obs_dim, act_dim, capacity):
        self.start = 0
        self.idx = 0
        self.size = 0
        self.capacity = capacity

        self.obs = torch.zeros((self.capacity, *obs_dim), requires_grad=False).float()
        self.act = torch.zeros((self.capacity, 1), requires_grad=False).long()
        self.r2g = torch.zeros((self.capacity, 1), requires_grad=False).float()
        self.adv = torch.zeros((self.capacity, 1), requires_grad=False).float()
        self.logp = torch.zeros((self.capacity, 1), requires_grad=False).float()

    def store(self, obs, action, logp):
        self.obs[self.idx] = obs
        self.act[self.idx] = action
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
        Experience = namedtuple('Experience', ['obs', 'act', 'r2g', 'adv', 'logp'])
        return Experience(
            self.obs[idx],
            self.act[idx],
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
        self.critic = simple_network(obs_dim, 1, hid_dim)
        self.experience = ExpBuff(obs_dim, act_dim, capacity)

        self.obs, _ = self.env.reset()
        self.obs = torch.from_numpy(self.obs).detach()

    @torch.no_grad()
    def step(self):
        logits = self.actor(self.obs)
        dist = Categorical(logits=logits)

        action = dist.sample()
        logp = dist.log_prob(action)
        value = self.critic(self.obs)

        return action, value, logp

    def finish_episode(self, ep_rews, ep_vals, trunc, hparams):
        # store estimation of reward if epsiode cut off early
        final = 0
        if trunc:
            _, value, _ = self.step()
            final = value.item()
        ep_rews.append(final)
        ep_vals.append(final)

        # reward to go
        r2g = calculate_reward_to_go(ep_rews, hparams['gamma'])

        # generalized advantage estimation
        advantage = calculate_advantage(ep_rews, ep_vals, hparams['gamma'], hparams['lambda'])

        # store r2g and advantage
        self.experience.store_finished(r2g[:-1], advantage)

    def collect_experience(self, hparams):
        epoch_rews = []
        ep_rews = []
        ep_vals = []
        num_eps = 0

        for t in range(hparams['steps']):
            action, value, logp = list(map(lambda x: x.item(), self.step()))

            next_obs, reward, term, trunc, _  = self.env.step(action)
            trunc = trunc or (t == hparams['steps']-1)
            done = term or trunc

            self.experience.store(self.obs, action, logp)

            ep_rews.append(reward)
            ep_vals.append(value)

            self.obs = torch.from_numpy(next_obs).detach()
            if done:
                num_eps += 1

                # calculate r2g and advantage
                self.finish_episode(ep_rews, ep_vals, trunc, hparams)

                # end of episode
                epoch_rews.append(sum(ep_rews))

                # reset
                ep_rews = []
                ep_vals = []
                self.obs, _ = self.env.reset()
                self.obs = torch.from_numpy(self.obs).detach()

        return epoch_rews, num_eps

    def update_actor(self, opt, hparams):
        exp = DataLoader(self.experience, batch_size=hparams['bs'], shuffle=True)

        def get_loss(data):
            logits = self.actor(data.obs)
            dist = Categorical(logits=logits)

            logp = dist.log_prob(data.act.reshape(-1))[:,None]
            logp_old = data.logp
            
            # normalize advantage
            advantage = data.adv
            advantage = (advantage - advantage.mean()) / (advantage.std()+1e-8)

            ratio = torch.exp(logp - logp_old)
            assert ratio.shape == (hparams['bs'], 1), f"{ratio.shape} != {(hparams['bs'], 1)}"
            clip_advantage = torch.clamp(ratio, 1-hparams['eps'], 1+hparams['eps'])*advantage

            loss = -(torch.min(ratio*advantage, clip_advantage)).mean()
            return loss

        losses = []
        for _ in range(hparams['actor_epochs']):
            for data in exp:
                loss = get_loss(data)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())

        return losses

    def update_critic(self, opt, hparams):
        exp = DataLoader(self.experience, batch_size=hparams['bs'], shuffle=True)

        def get_loss(data):
            # pred = self.critic(data.obs)[torch.arange(hparams['bs']), data.act.reshape(-1)][:,None]
            # assert pred.shape == (hparams['bs'], 1), f"{pred.shape} != {(hparams['bs'], 1)}"
            pred = self.critic(data.obs)
            diff = (pred - data.r2g).pow(2)
            mse = diff.mean()
            return mse

        losses = []
        for _ in range(hparams['critic_epochs']):
            for data in exp:
                opt.zero_grad()
                loss = get_loss(data)
                loss.backward()
                opt.step()
                losses.append(loss.item())

        return losses

    def fit(self, hparams, actor_opt, critic_opt):
        for epoch in range(hparams['epochs']):
            self.experience.reset()
            self.obs, _ = self.env.reset()
            self.obs = torch.from_numpy(self.obs).detach()

            # collect experience for trajectory length = hparams['steps']
            epoch_rews, num_eps = self.collect_experience(hparams)

            # update actor-critic
            actor_loss = self.update_actor(actor_opt, hparams)
            critic_loss = self.update_critic(critic_opt, hparams)

            print(f'Epoch: {epoch} - Num Episodes: {num_eps} - Avg Reward: {avg(epoch_rews):.4f} - Policy Loss: {avg(actor_loss):.4f} - Critic Loss: {avg(critic_loss):.4f}')

    @torch.no_grad()
    def play(self):
        ep_rews = []
        replay = []
        done = False

        self.obs, _ = self.env.reset()
        self.obs = torch.from_numpy(self.obs).float()
        while not done:
            # store frame of replay
            frame = self.env.render()
            replay.append(frame)

            # choose and take next action
            logits = self.actor(self.obs)
            action = Categorical(logits=logits).sample().item()
            self.obs, reward, term, trunc, _ = self.env.step(action)
            self.obs = torch.from_numpy(self.obs).float()
            ep_rews.append(reward)
            done = term or trunc

        # get final frame
        frame = self.env.render()
        replay.append(frame)

        
        return replay, sum(ep_rews)


