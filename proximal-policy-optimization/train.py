#!/usr/bin/env python3
from model import Agent
from torch.optim import Adam
import torch
import gymnasium as gym

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    epochs = 100
    steps = 2048
    bs = 32
    gamma = 0.99
    lam = 0.95
    eps = 0.2

    agent = Agent(env=env, capacity=steps)

    actor_epochs = critic_epochs = 50
    actor_lr = critic_lr = 1e-2

    actor_opt = Adam(agent.actor.parameters(), lr=actor_lr)
    critic_opt = Adam(agent.critic.parameters(), lr=critic_lr)

    hparams = {
        'epochs': epochs,
        'steps': steps,
        'gamma': gamma,
        'lambda': lam,
        'eps': eps,
        'bs': bs,

        'actor_epochs': actor_epochs,
        'critic_epochs': critic_epochs,
    }

    agent.fit(hparams, actor_opt, critic_opt)

    # compare to baseline agent
    baseline = Agent(env, capacity=steps)
    rews = []
    ep_rews = []
    for i in range(1_000):
        action, _, _ = baseline.step()
        action = action.item()
        obs, reward, term, trunc, _ = baseline.env.step(action)
        ep_rews.append(reward)
        if term or trunc:
            obs, _ = env.reset()
            rews.append(sum(ep_rews))
        baseline.obs = torch.from_numpy(obs)

    print('baseline:', sum(rews)/len(rews))

    # trained agent
    agent = Agent(env, capacity=steps)
    rews = []
    ep_rews = []
    for i in range(1_000):
        action, _, _ = agent.step()
        action = action.item()
        obs, reward, term, trunc, _ = agent.env.step(action)
        ep_rews.append(reward)
        if term or trunc:
            obs, _ = env.reset()
            rews.append(sum(ep_rews))
        agent.obs = torch.from_numpy(obs)

    print('agent:', sum(rews)/len(rews))

