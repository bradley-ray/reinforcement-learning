#!/usr/bin/env python3
from model import Agent
from torch.optim import Adam
import torch
import gymnasium as gym
import torchvision.transforms as T

def env_to_img(arr):
  to_img = T.ToPILImage()
  arr = torch.from_numpy(arr).permute(2,0,1)
  img = to_img(arr)
  return img

def save_replay(loc, name, buf, dur):
  buf = list(map(env_to_img, buf))
  buf[0].save(f'{loc}/{name}.gif', format='GIF', append_images=buf,
          save_all=True, duration=dur, loop=0)

if __name__ == '__main__':
    env_name = 'CartPole-v1'
    env_name = 'LunarLander-v2'

    env = gym.make(env_name, render_mode='rgb_array')

    # epochs = 10
    # steps = 4096
    epochs = 50
    steps = 8192
    bs = 32
    gamma = 0.99
    lam = 0.95
    eps = 0.2

    agent = Agent(env=env, capacity=steps)

    actor_epochs = critic_epochs = 15
    # actor_lr = critic_lr = 1e-2
    actor_lr = 1e-3
    critic_lr = 3e-3


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

    # train agent
    agent.fit(hparams, actor_opt, critic_opt)

    # save some replays of agent 
    for i in range(10):
        replay, reward = agent.play()
        save_replay(f'./replays', f'dqn-{env_name[:-3].lower()}-{i}', replay, 30)
        print(f'saved replay: {len(replay)} frames')
        print(f'reward: {reward}\n')

    print("Baseline Comparison")
    print("===================")

    # test trained agent
    env.reset()
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
            ep_rews = []
        agent.obs = torch.from_numpy(obs)
    print(f'agent: {sum(rews)/len(rews):.4f}')

    # compare to baseline random sampling
    env.reset()
    rews = []
    ep_rews = []
    for i in range(1_000):
        action = env.action_space.sample().item()
        _, reward, term, trunc, _ = env.step(action)
        ep_rews.append(reward)
        if term or trunc:
            env.reset()
            rews.append(sum(ep_rews))
            ep_rews = []

    print(f'baseline: {sum(rews)/len(rews):.4f}')


