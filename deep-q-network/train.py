#!/usr/bin/env python3
import torch
import torch.optim as optim
import torchvision.transforms as T
import os
import model as dqn
import gymnasium as gym

DEVICE = os.getenv("DEVICE", 'cpu')

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
    # env_name = 'LunarLander-v2'
    env = gym.make(env_name, render_mode='rgb_array')
    agent = dqn.Agent(env, dqn.DQNSimple, env.observation_space.shape[0], int(env.action_space.n), 128, device=DEVICE)
    # TODO: train working pixel model
    # agent = model.Agent(env, DQNPixel, int(env.action_space.n), 256, device=DEVICE)
    opt = optim.Adam(agent.q.parameters(), lr=1e-3)

    # only tested on cartpole
    eps = lambda t: max(0.1, 1 - t*1e-4)
    # gives decent results for cartpole & lunarlander
    #eps = lambda _: 0.1
    gamma = 0.99
    bs = 64

    agent.fill_memory(10_000)
    # cartpole
    num_eps = 500
    # lunarlander
    num_eps = 1500
    total_reward = 0
    step = 0
    for n in range(num_eps):
        agent.reset()
        reward, loss, steps = agent.fit(eps, gamma, opt, bs, step, 10_000)
        step += steps
        total_reward += reward
        if n % 50 == 0:
            print(f'Episode {n} - reward: {reward} - avg reward: {total_reward/50} - loss: {loss}')
            total_reward = 1

    # generate about 10 replays
    for i in range(10):
        replay, reward = agent.play(10_000)
        save_replay(f'./replays', f'dqn-{env_name[:-3].lower()}-{i}', replay, 30)
        print(f'saved replay: {len(replay)} frames')
        print(f'reward: {reward}')
        print()
    env.close()
