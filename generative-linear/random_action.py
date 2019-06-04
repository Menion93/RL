import gym
import numpy as np

env = gym.make('CartPole-v0')

s0 = env.reset()

box = env.observation_space
disc = env.action_space

EPISODES = 300
counts = []

for _ in range(EPISODES):
  count = 0
  done = False
  while not done:
    count+=1
    observation, reward, done, _ = env.step(env.action_space.sample())
  counts.append(count)
  env.reset()

print('Random actions per episode for Cartpole is:', round(np.mean(counts), 2))