import gym
from gym import wrappers
import numpy as np


def playEpisode(env, policy, max_len=10000):
  done = False
  s = env.reset()
  count = 0
  while not done and count < max_len:
    count += 1
    action = policy.next_action(s)
    s, _, done, _ = env.step(action)
  
  return count

class LinearPolicy:

  def __init__(self):
    self.weights = np.random.randn(4)
    self.threshold = np.random.randn(1)

  def next_action(self, state):
    return int(self.threshold > self.weights.dot(state))


def random_search(n_episodes=3000):
  env = gym.make('CartPole-v0').env
  best_count = 0
  best_policy = None

  for _ in range(n_episodes):
    policy = LinearPolicy()
    turns = playEpisode(env, policy)

    if turns > best_count:
      best_count = turns
      best_policy = policy
  
  # Show trainig results
  env = wrappers.Monitor(env, './video')
  playEpisode(env, best_policy)
  
  print('best model turns:', best_count)
  print('best model weights:', best_policy.weights)
  print('best model threshold:', best_policy.threshold)



random_search()