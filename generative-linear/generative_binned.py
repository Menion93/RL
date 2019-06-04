import gym
from gym import wrappers
from math import ceil
import numpy as np

cart_pos_range = (-5, 5)
cart_vel_range = (-2,2)
cart_angle_range = (-15, 15)
cart_pole_vel_range = (-2,2)

ranges = [cart_pos_range, cart_vel_range, cart_angle_range, cart_pole_vel_range]

def bin_state(state, n_bins=10):
  bins = []

  for s, c_range in zip(state, ranges):
    if s < c_range[0]:
      bins.append(0)
      continue
    if s > c_range[1]:
      bins.append(n_bins)
      continue

    bin_size = (c_range[1]-c_range[0]) / n_bins
    sel_bin = ceil((s - c_range[0]) / bin_size)
    normalized_bin = sel_bin / n_bins
    bins.append(normalized_bin)

  return bins


def playEpisode(env, policy, max_len=10000):
  done = False
  s = env.reset()
  count = 0
  while not done and count < max_len:
    count += 1
    action = policy.next_action(bin_state(s))
    s, _, done, _ = env.step(action)
  
  return count

class LinearPolicy:

  def __init__(self):
    self.weights = np.random.randn(4)
    self.threshold = np.random.randn(1)

  def next_action(self, state):
    return int(self.threshold > self.weights.dot(state))


def random_search(n_episodes=20000):
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
  #env = wrappers.Monitor(env, './video')
  playEpisode(env, best_policy)
  
  print('best model turns:', best_count)
  print('best model weights:', best_policy.weights)
  print('best model threshold:', best_policy.threshold)



random_search()
