import numpy as np
from math import ceil
import gym
from gym import wrappers


cart_pos_range = (-2.4, 2.4)
cart_vel_range = (-2,2)
cart_angle_range = (-0.4, 0.4)
cart_pole_vel_range = (-2,2)

ranges = [cart_pos_range, cart_vel_range, cart_angle_range, cart_pole_vel_range]

alpha = 0.1
discount = 0.9
eps = 0.1


class EPSGreedyPolicy:

  def __init__(self, eps):
    self.eps = eps

  def next_action(self, s, qs, env):
    if self.eps > np.random.rand():
        return env.action_space.sample()
    return argmax(env, qs, s)


class GreedyPolicy:
  def __init__(self, qs, env):
    self.qs = qs
    self.env = env

  def next_action(self, s):
    return argmax(self.env, self.qs, s)


def argmax(env, qs, s):
  actions_values = [qs.get((s, a), 0) for a in range(env.action_space.n)]
  return np.argmax(actions_values)


def get_return(env, qs, s):
  return np.max([qs.get((s, a), 0) for a in range(env.action_space.n)])


def bin_state(state, n_bins=10):
  bins = []

  for s, c_range in zip(state, ranges):
    if s < c_range[0]:
      bins.append(0)
      continue
    if s > c_range[1]:
      bins.append(1)
      continue

    bin_size = (c_range[1]-c_range[0]) / n_bins
    sel_bin = ceil((s - c_range[0]) / bin_size)
    normalized_bin = sel_bin / n_bins
    bins.append(normalized_bin)

  return str(bins)
  

def q_episode(env,
              policy, 
              qs, 
              n_episode,
              s_a_counts,
              max_len=200):
  s = bin_state(env.reset())
  done = False
  count = 0

  while not done and count < max_len:
    count += 1
    a = policy.next_action(s, qs, env)
    s_, r, done, _ = env.step(a)
    s_ = bin_state(s_)

    if done and count < max_len:
      r = -100
    elif count == max_len:
      r = 100
    else:
      r = 0

    # Update state action counts
    s_a_counts[(s,a)] = s_a_counts.get((s, a), 0) + 1

    decaying_alpha = alpha / s_a_counts[(s,a)]
    
    qs[(s, a)] = qs.get((s, a), 0) + decaying_alpha * (r + get_return(env, qs, s_) - qs.get((s, a), 0))

    s = s_
    

def q_learning(env, policy, episodes=10000):
  qs = {}
  s_a_counts = {}

  for n_episode in range(episodes):
    q_episode(env, policy, qs, n_episode + 1, s_a_counts)

  return qs


def playEpisode(env, policy, max_len=10000):
  done = False
  s = env.reset()
  count = 0
  while not done and count < max_len:
    count += 1
    action = policy.next_action(bin_state(s))
    s, _, done, _ = env.step(action)
  
  return count


if __name__ == '__main__':
  # print(bin_state([1., 0, 0, 0]))
  env = gym.make('CartPole-v0').env
  policy = EPSGreedyPolicy(eps)
  qs = q_learning(env, policy)

  # Show learned policy
  learned_policy = GreedyPolicy(qs, env)
  env = wrappers.Monitor(env, './video')
  print(np.mean([playEpisode(env, learned_policy) for _ in range(500)]))

