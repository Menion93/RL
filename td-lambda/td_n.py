import numpy as np
from math import ceil
import gym
from gym import wrappers


alpha = 0.001
discount = 0.9
eps = 0.1
n = 5

pos = (-1.2, 0.6)
vel = (-0.07, 0.07)

ranges = [pos, vel]


def bin_state(state, n_bins=2):
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


class EpsGreedyAgent:
  def next_action(self, s, qs, env, eps=0):
    if eps > np.random.rand():
        return env.action_space.sample()
    return np.argmax([qs.get((s, a), 0) for a in range(env.action_space.n)])


def q_max(env, qs, s):
  actions_values = [ qs.get((s, a), 0) for a in range(env.action_space.n) ]
  return np.max(actions_values)


def update_qs(env, history, qs, n, last_updates=False):
    if not last_updates and len(history) < n + 1:
      return history

    s, a, _, _ = history[0]

    G = np.sum([pow(discount, i) * r 
                  for i, (_, _, r, _) in enumerate(history)])

    if last_updates:
      qs[(s, a)] = qs.get((s, a), 0) + alpha * (G - qs.get((s, a), 0))
      return history[1:]
    
    _, _, _, s_ = history[-1]

    G += q_max(env, qs, s_) * pow(discount, n + 1)
    qs[(s, a)] = qs.get((s, a), 0) + alpha * (G - qs.get((s, a), 0))

    return history[1:]

    
def td_n(env,
         n,
         agent,
         episodes=1000,
         episode_max_len=5000):

  qs = {}

  for i in range(episodes):

    if (i+1) % 100 == 0:
      print('commencing ep ', (i+1))

    done = False
    count = 0
    history = []

    s = bin_state(env.reset())

    while not done and count < episode_max_len:
      count += 1
      a = agent.next_action(s, qs, env, eps=eps)
      s_, r, done, _ = env.step(a)
      s_ = bin_state(s_)

      history.append((s, a, r, s_))
      history = update_qs(env, history, qs, n)

      s = s_
    
    while len(history) > 0:
      history = update_qs(env, history, qs, n, last_updates=True)
  
  return qs


def playEpisode(env, agent, qs, max_len=500):
  done = False
  s = env.reset()
  totalReward = 0
  count = 0
  while not done and count < max_len:
    count += 1
    action = agent.next_action(bin_state(s), qs, env)
    s, r, done, _ = env.step(action)
    totalReward += r
  
  return totalReward


if __name__ == '__main__':
  env = gym.make('MountainCar-v0').env
  agent = EpsGreedyAgent()

  qs = td_n(env, n, agent, episodes=600)

  # Show learned policy
  env = wrappers.Monitor(env, './video/td-n-mountain-car')
  playEpisode(env, agent, qs)