import numpy as np
from math import ceil
import gym
from gym import wrappers
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.kernel_approximation import RBFSampler


discount = 0.9
eps = 0.1


class FeatureExtractor:
  def __init__(self, n_components=500):
    rbf_union =  FeatureUnion([('rbf1', RBFSampler(gamma=5.0, n_components=n_components)), 
                                ('rbf2', RBFSampler(gamma=2.0, n_components=n_components)),
                                ('rbf3', RBFSampler(gamma=1.0, n_components=n_components)), 
                                ('rbf4', RBFSampler(gamma=0.4, n_components=n_components))])
    self.feature_extractor = Pipeline([('scaler', StandardScaler()), ('rbf_kernel',rbf_union)])

  def fit_feature_extractor(self, env, size=10000):
    dataset = [env.observation_space.sample() for _ in range(size)]
    self.feature_extractor.fit(dataset)

  def extract_feature(self, s):
    return self.feature_extractor.transform(np.array(s).reshape(1, -1))


class RBFAgent:
  def __init__(self, num_actions, state_dim):
    self.num_actions = num_actions
    self.action2regressor = dict([(action, SGDRegressor()) 
        for action in range(self.num_actions)])

    # Initialize regressor calling fit once
    for _, regressor in self.action2regressor.items():
      regressor.partial_fit(np.random.randn(1, state_dim), [0])
    
  def update_agent(self, x, a, r, x_, discount=0.9):
    G = np.max([self.action2regressor[a].predict(x_) for a in range(self.num_actions)])
    G = r + discount*G
    self.action2regressor[a].partial_fit(x, [G])

  def next_action(self, x, env, eps=0):
    if eps > np.random.rand():
        return env.action_space.sample()
    return np.argmax([self.action2regressor[a].predict(x) for a in range(self.num_actions)])


def q_learning(env,
               agent, 
               f_extractor,
               episodes=1000,
               episode_max_len=1000):

  for n in range(episodes):

    if n % 100 == 0:
      print('commencing ep ', n)

    done = False
    count = 0

    s = f_extractor.extract_feature(env.reset())

    while not done and count < episode_max_len:
      count += 1
      a = agent.next_action(s, env, eps=eps)
      s_, r, done, _ = env.step(a)
      s_ = f_extractor.extract_feature(s_)

      agent.update_agent(s, a, r, s_, discount=discount)    

      s = s_


def playEpisode(env, agent, f_extractor, max_len=500):
  done = False
  s = env.reset()
  totalReward = 0
  count = 0
  while not done and count < max_len:
    count += 1
    action = agent.next_action(f_extractor.extract_feature(s), env)
    s, r, done, _ = env.step(action)
    totalReward += r
  
  return totalReward


if __name__ == '__main__':
  env = gym.make('MountainCar-v0').env

  feature_extractor = FeatureExtractor()
  feature_extractor.fit_feature_extractor(env)

  obs_dim = feature_extractor.extract_feature(env.observation_space.sample()).shape[1]
  agent = RBFAgent(env.action_space.n, obs_dim)

  q_learning(env, agent, feature_extractor, episodes=300)

  # Show learned policy
  env = wrappers.Monitor(env, './video/mountain-car')
  playEpisode(env, agent, feature_extractor)