import numpy as np
from math import ceil
import gym
from gym import wrappers
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


'''
Iy has been shown thata an approximation of TD-Lambda is just Q-Learning with elegibility Trace. 
That means add momentum to gradients during training.
'''

discount = 0.9999 # This is important! 0.9 is too low!!
eps = 0.1
alpha = 0.01
lambd = 0.5

class CustomSGDRegressor:
  def __init__(self, dim, lambd):
    self.theta = np.random.rand(dim) / np.sqrt(dim)
    self.traces = np.zeros(dim)
    self.lambd = lambd

  def partial_fit(self, X, y):
    self.traces = X[0] + self.lambd * discount * self.traces
    self.theta += alpha * (y-self.theta.dot(X[0])) * self.traces

  def predict(self, X):
    return self.theta.dot(X[0])


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
  def __init__(self, num_actions, state_dim, lambd):
    self.num_actions = num_actions
    self.action2regressor = dict([(action, CustomSGDRegressor(state_dim, lambd)) 
        for action in range(self.num_actions)])
    
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

    if (n+1) % 100 == 0:
      print('Starting episode: ', n+1)

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


def playEpisode(env, agent, f_extractor, max_len=10000):
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
  agent = RBFAgent(env.action_space.n, obs_dim, lambd)

  q_learning(env, agent, feature_extractor, episodes=300)

  # Show learned policy
  env = wrappers.Monitor(env, './video/td-lambda-MountainCar')
  playEpisode(env, agent, feature_extractor)