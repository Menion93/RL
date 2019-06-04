import numpy as np
from math import ceil
import gym
from gym import wrappers
import copy
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

tf.enable_eager_execution()

'''
Iy has been shown thata an approximation of TD-Lambda is just Q-Learning with elegibility Trace. 
That means add momentum to gradients during training.
'''


discount = 0.9999 # This is important! 0.9 is too low!!
eps = 0.1
p_hidden_layer = 200
v_hidden_layer = 200


class InputPreprocesser:
  def __init__(self):
    self.preprocesser = StandardScaler()

  def fit(self, env, size=20000):
    samples = [env.observation_space.sample() for _ in range(size)]
    self.preprocesser.fit(samples)

  def preprocess(self, x):
    x = x.reshape((1, 2))
    return self.preprocesser.transform(x)


class PolicyNetwork:
  def __init__(self, hidden_layer):
    self.optimizer = tf.train.AdamOptimizer(1e-03)
    self.hidden_layer = hidden_layer
    self.actions = 1 # mean and variance
    self.build()
  
  def build(self):
    self.dense_mean = tf.keras.layers.Dense(self.hidden_layer, activation='tanh')
    self.output_layer_mean = tf.keras.layers.Dense(self.actions, activation='tanh')

    self.dense_std = tf.keras.layers.Dense(self.hidden_layer, activation='tanh')
    self.output_layer_std = tf.keras.layers.Dense(self.actions, activation='softplus')

  def loss(self, X, a, adv):
    predict_op = self.predict(X)
    err = a - predict_op
    squared_err = err * err
    return squared_err * adv

  def partial_fit(self, X, a, adv):
    self.optimizer.minimize(lambda: self.loss(X, a, adv))

  def predict_mean(self, X):
    X = tf.convert_to_tensor(X, dtype='float')
    hidden_out = self.dense_mean(X)
    kk = self.output_layer_mean(hidden_out)
    return tf.reshape(kk, [-1])

  def predict_std(self, X):
    X = tf.convert_to_tensor(X, dtype='float')
    hidden_out = self.dense_std(X)
    kk = self.output_layer_std(hidden_out)
    return tf.reshape(kk, [-1])

  def predict(self, X):
    norm = tf.distributions.Normal(self.predict_mean(X), 
                                   self.predict_std(X))
    clipped_action = tf.clip_by_value(norm.sample(), -1, 1)
    return clipped_action


class ValueNetwork:
  def __init__(self, hidden_layer):
    self.optimizer = tf.train.GradientDescentOptimizer(0.001)
    self.hidden_layer = hidden_layer
    self.build()
  
  def build(self):
    self.dense1 = tf.keras.layers.Dense(self.hidden_layer, activation='tanh')
    self.output_layer = tf.keras.layers.Dense(1, activation=None)

  def mse_loss(self, X, y):
    y_ = self.predict(X)
    y_ = tf.reshape(y_, [-1])
    err = y - y_
    return tf.reduce_sum(err * err)

  def partial_fit(self, X, y):
    self.optimizer.minimize(lambda: self.mse_loss(X, y))

  def predict(self, X):
    X = tf.convert_to_tensor(X, dtype='float')
    hidden_out = self.dense1(X)
    return self.output_layer(hidden_out)


class Agent:
  def __init__(self, p_hidden_layer, v_hidden_layer):
    self.valueNetwork = ValueNetwork(v_hidden_layer)
    self.policyNetwork = PolicyNetwork(p_hidden_layer)
    
  def update_agent(self, x, a, r, x_, discount=0.9):
    G = r * discount + self.valueNetwork.predict(x_)
    adv = G  - self.valueNetwork.predict(x)
    self.policyNetwork.partial_fit(x, a, adv)
    self.valueNetwork.partial_fit(x, [G])

  def next_action(self, x, env, eps=0):
    if eps > np.random.rand():
        return env.action_space.sample()
    action = self.policyNetwork.predict(x)[0]
    return [max(-1, min(1, action))]


def policy_gradient(env,
                    agent, 
                    preprocesser,
                    episodes=1000,
                    episode_max_len=1000):

  for n in range(episodes):
    print(n)
    if (n+1) % 100 == 0:
      print('Starting episode: ', n+1)

    done = False
    count = 0
    totalReward = 0

    s = preprocesser.preprocess(env.reset())

    while not done and count < episode_max_len:
      count += 1
      a = agent.next_action(s, env, eps=eps)
      s_, r, done, _ = env.step(a)
      s_ = preprocesser.preprocess(s_)

      totalReward += r

      agent.update_agent(s, a, r, s_, discount=discount)    

      s = s_

    print(totalReward)


def playEpisode(env, agent, preprocesser, max_len=10000):
  done = False
  s = env.reset()
  totalReward = 0
  count = 0
  while not done and count < max_len:
    count += 1
    action = agent.next_action(preprocesser.preprocess(s), env)
    s, r, done, _ = env.step(action)
    totalReward += r
  
  return totalReward


if __name__ == '__main__':
  env = gym.make('MountainCarContinuous-v0').env

  agent = Agent(p_hidden_layer, v_hidden_layer)
  preprocesser = InputPreprocesser()
  preprocesser.fit(env)

  policy_gradient(env, agent, preprocesser, episodes=100)

  # Show learned policy
  env = wrappers.Monitor(env, './video/policy_gradient_mc_cc')
  playEpisode(env, agent, preprocesser)