import numpy as np
from math import ceil, exp
import gym
from gym import wrappers
import cv2
import gc
import keras
import time


'''
Iy has been shown thata an approximation of TD-Lambda is just Q-Learning with elegibility Trace. 
That means add momentum to gradients during training.
'''


discount = 0.99
eps = 1
hidden_layer = 100
steps = 4
resize_shape = (80, 80)
network_input = (resize_shape[0], resize_shape[1], steps)


def preprocess(x, show=True):
  x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
  x = cv2.resize(x, resize_shape)
  return x / 255


def get_deep_q_network(num_actions, hidden_layer, env_shape):
  input_ = keras.layers.Input(shape=(env_shape))
  c1_out = keras.layers.Conv2D(32, kernel_size=(3,3), data_format='channels_last')(input_)
  c2_out = keras.layers.Conv2D(64, kernel_size=(3,3), data_format='channels_last')(c1_out)
  c3_out = keras.layers.Conv2D(64, kernel_size=(3,3), data_format='channels_last')(c2_out)
  c3_reshaped = keras.layers.Flatten()(c3_out)
  hidden_output = keras.layers.Dense(512, activation='relu')(c3_reshaped)
  output = keras.layers.Dense(num_actions, activation='linear')(hidden_output)
  model = keras.Model(input_, output)
  optimizer = keras.optimizers.Adam()
  model.compile(optimizer=optimizer, loss='mse')
  return model


def update_state(state, obs_small):
  return np.append(state[:,:,1:], np.expand_dims(obs_small, 2), axis=2)


class DQNAgent:
  def __init__(self, 
               num_actions, 
               hidden_layer,
               env_shape, 
               update_freq,
               replay_max_size,
               batch_size):
    self.dqn = get_deep_q_network(num_actions, hidden_layer, env_shape)
    self.target_network = get_deep_q_network(num_actions, hidden_layer, env_shape)
    self.replay = []
    self.num_actions = num_actions
    self.updates = 0
    self.update_freq = update_freq
    self.replay_max_size = replay_max_size
    self.batch_size = batch_size
    self.env_shape = env_shape

  def update_agent(self, discount=0.9):
    if (self.updates + 1) % self.update_freq == 0:
      self.copy_dqn_to_target()
    # collect unusued memory also
    gc.collect()
    
    self.updates += 1

    self.replay = self.replay[:self.replay_max_size]
    sars_samples_indexes = np.random.choice(len(self.replay), size=self.batch_size, replace=False)
    sars_samples = np.array(self.replay)[sars_samples_indexes]

    X, Y = [], []
    for s, a, r, s_, done in sars_samples:
      X.append(s)
      next_est = 0
      if not done:
        next_est = np.max(self.target_network.predict(np.array([s_])), axis=1)
      y = self.dqn.predict(np.array([s]))
      y[0][a] = r + discount * next_est
      Y.append(y)

    X = np.array(X).reshape((self.batch_size, self.env_shape[0], self.env_shape[1], self.env_shape[2]))
    Y = np.array(Y).reshape((self.batch_size, self.num_actions))
    self.dqn.train_on_batch(X, Y)

  def copy_dqn_to_target(self):
    self.dqn.save_weights('breakout-weights/weights')
    self.target_network.load_weights('breakout-weights/weights')

  def next_action(self, x, env, eps=0):
    if eps > np.random.rand():
        return env.action_space.sample()
    return np.argmax(self.dqn.predict(np.array([x]))[0])


def dqn_episode(n_ep, env, agent, train_freq, warmup, episode_max_len):
    done = False
    count = 0

    s = preprocess(env.reset())
    s = np.array([s for _ in range(steps)]).reshape(network_input)

    totalReward = 0

    while not done and count < episode_max_len:
      count += 1
      c_eps =  1 /(1 + n_ep / 100) + (1 - 1 /(1 + n_ep / 100))*0.1
      a = agent.next_action(s, env, eps=c_eps)
      s_, r, done, _ = env.step(a)
      s_ = update_state(s, preprocess(s_))
      totalReward += r

      agent.replay.insert(0, (s, a, r, s_, done))

      s = s_

      if not warmup and (count + 1) % train_freq == 0:
        agent.update_agent(discount=discount)

    if not warmup:
      agent.update_agent(discount=discount)
    
    return totalReward


def deep_q_learning(env,
                    agent,
                    train_freq = 100,
                    warmup = 50,
                    episodes=1000,
                    episode_max_len=1000):

  avg_reward = []
  
  for n in range(episodes):
    if (n+1) % 100 == 0:
      print('Starting episode: ', n+1, 'avg reward:', np.mean(avg_reward))
      avg_reward = []

    totalReward = dqn_episode(n, env, agent, train_freq, n < warmup, episode_max_len)
    print(totalReward)
    avg_reward.append(totalReward)
      

def playEpisode(env, agent, max_len=10000):
  done = False
  s = preprocess(env.reset())
  s = np.array([s for _ in range(steps)]).reshape(network_input)
  totalReward = 0
  count = 0
  while not done and count < max_len:
    count += 1
    action = agent.next_action(preprocess(s), env)
    s_, r, done, _ = env.step(action)
    s = update_state(s, preprocess(s_))
    totalReward += r
  
  return totalReward


if __name__ == '__main__':
  env = gym.make('Breakout-v0').env

  agent = DQNAgent(env.action_space.n, 
                  hidden_layer, 
                  network_input,
                  1000, #copy target iterations
                  10000, # max mem size
                  32)  # batch size

  deep_q_learning(env, agent, episodes=3500)

  # Show learned policy
  env = wrappers.Monitor(env, './video/dqn-breakout')
  playEpisode(env, agent)