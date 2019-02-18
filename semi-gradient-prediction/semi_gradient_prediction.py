from grid_world import standard_grid, negative_grid
from helpers import *
import numpy as np

'''
bla bla bla
'''

alpha = 0.01
discount = 0.9
eps = 0.1

def eps_win_policy(s, env):
  '''
  Get a state s, a value state function vs and an env
  as in input, and returns the best action
  ''' 
  policy = {
    (0,0):'R',
    (0,1):'R',
    (0,2):'R',
    (1,0):'U',
    (1,2):'U',
    (2,0):'U',
    (2,1):'R',
    (2,2):'U',
    (2,3):'L'
  }

  if eps > np.random.rand():
    return np.random.choice(env.actions[s])

  return policy.get(s, None)

def random_policy(s, env):
  return np.random.choice(env.actions[s])

def preprocess_features(s):
  return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])

def td_episode(create_env, policy, theta):

  env = create_env()

  while not env.is_terminal(env.current_state()):
    s = env.current_state()

    reward = env.move(policy(s, env))
    new_s = env.current_state()

    x = preprocess_features(s)
    c_est = theta.dot(x)

    x_ = preprocess_features(new_s)
    next_est = theta.dot(x_)
    
    target = None
    if env.is_terminal(env.current_state()):
      target = reward
    else:
      target = reward + discount * next_est

    # Update current state
    theta = theta + alpha * (target - c_est) * x  
  return theta  


def semi_gradient_td(create_env, policy, episodes=100000):
  theta = np.random.randn(4) / 2

  for _ in range(episodes):
    theta = td_episode(create_env, policy, theta)

  return theta


if __name__ == '__main__':
  env = standard_grid()
  theta = semi_gradient_td(standard_grid, eps_win_policy)
  vs = get_value(env, theta, preprocess_features)
  render_vs(env, vs)

  print()

  env = negative_grid()
  theta = semi_gradient_td(negative_grid, eps_win_policy)
  vs = get_value(env, theta, preprocess_features)
  render_vs(env, vs)