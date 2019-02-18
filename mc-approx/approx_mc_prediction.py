from grid_world import standard_grid, negative_grid
from helpers import *
import numpy as np

discount = 0.9
alpha = 0.001
eps = 0.1

def eps_win_policy(env, s):
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

def random_policy(env, s):
  return np.random.choice(env.actions[s])

def preprocess_features(s):
  return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1] - 3, 1])

def monte_carlo_evaluation(create_env, policy, iterations=5000):
  theta = np.random.randn(4) / 2

  for _ in range(iterations):
    visited_states = []
    states, returns = run_episode(policy, create_env)

    for s, g in zip(states, returns):
      if s not in visited_states:
        # VS update with SGD
        x = preprocess_features(s)
        theta = theta + alpha * (g - theta.dot(x)) * x
        visited_states.append(s)
  
  return theta

def run_episode(policy, create_env):
  states, rewards, returns = [], [], []
  env = create_env()

  while not env.is_terminal(env.current_state()):
    s = env.current_state()
    a = policy(env, s)
    states.append(s)
    rewards.append(env.move(a))

  g = 0
  for r in rewards[::-1]:
    returns.append(discount * g + r)
    g = r + discount * g

  return states, returns[::-1]

def main():
  env = standard_grid()
  theta = monte_carlo_evaluation(standard_grid, eps_win_policy)
  print(theta)
  vs = get_value(env, theta, preprocess_features)
  render_vs(env, vs)

if __name__ == '__main__':
  main()