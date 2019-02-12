from grid_world import standard_grid, negative_grid
from helpers import *
import numpy as np

discount = 0.9

def win_policy(s):
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
    (2,1):'L',
    (2,2):'U',
    (2,3):'L'
  }

  return policy.get(s, None)


def monte_carlo_evaluation(create_env, policy, iterations=5000):
  vs = {}
  counts = {}

  for _ in range(iterations):
    visited_states = []
    states, returns = run_episode(policy, vs, create_env)

    for s, g in zip(states, returns):
      if s not in visited_states:
        counts[s] = counts.get(s, 0) + 1
        vs[s] = (1 - 1 / counts[s]) * vs.get(s, 0) + g * (1 / counts[s])
        visited_states.append(s)
  
  return vs

def run_episode(policy, vs, create_env):
  states, rewards, returns = [], [], []
  env = create_env()

  # Random start from 2,0 and 2,3
  initial_state = np.random.randint(0,2)
  env.set_state([(2,0), (2,3)][initial_state])

  while not env.is_terminal(env.current_state()):
    states.append(env.current_state())
    a = policy(env.current_state())
    if a == 'U':
      a = np.random.choice(['U', 'L', 'D', 'R'], p=[0.5, 0.5/3, 0.5/3, 0.5/3])
    rewards.append(env.move(a))

  states.append(env.current_state())
  rewards.append(0)

  g = 0
  for r in rewards[::-1]:
    returns.append(discount * g + r)
    g = r + discount * g

  return states, returns[::-1]

def main():
  env = standard_grid()
  vs = monte_carlo_evaluation(standard_grid, win_policy)
  render_vs(env, vs)

  env = standard_grid()
  vs = monte_carlo_evaluation(standard_grid, win_policy)
  render_vs(env, vs)

if __name__ == '__main__':
  main()
