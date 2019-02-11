from grid_world import standard_grid, negative_grid
from helpers import *
import numpy as np

discount = 0.9

def greedy_policy(env, vs):
  '''
  Get a state s, a value state function vs and an env
  as in input, and returns the best action
  '''
  s = env.current_state()
  actions = []
  for a in ['U', 'D', 'L', 'R']:
      reward = env.move(a)

      if not (env.current_state() == s):
        actions.append((a, vs.get(env.current_state(), 0), reward))
        env.undo_move(a)

  b_action = np.argmax(list(map(lambda x: x[2] + discount * x[1], actions)))
  return actions[b_action][0] 

def random_policy():
  '''
  Get a state s, a value state function vs and an env
  as in input, and returns the best action
  '''
  actions = ['U', 'D', 'L', 'R']
  return np.random.choice(actions)

def monte_carlo_evaluation(create_env, policy, iterations=5000):
  vs = {}

  for n in range(iterations):
    visited_states = []
    vs = {}
    counts = {}
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

  while not env.is_terminal(env.current_state()):
    states.append(env.current_state())
    a = policy()
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
  vs = monte_carlo_evaluation(standard_grid, random_policy)
  render_vs(env, vs)

  env = standard_grid()
  vs = monte_carlo_evaluation(standard_grid, random_policy)
  render_vs(env, vs)

if __name__ == '__main__':
  main()
