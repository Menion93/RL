from grid_world import standard_grid, negative_grid
from helpers import *
import numpy as np

'''
bla bla bla
'''

alpha = 0.01
discount = 0.9

def random_policy(s, env):
  return np.random.choice(env.actions[s])

def td_episode(create_env, policy, vs):

  env = create_env()

  while not env.is_terminal(env.current_state()):
    s = env.current_state()

    reward = env.move(policy(s, env))
    new_s = env.current_state()

    c_est = vs.get(s, 0)
    new_est = vs.get(new_s, 0)

    # Update current state
    vs[s] = c_est + alpha * (reward + discount*new_est - c_est)
  
  vs[env.current_state()] = 0
    


def td_learning(create_env, policy, episodes=20000):
  vs = {}

  for _ in range(episodes):
    td_episode(create_env, policy, vs)

  return vs



if __name__ == '__main__':
  env = standard_grid()
  vs = td_learning(standard_grid, random_policy)
  render_vs(env, vs)

  print()

  env = negative_grid()
  vs = td_learning(negative_grid, random_policy)
  render_vs(env, vs)