from grid_world import standard_grid, negative_grid
from helpers import *
import numpy as np

'''
In this example we are going to implement value iteration in the Windy Grid World we already seen in policy iteration random.
Value Iteration is a smart version of policy iteration that only takes one step of policy evaluation to update v(s).
In other words we just do iterative policy evaluation using the greedy policy.
'''

threshold = 0.0001
discount = 0.9

def greedy_policy(s, vs, env):
  '''
  Get a state s, a value state function vs and an env
  as in input, and returns the best action
  '''
  actions = []
  for a in ['D', 'L', 'R']:
    reward = env.move(a)

    if not (env.current_state() == s):
      actions.append([a, vs.get(env.current_state(), 0), reward])
      env.undo_move(a)

  # if action is UP
  reward = env.move('U')
  # action up is possible
  if not (env.current_state() == s):
    up_action = ['U', vs.get(env.current_state(), 0) * 0.5, reward * 0.5]

    for a, vs_, reward in actions:
      up_action[1] +=  vs_ * 0.5/3
      up_action[2] += reward * 0.5 / 3

    env.undo_move('U')
    stay_action = 3-len(actions)

    for _ in range(stay_action):
      up_action[1] +=  vs.get(env.current_state(),0) * 0.5/3

    actions.append(up_action)

  if len(actions) == 0:
    return []

  b_action = np.argmax(list(map(lambda x: x[2] + discount * x[1], actions)))
  return [(actions[b_action], 1)]


def update_vs(vs, env, old_vs):
  states = env.all_states()
  delta = 0
  for s in states:
    env.set_state(s)
    action_probs = greedy_policy(s, old_vs, env)
    computed_vs = 0
    for (_, vs_next, reward), p in action_probs:
      computed_vs += p * (reward + discount * vs_next)

    c_delta = abs(vs.get(env.current_state(), 0) - computed_vs)
    vs[env.current_state()] = computed_vs

    if c_delta > delta:
      delta = c_delta
  
  return delta

def value_iteration(env):
  vs = {}
  while True:
    delta = update_vs(vs, env, vs)
    if delta < threshold:
      break

  return vs

def policy_from_v(env, vs):
  policy = {}
  for s in env.all_states():
    if s not in policy:
      policy[s] = None
    
    env.set_state(s)
    b_action = greedy_policy(s, vs, env)

    if len(b_action) > 0 and b_action != policy[s]:
      policy[s] = b_action

  return policy

def main():
  print('Standard Grid')
  env = standard_grid()
  v_star = value_iteration(env)
  render_vs(env, v_star)
  render_policy(env, policy_from_v(env, v_star))

  print('Negative Grid:')
  env = negative_grid()
  v_star = value_iteration(env)
  render_vs(env, v_star)
  render_policy(env, policy_from_v(env, v_star))

# Entry Point
if __name__ == '__main__':
  main()