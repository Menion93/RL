from grid_world import standard_grid
from helpers import *
import numpy as np

'''
In this example we are going to show how to implement policy iteration to get the optimal policy in
a simple grid like world. This use iterative policy evaluation to update V(s) using the greedy policy.
When v(s) is updated, the actions of the greedy policy may change because it depends on v(s). If it continues to
change, then v(s) is converging to v*(s) which is the optimal state value function. This means that we also got the
optimal policy pi*.
state.
'''

threshold = 0.0001
discount = 0.9

def greedy_policy(s, vs, env):
  '''
  Get a state s, a value state function vs and an env
  as in input, and returns the best action
  '''
  actions = []
  for a in ['U', 'D', 'L', 'R']:
      reward = env.move(a)

      if not (env.current_state() == s):
        actions.append((a, vs.get(env.current_state(), 0), reward))
        env.undo_move(a)

  if len(actions) == 0:
    return []

  b_action = np.argmax(list(map(lambda x: x[2] + discount * x[1], actions)))
  return [(actions[b_action], 1)]


def update_vs(vs, env, old_vs, policy):
  states = env.all_states()
  delta = 0
  for s in states:
    env.set_state(s)
    action_probs = policy(s, old_vs, env)
    computed_vs = 0
    for (_, vs_next, reward), p in action_probs:
      computed_vs += p * (reward + discount * vs_next)

    c_delta = abs(vs.get(env.current_state(), 0) - computed_vs)
    vs[env.current_state()] = computed_vs

    if c_delta > delta:
      delta = c_delta
  
  return delta

def iterative_policy_evaluation(env, policy, old_vs):
  vs = {}
  while True:
    delta = update_vs(vs, env, old_vs, policy)
    if delta < threshold:
      break
  return vs

def policy_iteration(env):
  vs = {} #s: v
  policy = {} #s: a

  policy_changed = True
  i=0

  while policy_changed:
    i+=1
    # update vs accorgind to previous vs and greedy policy
    vs = iterative_policy_evaluation(env, greedy_policy, vs)

    policy_changed = False

    for s in env.all_states():
      if s not in policy:
        policy[s] = None
      
      env.set_state(s)
      b_action = greedy_policy(s, vs, env)

      if len(b_action) > 0 and b_action != policy[s]:
        policy[s] = b_action
        policy_changed = True

  return policy, vs

def main():
  env = standard_grid()
  policy, vs = policy_iteration(env)
  render_vs(env, vs)
  render_policy(env, policy)


# Entry Point
if __name__ == '__main__':
  main()