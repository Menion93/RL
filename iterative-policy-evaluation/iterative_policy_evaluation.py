from grid_world import standard_grid
import numpy as np

'''
In this example we are going to evaluate with iterative policy evaluation, a random policy.
The problem to solve is to land in the correct box in a Grid like world, avoiding bad terminal
state.
'''

threshold = 0.0001
discount = 0.9


def random_policy(s, vs, env):
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

  return [(action, 1 / len(actions)) for action in actions]

def update_vs(vs, env, policy):
  states = env.all_states()
  delta = 0
  for s in states:
    env.set_state(s)
    action_probs = policy(s, vs, env)
    computed_vs = 0
    for (_, vs_next, reward), p in action_probs:
      computed_vs += p * (reward + discount * vs_next)

    c_delta = abs(vs.get(env.current_state(), 0) - computed_vs)
    vs[env.current_state()] = computed_vs

    if c_delta > delta:
      delta = c_delta
  
  return delta

def iterative_policy_evaluation(env, policy):
  vs = {}

  while True:
    delta = update_vs(vs, env, policy)
    if delta < threshold:
      break

  return vs


def main():
  env = standard_grid()
  vs = iterative_policy_evaluation(env, random_policy)
  print(vs)


# Entry Point
if __name__ == '__main__':
  main()

  