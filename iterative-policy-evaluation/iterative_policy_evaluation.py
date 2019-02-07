from grid_world import standard_grid
import numpy as np

'''
In this example we are going to evaluate with iterative policy evaluation, a random policy.
The problem to solve is to land in the correct box in a Grid like world, avoiding bad terminal
state.
'''

threshold = 0.0001

def update_vs(vs, env):
  states = env.all_states()
  delta = 0
  for s in states:
    print('current state is ', s)
    env.set_state(s)
    rewards_vs = []
    for a in ['U', 'D', 'L', 'R']:
      print(env.current_state(), a)
      reward = env.move(a)
      if not (env.current_state() == s):
        rewards_vs.append(reward + vs.get(env.current_state(),0))
        env.undo_move(a)
    
    computed_vs = np.mean(rewards_vs) if len(rewards_vs) != 0 else 0
    c_delta = abs(vs.get(env.current_state(), 0) - computed_vs)
    vs[env.current_state()] = computed_vs

    if c_delta > delta:
      delta = c_delta
  
  return delta

def iterative_policy_evaluation(env):
  vs = {}
  
  while True:
    delta = update_vs(vs, env)

    if delta < threshold:
      break

  return vs


def main():
  env = standard_grid()
  vs = iterative_policy_evaluation(env)

  print(vs)


# Entry Point
if __name__ == '__main__':
  main()