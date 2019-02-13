
from grid_world import standard_grid, negative_grid
from helpers import *
import numpy as np

discount = 0.9
eps = 0.1

def epsilon_soft_greedy(s, env, qs):
  '''
  Get a state s, a q value function vs and an env
  as in input, and returns the greedy action
  '''
  state_actions = env.actions[s]
  p_best = 1-eps + eps / len(state_actions)
  p_other = eps / len(state_actions)

  q_s = qs.get(s, {})
  b_action = np.argmax([q_s.get(a, 0) for a in state_actions])
  max_value = q_s.get(state_actions[b_action], 0)
  action_candidates = [action for action in state_actions if q_s.get(action, 0) == max_value]
  selected_action = np.random.choice(action_candidates)
  other_actions = list(set(state_actions) - set(selected_action))
  ordered_action = [selected_action] + other_actions
  return np.random.choice(ordered_action, p=[p_best]+[p_other]*(len(state_actions)-1))

  
def eps_greedy_policy(s, env, qs):
  '''
  Get a state s, a q value function vs and an env
  as in input, and returns the greedy action
  '''
  if eps > np.random.random():
    return np.random.choice(env.actions[s])

  q_s = qs.get(s, {})
  b_action = np.argmax([q_s.get(a, 0) for a in env.actions[s]])
  max_value = q_s.get(env.actions[s][b_action], 0)
  # check if there are more actions with equal q value
  action_candidates = [action for action in env.actions[s] if q_s.get(action, 0) == max_value]
  selected_action = np.random.choice(action_candidates)
  return selected_action
  # return env.actions[s][b_action]  

def monte_carlo_control(create_env, policy, iterations=5000):
  qs = {}
  counts = {}

  for _ in range(iterations):
    visited_states = []
    states, actions, returns = run_episode(policy, qs, create_env)

    for s, a, g in zip(states, actions, returns):
      if (s, a) not in visited_states:
        if s not in counts:
          counts[s] = {}
        if s not in qs:
          qs[s] = {}
        counts[s][a] = counts.get(s, {}).get(a, 0) + 1
        qs[s][a] = (1 - 1 / counts[s][a]) * qs.get(s, {}).get(a, 0) + g * (1 / counts[s][a])
        visited_states.append((s, a))
  
  return qs

def run_episode(policy, qs, create_env):
  states, actions, rewards, returns = [], [], [], []
  env = create_env()

  while not env.is_terminal(env.current_state()):
    prev_state = env.current_state()
    states.append(prev_state)
    action = policy(prev_state, env, qs)
    if action == 'U':
      a = np.random.choice(['U', 'L', 'D', 'R'], p=[0.5, 0.5/3, 0.5/3, 0.5/3])
      rewards.append(env.move(a))
    else:
      rewards.append(env.move(action))

    actions.append(action)

  states.append(env.current_state())
  rewards.append(0)

  actions.append('NONE')

  g = 0
  for r in rewards[::-1]:
    returns.append(discount * g + r)
    g = r + discount * g

  return states, actions, returns[::-1]

def main():
  env = standard_grid()
  qs = monte_carlo_control(standard_grid, epsilon_soft_greedy)
  render_qs_policy(env, qs)
  print()
  env = negative_grid()
  qs = monte_carlo_control(negative_grid, epsilon_soft_greedy)
  render_qs_policy(env, qs)

if __name__ == '__main__':
  main()
