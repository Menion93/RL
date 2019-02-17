from grid_world import standard_grid, negative_grid
from helpers import *
import numpy as np

'''
bla bla bla
'''

alpha = 0.2
discount = 0.9
eps = 0.2 # in q learning we should explore more than in sarsa!

def epsilon_soft_greedy(s, env, qs, n_episode):
  '''
  Get a state s, a q value function vs and an env
  as in input, and returns the greedy action
  '''
  decaying_eps = eps / n_episode
  state_actions = env.actions[s]
  p_best = 1-decaying_eps + decaying_eps / len(state_actions)
  p_other = decaying_eps / len(state_actions)

  b_action = np.argmax([qs.get((s, a), 0) for a in state_actions])
  max_value = qs.get((s, state_actions[b_action]), 0)
  action_candidates = [action for action in state_actions if qs.get((s, action), 0) == max_value]
  selected_action = np.random.choice(action_candidates)
  other_actions = list(set(state_actions) - set(selected_action))
  ordered_action = [selected_action] + other_actions
  return np.random.choice(ordered_action, p=[p_best]+[p_other]*(len(state_actions)-1))

def argmax(env, qs, s):
  if s not in env.actions:
    return 0

  actions_values = [(a, qs.get((s, a), 0)) for a in env.actions[s]]
  b_action = np.argmax(map(lambda x: x[1], actions_values))
  return actions_values[b_action][1]


def q_episode(create_env,
              policy, 
              qs, 
              n_episode,
              s_a_counts):

  env = create_env()

  while not env.is_terminal(env.current_state()):
    s = env.current_state()
    a = policy(s, env, qs, n_episode)
    r = env.move(a)
    s_ = env.current_state()

    # Update state action counts
    s_a_counts[(s,a)] = s_a_counts.get((s, a), 0) + 1

    decaying_alpha = alpha / s_a_counts[(s,a)]
    qs[(s, a)] = qs.get((s, a), 0) + decaying_alpha * (r + argmax(env, qs, s_) - qs.get((s, a), 0))


def q_learning(create_env, policy, episodes=5000):
  qs = {}
  s_a_counts = {}

  for n_episode in range(episodes):
    q_episode(create_env, policy, qs, n_episode + 1, s_a_counts)

  return qs


if __name__ == '__main__':
  env = standard_grid()
  qs = q_learning(standard_grid, epsilon_soft_greedy)
  render_qs_policy(env, qs)

  print()

  env = negative_grid()
  qs = q_learning(negative_grid, epsilon_soft_greedy)
  render_qs_policy(env, qs)