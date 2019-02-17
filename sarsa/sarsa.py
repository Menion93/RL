from grid_world import standard_grid, negative_grid
from helpers import *
import numpy as np

'''
bla bla bla
'''

alpha = 0.1
discount = 0.9
eps = 0.1

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


def update_qs(sarsa_list, qs, s_a_counts, n_episode, last_update=False):
  if not last_update and len(sarsa_list) < 2:
    return sarsa_list

  s, a, r = sarsa_list[0]
  decaying_alpha = alpha / s_a_counts[(s, a)]

  if last_update:
    qs[(s, a)] = qs.get((s, a), 0) + decaying_alpha * (r - qs.get((s, a), 0))
    return

  s_, a_, _ = sarsa_list[1]
  qs[(s, a)] = qs.get((s, a), 0) + decaying_alpha * (r + qs.get((s_, a_), 0) - qs.get((s, a), 0))

  return sarsa_list[1:]


def sarsa_episode(create_env,
                  policy, 
                  qs, 
                  n_episode,
                  s_a_counts):

  env = create_env()
  sarsa_list = []

  while not env.is_terminal(env.current_state()):
    s = env.current_state()
    a = policy(s, env, qs, n_episode)
    r = env.move(a)

    # Update state action counts
    s_a_counts[(s,a)] = s_a_counts.get((s, a), 0) + 1

    sarsa_list.append((s, a, r))
    sarsa_list = update_qs(sarsa_list, qs, s_a_counts, n_episode)

  update_qs(sarsa_list, qs, s_a_counts, n_episode, last_update=True)


def sarsa(create_env, policy, episodes=5000):
  qs = {}
  s_a_counts = {}

  for n_episode in range(episodes):
    sarsa_episode(create_env, policy, qs, n_episode + 1, s_a_counts)

  return qs


if __name__ == '__main__':
  env = standard_grid()
  qs = sarsa(standard_grid, epsilon_soft_greedy)
  render_qs_policy(env, qs)

  print()

  env = negative_grid()
  qs = sarsa(negative_grid, epsilon_soft_greedy)
  render_qs_policy(env, qs)