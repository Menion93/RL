from grid_world import standard_grid, negative_grid
from helpers import *
import numpy as np
import math

alpha = 0.01
discount = 0.9
eps = 0.05

def epsilon_soft_greedy(s, env, theta, n_episodes):
  '''
  Get a state s, a q value function vs and an env
  as in input, and returns the greedy action
  '''
  decaying_eps = eps / (1 + n_episodes * 0.0001)
  state_actions = env.actions[s]
  p_best = 1-decaying_eps + decaying_eps / len(state_actions)
  p_other = decaying_eps / len(state_actions)

  action_values = [(a, theta.dot(preprocess_features(s, a))) for a in env.actions[s]]
  action_values.sort(key=lambda x:x[1], reverse=True)
  ordered_action_strings = list(map(lambda x: x[0], action_values))
  return np.random.choice(ordered_action_strings, p=[p_best] + [p_other] * (len(state_actions) - 1))

def preprocess_features(s, a):
  x = []

  for a_ in ['U', 'L', 'R', 'D']:
    action_taken = int(a == a_)
    x.extend([s[0] * action_taken - 1,
              (s[1] - 1.5)/1.5* action_taken,
              (s[0] ** 2- 3)/3 * action_taken,
              (s[1] ** 2-4.5)/4.5 * action_taken, 
              (s[0]*s[1]- 3)/3*action_taken,
              action_taken])

  x.append(1)
  return np.array(x)

def update_qs(sarsa_list, theta, s_a_counts, n_episode, last_update=False):
  if not last_update and len(sarsa_list) < 2:
    return sarsa_list, theta

  s, a, r = sarsa_list[0]
  decaying_alpha = alpha / (1 + n_episode * 0.0001)
  x = preprocess_features(s, a)
  c_est = theta.dot(x)

  if last_update:
    theta = theta + decaying_alpha * (r - c_est) * x
    return theta

  s_, a_, _ = sarsa_list[1]
  x_ = preprocess_features(s_, a_)
  next_est = theta.dot(x_)
  theta = theta + decaying_alpha * (r + discount * next_est - c_est) * x

  return sarsa_list[1:], theta


def sarsa_episode(create_env,
                  policy, 
                  theta, 
                  n_episode,
                  s_a_counts):

  env = create_env()
  sarsa_list = []
  i=0
  while not env.is_terminal(env.current_state()) and i < 5000:
    i += 1
    s = env.current_state()
    a = policy(s, env, theta, n_episode)
    r = env.move(a)

    # Update state action counts
    s_a_counts[(s,a)] = s_a_counts.get((s, a), 0) + 1

    sarsa_list.append((s, a, r))
    sarsa_list, theta = update_qs(sarsa_list, theta, s_a_counts, n_episode)

  if i == 5000:
    return theta

  theta = update_qs(sarsa_list, theta, s_a_counts, n_episode, last_update=True)
  return theta


def semi_gradient_sarsa(create_env, policy, episodes=20000):
  s_a_counts = {}
  theta = np.random.randn(25) / np.sqrt(25)

  for n_episode in range(episodes):
    theta = sarsa_episode(create_env, policy, theta, n_episode + 1, s_a_counts)

  return theta


if __name__ == '__main__':
  env = standard_grid()
  theta = semi_gradient_sarsa(standard_grid, epsilon_soft_greedy)
  qs = get_qs(env, theta, preprocess_features)
  render_qs_policy(env, qs)

  print()

  env = negative_grid()
  qs = semi_gradient_sarsa(negative_grid, epsilon_soft_greedy)
  qs = get_qs(env, theta, preprocess_features)
  render_qs_policy(env, qs)