import random
import numpy as np
import matplotlib.pyplot as plt

num_slot_machines = 3
max_iteration = 10 ** 5
warmup_iteration = 100
eps = [0.1, 0.05, 0.01]
optimistic_values = [1,2,3]

class SlotMachine:

  def __init__(self):
    # true slot machine distribution 
    self.random = random.randint(1,99) / 100
 
  def pull(self):
    # return a victory or a loss
    return np.random.binomial(1, self.random)

def update_distribution(distribution, index, value):
  prev_mean, count = distribution[index]
  count += 1
  # update mean
  next_mean = (1-1/count)*prev_mean + 1/count*value
  distribution[index] = (next_mean, count)

def compute_win_rate(distributions, iterations):
  return np.sum([mean * (count/iterations) for mean, count in distributions])

def epsilon_run(epsilon):
  distributions = [(0, 0) for _ in range(num_slot_machines)] # (mean, count)
  win_rates = []

  for i in range(max_iteration):
    eps_chance = random.random()
    index = None
    
    # Epsilon Exploration
    if eps_chance < epsilon or i < warmup_iteration:
      index = random.randint(0, num_slot_machines-1)
    else:
      # Greedy policy 
      index = np.argmax(list(map(lambda x: x[0], distributions)))

    # Simulation
    value = slot_machines[index].pull()
    # Update estimated slot machine win rate
    update_distribution(distributions, index, value)
    win_rates.append(compute_win_rate(distributions, i+1))

  # Compute win rate
  win_rate = compute_win_rate(distributions, max_iteration)
  print('Win rate of eps={0} is {1}'.format(epsilon, win_rate))
  return win_rates

def optimistic_run(optimistic_value):
  distributions = [(0, 0) for _ in range(num_slot_machines)] # (mean, count)
  # initilize the policy with optimistic values
  optimistic = [(optimistic_value, 1) for _ in range(num_slot_machines)] # (optimistic_mean, count)
  win_rates = []

  # Forget epsilon and do greedy policy
  for i in range(max_iteration):
    index = np.argmax(list(map(lambda x: x[0], optimistic)))
    value = slot_machines[index].pull()
    update_distribution(distributions, index, value)
    update_distribution(optimistic, index, value)
    win_rates.append(compute_win_rate(distributions, i+1))

  # Compute win rate
  win_rate = compute_win_rate(distributions, max_iteration)
  print('Win rate of the optimistic run is {0}'.format(win_rate))
  return win_rates

if __name__ == '__main__':
  slot_machines = [SlotMachine() for _ in range(num_slot_machines)] 
    
  print('Original Distribution')
  for sm in slot_machines:
    print('machine number {0}'.format(sm.random))

  for epsilon in eps:
    plt.plot(epsilon_run(epsilon), label='eps = {0}'.format(epsilon))

  for opt_val in optimistic_values:
    plt.plot(optimistic_run(opt_val), label='optimistic {0}'.format(opt_val))

  plt.legend()
  plt.xscale('log')
  plt.show()
