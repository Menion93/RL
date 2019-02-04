import random
import numpy as np
import matplotlib.pyplot as plt

num_slot_machines = 3
max_iteration = 10 ** 5
warmup_iteration = 100
eps = [0.1, 0.05, 0.01]

class SlotMachine:

  def __init__(self):
    self.random = random.randint(1,99) / 100

  def pull(self):
    return np.random.binomial(1, self.random)

def update_distribution(distribution, index, value):
  prev_mean, count = distribution[index]
  count += 1
  next_mean = (1-1/count)*prev_mean + 1/count*value
  distribution[index] = (next_mean, count)

def compute_win_rate(distributions, iterations):
  return np.sum([mean * (count/iterations) for mean, count in distributions])

def run(epsilon):
  distributions = [(0, 0), (0, 0), (0, 0)] # (mean, count)
  win_rates = []

  for i in range(max_iteration):
    eps_chance = random.random()
    index = None
    
    if eps_chance < epsilon or i < warmup_iteration:
      index = random.randint(0, num_slot_machines-1)
    else:
      index = np.argmax(list(map(lambda x: x[0], distributions)))

    value = slot_machines[index].pull()
    update_distribution(distributions, index, value)
    win_rates.append(compute_win_rate(distributions, i+1))

  # Compute win rate
  win_rate = compute_win_rate(distributions, max_iteration)
  print('Win rate is {0}'.format(win_rate))
  return win_rates

if __name__ == '__main__':
  slot_machines = [SlotMachine() for _ in range(num_slot_machines)] 
    
  print('Original Distribution')
  for sm in slot_machines:
    print('machine number {0}'.format(sm.random))

  for epsilon in eps:
    plt.plot(run(epsilon), label='eps = {0}'.format(epsilon))

  plt.legend()
  plt.xscale('log')
  plt.show()
