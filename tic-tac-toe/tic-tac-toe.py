import copy
import numpy as np

class TicTacToe:

  def __init__(self, player1, player2):
    self.reset()
    self.players = [player1, player2]

  def reset(self):
    self.board = np.array([[0,0,0] for _ in range(3)])

  def run_turn(self, player):
    i, j = player.next_action(self.board)
    self.board[i,j] = player.marker
    return copy.deepcopy(self.board)

  def run_episode(self):
    states = []
    turn = 0
    game_ended = False
    while not game_ended:
      states.append(self.run_turn(self.players[turn % 2]))
      game_ended, winner = self.players[turn % 2].is_final_and_won(self.board)
      turn += 1
    
    rewards = [0.5, 0.5]
    if winner:
      rewards[(turn-1)%2] = 1
      rewards[(turn)%2] = -1
      
    return states, rewards


class Player:

  def __init__(self, marker, alpha):
    self.alpha = alpha
    self.value_func = {}
    self.marker = marker

  def get_next_states(self, state):
    next_actions = [(i,j) for i in range(3) for j in range(3) if state[i,j] == 0]
    next_states = []
    for i,j in next_actions:
      c_copy = copy.deepcopy(state)
      c_copy[i,j] = self.marker
      next_states.append(c_copy)

    return [{'state':state, 'action':action} for state, action in zip(next_states, next_actions)]

  def get_value(self, state):
    return self.value_func[str(state)] if str(state) in self.value_func else 0

  def next_action(self, state):
    next_states = self.get_next_states(state)
    max_state = np.argmax([self.get_value(s['state']) for s in next_states])
    return next_states[max_state]['action']

  def is_final_and_won(self, state):
    won = False
    for i in range(3):
      won = won or state[:,i].tolist() == [self.marker] * 3
      won = won or state[i, :].tolist() == [self.marker] * 3

    won = won or np.diag(state).tolist() == [self.marker] * 3
    won = won or np.diag(state.T).tolist() == [self.marker] * 3
    return np.all([False for i in range(3) for j in range(3) if state[i,j] == 0]) or won, won

  def update_value_function(self, history, reward):
    rev_history = history[::-1]
    for i in range(len(history)-1):
      # update final state
      if i == 0:
        self.value_func[str(rev_history[i])] = reward
      # update intermediate states
      if str(rev_history[i+1]) not in self.value_func:
        self.value_func[str(rev_history[i+1])] = 0

      c_value = self.value_func[str(rev_history[i+1])]
      self.value_func[str(rev_history[i+1])] += self.alpha * ( self.value_func[str(rev_history[i])] - c_value )

if __name__ == '__main__':
  simulation_runs = 10 ** 5
  alpha = 0.05

  player1 = Player(1, alpha)
  player2 = Player(2, alpha)
  game = TicTacToe(player1, player2)

  win_p1 = 0
  win_p2 = 0

  # Simulate and update
  for run in range(simulation_runs):
    history, rewards = game.run_episode()
    # update value function
    player1.update_value_function(history, rewards[0])
    player2.update_value_function(history, rewards[1])

    win_p1 += rewards[0]
    win_p2 += rewards[1]

    game.reset()

    print('\r', sep='')
    print('Player1 Score {} bs Player2 score{}\r'.format(win_p1, win_p2), sep='')
    print(player1.value_func)