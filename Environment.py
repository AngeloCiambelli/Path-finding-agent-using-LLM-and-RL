import numpy as np
from Position import Position

class Environment:
    size : int
    state_values: np.ndarray

    def __init__(self, size: int, terminal_state: Position):
        self.size = size
        self.state_values = - np.random.random((size, size))
        self.state_values[terminal_state.x, terminal_state.y] = 0

    def __repr__(self):
        return str(self.state_values)

    def get_possible_actions_at_position(self, position: Position) -> list:
        # Corners
        if position.x == 0 and position.y == 0:
            possible_actions = ['down', 'right']
        elif position.x == 0 and position.y == self.size - 1:
            possible_actions = ['down', 'left']
        elif position.x == self.size - 1 and position.y == 0:
            possible_actions = ['up', 'right']
        elif position.x == self.size - 1 and position.y == self.size - 1:
            possible_actions = ['up', 'left']

        # Borders
        elif position.x == 0 and 0 < position.y < self.size - 1:
            possible_actions = ['up', 'down', 'left']
        elif 0 < position.x < self.size - 1 and position.y == 0:
            possible_actions = ['down', 'right', 'left']
        elif position.x == self.size - 1 and 0 < position.y < self.size - 1:
            possible_actions = ['up', 'down', 'left']
        elif 0 < position.x < self.size - 1 and position.y == self.size - 1:
            possible_actions = ['up', 'right', 'left']

        else:
            possible_actions = ['up', 'down', 'right', 'left']

        return possible_actions