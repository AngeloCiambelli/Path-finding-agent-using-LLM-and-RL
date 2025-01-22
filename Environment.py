import numpy as np
from Position import Position

class Environment:
    size : int
    state_values: np.ndarray
    terminal_state: Position

    def __init__(self, size: int, terminal_state: Position):
        self.size = size
        self.state_values = - np.random.random((size, size))
        self.terminal_state = terminal_state
        self.state_values[terminal_state.x, terminal_state.y] = 0

    def __repr__(self):
        return str(self.state_values)

    def compute_state_values_with_policy(self, policy: tuple):
        pass

    def compute_state_value_at_position(self, position: Position):
        pass


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


if __name__ == '__main__':
    size = 4
    pos_bottom_left_corner = Position(0, 0)
    pos_bottom_right_corner = Position(size - 1, 0)
    pos_top_left_corner = Position(0, size - 1)
    pos_top_right_corner = Position(size - 1, size - 1)

    env = Environment(size=size, terminal_state=pos_bottom_left_corner)

    # Testing getting the type of Position in a given environment
    assert pos_bottom_left_corner.get_nature_according_to_environment(env) == 'bottom left corner'
    assert pos_bottom_right_corner.get_nature_according_to_environment(env) == 'bottom right corner'
    assert pos_top_left_corner.get_nature_according_to_environment(env) == 'top left corner'
    assert pos_top_right_corner.get_nature_according_to_environment(env) == 'top right corner'

    pos_left_border = Position(0, np.random.randint(1, size - 2))
    pos_right_border = Position(size - 1, np.random.randint(1, size - 2))
    pos_bottom_border = Position(np.random.randint(1, size - 2), 0)
    pos_top_border = Position(np.random.randint(1, size - 2), size - 1)

    assert pos_left_border.get_nature_according_to_environment(env) == 'left border'
    assert pos_right_border.get_nature_according_to_environment(env) == 'right border'
    assert pos_bottom_border.get_nature_according_to_environment(env) == 'bottom border'
    assert pos_top_border.get_nature_according_to_environment(env) == 'top border'

    random_pos = Position(np.random.randint(1, size - 2), np.random.randint(1, size - 2))
    assert random_pos.get_nature_according_to_environment(env) == 'free'
