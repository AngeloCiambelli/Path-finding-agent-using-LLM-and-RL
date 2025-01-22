import numpy as np
from Position import Position

class Environment:
    size : int
    possible_actions: list
    state_values: np.ndarray
    action_values: list
    terminal_state: Position
    policy: list
    reward : float
    discount : float

    def __init__(self, size: int, terminal_state: Position, reward: float = -1., discount: float = 1.):
        self.size = size

        self.possible_actions = [Position(0, -1), Position(0, 1), Position(-1, 0), Position(1, 0)]

        self.terminal_state = terminal_state
        self.state_values = np.random.random((size, size))
        self.state_values[terminal_state.x, terminal_state.y] = 0

        self.action_values = np.random.random((size, size, 4))  # four possible maximum action for each state

        self.create_policy()

        self.reward = reward
        self.discount = discount

    def __repr__(self):
        return str(self.state_values)

    def create_policy(self):
        # policy is of dimension 4 * size * size, for each state of the environment, it stores the four probabilities
        # associated to each of its four possible action :
        # 0/3 array -> moving left action ; 1/3 -> moving up ; 2/3 -> moving right ; 3/3 -> moving down
        self.policy = 0.25 * np.ones((4, self.size, self.size))
        # There is a zero probability of stepping out of the environment
        self.policy[0, :, 0] = 0.
        self.policy[1, 0, :] = 0.
        self.policy[2, :, -1] = 0.
        self.policy[3, -1, :] = 0.

        self.policy /= np.sum(self.policy, axis=0, keepdims=True)

    def get_state_values_at_position(self, position: Position) -> float:
        if position.x < 0 or position.y < 0 or position.x > self.size - 1 or position.y > self.size - 1:
            return 0.0
        else:
            return self.state_values[position.x, position.y]

    def get_actions_probabilities_at_position(self, position: Position) -> list:
        return self.policy[:, position.y, position.x]

    def update_state_values(self):
        variation = 0
        for i in range(self.size):
            for j in range(self.size):
                # Don't parse the terminal state
                if i != self.terminal_state.x or j != self.terminal_state.y:
                    new_state_value = self.compute_new_state_value_at_position(Position(i, j))
                    variation += np.abs(new_state_value - self.state_values[i, j])
                    self.state_values[i, j] = new_state_value

        return variation / (self.size ** 2)

    def compute_new_state_value_at_position(self, position: Position) -> float:
        possible_future_positions = [position + Position(-1, 0), position + Position(0, -1),
                                     position + Position(1, 0), position + Position(0, 1)]
        action_probabilities = self.get_actions_probabilities_at_position(position)

        new_state_value = 0
        for pos, action_proba in zip(possible_future_positions, action_probabilities):
            new_state_value += action_proba * (self.reward + self.discount * self.get_state_values_at_position(pos))

        return new_state_value

    def update_action_values(self):
        for i in range(self.size):
            for j in range(self.size):
                if i != self.terminal_state.x and j != self.terminal_state.y:
                    for a, k in enumerate(self.possible_actions):
                        self.action_values[i, j, k] = self.compute_action_value_at_position_for_action(Position(i, j), a)

    def compute_action_value_at_position_for_action(self, position: Position, action: Position) -> float:
        return self.reward + self.discount * self.get_state_values_at_position(position + action)


if __name__ == '__main__':
    np.random.seed(seed=1)

    size = 4
    pos_bottom_left_corner = Position(0, 0)
    pos_bottom_right_corner = Position(size - 1, 0)
    pos_top_left_corner = Position(0, size - 1)
    pos_top_right_corner = Position(size - 1, size - 1)

    pos_left_border = Position(0, np.random.randint(1, size - 2))
    pos_right_border = Position(size - 1, np.random.randint(1, size - 2))
    pos_bottom_border = Position(np.random.randint(1, size - 2), 0)
    pos_top_border = Position(np.random.randint(1, size - 2), size - 1)

    env = Environment(size=size, terminal_state=pos_bottom_left_corner)
    print(env)
    print(env.policy[:, 0, 3])

    print(env.compute_new_state_value_at_position(pos_bottom_right_corner))
    print(env.compute_new_state_value_at_position(pos_top_left_corner))

    env.update_state_values()
    print(env)

    mean_variation = env.update_state_values()
    while mean_variation > 1e-2:
        mean_variation = env.update_state_values()
        print(env)

    # Testing getting the type of Position in a given environment
    assert pos_bottom_left_corner.get_nature_according_to_environment(env) == 'bottom left corner'
    assert pos_bottom_right_corner.get_nature_according_to_environment(env) == 'bottom right corner'
    assert pos_top_left_corner.get_nature_according_to_environment(env) == 'top left corner'
    assert pos_top_right_corner.get_nature_according_to_environment(env) == 'top right corner'

    assert pos_left_border.get_nature_according_to_environment(env) == 'left border'
    assert pos_right_border.get_nature_according_to_environment(env) == 'right border'
    assert pos_bottom_border.get_nature_according_to_environment(env) == 'bottom border'
    assert pos_top_border.get_nature_according_to_environment(env) == 'top border'

    random_pos = Position(np.random.randint(1, size - 2), np.random.randint(1, size - 2))
    assert random_pos.get_nature_according_to_environment(env) == 'free'
