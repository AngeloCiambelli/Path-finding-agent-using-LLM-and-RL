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
    exploration_exploitation_parameter : float

    def __init__(self, size: int, terminal_state: Position, reward: float = -1., discount: float = 1.,
                 exploration_exploitation_parameter: float = 0.):
        self.size = size

        self.possible_actions = [Position(-1, 0), Position(0, -1), Position(1, 0), Position(0, 1)]

        self.terminal_state = terminal_state
        self.state_values = np.random.random((size, size))
        self.state_values[terminal_state.x, terminal_state.y] = 0

        self.action_values = np.zeros((4, size, size))  # four possible action for each state

        self.create_policy()
        self.create_action_values()

        self.reward = reward
        self.discount = discount
        self.exploration_exploitation_parameter = exploration_exploitation_parameter

    def __repr__(self):
        return str(self.state_values)

    def create_policy(self):
        # policy is of dimension 4 * size * size, for each state of the environment, it stores the four probabilities
        # associated to each of its four possible action :
        # first N*N array -> moving left action ; Second N*N array -> moving up
        # Third N*N array -> moving right action ; Last of the four N*N array -> moving down
        self.policy = 0.25 * np.ones((4, self.size, self.size))
        # Setting all the probability to select actions that step out of the environment to zero
        self.policy[0, :, 0] = 0.
        self.policy[1, 0, :] = 0.
        self.policy[2, :, -1] = 0.
        self.policy[3, -1, :] = 0.

        self.policy /= np.sum(self.policy, axis=0)

    def create_action_values(self):
        self.action_values = np.random.random((4, size, size))  # four possible action for each state
        # Setting all the action values associated to actions that step out of the environment to -infinity so that they
        # will not be selected during policy improvement
        self.action_values[0, :, 0] = -np.inf
        self.action_values[1, 0, :] = -np.inf
        self.action_values[2, :, -1] = -np.inf
        self.action_values[3, -1, :] = -np.inf

    def update_state_values(self):
        variation = 0  # to quantify the convergence
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
            new_state_value += action_proba * (self.reward + self.discount * self.get_state_value_at_position(pos))

        return new_state_value

    def get_actions_probabilities_at_position(self, position: Position) -> list:
        return self.policy[:, position.y, position.x]

    def get_state_value_at_position(self, position: Position) -> float:
        if self.is_position_out_of_the_environment(position):
            return 0.0
        else:
            return self.state_values[position.x, position.y]

    def update_action_values(self):
        variation = 0.
        for i in range(self.size):
            for j in range(self.size):
                for k, a in enumerate(self.possible_actions):
                    # print(f"Action: {a}")
                    if not self.is_position_out_of_the_environment(Position(i, j) + a):  # test if action is taking us outside
                        new_action_value = self.compute_action_value_at_position_for_action(Position(i, j), a)
                        self.action_values[k, j, i] = new_action_value

    def is_position_out_of_the_environment(self, position: Position) -> bool:
        return position.x < 0 or position.y < 0 or position.x > self.size - 1 or position.y > self.size - 1

    def compute_action_value_at_position_for_action(self, position: Position, action: Position) -> float:
        return self.reward + self.discount * self.get_state_value_at_position_for_action(position + action)

    def get_state_value_at_position_for_action(self, position: Position) -> float:
        return self.state_values[position.x, position.y]

    def update_policy_softmax(self):
        self.policy = np.exp(self.action_values) / np.sum(np.exp(self.action_values), axis=0)

    def update_policy_greedily(self):
        pass

if __name__ == '__main__':
    np.random.seed(seed=1)

    size = 4

    # Some positions
    pos_bottom_left_corner = Position(0, 0)
    pos_bottom_right_corner = Position(size - 1, 0)
    pos_top_left_corner = Position(0, size - 1)
    pos_top_right_corner = Position(size - 1, size - 1)

    pos_left_border = Position(0, np.random.randint(1, size - 2))
    pos_right_border = Position(size - 1, np.random.randint(1, size - 2))
    pos_bottom_border = Position(np.random.randint(1, size - 2), 0)
    pos_top_border = Position(np.random.randint(1, size - 2), size - 1)

    env = Environment(size=size, terminal_state=pos_bottom_left_corner)
    print(f"State values:\n {env}\n")

    mean_variation = env.update_state_values()
    while mean_variation > 1e-5:  # stop when converged to stable state values
        mean_variation = env.update_state_values()

    print(f"State values (after convergence):\n {env}\n")

    print(f"Action values:\n {env.action_values}\n")

    env.update_action_values()
    print(f"Action values (after the update):\n {env.action_values}\n")


    print(f"Policy:\n {env.policy}\n")
    env.update_policy_softmax()
    print(f"Policy (after softmax update):\n {env.policy}\n")

