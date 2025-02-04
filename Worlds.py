import random
import math
import copy
import numpy as np

from script_environment import find_clue, make_2d_environment, make_1d_environment


# Define the grid-world environment
class GridWorld:
	def __init__(self, size=10, goal_position=((0, 0)), reward=10):
		env = make_2d_environment()
		self.size = size
		self.goal_position = goal_position
		self.reward = reward
		self.step_penalty = env[3]
		self.state = (0, 0)  # Start at top-left corner
    
	def reset(self):
		self.state = (0, 0)
		return self.state

	def get_next_state_from_action(self, action):
		x, y = copy.copy(self.state)
		if action == 0:  # Move up
			x = max(0, x - 1)
		elif action == 1:  # Move down
			x = min(self.size - 1, x + 1)
		elif action == 2:  # Move left
			y = max(0, y - 1)
		elif action == 3:  # Move right
			y = min(self.size - 1, y + 1)

		return x, y
	    
	def step(self, action):
		self.state = self.get_next_state_from_action(action)
		reward = self.reward if self.state in self.goal_position else self.step_penalty[x][y]
		done = self.state in self.goal_position
		return self.state, reward, done
    
	def get_state_space(self):
		return [(i, j) for i in range(self.size) for j in range(self.size)]
    
	def get_action_space(self):
		return [0, 1, 2, 3]  # Up, Down, Left, Right

	def compute_distance_to_goal(self, state):
		x, y = state
		goal = self.goal_position[0]  # the choice of the goal between all the goal positions doesn't matter
		return math.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2)

	def is_action_bringing_closer_to_goal(self, action):
		actual_dist_to_goal = self.compute_distance_to_goal(self.state)
		next_state = self.get_next_state_from_action(action)
		next_dist_to_goal = self.compute_distance_to_goal(next_state)

		return actual_dist_to_goal > next_dist_to_goal


class LineWorld:
	def __init__(self, size=10, goal_position=9, reward=10):
		self.size = size
		self.penalty_environment = make_1d_environment(size)
		self.goal_position = goal_position
		self.reward = reward
		self.state = 0  # Initialisation à l'extrémité gauche de la ligne

	def reset(self):
		self.state = 0
		return self.state

	def get_action_space(self):
		return [0, 1]  # Left, Right

	def get_state_space(self):
		return [i for i in range(self.size)]

	def get_penalty_space(self):
		return self.penalty_environment

	def step(self, action):
		x = self.state
		# Move left
		if action == 0: x = max(0, x - 1)
		# Move right
		else: x = min(self.size - 1, x + 1)

		self.state = x
		reward = self.reward if self.state is self.goal_position else self.penalty_environment[self.state]
		done = self.state is self.goal_position

		return self.state, reward, done

	def is_action_bringing_closer_to_goal(self, action):
		if self.goal_position > self.state:
			return action > 0
		else:
			return action == 0


if __name__ == "__main__":
	# 1D test
	terminal_state = 29  # Extrémité droite de la ligne
	line = LineWorld(size=terminal_state + 1, goal_position=terminal_state, reward=10)

	assert line.is_action_bringing_closer_to_goal(1) is True
	assert line.is_action_bringing_closer_to_goal(0) is False
	line.goal_position = 5
	line.state = 9
	assert line.is_action_bringing_closer_to_goal(0) is True
	assert line.is_action_bringing_closer_to_goal(1) is False

	# 2D test
	grid = GridWorld(size=10, goal_position=((0, 0), ))
	grid.state = (5, 5)

	assert grid.is_action_bringing_closer_to_goal(0) is True
	assert grid.is_action_bringing_closer_to_goal(1) is False
	assert grid.is_action_bringing_closer_to_goal(2) is True
	assert grid.is_action_bringing_closer_to_goal(3) is False

	grid.goal_position = ((5, 5), )
	grid.state = (0, 0)

	assert grid.is_action_bringing_closer_to_goal(0) is False
	assert grid.is_action_bringing_closer_to_goal(1) is True
	assert grid.is_action_bringing_closer_to_goal(2) is False
	assert grid.is_action_bringing_closer_to_goal(3) is True
