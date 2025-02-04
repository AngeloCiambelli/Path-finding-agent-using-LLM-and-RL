import random

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
	    
	def step(self, action):
		x, y = self.state
		if action == 0:  # Move up
			x = max(0, x - 1)
		elif action == 1:  # Move down
			x = min(self.size - 1, x + 1)
		elif action == 2:  # Move left
			y = max(0, y - 1)
		elif action == 3:  # Move right
			y = min(self.size - 1, y + 1)

		self.state = (x, y)
		reward = self.reward if self.state in self.goal_position else self.step_penalty[x][y]
		done = self.state in self.goal_position
		return self.state, reward, done
    
	def get_state_space(self):
		return [(i, j) for i in range(self.size) for j in range(self.size)]
    
	def get_action_space(self):
		return [0, 1, 2, 3]  # Up, Down, Left, Right


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
		return action > 0