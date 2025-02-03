import numpy as np
import random
from script_environment import find_clue, make_env


# Define the grid-world environment
class GridWorld:
	def __init__(self, size=10, goal_position=[(0,6),(0,7),(0,8),(1,6),(1,7),(1,8)], reward=10):
		env = make_env()
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

# Q-learning Algorithm
def q_learning(env, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):
	Q = {state: {action: 0 for action in env.get_action_space()} for state in env.get_state_space()}
    
	for episode in range(episodes):
		state = env.reset()
		done = False
        
		while not done:
			if random.uniform(0, 1) < epsilon:
				action = random.choice(env.get_action_space())  # Exploration
			else:
				action = max(Q[state], key=Q[state].get)  # Exploitation
		    
			next_state, reward, done = env.step(action)
			print(next_state, reward, done)
		    
			best_next_action = max(Q[next_state], key=Q[next_state].get)
			Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
		    
			state = next_state
    
	return Q

# Train the agent
env = GridWorld()
Q_table = q_learning(env)

# Print the learned Q-values
for state, actions in Q_table.items():
	print(f"State {state}: {actions}")

