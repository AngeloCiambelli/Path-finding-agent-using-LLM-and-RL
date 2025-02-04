import numpy as np
import random
import subprocess
import numpy.random as rd
from script_environment import find_clue, make_env

def llm_supervision(position, suggested_direction_untranslated, direction_certitude, history, objective="city"):
	x, y = position
	place_info = find_clue(x,y)	# Load the environment characteristics
	
	#convert to something the llm understand
	if (suggested_direction_untranslated)==0:
		suggested_direction = "up"
	elif (suggested_direction_untranslated)==1:
		suggested_direction = "down"
	elif (suggested_direction_untranslated)==2:
		suggested_direction = "left"
	elif (suggested_direction_untranslated)==3:
		suggested_direction = "right"
	
	#Make prompt
	prompt = "You are looking for a "+ objective +". You know the "+ objective + " isn't in a forest, you know there is road leading to it, but you don't know where it is. Therefore finding a road and following it may be a good strategy. You are now in a " + str(place_info[0]) + ", you smell " + str(place_info[1]) + " and you hear " + str(place_info[2]) + ". A reinforcement learning algorithm hint you to go " + suggested_direction + " with a certitude of " + str(direction_certitude) + ". Carefully considering what you know and the reinforcement learning idea, try to reason on what you know and say a direction (left, right, up, down) to go toward the "+ objective +""
	
	if (x == 0 and y == 9):  
		command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose right or up."]

	elif (x == 9 and y == 0): 
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose left or down."]

	elif (x == 9 and y == 9):  
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose right or down."]

	elif (x == 0 and y == 0):
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose up or left."]

	elif x == 0: 
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose up."]

	elif x == 9: 
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose down."]

	elif y == 0:
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose left."]

	elif y == 9:
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose right."]

	else:
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt]

	justification = subprocess.run(command2, capture_output=True, text=True)
	
	print((justification.stdout).lower())
	
	command_dir = ["ollama", "run", "mistral-openorca"] + ["Give ONLY the direction (left, right, up, down) this text suggest to go to, nothing more :" + justification.stdout]
	
	direction = subprocess.run(command_dir, capture_output=True, text=True)
	
	print((direction.stdout).lower())
	
	'''
	def find_last_occurring_string(text, string_list):
		last_string = None
		last_position = 0

		for string in string_list:
			pos = text.rfind(string)  # Find the last occurrence of the string
			if pos > last_position:   # Update if it's the latest occurring
				last_position = pos
				last_string = string
				
		return last_string
	
	
	supervised_direction = find_last_occurring_string((direction.stdout).lower(), ["left", "right", "up", "down"])
	
	if "right" == supervised_direction :
		supervised_direction = 3
		
	if "left" == supervised_direction:
		supervised_direction = 2
		
	if "up" == supervised_direction:
		supervised_direction = 0
		
	if "down" == supervised_direction:
		supervised_direction = 1
	'''
	
	
	if "right" in (direction.stdout).lower():
		supervised_direction = 3
		
	if "left" in (direction.stdout).lower():
		supervised_direction = 2
		
	if "up" in (direction.stdout).lower():
		supervised_direction = 0
		
	if "down" in (direction.stdout).lower():
		supervised_direction = 1
	
	history += "you said this in previous iterations : " + justification.stdout + ". "
	return(supervised_direction, )
	
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
	history = []
    	
	for episode in range(episodes):
		state = env.reset()
		done = False
        
		while not done:
			if random.uniform(0, 1) < epsilon:
				action = random.choice(env.get_action_space())  # Exploration
			else:
				action = max(Q[state], key=Q[state].get)  # Exploitation
				print(action, Q[state], max(Q[state].values()))
			
			supervised_action = llm_supervision(position = state, suggested_direction_untranslated = action, direction_certitude = max(Q[state].values()), history=history)
			
			print(supervised_action)
			
			next_state, reward, done = env.step(supervised_action)
			 
			
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

