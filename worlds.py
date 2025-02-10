import numpy as np

#================================================================================================================
#
# Ici se trouve les classes des environnement 1D et 2D. Elles receuille les variable de chaque classe 
# et les fonctions membres afin qu'on puisse utiliser les deux indifféremment dans nos algorithmes de RL et RL+LLM
#
#================================================================================================================



#================================================================================================================
#						Environnement 1D
#================================================================================================================


class Grid1D:
	def __init__(self, size=10):
		self.size = size
		self.goal = size -1
		self.dim = 1
		self.reset()
    
	def reset(self):
		self.agent_pos = 0 # Extremité gauche de la ligne
		return self.agent_pos
			
	def get_state_space(self):
		return [i for i in range(self.size)]
	
	def get_action_space(self):
		return [0, 1]  # Left, Right
		
	def is_action_bringing_closer_to_goal(self, action):
		if self.goal > self.agent_pos:
			return action > 0
		else:
			return action == 0
    
	def step(self, action, method):
		if action == 0:  							# Aller à Left
			self.agent_pos = max(0, self.agent_pos - 1)
		else:  									# Aller à Right
    			self.agent_pos = min(self.size - 1, self.agent_pos + 1)
        	
        	# Implémentation de recompenses différentes pour chaque méthode, ça marché mieux ainsi 
		if method == "policy" :
			reward = 1.0 if self.agent_pos == self.goal else -0.01
		elif method == "Q" :
			reward = 10.0 if self.agent_pos == self.goal else -1
				
		done = self.agent_pos == self.goal
		return self.agent_pos, reward, done
        
	@staticmethod
	def get_action_from_str(action: str):
		return 0 if action == 'Left' else 1
		
		
#================================================================================================================
#						Environnement 2D
#================================================================================================================


class Grid2D:
	def __init__(self, width=5, height=5):
		self.width = width
		self.height = height
		self.goal = (width - 1, height - 1)
		self.reset()
		self.dim = 2
    
	def reset(self):
		self.agent_pos = (0, 0) #Coin en haut à gauche
		return self.agent_pos

	def get_state_space(self):
		return [(x, y) for x in range(self.width) for y in range(self.height)]

	def get_action_space(self):
		return [0, 1, 2, 3]  # Left, Right, Up, Down

	def compute_distance_to_goal(self, state):
		x, y = state
		goal_x, goal_y = self.goal
		return abs(x - goal_x) + abs(y - goal_y)

	def is_action_bringing_closer_to_goal(self, action):
		actual_dist_to_goal = self.compute_distance_to_goal(self.agent_pos)
		next_pos, r, d = self.step(action, "Q")
		next_dist_to_goal = self.compute_distance_to_goal(next_pos)

		return actual_dist_to_goal > next_dist_to_goal
		
	def step(self, action, method):
		x, y = self.agent_pos

		if action == 0:  # Left
			x = max(0, x - 1)
		elif action == 1:  # Right
			x = min(self.width - 1, x + 1)
		elif action == 2:  # Up
			y = max(0, y - 1)
		elif action == 3:  # Down
			y = min(self.height - 1, y + 1)

		self.agent_pos = (x, y)
        	
        	# Implémentation de recompenses différentes pour chaque méthode, ça marché mieux ainsi 
		if method == "policy" :
			reward = 1.0 if self.agent_pos == self.goal else -0.01
		elif method == "Q" :
			reward = 10.0 if self.agent_pos == self.goal else -1
				
		done = self.agent_pos == self.goal
		return self.agent_pos, reward, done

	@staticmethod
	def get_action_from_str(action: str):
		mapping = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3}
		return mapping[action.strip()]
