import numpy as np
import random
import matplotlib.pyplot as plt

import subprocess
import json

# Define the grid-world environment
class GridWorld:
	def __init__(self, size=10, goal_position=9, reward=10, step_penalty=0):
		self.size = size
		self.goal_position = goal_position
		self.reward = reward
		self.step_penalty = step_penalty
		self.state = 0  
    
	def reset(self):
		self.state = 0
		return self.state
    
	def step(self, action):
		x = self.state
		if action == 0:  # Move left
			x = max(0, x - 1)
		elif action == 1:  # Move right
			x = min(self.size - 1, x + 1)
        
		self.state = x
		reward = self.reward if self.state == self.goal_position else self.step_penalty
		done = self.state == self.goal_position
		return self.state, reward, done
    
	def get_state_space(self):
		return [i for i in range(self.size)]
    
	def get_action_space(self):
		return [0, 1]  # Left, Right


# Q-learning Algorithm
def q_learning(env, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):
	Q = {state: {action: 0 for action in env.get_action_space()} for state in env.get_state_space()}
	time_per_episode = np.zeros(episodes)

	for episode in range(episodes):
		state = env.reset()
		done = False
		compteur = 0
		while not done:
			
			if random.uniform(0, 1) < epsilon:
				action = random.choice(env.get_action_space())  # Exploration
			else:
				action = max(Q[state], key=Q[state].get)  # Exploitation
		    	
			next_state, reward, done = env.step(action)
			best_next_action = max(Q[next_state], key=Q[next_state].get)
			Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
			state = next_state
			
			compteur+=1
			
		time_per_episode[episode] = compteur
	return Q, time_per_episode
	
def llm_supervision(suggested_direction_untranslated, model, history, indice):

	if (suggested_direction_untranslated)==0:
		suggested_direction = "left"
	elif (suggested_direction_untranslated)==1:
		suggested_direction = "right"
	
	user_input = f"You are collaborating with a reinforcement learning algorithm trying to find an objective. The reinforcement algorithm want to go {suggested_direction}, that get you {indice} from the objective. Knowing this say without justification ONLY one direction (right or left) to go closer to the objective."
	
	# Append user message to history
	history.append({"role": "God", "content": user_input})

	# Convert history to JSON string for the command
	history_json = json.dumps(history)

	# Run the Ollama command with full conversation history
	process = subprocess.run(["ollama", "run", model, history_json], capture_output=True, text=True)

	# Get Ollama's response
	response = process.stdout.strip()

	# Append Ollama's response to history
	history.append({"role": "You", "content": response})
	
	#print(process.stdout)
	
	if "right" in (process.stdout).lower():
		supervised_direction = 1
		
	elif "left" in (process.stdout).lower():
		supervised_direction = 0
	else :
		supervised_direction = suggested_direction_untranslated
		
	return(supervised_direction)
	
	
# Q-learning Algorithm coupled with llm model
def q_learning_llm(env, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):

	Q = {state: {action: 0 for action in env.get_action_space()} for state in env.get_state_space()}
	time_per_episode = np.zeros(episodes)

	model = "mistral-openorca"  # Change to your preferred model
	#history = [{"role" : "You are collaborating with a reinforcement learning algorithm trying to find an objective."}]  # Store conversation history
	
	
	for episode in range(episodes):
		
		print("Episode : ", episode)
		
		state = env.reset()
		done = False
		compteur = 0
		
		history = []  # Store conversation history
		
		while not done:
			
			print("state", state, "at iteration", compteur)
			
			if random.uniform(0, 1) < epsilon:
				action = random.choice(env.get_action_space())  # Exploration
			else:
				action = max(Q[state], key=Q[state].get)  # Exploitation
			
			next_state, reward, done = env.step(action)
			
			if abs(next_state - env.goal_position) < abs(state - env.goal_position) :
				indice = "closer"
			else : 
				indice = "farther"
		    	
			if (episode < 5):
				action = llm_supervision(action, model, history, indice)
				next_state, reward, done = env.step(action)
				print(action)
				print(env.step(action))
				print("next state :", next_state, "Done :", done)
		
			best_next_action = max(Q[next_state], key=Q[next_state].get)
			Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
			env.get_action_space[]
			state = next_state
			
			compteur+=1
			
		time_per_episode[episode] = compteur
	return Q, time_per_episode

# Train the agent
env = GridWorld()
nReplicat = 100
nEpisodes = 80
meanTime_per_episode_RL = np.zeros(nEpisodes)
meanTime_per_episode_RL_llm = np.zeros(nEpisodes)

Q_table_llm, time_per_episode_RL_llm = q_learning_llm(env, episodes = nEpisodes)

for i in range(0,nReplicat):
	Q_table_RL, time_per_episode_RL = q_learning(env, episodes = nEpisodes)

	meanTime_per_episode_RL += time_per_episode_RL/nReplicat
	meanTime_per_episode_RL_llm += time_per_episode_RL_llm/nReplicat

# Print the learned Q-values
for state, actions in Q_table_RL.items():
	print(f"RL, State {state}: {actions}")

for state, actions in Q_table_llm.items():
	print(f"RL + llm, State {state}: {actions}")

plt.plot(range(nEpisodes), meanTime_per_episode_RL[0:], label="RL")
plt.plot(range(nEpisodes), meanTime_per_episode_RL_llm[0:], label="RL+llm")
plt.xlabel("number of episode")
plt.ylabel("number of iteration")
plt.legend()
plt.show()
