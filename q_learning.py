import random
import numpy as np
import matplotlib.pyplot as plt
import llm_supervision as llm

#================================================================================================================
#
# Ici se trouve les fonctions implémentant les algorithme de Q learning simple et supervisé par un LLM. 
# Ainsi que de fonction d'affichage pour afficher les politique de choix de actions et les courbes de convergence.
#
#================================================================================================================


#================================================================================================================
#					Algorithme Q-learning simple
#================================================================================================================

def q_learning(env, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):
	iterations_by_episode = []
	Q = {state: {action: 0 for action in env.get_action_space()} for state in env.get_state_space()}

	traj_state = []
	traj_action = []

	for episode in range(episodes):
		state = env.reset()
		done = False
		iteration = 0

		while not done:
        
			if random.uniform(0, 1) < epsilon:
				action = random.choice(env.get_action_space())  # Exploration
			else:
				action = max(Q[state], key=Q[state].get)  # Exploitation

			next_state, reward, done = env.step(action, "Q")

			best_next_action = max(Q[next_state], key=Q[next_state].get)

			Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

			state = next_state

			iteration += 1

		iterations_by_episode.append(iteration)

	return Q, iterations_by_episode

#================================================================================================================
#					Algorithme Q-learning supervisé par LLM
#================================================================================================================

def q_learning_with_llm(env, episodes=100, episodes_with_llm=5, alpha=0.1, gamma=0.9, epsilon=0.1):
	iterations_by_episode = []
	Q = {state: {action: 0 for action in env.get_action_space()} for state in env.get_state_space()}

	traj_state = []
	traj_action = []

	for episode in range(episodes):
		state = env.reset()
		done = False
		iteration = 0

		traj_state_episode = []
		traj_action_episode = []

		
		if episode < episodes_with_llm:  # PArtie LLM
			while not done:
			
				if random.uniform(0, 1) < epsilon:
					action = random.choice(env.get_action_space())  # Exploration
				else:
					action = max(Q[state], key=Q[state].get)  # Exploitation

				is_getting_closer = env.is_action_bringing_closer_to_goal(action)
				
				if env.dim == 1 :
					prompt = llm.context_prompt_1d()
					prompt += llm.prompt_with_action_1d_and_proximity_cue(action, is_getting_closer)
					
				elif env.dim == 2 :
					prompt = llm.context_prompt_2d()
					prompt += llm.prompt_with_action_2d_and_proximity_cue(action, is_getting_closer, state)

				response = llm.get_response_from_ollama_prompt(prompt)
				
				#Afficher les action proposé par le RL et celle choisi par le LLM
				print(f"Épisode : {episode+1}.\nAction RL : {action}\nRéponse : {response}")

				action = env.get_action_from_str(response)

				next_state, reward, done = env.step(action, "Q")
				best_next_action = max(Q[next_state], key=Q[next_state].get)
				Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

				traj_state_episode.append(state)
				traj_action_episode.append(action)
				
				state = next_state

				iteration += 1
	
		else:
			
			if episode == episodes_with_llm:  # Quand RL et relaché par le LLM
				Q = {state: {action: 0 for action in env.get_action_space()} for state in env.get_state_space()}
				for states, actions in zip(traj_state, traj_action):
					for s, a in zip(states, actions):
						Q[s][a] += 1 / episodes_with_llm
				'''		
				# Calculate mean with old Q table
				for state, action_counts in Q.items():
					old_Q = Q[state]
					new_Q = Q_new[state]
					Q[state][0] = (new_Q[0] + old_Q[0]) / 2
					Q[state][1] = (new_Q[1] + old_Q[1]) / 2
				''' 
			
			while not done:
			
				# Partie RL
				if random.uniform(0, 1) < epsilon:
					action = random.choice(env.get_action_space())  # Exploration
				else:
					action = max(Q[state], key=Q[state].get)  # Exploitation

				next_state, reward, done = env.step(action, "Q")

				best_next_action = max(Q[next_state], key=Q[next_state].get)

				Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

				state = next_state

				iteration += 1

		traj_state.append(traj_state_episode)
		traj_action.append(traj_action_episode)

		iterations_by_episode.append(iteration)
		
		#print_q_table(Q)

	return Q, iterations_by_episode
    


def print_q_table(q):
    for state, actions in q.items():
        print(f"State {(state)}: {actions}")


def plot_time_by_episode(time_by_episode, label):
    for times, l in zip(time_by_episode, label):
        plt.plot(times, label=l)
    plt.xlabel('Number of episode')
    plt.ylabel('Time spent in each episode')
    plt.legend(loc='best')
    plt.show()
    #plt.savefig("plot_comparaison_methode_RL.pdf")
