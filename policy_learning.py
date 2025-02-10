import random
import numpy as np
import llm_supervision as llm

#================================================================================================================
#
# Ici se trouve les fonctions implémentant les algorithmes de Policy learning simple et supervisé par un LLM. 
#
#================================================================================================================


#================================================================================================================
#					Classe définissant les politiques tabulés 
#================================================================================================================

class PolicyTable:
	def __init__(self, env, initial_policy=None):
		self.env = env
		self.policy = initial_policy if initial_policy else {s: [1/len(env.get_action_space()) for i in range(0, len(env.get_action_space()))] for s in env.get_state_space()}
		#print(self.policy)

	def get_action_probs(self, state):
		return self.policy[state]

	def update(self, state, action, advantage, alpha):
		probs = self.policy[state]
		probs[action] += alpha * advantage
		for i in range(len(probs)):
			if i != action:
				probs[i] -= alpha * advantage / (len(probs) - 1)
				
		# S'assurer de probas qui somme à 1 et positives
		probs = np.clip(probs, 0., 1.)
		self.policy[state] = probs / np.sum(probs)  

#================================================================================================================
#					Algorithme Policy-learning simple (type reinforce)
#================================================================================================================

def policy_learning(env, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):
    iterations_by_episode = []
    policy = PolicyTable(env)
    
    for episode in range(episodes):
        log_probs = []
        rewards = []
        states = []
        actions = []
        state = env.reset()
        done = False
        iteration = 0
        
        while not done:
            action_probs = policy.get_action_probs(state)
            #print(policy.get_action_probs(state), sum(policy.get_action_probs(state)))
            action = np.random.choice(env.get_action_space(), p=action_probs) if np.random.rand() > epsilon else np.random.choice(env.get_action_space())
            next_state, reward, done = env.step(action, "policy")
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            iteration += 1
        
        # Remonte les récompenses
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Mise à jour de la politique
        for s, a, G in zip(states, actions, returns):
            policy.update(s, a, G, alpha)
        
        iterations_by_episode.append(iteration)
        #print(policy.policy)
    
    return policy.policy, iterations_by_episode

#================================================================================================================
#					Algorithme Policy-learning supervisé par LLM
#================================================================================================================

def policy_learning_with_llm(env, episodes=100, episodes_with_llm=5, alpha=0.1, gamma=0.9, epsilon=0.1, errorRate=0.0):
	iterations_by_episode = []
	
	# Politique a priori (dictionnaire des probabilités initiales)
	policy = PolicyTable(env)	
	traj_state = []
	traj_action = []
    
	for episode in range(episodes):
		log_probs = []
		rewards = []
		states = []
		actions = []
		state = env.reset()
		done = False
		iteration = 0

		traj_state_episode = []
		traj_action_episode = []
		
		# Partie supervision LLM
		if episode < episodes_with_llm:
			while not done:
				
				action_probs = policy.get_action_probs(state)
				action = np.random.choice(env.get_action_space(), p=action_probs) if np.random.rand() > epsilon else np.random.choice(env.get_action_space())
				
				if np.random.rand() > errorRate :
					is_getting_closer = env.is_action_bringing_closer_to_goal(action)

					if env.dim == 1 :
						prompt = llm.context_prompt_1d()
						prompt += llm.prompt_with_action_1d_and_proximity_cue(action, is_getting_closer)
						
					elif env.dim == 2 :
						prompt = llm.context_prompt_2d()
						prompt += llm.prompt_with_action_2d_and_proximity_cue(action, is_getting_closer, state)
						
					response = llm.get_response_from_ollama_prompt(prompt)
					print(f"Épisode : {episode+1}.\nAction RL : {action}\nRéponse : {response}")

					action = env.get_action_from_str(response)
				
				else:
					action = 0

				next_state, reward, done = env.step(action, "policy")
				
				states.append(state)
				actions.append(action)
				rewards.append(reward)
				
				traj_state_episode.append(state)
				traj_action_episode.append(action)
				
				state = next_state 
				iteration += 1
				
			# Remonte les récompenses
			G = 0
			returns = []
			for r in reversed(rewards):
				G = r + gamma * G
				returns.insert(0, G)

			# Mise à jour de la policy
			for s, a, G in zip(states, actions, returns):
				policy.update(s, a, G, alpha)
	

		else:	
			# Moyenner la politique apprise pendant la supervision et les choix du LLM
			if episode == episodes_with_llm:
				Q = {state: {action: 0 for action in env.get_action_space()} for state in env.get_state_space()}
				for s_s, a_s in zip(traj_state, traj_action):
					for s, a in zip(s_s, a_s):
						#print(s, a)
						Q[s][a] += 1
				
				#Nouvelle policy
				P = {}
				for state, action_counts in Q.items():
					total = sum(action_counts.values())  # Nombre total d'actions effectuées dans cet état

					if total > 0:
						P[state] = [(action_counts.get(i, 0) / total + policy.policy[state][i]) / 2 for i in range(0, len(env.get_action_space()))]
					else:
						P[state] = [(1/len(env.get_action_space()) + policy.policy[state][i]) / 2 for i in range(0, len(env.get_action_space()))] 

				#print(P)
				policy = PolicyTable(env, P)
				
			while not done:	
			
				action_probs = policy.get_action_probs(state)
				action = np.random.choice(env.get_action_space(), p=action_probs) if np.random.rand() > epsilon else np.random.choice(env.get_action_space())
				next_state, reward, done = env.step(action, "policy")

				states.append(state)
				actions.append(action)
				rewards.append(reward)

				state = next_state
				iteration += 1

			# Remonte les récompenses
			G = 0
			returns = []
			for r in reversed(rewards):
				G = r + gamma * G
				returns.insert(0, G)

			# Mise à jour de la policy
			for s, a, G in zip(states, actions, returns):
				policy.update(s, a, G, alpha)
			
		traj_state.append(traj_state_episode)
		traj_action.append(traj_action_episode)
		iterations_by_episode.append(iteration)
		
		#print(policy.policy)

	return policy.policy, iterations_by_episode



