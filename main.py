import random
import numpy as np
from worlds import *
from q_learning import *
from policy_learning import *

# Make the variables
terminal_state = 9  			# Extrémité droite de la ligne
nEpisode = 60				# Nombre d'episodes d'apprentissage
replicates = 4				# Nombre de réplicats d'apprentissage (pour régulariser les courbes)

# Stockage du nombre d'itération/episode pour chaque réplicat
replicates_time_Q = []			
replicates_time_Q_llm = []		
replicates_time_policy = []
replicates_time_policy_llm = []
replicates_time_policy_llm10 = []
replicates_time_policy_llm50 = []
replicates_time_policy_llm80 = []

#Faire les environnements 1D et 2D
env1D = Grid1D(terminal_state+1)
env2D = Grid2D()

for i in range(replicates):
	print("Réplicat : ", i)
	
	q_table, time_in_episodes_Q = q_learning(env1D, episodes=nEpisode, alpha=0.1, gamma=0.9, epsilon=0.1)
	q_table_with_llm, time_in_episodes_Q_llm = q_learning_with_llm(env1D, episodes=nEpisode, episodes_with_llm=3, alpha=0.1, gamma=0.9, epsilon=0.1)
	
	policy_end, time_in_episodes_policy = policy_learning(env1D, episodes=nEpisode, gamma=0.9, epsilon=0.1)
	p_table_with_llm, time_in_episodes_policy_llm = policy_learning_with_llm(env1D, episodes=nEpisode, episodes_with_llm=3, alpha=0.1, gamma=0.9, epsilon=0.1)
	#p_table_with_llm10, time_in_episodes_policy_llm10 = policy_learning_with_llm(terminal_state+1, episodes=nEpisode, episodes_with_llm=3, alpha=0.1, gamma=0.9, epsilon=0.1, errorRate=0.1)
	#p_table_with_llm50, time_in_episodes_policy_llm50 = policy_learning_with_llm(terminal_state+1, episodes=nEpisode, episodes_with_llm=3, alpha=0.1, gamma=0.9, epsilon=0.1, errorRate=0.5)
	#p_table_with_llm80, time_in_episodes_policy_llm80 = policy_learning_with_llm(terminal_state+1, episodes=nEpisode, episodes_with_llm=3, alpha=0.1, gamma=0.9, epsilon=0.1, errorRate=0.8)
	
	
	replicates_time_Q.append(time_in_episodes_Q)
	replicates_time_Q_llm.append(time_in_episodes_Q_llm)
	
	replicates_time_policy.append(time_in_episodes_policy)
	replicates_time_policy_llm.append(time_in_episodes_policy_llm)
	#replicates_time_policy_llm10.append(time_in_episodes_policy_llm10)
	#replicates_time_policy_llm50.append(time_in_episodes_policy_llm50)
	#replicates_time_policy_llm80.append(time_in_episodes_policy_llm80)
    
median_times_in_episodes_Q = np.median(replicates_time_Q, axis=0)
median_times_in_episodes_Q_llm = np.median(replicates_time_Q_llm, axis=0)

median_times_in_episodes_policy = np.median(replicates_time_policy, axis=0)
median_times_in_episodes_policy_llm = np.median(replicates_time_policy_llm, axis=0)
#median_times_in_episodes_policy_llm10 = np.median(replicates_time_policy_llm10, axis=0)
#median_times_in_episodes_policy_llm50 = np.median(replicates_time_policy_llm50, axis=0)
#median_times_in_episodes_policy_llm80 = np.median(replicates_time_policy_llm80, axis=0)

plot_time_by_episode([median_times_in_episodes_policy, median_times_in_episodes_Q, median_times_in_episodes_Q_llm, median_times_in_episodes_policy_llm], label=["Policy","Q", "Q+LLM", "Policy+LLM"])
#plot_time_by_episode([mean_times_in_episodes_policy, mean_times_in_episodes_Q], label=["Policy","Q"])
#plot_time_by_episode([median_times_in_episodes_policy, median_times_in_episodes_policy_llm], label=["Policy","Policy+LLM"])
#plot_time_by_episode([median_times_in_episodes_policy, median_times_in_episodes_policy_llm, median_times_in_episodes_policy_llm10, median_times_in_episodes_policy_llm50], label=["Policy","Policy+LLM", "Policy+LLM10", "Policy+LLM50"])


print("\nPolitique finale après entraînement:")
print(policy_end)
