import os
import sys
from script_environment import make_env, find_clue


rl_action = sys.argv[1]		# Where does the RL algorithm want to go
rl_weight = sys.argv[2]		# What is the certitude of the direction
coord1 = int(sys.argv[3])		# 1st coordinate of your place in the env grid
coord2 = int(sys.argv[4])		# 2nd coordinate of your place in the env grid
place_info = find_clue(coord1,coord2)	# Load the environment characteristics
objective = int(sys.argv[5])

#Contexte
print("You are living in a virtual world and looking for a "+ objective +". Use the visual, olfactive and sound clues of the environment and what the reinforcement learning program suggests. You are now in place " + str(coord1) + " et " + str(coord2) + " the reinforcement learning algoritm suggest going " + str(rl_action) + " with certitude " + str(rl_weight) + ". Knowing additionally your previous mouvement and that in your actual place you are in a " + str(place_info[0]) + ", you smell " + str(place_info[1]) + " and you hear " + str(place_info[2]) + ", give a direction (left, right, up, down) where you want to go toward the "+ objective +".")
