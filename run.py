import subprocess
import numpy.random as rd
from script_environment import find_clue, make_2d_environment

def main():
	
	maxIter = 200
	objective = "city"
	history = ""
	mean_n = 0
	
	for i in range(0,100):
		
		coord1 = rd.randint(0,9)
		coord2 = rd.randint(0,9)
		surroundings = find_clue(coord1, coord2)[0]
		n = 0
		
		while (n < maxIter) and (surroundings != objective):
			
			rl_action = "up"	# Where does the RL algorithm want to go
			rl_weight = 0		# What is the certitude of the direction
			place_info = find_clue(coord1,coord2)	# Load the environment characteristics
			
			# prompt = "You are looking for a "+ objective +". Use the visual, olfactive and sound clues of the environment and what the reinforcement learning program suggests. Historically, "+ history +". You are now in place (" + str(coord1) + "," + str(coord2) + ") the reinforcement learning algoritm suggest going " + str(rl_action) + " with certitude " + str(rl_weight) + " (knowing 0 is bad and 1 is good, you may choose something else). Using your history and knowing you are now in a " + str(place_info[0]) + ", you smell " + str(place_info[1]) + " and you hear " + str(place_info[2]) + ", say one direction (left, right, up, down) to go toward the "+ objective +" and state shortly what you know now of the environnement."
			
			#prompt = "You are looking for a "+ objective +". Use the visual, olfactive and sound clues of the environment. You should follow the roads and go back on them as much as possible. Using what you remember and knowing you are now in a " + str(place_info[0]) + ", you smell " + str(place_info[1]) + " and you hear " + str(place_info[2]) + ", say one direction (left, right, up, down) to go toward the "+ objective +" and state very shortly why, be logical. "
			
			#Make prompt
			prompt = "You are looking for a "+ objective +". You are now in a " + str(place_info[0]) + ", you smell " + str(place_info[1]) + " and you hear " + str(place_info[2]) + ". Say only one direction (left, right, up, down) to go toward the "+ objective +", be logical following roads usually leads to the city ! "
			
			if (coord1 == 0 and coord2 == 9):  
	    			command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose right or up."]

			elif (coord1 == 9 and coord2 == 0): 
			    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose left or down."]

			elif (coord1 == 9 and coord2 == 9):  
			    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose right or down."]

			elif (coord1 == 0 and coord2 == 0):
			    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose up or left."]

			elif coord1 == 0: 
			    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose up."]

			elif coord1 == 9: 
			    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose down."]

			elif coord2 == 0:
			    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose left."]

			elif coord2 == 9:
			    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose right."]

			else:
			    command2 = ["ollama", "run", "mistral-openorca"] + [prompt]


			direction = subprocess.run(command2, capture_output=True, text=True)
			
			if "right" in (direction.stdout).lower():
				d = "right"
				coord2 += 1
				
			if "left" in (direction.stdout).lower():
				d = "left"	
				coord2 -= 1
				
			if "up" in (direction.stdout).lower():
				d = "up"
				coord1 -= 1
				
			if "down" in (direction.stdout).lower():
				d = "down"
				coord1 += 1	
				
			surroundings = find_clue(coord1, coord2)[0]
			n += 1
			history += "you were in a " + str(place_info[0]) + ", you smelled " + str(place_info[1]) + " and you heard " + str(place_info[2]) + " and you went to the " + d + "; " 
			#print(coord1, coord2, direction.stdout)
		
		print("Bravo you found a city")
		mean_n += n/100
	
	print(mean_n)
	
	
if __name__ == "__main__":
    main()
