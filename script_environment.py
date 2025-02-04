import numpy as np
import matplotlib.pyplot as plt
import sys

def make_2d_environment():
	#Make visual environment 

	#  f = forest, g = grass, c = city, p = path
	visual_env = [	["f","f","f","f","g","g","c","c","c","g"],
			["f","f","f","f","g","g","c","c","c","g"],
			["f","g","f","f","g","g","g","p-u","g","g"],
			["f","g","f","f","g","g","g","p-u","g","g"],
			["f","f","f","f","f","g","g","p-u","g","g"],
			["f","f","f","f","f","f","g","p-u","g","g"],
			["p-r","p-r","p-r-d","p-r","p-r","p-r","p-r","p-u","g","g"], 
			["g","g","p-d","f","g","g","g","g","g","g"],
			["g","g","p-r","p","g","g","g","g","g","g"],
			["g","g","g","g","g","g","g","g","g","g"]]
	
	fatigue_env = [	[-3,-3,-3,-3,-2,-2,10,10,10,-1],
			[-3,-3,-3,-3,-2,-2,10,10,10,-2],
			[-3,-2,-3,-3,-2,-2,-2,-1,-2,-2],
			[-3,-2,-3,-3,-2,-2,-2,-1,-2,-2],
			[-3,-3,-3,-3,-3,-2,-2,-1,-2,-2],
			[-3,-3,-3,-3,-3,-3,-2,-1,-2,-2],
			[-1,-1,-1,-1,-1,-1,-1,-1,-2,-2], 
			[-2,-2,-1,-3,-2,-2,-2,-2,-2,-2],
			[-2,-2,-1,-1,-2,-2,-2,-2,-2,-2],
			[-2,-2,-2,-2,-2,-2,-2,-2,-2,-2]]

	#  m = moss, g = grass, s = sewage
	odor_env = [	["m","m","m"," ","g","s","s","s","s","s"],
			["m","m","m"," ","g","s","s","s","s","s"],
			["m","m"," "," ","g","s","s","s","s","s"],
			["m","m","m"," ","g","g","g"," ","g","g"],
			[" "," ","m"," "," ","g","g"," ","g","g"],
			[" "," "," "," "," "," "," "," ","g","g"],
			[" "," "," "," "," "," "," "," ","g","g"], 
			["g","g"," "," ","g","g","g","g","g","g"],
			["g","g"," "," ","g","g","g","g","g","g"],
			["g","g","g","g","g","g","g","g","g","g"]]

	#  m = market, b = birds, e = sheep, h = horse drawn carriage
	sound_env = [	["b","b","b"," "," ","m","m"," "," "," "],
			["b","b","b"," "," ","m","m"," "," "," "],
			["b","b","b"," "," "," "," ","m"," "," "],
			["b","b","b"," ","b"," "," ","h","h"," "],
			[" "," ","b"," "," "," ","h","h","h"," "],
			[" "," "," "," ","h"," ","h"," "," "," "],
			[" ","h","h"," "," ","h"," ","h"," "," "], 
			["h","h"," ","h"," "," ","e","e","e","e"],
			[" "," "," "," "," "," "," "," ","e","e"],
			[" "," "," "," "," "," "," "," ","e","e"]]		

	env = [visual_env, odor_env, sound_env, fatigue_env]
	return(env)


def make_1d_environment(size: int):
	return -1 * np.ones((size))



def find_clue(coord1, coord2):
	env = make_2d_environment()
	cripted_clues = []
	clues = []
	cripted_clues.append(env[0][coord1][coord2])
	cripted_clues.append(env[1][coord1][coord2])
	cripted_clues.append(env[2][coord1][coord2])
	
	#Decoding of visual clues
	if cripted_clues[0] == "g":
		clues.append("grass")
		
	elif cripted_clues[0] == "f":
		clues.append("forest")
	
	elif cripted_clues[0] == "c":
		clues.append("city")
		
	elif cripted_clues[0] == "p":
		clues.append("end of the road")
	
	elif cripted_clues[0] == "p-u":
		clues.append("road going up")
	
	elif cripted_clues[0] == "p-r":
		clues.append("road going right")
	
	elif cripted_clues[0] == "p-d":
		clues.append("road going down")
		
	elif cripted_clues[0] == "p-r-d":
		clues.append("road bifurcating and going right and down")
	
	#Decoding of odorant clues
	if cripted_clues[1] == "s":
		clues.append("sewage")
		
	elif cripted_clues[1] == "m":
		clues.append("moss")
	
	elif cripted_clues[1] == "g":
		clues.append("grass")
		
	elif cripted_clues[1] == " ":
		clues.append("nothing")
		
	#Decoding of sound clues
	if cripted_clues[2] == "e":
		clues.append("sheep")
		
	elif cripted_clues[2] == "b":
		clues.append("birds")
	
	elif cripted_clues[2] == "m":
		clues.append("market")
		
	elif cripted_clues[2] == "h":
		clues.append("horse drawn carriage")
	
	elif cripted_clues[2] == " ":
		clues.append("nothing")
	
	return(clues)
	
