import numpy as np
import matplotlib.pyplot as plt
import sys

#Get place of the agent in the grid [0,9]x[0,9]
coord1 = int(sys.argv[1])
coord2 = int(sys.argv[2])

if (coord1>9) or (coord1<0) or (coord2>9) or (coord2<0) :
	print("The coordinantes are out of range of [0,9]x[0,9]")
	exit()

env = []

#Make visual environment 
visual_env = [	["f","f","f","f","g","g","c","c","c","g"],
		["f","f","f","f","g","g","c","c","c","g"],
		["f","g","f","f","g","g","g","p","g","g"],
		["f","g","f","f","g","g","g","p","g","g"],
		["f","f","f","f","f","g","g","p","g","g"],
		["f","f","f","f","f","f","g","p","g","g"],
		["p","p","p","p","p","p","p","p","g","g"], 
		["g","g","p","f","g","g","g","g","g","g"],
		["g","g","p","p","g","g","g","g","g","g"],
		["g","g","g","g","g","g","g","g","g","g"]]

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
		
sound_env = [	["b","b","b"," "," ","m","m"," "," "," "],
		["b","b","b"," "," ","m","m"," "," "," "],
		["b","b","b"," "," "," "," ","m"," "," "],
		["b","b","b"," ","b"," "," "," ","h"," "],
		[" "," ","b"," "," "," ","h"," ","h"," "],
		[" "," "," "," ","h"," ","h"," "," "," "],
		[" "," "," "," "," ","h"," ","h"," "," "], 
		["h","h"," ","h"," "," ","e","e","e","e"],
		[" "," "," "," "," "," "," "," ","e","e"],
		[" "," "," "," "," "," "," "," ","e","e"]]		

env = [visual_env, odor_env, sound_env]

def find_clue(coord1, coord2):
	clues = []
	clues.append(env[0][coord1][coord2])
	clues.append(env[1][coord1][coord2])
	clues.append(env[2][coord1][coord2])
	return(clues)
	
print(find_clue(coord1,coord2))
