import random
import numpy as np
import requests

def get_response_from_ollama_prompt(prompt:str, model:str="mistral-openorca") -> list[str]:
	# Requête à Ollama
	response = requests.post("http://localhost:11434/api/generate",
							 json={"model": model, "prompt": prompt, "stream": False})

	return response.json()["response"]


def context_prompt_2d() -> str:
	prompt = "You are placed on a two dimensional environment and you must collaborate with another agent to find the"
	prompt += "objective state. You must choose a direction between Left, Right, Up and Down. Answer with only one of these four "
	prompt += "words, without any justification"

	return prompt


def context_prompt_1d() -> str:
	prompt = "You are placed on a one dimensional environment and you must collaborate with another agent to find the"
	prompt += "objective state. You must choose a direction between Left and Right. Answer with only one of these two "
	prompt += "words, without any justification"

	return prompt


def prompt_with_action_1d_and_proximity_cue(action: int, is_action_bringing_closer_to_goal: bool):
	translated_action = "Left" if action == 0 else "Right"
	prompt = f"The agent wants to go {translated_action}. "
	prompt += f"This action moves you {'closer to your objective' if is_action_bringing_closer_to_goal else 'away from your objective'}."
	prompt += "Which direction do you choose ?"

	return prompt


def prompt_with_action_2d_and_proximity_cue(action: int, is_action_bringing_closer_to_goal: bool, state: tuple[int, int]):
	x, y = state
	
	if action == 0:
		translated_action = "Left"
	elif action == 1:
		translated_action = "Right"
	elif action == 2:
		translated_action = "Up"
	else:
		translated_action = "Down"
		
	prompt = f"The agent wants to go {translated_action}. "
	prompt += f"This action moves you {'closer to your objective' if is_action_bringing_closer_to_goal else 'away from your objective'}."
		
	if (x == 0 and y == 4):  
		prompt += "You are on the edge, so forbidden to choose Left or Down."

	elif (x == 4 and y == 0): 
		prompt += "You are on the edge, forbidden to choose Right or Up."

	elif (x == 4 and y == 4):  
		prompt += "You are on the edge, forbidden to choose Right or Down."

	elif (x == 0 and y == 0):
		prompt += "You are on the edge, forbidden to choose Up or Left."

	elif x == 0: 
		prompt += "You are on the edge, forbidden to choose Left."

	elif x == 4: 
		prompt += "You are on the edge, forbidden to choose Right."

	elif y == 0:
		prompt += "You are on the edge, forbidden to choose Up."

	elif y == 4:
	    	prompt += "You are on the edge, forbidden to choose Down."
	    	
	prompt += "Which direction do you choose, saying only (Right, Left, Up or Down)?"

	return prompt

