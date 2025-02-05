import requests
import subprocess
from openai import OpenAI
from script_environment import find_clue


def get_openai_apikey() -> str:
	with open("OPENAI_API_KEY", "r") as f:
		key = f.read()
	return key


def get_openai_client() -> OpenAI:
	client = OpenAI(api_key=get_openai_apikey())
	return client


def get_response_from_openai_prompt(client, prompt: str, context: str ,model:str="gpt-3.5-turbo"):
	response = client.chat.completions.create(
		model=model,
		messages=[
			{"role": "system", "content": context},
			{"role": "user", "content": prompt}
		],
		temperature=0.7,  # Plus élevé = plus créatif, plus bas = plus précis
	)
	return response.choices[0].message.content


def get_response_from_ollama_prompt(prompt:str, model:str="mistral-openorca") -> list[str]:
	# Requête à Ollama
	response = requests.post("http://localhost:11434/api/generate",
							 json={"model": model, "prompt": prompt, "stream": False})

	return response.json()["response"]


def context_prompt_2d() -> str:
	prompt = "You are placed in a two dimensional environment. Your goal is to find a city using visual, olfactive and"
	prompt += " auditive clues I will provide. You must choose a direction, between Left, Right, Up and Down. "
	prompt += "There is three type of environment. Forest, grass fields and road. Forest is the worse environment for "
	prompt += "you. Grass field is the second worse. Road is the best type of environment for you. "
	prompt += "You know that the road is leading to the city somehow, but you don't know where is the city."
	prompt += "Give me only the direction you choose, without a justification. If you are on in a corner or on a border"
	prompt += " you won't be able to choose certain direction, but I will tell you when it is the case."

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


def prompt_with_action_2d_and_proximity_cue(action: int, is_action_bringing_closer_to_goal: bool):
	if action == 0:
		translated_action = "Up"
	elif action == 1:
		translated_action = "Down"
	elif action == 2:
		translated_action = "Left"
	else:
		translated_action = "Right"

	prompt = f"The agent wants to go {translated_action}. "
	prompt += f"This action moves you {'closer to your objective' if is_action_bringing_closer_to_goal else 'away from your objective'}."
	prompt += "Which direction do you choose ?"

	return prompt


def llm_supervision(position, suggested_direction_untranslated, direction_certitude, history, objective="city"):
	x, y = position
	place_info = find_clue(x,y)	# Load the environment characteristics
	
	#convert to something the llm understand
	if (suggested_direction_untranslated)==0:
		suggested_direction = "up"
	elif (suggested_direction_untranslated)==1:
		suggested_direction = "down"
	elif (suggested_direction_untranslated)==2:
		suggested_direction = "left"
	elif (suggested_direction_untranslated)==3:
		suggested_direction = "right"
	
	#Make prompt
	prompt = "You are looking for a "+ objective +". You know the "+ objective + " isn't in a forest, you know there is road leading to it, but you don't know where it is. Therefore finding a road and following it may be a good strategy. You are now in a " + str(place_info[0]) + ", you smell " + str(place_info[1]) + " and you hear " + str(place_info[2]) + ". A reinforcement learning algorithm hint you to go " + suggested_direction + " with a certitude of " + str(direction_certitude) + ". Carefully considering what you know and the reinforcement learning idea, try to reason on what you know and say a direction (left, right, up, down) to go toward the "+ objective +""
	
	if (x == 0 and y == 9):  
		command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose right or up."]

	elif (x == 9 and y == 0): 
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose left or down."]

	elif (x == 9 and y == 9):  
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose right or down."]

	elif (x == 0 and y == 0):
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose up or left."]

	elif x == 0: 
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose up."]

	elif x == 9: 
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose down."]

	elif y == 0:
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose left."]

	elif y == 9:
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt + "Forbidden to choose right."]

	else:
	    command2 = ["ollama", "run", "mistral-openorca"] + [prompt]

	justification = subprocess.run(command2, capture_output=True, text=True)
	
	print((justification.stdout).lower())
	
	command_dir = ["ollama", "run", "mistral-openorca"] + ["Give ONLY the direction (left, right, up, down) this text suggest to go to, nothing more :" + justification.stdout]
	
	direction = subprocess.run(command_dir, capture_output=True, text=True)
	
	print((direction.stdout).lower())
	
	'''
	def find_last_occurring_string(text, string_list):
		last_string = None
		last_position = 0

		for string in string_list:
			pos = text.rfind(string)  # Find the last occurrence of the string
			if pos > last_position:   # Update if it's the latest occurring
				last_position = pos
				last_string = string
				
		return last_string
	
	
	supervised_direction = find_last_occurring_string((direction.stdout).lower(), ["left", "right", "up", "down"])
	
	if "right" == supervised_direction :
		supervised_direction = 3
		
	if "left" == supervised_direction:
		supervised_direction = 2
		
	if "up" == supervised_direction:
		supervised_direction = 0
		
	if "down" == supervised_direction:
		supervised_direction = 1
	'''
	
	
	if "right" in (direction.stdout).lower():
		supervised_direction = 3
		
	if "left" in (direction.stdout).lower():
		supervised_direction = 2
		
	if "up" in (direction.stdout).lower():
		supervised_direction = 0
		
	if "down" in (direction.stdout).lower():
		supervised_direction = 1
	
	history += "you said this in previous iterations : " + justification.stdout + ". "
	return(supervised_direction, )


if "__main__" == __name__:
	resp = get_response_from_ollama_prompt("This is a test prompt say a few words")
	print(resp)

	# openai_client = get_openai_client()
	#
	# resp = get_response_from_openai_prompt(openai_client, prompt="Test prompt. Say test",
	# 									   context="You are an assistant")
	# # Afficher la réponse du modèle
	# print(resp)


