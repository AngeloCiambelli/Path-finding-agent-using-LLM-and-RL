import subprocess
import json

def chat_with_ollama():
    model = "mistral-openorca"  # Change to your preferred model
    history = []  # Store conversation history
    
    print("Starting chat with Ollama. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat.")
            break

        # Append user message to history
        history.append({"role": "user", "content": user_input})

        # Convert history to JSON string for the command
        history_json = json.dumps(history)

        # Run the Ollama command with full conversation history
        process = subprocess.run(
            ["ollama", "run", model, history_json],
            capture_output=True,
            text=True
        )

        # Get Ollama's response
        response = process.stdout.strip()
        print("Ollama:", response)

        # Append Ollama's response to history
        history.append({"role": "assistant", "content": response})

# Run the chatbot
chat_with_ollama()

