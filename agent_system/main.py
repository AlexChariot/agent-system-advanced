# from graph import build_graph

import logging
from unittest import result
import subprocess


def choose_llm_model():
    """
    Allow user to choose an LLM model from available Ollama models.
    
    Returns:
        str: The selected model name, or None if no models are available or user cancels.
    """
    try:
        # Get list of available models from ollama
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split("\n")
        
        # Skip header line and parse model names
        if len(lines) <= 1:
            print("No Ollama models available. Please install a model first using 'ollama pull <model>'")
            return None
        
        models = []
        for line in lines[1:]:  # Skip header
            if line.strip():
                # Extract model name (first column)
                parts = line.split()
                if parts:
                    models.append(parts[0])
        
        if not models:
            print("No Ollama models found.")
            return None
        
        print("\nAvailable Ollama Models:")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = input(f"\nSelect a model (1-{len(models)}) or press Enter to skip: ").strip()
                if choice == "":
                    return None
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(models):
                    selected_model = models[choice_idx]
                    print(f"Selected model: {selected_model}")
                    return selected_model
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(models)}.")
            except ValueError:
                print(f"Invalid input. Please enter a number between 1 and {len(models)}.")
    
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please ensure Ollama is installed and in your PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running 'ollama list': {e.stderr}")
        return None
    except Exception as e:
        print(f"An error occurred while fetching models: {e}")
        return None


def main():
    """
    Main function to run the Agent System.

    This function builds the graph and allows the user to input goals.
    It then invokes the graph with the goal and prints the final result.
    """
    from agent_system.graph import build_graph
    print("Welcome to the Agent System!\n")
    print("Default LLM Model: llama3.1")
    print("(Select option 3 to change the model)\n")
    
    silence_logs = True  # Set to False to enable logging
    if silence_logs:
        logging.disable(logging.WARNING)    # Disable warnings and below (including info and debug logs)

    graph = build_graph()
    history = []
    selected_model = None

    while True:
        print("\nOptions:")
        print("1. Enter a new goal")
        print("2. View history")
        print("3. Choose LLM Model")
        print("4. Quit")

        choice = input("Choose an option (1-4): ")

        if choice == "1":
            goal = input("\nGoal: ")

            if not goal.strip():
                print("Error: Goal cannot be empty.")
                continue

            try:
                result = graph.invoke({
                    "goal": goal,
                    "plan": [],
                    "history": history,
                    "selected_model": selected_model or "llama3.1"
                })
                
                
                
                print("\nFINAL RESULT\n")
                print(result["result"])

                # Update history with the goal and result
                history.append({
                    "goal": goal,
                    "result": result["result"]
                })

            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == "2":
            if not history:
                print("No history available.")
            else:
                print("\nHistory:")
                for i, item in enumerate(history, 1):
                    print(f"{i}. Goal: {item['goal']}")
                    print(f"   Result: {item['result']}\n")

        elif choice == "3":
            selected_model = choose_llm_model()
            if selected_model:
                print(f"Current model: {selected_model}")

        elif choice == "4":
            print("Exiting the Agent System.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
