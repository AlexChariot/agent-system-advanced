# from graph import build_graph

import logging
from unittest import result


def main():
    """
    Main function to run the Agent System.

    This function builds the graph and allows the user to input goals.
    It then invokes the graph with the goal and prints the final result.
    """
    from agent_system.graph import build_graph
    print("Welcome to the Agent System!\n")
    
    silence_logs = True  # Set to False to enable logging
    if silence_logs:
        logging.disable(logging.WARNING)    # Disable warnings and below (including info and debug logs)

    graph = build_graph()
    history = []

    while True:
        print("\nOptions:")
        print("1. Enter a new goal")
        print("2. View history")
        print("3. Quit")

        choice = input("Choose an option (1-3): ")

        if choice == "1":
            goal = input("\nGoal: ")

            if not goal.strip():
                print("Error: Goal cannot be empty.")
                continue

            try:
                result = graph.invoke({
                    "goal": goal,
                    "plan": [],
                    "history": history
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
            print("Exiting the Agent System.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 3.")

if __name__ == "__main__":
    main()
