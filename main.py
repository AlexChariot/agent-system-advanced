from graph import build_graph

def main():
    
    print("Welcome to the Agent System!\n")

    graph = build_graph()

    while True:

        goal = input("\nGoal: ")

        result = graph.invoke({
            "goal": goal,
            "plan": [],
            "history": []
        })

        print("\nFINAL RESULT\n")
        print(result["result"])


if __name__ == "__main__":
    main()