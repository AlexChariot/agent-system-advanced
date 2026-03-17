from agent_system.tools.web_search import search

def researcher(state):
    
    print("***Researcher conducting research...***")

    task = state["current_task"]

    data = search(task)
    
    print("\n\t***Research conducted with data:", data)

    return {"research": data}