from tools.web_search import search
from memory.vector_memory import recall_memory

def researcher(state):
    print("***Researcher researching the task...***")

    task = state["current_task"]

    web = search(task)
    memory = recall_memory(task)
    
    print("\n\t***Web research:", web)
    print("\n\t***Memory recall:", memory)

    return {
        "research": f"""
WEB:
{web}

MEMORY:
{memory}
"""
    }