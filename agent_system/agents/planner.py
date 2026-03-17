#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3")

def planner(state):
    
    print("***Planner breaking down the goal into tasks...***")

    goal = state["goal"]

    prompt = f"""
Break this goal into tasks.

Goal:
{goal}

Return a list of tasks.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    tasks = response.content.split("\n")
    
    print("\n\t***Tasks planned:", tasks)

    return {
        "plan": tasks,
        "current_task": tasks[0]
    }