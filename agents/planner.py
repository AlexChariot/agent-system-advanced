#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3")

def planner(state):
    print("***Planner creating a plan...***")   

    goal = state["goal"]

    prompt = f"""
Create a step-by-step plan to solve this goal.

Goal:
{goal}

Return a list of tasks.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    tasks = response.content.split("\n")
    
    print("\n\t***Plan created:", tasks)

    return {
        "plan": tasks,
        "current_task": tasks[0]
    }