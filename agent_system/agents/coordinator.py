#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3")

def coordinator(state):
    
    print("***Coordinator determining the next agent...***")

    goal = state["goal"]

    prompt = f"""
Determine which type of agent should work next.

Goal: {goal}

Options:
planner
researcher
analyst
executor

Return only the agent name.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("\n\t***Next agent determined:", response.content)

    return {"next_agent": response.content.strip()}