#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3")

def executor(state):
    
    print("***Executor executing the task...***")

    analysis = state["analysis"]
    goal = state["goal"]

    prompt = f"""
Use this analysis to produce the final result.

Goal:
{goal}

Analysis:
{analysis}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("\n\t***Execution completed with result:", response.content)

    return {"result": response.content}