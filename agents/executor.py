#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from memory.vector_memory import store_memory

llm = ChatOllama(model="llama3")

def executor(state):
    print("***Executor executing the task...***") 
    print("\n\tCurrent state:", state)
    print("\n\tCurrent task:", state.get("current_task"))  

    task = state["current_task"]
    research = state["research"]

    prompt = f"""
Execute this task.

Task:
{task}

Research:
{research}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    result = response.content

    store_memory(result)
    
    print("\n\t***Task executed with result:", result)

    return {"result": result}