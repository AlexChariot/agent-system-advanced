#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from agent_system.memory.vector_memory import store_memory

llm = ChatOllama(model="llama3.1")  # initialisation du modèle une fois pour tous les appels à executor

def executor(state):

    context = state.get("context", "")
    analysis = state["analysis"]
    goal = state["goal"]

    history = state.get("history", [])

    prompt = f"""
Context:
{context}

Goal:
{goal}

Analysis:
{analysis}

Produce the best final result.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    result = response.content
    
    store_memory(f"Goal: {goal}\nResult: {result}")

    # 🔥 mise à jour mémoire court terme
    # history.append({
    #     "agent": "executor",
    #     "result": result
    # })
    history = history + [{
        "agent": "executor",
        "result": result
    }]    

    return {
        "result": result,
        "history": [{"agent": "executor", "result": result}]  # juste le nouvel élément
    }
    