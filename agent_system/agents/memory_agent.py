from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from agent_system.memory.vector_memory import recall_memory

llm = ChatOllama(model="llama3.1")  # initialisation du modèle une fois pour tous les appels à memory_agent

def memory_agent(state):

    history = state.get("history", [])
    goal = state["goal"]

    # Récupération mémoire long terme
    retrieved = recall_memory(goal)

    # Compression du contexte
    prompt = f"""
Summarize the important context for solving the goal.

Goal:
{goal}

History:
{history}

Relevant past memory:
{retrieved}

Return a concise context.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "context": response.content,
        "retrieved_memory": retrieved
    }