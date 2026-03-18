from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from agent_system.memory.vector_memory import recall_memory

def memory_agent(state):
    """
    Manage and compress context using an LLM.

    Args:
        state (dict): The state containing history, goal, and selected_model.

    Returns:
        dict: Updated state with context and retrieved_memory.
    """
    history = state.get("history", [])
    goal = state["goal"]
    model = state.get("selected_model", "llama3.1")

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

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "context": response.content,
        "retrieved_memory": retrieved
    }