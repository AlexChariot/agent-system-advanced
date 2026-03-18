from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from agent_system.memory.vector_memory import store_memory
import logging

logger = logging.getLogger(__name__)


def executor(state: dict) -> dict:
    """
    Produces the final result from the context, goal, and analysis.

    Also stores the result in long-term vector memory for future retrieval.

    Args:
        state (dict): Current state containing `context`, `analysis`, `goal`,
                      and `selected_model`.

    Returns:
        dict: `result` and a `history` entry.
    """
    context = state.get("context", "")
    analysis = state.get("analysis", "")
    goal = state.get("goal", "")
    model = state.get("selected_model", "llama3.1")

    if not analysis:
        raise ValueError("[Executor] The analysis field is empty.")
    if not goal:
        raise ValueError("[Executor] The goal field is empty.")

    prompt = f"""You are an executor agent. Produce the best possible final result.

Context:
{context}

Goal:
{goal}

Analysis:
{analysis}

Produce a complete, directly usable response.
"""

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])

    result = response.content
    store_memory(f"Goal: {goal}\nResult: {result}")

    logger.info(f"[Executor] Result produced ({len(result)} chars).")

    return {
        "result": result,
        "history": [{"agent": "executor", "chars": len(result)}],
    }