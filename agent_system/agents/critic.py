from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)


def critic(state: dict, output_format: str = "text") -> dict:
    """
    Evaluates whether the result satisfies the goal.

    Returns "YES" to end the workflow, or "NO" to send it back to the manager.

    Args:
        state (dict): Current state containing `result`, `goal`, and `selected_model`.
        output_format (str): "text" (default) or "boolean".

    Returns:
        dict: `evaluation` ("YES" | "NO") and a `history` entry.
    """
    result = state.get("result", "")
    goal = state.get("goal", "")
    model = state.get("selected_model", "llama3.1")

    if not result:
        raise ValueError("[Critic] The result field is empty.")
    if not goal:
        raise ValueError("[Critic] The goal field is empty.")

    prompt = f"""You are a critic agent. Evaluate whether this result satisfies the goal.

Goal:
{goal}

Result:
{result}

Reply with ONLY YES or NO.
- YES: the result is complete and satisfactory.
- NO:  the result is incomplete, incorrect, or insufficient.
"""

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])

    if not response.content:
        raise ValueError("[Critic] The LLM response is empty.")

    raw = response.content.strip().upper()

    if "YES" in raw:
        evaluation = "YES"
    elif "NO" in raw:
        evaluation = "NO"
    else:
        logger.warning(f"[Critic] Ambiguous response '{raw}', falling back to NO.")
        evaluation = "NO"

    if output_format == "boolean":
        return {"evaluation": evaluation == "YES"}

    logger.info(f"[Critic] Evaluation: {evaluation}")

    return {
        "evaluation": evaluation,
        "history": [{"agent": "critic", "evaluation": evaluation}],
    }