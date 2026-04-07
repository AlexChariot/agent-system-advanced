from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)


def planner(state: dict) -> dict:
    """
    Breaks the goal down into an ordered list of atomic tasks via the LLM.

    Always returns `plan` as a List[str], consistent with AgentState, which
    guarantees compatibility with the manager and the researcher.

    Args:
        state (dict): Current state containing `goal` and `selected_model`.

    Returns:
        dict: `plan` (List[str]), `current_task` (str), and a `history` entry.
    """
    goal = state["goal"]
    model = state.get("selected_model", "llama3.1")

    prompt = f"""Break this goal down into ordered, atomic tasks.

Goal:
{goal}

Return one task per line, with no numbering, no bullet points and no introductive text.
"""

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])

    if not response.content:
        raise ValueError("[Planner] The LLM response is empty.")

    tasks: list[str] = [t.strip() for t in response.content.split("\n") if t.strip()]

    if not tasks:
        raise ValueError("[Planner] No tasks were generated.")

    logger.info(f"[Planner] {len(tasks)} task(s) planned.")

    return {
        "plan": tasks,
        "current_task": tasks[0],
        "research": "",
        "analysis": "",
        "result": "",
        "evaluation": "",
        "completed_tasks": [],
        "history": [{"agent": "planner", "tasks": tasks}],
    }
