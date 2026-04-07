from agent_system.tools.web_search import search
import logging

logger = logging.getLogger(__name__)


def researcher(state: dict) -> dict:
    """
    Performs a web search for the current task.

    Leaves task advancement to the analyst so the manager can always finish
    analysis for the task that was just researched before moving on.

    Args:
        state (dict): Current state containing `current_task` and `plan`.

    Returns:
        dict: `research` for the current task and a `history` entry.
    """
    task = state.get("current_task")
    if not task:
        raise ValueError("[Researcher] No current task found in state.")

    logger.info(f"[Researcher] Searching for: {task}")
    data = search(task)

    if not data:
        logger.warning("[Researcher] No results found.")
        data = "No relevant information found."

    logger.info(f"[Researcher] Search complete ({len(data)} chars).")

    return {
        "research": data,
        "history": [{"agent": "researcher", "task": task, "chars": len(data)}],
    }
