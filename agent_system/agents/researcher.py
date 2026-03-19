from agent_system.tools.web_search import search
import logging

logger = logging.getLogger(__name__)


def researcher(state: dict) -> dict:
    """
    Performs a web search for the current task.

    Also advances the plan: removes the processed task and sets `current_task`
    to the next one if available. When the plan is exhausted, `current_task`
    is set to None to signal the manager that there are no more tasks.

    Args:
        state (dict): Current state containing `current_task` and `plan`.

    Returns:
        dict: `research`, updated `plan`, `current_task`, and a `history` entry.
    """
    task = state.get("current_task")
    plan = state.get("plan", [])

    if not task:
        raise ValueError("[Researcher] No current task found in state.")

    logger.info(f"[Researcher] Searching for: {task}")
    data = search(task)

    if not data:
        logger.warning("[Researcher] No results found.")
        data = "No relevant information found."

    # Advance the plan by removing only the first instance of the completed task
    remaining_plan = plan.copy()
    try:
        remaining_plan.remove(task)
    except ValueError:
        # Task not in plan, this shouldn't happen but handle gracefully
        logger.warning(f"[Researcher] Task '{task}' not found in plan.")

    # None signals to the manager that the plan is exhausted
    next_task = remaining_plan[0] if remaining_plan else None

    if next_task is None:
        logger.info("[Researcher] Plan exhausted — all tasks have been processed.")
    else:
        logger.info(f"[Researcher] Next task: {next_task}")

    logger.info(f"[Researcher] Search complete ({len(data)} chars).")

    return {
        "research": data,
        "plan": remaining_plan,
        "current_task": next_task,
        "history": [{"agent": "researcher", "task": task, "chars": len(data)}],
    }