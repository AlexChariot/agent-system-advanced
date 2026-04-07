from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import logging
import json

logger = logging.getLogger(__name__)


def analyst(state: dict, output_format: str = "text", insight_types: list = None) -> dict:
    """
    Analyzes research data and extracts key insights via the LLM.

    Args:
        state (dict): Current state containing `research` and `selected_model`.
        output_format (str): "text" (default) or "json".
        insight_types (list): Optional list of insight categories to focus on.

    Returns:
        dict: Aggregated `analysis`, updated task progress, and a `history` entry.
    """
    research = state.get("research", "")
    current_task = state.get("current_task")
    plan = state.get("plan", [])
    completed_tasks = state.get("completed_tasks", [])
    previous_analysis = state.get("analysis", "")
    model = state.get("selected_model", "llama3.1")

    if not research:
        raise ValueError("[Analyst] The research field is empty.")
    if not current_task:
        raise ValueError("[Analyst] No current task found in state.")

    prompt = f"""Analyze this research and extract the key insights.

Research:
{research}
"""
    if insight_types:
        prompt += f"\n\nFocus on: {', '.join(insight_types)}"

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])

    if not response.content:
        raise ValueError("[Analyst] The LLM response is empty.")

    if output_format == "json":
        try:
            task_analysis = json.loads(response.content)
        except json.JSONDecodeError:
            task_analysis = {"insights": response.content}
    else:
        task_analysis = response.content

    if output_format == "json":
        aggregated_analysis = {
            "completed_tasks": [*completed_tasks, current_task],
            "task_analysis": task_analysis,
        }
    else:
        section_header = f"Task: {current_task}"
        section_body = str(task_analysis).strip()
        aggregated_analysis = (
            f"{previous_analysis}\n\n{section_header}\n{section_body}".strip()
            if previous_analysis
            else f"{section_header}\n{section_body}"
        )

    remaining_plan = plan.copy()
    if remaining_plan and remaining_plan[0] == current_task:
        remaining_plan = remaining_plan[1:]
    else:
        try:
            remaining_plan.remove(current_task)
        except ValueError:
            logger.warning(f"[Analyst] Task '{current_task}' not found in plan.")

    next_task = remaining_plan[0] if remaining_plan else None

    logger.info("[Analyst] Analysis complete.")

    return {
        "analysis": aggregated_analysis,
        "completed_tasks": [*completed_tasks, current_task],
        "plan": remaining_plan,
        "current_task": next_task,
        "research": "",
        "result": "",
        "evaluation": "",
        "history": [{"agent": "analyst", "task": current_task, "chars": len(str(task_analysis))}],
    }
