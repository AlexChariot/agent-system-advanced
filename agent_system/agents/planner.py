#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def planner(state, output_format="list"):
    """
    Break down the goal into tasks using an LLM.

    Args:
        state (dict): The state containing the goal and selected_model.
        output_format (str): The format of the output. Can be "list" or "json". Default is "list".

    Returns:
        dict: A dictionary containing the plan and the current task.
    """
    # print("***Planner breaking down the goal into tasks...***")

    goal = state["goal"]
    model = state.get("selected_model", "llama3.1")

    prompt = f"""
Break this goal into tasks.

Goal:
{goal}

Return a list of tasks, one per line, in the order they should be executed. Do not include numbering or bullet points.
"""

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])

    if not response.content:
        raise ValueError("The LLM response is empty or invalid.")

    tasks = response.content.split("\n")

    # Validate tasks
    tasks = [task.strip() for task in tasks if task.strip()]

    if not tasks:
        raise ValueError("No valid tasks were generated.")

    # print("\n\t***Tasks planned:", tasks)

    if output_format == "json":
        import json
        tasks = json.dumps(tasks)

    return {
        "plan": tasks,
        "current_task": tasks[0] if isinstance(tasks, list) else json.loads(tasks)[0]
    }