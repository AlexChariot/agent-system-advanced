#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def planner(state, model="llama3", output_format="list"):
    """
    Break down the goal into tasks using an LLM.

    Args:
        state (dict): The state containing the goal.
        model (str): The Ollama model to use. Default is "llama3".
        output_format (str): The format of the output. Can be "list" or "json". Default is "list".

    Returns:
        dict: A dictionary containing the plan and the current task.
    """
    print("***Planner breaking down the goal into tasks...***")

    goal = state["goal"]

    prompt = f"""
Break this goal into tasks.

Goal:
{goal}

Return a list of tasks.
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

    print("\n\t***Tasks planned:", tasks)

    if output_format == "json":
        import json
        tasks = json.dumps(tasks)

    return {
        "plan": tasks,
        "current_task": tasks[0] if isinstance(tasks, list) else json.loads(tasks)[0]
    }