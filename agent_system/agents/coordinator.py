#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def coordinator(state, model="llama3.1", agent_options=None):
    """
    Determine the next agent to work on the goal using an LLM.

    Args:
        state (dict): The state containing the goal.
        model (str): The Ollama model to use. Default is "llama3.1".
        agent_options (list): The list of available agent options. Default is ["planner", "researcher", "analyst", "executor"].

    Returns:
        dict: A dictionary containing the next agent to work on the goal.
    """
    # print("***Coordinator determining the next agent...***")

    goal = state["goal"]

    if not goal:
        raise ValueError("Goal must not be empty.")

    if agent_options is None:
        agent_options = ["planner", "researcher", "analyst", "executor"]

    prompt = f"""
Determine which type of agent should work next.

Goal: {goal}

Options:
{', '.join(agent_options)}

Return only the agent name.
"""

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])

    if not response.content:
        raise ValueError("The LLM response is empty or invalid.")

    next_agent = response.content.strip().lower()

    if next_agent not in agent_options:
        raise ValueError(f"The LLM response must be one of the following: {', '.join(agent_options)}.")

    # print("\n\t***Next agent determined:", next_agent)

    return {"next_agent": next_agent}
