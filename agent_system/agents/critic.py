#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def critic(state, model="llama3", output_format="text"):
    """
    Evaluate if the result satisfies the goal using an LLM.

    Args:
        state (dict): The state containing the result and goal.
        model (str): The Ollama model to use. Default is "llama3".
        output_format (str): The format of the output. Can be "text" or "boolean". Default is "text".

    Returns:
        dict: A dictionary containing the evaluation of the result.
    """
    result = state["result"]
    goal = state["goal"]

    if not result or not goal:
        raise ValueError("Result and goal must not be empty.")

    prompt = f"""
Evaluate if this result satisfies the goal.

Goal:
{goal}

Result:
{result}

Answer YES or NO.
"""

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])

    if not response.content:
        raise ValueError("The LLM response is empty or invalid.")

    evaluation = response.content.strip().upper()

    if evaluation not in ["YES", "NO"]:
        raise ValueError("The LLM response must be either 'YES' or 'NO'.")

    if output_format == "boolean":
        evaluation = evaluation == "YES"

    return {"evaluation": evaluation}
