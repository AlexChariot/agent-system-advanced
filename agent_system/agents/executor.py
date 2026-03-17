#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def executor(state, model="llama3", output_format="text"):
    """
    Execute the task using an LLM based on the analysis and goal.

    Args:
        state (dict): The state containing the analysis and goal.
        model (str): The Ollama model to use. Default is "llama3".
        output_format (str): The format of the output. Can be "text" or "json". Default is "text".

    Returns:
        dict: A dictionary containing the result of the execution.
    """
    print("***Executor executing the task...***")

    analysis = state["analysis"]
    goal = state["goal"]

    if not analysis or not goal:
        raise ValueError("Analysis and goal must not be empty.")

    prompt = f"""
Use this analysis to produce the final result.

Goal:
{goal}

Analysis:
{analysis}
"""

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    if not response.content:
        raise ValueError("The LLM response is empty or invalid.")

    print("\n\t***Execution completed with result:", response.content)

    if output_format == "json":
        import json
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {"result": response.content}
    else:
        result = response.content

    return {"result": result}
