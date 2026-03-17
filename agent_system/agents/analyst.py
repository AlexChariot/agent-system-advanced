#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def analyst(state, model="llama3", output_format="text", insight_types=None):
    """
    Analyze the research and extract key insights using an LLM.

    Args:
        state (dict): The state containing the research.
        model (str): The Ollama model to use. Default is "llama3".
        output_format (str): The format of the output. Can be "text" or "json". Default is "text".
        insight_types (list): The types of insights to extract. Default is None.

    Returns:
        dict: A dictionary containing the analysis of the research.
    """
    print("***Analyst analyzing the research...***")

    research = state["research"]

    if not research:
        raise ValueError("Research must not be empty.")

    prompt = f"""
Analyze this research and extract key insights.

Research:
{research}
"""

    if insight_types:
        prompt += f"\n\nFocus on the following types of insights: {', '.join(insight_types)}"

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    if not response.content:
        raise ValueError("The LLM response is empty or invalid.")

    print("\n\t***Analysis completed with insights:", response.content)

    if output_format == "json":
        import json
        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            analysis = {"insights": response.content}
    else:
        analysis = response.content

    return {"analysis": analysis}
