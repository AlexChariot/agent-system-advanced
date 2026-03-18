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
        dict: `analysis` and a `history` entry.
    """
    research = state.get("research", "")
    model = state.get("selected_model", "llama3.1")

    if not research:
        raise ValueError("[Analyst] The research field is empty.")

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
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            analysis = {"insights": response.content}
    else:
        analysis = response.content

    logger.info("[Analyst] Analysis complete.")

    return {
        "analysis": analysis,
        "history": [{"agent": "analyst", "chars": len(str(analysis))}],
    }