#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3")

def critic(state):

    result = state["result"]
    goal = state["goal"]

    prompt = f"""
Evaluate if this result satisfies the goal.

Goal:
{goal}

Result:
{result}

Answer YES or NO.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"evaluation": response.content}