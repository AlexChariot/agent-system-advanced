#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3")

def analyst(state):
    
    print("***Analyst analyzing the research...***")

    research = state["research"]

    prompt = f"""
Analyze this research and extract key insights.

Research:
{research}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("\n\t***Analysis completed with insights:", response.content)

    return {"analysis": response.content}