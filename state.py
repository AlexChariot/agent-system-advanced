from typing import TypedDict, List

class AgentState(TypedDict):
    
    print("***Defining the agent state...***")

    goal: str
    plan: List[str]
    current_task: str
    research: str
    result: str
    history: List[str]