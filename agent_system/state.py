from typing import TypedDict, List

class AgentState(TypedDict):
    
    print("***Initializing agent state...***")

    goal: str
    plan: List[str]
    current_task: str

    research: str
    analysis: str
    result: str

    history: List[str]
    evaluation: str