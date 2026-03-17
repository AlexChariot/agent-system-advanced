from typing import TypedDict, List

class AgentState(TypedDict):
    """
    Represents the state of an agent in the system.

    Attributes:
        goal (str): The main objective or goal that the agent is trying to achieve.
        plan (List[str]): A list of tasks or steps that the agent plans to execute to achieve the goal.
        current_task (str): The task that the agent is currently working on.

        research (str): Information gathered during the research phase.
        analysis (str): Results of the analysis performed on the research data.
        result (str): The final outcome or result after execution.

        history (List[str]): A record of previous actions or states for reference.
        evaluation (str): Assessment of the current state or performance.
    """

    print("***Initializing agent state...***")

    goal: str
    plan: List[str]
    current_task: str

    research: str
    analysis: str
    result: str

    history: List[str]
    evaluation: str