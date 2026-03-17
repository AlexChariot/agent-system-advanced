from typing import TypedDict, List, Dict, Annotated
import operator

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

        history (List[Dict]): A record of previous actions or states for reference.
        context (str): Compressed information that provides context for the agent's decisions.
        retrieved_memory (str): Long-term memory that can be retrieved for use in decision-making.
        evaluation (str): Assessment of the current state or performance.
    """

    goal: str
    plan: List[str]
    current_task: str

    research: str
    analysis: str
    result: str

    history: Annotated[List[Dict], operator.add]    # historique des actions précédentes
    context: str                                    # informations compressées pour fournir du contexte aux agents
    retrieved_memory: str                           # mémoire long terme récupérée pour être utilisée dans la prise de décision
    evaluation: str                                 # évaluation de l'état actuel ou de la performance