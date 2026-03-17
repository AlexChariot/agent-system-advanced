from langgraph.graph import StateGraph, END

from agent_system.state import AgentState
from agent_system.agents.planner import planner
from agent_system.agents.researcher import researcher
from agent_system.agents.analyst import analyst
from agent_system.agents.executor import executor
from agent_system.agents.critic import critic

def build_graph():

    print("***Building the agent graph...***")

    graph = StateGraph(AgentState)

    # Add all agent nodes to the graph
    graph.add_node("planner", planner)      # Initial planning node
    graph.add_node("researcher", researcher)  # Research and information gathering
    graph.add_node("analyst", analyst)      # Analysis of gathered information
    graph.add_node("executor", executor)    # Execution of actions based on analysis
    graph.add_node("critic", critic)        # Evaluation of results

    # Set the entry point of the graph
    graph.set_entry_point("planner")

    # Define the main workflow sequence
    graph.add_edge("planner", "researcher")     # Planner -> Researcher
    graph.add_edge("researcher", "analyst")     # Researcher -> Analyst
    graph.add_edge("analyst", "executor")       # Analyst -> Executor
    graph.add_edge("executor", "critic")        # Executor -> Critic

    def loop(state):
        """
        Conditional edge function to determine if the workflow should continue or end.
        Args:
            state: The current state of the graph
        Returns:
            The next node to execute or END if the workflow is complete
        """
        if "YES" in state["evaluation"]:
            return END

        return "planner"

    # Add conditional edges from critic to either continue the loop or end
    graph.add_conditional_edges(
        "critic",
        loop,
        {
            "planner": "planner",  # Continue the loop if evaluation is not "YES"
            END: END                # End the workflow if evaluation is "YES"
        }
    )
    
    print("\n***Agent graph built successfully!***")

    return graph.compile()