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

    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("executor", executor)
    graph.add_node("critic", critic)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "executor")
    graph.add_edge("executor", "critic")

    def loop(state):

        if "YES" in state["evaluation"]:
            return END

        return "planner"

    graph.add_conditional_edges(
        "critic",
        loop,
        {
            "planner": "planner",
            END: END
        }
    )
    
    print("\n***Agent graph built successfully!***")

    return graph.compile()