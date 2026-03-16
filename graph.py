from langgraph.graph import StateGraph, END

from agents.planner import planner
from agents.researcher import researcher
from agents.executor import executor
from agents.critic import critic
from state import AgentState


def build_graph():
    
    print("***Building the agent graph...***")

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("executor", executor)
    graph.add_node("critic", critic)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "executor")
    graph.add_edge("executor", "critic")

    def should_continue(state):

        if "YES" in state["evaluation"]:
            return END

        return "planner"

    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "planner": "planner",
            END: END,
        },
    )
    
    print("\n\t***Graph built successfully!***")
    print("\n\tNodes:", graph.nodes)
    print("\n\tEdges:", graph.edges)

    return graph.compile()