from langgraph.graph import StateGraph, END

from agent_system.state import AgentState
from agent_system.agents.manager import manager
from agent_system.agents.planner import planner
from agent_system.agents.researcher import researcher
from agent_system.agents.analyst import analyst
from agent_system.agents.executor import executor
from agent_system.agents.critic import critic
from agent_system.agents.memory_agent import memory_agent


def _route_from_manager(state: AgentState) -> str:
    """
    Pure routing function: reads `next_agent` from the state and returns
    the name of the next node.
    Kept separate from the manager node to avoid double invocation.
    """
    return state.get("next_agent", "planner")


def _should_end(state: AgentState) -> str:
    """
    After the critic: ends the workflow if the evaluation is positive,
    otherwise routes back to the manager.
    """
    if state.get("evaluation", "") == "YES":
        return END
    return "manager"


def build_graph():
    graph = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("manager", manager)            # decides next_agent
    graph.add_node("memory_agent", memory_agent)  # enriches context after planning
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("executor", executor)
    graph.add_node("critic", critic)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("manager")

    # ── Routing from the manager ──────────────────────────────────────────────
    graph.add_conditional_edges(
        "manager",
        _route_from_manager,
        {
            "planner":    "planner",
            "researcher": "researcher",
            "analyst":    "analyst",
            "executor":   "executor",
            "critic":     "critic",
        },
    )

    # ── After the planner: memory enrichment, then back to the manager ────────
    graph.add_edge("planner", "memory_agent")
    graph.add_edge("memory_agent", "manager")

    # ── All other agents route back through the manager ───────────────────────
    graph.add_edge("researcher", "manager")
    graph.add_edge("analyst", "manager")
    graph.add_edge("executor", "manager")

    # ── Conditional exit from the critic ─────────────────────────────────────
    graph.add_conditional_edges(
        "critic",
        _should_end,
        {
            "manager": "manager",
            END: END,# The `invoke` method is being called on the `graph` object in the `run`
            # command function. This method is used to trigger the execution of the agent
            # system with a specific goal. The `invoke` method takes a dictionary
            # containing information about the goal, plan, history, and selected model as
            # input parameters. It then processes this information within the agent system
            # to determine the result of the execution based on the defined logic and rules
            # within the system. The result of the `invoke` method call is then displayed
            # to the user as the final output of the execution.
            
        },
    )

    return graph.compile()