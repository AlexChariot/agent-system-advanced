from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)

# Available agents and their roles (used in the LLM prompt)
AGENT_DESCRIPTIONS = {
    "planner":    "Breaks the goal down into an ordered list of atomic tasks.",
    "researcher": "Performs a web search for the current task.",
    "analyst":    "Analyzes the research data and extracts key insights.",
    "executor":   "Produces the final result from the analysis.",
    "critic":     "Evaluates whether the result satisfies the goal and decides whether to stop or continue.",
}


def manager(state: dict) -> dict:
    """
    Manager agent: decides which agent to call next based on the current state.

    Strategy:
    1. Deterministic rules first (fast and reliable for obvious cases).
    2. LLM fallback when the state is ambiguous (e.g. after a negative evaluation).

    Args:
        state (dict): The current system state.

    Returns:
        dict: Updated state with `next_agent` and a `history` entry.
    """
    plan = state.get("plan", [])
    research = state.get("research", "")
    analysis = state.get("analysis", "")
    result = state.get("result", "")
    evaluation = state.get("evaluation", "")
    model = state.get("selected_model", "llama3.1")

    # ── 1. Deterministic rules ───────────────────────────────────────────────

    if not plan:
        next_agent = "planner"
        reason = "No plan available — starting with the planner."

    elif not research:
        next_agent = "researcher"
        reason = "Plan ready, but no research has been done yet."

    elif not analysis:
        next_agent = "analyst"
        reason = "Research available, but no analysis has been performed yet."

    elif not result:
        next_agent = "executor"
        reason = "Analysis available, but no result has been produced yet."

    elif result and not evaluation:
        next_agent = "critic"
        reason = "Result available — sending to the critic for evaluation."

    elif evaluation == "NO":
        next_agent = _llm_decide(state, model)
        reason = f"Negative evaluation — LLM re-routing decision: {next_agent}."

    else:
        next_agent = _llm_decide(state, model)
        reason = f"Ambiguous state — LLM routing decision: {next_agent}."

    logger.info(f"[Manager] → {next_agent} | {reason}")

    return {
        "next_agent": next_agent,
        "history": [{"agent": "manager", "decision": next_agent, "reason": reason}],
    }


def _llm_decide(state: dict, model: str) -> str:
    """
    Delegates the routing decision to the LLM when deterministic rules are
    insufficient (e.g. negative evaluation, ambiguous state).

    Args:
        state (dict): The current state.
        model (str): The Ollama model to use.

    Returns:
        str: The name of the next agent (always a key in AGENT_DESCRIPTIONS).
    """
    agent_list = "\n".join(
        f"- {name}: {desc}" for name, desc in AGENT_DESCRIPTIONS.items()
    )

    prompt = f"""You are the manager of a multi-agent system. Your role is to choose which agent to call next.

=== CURRENT STATE ===
Goal       : {state.get('goal', 'N/A')}
Plan       : {state.get('plan', [])}
Task       : {state.get('current_task', 'N/A')}
Research   : {'✓ present' if state.get('research') else '✗ absent'}
Analysis   : {'✓ present' if state.get('analysis') else '✗ absent'}
Result     : {'✓ present' if state.get('result') else '✗ absent'}
Evaluation : {state.get('evaluation', 'N/A')}

=== AVAILABLE AGENTS ===
{agent_list}

Reply with ONLY the exact agent name (e.g. researcher). No explanation.
"""

    llm = ChatOllama(model=model)
    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()

    if decision not in AGENT_DESCRIPTIONS:
        logger.warning(
            f"[Manager/LLM] Unexpected response '{decision}', falling back to 'planner'."
        )
        decision = "planner"

    return decision