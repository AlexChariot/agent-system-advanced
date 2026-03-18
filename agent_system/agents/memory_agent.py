from agent_system.memory.vector_memory import recall_memory
import logging

logger = logging.getLogger(__name__)


def memory_agent(state: dict) -> dict:
    """
    Enriches the state with long-term memories relevant to the current goal.

    Called after the planner to inject context from the vector memory before
    the other agents begin working.

    Args:
        state (dict): Current state containing at least `goal`.

    Returns:
        dict: Updated state with `retrieved_memory`, `context`, and a `history` entry.
    """
    goal = state.get("goal", "")
    plan = state.get("plan", [])

    if not goal:
        logger.warning("[MemoryAgent] No goal found — skipping memory enrichment.")
        return {}

    # Query on goal + first task for more precise retrieval
    query = goal
    if plan:
        query = f"{goal} — {plan[0]}"

    logger.info(f"[MemoryAgent] Retrieving memories for: {query[:80]}...")
    retrieved = recall_memory(query)

    # Context compresses memories for downstream agents
    context = f"[Retrieved long-term memory]\n{retrieved}" if retrieved else ""

    logger.info(f"[MemoryAgent] Context injected ({len(retrieved)} chars).")

    return {
        "retrieved_memory": retrieved,
        "context": context,
        "history": [{"agent": "memory_agent", "retrieved_chars": len(retrieved)}],
    }