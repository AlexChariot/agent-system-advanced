from agent_system.tools.web_search import search
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def researcher(state):
    """
    Conducts research based on the current task in the state.

    Args:
        state (dict): The current state containing the task to research

    Returns:
        dict: Updated state with the research data
    """
    logger.info("***Researcher conducting research...***")

    try:
        task = state.get("current_task")
        if not task:
            raise ValueError("No current task found in state")

        logger.info(f"Researching task: {task}")
        data = search(task)

        if not data:
            logger.warning("No research data found for the task")
            data = "No relevant information found."

        logger.info(f"Research completed with data: {data[:100]}...")  # Log first 100 chars of data
        return {"research": data}

    except Exception as e:
        logger.error(f"Error during research: {str(e)}")
        return {"research": f"An error occurred during research: {str(e)}"}
