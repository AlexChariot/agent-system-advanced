# from duckduckgo_search import DDGS
from ddgs import DDGS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search(query, max_results=5):
    """
    Perform a web search using DuckDuckGo and return the results.

    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return (default: 5)

    Returns:
        str: Combined search results as a single string
    """
    logger.info(f"Searching for: {query}")

    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                if "body" in r and r["body"]:
                    results.append(r["body"])
                    logger.debug(f"Found result: {r['body'][:100]}...")  # Log first 100 chars of each result

        if not results:
            logger.warning("No results found for the query")
            return "No results found."

        logger.info(f"Found {len(results)} results")
        return "\n".join(results)

    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        return f"An error occurred during search: {str(e)}"
