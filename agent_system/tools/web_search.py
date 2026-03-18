from ddgs import DDGS
import logging

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search using DuckDuckGo and return the results.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return (default: 5).

    Returns:
        str: Combined search results as a single string.
    """
    logger.info(f"[WebSearch] Searching for: {query}")

    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                if "body" in r and r["body"]:
                    results.append(r["body"])
                    logger.debug(f"[WebSearch] Result snippet: {r['body'][:100]}...")

        if not results:
            logger.warning("[WebSearch] No results found for the query.")
            return "No results found."

        logger.info(f"[WebSearch] Found {len(results)} result(s).")
        return "\n".join(results)

    except Exception as e:
        logger.error(f"[WebSearch] Error during search: {e}")
        return f"An error occurred during search: {e}"