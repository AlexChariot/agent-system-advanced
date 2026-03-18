from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import logging

logger = logging.getLogger(__name__)

# Lazy singletons — initialized on first use, not at import time
_embedding = None
_vectorstore = None


def _get_vectorstore() -> Chroma:
    """
    Returns the vectorstore, initializing it on the first call.

    Lazy initialization prevents a crash at import time if Ollama is not
    running yet.

    Returns:
        Chroma: The vectorstore instance.

    Raises:
        RuntimeError: If Ollama is unreachable or initialization fails.
    """
    global _embedding, _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    try:
        logger.info("[VectorMemory] Initializing vectorstore...")
        _embedding = OllamaEmbeddings(model="llama3.1")
        _vectorstore = Chroma(
            collection_name="long_term_memory",
            embedding_function=_embedding,
        )
        logger.info("[VectorMemory] Vectorstore initialized successfully.")
    except Exception as e:
        logger.error(f"[VectorMemory] Initialization failed: {e}")
        raise RuntimeError(
            f"Could not initialize the vectorstore. "
            f"Make sure Ollama is running (`ollama serve`). Detail: {e}"
        ) from e

    return _vectorstore


def store_memory(text: str) -> bool:
    """
    Stores a text in the vector memory.

    Args:
        text (str): The text to store.

    Returns:
        bool: True if storage succeeded, False otherwise.
    """
    if not text or not isinstance(text, str):
        logger.error("[VectorMemory] Invalid text provided for storage.")
        return False

    try:
        vs = _get_vectorstore()
        logger.info(f"[VectorMemory] Storing: {text[:100]}...")
        vs.add_texts([text])
        logger.info("[VectorMemory] Memory stored successfully.")
        return True
    except Exception as e:
        logger.error(f"[VectorMemory] Error during storage: {e}")
        return False


def recall_memory(query: str, k: int = 5) -> str:
    """
    Retrieves memories from the vectorstore based on a query.

    Args:
        query (str): The search query.
        k (int): Number of results to return (default: 5).

    Returns:
        str: Retrieved memories concatenated as a single string.
    """
    if not query or not isinstance(query, str):
        logger.error("[VectorMemory] Invalid query provided for recall.")
        return "Invalid query."

    try:
        vs = _get_vectorstore()
        logger.info(f"[VectorMemory] Recalling memories for: {query[:80]}...")
        docs = vs.similarity_search(query, k=k)

        if not docs:
            logger.warning("[VectorMemory] No relevant memories found.")
            return "No relevant memories found."

        recalled_text = "\n".join([d.page_content for d in docs])
        logger.info(f"[VectorMemory] {len(docs)} memory/memories recalled.")
        return recalled_text

    except Exception as e:
        logger.error(f"[VectorMemory] Error during recall: {e}")
        return f"Error during memory recall: {e}"