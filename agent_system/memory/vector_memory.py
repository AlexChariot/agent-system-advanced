#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embeddings and vector store
try:
    embedding = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma(
        collection_name="long_term_memory",
        embedding_function=embedding,
    )
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {str(e)}")
    raise

def store_memory(text):
    """
    Stores text in the vector memory.

    Args:
        text (str): The text to store in memory

    Returns:
        bool: True if storage was successful, False otherwise
    """
    if not text or not isinstance(text, str):
        logger.error("Invalid text provided for storage")
        return False

    try:
        logger.info(f"Storing in memory: {text[:100]}...")  # Log first 100 chars
        vectorstore.add_texts([text])
        logger.info("Memory stored successfully")
        return True
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        return False

def recall_memory(query, k=5):
    """
    Recalls memories from the vector store based on a query.

    Args:
        query (str): The query to search for in memory
        k (int): Number of results to return (default: 5)

    Returns:
        str: Combined recalled memories as a single string
    """
    if not query or not isinstance(query, str):
        logger.error("Invalid query provided for recall")
        return "Invalid query provided."

    try:
        logger.info(f"Recalling from memory with query: {query}")
        docs = vectorstore.similarity_search(query, k=k)

        if not docs:
            logger.warning("No relevant memories found")
            return "No relevant memories found."

        recalled_text = "\n".join([d.page_content for d in docs])
        logger.info(f"Recalled {len(docs)} memories")
        return recalled_text
    except Exception as e:
        logger.error(f"Error recalling memory: {str(e)}")
        return f"An error occurred during recall: {str(e)}"
