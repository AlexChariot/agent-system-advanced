# from duckduckgo_search import DDGS
from ddgs import DDGS

def search(query):
    print("Searching for:", query)

    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r["body"])
            

    print("Search results:", results)

    return "\n".join(results)