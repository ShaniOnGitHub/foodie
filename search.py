"""
search.py
---------
Standalone CLI semantic search utility for Foodie recommendation system.
Allows querying the ChromaDB food collection directly from the command line.
"""
import sys
import os
from typing import List, Dict, Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ingest import get_collection


def search_food(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Query the foods collection in ChromaDB for semantic matches.
    Returns a list of result dictionaries containing metadata and distance scores.
    """
    if not query_text.strip():
        return []

    collection = get_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )

    items = []
    if results and "metadatas" in results and results["metadatas"]:
        metas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]
        for idx, meta in enumerate(metas):
            dist = distances[idx] if idx < len(distances) else None
            items.append({
                "name": meta.get("name"),
                "cuisine": meta.get("cuisine"),
                "calories": meta.get("calories"),
                "dietary_type": meta.get("dietary_type"),
                "description": meta.get("description"),
                "distance": dist
            })
    return items


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "healthy vegan salad"
    print(f"\nSearching food items for: '{query}'...")
    results = search_food(query, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['name']} ({res['cuisine']}, {res['dietary_type']}) - {res['calories']} cal")
        print(f"   Description: {res['description']}\n")
