"""
ingest.py
---------
ChromaDB client setup and data ingestion for the Food Recommender.

Responsibilities:
  - Create (or reuse) the ChromaDB collection with cosine space + mxbai EF
  - Convert each food item dict into a natural-language string before embedding
  - upsert() all 30 items (idempotent — safe to call on every app startup)
  - Gate: skip ingestion entirely if collection already has 30 documents
"""

import json
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(_HERE, "food_data.json")
CHROMA_PATH = os.path.join(_HERE, "chroma_data")

COLLECTION_NAME = "foods"
EMBED_MODEL     = "mixedbread-ai/mxbai-embed-large-v1"
EXPECTED_COUNT  = 30


# ---------------------------------------------------------------------------
# Embedding function (module-level so it is created once)
# ---------------------------------------------------------------------------
def _get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


# ---------------------------------------------------------------------------
# ChromaDB client + collection — cached at module level
# ---------------------------------------------------------------------------
def get_collection() -> chromadb.Collection:
    """
    Return the foods collection, creating it if it does not yet exist.
    The EF is bound to the collection at creation time — every subsequent
    add/query/update uses it automatically without being passed again.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = _get_embedding_function()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Text conversion — dict -> natural-language string
# ---------------------------------------------------------------------------
def record_to_text(item: dict) -> str:
    """
    Convert a food item dict into one embeddable prose sentence.

    We do NOT embed raw JSON key-value pairs. Instead we construct a sentence
    so the embedding model has semantic context for every field. Numbers like
    calories are written as prose ("450 calories") so the model understands
    their meaning rather than treating them as bare tokens.
    """
    return (
        f"{item['name']} is a {item['dietary_type']} {item['cuisine']} dish "
        f"with {item['calories']} calories, cooked by {item['cooking_method']}. "
        f"Taste profile: {item['taste_profile']}. "
        f"{item['description']}"
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest(collection: chromadb.Collection) -> None:
    """
    Load food_data.json and upsert all 30 items into ChromaDB.

    Gate: if the collection already holds EXPECTED_COUNT documents, skip
    ingestion entirely. This means the first app startup embeds everything
    (slow, one-time) and every subsequent restart is instant.

    Uses upsert() — not add() — so re-running never raises DuplicateIDError.

    Metadata field `calories` is stored as int so ChromaDB's $gte/$lte
    numeric operators work correctly in Tab 2's calorie filter.
    """
    current_count = collection.count()
    if current_count >= EXPECTED_COUNT:
        print(f"[ingest] Collection already has {current_count} docs. Skipping.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        foods = json.load(f)

    texts     = [record_to_text(item) for item in foods]
    ids       = [item["id"] for item in foods]
    metadatas = [
        {
            "name":           item["name"],
            "cuisine":        item["cuisine"],
            "calories":       int(item["calories"]),   # must be int for $gte/$lte
            "dietary_type":   item["dietary_type"],
            "cooking_method": item["cooking_method"],
            "taste_profile":  item["taste_profile"],
            "description":    item["description"],
        }
        for item in foods
    ]

    collection.upsert(documents=texts, ids=ids, metadatas=metadatas)
    print(f"[ingest] Upserted {len(ids)} food items. "
          f"Collection now has {collection.count()} docs.")


# ---------------------------------------------------------------------------
# CLI entry point — run directly to pre-warm the collection
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    col = get_collection()
    ingest(col)
    print(f"[ingest] Done. Total docs in collection: {col.count()}")
