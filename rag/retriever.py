from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.config import (
    MILVUS_URI, COLLECTION_NAME,
    EMBEDDING_MODEL, TOP_K_RESULTS
)

# Load once at import time — no need to reload on every query
_model  = None
_client = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def _get_client():
    global _client
    if _client is None:
        _client = MilvusClient(MILVUS_URI)
    return _client

def retrieve(query: str, top_k: int = TOP_K_RESULTS) -> str:
    """
    Takes a patient question, searches Milvus,
    returns the most relevant clinic knowledge as a string.
    """
    model  = _get_model()
    client = _get_client()

    # Embed the query
    query_vector = model.encode(query).tolist()
    
    # Load collection into memory before searching
    client.load_collection(COLLECTION_NAME)

    # Search Milvus
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=["text"],
    )

    # Extract and join the top chunks
    chunks = [hit["entity"]["text"] for hit in results[0]]

    if not chunks:
        return ""

    context = "\n\n---\n\n".join(chunks)
    return context


if __name__ == "__main__":
    # Quick test
    test_query = "Quels sont vos horaires?"
    print(f"Query: {test_query}\n")
    print("Retrieved context:")
    print(retrieve(test_query))