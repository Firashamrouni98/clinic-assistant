from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.config import (
    MILVUS_URI, COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIM
)

def load_docs():
    faq_path = os.path.join(os.path.dirname(__file__), "../data/clinic_faq.txt")
    with open(faq_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 30]
    return chunks

def embed_and_store():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Reading clinic documents...")
    chunks = load_docs()
    print(f"Found {len(chunks)} chunks to embed.")
    print("Connecting to Milvus...")
    client = MilvusClient(MILVUS_URI)
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print("Dropped existing collection.")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIM,
    )
    print("Embedding and inserting chunks...")
    embeddings = model.encode(chunks).tolist()
    data = [
        {"id": i, "vector": embeddings[i], "text": chunks[i]}
        for i in range(len(chunks))
    ]
    client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"Done! {len(chunks)} chunks stored in Milvus at {MILVUS_URI}")

if __name__ == "__main__":
    embed_and_store()