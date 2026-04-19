import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "market_insights"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def search(query: str, k: int = 4) -> tuple[list, float]:
    model = _get_model()
    embedding = model.encode([query])[0].tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "distances"],
    )

    docs = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    if not docs:
        return [], 0.0

    scores = [1.0 / (1.0 + d) for d in distances]
    return docs, sum(scores) / len(scores)
