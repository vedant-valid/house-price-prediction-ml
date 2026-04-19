import os
import numpy as np
from sentence_transformers import SentenceTransformer

MARKET_DATA_DIR = "market_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model = None
_docs = None
_embeddings = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _load_docs() -> list[str]:
    global _docs, _embeddings
    if _docs is not None:
        return _docs

    _docs = []
    for fname in sorted(os.listdir(MARKET_DATA_DIR)):
        if fname.endswith(".txt"):
            path = os.path.join(MARKET_DATA_DIR, fname)
            with open(path) as f:
                text = f.read().strip()
            # split into ~paragraph chunks
            chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 30]
            _docs.extend(chunks)

    model = _get_model()
    _embeddings = model.encode(_docs, normalize_embeddings=True)
    return _docs


def search(query: str, k: int = 4) -> tuple[list, float]:
    docs = _load_docs()
    model = _get_model()

    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(_embeddings, q_emb)

    top_idx = np.argsort(scores)[::-1][:k]
    top_docs = [docs[i] for i in top_idx]
    top_scores = [float(scores[i]) for i in top_idx]

    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    return top_docs, avg_score
