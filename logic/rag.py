import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MARKET_DATA_DIR = "data/market_data"

_vectorizer = None
_matrix = None
_docs = None


def _load():
    global _vectorizer, _matrix, _docs
    if _docs is not None:
        return

    _docs = []
    for fname in sorted(os.listdir(MARKET_DATA_DIR)):
        if fname.endswith(".txt"):
            path = os.path.join(MARKET_DATA_DIR, fname)
            with open(path) as f:
                text = f.read().strip()
            chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 30]
            _docs.extend(chunks)

    _vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    _matrix = _vectorizer.fit_transform(_docs)


def search(query: str, k: int = 4) -> tuple[list, float]:
    _load()
    q_vec = _vectorizer.transform([query])
    scores = cosine_similarity(q_vec, _matrix)[0]

    top_idx = np.argsort(scores)[::-1][:k]
    top_docs = [_docs[i] for i in top_idx]
    top_scores = [float(scores[i]) for i in top_idx]

    avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    return top_docs, avg_score
