"""
Step 3: Vector store and retrieval.

A minimal in-memory store: hold chunk texts alongside their embedding
vectors, and find the chunks most relevant to a query. Three retrieval
strategies are available:

- Cosine similarity: plain nearest neighbours.
- MMR (maximal marginal relevance): trades some relevance for diversity, so
  the top-k aren't five near-duplicates of the same passage.
- Hybrid: blends dense cosine scores with BM25 keyword scores, which helps
  for names, numbers and rare terms that embeddings can blur.
"""

import math
import re
from collections import Counter

import numpy as np

from options import Param, Technique

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class VectorStore:
    def __init__(self):
        self.texts: list[str] = []
        self.vectors: np.ndarray | None = None  # (n, d)
        self._bm25: "BM25 | None" = None

    def add(self, texts: list[str], vectors: np.ndarray) -> None:
        """Add a batch of chunks and their corresponding embedding vectors."""
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.texts.extend(texts)
        self._bm25 = None

    def cosine_scores(self, query_vector: np.ndarray) -> np.ndarray:
        return _normalize(self.vectors) @ _normalize(query_vector)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """(text, score) pairs for the k most similar chunks, best first."""
        return [(self.texts[i], s) for i, s in search_cosine(self, query_vector, "", k)]

    @property
    def bm25(self) -> "BM25":
        if self._bm25 is None:
            self._bm25 = BM25(self.texts)
        return self._bm25


class BM25:
    """Okapi BM25 keyword scoring over the stored chunks."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(_tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if len(self.docs) else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        terms = [t for t in set(_tokenize(query)) if t in self.idf]
        for i, doc in enumerate(self.docs):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    out[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
        return out


def _top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    k = min(k, len(scores))
    idx = np.argsort(scores)[-k:][::-1]
    return [(int(i), float(scores[i])) for i in idx]


def search_cosine(store: VectorStore, query_vector, query_text, k=5):
    return _top_k(store.cosine_scores(query_vector), k)


def search_mmr(store: VectorStore, query_vector, query_text, k=5, diversity=0.3, candidates=25):
    """
    Greedily pick chunks that are relevant to the query but dissimilar to
    the ones already picked: score = (1-diversity)*relevance - diversity*redundancy.
    """
    relevance = store.cosine_scores(query_vector)
    pool = [i for i, _ in _top_k(relevance, max(candidates, k))]
    vecs = _normalize(store.vectors)

    selected: list[int] = []
    while pool and len(selected) < k:
        if selected:
            redundancy = (vecs[pool] @ vecs[selected].T).max(axis=1)
        else:
            redundancy = np.zeros(len(pool))
        mmr = (1 - diversity) * relevance[pool] - diversity * redundancy
        selected.append(pool.pop(int(np.argmax(mmr))))
    return [(i, float(relevance[i])) for i in selected]


def search_hybrid(store: VectorStore, query_vector, query_text, k=5, keyword_weight=0.4):
    """Min-max normalize dense and BM25 scores, then take a weighted sum."""

    def minmax(x: np.ndarray) -> np.ndarray:
        span = x.max() - x.min()
        return (x - x.min()) / span if span > 0 else np.zeros_like(x)

    dense = minmax(store.cosine_scores(query_vector))
    sparse = minmax(store.bm25.scores(query_text))
    return _top_k((1 - keyword_weight) * dense + keyword_weight * sparse, k)


RETRIEVERS: dict[str, Technique] = {
    t.name: t
    for t in [
        Technique(
            "Cosine similarity",
            "Plain nearest-neighbour search on the embeddings.",
            search_cosine,
            [Param("k", "Chunks to retrieve (top-k)", 5, 1, 20, 1)],
        ),
        Technique(
            "MMR (diverse)",
            "Maximal Marginal Relevance: avoids returning several near-duplicate chunks.",
            search_mmr,
            [
                Param("k", "Chunks to retrieve (top-k)", 5, 1, 20, 1),
                Param("diversity", "Diversity (0 = pure relevance)", 0.3, 0.0, 1.0, 0.05),
                Param("candidates", "Candidate pool size", 25, 5, 100, 5),
            ],
        ),
        Technique(
            "Hybrid (dense + BM25)",
            "Blends semantic similarity with keyword matching. Strong for names, numbers and jargon.",
            search_hybrid,
            [
                Param("k", "Chunks to retrieve (top-k)", 5, 1, 20, 1),
                Param("keyword_weight", "Keyword (BM25) weight", 0.4, 0.0, 1.0, 0.05),
            ],
        ),
    ]
}
