"""
Step 2: Embeddings.

Turn text into vectors so that semantic similarity can later be measured as
vector similarity. Three interchangeable embedders are available:

- Voyage AI (remote, high quality, needs VOYAGE_API_KEY)
- Ollama (local neural embedding model, e.g. nomic-embed-text)
- TF-IDF (local, lexical, no model at all -- a useful baseline)

All of them expose the same interface: `embed_documents` for the chunks being
indexed and `embed_query` for search queries.
"""

import os
from typing import Callable

import numpy as np
from dotenv import load_dotenv

from options import Param, Technique

load_dotenv()

Progress = Callable[[int, int], None] | None


class Embedder:
    name = "base"

    def embed_documents(self, texts: list[str], on_progress: Progress = None) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


def _batched(texts: list[str], size: int):
    for i in range(0, len(texts), size):
        yield i, texts[i : i + size]


class VoyageEmbedder(Embedder):
    """
    Documents and queries are embedded with different `input_type`s: Voyage's
    models are trained asymmetrically, so using the right one on each side
    improves retrieval quality.
    """

    def __init__(self, model: str = "voyage-3", batch_size: int = 64):
        import voyageai

        key = os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError("VOYAGE_API_KEY is not set (add it to .env).")
        self.client = voyageai.Client(api_key=key)
        self.model = model
        self.batch_size = batch_size
        self.name = f"Voyage {model}"

    def embed_documents(self, texts, on_progress=None):
        vectors = []
        for i, batch in _batched(texts, self.batch_size):
            result = self.client.embed(batch, model=self.model, input_type="document")
            vectors.extend(result.embeddings)
            if on_progress:
                on_progress(i + len(batch), len(texts))
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text):
        result = self.client.embed([text], model=self.model, input_type="query")
        return np.array(result.embeddings[0], dtype=np.float32)


class OllamaEmbedder(Embedder):
    def __init__(self, model: str = "nomic-embed-text", batch_size: int = 32):
        import ollama

        self.client = ollama
        self.model = model
        self.batch_size = batch_size
        self.name = f"Ollama {model}"

    def embed_documents(self, texts, on_progress=None):
        vectors = []
        for i, batch in _batched(texts, self.batch_size):
            vectors.extend(self.client.embed(model=self.model, input=batch)["embeddings"])
            if on_progress:
                on_progress(i + len(batch), len(texts))
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text):
        return np.array(self.client.embed(model=self.model, input=text)["embeddings"][0], dtype=np.float32)


class TfidfEmbedder(Embedder):
    """
    Sparse lexical vectors: a word scores high in a chunk if it is frequent
    there but rare across the document. Matches exact terms well, has no
    notion of synonyms. The vocabulary is fitted on the indexed chunks.
    """

    def __init__(self, max_features: int = 4096, ngram_max: int = 2):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, ngram_max), stop_words="english"
        )
        self.name = f"TF-IDF ({max_features} features)"

    def embed_documents(self, texts, on_progress=None):
        matrix = self.vectorizer.fit_transform(texts)
        if on_progress:
            on_progress(len(texts), len(texts))
        return matrix.toarray().astype(np.float32)

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0].astype(np.float32)


def ollama_models() -> list[str]:
    """Names of locally installed Ollama models ([] if Ollama isn't running)."""
    try:
        import ollama

        return sorted(m.model for m in ollama.list().models)
    except Exception:
        return []


def ollama_embedding_models() -> list[str]:
    hints = ("embed", "bge", "minilm", "e5", "gte", "arctic")
    return [m for m in ollama_models() if any(h in m.lower() for h in hints)]


# Embedding models the UI offers to download when none are installed.
PULLABLE_EMBEDDERS = {
    "nomic-embed-text": "~274 MB, strong general-purpose default",
    "all-minilm": "~46 MB, smallest and fastest, lower quality",
    "mxbai-embed-large": "~670 MB, highest quality of the three",
}


def pull_ollama_model(model: str):
    """Download an Ollama model, yielding progress updates (status/completed/total)."""
    import ollama

    yield from ollama.pull(model, stream=True)


EMBEDDERS: dict[str, Technique] = {
    t.name: t
    for t in [
        Technique(
            "Ollama (local)",
            "Neural embeddings from a local Ollama model. Private and free. "
            "Install one with `ollama pull nomic-embed-text`.",
            OllamaEmbedder,
            [Param("model", "Embedding model", "nomic-embed-text", choices=[])],  # filled by UI
        ),
        Technique(
            "Voyage AI (remote)",
            "High-quality hosted embeddings with asymmetric query/document modes. Needs VOYAGE_API_KEY.",
            VoyageEmbedder,
            [Param("model", "Model", "voyage-3", choices=["voyage-3", "voyage-3-lite", "voyage-3-large"])],
        ),
        Technique(
            "TF-IDF (local, lexical)",
            "Keyword-weighted vectors with no model at all. Instant, and a good baseline to compare against.",
            TfidfEmbedder,
            [
                Param("max_features", "Vocabulary size", 4096, 512, 20000, 512),
                Param("ngram_max", "Max n-gram length", 2, 1, 3, 1),
            ],
        ),
    ]
}


# Kept for the CLI (`main.py`) and older callers.
_default: Embedder | None = None


def _get_default() -> Embedder:
    global _default
    if _default is None:
        _default = VoyageEmbedder()
    return _default


def embed_documents(texts: list[str]) -> np.ndarray:
    return _get_default().embed_documents(texts)


def embed_query(text: str) -> np.ndarray:
    return _get_default().embed_query(text)


if __name__ == "__main__":
    emb = TfidfEmbedder()
    doc_vecs = emb.embed_documents(["The sky is blue.", "Cats are mammals."])
    query_vec = emb.embed_query("What color is the sky?")
    print("doc embeddings shape:", doc_vecs.shape)
    print("query embedding shape:", query_vec.shape)
