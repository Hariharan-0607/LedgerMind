"""ChromaDB persistence and embedding-function selection."""

from __future__ import annotations

import hashlib
import logging
import math
import re

from config import CHROMA_DIR, EMBEDDING_MODEL

log = logging.getLogger(__name__)

COLLECTION_NAME = "annual_reports"
_collection = None


class HashEmbeddingFunction:
    """Deterministic bag-of-words hashing embedder.

    Used for tests and as an offline fallback. It is lexical, not semantic —
    retrieval quality is lower than a real Sentence Transformers model.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "hash-bow"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def __call__(self, input):  # noqa: A002 - chromadb's protocol uses `input`
        return [self._embed(t) for t in input]

    def embed_documents(self, input):  # noqa: A002
        return self(input)

    def embed_query(self, input):  # noqa: A002
        return self(input)

    def is_legacy(self) -> bool:
        return False

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config):
        return HashEmbeddingFunction(**config)


def get_embedding_function():
    """Prefer Sentence Transformers; fall back to Chroma's ONNX MiniLM, then hashing."""
    if EMBEDDING_MODEL == "hash":
        return HashEmbeddingFunction()
    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    except Exception as exc:
        log.info("sentence-transformers unavailable (%s); trying Chroma default", type(exc).__name__)
    try:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return DefaultEmbeddingFunction()
    except Exception as exc:
        log.warning("Chroma default embedder unavailable (%s); using hash fallback", type(exc).__name__)
        return HashEmbeddingFunction()


def get_collection(path=CHROMA_DIR, embedding_function=None):
    global _collection
    if _collection is not None:
        return _collection
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function or get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def set_collection(collection) -> None:
    """Inject a collection (used by tests and evaluation)."""
    global _collection
    _collection = collection
