from __future__ import annotations

import hashlib
from typing import Protocol

from corpus.config import Config


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...
    # Higher score = more relevant, the cross-encoder convention -- the
    # opposite of Chunk.distance, where lower is better everywhere else.
    # retrieve() negates when converting these into Chunk.distance so "lower
    # is always better" still holds uniformly across every mode.


class CrossEncoderReranker:
    """Wraps a local sentence-transformers CrossEncoder (PLAN.md mode 4:
    "bge-reranker-v2-m3 or similar, local, ~100ms, $0"). Using
    ms-marco-MiniLM-L-6-v2 instead of bge-reranker-v2-m3 itself: it's the
    canonical, well-tested checkpoint for sentence-transformers' CrossEncoder
    class, an order of magnitude smaller, and this machine already hit a real
    MPS OOM on the much larger 300M embedding model during Phase 2 indexing
    -- no reason to risk that again for a reranker that only needs to beat
    hybrid's ranking, not be state-of-the-art.
    """

    def __init__(self, config: Config) -> None:
        # Imported lazily -- same reasoning as GemmaEmbedder: keep this cost
        # out of every test that imports corpus.retrieve but never actually
        # reranks.
        from sentence_transformers import CrossEncoder

        # Forced CPU to match GemmaEmbedder's fix for the same MPS OOM class
        # of failure; a cross-encoder over ~50 candidates at query time has
        # no throughput need for a GPU anyway.
        self._model = CrossEncoder(config.reranking_model_name, device="cpu")

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, t) for t in texts]
        return self._model.predict(pairs).tolist()


class HashReranker:
    """Deterministic, model-free stand-in for tests -- same role as
    HashEmbedder. The score is a stable hash of (query, text), not a real
    relevance signal; it exists so mode="rerank" can be exercised in the fast
    suite without loading a model.
    """

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores = []
        for text in texts:
            digest = hashlib.sha256(f"{query}||{text}".encode("utf-8")).digest()
            scores.append(int.from_bytes(digest[:8], "big") / 2**64)
        return scores
