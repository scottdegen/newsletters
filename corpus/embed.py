from __future__ import annotations

import hashlib
import os
from typing import Literal, Protocol

import numpy as np

from corpus.config import Config


class Embedder(Protocol):
    def encode(self, texts: list[str], kind: Literal["query", "doc"]) -> np.ndarray: ...

# Read directly from SentenceTransformer("google/embeddinggemma-300m").prompts
# on 2026-08-17 — not transcribed from memory or docs. If the model version
# ever changes, re-verify against model.prompts rather than editing these by
# hand; a wrong prefix degrades retrieval silently (CLAUDE.md invariant 1).
# These constants are the only place task prefixes may appear anywhere in
# this codebase.
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "


class GemmaEmbedder:
    """Wraps EmbeddingGemma-300M. Native output is 768-dim; truncated to
    config.embedding_dim via Matryoshka (a prefix of the vector is itself a
    valid, if lower-fidelity, embedding for this model family) and
    re-normalized, since a truncated unit vector is no longer unit-length.
    """

    def __init__(self, config: Config) -> None:
        # Imported lazily: torch + sentence_transformers cost real seconds
        # just to import, even before doing anything. corpus.embed is
        # imported by modules (versioning.py, retrieve.py) that fast tests
        # touch constantly without ever needing the real model — keeping this
        # import inside __init__ means only tests that actually construct a
        # GemmaEmbedder (marked @pytest.mark.slow) pay that cost.
        import torch
        from sentence_transformers import SentenceTransformer

        # Forced CPU: sentence-transformers defaults to Apple's MPS backend on
        # this machine, which hit repeated kIOGPUCommandBufferCallbackError
        # OutOfMemory errors mid-batch — ~15 min/batch of 32 chunks (a full
        # index would have taken 15+ hours). CPU inference on a 300M model is
        # reliable and, on this hardware, actually the faster path.
        cpu_count = os.cpu_count()
        if cpu_count:
            torch.set_num_threads(cpu_count)  # default under-uses cores (4 of 8 seen)
        self._model = SentenceTransformer(config.embedding_model_name, device="cpu")
        self._dim = config.embedding_dim
        self._batch_size = config.embedding_batch_size

    def encode(
        self, texts: list[str], kind: Literal["query", "doc"], show_progress: bool = False
    ) -> np.ndarray:
        prefix = QUERY_PREFIX if kind == "query" else DOCUMENT_PREFIX
        prefixed = [prefix + t for t in texts]
        embeddings = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        truncated = embeddings[:, : self._dim]
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        return truncated / norms


class HashEmbedder:
    """Deterministic vectors from a text hash — no model, no I/O. Lets the
    full retrieval pipeline (filters, DB round-trips, future RRF/fusion) run
    in well under a second with nothing loaded in memory. Real embeddings
    only in tests marked @pytest.mark.slow; everything else uses this.

    `kind` is folded into the hash so query/doc embeddings differ for the
    same text, mirroring (structurally, not semantically) GemmaEmbedder's
    prefix behavior — enough for tests that check the two are treated
    differently without needing the real model to prove it.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def encode(self, texts: list[str], kind: Literal["query", "doc"]) -> np.ndarray:
        vectors = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(f"{kind}:{text}".encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            vectors[i] = np.random.default_rng(seed).standard_normal(self._dim)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
