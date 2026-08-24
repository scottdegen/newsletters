from __future__ import annotations

from dataclasses import dataclass

# Word count is used as a fast, dependency-free proxy for token count so this
# module stays pure and testable with no model loaded (CLAUDE.md: chunk() gets
# fast unit tests with no model loaded). It's an approximation of the real
# EmbeddingGemma tokenizer's count, not an exact match.
TARGET_TOKENS = 512
OVERLAP_RATIO = 0.15


@dataclass(frozen=True)
class TextChunk:
    text: str
    token_count: int  # approximate; see module docstring


def chunk(
    text: str,
    target_tokens: int = TARGET_TOKENS,
    overlap_ratio: float = OVERLAP_RATIO,
) -> list[TextChunk]:
    """Pure: split one item's text into overlapping chunks.

    Never crosses an item boundary by construction — call once per item, with
    that item's own text only.
    """
    words = text.split()
    if not words:
        return []

    step = max(1, int(target_tokens * (1 - overlap_ratio)))
    chunks: list[TextChunk] = []
    start = 0
    while start < len(words):
        end = min(start + target_tokens, len(words))
        piece = words[start:end]
        chunks.append(TextChunk(text=" ".join(piece), token_count=len(piece)))
        if end == len(words):
            break
        start += step
    return chunks
