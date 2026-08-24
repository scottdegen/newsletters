from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """RRF: score(id) = sum, over lists containing it, of 1/(k + rank) with
    rank 1-indexed. Pure function — takes and returns plain structures, no I/O.

    Returns (id, score) sorted by score descending. Ties broken by id
    ascending — CLAUDE.md invariant 7 (deterministic retrieval; never leave
    ties to dict/insertion ordering, or identical inputs across two runs can
    silently produce different results).
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
