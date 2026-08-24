from __future__ import annotations

import random
from datetime import datetime, timezone

from corpus import db
from corpus.config import Config
from corpus.embed import Embedder
from corpus.rerank import Reranker
from corpus.retrieve import retrieve
from corpus.versioning import compute_index_version

# Fixed per CLAUDE.md invariant 7 (deterministic retrieval/eval) — re-running
# `corpus pool` on unchanged data reproduces the exact same shuffle.
POOL_SHUFFLE_SEED = 42

# dense (Phase 2.5); sparse, hybrid, rerank, parent, temporal (Phase 3 —
# rerank/parent/temporal can surface items the first three never ranked into
# their own top-k, so they need to be in the pool too, or ground truth would
# quietly exclude exactly the candidates those modes exist to test).
WIRED_MODES = ["dense", "sparse", "hybrid", "rerank", "parent", "temporal"]


def build_pool(
    config: Config,
    queries: list[dict],
    k: int = 10,
    modes: list[str] | None = None,
    conn=None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> dict:
    """For each query: run every wired mode, union top-k by item (deduped —
    a judge rates an article, not a text fragment), shuffle per-query with a
    fixed seed so mode-of-origin isn't order-visible, return the judgment
    sheet structure. `relevant` is always emitted as null — filling it in is
    a human's job, never fabricated here (PLAN.md: never construct ground
    truth from search output).
    """
    modes = modes or WIRED_MODES
    conn = conn or db.get_connection(config.db_path)
    embedder = embedder or _default_embedder(config)
    reranker = reranker or (_default_reranker(config) if "rerank" in modes else None)
    index_version = compute_index_version(config)

    entries = []
    for q in queries:
        seen_items: dict[int, dict] = {}
        for mode in modes:
            results = retrieve(
                q["text"], k, mode, conn=conn, embedder=embedder, config=config, reranker=reranker
            )
            for r in results:
                if r.item_id not in seen_items:
                    seen_items[r.item_id] = {
                        "query_id": q["id"],
                        "query_text": q["text"],
                        "query_tag": q["tag"],
                        "item_id": r.item_id,
                        "headline": r.headline,
                        "canonical_url": r.canonical_url,
                        "source_slug": r.source_slug,
                        "snippet": r.text[:200],
                        "modes": [mode],
                        "relevant": None,
                    }
                else:
                    seen_items[r.item_id]["modes"].append(mode)

        pooled = list(seen_items.values())
        # Seeded per-query (not globally) so adding/removing a later query
        # never reshuffles earlier ones' judgment order.
        rng = random.Random(f"{POOL_SHUFFLE_SEED}:{q['id']}")
        rng.shuffle(pooled)
        entries.extend(pooled)

    return {
        "index_version": index_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judgments": entries,
    }


def merge_judgments(new_sheet: dict, old_sheet: dict | None) -> dict:
    """Carry forward existing relevance judgments onto a freshly rebuilt pool
    (e.g. after wiring a new mode), matched by (query_id, item_id). New
    candidates a prior pool never surfaced stay unjudged, same as always —
    this only preserves work already done, never invents anything.
    """
    if old_sheet is None:
        return new_sheet
    old_by_key = {
        (j["query_id"], j["item_id"]): j["relevant"]
        for j in old_sheet["judgments"]
        if j["relevant"] is not None
    }
    for j in new_sheet["judgments"]:
        key = (j["query_id"], j["item_id"])
        if key in old_by_key:
            j["relevant"] = old_by_key[key]
    return new_sheet


def _default_embedder(config: Config) -> Embedder:
    from corpus.embed import GemmaEmbedder

    return GemmaEmbedder(config)


def _default_reranker(config: Config) -> Reranker:
    from corpus.rerank import CrossEncoderReranker

    return CrossEncoderReranker(config)
