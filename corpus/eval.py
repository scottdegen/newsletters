from __future__ import annotations

from collections import defaultdict

import numpy as np

from corpus.config import Config
from corpus.embed import Embedder
from corpus.retrieve import retrieve

QUERY_TAGS = {"entity", "paraphrase", "temporal", "multi-source"}


def _relevant_items_by_query(judgments: list[dict]) -> dict[str, set[int]]:
    relevant: dict[str, set[int]] = defaultdict(set)
    for j in judgments:
        if j["relevant"] is True:
            relevant[j["query_id"]].add(j["item_id"])
    return relevant


def _judged_query_ids(judgments: list[dict]) -> set[str]:
    """Queries with at least one filled-in (non-null) judgment."""
    return {j["query_id"] for j in judgments if j["relevant"] is not None}


def per_query_scores(
    queries: list[dict],
    judgments: list[dict],
    mode: str,
    k: int,
    conn,
    embedder: Embedder,
    config: Config,
) -> list[dict]:
    """Recall@k and reciprocal rank per query, restricted to queries that have
    been judged (relevant != null for at least one pooled item) and have at
    least one item marked relevant (otherwise recall is undefined, not zero).
    """
    relevant_by_query = _relevant_items_by_query(judgments)
    judged_ids = _judged_query_ids(judgments)

    scores = []
    for q in queries:
        if q["id"] not in judged_ids:
            continue
        relevant = relevant_by_query.get(q["id"], set())
        if not relevant:
            continue
        results = retrieve(q["text"], k, mode, conn=conn, embedder=embedder, config=config)
        retrieved_ids = [r.item_id for r in results]
        hit = set(retrieved_ids) & relevant
        recall = len(hit) / len(relevant)
        rr = 0.0
        for rank, item_id in enumerate(retrieved_ids, start=1):
            if item_id in relevant:
                rr = 1.0 / rank
                break
        scores.append({"query_id": q["id"], "tag": q["tag"], "recall_at_k": recall, "rr": rr})
    return scores


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, seed: int = 42
) -> tuple[float, float]:
    """95% bootstrap CI on the mean. Fixed seed — CLAUDE.md invariant 7."""
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "recall_at_k": None, "mrr": None, "recall_ci95": None}
    recalls = [r["recall_at_k"] for r in rows]
    rrs = [r["rr"] for r in rows]
    return {
        "n": len(rows),
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        "recall_ci95": bootstrap_ci(recalls),
    }


def summarize(scores: list[dict]) -> dict:
    """Overall + per-tag breakdown. The per-tag table is the real writeup
    material — an aggregate alone hides which mode fails where (PLAN.md).
    """
    by_tag = {tag: _aggregate([s for s in scores if s["tag"] == tag]) for tag in QUERY_TAGS}
    return {"overall": _aggregate(scores), "by_tag": by_tag}


def compare_modes(scores_a: list[dict], scores_b: list[dict]) -> dict:
    """Paired comparison of two modes' per-query Recall@k via Wilcoxon
    signed-rank. A few-point difference at n=40 is noise; this tells you
    whether it's signal. Requires overlapping query_ids (paired samples).
    """
    from scipy import stats

    a_by_id = {s["query_id"]: s["recall_at_k"] for s in scores_a}
    b_by_id = {s["query_id"]: s["recall_at_k"] for s in scores_b}
    common = sorted(set(a_by_id) & set(b_by_id))
    if len(common) < 2:
        return {"n": len(common), "mean_delta": None, "wilcoxon_p": None}

    a_vals = [a_by_id[q] for q in common]
    b_vals = [b_by_id[q] for q in common]
    mean_delta = sum(b - a for a, b in zip(a_vals, b_vals)) / len(common)

    if all(a == b for a, b in zip(a_vals, b_vals)):
        return {"n": len(common), "mean_delta": 0.0, "wilcoxon_p": 1.0}

    result = stats.wilcoxon(a_vals, b_vals)
    return {"n": len(common), "mean_delta": mean_delta, "wilcoxon_p": float(result.pvalue)}
