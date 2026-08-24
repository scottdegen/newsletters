import dataclasses
import sqlite3

import sqlite_vec

from corpus import db, eval as eval_mod
from corpus.config import load_config
from corpus.embed import HashEmbedder
from corpus.retrieve import retrieve

# Synthetic fixtures throughout — hand-authored constants, not derived from
# real search output. This tests the statistics arithmetic, not retrieval
# quality; PLAN.md's "never construct ground truth from search output" rule
# is about the real evals/judgments.yaml, not ordinary unit-test fixtures.


def _seeded_conn_and_config(dim: int = 16):
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(db.SCHEMA)
    db.ensure_chunk_vec(conn, dim)

    embedder = HashEmbedder(dim=dim)
    source_id = db.upsert_source(
        conn, "test-src", "Test Source", "https://x.test/feed", None, "wordpress"
    )
    item_ids = {}
    for key, text in [("a", "apple banana cherry"), ("b", "durian elderberry fig")]:
        url = f"https://x.test/{key}"
        issue_id, _ = db.insert_issue(
            conn, source_id, url, key, "2026-06-01T00:00:00Z",
            "raw/x.html", "hash", "2026-06-01T00:00:00Z",
        )
        item_id = db.insert_item(conn, issue_id, 0, key, text, url)
        chunk_id = db.insert_chunk(conn, item_id, 0, text, len(text.split()))
        vec = embedder.encode([text], kind="doc")[0]
        db.insert_chunk_vec(
            conn, chunk_id, "2026-06", vec.astype("float32").tobytes(),
            source_id, "2026-06-01T00:00:00Z",
        )
        item_ids[key] = item_id
    conn.commit()

    config = dataclasses.replace(load_config("config.toml"), embedding_dim=dim)
    return conn, config, embedder, item_ids


def test_recall_is_1_when_k_covers_every_item():
    """With k >= total items, every item is returned, so any judged-relevant
    item is necessarily included — true regardless of ranking order.
    """
    conn, config, embedder, item_ids = _seeded_conn_and_config()
    queries = [{"id": "q01", "tag": "entity", "text": "apple banana cherry"}]
    judgments = [{"query_id": "q01", "item_id": item_ids["a"], "relevant": True}]

    scores = eval_mod.per_query_scores(
        queries, judgments, mode="dense", k=2, conn=conn, embedder=embedder, config=config
    )
    assert len(scores) == 1
    assert scores[0]["recall_at_k"] == 1.0


def test_per_query_scores_matches_hand_computed_recall_and_rr():
    conn, config, embedder, item_ids = _seeded_conn_and_config()
    queries = [{"id": "q01", "tag": "entity", "text": "apple banana cherry"}]
    judgments = [{"query_id": "q01", "item_id": item_ids["a"], "relevant": True}]

    results = retrieve("apple banana cherry", k=1, mode="dense", conn=conn, embedder=embedder, config=config)
    expected_recall = 1.0 if results[0].item_id == item_ids["a"] else 0.0
    expected_rr = 1.0 if results[0].item_id == item_ids["a"] else 0.0

    scores = eval_mod.per_query_scores(
        queries, judgments, mode="dense", k=1, conn=conn, embedder=embedder, config=config
    )
    assert scores[0]["recall_at_k"] == expected_recall
    assert scores[0]["rr"] == expected_rr


def test_unjudged_queries_are_excluded():
    conn, config, embedder, item_ids = _seeded_conn_and_config()
    queries = [
        {"id": "q01", "tag": "entity", "text": "apple banana cherry"},
        {"id": "q02", "tag": "entity", "text": "durian elderberry fig"},
    ]
    judgments = [{"query_id": "q01", "item_id": item_ids["a"], "relevant": True}]

    scores = eval_mod.per_query_scores(
        queries, judgments, mode="dense", k=2, conn=conn, embedder=embedder, config=config
    )
    assert {s["query_id"] for s in scores} == {"q01"}


def test_judged_but_nothing_relevant_is_excluded():
    conn, config, embedder, item_ids = _seeded_conn_and_config()
    queries = [{"id": "q01", "tag": "entity", "text": "apple banana cherry"}]
    # Judged (relevant is non-null) but explicitly false -> no relevant items.
    judgments = [{"query_id": "q01", "item_id": item_ids["a"], "relevant": False}]

    scores = eval_mod.per_query_scores(
        queries, judgments, mode="dense", k=2, conn=conn, embedder=embedder, config=config
    )
    assert scores == []


def test_summarize_aggregates_overall_and_by_tag():
    scores = [
        {"query_id": "q1", "tag": "entity", "recall_at_k": 1.0, "rr": 1.0},
        {"query_id": "q2", "tag": "entity", "recall_at_k": 0.0, "rr": 0.0},
        {"query_id": "q3", "tag": "paraphrase", "recall_at_k": 0.5, "rr": 0.5},
    ]
    summary = eval_mod.summarize(scores)
    assert summary["overall"]["n"] == 3
    assert summary["overall"]["recall_at_k"] == 0.5
    assert summary["by_tag"]["entity"]["n"] == 2
    assert summary["by_tag"]["entity"]["recall_at_k"] == 0.5
    assert summary["by_tag"]["paraphrase"]["n"] == 1
    assert summary["by_tag"]["temporal"]["n"] == 0
    assert summary["by_tag"]["temporal"]["recall_at_k"] is None


def test_bootstrap_ci_is_deterministic_and_contains_mean():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 0.5]
    lo1, hi1 = eval_mod.bootstrap_ci(values)
    lo2, hi2 = eval_mod.bootstrap_ci(values)
    assert (lo1, hi1) == (lo2, hi2)
    mean = sum(values) / len(values)
    assert lo1 <= mean <= hi1


def test_compare_modes_zero_delta_when_identical():
    scores = [
        {"query_id": "q1", "recall_at_k": 1.0},
        {"query_id": "q2", "recall_at_k": 0.5},
    ]
    result = eval_mod.compare_modes(scores, scores)
    assert result["mean_delta"] == 0.0
    assert result["wilcoxon_p"] == 1.0


def test_compare_modes_detects_consistent_improvement():
    scores_a = [{"query_id": f"q{i}", "recall_at_k": 0.0} for i in range(10)]
    scores_b = [{"query_id": f"q{i}", "recall_at_k": 1.0} for i in range(10)]
    result = eval_mod.compare_modes(scores_a, scores_b)
    assert result["mean_delta"] == 1.0
    assert result["wilcoxon_p"] < 0.05


def test_compare_modes_handles_too_few_paired_queries():
    result = eval_mod.compare_modes(
        [{"query_id": "q1", "recall_at_k": 1.0}], [{"query_id": "q1", "recall_at_k": 0.5}]
    )
    assert result["n"] == 1
    assert result["wilcoxon_p"] is None
