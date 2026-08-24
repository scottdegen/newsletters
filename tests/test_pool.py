import dataclasses
import sqlite3

import sqlite_vec

from corpus import db, pool
from corpus.config import load_config
from corpus.embed import HashEmbedder
from corpus.rerank import HashReranker


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

    def add(url, headline, text):
        issue_id, _ = db.insert_issue(
            conn, source_id, url, headline, "2026-06-01T00:00:00Z",
            "raw/x.html", "hash", "2026-06-01T00:00:00Z",
        )
        item_id = db.insert_item(conn, issue_id, 0, headline, text, url)
        chunk_id = db.insert_chunk(conn, item_id, 0, text, len(text.split()))
        vec = embedder.encode([text], kind="doc")[0]
        db.insert_chunk_vec(
            conn, chunk_id, "2026-06", vec.astype("float32").tobytes(),
            source_id, "2026-06-01T00:00:00Z",
        )

    add("https://x.test/a", "Nuclear talks", "nuclear negotiations vienna")
    add("https://x.test/b", "Basketball recap", "basketball championship game")
    add("https://x.test/c", "Election polling", "midterm election polling forecast")
    conn.commit()

    config = dataclasses.replace(load_config("config.toml"), embedding_dim=dim)
    return conn, config, embedder


def test_build_pool_marks_all_judgments_unjudged():
    conn, config, embedder = _seeded_conn_and_config()
    queries = [{"id": "q01", "tag": "entity", "text": "nuclear talks"}]

    sheet = pool.build_pool(config, queries, k=3, conn=conn, embedder=embedder, reranker=HashReranker())
    assert all(j["relevant"] is None for j in sheet["judgments"])


def test_build_pool_dedupes_by_item_across_modes():
    conn, config, embedder = _seeded_conn_and_config()
    queries = [{"id": "q01", "tag": "entity", "text": "nuclear talks"}]

    sheet = pool.build_pool(
        config, queries, k=3, conn=conn, embedder=embedder, modes=["dense", "dense"]
    )
    item_ids = [j["item_id"] for j in sheet["judgments"]]
    assert len(item_ids) == len(set(item_ids))
    assert sheet["judgments"][0]["modes"] == ["dense", "dense"]


def test_build_pool_shuffle_is_deterministic():
    conn, config, embedder = _seeded_conn_and_config()
    queries = [{"id": "q01", "tag": "entity", "text": "nuclear talks"}]

    sheet1 = pool.build_pool(config, queries, k=3, conn=conn, embedder=embedder, reranker=HashReranker())
    sheet2 = pool.build_pool(config, queries, k=3, conn=conn, embedder=embedder, reranker=HashReranker())
    order1 = [j["item_id"] for j in sheet1["judgments"]]
    order2 = [j["item_id"] for j in sheet2["judgments"]]
    assert order1 == order2


def test_build_pool_stamps_index_version():
    conn, config, embedder = _seeded_conn_and_config()
    queries = [{"id": "q01", "tag": "entity", "text": "nuclear talks"}]

    sheet = pool.build_pool(config, queries, k=3, conn=conn, embedder=embedder, reranker=HashReranker())
    assert sheet["index_version"]
    assert len(sheet["index_version"]) == 12


def test_merge_judgments_carries_forward_matching_entries():
    old_sheet = {"judgments": [
        {"query_id": "q01", "item_id": 1, "relevant": True},
        {"query_id": "q01", "item_id": 2, "relevant": False},
    ]}
    new_sheet = {"judgments": [
        {"query_id": "q01", "item_id": 1, "relevant": None},
        {"query_id": "q01", "item_id": 2, "relevant": None},
        {"query_id": "q01", "item_id": 3, "relevant": None},  # new candidate, e.g. from sparse
    ]}
    merged = pool.merge_judgments(new_sheet, old_sheet)
    by_item = {j["item_id"]: j["relevant"] for j in merged["judgments"]}
    assert by_item == {1: True, 2: False, 3: None}


def test_merge_judgments_handles_no_prior_sheet():
    new_sheet = {"judgments": [{"query_id": "q01", "item_id": 1, "relevant": None}]}
    assert pool.merge_judgments(new_sheet, None) is new_sheet


def test_merge_judgments_never_overwrites_with_null():
    # An old judgment of True/False must survive even if merge order changed.
    old_sheet = {"judgments": [{"query_id": "q01", "item_id": 1, "relevant": True}]}
    new_sheet = {"judgments": [{"query_id": "q01", "item_id": 1, "relevant": None}]}
    merged = pool.merge_judgments(new_sheet, old_sheet)
    assert merged["judgments"][0]["relevant"] is True
