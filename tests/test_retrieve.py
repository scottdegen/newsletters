import dataclasses
import sqlite3

import pytest

from corpus import db, retrieve
from corpus.config import load_config
from corpus.embed import HashEmbedder
from corpus.rerank import HashReranker


def _seeded_conn(dim: int):
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(db.SCHEMA)
    db.ensure_chunk_vec(conn, dim)

    embedder = HashEmbedder(dim=dim)
    source_id = db.upsert_source(
        conn, "test-src", "Test Source", "https://x.test/feed", None, "wordpress"
    )

    def add(url: str, headline: str, text: str, published_at: str = "2026-06-01T00:00:00Z") -> int:
        issue_id, _ = db.insert_issue(
            conn, source_id, url, headline, published_at,
            "raw/x.html", "hash", published_at,
        )
        item_id = db.insert_item(conn, issue_id, 0, headline, text, url)
        chunk_id = db.insert_chunk(conn, item_id, 0, text, len(text.split()))
        vec = embedder.encode([text], kind="doc")[0]
        db.insert_chunk_vec(
            conn, chunk_id, published_at[:7], vec.astype("float32").tobytes(),
            source_id, published_at,
        )
        return chunk_id

    add("https://x.test/nuclear", "Iran nuclear talks", "nuclear negotiations vienna")
    add("https://x.test/basketball", "NBA recap", "basketball championship game seven")
    conn.commit()
    return conn, embedder


def test_retrieve_dense_returns_chunks_with_full_citation_metadata():
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve(
        "nuclear negotiations vienna", k=2, mode="dense",
        conn=conn, embedder=embedder, config=config,
    )
    assert len(results) == 2
    top = results[0]
    assert top.headline == "Iran nuclear talks"
    assert top.canonical_url == "https://x.test/nuclear"
    assert top.source_slug == "test-src"
    # HashEmbedder folds `kind` into the hash, so even a verbatim-matching
    # query/doc pair won't land at distance 0 — just assert correct ordering.
    assert top.distance < results[1].distance


def test_retrieve_runs_fast_with_no_model_loaded():
    import time

    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    start = time.monotonic()
    retrieve.retrieve(
        "nuclear negotiations", k=2, mode="dense",
        conn=conn, embedder=embedder, config=config,
    )
    assert time.monotonic() - start < 1.0


def test_recency_breaks_a_tie_toward_the_newer_item():
    import datetime as dt

    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(db.SCHEMA)
    db.ensure_chunk_vec(conn, 16)
    embedder = HashEmbedder(dim=16)
    source_id = db.upsert_source(
        conn, "test-src", "Test Source", "https://x.test/feed", None, "wordpress"
    )

    # Identical text -> identical HashEmbedder vector -> tied base distance.
    # Only recency should be able to separate them.
    same_text = "shared content identical for both items"
    for key, published_at in [("old", "2020-01-01T00:00:00Z"), ("new", "2026-06-01T00:00:00Z")]:
        url = f"https://x.test/{key}"
        issue_id, _ = db.insert_issue(
            conn, source_id, url, key, published_at, "raw/x.html", "hash", published_at,
        )
        item_id = db.insert_item(conn, issue_id, 0, key, same_text, url)
        chunk_id = db.insert_chunk(conn, item_id, 0, same_text, len(same_text.split()))
        vec = embedder.encode([same_text], kind="doc")[0]
        db.insert_chunk_vec(
            conn, chunk_id, published_at[:7], vec.astype("float32").tobytes(),
            source_id, published_at,
        )
    conn.commit()
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)
    as_of = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)

    without_recency = retrieve.retrieve(
        same_text, k=2, mode="dense", conn=conn, embedder=embedder, config=config,
    )
    assert without_recency[0].distance == without_recency[1].distance  # confirmed tie

    with_recency = retrieve.retrieve(
        same_text, k=2, mode="dense", conn=conn, embedder=embedder, config=config,
        recency=True, as_of=as_of,
    )
    assert with_recency[0].canonical_url == "https://x.test/new"
    assert with_recency[0].distance < with_recency[1].distance


def test_recency_does_not_change_which_items_are_retrieved():
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    without = retrieve.retrieve(
        "nuclear negotiations", k=2, mode="dense", conn=conn, embedder=embedder, config=config,
    )
    with_recency = retrieve.retrieve(
        "nuclear negotiations", k=2, mode="dense", conn=conn, embedder=embedder,
        config=config, recency=True,
    )
    assert {c.chunk_id for c in without} == {c.chunk_id for c in with_recency}


def test_recency_defaults_to_off():
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    default = retrieve.retrieve(
        "nuclear negotiations", k=2, mode="dense", conn=conn, embedder=embedder, config=config,
    )
    explicit_off = retrieve.retrieve(
        "nuclear negotiations", k=2, mode="dense", conn=conn, embedder=embedder,
        config=config, recency=False,
    )
    assert default == explicit_off


ALL_MODES = ["dense", "sparse", "hybrid", "rerank", "parent", "temporal"]


def _mode_kwargs(mode: str) -> dict:
    return {"reranker": HashReranker()} if mode == "rerank" else {}


@pytest.mark.parametrize("mode", ALL_MODES)
def test_source_filter_excludes_other_sources(mode):
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve(
        "nuclear negotiations vienna", k=5, mode=mode,
        conn=conn, embedder=embedder, config=config, source="test-src",
        **_mode_kwargs(mode),
    )
    assert all(r.source_slug == "test-src" for r in results)


def test_source_filter_unknown_slug_returns_nothing():
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve(
        "nuclear negotiations", k=5, mode="dense",
        conn=conn, embedder=embedder, config=config, source="does-not-exist",
    )
    assert results == []


@pytest.mark.parametrize("mode", ALL_MODES)
def test_before_filter_excludes_everything(mode):
    # Both fixture items are published 2026-06-01; a `before` cutoff earlier
    # than that should exclude them all.
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve(
        "nuclear negotiations", k=5, mode=mode,
        conn=conn, embedder=embedder, config=config, before="2026-01-01T00:00:00Z",
        **_mode_kwargs(mode),
    )
    assert results == []


@pytest.mark.parametrize("mode", ALL_MODES)
def test_after_filter_includes_matching_items(mode):
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve(
        "nuclear negotiations", k=5, mode=mode,
        conn=conn, embedder=embedder, config=config, after="2026-01-01T00:00:00Z",
        **_mode_kwargs(mode),
    )
    assert len(results) > 0


def test_retrieve_sparse_matches_keyword():
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve("nuclear vienna", k=2, mode="sparse", conn=conn, config=config)
    assert results[0].headline == "Iran nuclear talks"


def test_retrieve_sparse_needs_no_embedder():
    conn, _ = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)
    # No embedder passed at all -- sparse mode must not require one.
    results = retrieve.retrieve("basketball", k=2, mode="sparse", conn=conn, config=config)
    assert results[0].headline == "NBA recap"


def test_retrieve_hybrid_surfaces_relevant_item_first():
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve(
        "nuclear negotiations vienna", k=2, mode="hybrid",
        conn=conn, embedder=embedder, config=config,
    )
    assert len(results) == 2
    assert results[0].headline == "Iran nuclear talks"


def test_retrieve_hybrid_distance_is_lower_is_better_like_every_other_mode():
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve(
        "nuclear negotiations vienna", k=2, mode="hybrid",
        conn=conn, embedder=embedder, config=config,
    )
    assert results[0].distance <= results[1].distance


def test_retrieve_raises_for_unwired_mode():
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    with pytest.raises(NotImplementedError):
        retrieve.retrieve(
            "q", k=5, mode="nonexistent", conn=conn, embedder=embedder, config=config
        )


def test_retrieve_rerank_returns_k_chunks_reordered_by_reranker_score():
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve(
        "nuclear negotiations vienna", k=2, mode="rerank",
        conn=conn, embedder=embedder, config=config, reranker=HashReranker(),
    )
    assert len(results) == 2
    # Reranker score is higher-is-better; retrieve() must negate it into
    # distance so lower-is-better still holds, same as every other mode.
    assert results[0].distance <= results[1].distance


def test_retrieve_rerank_is_deterministic():
    conn, embedder = _seeded_conn(dim=16)
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    a = retrieve.retrieve(
        "nuclear negotiations vienna", k=2, mode="rerank",
        conn=conn, embedder=embedder, config=config, reranker=HashReranker(),
    )
    b = retrieve.retrieve(
        "nuclear negotiations vienna", k=2, mode="rerank",
        conn=conn, embedder=embedder, config=config, reranker=HashReranker(),
    )
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_retrieve_parent_collapses_multiple_chunks_to_one_item():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(db.SCHEMA)
    db.ensure_chunk_vec(conn, 16)
    embedder = HashEmbedder(dim=16)
    source_id = db.upsert_source(
        conn, "test-src", "Test Source", "https://x.test/feed", None, "wordpress"
    )

    published_at = "2026-06-01T00:00:00Z"
    full_text = "nuclear negotiations vienna talks continue into a second week"
    issue_id, _ = db.insert_issue(
        conn, source_id, "https://x.test/nuclear", "Iran nuclear talks",
        published_at, "raw/x.html", "hash", published_at,
    )
    item_id = db.insert_item(conn, issue_id, 0, "Iran nuclear talks", full_text, "https://x.test/nuclear")
    for ordinal, piece in enumerate(["nuclear negotiations vienna", "talks continue second week"]):
        chunk_id = db.insert_chunk(conn, item_id, ordinal, piece, len(piece.split()))
        vec = embedder.encode([piece], kind="doc")[0]
        db.insert_chunk_vec(
            conn, chunk_id, published_at[:7], vec.astype("float32").tobytes(),
            source_id, published_at,
        )
    conn.commit()
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve(
        "nuclear negotiations vienna", k=5, mode="parent",
        conn=conn, embedder=embedder, config=config,
    )
    # Two chunks, same item -- parent mode must collapse to one result and
    # swap in the full item text, not a single ~half-length chunk fragment.
    assert len(results) == 1
    assert results[0].item_id == item_id
    assert results[0].text == full_text


def test_retrieve_temporal_diversifies_across_months():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(db.SCHEMA)
    db.ensure_chunk_vec(conn, 16)
    embedder = HashEmbedder(dim=16)
    source_id = db.upsert_source(
        conn, "test-src", "Test Source", "https://x.test/feed", None, "wordpress"
    )

    # Same text every month -> tied relevance ranking, so only time
    # diversification (not the underlying score) can explain a spread of
    # distinct months in the result.
    months = ["2026-01", "2026-02", "2026-03", "2026-04"]
    for ym in months:
        published_at = f"{ym}-15T00:00:00Z"
        text = "nuclear negotiations vienna"
        url = f"https://x.test/{ym}"
        issue_id, _ = db.insert_issue(
            conn, source_id, url, ym, published_at, "raw/x.html", "hash", published_at,
        )
        item_id = db.insert_item(conn, issue_id, 0, ym, text, url)
        chunk_id = db.insert_chunk(conn, item_id, 0, text, len(text.split()))
        vec = embedder.encode([text], kind="doc")[0]
        db.insert_chunk_vec(
            conn, chunk_id, ym, vec.astype("float32").tobytes(), source_id, published_at,
        )
    conn.commit()
    config = dataclasses.replace(load_config("config.toml"), embedding_dim=16)

    results = retrieve.retrieve(
        "nuclear negotiations vienna", k=4, mode="temporal",
        conn=conn, embedder=embedder, config=config,
    )
    seen_months = {r.published_at[:7] for r in results}
    assert seen_months == set(months)


def _chunk(chunk_id: int, published_at: str | None) -> retrieve.Chunk:
    return retrieve.Chunk(
        chunk_id=chunk_id, text="x", distance=float(chunk_id), item_id=chunk_id,
        headline=None, canonical_url=None, source_slug="test-src", published_at=published_at,
    )


def test_time_diversify_round_robins_oldest_bucket_first():
    chunks = [
        _chunk(1, "2026-03-01T00:00:00Z"),
        _chunk(2, "2026-01-01T00:00:00Z"),
        _chunk(3, "2026-02-01T00:00:00Z"),
        _chunk(4, "2026-01-05T00:00:00Z"),  # second item in Jan bucket
    ]
    result = retrieve._time_diversify(chunks, k=3)
    # Round 1 visits buckets chronologically (Jan, Feb, Mar): id 2, then 3, then 1.
    assert [c.chunk_id for c in result] == [2, 3, 1]


def test_time_diversify_returns_everything_when_k_exceeds_pool():
    chunks = [_chunk(1, "2026-01-01T00:00:00Z"), _chunk(2, "2026-02-01T00:00:00Z")]
    result = retrieve._time_diversify(chunks, k=10)
    assert len(result) == 2


def test_time_diversify_is_deterministic():
    chunks = [_chunk(i, f"2026-{(i % 6) + 1:02d}-01T00:00:00Z") for i in range(1, 20)]
    a = retrieve._time_diversify(chunks, k=8)
    b = retrieve._time_diversify(chunks, k=8)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
