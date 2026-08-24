import sqlite3

from corpus import db


def _memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(db.SCHEMA)
    return conn


def _seed_item(conn: sqlite3.Connection) -> int:
    source_id = db.upsert_source(
        conn, "test-src", "Test Source", "https://x.test/feed", None, "wordpress"
    )
    issue_id, _ = db.insert_issue(
        conn, source_id, "https://x.test/a", "Title", "2026-06-01T00:00:00Z",
        "raw/a.html", "hash-a", "2026-06-01T00:00:00Z",
    )
    return db.insert_item(conn, issue_id, 0, "Headline", "some item text", "https://x.test/a")


def test_get_unindexed_items_returns_items_with_no_chunks():
    conn = _memory_conn()
    item_id = _seed_item(conn)

    unindexed = db.get_unindexed_items(conn)
    assert [row[0] for row in unindexed] == [item_id]


def test_get_unindexed_items_excludes_chunked_items():
    conn = _memory_conn()
    item_id = _seed_item(conn)
    db.insert_chunk(conn, item_id, 0, "some item text", 3)

    unindexed = db.get_unindexed_items(conn)
    assert unindexed == []


def test_insert_chunk_keeps_fts_rowid_in_sync():
    conn = _memory_conn()
    item_id = _seed_item(conn)
    chunk_id = db.insert_chunk(conn, item_id, 0, "nuclear policy discussion", 3)

    hits = conn.execute(
        "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'nuclear'"
    ).fetchall()
    assert hits == [(chunk_id,)]
