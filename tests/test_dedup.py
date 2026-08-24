import sqlite3

from corpus import db


def _memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(db.SCHEMA)
    return conn


def test_insert_issue_is_idempotent_on_canonical_url():
    conn = _memory_conn()
    source_id = db.upsert_source(
        conn, "test-src", "Test Source", "https://x.test/feed", None, "wordpress"
    )

    id1, inserted1 = db.insert_issue(
        conn, source_id, "https://x.test/a", "Title A", "2026-08-01T00:00:00Z",
        "raw/a.html", "hash-a", "2026-08-01T00:00:00Z",
    )
    id2, inserted2 = db.insert_issue(
        conn, source_id, "https://x.test/a", "Title A", "2026-08-01T00:00:00Z",
        "raw/a.html", "hash-a", "2026-08-02T00:00:00Z",
    )

    assert inserted1 is True
    assert inserted2 is False
    assert id1 == id2
    assert conn.execute("SELECT COUNT(*) FROM issue").fetchone()[0] == 1


def test_issue_exists():
    conn = _memory_conn()
    source_id = db.upsert_source(
        conn, "test-src", "Test Source", "https://x.test/feed", None, "wordpress"
    )
    assert db.issue_exists(conn, "https://x.test/a") is False
    db.insert_issue(
        conn, source_id, "https://x.test/a", "T", None,
        "raw/a.html", "h", "2026-08-01T00:00:00Z",
    )
    assert db.issue_exists(conn, "https://x.test/a") is True


def test_upsert_source_is_stable_across_calls():
    conn = _memory_conn()
    id1 = db.upsert_source(conn, "slug", "Name", "https://x.test/feed", None, "wordpress")
    id2 = db.upsert_source(conn, "slug", "Renamed", "https://x.test/feed2", None, "wordpress")
    assert id1 == id2
    assert conn.execute("SELECT COUNT(*) FROM source").fetchone()[0] == 1
