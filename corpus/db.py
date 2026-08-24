from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS source (
    id INTEGER PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    feed_url TEXT NOT NULL,
    home_url TEXT,
    platform TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issue (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES source(id),
    canonical_url TEXT UNIQUE NOT NULL,
    title TEXT,
    published_at TEXT,
    raw_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS item (
    id INTEGER PRIMARY KEY,
    issue_id INTEGER NOT NULL REFERENCES issue(id),
    ordinal INTEGER NOT NULL,
    headline TEXT,
    text TEXT NOT NULL,
    canonical_url TEXT
);

CREATE TABLE IF NOT EXISTS link_map (
    tracking_url TEXT PRIMARY KEY,
    resolved_url TEXT NOT NULL,
    resolved_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk (
    id INTEGER PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES item(id),
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text, content='chunk', content_rowid='id'
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(SCHEMA)
    return conn


def ensure_chunk_vec(conn: sqlite3.Connection, dim: int) -> None:
    """chunk_vec's embedding width is config-driven (CLAUDE.md: no hardcoded
    dimensions in logic), so it can't live in the static SCHEMA string.
    """
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            ym TEXT PARTITION KEY,
            embedding FLOAT[{dim}],
            +source_id INTEGER,
            +published_at TEXT
        )
        """
    )


def upsert_source(
    conn: sqlite3.Connection,
    slug: str,
    name: str,
    feed_url: str,
    home_url: str | None,
    platform: str,
) -> int:
    conn.execute(
        """
        INSERT INTO source (slug, name, feed_url, home_url, platform)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(slug) DO UPDATE SET
            name = excluded.name,
            feed_url = excluded.feed_url,
            home_url = excluded.home_url,
            platform = excluded.platform
        """,
        (slug, name, feed_url, home_url, platform),
    )
    row = conn.execute("SELECT id FROM source WHERE slug = ?", (slug,)).fetchone()
    return row[0]


def issue_exists(conn: sqlite3.Connection, canonical_url: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM issue WHERE canonical_url = ?", (canonical_url,)
    ).fetchone()
    return row is not None


def insert_issue(
    conn: sqlite3.Connection,
    source_id: int,
    canonical_url: str,
    title: str | None,
    published_at: str | None,
    raw_path: str,
    content_hash: str,
    fetched_at: str,
) -> tuple[int, bool]:
    """Returns (issue_id, was_newly_inserted). Idempotent on canonical_url."""
    cur = conn.execute(
        """
        INSERT INTO issue
            (source_id, canonical_url, title, published_at, raw_path, content_hash, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(canonical_url) DO NOTHING
        """,
        (source_id, canonical_url, title, published_at, raw_path, content_hash, fetched_at),
    )
    row = conn.execute(
        "SELECT id FROM issue WHERE canonical_url = ?", (canonical_url,)
    ).fetchone()
    return row[0], cur.rowcount > 0


def insert_item(
    conn: sqlite3.Connection,
    issue_id: int,
    ordinal: int,
    headline: str | None,
    text: str,
    canonical_url: str | None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO item (issue_id, ordinal, headline, text, canonical_url)
        VALUES (?, ?, ?, ?, ?)
        """,
        (issue_id, ordinal, headline, text, canonical_url),
    )
    return cur.lastrowid


def get_unindexed_items(conn: sqlite3.Connection) -> list[tuple[int, str, int, str | None]]:
    """Items with no chunks yet: (item_id, text, source_id, published_at).

    Basis for `corpus index` resumability — items already chunked are skipped.
    """
    return conn.execute(
        """
        SELECT item.id, item.text, issue.source_id, issue.published_at
        FROM item
        JOIN issue ON issue.id = item.issue_id
        LEFT JOIN chunk ON chunk.item_id = item.id
        WHERE chunk.id IS NULL
        """
    ).fetchall()


def insert_chunk(
    conn: sqlite3.Connection, item_id: int, ordinal: int, text: str, token_count: int
) -> int:
    cur = conn.execute(
        "INSERT INTO chunk (item_id, ordinal, text, token_count) VALUES (?, ?, ?, ?)",
        (item_id, ordinal, text, token_count),
    )
    chunk_id = cur.lastrowid
    conn.execute("INSERT INTO chunk_fts (rowid, text) VALUES (?, ?)", (chunk_id, text))
    return chunk_id


def insert_chunk_vec(
    conn: sqlite3.Connection,
    chunk_id: int,
    ym: str,
    embedding: bytes,
    source_id: int,
    published_at: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO chunk_vec (chunk_id, ym, embedding, source_id, published_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (chunk_id, ym, embedding, source_id, published_at),
    )


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def get_cached_link(conn: sqlite3.Connection, tracking_url: str) -> str | None:
    row = conn.execute(
        "SELECT resolved_url FROM link_map WHERE tracking_url = ?", (tracking_url,)
    ).fetchone()
    return row[0] if row else None


def cache_link(
    conn: sqlite3.Connection, tracking_url: str, resolved_url: str, resolved_at: str
) -> None:
    conn.execute(
        """
        INSERT INTO link_map (tracking_url, resolved_url, resolved_at)
        VALUES (?, ?, ?)
        ON CONFLICT(tracking_url) DO UPDATE SET
            resolved_url = excluded.resolved_url,
            resolved_at = excluded.resolved_at
        """,
        (tracking_url, resolved_url, resolved_at),
    )
