import sqlite3

from corpus import db, links
from corpus.links import _is_tracker


def test_is_tracker_matches_known_esp_domains():
    assert _is_tracker("https://link.mail.beehiiv.com/ss/c/abc123")
    assert _is_tracker("https://example.substack.com/redirect?url=https://x.test")


def test_is_tracker_ignores_plain_article_links():
    assert not _is_tracker("https://example-news-site.com/2026/08/some-article/")
    assert not _is_tracker("https://www.reuters.com/world/some-story")


def test_unwrap_url_caches_after_first_resolve(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.executescript(db.SCHEMA)

    calls = []

    def fake_resolve(url, config, session, limiter):
        calls.append(url)
        return "https://resolved.example/final"

    monkeypatch.setattr(links, "_resolve", fake_resolve)

    tracker_url = "https://link.mail.beehiiv.com/x"
    first = links.unwrap_url(tracker_url, None, None, None, conn)
    second = links.unwrap_url(tracker_url, None, None, None, conn)

    assert first == "https://resolved.example/final"
    assert second == "https://resolved.example/final"
    assert len(calls) == 1  # second call served from link_map, not _resolve
