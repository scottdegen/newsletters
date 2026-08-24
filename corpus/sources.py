from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import feedparser
import requests

from corpus.config import Config


@dataclass(frozen=True)
class DiscoveredIssue:
    canonical_url: str
    title: str | None
    published_at: str | None


class RateLimiter:
    """Enforces a minimum interval between successive requests."""

    def __init__(self, per_sec: float) -> None:
        self._min_interval = 1.0 / per_sec if per_sec > 0 else 0.0
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


def fetch(
    url: str,
    config: Config,
    session: requests.Session,
    limiter: RateLimiter,
    **kwargs,
) -> requests.Response:
    limiter.wait()
    resp = session.get(
        url,
        headers={"User-Agent": config.user_agent},
        timeout=config.timeout_seconds,
        **kwargs,
    )
    resp.raise_for_status()
    return resp


def discover_issues(
    source: dict,
    config: Config,
    session: requests.Session,
    limiter: RateLimiter,
    since: str | None = None,
) -> Iterator[DiscoveredIssue]:
    """`since`, if given, is an ISO 8601 date string; older issues are skipped."""
    platform = source["platform"]
    if platform == "substack":
        yield from _discover_substack(source, config, session, limiter, since)
    elif platform == "csis-filtered":
        yield from _discover_feed(
            source, config, session, limiter,
            program_filter=source["program_filter"], since=since,
        )
    elif platform == "wordpress":
        yield from _discover_wordpress_paginated(source, config, session, limiter, since)
    else:  # custom-rss: no known deep-history mechanism, single fetch only
        yield from _discover_feed(source, config, session, limiter, since=since)


def _entry_to_issue(entry, program_filter: str | None) -> DiscoveredIssue | None:
    if program_filter is not None:
        haystack = entry.get("summary", "") + entry.get("title", "")
        if program_filter.lower() not in haystack.lower():
            return None
    published_at = None
    if entry.get("published_parsed"):
        published_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", entry.published_parsed)
    return DiscoveredIssue(
        canonical_url=entry.link, title=entry.get("title"), published_at=published_at
    )


def _discover_feed(
    source: dict,
    config: Config,
    session: requests.Session,
    limiter: RateLimiter,
    program_filter: str | None = None,
    since: str | None = None,
) -> Iterator[DiscoveredIssue]:
    resp = fetch(source["feed_url"], config, session, limiter)
    parsed = feedparser.parse(resp.content)
    for entry in parsed.entries:
        issue = _entry_to_issue(entry, program_filter)
        if issue is None:
            continue
        if since is not None and issue.published_at is not None and issue.published_at < since:
            continue
        yield issue


# WordPress's default feed returns ~10 recent items only. `?paged=N` walks
# further back; a site off the end of its pagination either 404s or loops
# back to page 1 (rather than erroring), so both are treated as "no more
# pages." The cap bounds a misbehaving site from paginating forever.
_WORDPRESS_PAGE_CAP = 100


def _discover_wordpress_paginated(
    source: dict,
    config: Config,
    session: requests.Session,
    limiter: RateLimiter,
    since: str | None = None,
) -> Iterator[DiscoveredIssue]:
    feed_url = source["feed_url"]
    first_link_seen: str | None = None
    for page_num in range(1, _WORDPRESS_PAGE_CAP + 1):
        url = feed_url if page_num == 1 else f"{feed_url.rstrip('/')}/?paged={page_num}"
        try:
            resp = fetch(url, config, session, limiter)
        except requests.HTTPError:
            return
        parsed = feedparser.parse(resp.content)
        if not parsed.entries:
            return
        if parsed.entries[0].link == first_link_seen:
            return  # site looped back to page 1: pagination has run out
        first_link_seen = parsed.entries[0].link if page_num == 1 else first_link_seen

        reached_cutoff = False
        for entry in parsed.entries:
            issue = _entry_to_issue(entry, program_filter=None)
            if issue is None:
                continue
            if since is not None and issue.published_at is not None and issue.published_at < since:
                reached_cutoff = True
                continue
            yield issue
        if reached_cutoff:
            return


def _discover_substack(
    source: dict,
    config: Config,
    session: requests.Session,
    limiter: RateLimiter,
    since: str | None = None,
) -> Iterator[DiscoveredIssue]:
    base = source["feed_url"].rsplit("/feed", 1)[0]
    offset = 0
    page_size = config.substack_archive_page_size
    while True:
        url = f"{base}/api/v1/archive?sort=new&offset={offset}&limit={page_size}"
        resp = fetch(url, config, session, limiter)
        page = resp.json()
        if not page:
            return
        for post in page:
            post_date = post.get("post_date")
            if since is not None and post_date is not None and post_date < since:
                # sort=new means everything from here on is even older.
                return
            yield DiscoveredIssue(
                canonical_url=post["canonical_url"],
                title=post.get("title"),
                published_at=post_date,
            )
        if len(page) < page_size:
            return
        offset += page_size
