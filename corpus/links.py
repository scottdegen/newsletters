from __future__ import annotations

from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from corpus import db
from corpus.config import Config
from corpus.sources import RateLimiter

# Only links on these domains get a resolve request. Most links inside an
# already-fetched article page are plain article URLs; blindly HEAD-ing every
# link would multiply request volume per issue for little benefit. This list
# targets the ESP/redirect domains Phase 1 actually needs to unwrap.
_TRACKER_DOMAINS = (
    "link.mail.beehiiv.com",
    "substack.com/redirect",
    "list-manage.com",
    "sendgrid.net",
    "mailchimp.com",
    "click.convertkit-mail",
    "url.avanan.click",
    "clicks.substack.com",
)


def _is_tracker(url: str) -> bool:
    return any(domain in url for domain in _TRACKER_DOMAINS)


def _resolve(
    url: str, config: Config, session: requests.Session, limiter: RateLimiter
) -> str:
    limiter.wait()
    try:
        resp = session.head(
            url,
            headers={"User-Agent": config.user_agent},
            timeout=config.timeout_seconds,
            allow_redirects=True,
        )
        if resp.status_code < 400:
            return resp.url
    except requests.RequestException:
        pass
    try:
        limiter.wait()
        resp = session.get(
            url,
            headers={"User-Agent": config.user_agent},
            timeout=config.timeout_seconds,
            allow_redirects=True,
            stream=True,
        )
        resp.close()
        return resp.url
    except requests.RequestException:
        return url


def unwrap_url(
    url: str,
    config: Config,
    session: requests.Session,
    limiter: RateLimiter,
    conn,
) -> str:
    cached = db.get_cached_link(conn, url)
    if cached is not None:
        return cached
    resolved = _resolve(url, config, session, limiter)
    db.cache_link(conn, url, resolved, datetime.now(timezone.utc).isoformat())
    return resolved


def unwrap_html_links(
    html: str,
    config: Config,
    session: requests.Session,
    limiter: RateLimiter,
    conn,
) -> str:
    """Rewrites tracker/redirect hrefs to their resolved destination.

    Non-tracker links pass through untouched. Impure (network + DB cache) —
    call this before parse.parse(), which must stay pure.
    """
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if _is_tracker(href):
            a["href"] = unwrap_url(href, config, session, limiter, conn)
    return str(soup)
