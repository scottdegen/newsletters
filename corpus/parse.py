from __future__ import annotations

from typing import NamedTuple

import html2text
from bs4 import BeautifulSoup

# Tried in order; first match wins. Covers WordPress, Substack, and generic
# article markup. Falls back to <body> minus chrome if nothing matches.
_CONTENT_SELECTORS = [
    "div.available-content",  # Substack
    "div.body.markup",  # Substack (older theme)
    "div.post-primary",  # Just Security (WordPress, this theme) — narrower
    # than div.entry-content: excludes the "Listen to Article" / "About the
    # Author" chrome that sits elsewhere inside <main> on this theme.
    "div.entry-content",  # WordPress
    "div.post-content",
    "article",
    "main",
]

# Removed unconditionally, wherever they appear in the document.
_ALWAYS_STRIP = [
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "img",
    "svg",
    "picture",
]

_STRIP_LINK_KEYWORDS = (
    "unsubscribe",
    "view in browser",
    "view this email in your browser",
    "privacy policy",
    "manage preferences",
    "manage your subscription",
    "share this post",
    "terms of service",
)


def _extract_content(soup: BeautifulSoup) -> BeautifulSoup:
    for selector in _CONTENT_SELECTORS:
        found = soup.select_one(selector)
        if found is not None:
            return found
    return soup.body or soup


def _clean(content: BeautifulSoup) -> None:
    """Mutates content in place: strips chrome tags and subscribe/share CTA links."""
    for tag in content.find_all(_ALWAYS_STRIP):
        tag.decompose()

    for a in content.find_all("a"):
        text = a.get_text(strip=True).lower()
        if any(kw in text for kw in _STRIP_LINK_KEYWORDS):
            a.decompose()


def _to_markdown(node) -> str:
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.ignore_images = True
    converter.body_width = 0
    return converter.handle(str(node)).strip()


def parse(html: str) -> str:
    """Pure: raw fetched page HTML -> normalized Markdown body.

    No network I/O. Strips site chrome, nav, footers, subscribe/share CTAs,
    and images (Phase 1 is text-only; see CLAUDE.md).
    """
    soup = BeautifulSoup(html, "html.parser")
    content = _extract_content(soup)
    _clean(content)
    return _to_markdown(content)


class Item(NamedTuple):
    headline: str | None  # None means "use the issue title" (default, whole-issue item)
    text: str


# Blocks tried, in document order, as digest section boundaries.
_BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "ul", "ol", "blockquote", "hr"]
_SECTION_HEADER_MAX_LEN = 80


def _section_header_text(block) -> str | None:
    """A <p> that is one bold+italic run of short all-caps text is the
    "bold-headline" digest pattern PLAN.md calls for splitting on — e.g. Just
    Security's "Early Edition" marks each topic with a standalone paragraph
    like `**_IRAN WAR_**` instead of a real <h2>. Real <h2>/<hr> boundaries
    (Axios-style digests) fall out of _BLOCK_TAGS directly and don't need
    this check.
    """
    if block.name != "p":
        return None
    if not (block.find(["b", "strong"]) and block.find(["i", "em"])):
        return None
    text = block.get_text(strip=True)
    letters = [c for c in text if c.isalpha()]
    if not letters or not all(c.isupper() for c in letters):
        return None
    if not (0 < len(text) < _SECTION_HEADER_MAX_LEN):
        return None
    return text


def segment(html: str) -> list[Item]:
    """Pure: raw issue HTML -> one Item per digest section.

    Digest-style issues (e.g. Just Security's "Early Edition") bundle many
    unrelated stories into one issue. Treating the whole issue as a single
    item makes that item topically diffuse — see README's rerank diagnosis:
    a cross-encoder over-scores roundup chunks that mention a query topic in
    passing, crowding out chunks actually about it. Splitting on the digest's
    section-header pattern keeps each item on one topic.

    Falls back to a single whole-issue Item (headline=None) when fewer than
    two section headers are found — i.e. every non-digest issue, which is
    the PLAN's stated default ("item segmentation... default = whole issue").
    """
    soup = BeautifulSoup(html, "html.parser")
    content = _extract_content(soup)
    _clean(content)

    blocks = content.find_all(_BLOCK_TAGS)
    headers = [(i, h) for i, b in enumerate(blocks) if (h := _section_header_text(b))]

    if len(headers) < 2:
        return [Item(None, _to_markdown(content))]

    items: list[Item] = []
    for pos, (start, headline) in enumerate(headers):
        end = headers[pos + 1][0] if pos + 1 < len(headers) else len(blocks)
        section_blocks = blocks[start + 1 : end]
        if not section_blocks:
            continue
        fragment = BeautifulSoup("", "html.parser")
        for b in section_blocks:
            fragment.append(b)
        text = _to_markdown(fragment)
        if text:
            items.append(Item(headline, text))

    return items or [Item(None, _to_markdown(content))]
