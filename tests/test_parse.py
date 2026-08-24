from pathlib import Path

import pytest

from corpus.parse import parse, segment

# Real newsletter HTML is excluded from the public repo for privacy — it would
# reveal exactly which publications are being read. Locally, drop fixtures
# under tests/fixtures/ (see README) and these tests run normally; on a fresh
# clone without them, they skip rather than fail.
FIXTURES = Path(__file__).parent / "fixtures"
ALL_FIXTURES = sorted(FIXTURES.glob("*.html"))
requires_fixtures = pytest.mark.skipif(
    not ALL_FIXTURES, reason="tests/fixtures/ not present (excluded from public repo)"
)


@requires_fixtures
def test_parse_strips_script_tags():
    html = (FIXTURES / "generic-article-1.html").read_text()
    md = parse(html)
    assert "<script" not in md
    assert len(md) > 200


@requires_fixtures
def test_parse_strips_unsubscribe_style_links():
    html = (FIXTURES / "generic-article-1.html").read_text()
    md = parse(html)
    assert "unsubscribe" not in md.lower()


@requires_fixtures
def test_parse_produces_readable_text_on_substack_layout():
    html = (FIXTURES / "substack-post-1.html").read_text()
    md = parse(html)
    assert len(md) > 200


@requires_fixtures
def test_parse_is_pure_and_deterministic():
    html = (FIXTURES / "generic-article-1.html").read_text()
    assert parse(html) == parse(html)


@requires_fixtures
@pytest.mark.parametrize("fixture_path", ALL_FIXTURES, ids=lambda p: p.name)
def test_parse_handles_every_source_template_without_crashing(fixture_path):
    html = fixture_path.read_text()
    md = parse(html)
    # Deliberately low bar: some fixtures are paywalled posts where an
    # anonymous fetch only gets a short teaser — expected per PLAN.md's
    # "free-tier truncation accepted" non-goal, not a parser bug. This test
    # is about "didn't crash and didn't leak markup," not "produced
    # substantial content."
    assert len(md) > 0
    assert "<script" not in md
    assert "<style" not in md


@requires_fixtures
def test_segment_splits_a_digest_issue_into_one_item_per_section():
    html = (FIXTURES / "digest-bold-headers-1.html").read_text()
    items = segment(html)
    headlines = [i.headline for i in items]
    assert headlines == [
        "IRAN WAR",
        "WEST BANK VIOLENCE",
        "U.S. CARIBBEAN AND PACIFIC OPERATIONS",
        "RUSSIA-UKRAINE WAR",
        "OTHER GLOBAL DEVELOPMENTS",
        "TECH DEVELOPMENTS",
        "U.S. IMMIGRATION DEVELOPMENTS",
        "U.S. DOMESTIC DEVELOPMENTS",
        "TRUMP ADMINISTRATION ACTIONS",
        "TRUMP ADMINISTRATION LITIGATION",
    ]
    assert all(len(i.text) > 0 for i in items)
    # Each section stays on its own topic — no bleed across boundaries.
    assert "iran" in items[0].text.lower()
    assert "iran" not in items[1].text.lower()


@requires_fixtures
def test_segment_falls_back_to_one_whole_issue_item_for_non_digest_html():
    html = (FIXTURES / "digest-fallback-1.html").read_text()
    items = segment(html)
    assert len(items) == 1
    assert items[0].headline is None
    assert len(items[0].text) > 0


@requires_fixtures
@pytest.mark.parametrize("fixture_path", ALL_FIXTURES, ids=lambda p: p.name)
def test_segment_handles_every_source_template_without_crashing(fixture_path):
    html = fixture_path.read_text()
    items = segment(html)
    assert len(items) >= 1
    for item in items:
        assert "<script" not in item.text
        assert "<style" not in item.text


@requires_fixtures
def test_segment_is_pure_and_deterministic():
    html = (FIXTURES / "digest-bold-headers-1.html").read_text()
    assert segment(html) == segment(html)


@requires_fixtures
def test_fixtures_cover_multiple_source_templates():
    # Regression guard for "every source in sources.yaml has a fixture" —
    # kept template-name-agnostic on purpose (sources.yaml is private; see
    # README), so this only checks breadth, not which specific sources.
    covered = {f.stem.rsplit("-", 1)[0] for f in ALL_FIXTURES}
    assert len(covered) >= 8
