from pathlib import Path

import yaml

QUERIES_PATH = Path(__file__).parent.parent / "evals" / "queries.yaml"
VALID_TAGS = {"entity", "paraphrase", "temporal", "multi-source"}


def _load_queries():
    return yaml.safe_load(QUERIES_PATH.read_text())["queries"]


def test_exactly_40_queries():
    assert len(_load_queries()) == 40


def test_query_ids_are_unique():
    ids = [q["id"] for q in _load_queries()]
    assert len(ids) == len(set(ids))


def test_all_tags_valid_and_evenly_distributed():
    queries = _load_queries()
    for q in queries:
        assert q["tag"] in VALID_TAGS
    counts = {}
    for q in queries:
        counts[q["tag"]] = counts.get(q["tag"], 0) + 1
    assert counts == {tag: 10 for tag in VALID_TAGS}


def test_no_empty_query_text():
    for q in _load_queries():
        assert q["text"].strip()
