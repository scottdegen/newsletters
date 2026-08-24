from corpus.retrieve import _sanitize_fts_query


def test_sanitize_handles_apostrophes_without_crashing():
    # Original bug: raw text with an apostrophe raised
    # "sqlite3.OperationalError: fts5: syntax error near '''".
    result = _sanitize_fts_query("What were the results of JD Vance's Iran negotiations?")
    assert "'" not in result  # apostrophe consumed by \w+ tokenization
    assert '"JD"' in result
    assert '"Vance"' in result


def test_sanitize_drops_stopwords():
    result = _sanitize_fts_query("What is the plan for Iran?")
    assert '"What"' not in result
    assert '"is"' not in result
    assert '"the"' not in result
    assert '"Iran"' in result


def test_sanitize_joins_with_or():
    result = _sanitize_fts_query("Iran nuclear talks")
    assert result == '"Iran" OR "nuclear" OR "talks"'


def test_sanitize_handles_all_stopword_input():
    # Degenerate case: nothing survives the stopword filter.
    result = _sanitize_fts_query("What is the")
    assert result  # must not be empty -- empty string is an invalid MATCH query


def test_sanitize_never_leaves_a_bare_double_quote_in_a_token():
    # \w+ tokenization already strips quote characters before escaping runs,
    # so there's never an embedded quote in practice -- this just confirms
    # the output is always a clean sequence of "token" OR "token" groups.
    result = _sanitize_fts_query('say "hello" now')
    assert result == '"say" OR "hello" OR "now"'
