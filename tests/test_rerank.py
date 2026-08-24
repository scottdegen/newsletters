import pytest

from corpus.rerank import HashReranker

# corpus.rerank's CrossEncoderReranker imports sentence_transformers at
# __init__ time, not module level, so importing the module itself is cheap --
# only tests that actually construct one need @pytest.mark.slow.


def test_hash_reranker_is_deterministic():
    reranker = HashReranker()
    texts = ["nuclear talks in vienna", "basketball championship recap"]
    a = reranker.score("nuclear negotiations", texts)
    b = reranker.score("nuclear negotiations", texts)
    assert a == b


def test_hash_reranker_scores_differ_by_query():
    reranker = HashReranker()
    texts = ["nuclear talks in vienna"]
    a = reranker.score("nuclear negotiations", texts)
    b = reranker.score("basketball", texts)
    assert a != b


@pytest.mark.slow
def test_cross_encoder_reranker_ranks_relevant_text_higher():
    from corpus.config import load_config
    from corpus.rerank import CrossEncoderReranker

    config = load_config("config.toml")
    reranker = CrossEncoderReranker(config)

    scores = reranker.score(
        "nuclear negotiations",
        [
            "Iran and world powers resumed nuclear talks in Vienna.",
            "The basketball championship concluded last night.",
        ],
    )
    assert scores[0] > scores[1]
