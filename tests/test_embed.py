import pytest

# corpus.embed imports sentence_transformers/torch at module level, which costs
# real seconds even just to *collect* this file. Import lazily inside each
# test so a normal `pytest` run (slow tests deselected) never pays that cost.


@pytest.mark.slow
def test_embed_output_shape_and_normalization():
    import numpy as np

    from corpus.config import load_config
    from corpus.embed import GemmaEmbedder

    config = load_config("config.toml")
    embedder = GemmaEmbedder(config)

    vecs = embedder.encode(["a test document about nuclear policy"], kind="doc")
    assert vecs.shape == (1, config.embedding_dim)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.slow
def test_query_and_document_prefixes_differ():
    import numpy as np

    from corpus.config import load_config
    from corpus.embed import DOCUMENT_PREFIX, QUERY_PREFIX, GemmaEmbedder

    assert QUERY_PREFIX != DOCUMENT_PREFIX

    config = load_config("config.toml")
    embedder = GemmaEmbedder(config)

    same_text = "nuclear negotiations"
    query_vec = embedder.encode([same_text], kind="query")[0]
    doc_vec = embedder.encode([same_text], kind="doc")[0]
    # Same raw text, different task prefix -> different embeddings.
    assert not np.allclose(query_vec, doc_vec)


@pytest.mark.slow
def test_relevant_text_ranks_above_irrelevant_text():
    from corpus.config import load_config
    from corpus.embed import GemmaEmbedder

    config = load_config("config.toml")
    embedder = GemmaEmbedder(config)

    query_vec = embedder.encode(["nuclear negotiations"], kind="query")[0]
    docs = embedder.encode(
        [
            "Iran and world powers resumed nuclear talks in Vienna.",
            "The basketball championship concluded last night.",
        ],
        kind="doc",
    )
    relevant_sim = query_vec @ docs[0]
    irrelevant_sim = query_vec @ docs[1]
    assert relevant_sim > irrelevant_sim
