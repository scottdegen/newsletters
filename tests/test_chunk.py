from corpus.chunk import chunk


def test_chunk_empty_text_returns_nothing():
    assert chunk("") == []
    assert chunk("   ") == []


def test_chunk_short_text_is_a_single_chunk():
    text = "one two three four five"
    chunks = chunk(text, target_tokens=512, overlap_ratio=0.15)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].token_count == 5


def test_chunk_splits_long_text_with_overlap():
    words = [f"w{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk(text, target_tokens=100, overlap_ratio=0.15)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 100

    # Consecutive chunks overlap: the tail of one reappears at the head of the next.
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert first_words[-1] != second_words[0]  # sanity: not identical single word
    overlap = set(first_words[-10:]) & set(second_words[:10])
    assert len(overlap) > 0


def test_chunk_covers_all_words_without_gaps():
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)
    chunks = chunk(text, target_tokens=20, overlap_ratio=0.15)
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    assert covered == set(words)


def test_chunk_is_deterministic():
    text = " ".join(f"w{i}" for i in range(300))
    assert chunk(text, target_tokens=50) == chunk(text, target_tokens=50)
