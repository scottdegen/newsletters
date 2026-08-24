import dataclasses

from corpus.config import load_config
from corpus.versioning import compute_index_version


def test_index_version_is_deterministic():
    config = load_config("config.toml")
    assert compute_index_version(config) == compute_index_version(config)


def test_index_version_changes_with_chunk_config():
    config = load_config("config.toml")
    other = dataclasses.replace(config, chunk_target_tokens=256)
    assert compute_index_version(config) != compute_index_version(other)


def test_index_version_changes_with_embedding_dim():
    config = load_config("config.toml")
    other = dataclasses.replace(config, embedding_dim=128)
    assert compute_index_version(config) != compute_index_version(other)


def test_index_version_is_short_hex():
    config = load_config("config.toml")
    version = compute_index_version(config)
    assert len(version) == 12
    int(version, 16)  # raises if not valid hex
