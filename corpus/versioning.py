from __future__ import annotations

import hashlib
from pathlib import Path

from corpus.config import Config
from corpus.embed import DOCUMENT_PREFIX, QUERY_PREFIX


def _parser_version() -> str:
    """Auto-derived from parse.py's own source, so index_version changes the
    moment parsing logic changes — nothing to remember to bump by hand.
    """
    parse_py = Path(__file__).parent / "parse.py"
    return hashlib.sha256(parse_py.read_bytes()).hexdigest()[:12]


def compute_index_version(config: Config) -> str:
    """sha256(model, dim, chunk_size, overlap, prefixes, parser_version)[:12].

    Stamped on every index build and every eval result. A mismatch between
    two stamps means the runs are not comparable — different configs produced
    them, so a score delta might just be config drift, not a real change.
    """
    parts = "|".join(
        [
            config.embedding_model_name,
            str(config.embedding_dim),
            str(config.chunk_target_tokens),
            str(config.chunk_overlap_ratio),
            QUERY_PREFIX,
            DOCUMENT_PREFIX,
            _parser_version(),
        ]
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:12]
