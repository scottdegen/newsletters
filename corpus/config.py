from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    raw_dir: Path
    db_path: Path
    user_agent: str
    rate_limit_per_sec: float
    timeout_seconds: float
    substack_archive_page_size: int
    chunk_target_tokens: int
    chunk_overlap_ratio: float
    embedding_model_name: str
    embedding_dim: int
    embedding_batch_size: int
    retrieval_recency_decay_per_day: float
    reranking_model_name: str


def load_config(path: str | Path = "config.toml") -> Config:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Config(
        raw_dir=Path(data["paths"]["raw_dir"]),
        db_path=Path(data["paths"]["db_path"]),
        user_agent=data["fetch"]["user_agent"],
        rate_limit_per_sec=data["fetch"]["rate_limit_per_sec"],
        timeout_seconds=data["fetch"]["timeout_seconds"],
        substack_archive_page_size=data["fetch"]["substack_archive_page_size"],
        chunk_target_tokens=data["chunk"]["target_tokens"],
        chunk_overlap_ratio=data["chunk"]["overlap_ratio"],
        embedding_model_name=data["embedding"]["model_name"],
        embedding_dim=data["embedding"]["dim"],
        embedding_batch_size=data["embedding"]["batch_size"],
        retrieval_recency_decay_per_day=data["retrieval"]["recency_decay_per_day"],
        reranking_model_name=data["reranking"]["model_name"],
    )
