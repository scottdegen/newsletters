from __future__ import annotations

import dataclasses
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from corpus import db
from corpus.config import Config, load_config
from corpus.embed import Embedder, GemmaEmbedder
from corpus.fuse import reciprocal_rank_fusion
from corpus.rerank import CrossEncoderReranker, Reranker

# Small, standard English stopword list — just enough to keep sparse mode's
# FTS5 query from being diluted by near-universal terms. Sparse is an
# intentional baseline (PLAN.md: "not for use... exists so mode 3's [hybrid]
# win is demonstrable"), so this isn't tuned for maximum sparse quality.
_FTS_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "what", "who", "which", "how", "why", "when", "where",
    "has", "have", "had", "be", "been", "do", "does", "did", "with", "by",
    "from", "that", "this", "these", "those", "it", "its", "as", "at", "but",
    "not", "can", "will", "would", "could", "should", "about",
}

# CLAUDE.md invariant 3 fixes this public signature — new retrieval strategies
# are new `mode` values, never new mode-specific parameters. `conn`,
# `embedder`, and `reranker` are the exception: they're infrastructure
# injection points, not retrieval-strategy knobs, and exist so tests can pass
# a HashEmbedder/HashReranker + throwaway DB instead of loading real models.
# All default to real ones when omitted, so callers unaware of testing
# concerns see the exact signature CLAUDE.md specifies. `reranker` is only
# consulted by mode="rerank", same as `embedder` being irrelevant to
# mode="sparse" — "used by every mode" was never literally true even before
# this, since sparse never touches embedder either.

# Hybrid fuses each single-mode candidate list at this depth before RRF —
# per PLAN.md: "1 + 2 at top-50 each, fused by RRF".
HYBRID_CANDIDATE_K = 50

# sqlite-vec 0.1.9 rejects any WHERE constraint on a vec0 auxiliary column
# combined with a KNN clause ("illegal WHERE constraint... in a KNN query",
# verified directly, not assumed) — there's no native pre-filtered KNN here.
# So filtered dense/hybrid queries over-fetch this many unfiltered candidates
# from the KNN, then filter and truncate to k in an outer query. Approximate:
# a very restrictive filter can legitimately return fewer than k results.
CANDIDATE_OVERFETCH_MULTIPLIER = 20
MAX_CANDIDATES = 500

# Mode 4 (rerank): "Hybrid top-50 -> local cross-encoder -> top-k" (PLAN.md,
# verbatim depth).
RERANK_POOL_K = 50

# Mode 5 (parent): candidate chunks before collapsing to distinct parent
# items. Needs to be well above k since several chunks routinely share one
# item_id and collapse into a single result.
PARENT_CANDIDATE_K = 50

# Mode 6 (temporal): candidate pool before time-diversified sampling. Wider
# than the other modes' pools on purpose -- diversifying across months only
# works if the pool actually spans many months to begin with.
TEMPORAL_CANDIDATE_K = 200


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    text: str
    distance: float  # lower is always better, across every mode (see below)
    item_id: int
    headline: str | None
    canonical_url: str | None
    source_slug: str
    published_at: str | None


def retrieve(
    query: str,
    k: int,
    mode: str,
    *,
    after: str | None = None,
    before: str | None = None,
    source: str | None = None,
    recency: bool = False,
    conn: sqlite3.Connection | None = None,
    embedder: Embedder | None = None,
    config: Config | None = None,
    as_of: datetime | None = None,
    reranker: Reranker | None = None,
) -> list[Chunk]:
    if mode not in ("dense", "sparse", "hybrid", "rerank", "parent", "temporal"):
        raise NotImplementedError(f"mode={mode!r} not wired.")

    config = config or load_config()
    conn = conn or db.get_connection(config.db_path)
    db.ensure_chunk_vec(conn, config.embedding_dim)

    if mode == "dense":
        embedder = embedder or GemmaEmbedder(config)
        scored = _dense_scored_ids(query, k, conn, embedder, after, before, source)
        chunks = _chunks_from_scored_ids(conn, scored)
    elif mode == "sparse":
        scored = _sparse_scored_ids(query, k, conn, after, before, source)
        chunks = _chunks_from_scored_ids(conn, scored)
    elif mode == "hybrid":
        scored = _hybrid_scored_ids(query, HYBRID_CANDIDATE_K, conn, embedder, after, before, source, config)
        chunks = _chunks_from_scored_ids(conn, scored[:k])
    elif mode == "rerank":
        # Mode 4: hybrid's top-50 candidates re-scored by a cross-encoder
        # that sees the actual (query, chunk text) pair, not just two
        # independent single-vector similarities. Largest expected gain
        # per PLAN.md, at the cost of one extra local model pass.
        scored = _hybrid_scored_ids(query, RERANK_POOL_K, conn, embedder, after, before, source, config)
        candidates = _chunks_from_scored_ids(conn, scored)
        reranker = reranker or CrossEncoderReranker(config)
        chunks = _rerank_chunks(candidates, query, reranker, k)
    elif mode == "parent":
        # Mode 5: rank on chunks (the embedding unit) but return whole items
        # (CLAUDE.md data model: the item is "the unit of meaning") -- a
        # judge or an LLM synthesizing an answer wants the full story, not
        # one ~512-token fragment of it.
        scored = _hybrid_scored_ids(query, PARENT_CANDIDATE_K, conn, embedder, after, before, source, config)
        candidates = _chunks_from_scored_ids(conn, scored)
        chunks = _dedupe_to_parent_items(conn, candidates, k)
    else:  # temporal
        # Mode 6: the longitudinal thesis. Hybrid ranks by relevance alone,
        # which tends to cluster around whichever months are highest-volume
        # or most topical right now; this deliberately spreads the result
        # set across every month present in the candidate pool instead.
        scored = _hybrid_scored_ids(query, TEMPORAL_CANDIDATE_K, conn, embedder, after, before, source, config)
        candidates = _chunks_from_scored_ids(conn, scored)
        chunks = _time_diversify(candidates, k)

    if recency:
        # Temporal's whole point is a deliberately non-distance-sorted
        # order; re-sorting by recency-adjusted distance afterward would
        # silently collapse it back into an ordinary ranked list -- exactly
        # the "one irreversible mistake" CLAUDE.md invariant 2 warns about,
        # just arrived at from the composition of two features instead of
        # one. So temporal still gets recency-adjusted distances (useful
        # for display/debugging) but keeps its own order.
        chunks = _apply_recency(
            chunks, as_of or datetime.now(timezone.utc), config, resort=(mode != "temporal")
        )
    return chunks


def _hybrid_scored_ids(
    query: str,
    pool_k: int,
    conn: sqlite3.Connection,
    embedder: Embedder | None,
    after: str | None,
    before: str | None,
    source: str | None,
    config: Config,
) -> list[tuple[int, float]]:
    """RRF fusion of dense + sparse, each fetched at `pool_k` -- the shared
    candidate-generation step behind hybrid and every mode built on top of it
    (rerank, parent, temporal). Returns the full fused ranking, already
    sorted best-first; callers slice or otherwise post-process from there.
    """
    embedder = embedder or GemmaEmbedder(config)
    dense = _dense_scored_ids(query, pool_k, conn, embedder, after, before, source)
    sparse = _sparse_scored_ids(query, pool_k, conn, after, before, source)
    fused = reciprocal_rank_fusion([[cid for cid, _ in dense], [cid for cid, _ in sparse]])
    # RRF score is higher-is-better; negate so "lower distance = better"
    # holds uniformly across every mode, not just dense/sparse.
    return [(cid, -score) for cid, score in fused]


def _rerank_chunks(candidates: list[Chunk], query: str, reranker: Reranker, k: int) -> list[Chunk]:
    if not candidates:
        return []
    scores = reranker.score(query, [c.text for c in candidates])
    # Reranker score is higher-is-better (cross-encoder convention); negate
    # into distance so lower-is-better holds here too. Ties broken by
    # chunk_id ascending -- CLAUDE.md invariant 7.
    rescored = sorted(
        zip(candidates, scores), key=lambda pair: (-pair[1], pair[0].chunk_id)
    )
    return [dataclasses.replace(c, distance=-s) for c, s in rescored[:k]]


def _dedupe_to_parent_items(
    conn: sqlite3.Connection, chunks: list[Chunk], k: int
) -> list[Chunk]:
    """Collapses ranked chunks down to their parent item, keeping only each
    item's best-ranked (first-seen) chunk, then swaps in the item's full
    text. `chunks` is already sorted best-first, so first-seen == best.
    """
    seen_items: set[int] = set()
    picked: list[Chunk] = []
    for c in chunks:
        if c.item_id in seen_items:
            continue
        seen_items.add(c.item_id)
        picked.append(c)
        if len(picked) == k:
            break
    if not picked:
        return []

    ids = [c.item_id for c in picked]
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, text FROM item WHERE id IN ({placeholders})", ids
    ).fetchall()
    text_by_item = {row[0]: row[1] for row in rows}
    return [dataclasses.replace(c, text=text_by_item.get(c.item_id, c.text)) for c in picked]


def _time_diversify(chunks: list[Chunk], k: int) -> list[Chunk]:
    """Round-robins across the year-month buckets present in `chunks`,
    visiting buckets in a fixed chronological order each round (oldest
    first; missing dates bucket last since "unknown" sorts after any real
    "YYYY-MM" string), so the result covers the whole date range in the
    candidate pool rather than being dominated by whichever months rank
    highest by relevance alone -- PLAN.md mode 6, the longitudinal thesis.
    Order within each bucket is preserved from `chunks` (already best-first
    from hybrid fusion), so this trades pure relevance rank for date spread,
    not for quality within a given month.
    """
    buckets: dict[str, list[Chunk]] = {}
    for c in chunks:
        ym = (c.published_at or "unknown")[:7]
        buckets.setdefault(ym, []).append(c)
    order = sorted(buckets)

    result: list[Chunk] = []
    i = 0
    while len(result) < k and any(buckets[ym] for ym in order):
        ym = order[i % len(order)]
        if buckets[ym]:
            result.append(buckets[ym].pop(0))
        i += 1
    return result


def _apply_recency(
    chunks: list[Chunk], as_of: datetime, config: Config, resort: bool = True
) -> list[Chunk]:
    """Post-retrieval multiplier only — CLAUDE.md invariant 2: never baked
    into the embedding or index. Applied uniformly after whichever mode ran,
    so it composes with every mode rather than being mode-specific.
    distance *= (1 + decay_per_day * age_in_days); older items' distances
    grow, so they sort worse, without changing what was actually retrieved.

    `resort=False` (used by mode="temporal") still adjusts each chunk's
    distance but leaves list order untouched -- temporal's order is a
    deliberate time-diversified sample, not a distance ranking, and
    re-sorting it would silently discard the diversification.
    """
    adjusted = []
    for c in chunks:
        age_days = 0.0
        if c.published_at:
            try:
                published = datetime.fromisoformat(c.published_at.replace("Z", "+00:00"))
                age_days = max((as_of - published).total_seconds() / 86400, 0.0)
            except ValueError:
                pass
        new_distance = c.distance * (1 + config.retrieval_recency_decay_per_day * age_days)
        adjusted.append(dataclasses.replace(c, distance=new_distance))
    if resort:
        adjusted.sort(key=lambda c: (c.distance, c.chunk_id))
    return adjusted


def _resolve_source_id(conn: sqlite3.Connection, source_slug: str) -> int:
    row = conn.execute("SELECT id FROM source WHERE slug = ?", (source_slug,)).fetchone()
    return row[0] if row else -1  # sentinel: matches no real source_id


def _filter_sql(
    conn: sqlite3.Connection, after: str | None, before: str | None, source: str | None
) -> tuple[str, list]:
    """WHERE fragment + params filtering on chunk_vec's own aux columns
    (source_id, published_at) — avoids joining all the way to item/issue/
    source just to filter, since chunk_vec already carries both directly.
    """
    clauses = []
    params: list = []
    if source is not None:
        clauses.append("cv.source_id = ?")
        params.append(_resolve_source_id(conn, source))
    if after is not None:
        clauses.append("cv.published_at >= ?")
        params.append(after)
    if before is not None:
        clauses.append("cv.published_at <= ?")
        params.append(before)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _dense_scored_ids(
    query: str,
    k: int,
    conn: sqlite3.Connection,
    embedder: Embedder,
    after: str | None = None,
    before: str | None = None,
    source: str | None = None,
) -> list[tuple[int, float]]:
    query_vec = embedder.encode([query], kind="query")[0]
    extra_where, extra_params = _filter_sql(conn, after, before, source)
    has_filters = bool(extra_where)
    candidate_k = min(k * CANDIDATE_OVERFETCH_MULTIPLIER, MAX_CANDIDATES) if has_filters else k

    # sqlite-vec 0.1.9 (verified by bisecting real failures, not assumed) is
    # strict here: `k = ?` must live inside the vec0 clause itself (an outer
    # LIMIT isn't a valid KNN query shape); the deterministic tie-break
    # column must come from a *joined real table*, not the KNN subquery's own
    # chunk_id ("Only a single 'ORDER BY distance' clause is allowed" if you
    # do); and auxiliary-column filters (source_id, published_at) can't sit
    # inside the KNN WHERE clause at all ("illegal WHERE constraint... in a
    # KNN query") — hence the second chunk_vec join for filtering, and the
    # over-fetch since there's no native pre-filtered KNN to lean on.
    rows = conn.execute(
        f"""
        SELECT chunk.id, nn.distance FROM (
            SELECT chunk_id, distance FROM chunk_vec
            WHERE embedding MATCH ? AND k = ?
        ) nn
        JOIN chunk ON chunk.id = nn.chunk_id
        JOIN chunk_vec cv ON cv.chunk_id = nn.chunk_id
        WHERE 1=1{extra_where}
        ORDER BY nn.distance ASC, chunk.id ASC
        LIMIT ?
        """,
        (query_vec.astype("float32").tobytes(), candidate_k, *extra_params, k),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _sanitize_fts_query(text: str) -> str:
    """Free text -> a safe FTS5 MATCH expression.

    FTS5's MATCH syntax is a small query language, not literal text — an
    apostrophe (e.g. "Vance's") is a syntax character, not punctuation, and
    raises a syntax error un-escaped. Each surviving token gets wrapped in
    double quotes (forces literal matching, sidesteps the syntax parser) and
    OR-joined, dropping stopwords so near-universal terms don't dilute the
    real content words. An AND join is more precise but returns zero rows
    the moment any one word (even an incidental one) isn't present verbatim
    in some chunk — too brittle for full-sentence queries.
    """
    tokens = re.findall(r"\w+", text)
    tokens = [t for t in tokens if len(t) > 1 and t.lower() not in _FTS_STOPWORDS]
    if not tokens:
        tokens = re.findall(r"\w+", text) or ["_"]  # degenerate: query was all stopwords
    escaped = [t.replace('"', '""') for t in tokens]
    return " OR ".join(f'"{t}"' for t in escaped)


def _sparse_scored_ids(
    query: str,
    k: int,
    conn: sqlite3.Connection,
    after: str | None = None,
    before: str | None = None,
    source: str | None = None,
) -> list[tuple[int, float]]:
    # bm25() is negative in SQLite's FTS5, more negative = better match, so
    # ascending order already means "best first" (verified against the
    # installed FTS5, not assumed). No over-fetch needed here — FTS5 filters
    # exactly, unlike vec0's KNN.
    extra_where, extra_params = _filter_sql(conn, after, before, source)
    rows = conn.execute(
        f"""
        SELECT chunk_fts.rowid, bm25(chunk_fts) AS score
        FROM chunk_fts
        JOIN chunk_vec cv ON cv.chunk_id = chunk_fts.rowid
        WHERE chunk_fts MATCH ?{extra_where}
        ORDER BY score ASC, chunk_fts.rowid ASC
        LIMIT ?
        """,
        (_sanitize_fts_query(query), *extra_params, k),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _chunks_from_scored_ids(
    conn: sqlite3.Connection, scored: list[tuple[int, float]]
) -> list[Chunk]:
    if not scored:
        return []
    ids = [cid for cid, _ in scored]
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""
        SELECT chunk.id, chunk.text, item.id, item.headline, item.canonical_url,
               source.slug, chunk_vec.published_at
        FROM chunk
        JOIN item ON item.id = chunk.item_id
        JOIN chunk_vec ON chunk_vec.chunk_id = chunk.id
        JOIN source ON source.id = chunk_vec.source_id
        WHERE chunk.id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    by_id = {row[0]: row for row in rows}

    result = []
    for chunk_id, distance in scored:
        row = by_id.get(chunk_id)
        if row is None:
            continue  # chunk existed at retrieval time but not at fetch time; skip
        result.append(
            Chunk(
                chunk_id=row[0], text=row[1], distance=distance, item_id=row[2],
                headline=row[3], canonical_url=row[4], source_slug=row[5],
                published_at=row[6],
            )
        )
    return result
