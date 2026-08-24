from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from corpus import chunk as chunk_mod
from corpus import db, embed, eval as eval_mod, links, parse, pool
from corpus.config import Config, load_config
from corpus.retrieve import retrieve
from corpus.sources import RateLimiter, discover_issues, fetch
from corpus.versioning import compute_index_version

INDEX_BATCH_SIZE = 200

logger = logging.getLogger("corpus")


def _date_parts(iso_str: str | None, fallback: datetime) -> tuple[str, str, str]:
    if iso_str and len(iso_str) >= 10:
        y, m, d = iso_str[:10].split("-")
        return y, m, d
    return f"{fallback.year:04d}", f"{fallback.month:02d}", f"{fallback.day:02d}"


def _write_raw(raw_dir: Path, slug: str, date_parts: tuple[str, str, str],
               content_hash: str, html: str, markdown: str) -> Path:
    y, m, d = date_parts
    out_dir = raw_dir / y / m / d / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{content_hash}.html").write_text(html, encoding="utf-8")
    (out_dir / f"{content_hash}.md").write_text(markdown, encoding="utf-8")
    return out_dir / f"{content_hash}.html"


def cmd_ingest(config: Config, sources_path: Path, since: str | None = None) -> None:
    sources = yaml.safe_load(sources_path.read_text())["sources"]
    conn = db.get_connection(config.db_path)
    session = requests.Session()
    limiter = RateLimiter(config.rate_limit_per_sec)

    new_issues = 0
    seen_issues = 0
    errors = 0

    for source in sources:
        slug = source["id"]
        source_id = db.upsert_source(
            conn, slug, source["name"], source["feed_url"],
            source.get("home_url"), source["platform"],
        )
        logger.info("[%s] discovering issues", slug)
        try:
            discovered = list(discover_issues(source, config, session, limiter, since))
        except Exception:
            logger.exception("[%s] failed to discover issues", slug)
            errors += 1
            continue

        for issue in discovered:
            if db.issue_exists(conn, issue.canonical_url):
                seen_issues += 1
                continue
            try:
                resp = fetch(issue.canonical_url, config, session, limiter)
                html = links.unwrap_html_links(resp.text, config, session, limiter, conn)
                markdown = parse.parse(html)
                content_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()
                fetched_at = datetime.now(timezone.utc).isoformat()
                raw_path = _write_raw(
                    config.raw_dir, slug,
                    _date_parts(issue.published_at, datetime.now(timezone.utc)),
                    content_hash, html, markdown,
                )
                issue_id, inserted = db.insert_issue(
                    conn, source_id, issue.canonical_url, issue.title,
                    issue.published_at, str(raw_path), content_hash, fetched_at,
                )
                if inserted:
                    for ordinal, seg in enumerate(parse.segment(html)):
                        db.insert_item(
                            conn, issue_id, ordinal, seg.headline or issue.title,
                            seg.text, issue.canonical_url,
                        )
                    new_issues += 1
                    logger.info("[%s] new: %s", slug, issue.title)
                else:
                    seen_issues += 1
            except Exception:
                logger.exception("[%s] failed on %s", slug, issue.canonical_url)
                errors += 1
                continue

        conn.commit()

    print(json.dumps({"new_issues": new_issues, "seen_issues": seen_issues, "errors": errors}))


def cmd_index(config: Config) -> None:
    conn = db.get_connection(config.db_path)
    db.ensure_chunk_vec(conn, config.embedding_dim)

    # Written on every invocation, not just when new items land, so meta
    # always reflects the current config. Note: this does NOT detect or
    # rebuild items chunked under a stale config — get_unindexed_items()
    # only sees items with zero chunks. Changing chunk/embedding config
    # requires manually clearing chunk/chunk_vec before re-running index.
    index_version = compute_index_version(config)
    db.set_meta(conn, "index_version", index_version)
    conn.commit()

    items = db.get_unindexed_items(conn)
    if not items:
        print(json.dumps({
            "items_indexed": 0, "chunks_created": 0, "index_version": index_version
        }))
        return

    logger.info("indexing %d unindexed items", len(items))
    embedder = embed.GemmaEmbedder(config)

    items_indexed = 0
    chunks_created = 0
    for batch_start in range(0, len(items), INDEX_BATCH_SIZE):
        batch = items[batch_start : batch_start + INDEX_BATCH_SIZE]

        flat: list[tuple[int, int, int, str | None, chunk_mod.TextChunk]] = []
        for item_id, text, source_id, published_at in batch:
            text_chunks = chunk_mod.chunk(
                text, config.chunk_target_tokens, config.chunk_overlap_ratio
            )
            for ordinal, tc in enumerate(text_chunks):
                flat.append((item_id, ordinal, source_id, published_at, tc))

        if flat:
            texts = [f[4].text for f in flat]
            embeddings = embedder.encode(texts, kind="doc", show_progress=True)
            for (item_id, ordinal, source_id, published_at, tc), vec in zip(flat, embeddings):
                ym = (published_at or "unknown")[:7]
                chunk_id = db.insert_chunk(conn, item_id, ordinal, tc.text, tc.token_count)
                db.insert_chunk_vec(
                    conn, chunk_id, ym, vec.astype("float32").tobytes(), source_id, published_at
                )
            chunks_created += len(flat)
        conn.commit()

        items_indexed += len(batch)
        logger.info(
            "indexed %d/%d items (%d chunks so far)", items_indexed, len(items), chunks_created
        )

    print(json.dumps({
        "items_indexed": items_indexed, "chunks_created": chunks_created,
        "index_version": index_version,
    }))


def cmd_search(
    config: Config, query_text: str, k: int, mode: str,
    after: str | None, before: str | None, source: str | None, recency: bool,
) -> None:
    results = retrieve(
        query_text, k, mode,
        after=after, before=before, source=source, recency=recency, config=config,
    )
    for r in results:
        print(f"[{r.distance:.4f}] {r.headline}  ({r.canonical_url})")
        print(f"  {r.text[:200]}...")


def cmd_pool(config: Config, queries_path: Path, output_path: Path, k: int) -> None:
    queries = yaml.safe_load(queries_path.read_text())["queries"]
    old_sheet = yaml.safe_load(output_path.read_text()) if output_path.exists() else None

    sheet = pool.build_pool(config, queries, k=k)
    sheet = pool.merge_judgments(sheet, old_sheet)
    carried = sum(1 for j in sheet["judgments"] if j["relevant"] is not None)

    output_path.write_text(yaml.safe_dump(sheet, sort_keys=False, allow_unicode=True))
    print(json.dumps({
        "queries": len(queries), "pooled_entries": len(sheet["judgments"]),
        "judgments_carried_forward": carried,
        "index_version": sheet["index_version"], "output": str(output_path),
    }))


def cmd_eval(
    config: Config, queries_path: Path, judgments_path: Path, mode: str, k: int
) -> None:
    queries = yaml.safe_load(queries_path.read_text())["queries"]
    if not judgments_path.exists():
        raise SystemExit(
            f"{judgments_path} not found. Run `corpus pool` first, then fill in its "
            "`relevant` fields (true/false) before evaluating."
        )
    sheet = yaml.safe_load(judgments_path.read_text())
    judgments = sheet["judgments"]

    current_version = compute_index_version(config)
    if sheet["index_version"] != current_version:
        raise SystemExit(
            f"judgments.yaml is stamped index_version={sheet['index_version']!r} but the "
            f"current corpus is {current_version!r} — index_version exists precisely to "
            "catch this: these two are not comparable. Re-run `corpus pool` against the "
            "current index before evaluating."
        )

    judged_count = sum(1 for j in judgments if j["relevant"] is not None)
    if judged_count == 0:
        raise SystemExit(
            f"No judged entries in {judgments_path} — every `relevant` field is still null. "
            "Fill in at least some before running eval."
        )

    conn = db.get_connection(config.db_path)
    db.ensure_chunk_vec(conn, config.embedding_dim)
    embedder = embed.GemmaEmbedder(config)

    scores = eval_mod.per_query_scores(queries, judgments, mode, k, conn, embedder, config)
    summary = eval_mod.summarize(scores)

    result = {
        "mode": mode, "k": k, "index_version": current_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judged_entries": judged_count, "scored_queries": len(scores),
        "summary": summary,
    }
    results_dir = Path("evals/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{mode}_{current_version}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nwritten to {out_path}", file=sys.stderr)


def cmd_stats(config: Config) -> None:
    conn = db.get_connection(config.db_path)
    n_sources = conn.execute("SELECT COUNT(*) FROM source").fetchone()[0]
    n_issues = conn.execute("SELECT COUNT(*) FROM issue").fetchone()[0]
    n_items = conn.execute("SELECT COUNT(*) FROM item").fetchone()[0]
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    date_range = conn.execute(
        "SELECT MIN(published_at), MAX(published_at) FROM issue"
    ).fetchone()
    per_source = conn.execute(
        """
        SELECT source.slug, COUNT(issue.id)
        FROM source LEFT JOIN issue ON issue.source_id = source.id
        GROUP BY source.slug ORDER BY source.slug
        """
    ).fetchall()

    print(f"sources: {n_sources}")
    print(f"issues:  {n_issues}")
    print(f"items:   {n_items}")
    print(f"chunks:  {n_chunks}")
    print(f"date range: {date_range[0]} .. {date_range[1]}")
    print("per source:")
    for slug, count in per_source:
        print(f"  {slug}: {count}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )

    parser = argparse.ArgumentParser(prog="corpus")
    parser.add_argument("--config", default="config.toml", type=Path)
    parser.add_argument("--sources", default="sources.yaml", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_parser = sub.add_parser("ingest")
    ingest_parser.add_argument(
        "--since", default=None,
        help="ISO 8601 date (e.g. 2026-02-16); skip issues published before it",
    )
    sub.add_parser("stats")
    sub.add_parser("index")
    search_parser = sub.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("-k", type=int, default=10)
    search_parser.add_argument(
        "--mode", default="dense",
        choices=["dense", "sparse", "hybrid", "rerank", "parent", "temporal"],
        help="retrieval mode (Phase 3, all 6 wired)",
    )
    search_parser.add_argument("--after", default=None, help="ISO 8601 date; only issues on/after")
    search_parser.add_argument("--before", default=None, help="ISO 8601 date; only issues on/before")
    search_parser.add_argument("--source", default=None, help="Source slug, e.g. my-source-slug")
    search_parser.add_argument(
        "--recency", action="store_true", help="Post-retrieval recency boost (never in the index)"
    )
    pool_parser = sub.add_parser("pool")
    pool_parser.add_argument("--queries", type=Path, default=Path("evals/queries.yaml"))
    pool_parser.add_argument("--output", type=Path, default=Path("evals/judgments.yaml"))
    pool_parser.add_argument("-k", type=int, default=10)
    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument(
        "--mode", default="dense",
        choices=["dense", "sparse", "hybrid", "rerank", "parent", "temporal"],
    )
    eval_parser.add_argument("--queries", type=Path, default=Path("evals/queries.yaml"))
    eval_parser.add_argument("--judgments", type=Path, default=Path("evals/judgments.yaml"))
    eval_parser.add_argument("-k", type=int, default=10)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "ingest":
        cmd_ingest(config, args.sources, args.since)
    elif args.command == "stats":
        cmd_stats(config)
    elif args.command == "index":
        cmd_index(config)
    elif args.command == "search":
        cmd_search(
            config, args.query, args.k, args.mode,
            args.after, args.before, args.source, args.recency,
        )
    elif args.command == "pool":
        cmd_pool(config, args.queries, args.output, args.k)
    elif args.command == "eval":
        cmd_eval(config, args.queries, args.judgments, args.mode, args.k)


if __name__ == "__main__":
    main()
