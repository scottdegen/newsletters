# PLAN.md — Current-Events RAG Corpus

## Goal

Build a local RAG system over a longitudinal corpus of newsletter content about
current events. The value is **time depth**: the ability to ask how coverage of a
topic evolved, not just what the latest issue said.

Phase 1 builds the corpus from public RSS/Substack archives. This backfill is
both the prototype corpus and the eval set. Gmail ingestion comes later.

## Stack

Python 3.11+ · SQLite (sqlite-vec, FTS5) · EmbeddingGemma-300M · uv for deps.
No cloud services in Phase 1–3. No Docker.

## Non-Goals

- No Gmail, OAuth, or daemon until Phase 4.
- No web UI. CLI only.
- No LLM answer synthesis until retrieval quality is measured.
- No paid-tier content. Free-tier truncation is accepted for now.

---

## Data Model

Four levels. The **item** is the unit of meaning; the **chunk** is the unit of embedding.

```
source  (a publication)
  └── issue  (one post / one email, dated)
        └── item  (one story or section — has headline, date, canonical URL)
              └── chunk  (embedding unit, ~512 tokens, ~15% overlap)
```

For long-form Substacks, `issue` and `item` are 1:1. For curated digests
(Axios-style), one issue splits into 5–8 items. Split on `<h2>` / `<hr>` /
bold-headline patterns, per source.

Raw content is stored on disk, content-addressed and date-partitioned:

```
corpus/raw/2026/08/16/<source_slug>/<sha256>.html
corpus/raw/2026/08/16/<source_slug>/<sha256>.md    # normalized
corpus/corpus.db
```

**Rationale:** re-parsing and re-embedding will happen repeatedly (chunker
changes, dimension changes, template changes). Re-deriving from local raw is
cheap; re-fetching is not.

Dedup key: canonical URL + content SHA-256.

---

## Schema

```sql
CREATE TABLE source (
  id INTEGER PRIMARY KEY, slug TEXT UNIQUE, name TEXT,
  feed_url TEXT, home_url TEXT, platform TEXT   -- substack|beehiiv|ghost|rss
);

CREATE TABLE issue (
  id INTEGER PRIMARY KEY, source_id INTEGER REFERENCES source(id),
  canonical_url TEXT UNIQUE, title TEXT,
  published_at TEXT,          -- ISO 8601 UTC
  raw_path TEXT, content_hash TEXT, fetched_at TEXT
);

CREATE TABLE item (
  id INTEGER PRIMARY KEY, issue_id INTEGER REFERENCES issue(id),
  ordinal INTEGER, headline TEXT, text TEXT, canonical_url TEXT
);

CREATE TABLE chunk (
  id INTEGER PRIMARY KEY, item_id INTEGER REFERENCES item(id),
  ordinal INTEGER, text TEXT, token_count INTEGER
);

CREATE VIRTUAL TABLE chunk_fts USING fts5(
  text, content='chunk', content_rowid='id'
);

-- Verify vec0 syntax against the pinned sqlite-vec version before writing.
CREATE VIRTUAL TABLE chunk_vec USING vec0(
  chunk_id INTEGER PRIMARY KEY,
  ym TEXT PARTITION KEY,        -- '2026-08'
  embedding FLOAT[256],
  +source_id INTEGER,
  +published_at TEXT
);
```

`ym` as partition key lets sqlite-vec skip whole segments on date-filtered
queries. This is the largest latency lever in the system.

---

## Phases

### Phase 1 — Backfill ingestion
- [ ] `sources.yaml` with 10–15 newsletters (slug, feed URL, platform).
- [ ] Feed fetcher: Substack `/feed`, Ghost/Beehiiv RSS. Respect `robots.txt`,
      rate-limit to ~1 req/sec, cache ETag/Last-Modified.
- [ ] Substack archive endpoint for deeper history than `/feed` returns
      (`/api/v1/archive?sort=new&offset=N&limit=12`) — verify it still works
      before depending on it; fall back to feed-only if not.
- [ ] HTML → normalized Markdown. Strip nav, footers, subscribe CTAs, share
      buttons, images.
- [ ] Unwrap tracking redirects in links (`link.mail.beehiiv.com`,
      Substack redirect URLs) via HEAD + follow, cached in a `link_map` table.
- [ ] Item segmentation with a per-source strategy; default = whole issue.
- [ ] Write raw + normalized to disk; populate `source`/`issue`/`item`.

**Done when:** `python -m corpus ingest` is idempotent (re-running adds zero
rows), and `corpus stats` reports ≥1,000 items across ≥10 sources spanning
≥6 months.

### Phase 2 — Index
- [ ] Chunker: ~512 tokens, ~15% overlap, never crossing an item boundary.
- [ ] EmbeddingGemma-300M via sentence-transformers. **Task prefixes matter** —
      query and document prefixes differ. Define both as constants in ONE
      module so index-time and query-time cannot drift.
- [ ] Embed at 768 dims, store truncated to 256 (Matryoshka). Keep the
      dimension in config; make re-indexing a single command.
- [ ] Populate `chunk`, `chunk_fts`, `chunk_vec`.
- [ ] Batch embed with progress output; resumable if interrupted.

**Done when:** `corpus index` completes on the full backfill and
`corpus search "<text>"` returns plausible results in <500ms.

### Phase 2.5 — Testing architecture + ground truth

Build this **before** Phase 3 retrieval work, with only mode 1 wired. Every
later change then gets measured the moment it lands.

**Seams.** Keep pure functions pure — `parse(html)→items`, `chunk(item)→chunks`,
`fuse(ranked_lists)→ranking` take structures and return structures. HTTP,
embedding, and SQLite are the only I/O. Pure functions get millisecond unit
tests with no model loaded.

- [ ] `tests/fixtures/` — ~20 real HTML issues committed to the repo, covering
      every source template supported. Tests never hit the network. When a
      publisher changes markup, add the new fixture; the old ones prove no
      regression.
- [ ] `Embedder` protocol with two implementations:
      ```python
      class Embedder(Protocol):
          def encode(self, texts: list[str],
                     kind: Literal["query", "doc"]) -> np.ndarray: ...
      ```
      `HashEmbedder` returns deterministic vectors from a text hash — the full
      retrieval pipeline (RRF, filters, parent expansion, DB round-trips) runs
      in <1s with no 300M model in memory. Real model only in tests marked
      `@pytest.mark.slow`.
- [ ] **Index versioning.** In a `meta` table:
      `index_version = sha256(model, dim, chunk_size, overlap, prefixes,
      parser_version)[:12]`. Every eval result is stamped with it. This makes
      eval caching correct, prevents comparing runs across incompatible
      indexes, and answers "what changed between these two scores."
- [ ] **Determinism.** Fixed seeds. Break RRF ties by `chunk_id` ascending —
      never leave ties to dict ordering, or two runs of identical config
      produce different Recall@10 and you chase a phantom improvement.

**Ground truth — pooled relevance judgment.** Do NOT build `queries.yaml` by
running search and marking what looks good; that defines truth as "what the
default mode returns" and guarantees it wins by construction. Instead:

1. Write all 40 queries first, before looking at any results.
2. For each query, run **every** mode and take the union of their top-10.
3. Shuffle the pool so mode-of-origin is not visible.
4. Judge relevance on the shuffled pool.

- [ ] `corpus pool --queries evals/queries.yaml` emits shuffled judgment sheets.
- [ ] Judgments stored separately from queries, stamped with `index_version`.

**Statistics at n=40.** A 3-point Recall@10 difference is noise. Compare
**paired** (same queries across modes, per-query deltas), report a bootstrap
CI rather than a bare number, and run `scipy.stats.wilcoxon` on paired
per-query scores. A 12-point gap needs no statistics; a 4-point gap does —
knowing which case you're in is the point.

**Done when:** `pytest` runs green in <10s without loading a model, and
`corpus eval --mode dense` produces a stamped, reproducible number.

### Phase 3 — Retrieval modes + eval

Retrieval is a **pluggable mode** behind one interface, not a single path:

```python
def retrieve(query: str, k: int, mode: str, *, after=None, before=None,
             source=None, recency=False) -> list[Chunk]: ...
```

Implement modes 1–3 now, 4–6 after the eval harness exists.

| # | Mode | Mechanism | Purpose |
|---|------|-----------|---------|
| 1 | `dense` | EmbeddingGemma query embed → cosine over `chunk_vec`, top-k | Baseline |
| 2 | `sparse` | BM25 via FTS5, top-k | Baseline |
| 3 | `hybrid` | 1 + 2 at top-50 each, fused by RRF `Σ 1/(60+rank)` | Expected default |
| 4 | `rerank` | Hybrid top-50 → local cross-encoder → top-k | Largest expected gain |
| 5 | `parent` | Retrieve on chunks, return whole parent `item` | Coherence for synthesis |
| 6 | `temporal` | Hybrid + time-diversified sampling across months | The longitudinal thesis |

Notes per mode:

- **1 and 2 are not for use.** They exist so mode 3's win is demonstrable. A
  hybrid score with no baseline is a claim, not a finding.
- **3** — dense should win on paraphrase queries, lose on proper nouns, bill
  numbers, tickers, and names postdating the encoder's training data. That gap
  is the reason hybrid is the default.
- **4** — bge-reranker-v2-m3 or similar, local, ~100ms, $0.
- **5** — chunks are fragments; the `item` is the coherent story. Schema already
  has the parent link.
- **6** — recency weighting and time diversification are *opposites*, both
  needed. "What's the latest on X" wants recency; "how did coverage of X shift"
  wants deliberate sampling across the date range. Recency is always an
  **optional post-retrieval multiplier**, never baked into the embedding or the
  index — baking it in permanently destroys mode 6.

Tasks:
- [ ] Modes 1–3 behind the `retrieve()` interface.
- [ ] Filters: `--source`, `--after`, `--before`, `--recency`.
- [ ] `evals/queries.yaml`: 30–50 hand-written queries with known-relevant item
      IDs from the backfill. Tag each query by type: `entity`, `paraphrase`,
      `temporal`, `multi-source`.
- [ ] `corpus eval --mode <m>` reporting Recall@10 and MRR, **broken out by
      query tag** — the aggregate hides which mode fails where.
- [ ] Per-query failure table written to `evals/results/`. This is the writeup
      material, more than the aggregate score.
- [ ] Modes 4–6 once the harness works; re-run the full sweep on each.

**Done when:** hybrid beats both single-mode baselines on Recall@10, the
per-tag table is in the README, and re-running the sweep is one command.

**Eval cost discipline:** modes 1–6 are all local and cost $0 — sweep as often
as you like. Cost appears only when an LLM judge enters the loop. If answer
quality scoring is added later, run the judge on a fixed 20-query subset, cache
results by `(query_id, mode, index_version)`, and use a cheap model. Judge cost
routinely exceeds production cost by 5–10× at this stage.

### Phase 4 — Gmail (later; do not start until Phase 3 is done)
- Server-side Gmail filter routes newsletters to a `newsletters` label.
- Incremental sync via `users.history.list` + stored `historyId`; full
  date-range re-sync as fallback (history IDs expire after ~1 week, and the
  daemon will be offline sometimes).
- Dedup on RFC 5322 `Message-ID` header, not the Gmail API message id.
- Same parse → item → chunk path as Phase 1.

---

## Conventions

- All timestamps ISO 8601 UTC in the DB; format for display only at output.
- Config in `config.toml`; no hardcoded paths or model names in logic.
- Every stage is a separate CLI subcommand and independently re-runnable.
- Log to stderr, data to stdout, so commands compose.
- Tests for: chunk boundaries, dedup idempotence, link unwrapping, RRF math.
- Pin sqlite-vec exactly — its API has changed across minor versions.

## Repo Layout

```
corpus/
  __main__.py      # CLI
  config.py
  sources.py       # feed discovery + fetch
  parse.py         # HTML → items
  links.py         # tracking-redirect unwrapping
  chunk.py
  embed.py         # prefixes live here, nowhere else
  embedder.py      # Embedder protocol; Gemma + HashEmbedder impls
  index.py
  retrieve.py      # retrieve() dispatcher + all modes
  fuse.py          # RRF, recency multiplier, time diversification
  rerank.py        # phase 3, mode 4
  db.py            # schema, migrations
  usage.py         # token/cost logging if an LLM is added
evals/
  queries.yaml     # tagged: entity | paraphrase | temporal | multi-source
  judgments.yaml   # pooled, stamped with index_version
  run.py
  results/
tests/
  fixtures/        # ~20 real HTML issues, committed
  test_parse.py test_chunk.py test_fuse.py test_dedup.py test_links.py
sources.yaml
config.toml
PLAN.md
README.md
```