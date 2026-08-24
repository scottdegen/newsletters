# newsletters

Local RAG over a longitudinal newsletter corpus about current events. See
`PLAN.md` for the roadmap and `CLAUDE.md` for the invariants. No cloud
services, no Docker, no web UI — CLI only.

## Corpus (as of this writeup)

10 sources, 2,879 issues, 4,194 items, 16,567 chunks, spanning 2026-02-16 to
2026-08-16.

## Privacy note

`sources.yaml`, `newsletter_senders.yaml`, `tests/fixtures/`, and
`evals/judgments*.yaml` are gitignored and not in this repo — they'd reveal
exactly which newsletters/publications are being read. `sources.example.yaml`
and `newsletter_senders.example.yaml` show the expected shape; copy one to
the real filename and fill in your own feeds. Fixture-dependent tests skip
gracefully (rather than fail) when `tests/fixtures/` is absent.

## Commands

```
corpus ingest              # fetch feeds -> raw disk -> source/issue/item
corpus index                # chunk -> embed -> chunk_vec + chunk_fts
corpus search "<q>" --mode hybrid
corpus pool                 # regenerate the shuffled judgment pool
corpus eval --mode <m>       # Recall@10, MRR, broken out by query tag
corpus stats
pytest                       # fast tests only (<10s, no model loaded)
pytest -m slow                # real-model tests
```

## Retrieval modes

All six modes from PLAN.md are wired behind one `retrieve()` interface:

| # | mode | mechanism |
|---|------|-----------|
| 1 | `dense` | EmbeddingGemma query embed -> cosine over `chunk_vec`, top-k |
| 2 | `sparse` | BM25 via FTS5, top-k |
| 3 | `hybrid` | dense + sparse at top-50 each, fused by RRF |
| 4 | `rerank` | hybrid top-50 -> local cross-encoder (ms-marco-MiniLM-L-6-v2) -> top-k |
| 5 | `parent` | hybrid candidates, deduped to one result per parent item, full item text returned |
| 6 | `temporal` | hybrid candidates, round-robin sampled across the months present in the pool |

`--source`, `--after`, `--before`, and `--recency` compose with every mode.
Recency is always a post-retrieval distance multiplier — never baked into the
embedding or index (CLAUDE.md invariant 2), since that would permanently
destroy `temporal`'s time-diversified sampling, which is the whole point of
the project.

## Eval results

**Ground truth coverage note:** relevance judgments are hand-entered by the
maintainer against a pooled, shuffled candidate set (never constructed from
search output — see PLAN.md's Phase 2.5 methodology). As of this writeup,
90 of 852 pooled entries are judged, covering 8 of 40 queries with at least
one relevant item (`n=8` below). That's still thin — a single query swinging
is enough to move the aggregate — so treat these as directional, not final.

| mode | n | Recall@10 | 95% CI | MRR |
|---|---|---|---|---|
| dense | 8 | 0.634 | [0.41, 0.85] | 0.688 |
| sparse | 8 | 0.411 | [0.26, 0.55] | 0.617 |
| hybrid | 8 | 0.647 | [0.47, 0.81] | 0.608 |
| rerank | 8 | 0.277 | [0.07, 0.52] | 0.239 |
| parent | 8 | 0.742 | [0.57, 0.90] | 0.609 |
| temporal | 8 | 0.355 | [0.15, 0.68] | 0.323 |

**Digest-issue segmentation (splitting a roundup-style issue into one item
per topic, instead of one diffuse item per issue) has since shipped**, aimed
at `rerank`'s known weakness: a cross-encoder scores chunks by raw topical
density, so a roundup chunk that mentions a query's topic in passing can
out-score a real single-story chunk that's actually about it. Whether that
fix helped isn't resolved by the table above — the two queries that touch
split issues also got *stricter* ground truth (retrieval now has to land on
the specific split-out section, not any chunk from the whole roundup), and
with only 8 scored queries total that's enough on its own to move every
mode's aggregate. Confirming the real effect needs more judged queries
(PLAN.md's original target was 30-50) to average out that noise — not
something this eval run alone can settle.

**`temporal`'s lower Recall@10 is expected, not necessarily a regression.**
It deliberately trades relevance rank for date-range spread — Recall@10
isn't the metric that would show whether it's succeeding at its actual job
(covering the full date range for "how did coverage of X shift" queries).
Judging that properly needs a metric that isn't in this eval harness yet
(e.g. distinct months touched, or judgments that account for the shift
question the query is actually asking) rather than more Recall@10 samples.

## Testing

Pure functions (`chunk`, `fuse`, `_sanitize_fts_query`, `_time_diversify`)
get fast unit tests with no model loaded — `HashEmbedder`/`HashReranker`
stand in for the real models. Fixture-dependent tests use real HTML under
`tests/fixtures/`, which is gitignored (see Privacy note above) and skip
gracefully when absent; tests never hit the network. `pytest` runs the fast
suite in under 10 seconds; real-model tests are marked `@pytest.mark.slow`.
