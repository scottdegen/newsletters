# CLAUDE.md

Local RAG over a longitudinal newsletter corpus. Read `PLAN.md` for the roadmap.
This file is the invariants — it wins over anything else if they conflict.

## Stack

Python 3.11+ · SQLite (sqlite-vec, FTS5) · EmbeddingGemma-300M · uv · pytest.
No cloud services. No Docker. No web UI. CLI only.

## Invariants — never change these without being asked

1. **Embedding prefixes live in `corpus/embed.py` and nowhere else.**
   EmbeddingGemma uses different task prefixes for queries vs. documents.
   Never inline a prefix string at a call site. Drift here degrades retrieval
   silently rather than loudly, so it will not show up as a failing test.

2. **Recency is a post-retrieval multiplier, never in the embedding or index.**
   Baking it in permanently destroys the time-diversified retrieval mode, which
   is the point of the project. This is the one irreversible mistake available.

3. **`retrieve()` keeps its signature.** New retrieval strategies are new
   `mode` values, not new functions or new parameters:
   ```python
   def retrieve(query: str, k: int, mode: str, *, after=None, before=None,
                source=None, recency=False) -> list[Chunk]: ...
   ```

4. **All timestamps are ISO 8601 UTC in the database.** Convert for display
   only, at the output layer.

5. **Raw source content is written to disk before parsing.** Content-addressed
   under `corpus/raw/YYYY/MM/DD/<slug>/<sha256>.html`. Re-parsing and
   re-embedding happen constantly; re-fetching is expensive and sometimes
   impossible.

6. **Every stage is independently re-runnable and idempotent.** Re-running
   ingest adds zero rows. Re-running index is a full rebuild or a no-op, never
   a partial mutation.

7. **Retrieval is deterministic.** Fixed seeds; RRF ties break by `chunk_id`
   ascending. Non-determinism makes eval numbers meaningless.

## Data model

`source → issue → item → chunk`. The **item** is the unit of meaning (one
story, with a headline and canonical URL). The **chunk** is the unit of
embedding. Chunks never cross item boundaries.

## Working style

- Read `PLAN.md` for the current phase only. Do not implement ahead of it.
- Ask before adding a dependency, changing the schema, or introducing a
  framework. LangChain / LlamaIndex are out of scope — the point is to build
  the pipeline directly.
- Small commits, one concern each.
- Prefer editing existing files over creating new ones.
- Log to stderr, data to stdout, so commands compose.
- Config in `config.toml`. No hardcoded paths, model names, or dimensions in
  logic.
- Pin `sqlite-vec` exactly. Its API has changed across minor versions — verify
  `vec0` syntax against the installed version rather than assuming.

## Testing

- Pure functions (`parse`, `chunk`, `fuse`) get fast unit tests with no model
  loaded. Use `HashEmbedder`, not the real model.
- Fixtures in `tests/fixtures/` are real committed HTML. Tests never hit the
  network.
- `pytest` must run green in under 10 seconds. Real-model tests are marked
  `@pytest.mark.slow` and excluded by default.
- Never construct eval ground truth from search output. Judgments are pooled
  across all modes and shuffled before judging — see PLAN.md Phase 2.5.

## Commands

```
corpus ingest              # fetch feeds → raw disk → source/issue/item
corpus index               # chunk → embed → chunk_vec + chunk_fts
corpus search "<q>" --mode hybrid
corpus eval --mode <m>     # Recall@10, MRR, broken out by query tag
corpus stats
pytest                     # fast tests only
pytest -m slow             # real model
```