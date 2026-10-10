# Sessions and storage

Companion to §6. Records the storage design and why it looks like this.

## A session is a data boundary, not a compute boundary

A session is a titled, isolated workspace: its own documents, chunks, embeddings,
and evaluation runs. This is the single most important thing to get right, and it
is worth stating as a table because the alternative reading is so plausible:

| Shared across all sessions | Scoped to one session |
|---|---|
| `bge-m3` embedder (GPU 1) | documents |
| `bge-reranker-v2-m3` (GPU 1) | chunks |
| `qwen3:8b` via Ollama (GPU 0) | embeddings |
| the FastAPI process | evaluation runs |
| the 20GB store root | title and config |

Ten sessions cost ten indexes and share one set of loaded weights. The
alternative — a process per session — is not a design choice, it is impossible
here: 16GB of VRAM per T4 cannot hold ten copies of an embedder, and there is one
4-core CPU.

### Isolation is a feature

Two properties follow, and they are enforced rather than assumed:

1. **A query in session A can never read session B's data.** Every store call
   takes a `session_id`, and no store method has an "all sessions" default.
2. **That isolation is worth having.** An untrusted or exploratory corpus cannot
   contaminate the corpus you rely on. If a cross-session query is ever wanted it
   is a separate explicit operation over a named set of sessions, never an
   implicit default.

The API makes this structural rather than procedural: the session id is a **URL
path segment** on every route that touches corpus, index, query, or search state.
There is no `/api/index` and no `/api/query` to write by accident, and the
guarantee is checkable by reading the routes instead of auditing the store.

### Identity is generated, never the title

`session_id` is a generated id. Titles must be human-readable, editable, and
allowed to contain spaces and punctuation, so they cannot be directory names.
Renaming a session therefore never moves files — which removes a whole class of
data-loss bug from the rename path.

## Where things live

| Tier | Path | Survives session end |
|---|---|---|
| Durable store | `/kaggle/working/ragstore/` | **Yes**, up to 20GB |
| Scratch | `/kaggle/working/ragstore/tmp/` | Yes, but disposable |
| Session memory | RAM | **No** |
| Anything outside `/kaggle/working` | — | **No** |

The non-obvious consequence: **saved output does not come back at the same path.**
Kaggle re-attaches a previous run's output through the notebook editor's Input
pane, under `/kaggle/input/<notebook-slug>/`. The next session starts with the
store absent from `/kaggle/working` and present read-only under `/kaggle/input/`.

That drives the decision which actually protects the work:

> **Write-through, never write-on-shutdown.** The index is written to disk as soon
> as it is built, and every mutation is flushed immediately. Nothing waits for a
> clean exit.

The failure this avoids is documented, not hypothetical: pipelines have been lost
when a session ended and auto-save did not finish. A 12-hour session killed by
timeout is exactly that case, and it is also the case where the quota was already
spent.

This is why durability moved from Phase 8 into Phase 1. Retrofitting write-through
onto an index designed to live in RAM is precisely the refactor that produces the
lost-index failure.

## The invariant everything else depends on

```text
row i of vectors.npy  ==  line i of chunks.jsonl  ==  chunk.ordinal
```

`chunk.ordinal` is session-global, monotonically increasing, never reused. A
deleted chunk's ordinal stays burned.

Embeddings are a plain `.npy` — not `vectors.f16.npy` — precisely so this holds
and so the array opens with `np.load(path, mmap_mode="r")`. The dtype lives in
`manifest.json` and in the `.npy` header, never in the filename, because §5.5 may
switch it to fp32 and a rename would break every path that resolves the file.

That one choice buys three things at once:

- **Exact search without loading the index.** The matmul runs over memory-mapped
  pages; opening a 400MB index costs no resident RAM.
- **Incremental appends.** A new document's chunks append to the end of both
  files, so an incremental ingest rewrites nothing already indexed.
- **No database.** No SQLite, no FAISS, no server. A directory of files numpy can
  read is the entire storage engine, and it is the easiest thing to inspect when
  something is wrong.

A test asserts the invariant after every append. Corruption here is silent, and it
poisons every retrieval metric downstream.

## Deletion is the one place the invariant bends

Removing a document means removing chunks scattered through the middle of both
files. Deleting rows would break the ordinal invariant and force a rewrite, so
deletion is **logical, not physical**:

- Chunks are never removed from `chunks.jsonl` or `vectors.npy`. Rows are only
  appended.
- A `live` boolean on the chunk record marks dead rows. `doc_id` presence in
  `documents.jsonl` is the authoritative membership test.
- Dense search multiplies against all rows including dead ones and filters after.
  At this scale that is cheap, and it keeps the vector file append-only forever.
- BM25 filters to live rows before building postings, because that index is
  derived and a dead row left in it would be returned by every query forever.

Both counts are reported, because their divergence is otherwise invisible and
reads as storage drift:

| Field | Meaning |
|---|---|
| `chunk_count` | live rows — the effective count |
| `chunk_rows` | physical rows in the files |

So the invariant test asserts `rows == physical_chunk_lines == manifest.chunk_rows`,
and a *separate* test asserts `live_count <= rows` and that deletion moves both
counts in step.

## Persisted versus derived

Storing only irreducible artefacts means exactly one source of truth per fact and
no class of stale-derived-index bug.

| Artefact | Persisted | Why |
|---|---|---|
| Uploaded originals | Yes | needed to re-chunk and to show source documents |
| Chunks with offsets | Yes | expensive to recompute, and offsets *are* the citation contract |
| Embeddings | Yes | the expensive artefact |
| BM25 index | **No** | derived from live chunks; a persisted copy can only be stale |
| Chunk-text → vector cache | **Yes**, shared | content-addressed, so chunking ablations reuse embeddings |
| `bge-m3` sparse / ColBERT weights | **No** | derived when enabled; ColBERT is infeasible at this scale |
| Query history, eval results | Yes | cheap, and they are the record of what was learned |
| Model weights | No | re-downloaded each session |

## Detectability is not enough — mismatched components are rejected

Detecting a component change and then appending anyway is *worse* than not
detecting it. Build with `fixed/512`, change the chunker to `structural`, index
again, and the index now holds two chunking regimes in one matrix: the invariant
passes, recall degrades, and nothing errors anywhere.

So `POST /api/sessions/{id}/index` compares the session's current index signature
against what the manifest recorded:

- **all equal** → append.
- **any differ** → **409**, naming the differing component and offering
  `?rebuild=true`.

`Settings.index_signature` holds exactly the components that shape stored data:
`chunker`, `embed_model`, `embed_dim`, `vector_dtype`. Query-time settings —
`retrieve_top_k`, `context_top_n`, transform flags — are deliberately absent and
are always free to change.

That distinction is now tested. Two details worth noting:

- The chunker signature records each strategy's **effective** parameters after
  defaults resolve. `get_chunker("fixed", size=256)` leaves `overlap` at 64, and a
  signature built only from explicitly-passed parameters would make two
  differently-configured chunkers hash identically — so §6.5 would let them
  append into one matrix.
- `structural` **rejects** a nonzero `overlap` rather than clamping it. Overlap is
  not merely unhelpful there, it is contradictory: it reintroduces text across the
  heading boundary the strategy exists to preserve. A silent clamp would let
  `chunk_overlap=64` look applied to a `structural` build while changing nothing.

## Size arithmetic

Storage is the least binding constraint, so it is worth proving rather than
hoping. At 1024 dims, one fp16 row is 2,048 bytes exactly:

| Chunks | Vectors | Text |
|---:|---:|---:|
| 10,000 | ~20MB | ~4MB |
| 100,000 | ~205MB | ~40MB |
| 1,000,000 | ~2.0GB | ~400MB |

Sessions of 50k chunks that fill 20GB: about 100. If §5.5 measures that fp16
degrades retrieval, vectors become 4,096 bytes per row, doubling every figure and
still leaving roughly 50 sessions at that size.

Storage is not the constraint. VRAM and CPU time are. The row size is computed in
`Settings.vector_nbytes_per_row` rather than hardcoded anywhere, and a test pins
it to 2,048 because the plan quotes that number repeatedly.

`GET /api/sessions/{id}/index/status` therefore reports index bytes per session
and total bytes used, plus both chunk counts, so growth is visible before the
ceiling is hit rather than after.

## Backup, because auto-save is not a backup

A 20GB store whose only copy is inside a notebook's output is one hard kill from
gone. `scripts/export_index.py` (Phase 8) packages one session or all of them
into a `.zip` under `/kaggle/working`, optionally publishes to a **private** Kaggle
dataset via the CLI, and prints a `FileLink`.

The store is rebuildable from `files/` plus the manifest, so the irreplaceable
assets are the uploaded originals and the evaluation question sets. Both are
small, and both are worth exporting deliberately.

## Current limitations

- The store itself is Phase 1. What exists today is `Settings`, the index
  signature, the error types, and the tests around them — no `session.json`, no
  `chunks.jsonl`, no write-through yet.
- The measured BM25 build cost (~0.7–0.9 ms/chunk) is what makes the LRU
  eviction order worth re-examining; see
  [`measurements.md`](measurements.md).
