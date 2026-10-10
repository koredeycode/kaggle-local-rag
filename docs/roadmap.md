# Phase roadmap

The phase gates from §20, with what is actually done. Each phase is one PR, one
doc update, and one measured delta. Phases 0 and 1 need no GPU.

## Phase 0 — scaffold, mock profile, metrics, eval skeleton — **done**

**Gate:**

| | |
|---|---|
| Test suite green | 225 tests, `make test` |
| Eval harness runs with zero GPU | done |
| BM25 build time measured (§6.7) | done — [measurements](measurements.md) |
| Embedding-cache hit rate measured (§20) | done — [measurements](measurements.md) |
| fp16 vs fp32 agreement measured (§5.5) | shipped, **not run** — needs Kaggle |
| POST-streaming proven through a tunnel (§19.3) | **not started** — needs the service |

Two gates are hardware-gated and honestly reported as such. Both are cheap on
Kaggle relative to the phases that gate on them.

**Two findings changed the plan rather than confirming it:**

- BM25 build cost is ~0.7–0.9 ms/chunk — 8.6s at 10k chunks. The plan suspected
  "seconds"; it is tens of seconds at realistic corpus sizes. Session-open moves
  outside the query path, and the LRU eviction order needs re-examining.
- The embedding-cache hit rate across chunker settings is **12.8%**, not the
  "mostly cache hits" the 8–16 GPU-hour grid budget assumed. §15a's fallback —
  narrow the grid, drop `semantic`, ablate sizes on `fixed` and `structural` — is
  now the plan, halving the largest quota risk in the project.

## Phase 1 — loaders, cleaning, offset map, manifest, session store, persistence

**Gate:** re-ingest skips unchanged docs; offsets round-trip both directions; two
sessions populated and proven mutually invisible; a killed process loses nothing
already written.

The offset map is the centrepiece. Everything downstream assumes clean-space
offsets are correct, and §22 requires a round-trip test on a fixture containing
text the cleaner rewrites, with unmappable regions reporting `None` rather than a
wrong number.

## Phase 2 — fixed + recursive chunking, dense index, exact search, author tool, first 60 questions

**Gate:** baseline recall@5 recorded against a real gold set.

This is where `eval/author.py` earns its keep. The seed set in
[`eval/README.md`](../eval/README.md) has `gold_docs` but empty `gold_spans`, and
cannot produce span-level recall until the operator clicks supporting chunks
through the tool. Nothing owned that work before, and Phases 2–6 all gate on gold
spans that do not exist without it.

## Phase 3 — BM25 + RRF

**Gate:** hybrid-versus-dense delta measured. Both stages exist; this is the first
point the delta is computable.

## Phase 4 — cross-encoder rerank, 512-token windowing

**Gate:** rerank delta measured, expected large.

Expect this to be the single largest quality jump in the project, and expect the
most quota.

## Phase 5 — transforms and the relevance gate

**Gate:** transform deltas measured; gate delta measured **with the abstention
rate reported alongside correct abstention**.

## Phase 6 — small-to-big, compression, citations, packing

**Gate:** compression delta; citation precision and recall measured.

## Phase 7 — session routes, FastAPI, streaming, UI, tunnel

**Gate:** Setup and Ask work per session; all §17.4 states render at
320/390/768/1280px; streaming survives phone backgrounding.

Also where §19.3's streaming claim finally gets proven against a real tunnel.

## Phase 8 — Input-pane restore, export, notebook QR

**Gate:** a new Kaggle session restores from `/kaggle/input/` with no re-embedding.

Durability itself was pulled forward into Phase 1, because retrofitting
write-through onto an index designed to live in RAM is exactly the refactor that
produces the lost-index failure.

## Phase 9 — judge calibration, final ablation matrix, README

**Gate:** full table; human-versus-judge gap reported.

## Three re-orderings from the original plan

Each fixes a sequencing error rather than changing scope:

- **Durability moved from Phase 8 into Phase 1.** §6.2 argues persistence is the
  constraint that shapes everything, and §23 lists auto-save failure as a top
  risk. Waiting until Phase 8 to retrofit write-through is the refactor that
  produces the failure.
- **The eval set moved into Phase 2.** Nothing owned it, and Phases 2–6 all gate
  on gold spans that do not exist until it does.
- **The embedding cache landed in Phase 1**, because the chunking grid in Phase 2
  is where it pays off and it must exist before the first rebuild. Its measured
  hit rate is now known: 12.8%.

## Quota budget

30 GPU-hours per week, 12-hour session cap, T4 x2. Roughly 2.5 sessions per week.

| Activity | Estimate |
|---|---|
| Bootstrap, model pull, preflight | 0.5–1.0h |
| Ingest and embed 1k docs | 0.5–1.0h |
| Ablation: retrieval only, warm index | 0.5–1.5h |
| Ablation: full with generation | 1.5–3.0h |
| **Chunking grid, 8 cells after the measurement** | **4–8h** |
| UI iteration | 1–2h |

The chunking grid was the unbudgeted item. Phase 0's cache measurement halved it
from 16 cells to 8, which is the single largest quota decision made so far.

Comfortable steady state is about two sessions a week, ~5 hours total. One full
grid used to eat most of a week.
