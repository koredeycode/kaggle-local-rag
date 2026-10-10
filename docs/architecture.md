# Architecture

Companion to §4 of the plan. Records what the system does, why, and what is not
built yet.

## The problem this shape solves

Kaggle gives you `T4 x2`: one 16GB device seen twice, 4 CPU cores, 29GB of RAM, a
12-hour session cap, and 30 GPU-hours a week. Ollama will not split a sub-16GB
model across both cards, so the naive arrangement leaves the second T4 idle for
the entire session while the generator and the embedder fight over the first.

## Split by workload, not by model

| | GPU 0 — generation | GPU 1 — retrieval |
|---|---|---|
| Process | Ollama | `sentence-transformers` worker |
| Models | `qwen3:8b` generator, and the same model as judge | `bge-m3` embedder, `bge-reranker-v2-m3` |
| Access pattern | Sequential, low concurrency | Batched, high throughput |
| Isolation | `CUDA_VISIBLE_DEVICES=0` | `CUDA_VISIBLE_DEVICES=1` |

The reasoning is about *access pattern*, and that is what makes it hold rather
than merely sound plausible:

- Reranking is the highest-return component in RAG and it is throughput-bound,
  not latency-bound. Batching it on a dedicated card costs the generator nothing.
- The generator is latency-bound and wants its VRAM undivided.
- `CUDA_VISIBLE_DEVICES` is a cheap, **hard** boundary. Neither process can see
  the other's memory, so neither can OOM the other.

Counting visible devices is the weaker check and would pass even if both landed
on GPU 0, which is why `scripts/preflight.py` resolves PIDs to physical GPU
indices via `nvidia-smi --query-compute-apps=pid,...,gpu_uuid` rather than
counting.

## Component flow

```text
                    ┌──────────────── GPU 0 ────────────────┐
                    │  Ollama :11434                        │
                    │  qwen3:8b  ├─ generate               │
                    │  (same)    └─ judge (JSON mode)       │
                    └───────────────────▲────────────────────┘
                                        │ loopback HTTP
┌──────────┐   ┌──────────┐   ┌────────┴─────────┐   ┌──────────┐
│  ingest  │──▶│  chunk   │──▶│  retrieve/transform│──▶│ generate │
│ loaders  │   │ strategies│   │  rerank / gate     │   │  + cite  │
└──────────┘   └──────────┘   └────────▲─────────┘   └──────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │  dense store (exact) + BM25     │
                    │  /kaggle/working/ragstore (20GB)│
                    └────────────────▲────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │  GPU 1 — sentence-transformers  │
                    │  bge-m3 embedder                │
                    │  bge-reranker-v2-m3 reranker    │
                    │  CUDA_VISIBLE_DEVICES=1         │
                    └─────────────────────────────────┘
```

## Exact search, not approximate

At 1024 dimensions and fp16, a 100,000-chunk index is ~200MB. Brute-force exact
search over that is a single matrix multiply, so there is no reason to accept an
approximate index's error.

That matters more than the speed does. Phase 2 compares chunking strategies and
attributes recall changes to the chunker; an ANN index adds its own recall loss to
every cell of that table, and the attribution becomes unrecoverable. Ship
approximate search only when a measured corpus exceeds roughly 250k chunks, and
re-baseline everything when you do.

**Where the matmul runs, and the footgun that decides it:**

| Index size | Path |
|---|---|
| ≤ `dense_gpu_threshold` (50k) | `np.load(mmap_mode="r")`, **upcast fp16→fp32**, matmul on CPU |
| above | one fp32 matrix copied to GPU 1 per session, reused across queries |

The upcast is not an optimisation, it is a correctness requirement. numpy has no
fast `float16 @ float16` BLAS path — it falls back to a slow reference loop — so
a naive fp16 matmul turns "effectively instant" into seconds. The symptom reads as
"exact search is slow", i.e. as a design flaw, when it is a dtype bug. That is
exactly why §22 requires a dense-search test against a brute-force reference.

Below the threshold, dense search runs on CPU and therefore does not contend with
the embedder for GPU 1 in the common case. Most sessions stay on CPU.

## Concurrency with one embedder

Every session shares GPU 1's embedder and reranker, so contention is deliberate
rather than incidental:

- Embedding and reranking run behind an `asyncio.Semaphore`. Without it, several
  concurrent queries on 4 cores and one GPU thrash, and every request slows down
  including the first.
- Reranking is **batched across requests**, because the cross-encoder is
  throughput-bound and batching is where its throughput comes from. Cross-request
  batching is the single easiest place to write a bug that is invisible in
  testing: misattributing a score to the wrong query is trivial to write and
  catastrophic to retrieval quality. §22 requires an association test.
- Ollama handles its own concurrency, but `keep_alive` must be set or the model
  reloads between queries and the first request after an idle gap pays a
  multi-second penalty.

### The BM25 finding

Measured, not assumed — see [`measurements.md`](measurements.md). BM25 costs
~0.7–0.9 ms per chunk to build: about **8.6 seconds for a 10k-chunk index** and
roughly **70–90 seconds at 100k**.

That is felt by a user. So session-open moves outside the query path (onto
`POST /index` or session selection), and the semaphore covers index-open as well
as embed and rerank. A user who switches sessions and then asks a question should
never sit through that pause.

It also reframes the §6.7 eviction order. The plan evicts BM25 before handles
because postings are the memory cost and handles are nearly free; the measurement
shows postings are also expensive to *build*, so that eviction is not free either.
The order is still right for memory pressure. It is worth re-measuring against a
real corpus before treating it as settled.

## What is not built yet

Phase 0 covers configuration, types, errors, metrics, profiles, chunking, BM25,
RRF, and the evaluation harness. The following directories exist with docstrings
and are **empty of implementation**: `ingest/`, `sessions/`, `retrieval/`,
`transforms/`, `assemble/`, `generate/`, `service/`, `bootstrap/`. The vector
store, exact dense search, the embedding cache, and the cross-encoder all arrive
in Phases 1–4.

## Current limitations

- Every quality and latency claim about the models is unmeasured. §3 is explicit
  that local development validates logic, types, and API contracts only; quality
  numbers come from Kaggle runs or they do not exist.
- `vector_dtype` defaults to `float16` because that is the plan's default, not
  because §5.5 has confirmed it is safe. `scripts/measure_precision.py` exists to
  settle it and has not been run.
- The chunker signature (§6.5) records each strategy's *effective* parameters
  after defaults resolve, so `fixed` at 256 and `fixed` at 512 produce different
  index signatures and a re-index between them is refused. This is verified in
  tests but the 409 endpoint that consumes it is Phase 7.
