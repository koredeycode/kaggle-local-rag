# kaggle-local-rag

Citation-backed RAG running entirely on local models inside a free Kaggle T4 x2
session. No API key, ever.

This is a **curriculum project**: every design decision in it is also a thing it
tries to teach. The full design is in [`PLAN.md`](PLAN.md);
[`docs/`](docs/) records the reasoning as it is made rather than reconstructed
later.

---

## Status: Phase 0

The scaffold, the core logic, and two of the four Phase 0 measurements are done.

| Phase 0 gate (§20) | State |
|---|---|
| `pytest` green | **225 tests**, run locally with stdlib `unittest` (see below) |
| Eval harness runs with zero GPU | **done** — retrieval metrics, citation metrics, the abstention pair, reporting |
| BM25 build time measured (§6.7) | **done** — [`docs/measurements.md`](docs/measurements.md) |
| Embedding-cache hit rate measured (§20) | **done** — [`docs/measurements.md`](docs/measurements.md) |
| fp16 vs fp32 embedding agreement (§5.5) | **shipped, not yet run** — needs `bge-m3`, i.e. Kaggle |
| POST-streaming proven through a Quick Tunnel (§19.3) | **not started** — needs a real tunnel, i.e. Kaggle |

The two unrun gates need hardware this box does not have, and §3 is explicit that
local runs validate logic, types, and API contracts rather than model quality or
latency. Both ship as scripts (`scripts/measure_precision.py`,
`scripts/measure_cache_hits.py`) rather than as TODOs.

## What works today

No GPU, no model download, no network:

```bash
python3 -m unittest discover -s tests -t .     # 225 tests
python3 scripts/measure_bm25.py                # §6.7 gate
python3 scripts/measure_cache_hits.py          # §20 gate
python3 scripts/preflight.py                   # §19.1 report
python3 scripts/cleanup.py --dry-run           # §19.1 teardown
```

Implemented and tested: core types with the two offset spaces (§9), frozen config
with the index/query setting split (§6.5), the closed error-code set and single
envelope (§16), dependency-free p50/p95 metrics with enforced redaction (§16),
the three profiles and `mock`/`tiny`/`local` resolution (§8), all four chunking
strategies behind one interface (§10), in-repo BM25 (§11), RRF (§11), and the
retrieval plus generation evaluation metrics including the abstention pair (§15).

## What is not built yet

Phase 1 onwards, and named so nobody assumes it exists: loaders and cleaning,
the session store and write-through persistence, the vector store and exact
search, the cross-encoder, query transforms, the relevance gate, context
assembly, generation and citation resolution, the FastAPI service, and the
browser client. Their module directories exist with docstrings explaining what
they own; the modules inside them do not.

## Two things worth knowing before you read the code

**The two offset spaces on every chunk are load-bearing.** `clean_start`/`clean_end`
index the *cleaned* text — what retrieval ranked and what span verification runs
against. `orig_start`/`orig_end` index the *file on disk* — what a citation points
at so a human can check it. They differ wherever cleaning rewrote anything, and
conflating them produces citations that name the wrong words plus a recall@k that
is undefined rather than wrong.

**`correct_abstention` is never reported without `abstention_rate`.** Scoring
refusals as correct on unanswerable questions rewards the right behaviour, and on
its own it is trivially gameable: a system that refuses everything scores 15/15.
`GenerationMetrics.validate()` raises if you try to render a row carrying one
without the other.

## Installing

```bash
pip install -e .                 # core: numpy only
pip install -e '.[local]'        # Kaggle: torch, sentence-transformers, FastAPI
pip install -e '.[dev]'          # pytest, mypy, ruff
```

## Testing without pytest

This project was built on a box with **no network and no pip**, so `pytest` cannot
be installed here. Tests are therefore written as `unittest.TestCase` classes:
`python3 -m unittest` runs them today, and `pytest` collects the same classes
unchanged on Kaggle, satisfying the §22 gate where pip works. `make test` and
`make test-kaggle` run the same suite through the two runners.

## Documentation

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Components, data flow, and why the split is by workload rather than by model |
| [`docs/sessions-and-storage.md`](docs/sessions-and-storage.md) | Session model, on-disk layout, the append invariant, logical deletion |
| [`docs/chunking.md`](docs/chunking.md) | The four strategies and the reranker's 512-token ceiling |
| [`docs/retrieval.md`](docs/retrieval.md) | Dense, BM25, RRF, reranking, the gate |
| [`docs/ingestion.md`](docs/ingestion.md) | Loaders, cleaning, and the clean↔original offset map |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics, the judge, and the two ablation matrices |
| [`docs/security.md`](docs/security.md) | Threat model, loopback discipline, tunnel policy |
| [`docs/measurements.md`](docs/measurements.md) | **Phase 0 measurements, with the numbers this box produced** |
| [`docs/notebook-runbook.md`](docs/notebook-runbook.md) | Cell-by-cell notebook structure and the Kaggle workaround catalogue |
| [`eval/README.md`](eval/README.md) | The seed question set and why its gold spans are empty for now |

## Safety

This serves a single operator over a temporary Cloudflare Quick Tunnel. That
tunnel is public, random, temporary, and has no SLA — Cloudflare describes it as a
demo product with deliberate guardrails. Everything binds `127.0.0.1`; the tunnel
is the only ingress; a locally generated runtime token is printed once and never
written to disk. Ollama is never exposed. See [`docs/security.md`](docs/security.md).

## License

Apache-2.0, matching the models this project serves.
