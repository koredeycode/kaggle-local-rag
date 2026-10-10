# docs/

Written as decisions are made, not retrofitted, so the reasoning is recorded at
the moment it is formed rather than reconstructed later.

Every document states what the system does, **why**, and its current limitations.
No aspirational descriptions of unimplemented behaviour — an explicit
"Current limitations" section is the model.

## Read in this order

| | |
|---|---|
| [architecture.md](architecture.md) | Components, data flow, and why the GPU split is by workload rather than by model |
| [sessions-and-storage.md](sessions-and-storage.md) | Session model, on-disk layout, the append invariant, logical deletion, write-through |
| [ingestion.md](ingestion.md) | Loaders, cleaning, and the clean↔original offset map |
| [chunking.md](chunking.md) | Four strategies, trade-offs, and the reranker's 512-token ceiling |
| [retrieval.md](retrieval.md) | Dense, BM25, RRF, reranking, transforms, the relevance gate |
| [evaluation.md](evaluation.md) | Metrics, the judge, and the two ablation matrices |
| [security.md](security.md) | Threat model, loopback discipline, tunnel policy |
| [development.md](development.md) | Local setup, the unittest/pytest split, style |
| [roadmap.md](roadmap.md) | Phase gates, what is done, and what the measurements changed |
| [measurements.md](measurements.md) | **Phase 0 measurements, with the numbers this machine produced** |
| [notebook-runbook.md](notebook-runbook.md) | Cell-by-cell notebook structure and the Kaggle workaround catalogue |
| [../PLAN.md](../PLAN.md) | The design document every `§n` reference points into |

## Not yet written

These are Phase-deliverable documents. They are listed so their absence is
visible rather than mistaken for an oversight:

`glossary.md`, `api.md`, `ui.md`, `observability.md`, `troubleshooting.md`,
`production-migration.md`, `kaggle-operations.md`, `models.md`.

`glossary.md` deserves a note. §2 of the plan is a glossary *inside* a plan,
which is a reasonable place to learn the vocabulary and a poor place to look it
up while coding. The standalone version is generated from the same source so the
two cannot drift — which means writing the generator, not copying the text by
hand.

## Section references

`§n` throughout the codebase and these documents refers to sections of
[`../PLAN.md`](../PLAN.md). Citing the section next to a claim is how a reader
tells which decisions are settled and which are still open.
