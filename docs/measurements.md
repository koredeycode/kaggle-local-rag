# Phase 0 measurements

§20 lists four measurements in the Phase 0 acceptance gate. Three need no GPU
and were completed on a laptop; one needs `bge-m3` and ships as a script.

Every number below was produced by a committed script, and every script is
seeded so the run is reproducible. An unseeded measurement cannot be compared
with a later one, and comparison is the entire point of measuring.

**These figures are from the development box, not from Kaggle.** It has 8 CPU
cores where Kaggle has 4, no GPU, and a different BLAS. Read the shape of each
result, not the absolute value, and re-run on Kaggle before spending quota on it.

---

## 1. BM25 build cost — §6.7 gate, complete

```bash
python3 scripts/measure_bm25.py --chunks 10000 25000 --repeats 1
```

| chunks | corpus MB | build ms | µs/chunk | query p50 ms | query p95 ms | terms |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 20.5 | 8,586 | 859 | 84.3 | 85.3 | 167 |
| 25,000 | 51.1 | 17,755 | 710 | 216.9 | 227.0 | 167 |

### What this changes

**§6.7's suspicion was right, and understated.** The plan says BM25 build time is
"plausibly seconds, not 'well under a second'" and should be measured rather than
assumed. It is ~0.7–0.9 ms per chunk, which puts a 10k-chunk index at **8.6
seconds** and a 100k-chunk index at roughly **70–90 seconds**. Those are the
numbers §6.7 was reaching for.

Consequences, all of which follow directly:

- Session-open must move **outside** the query path: build on `POST /index`, or
  on session selection in the UI. A user who switches sessions and then asks a
  question will otherwise sit through a multi-second pause before the first stage
  of the pipeline begins.
- The §6.7 semaphore must cover index-open, not just embed and rerank.
- The §6.7 LRU eviction order — BM25 evicted before handles — **matters much more
  than the plan assumed**. The plan calls memory-mapped handles nearly free and
  BM25 postings "the real cost"; this measurement says BM25 postings are also the
  expensive thing to *build*, so evicting them first means paying the cost again on
  the next switch. The eviction order is still right for memory, but it is not
  free, and it is worth re-measuring after the first real corpus.

### What this does not tell us

The query columns are an **upper bound**, and deliberately so. The synthetic
corpus has 167 distinct terms, so every query term walks a posting list of
`corpus/167` rows. A real corpus has orders of magnitude more vocabulary, so its
posting lists are far shorter. Do not carry the 84ms query figure into any latency
claim — the build column is the one that needs acting on.

### Not yet measured

A 100k-chunk build, directly. The 70–90 s figure is a linear extrapolation from
10k and 25k. The extrapolation is safe because the cost is per-token and
linearity holds across a 2.5× range, but it is still an extrapolation:
`--chunks 100000 --repeats 1` confirms it, and costs about 90 seconds of CPU to do
so. Nobody has run that yet.

---

## 2. Embedding-cache hit rate — §20 gate, complete

```bash
python3 scripts/measure_cache_hits.py --documents 30 --strategies fixed recursive structural --sizes 256 512
```

| strategy | size | distinct texts | max prior hit | mean prior hit |
|---|---:|---:|---:|---:|
| fixed | 512 | 31 | 32.3% | 32.3% |
| recursive | 256 | 52 | 0.0% | 0.0% |
| recursive | 512 | 31 | 35.5% | 11.8% |
| structural | 256 | 81 | 0.0% | 0.0% |
| structural | 512 | 81 | 100.0% | 20.0% |

Mean prior-cell hit rate: **12.8%**.

### What this changes

The §21 chunking grid is 16 fresh index builds at 8–16 GPU-hours — more than half
the weekly quota for one phase. The plan's stated lever is that "heavily
overlapping outputs across strategies mean most second-and-later builds are mostly
cache hits", and it says Phase 0 should measure the real rate rather than assume
it.

**The assumption does not hold.** Exact-text overlap between chunker settings is
low, because a chunker that cuts anywhere else produces chunk text that differs
everywhere. Even the best-matched pair — `recursive/512` against `fixed/512` —
shares only 35% of its chunks, and the mean across the grid is 12.8%.

This makes §15a's fallback the live plan rather than the contingency:

> If the measured cache hit rate is low and the corpus is large, the honest move is
> to narrow the grid — drop `semantic` and ablate sizes only on `fixed` and
> `structural`.

So the grid becomes 8 cells rather than 16, halving the largest single quota risk
in the project. That is a better outcome than a cache that worked: the measurement
is what makes the decision defensible, and `structural` is the strategy §10
expects to win on a structured corpus anyway.

### What this does not tell us

This synthetic corpus shares paragraph phrasing across documents, which
**inflates** the hit rate. A real corpus of distinct documents will be lower. The
direction of the finding is therefore robust — "low" is generous here — but the
number is not a prediction. Re-run against the real corpus before relying on 12.8%
specifically.

Note also that the cache is keyed by chunk-text hash, so this measurement is
**model-independent**: it does not change when the embedder does. That is why it
could be taken on a laptop at all.

---

## 3. fp16 vs fp32 embedding agreement — §5.5, not yet run

```bash
# Kaggle only. Needs bge-m3, which is a 2.3GB download on a GPU box.
python3 scripts/measure_precision.py --json eval/runs/measure_precision.json
```

§5.5 is explicit that `bge-m3`'s "slight performance degradation" in fp16 is not a
number, and that silent embedding degradation would corrupt every retrieval metric
downstream. The plan forbids guessing.

The script measures two things, and the second is the one that matters:

1. **Per-vector cosine agreement** between the fp32 and fp16 embedding matrices —
   mean and minimum, because a good mean can hide one chunk that moved a lot.
2. **Retrieval consequence** — whether a sample query set returns an identical
   top-k neighbour set, and at what rank the ordering first diverges. Cosine
   agreement does not imply identical rankings, and rankings are what recall is
   computed from.

Exit code 3 means fp16 failed the check, and the fix is `vector_dtype=float32`.
§5.3 budgets `bge-m3` fp32 at ~2.3GB on GPU 1, which fits comfortably, so the cost
is index size (§6.6: 4,096 bytes per row instead of 2,048) rather than VRAM
pressure. Note that switching dtype invalidates the index and forces a re-embed,
which is why this is measured before the Phase 2 baseline rather than after.

---

## 4. POST-streaming through a Quick Tunnel — §19.3, not started

The decision is already made and needs no measurement to justify: Quick Tunnel
buffers `GET` responses and flushes on close
([cloudflared#1449](https://github.com/cloudflare/cloudflared/issues/1449),
observed, and still open), and `EventSource` can only issue `GET`. So streaming is
`POST` + `fetch` + `ReadableStream` with standard SSE framing.

What is untested is whether that actually works end to end through a real tunnel,
which is what §20 asks for. That needs a tunnel, an Ollama server, and a real
generation pass — none of which exist on this box. The client code is written so
that switching back to `EventSource` is a one-line method change, so re-running
this gate later is cheap.

§19.3 also warns against treating this as a law: Cloudflare's maintainer replied
that quick tunnels carry deliberate guardrails because they are a demo product, and
a flush-on-write fix landed on their edge in mid-2025. Re-verify each time, and
prefer a named tunnel the moment one is available.

---

## Summary

| Gate | State | Cost to finish |
|---|---|---|
| BM25 build time (§6.7) | complete | — |
| Embedding-cache hit rate (§20) | complete | — |
| fp16 vs fp32 agreement (§5.5) | shipped, unrun | one Kaggle session, minutes of GPU |
| POST-streaming (§19.3) | not started | one Kaggle session, needs the service from Phase 7 |

Both unrun gates belong to a Kaggle session, and neither is on the critical path
for Phase 1. Both are cheap there relative to the work that gates on them.
