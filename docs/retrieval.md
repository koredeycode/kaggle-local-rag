# Retrieval

Companion to §11 and §12. Records dense search, BM25, fusion, reranking, query
transforms, and the relevance gate.

## The pipeline, in order

1. **Query embedding** via `bge-m3`.
2. **Dense search** — exact; CPU matmul below `dense_gpu_threshold`, GPU 1 above.
3. **BM25** — in-repo Okapi, over live rows only.
4. **Metadata filters** applied *before* both stages, not after.
5. **RRF fusion** — `score = Σ 1/(k + rank)`, k = 60.
6. **Cross-encoder rerank** on the top ~20 fused candidates, windowed to 512.
7. **Parent expansion** on survivors.
8. **Relevance gate** on post-rerank survivors, not on every hit.

## BM25, in-repo

Roughly 100 lines plus tests, chosen deliberately: it is core RAG curriculum, it
adds no dependency, and §11 wants it taught rather than delegated. `rank_bm25`
would be an acceptable substitute if it ever saved time; the ablation table does
not care which implementation produced the numbers, only that they are correct.

Two properties make hybrid retrieval worth the extra work, and both are pinned by
tests against hand-computed cases rather than regression output:

**Term frequency saturates.** With `k1 = 1.5`, the tf factor is 1.000 at tf=1 and
2.174 at tf=10. Ten occurrences score ~2.2× one, not 10×. The test holds document
length constant across the pair so the ratio isolates saturation from length
normalisation.

**Long documents are penalised.** `b = 0.75` normalises for length, so a
5000-token chunk does not beat a 300-token chunk by containing more.

And one property that is easy to omit and produces a bug that shows up on small
test corpora and hides on large ones: **IDF must never go negative.** The
`ln(1 + (N - df + 0.5)/(df + 0.5))` guard matters, because an unguarded formula
gives a negative IDF to a term appearing in more than half the documents, so
containing it *lowers* a score.

### Derived, never persisted

The BM25 index is built from live `chunks.jsonl` on load and is not written to
disk. A persisted copy can only ever be stale relative to the live chunk set, and
stale is worse than slow to rebuild.

That trade is now quantified rather than asserted — see
[`measurements.md`](measurements.md). BM25 costs **~0.7–0.9 ms per chunk to
build**: about 8.6 seconds for a 10k-chunk index, roughly 70–90 seconds at 100k.

That is felt by a user, so session-open moves outside the query path. It also
reframes the eviction order: the plan evicts BM25 before handles because postings
are the memory cost, and the measurement adds that postings are also the expensive
thing to rebuild.

### Tokenisation

Lowercase word tokens, `\w+` with `re.UNICODE` rather than an ASCII pattern, so
Yoruba or Hausa text tokenises instead of vanishing. §24 takes multilingual
support seriously enough that a lexical stage assuming English would quietly
contradict the embedder.

The stopword list is deliberately tiny — 23 terms that appear in every document in
a typical corpus. Not a linguistic list: a large table would change what the
lexical stage measures, and an unrecorded stopword change invalidates the
hybrid-versus-dense delta.

### Learned sparse weights

`bge-m3` emits per-token weights, effectively a learned BM25, "without any
additional cost". `BM25Index.add` accepts `weights` for them — as an **explicit
argument**, because supplying them changes what the lexical stage measures and
should never be a silent path.

Keeping in-repo BM25 anyway gives the ablation three lexical variants to compare:
BM25 alone, `bge-m3` sparse alone, and both fused. Picking one on faith would be
worse than picking one on evidence.

## RRF, and why it is rank-based

Fusion sums `1/(k + rank)` over every list containing a row.

The reason it is rank-based is the usual source of hybrid bugs. A dense cosine
similarity lives in [−1, 1]; a BM25 score is unbounded and corpus-dependent.
Combining them directly requires a normalisation step, and every normalisation
step has to answer "normalise against what population?", which can be wrong in
ways that produce plausible-looking rankings.

Fusion on *positions* sidesteps the question: rank 1 in both lists beats rank 1 in
one and rank 30 in the other, whatever either system's score scale means.

Properties the tests pin:

- **Agreement beats confidence.** A row in both lists outranks a row at the same
  rank in one. This is the whole premise of hybrid retrieval.
- **Promoting a row never lowers its own score.** The claim is per-row: promoting
  `z` from rank 2 to rank 0 raises `z`, while the rows it passed fall by exactly
  the same amount.
- **Determinism.** Ties break on first appearance across sources. An ablation
  table has to be diffable, and a tie broken by dict iteration order is not.
- **`k = 60` damps the top.** Without the offset, a single rank-1 result scores
  60× a rank-40 one; with it, under 3×. `k` is a constant from the original paper
  and is notably insensitive, which is why it is not a tuning knob.

## Exact search, not approximate

See [`architecture.md`](architecture.md). The short version: exact search returns
true recall@k, which is what makes Phase 2's chunking comparison attributable to
the chunker rather than to an ANN index's error. And the fp16→fp32 upcast before
the matmul is a correctness requirement, not a tuning choice — numpy has no fast
`float16` BLAS path, so a naive fp16 matmul reads as "exact search is slow" when
it is a dtype bug.

## Reranking

A bi-encoder embeds query and passage independently, so a perfectly relevant
passage can embed poorly in isolation. A cross-encoder reads them together and
can tell a passage is relevant *for this question* even when it looks like noise
in isolation. Expect this to be the single largest quality jump in the project.

**Scores are unbounded logits**, roughly −8 to +5 for `bge-reranker-v2-m3`, not
probabilities. `sigmoid` maps to [0,1] for display and thresholding; never compare
raw scores across queries, because the scale shifts with query difficulty.

**512-token ceiling.** Longer chunks are windowed to a query-aware head rather than
truncated. See [`chunking.md`](chunking.md) — this reshapes the ablation grid, not
just the reranker.

**Batched across requests**, because the cross-encoder is throughput-bound. This
is also the easiest place to write an invisible bug: misattributing a score to the
wrong query is trivial to write and catastrophic to retrieval quality. §22
requires an association test.

## Query transforms

Each is a flag, each is measured, none is on by default. They cost latency and
tokens, so they earn their place through ablation deltas or they get removed.

| Transform | What it does | Cost |
|---|---|---|
| `rewrite` | Make a conversation-scoped question standalone | One small call |
| `hyde` | Embed a plausible fake *answer* instead of the question | One generation + one embed |
| `decompose` | Split multi-hop into sub-questions, retrieve each, merge | One generation + N searches |
| `multiquery` | Paraphrase fan-out, fuse the lists | One generation + N searches |

HyDE works because a fake answer is closer in style and vocabulary to the real
answer than a question is — so it helps most when the query is short or uses
vocabulary unlike the corpus. Decomposition is the standard route to questions
that need evidence stitched from two places.

## The relevance gate is not a transform

The gate grades post-rerank survivors as `relevant` / `ambiguous` / `irrelevant`:

- confidently relevant → proceed
- mostly irrelevant → re-retrieve once with an adjusted query
- still nothing → **abstain explicitly**, naming what is missing

It is separated from the transforms above because transforms change *what is
searched* and are off by default pending ablation deltas, while the gate changes
*whether the system speaks at all* — a correctness property, so `gate_enabled`
defaults to `True`.

### Its cost is the part that is easy to forget

One LLM call over the assembled candidates is not free on a T4: several seconds of
prefill before generation even starts. That makes the gate plausibly the **slowest
pre-generation stage** — not reranking — which is why `gate_ms` is its own field in
`StageTimings` and its own row in the UI stage readout rather than being folded
into `retrieve_ms`. A user waiting on an unexplained multi-second pause is worse
served than one told it is grading evidence.

Three consequences, all specified:

- The gate runs **after** reranking, over `gate_max_spans` (6) survivors. Grading
  all 40 would multiply the cost by ~7 for no benefit — rerank already removed the
  obviously-bad candidates.
- It is **one batched JSON-mode call** returning a verdict per span, not one call
  per span.
- Re-retrieval happens **once**, not in a loop.

### Abstention is a feature

A system that answers confidently from bad retrieval is worse than one that says
"the corpus does not cover this". The eval harness scores refusals as correct when
gold evidence is absent — but only ever alongside the overall abstention rate,
because a system that abstains on everything would otherwise score 15/15.

That is why `abstention_metrics` **raises** when called with no unanswerable
questions, and why `GenerationMetrics.validate()` refuses to render a row carrying
`correct_abstention` without `abstention_rate`. See [`evaluation.md`](evaluation.md).

## Current limitations

- Only BM25 and RRF are implemented. Dense search, filters, reranking, parent
  expansion, transforms, and the gate are Phase 2–5.
- The `MockEmbedder` used in tests is a hashed bag of words. It is deterministic
  and dependency-free, which is what tests need, and it preserves exact-token
  overlap, which makes the dense/BM25 complementarity test meaningful. It is **not
  semantic** and carries no quality claim — its `model_id` says so loudly.
- `MockReranker` does see query and passage together, deliberately: a stub that
  scored passages independently would make the rerank ablation untestable.
- No retrieval metric has been computed against a real index. Every delta quoted
  in the plan is a projection.
