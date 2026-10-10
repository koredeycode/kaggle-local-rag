# Evaluation

Companion to §15 and §15a. This is the deliverable that turns the project from a
demo into an engineering artefact, so it gets more care than its size suggests.

## Evaluation is per session

Every metric is only meaningful against a specific corpus. An eval set lives
inside its session's `eval/` directory, is versioned, and travels with the corpus
it describes. Results from two sessions are never placed in one table, even when
the configuration is identical, because the corpora differ.

This is enforced rather than documented: `assert_comparable` raises when two runs
differ on session, embedder, embedder revision, or judge. `EvalSet` refuses to
exist without a `session_id`.

## The seed set, and its honest gap

[`eval/questions.sample.jsonl`](../eval/questions.sample.jsonl) has 34 questions:
24 answerable, 10 unanswerable.

`gold_docs` is filled in. **`gold_spans` is empty, and that is deliberate.**

§15 requires gold spans in **clean space** (§9) — character offsets into the
*cleaned* text, because recall@k compares gold against what retrieval actually
ranked. A span in original-file coordinates is a different space, and the
comparison is undefined rather than wrong.

Those offsets cannot be written by hand; counting characters into a parsed document
is not something anyone does correctly twice. `eval/author.py` (Phase 2) has the
operator type a question, retrieve against the warm index, and click the
supporting chunk — the tool emits the exact `clean_start`/`clean_end`. Humans judge
relevance by reading; the tool handles coordinates.

So the set supports **doc-level** retrieval scoring only until Phase 2. That is
stated in `eval/README.md` and pinned by a test, rather than hidden behind a
plausible-looking number.

## Metrics

**Retrieval** — recall@k, hit-rate@k, MRR@10, nDCG@10, plus a span-coverage
measure.

Two conventions worth stating:

- **Unanswerable questions are excluded from retrieval averages.** A question with
  no gold evidence has no recall to score; counting it as zero would make an
  unanswerable question look like a retrieval failure, which is the opposite of
  what it is.
- **A missing result scores zero rather than being dropped.** An ablation cell that
  failed to retrieve for one question should show a zero in that row. Skipping it
  would make a broken run look better than a working one.

`span_coverage` is the partial-credit measure: a 1024-token chunk overlapping 80%
of a gold span is a different outcome from missing it, and plain recall@k calls
both zero. This is the metric that matters when chunk size is the variable under
study. Overlapping retrieved spans are merged before counting, so two chunks each
covering half a gold span score 1.0 rather than double-counting, and coverage is
matched per document — byte 100 of one document has nothing to do with byte 100 of
another.

## The abstention pair

The most important thing in this document.

Scoring refusals as correct when gold evidence is absent rewards the right
behaviour. On its own it is also trivially gameable: a system that refuses
everything scores 15/15 on the unanswerable set while being useless.

So the eval reports a **pair**:

| Metric | Reads as |
|---|---|
| `correct_abstention` | of the unanswerable questions, how many were correctly refused |
| `abstention_rate` | of **all** questions, how many were refused at all |

A good system sits high on the first and low on the second. A degenerate one
shows up as `correct_abstention` at ceiling with `abstention_rate` climbing
toward 1.0.

Reporting either number alone is worse than reporting neither, because it invites
exactly that conclusion. So the harness enforces it in code:

- `abstention_metrics` **raises** when called with no unanswerable questions.
- `GenerationMetrics.validate()` **raises** when a row carries no
  `abstention` block at all.
- `report.py` renders both columns together and carries a footnote explaining why.
- `is_degenerate()` flags a row at ≥0.8 on both.

`false_abstentions` is reported alongside: the count of answerable questions the
system refused. `correct_abstention` alone hides exactly that.

## Citation precision, defined narrowly

Every marker the model emits that resolves is, mechanically, a valid citation. So
the only precision failure available is **over-citation**, and that is what
`precision` measures:

```text
precision = min(markers_used / markers_supplied,
                1 / markers_per_cited_sentence)
```

Both terms are 1.0 in the healthy case — each source used, one marker per claim.
The first punishes spraying every source onto every claim; the second punishes
citing several sources for one claim. `citation_recall` (sentences with a citation
over sentences making a claim) catches unsupported claims, which are the dangerous
kind because they read as correct.

`precision` is not reported on fewer than three cited sentences: a precision
computed over one sentence is a boolean, and printing `1.0` next to it invites
reading it as a quality number.

## The judge, and what it cannot tell you

`qwen3:8b` on GPU 0, JSON mode, thinking disabled. No external judge, no API key.
Three consequences, all acknowledged in the plan:

- **An 8B judge is a weak judge.** Absolute faithfulness numbers are pessimistic
  and only comparable against themselves.
- **The judge is also the generator**, which is textbook self-preference bias: the
  model is most likely to rate its own output as faithful. This cannot be fixed by
  model choice — a `qwen3:14b` judge does not fit beside an 8B generator on one
  16GB T4 (9.3 + 5.2 + KV > 16). The two mitigations that do fit are both free:
  blind the judge to which variant produced an answer, and randomise presentation
  order within a run.
- For that reason, keep a fixed human-rated sample of ~20 answers as a calibration
  anchor and report the human-versus-judge gap periodically. That sample is doing
  more work than the plan credits it for.

`report.py` prints this bias in a footnote under every generation table, so
somebody reading only the table still meets it.

**Rule: never compare numbers across different corpora, embedders, or judges.**
Comparisons are valid only within a fixed setup.

## Two matrices, because one grid costs more than the quota

| | Retrieval matrix | Generation matrix |
|---|---|---|
| Metrics | recall@k, hit-rate@k, MRR@10, nDCG@10, p95, coverage | + faithfulness, relevancy, citation P/R, abstention pair |
| Cost per config | embed once, then seconds per question | plus one 8B generation per question |
| Grid | large — the whole chunking × retrieval space | small — only the 3–4 configs worth a faithfulness claim |

Retrieval-only metrics run through `/api/sessions/{id}/search` and generate
nothing. That is not an incidental simplification — it is the difference between
a feasible ablation programme and an infeasible one. Phase gates about baseline
recall, hybrid delta, and rerank delta run the cheap matrix and can therefore be
large; gates about grounding run the generation matrix on a question subset.

## The chunking grid is an embedding cost

4 strategies × 4 sizes = 16 index builds, at 8–16 GPU-hours per §21 — more than
half the weekly quota for one phase.

**Measured, not assumed.** The content-addressed embedding cache was supposed to
make second-and-later builds mostly cache hits. It is not. Exact-text overlap
between chunker settings averages **12.8%**, because a chunker that cuts anywhere
else produces chunk text that differs everywhere. See
[`measurements.md`](measurements.md).

That makes §15a's fallback the live plan: drop `semantic` and ablate sizes only on
`fixed` and `structural`. Eight cells instead of sixteen, halving the largest
single quota risk in the project. A measurement that made the grid *smaller* is a
better outcome than a cache that worked, because the decision is now defensible.

## Determinism

Every run records: seeds, model identity **and revision**, the full retrieval
config hash, `rerank_window`, the live and physical chunk counts, and library
versions. Without these a 3-point "improvement" is uninterpretable — you cannot
tell a real delta from a changed setting.

`Settings.retrieval_config_hash` is computed from the query-time signature, and
`Settings.index_signature` separately from the stored-data components (§6.5). Two
runs are only comparable within one config hash, and `report.py` partitions rows
into groups by hash rather than tabulating them together.

## Current limitations

- Only the retrieval half and the citation/abstention arithmetic exist. The judge,
  the ablation runner, and `eval/author.py` are Phase 2 and Phase 9.
- No metric has been computed against a real index. Every delta in the plan is a
  projection.
- The seed set's 24/10 split is smaller than the plan's 45/15. It is large enough
  to exercise every metric path and small enough for an operator to read in one
  sitting, which matters more than hitting a number with questions nobody has
  read. `meets_recommended_composition()` reports the gap rather than hiding it.
