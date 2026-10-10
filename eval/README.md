# Seed eval set

`questions.sample.jsonl` is the committed starter set described in §15. It is
copied into a session's `eval/` directory on creation, and from then on it
travels with that corpus (§15: evaluation is per session, because every metric is
only meaningful against a specific corpus).

## Current state: questions yes, gold spans no

`gold_docs` is filled in. **`gold_spans` is empty, and that is deliberate at
Phase 0.**

§15 requires gold spans in **clean space** (§9) — character offsets into the
*cleaned* document text, because recall@k compares gold against what retrieval
actually ranked. A span in original-file coordinates is a different space, and the
comparison is undefined rather than merely wrong.

Those offsets cannot be written by hand. Counting characters into a parsed
document is not something anybody does correctly twice, which is why
`eval/author.py` exists in Phase 2: the operator types a question, the tool
retrieves against the warm index, and the operator clicks the supporting chunk.
The tool emits the exact `clean_start`/`clean_end`. Humans judge relevance by
reading; the tool handles coordinates.

Until then, the set supports **doc-level** retrieval scoring only. `retrieve`
metrics compute over `gold_docs` when `gold_spans` is empty; `span_coverage`
returns 0.0 and is not meaningful. That limitation is stated here rather than
hidden behind a plausible-looking number.

## Composition

| | Count |
|---|---|
| Total | 34 |
| Answerable | 24 |
| Unanswerable | 10 |

§15 asks for 45 answerable and 15 unanswerable. This is a starter set, not the
final one: it is large enough to exercise every metric path and small enough for
an operator to read in one sitting, which matters more than hitting a target
number with questions nobody has read.

The 3:1 answerable-to-unanswerable ratio is preserved, and that ratio is the part
that is not arbitrary. A set dominated by unanswerable questions scores abstention
beautifully and retrieval terribly.

## Tags

Every question carries the retrieval property it probes, because a single blended
recall number hides which case is failing (§15). Two are worth watching first:

- **`exact-identifier`** (q006, q022) — statute numbers and verbatim clause text.
  This is the tag to check when the hybrid delta is disappointing; if dense-only
  retrieval is losing here, BM25 is doing its job.
- **`paraphrased`** (q018-q020) — the same fact asked in different words. This is
  the tag to check when the lexical stage is winning alone.

## Adding questions

1. Add a line. `qid` is the join key for every result row, so it must be unique.
2. Tag it. An untagged question is a blended question.
3. Unanswerable questions get `answerable: false` and **no** gold. A question
   labelled unanswerable that carries gold evidence is a labelling error, not a
   hard case, and `EvalQuestion.validate` rejects it.
4. `EvalSet.validate` runs on load. Empty spans, negative offsets, duplicate qids,
   unknown tags, and an answerable question with neither spans nor docs are all
   errors rather than warnings.

## Validating this file

```bash
python3 -c "
from raglab.eval.dataset import load_questions
qs = load_questions('eval/questions.sample.jsonl')
print(len(qs), 'questions,', sum(q.answerable for q in qs), 'answerable')
"
```
