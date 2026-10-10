# Chunking

Companion to §10. Records the four strategies, their trade-offs, and the
constraint that reshapes the ablation grid.

## Why this is the highest-leverage decision

Chunking is the single highest-leverage decision in naive RAG, and it is the main
subject of the ablation grid. Three reasons it dominates:

- A chunk boundary is a hard decision the retriever cannot undo. A fact split
  across two chunks may match neither, or match only the half that reads as
  nonsense in isolation.
- Chunk size trades two failure modes against each other. Too small and the
  generator receives fragments it cannot reason across — "Term 1 is ₦45,000" then
  "…which is payable before the resumption date". Too large and the embedding is
  diluted by content that does not match the query, so recall falls.
- Errors here are silent. Retrieval just returns worse results, and the cause is
  three layers upstream of where anyone looks.

## One interface, four strategies

```python
class Chunker(Protocol):
    name: str
    def chunk(self, doc: Document) -> list[Chunk]: ...
```

One interface is what makes the grid possible. Without it each strategy needs its
own indexing path and the comparison is not controlled.

The interface is split deliberately: a strategy returns **spans** (offsets), and
the base class resolves offsets into text and builds `Chunk` records. No strategy
can forget to record `clean_start`/`clean_end` and silently break the citation
contract. Offsets are the thing that must be right; text is derivable.

## The strategies

| Strategy | Idea | Strengths | Weaknesses |
|---|---|---|---|
| `fixed` | N tokens, overlap | Predictable, no model needed, deterministic | Splits mid-thought |
| `recursive` | Split on the best separator, recurse | Respects natural boundaries | Still ignores meaning |
| `structural` | Never cross a heading boundary | Best for MD/DOCX/HTML | Coarse; useless for flat prose |
| `semantic` | Break where embedding similarity dips | Tracks topic shifts | An embedding pass per document |

Defaults per §10: `fixed` at 512 with 64 overlap, `context_top_n = 6`, then
`structural`.

### `fixed` — the baseline

Token windows at a fixed stride, implemented over token spans rather than
characters. `chunk_size` is specified in tokens throughout, and a character count
drifts silently on any non-Latin text — the exact failure §13 warns about for
context packing, avoided here at the source.

Overlap is why chunks can catch a fact spanning a boundary, and why the index
grows and near-duplicates start crowding the top-k (§2.1).

### `recursive` — separator preference

The point is *preference order*: a paragraph break beats a sentence break beats a
space beats mid-word. `fixed` has no preference and cuts wherever the token count
lands.

Two details that matter more than they look:

- **Merge-back.** Consecutive small pieces are coalesced up to `size`, measured in
  tokens with the real tokenizer. Without it, a document of short paragraphs
  produces 200-token chunks when 512 were requested, and the chunk-size ablation
  measures the wrong thing.
- **Offsets stay relative during recursion.** Each level tracks a cursor relative
  to its own base offset. Accumulating an absolute end and re-adding the base
  compounds at every piece, producing spans that run past the document and land in
  the wrong place — while still looking plausible. This was a real bug found by
  the ordering assertion.

### `structural` — the heading tree

Each heading section becomes one chunk, and every chunk carries its full
breadcrumb. §9 already prepends that breadcrumb to the embedded text, so a chunk
reading only "Term 1 is ₦45,000" becomes "Lagos State > Education > Fee Structure
2026 / Term 1 is ₦45,000" and becomes retrievable. This costs nothing at inference
time and reliably beats larger chunk sizes in practice.

It follows the document's own heading tree rather than re-deriving structure from
text, because parsing already recovered it.

Two properties that are not subtleties:

- **It does not support overlap.** Overlap reintroduces text across the heading
  boundary the strategy exists to preserve. `overlap != 0` raises rather than
  clamping — a silent clamp would let `chunk_overlap=64` look applied to a
  `structural` build while changing nothing.
- **It degrades rather than failing on flat prose.** A document with no headings
  yields one implicit root section, which is then size-capped like any other. It
  becomes `fixed`, not one unusable 5000-token chunk. That is why `fixed` remains
  the baseline.

`min_size` folds runs of tiny sections into the section that *follows* them, not
the one before, so the merged chunk takes its successor's breadcrumb. Merging
backward would leave the chunk labelled with the earlier, shallower heading while
its body is mostly the later section's content.

### `semantic` — where meaning shifts

Sentences are grouped into windows, embedded in one batch, and a boundary is
proposed wherever consecutive window cosine similarity falls below `threshold`.
Windows are then packed to `size`, so the output count is still governed by `size`
and the semantic signal only decides *where inside* that budget to cut.

The cost is real and it is why §15a nominates `semantic` as the first arm to drop:
it needs an embedding pass per document, on top of the pass for the resulting
chunks.

`threshold` is a genuine parameter with no principled default. `0.5` is a
starting point to be measured, not a constant, and it is part of the §6.5 index
signature so changing it forces a rebuild rather than a mixed index.

## The reranker's 512-token ceiling reshapes the grid

`bge-reranker-v2-m3` has `max_len 512`. Any longer chunk is **silently truncated**
inside the cross-encoder — the component this project expects to produce its
single largest quality jump. Ablating 1024 and 2048 without handling this measures
truncation, not chunking.

The plan takes option 3: window to a **query-aware head** — `heading_path`
prepended, plus a query-aware selection of the most relevant sentences inside the
chunk. That costs one embedding pass per candidate rather than several
cross-encoder passes, and it composes with §13 compression instead of duplicating
it. It is also an ablation worth having on its own: compress-then-rerank versus
rerank-then-compress.

Whichever is implemented, `rerank_window` is a **column in the ablation table**,
not a footnote, so a 2048-token row is never read against a 512-token row as if
the pipelines matched.

## Small-to-big

Index small chunks (~200 tokens) for precise matching, return the parent region
(~1000 tokens) to the generator. Small chunks retrieve sharply; large chunks read
coherently.

It resolves the precision-versus-recall tension that otherwise forces a single
compromise. Parents are contiguous regions of the same document, and expansion
happens *after* ranking, so it costs nothing when the gate abstains.

It wraps a child chunker rather than being one — the child strategy determines the
boundaries, this adds the parent layer.

`expand()` falls back to the child's own text when no parent span is recorded, and
clamps a stale parent span to the document length. Returning truncated context is
strictly better than raising: a citation panel must never be the thing that
crashes a query.

## What the tests pin

Each of these caught or guards a real failure:

- Size bounds, offset validity, and offset/text agreement for every strategy.
- **Ordering.** Spans must be ordered even when overlap is allowed. Without the
  check, an overlap longer than one chunk produces spans running backwards through
  the document.
- **Overlap is per-strategy.** `fixed` and `recursive` declare `allows_overlap`;
  `structural` and `semantic` do not, and validation enforces the declaration
  rather than trusting each strategy to remember.
- **Determinism.** Same input, same spans. Boundaries feed the §6.4 ordinal
  invariant and the citation offsets, so a chunker returning different spans on
  identical input makes every stored offset quietly unreproducible.
- **`structural` starts at headings**, or at 0 when the document opens with
  content before any heading.
- Out-of-range and empty spans are rejected by `validate`, not trusted.

## Current limitations

- The `mock` embedder used as `semantic`'s default makes **no** semantic claim. It
  makes boundary *positions* plausible, which is exactly what makes it dangerous
  if one leaks into a quality claim. `docs/chunking.md` says so; so should
  anything reporting a `semantic` delta.
- `structural`'s section detection trusts `Document.heading_path_at` from the
  loaders, which are Phase 1. It currently filters empty headings and collapses
  duplicate offsets, because a trailing newline parsed as an empty heading at
  offset 0 produces a second section starting at 0 — duplicating the document
  start and yielding out-of-order spans.
- No strategy has been measured against a real corpus. Every quality statement
  here is a hypothesis awaiting Phase 2.
