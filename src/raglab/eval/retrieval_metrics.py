"""Retrieval-only metrics: recall@k, hit-rate@k, MRR@10, nDCG@10 (§15).

All pure functions over `(question, ranked keys)` pairs. No I/O, no models, no
ordering assumptions beyond "rank 0 is best". That makes them cheap enough to run
over the whole retrieval matrix without touching the quota (§15: the retrieval
matrix is the difference between a feasible and an infeasible ablation
programme).

The graded-relevance convention, since it decides whether nDCG is even defined:

- A gold item that appears at rank *r* is judged *fully* relevant. Partial
  credit is not used, because there is no principled way to assign it to an
  overlap between a gold span and a retrieved chunk without a second judgement
  per candidate per question. `recall_overlap` below is provided for the case
  where that is wanted.
- `gold_docs` alone (no `gold_spans`) falls back to graded doc-level relevance:
  a result in a gold document counts 1.0, in another document 0.0.

`Q` is keyed by `qid`, never by question text. Two questions can share text, and
keying by text would merge their gold sets.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from raglab.index.rrf import DEFAULT_RRF_K


@runtime_checkable
class Question(Protocol):
    """Structural requirements for an eval question.

    Matches `eval.dataset.EvalQuestion`; declared as a protocol so the metrics
    module has no dependency on the dataset module and can be used against a
    plain dict in a test.
    """

    qid: str
    answerable: bool
    gold_spans: tuple[tuple[str, int, int], ...]
    gold_docs: tuple[str, ...]


def _gold_keys(q: Question) -> list[tuple[str, int, int]]:
    """Gold keys for recall, preferring spans over documents.

    Spans when present: they are what retrieval actually ranked, so recall@k
    computed on spans is defined (§9, §15). Documents alone are a coarser
    fallback and are reported as such.
    """
    if q.gold_spans:
        return list(q.gold_spans)
    return []


def doc_keys(q: Question) -> set[str]:
    return set(q.gold_docs)


def recall_at_k(
    ranked: Sequence[Any],
    gold: Iterable[Any],
    k: int,
    *,
    key_fn: Callable[[Any], Any] | None = None,
) -> float:
    """Fraction of gold items present in the top *k*.

    0.0 when there is no gold: a question with no evidence has no recall to
    score, and returning 0 would drag a corpus-wide average down for a question
    that was never answerable. `answerable = False` questions are excluded from
    retrieval averages by `aggregate()` for the same reason.

    With a single gold item this equals hit-rate@k (§2.4), which is why both are
    reported.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    if k <= 0:
        return 0.0
    top = list(ranked)[:k]
    keys = {key_fn(r) if key_fn else r for r in top}
    return len(gold_set & keys) / len(gold_set)


def hit_rate_at_k(
    ranked: Sequence[Any],
    gold: Iterable[Any],
    k: int,
    *,
    key_fn: Callable[[Any], Any] | None = None,
) -> float:
    """1.0 if *any* gold item is in the top *k*, else 0.0."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    if k <= 0:
        return 0.0
    keys = {key_fn(r) if key_fn else r for r in list(ranked)[:k]}
    return 1.0 if gold_set & keys else 0.0


def reciprocal_rank(
    ranked: Sequence[Any],
    gold: Iterable[Any],
    *,
    key_fn: Callable[[Any], Any] | None = None,
) -> float:
    """`1 / position of the first correct result`. 1-based position, so 1.0 for a
    top hit and 0.0 for no hit anywhere in the list."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for i, r in enumerate(ranked, start=1):
        key = key_fn(r) if key_fn else r
        if key in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    ranked: Sequence[Any],
    gold: Iterable[Any],
    k: int,
    *,
    key_fn: Callable[[Any], Any] | None = None,
) -> float:
    """Normalised discounted cumulative gain over the top *k*.

    `DCG = Σ (2^rel - 1) / log2(rank + 1)` with 1-based ranks, and `IDCG` from an
    ideal ordering of the same gold items. Gain is exponential in relevance so a
    relevant hit at rank 1 outweighs the same hit at rank 8 substantially rather
    than marginally.

    Returns 0.0 when there is no gold (IDCG is 0, and 0/0 is not a number worth
    propagating into an average).
    """
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0

    dcg = 0.0
    for i, r in enumerate(list(ranked)[:k], start=1):
        key = key_fn(r) if key_fn else r
        if key in gold_set:
            dcg += (2**1 - 1) / math.log2(i + 1)

    n_hits = min(len(gold_set), k)
    idcg = sum((2**1 - 1) / math.log2(i + 1) for i in range(1, n_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def span_coverage(
    ranked_keys: Sequence[Any],
    ranked_spans: dict[Any, tuple[str, int, int]],
    gold: Sequence[tuple[str, int, int]],
    k: int,
) -> float:
    """Character coverage of gold spans by the top-*k* results.

    The partial-credit metric, and the one that matters when chunk size is the
    variable under study: a 1024-token chunk overlapping 80% of a gold span is a
    different outcome from missing it, and plain recall@k calls both zero.

    `ranked_spans` maps a retrieved key to `(doc_id, start, end)` in **clean**
    space (§9) — the same space `gold_spans` lives in, which is the only reason
    this number is comparable to anything.

    Overlap is summed per gold span after merging, so two chunks each covering
    half a span score 1.0 rather than double counting.
    """
    if not gold:
        return 0.0
    top = list(ranked_keys)[:k]
    if not top:
        return 0.0
    by_doc: dict[str, list[tuple[int, int]]] = {}
    for key in top:
        span = ranked_spans.get(key)
        if span is None:
            continue
        doc_id, start, end = span
        by_doc.setdefault(str(doc_id), []).append((int(start), int(end)))

    scores: list[float] = []
    for gdoc, gstart, gend in gold:
        spans = by_doc.get(str(gdoc), [])
        if not spans:
            scores.append(0.0)
            continue
        # Merge overlapping retrieved spans first, so two chunks sharing
        # characters are not counted twice against one gold span.
        covered = 0
        cur_s, cur_e = spans[0]
        for s, e in sorted(spans)[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                covered += cur_e - cur_s
                cur_s, cur_e = s, e
        covered += cur_e - cur_s
        scores.append(min(1.0, max(0.0, covered) / max(1, gend - gstart)))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10, 20, 40)


@dataclass(slots=True)
class QuestionResult:
    """Per-question scores. Kept so a report can show which case failed."""

    qid: str
    answerable: bool
    recall_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    rr: float = 0.0
    ndcg: float = 0.0
    coverage: float = 0.0
    retrieved: int = 0
    latency_ms: float = 0.0


@dataclass(slots=True)
class RetrievalMetrics:
    """Corpus-level retrieval health.

    Recall/hit-rate/rr/ndcg are averaged over *answerable* questions only.
    `n_unanswerable` is carried so the report can state the split explicitly —
    a recall number averaged over a set that includes questions with no gold
    evidence is not comparable with one that does not, and silently mixing them
    is the most common way a retrieval table misleads.
    """

    n_questions: int = 0
    n_answerable: int = 0
    n_unanswerable: int = 0
    recall_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    mrr_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    coverage: float = 0.0
    latency: dict[str, float] = field(default_factory=dict)
    per_question: list[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_questions": self.n_questions,
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "hit_rate_at_k": {str(k): round(v, 4) for k, v in self.hit_rate_at_k.items()},
            "mrr_at_10": round(self.mrr_at_10, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "coverage": round(self.coverage, 4),
            "latency": {k: round(v, 2) for k, v in self.latency.items()},
        }


def score_question(
    q: Question,
    ranked_keys: Sequence[Any],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    ranked_spans: dict[Any, tuple[str, int, int]] | None = None,
    latency_ms: float = 0.0,
) -> QuestionResult:
    """Score one question against its ranked results."""
    gold = _gold_keys(q)
    res = QuestionResult(
        qid=q.qid,
        answerable=q.answerable,
        retrieved=len(ranked_keys),
        latency_ms=latency_ms,
    )
    if q.answerable and gold:
        for k in ks:
            res.recall_at_k[k] = recall_at_k(ranked_keys, gold, k)
            res.hit_rate_at_k[k] = hit_rate_at_k(ranked_keys, gold, k)
        res.rr = reciprocal_rank(ranked_keys, gold)
        res.ndcg = ndcg_at_k(ranked_keys, gold, 10)
        if ranked_spans:
            res.coverage = span_coverage(ranked_keys, ranked_spans, gold, 10)
    return res


def aggregate(results: Iterable[QuestionResult]) -> RetrievalMetrics:
    """Corpus-level aggregation. Averages over answerable questions only."""
    items = list(results)
    answerable = [r for r in items if r.answerable]
    m = RetrievalMetrics(
        n_questions=len(items),
        n_answerable=len(answerable),
        n_unanswerable=len(items) - len(answerable),
        per_question=items,
    )
    if not answerable:
        return m
    ks = sorted({k for r in answerable for k in r.recall_at_k})
    for k in ks:
        vals = [r.recall_at_k[k] for r in answerable if k in r.recall_at_k]
        if vals:
            m.recall_at_k[k] = sum(vals) / len(vals)
    ks = sorted({k for r in answerable for k in r.hit_rate_at_k})
    for k in ks:
        vals = [r.hit_rate_at_k[k] for r in answerable if k in r.hit_rate_at_k]
        if vals:
            m.hit_rate_at_k[k] = sum(vals) / len(vals)
    m.mrr_at_10 = sum(r.rr for r in answerable) / len(answerable)
    m.ndcg_at_10 = sum(r.ndcg for r in answerable) / len(answerable)
    cov = [r.coverage for r in answerable if r.coverage]
    if cov:
        m.coverage = sum(cov) / len(cov)
    lat = [r.latency_ms for r in items if r.latency_ms]
    if lat:
        from raglab.metrics import percentile

        m.latency = {
            "mean_ms": sum(lat) / len(lat),
            "p50_ms": percentile(lat, 50),
            "p95_ms": percentile(lat, 95),
        }
    return m


def evaluate(
    questions: Iterable[Question],
    ranked: dict[str, Sequence[Any]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    ranked_spans: dict[str, dict[Any, tuple[str, int, int]]] | None = None,
) -> RetrievalMetrics:
    """Score a whole question set against a `qid -> ranked keys` mapping.

    A question with no entry in `ranked` scores zero, not an exception: an
    ablation cell that failed to retrieve for one question should show a zero in
    that row, because an exception would silently drop it from the denominator and
    make a broken run look better than a working one.
    """
    spans = ranked_spans or {}
    results = [
        score_question(
            q,
            ranked.get(q.qid, []),
            ks=ks,
            ranked_spans=spans.get(q.qid),
        )
        for q in questions
    ]
    return aggregate(results)


def rrf_config_hash(lists: dict[str, Sequence[Any]], k: int = DEFAULT_RRF_K) -> str:
    """Small helper so an eval row records the fusion constant it used."""
    import hashlib
    import json

    payload = json.dumps({"k": k, "n_lists": len(lists)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
