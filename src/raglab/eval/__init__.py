"""Evaluation harness (§15).

Retrieval metrics, generation metrics, question sets, and reporting. The judge
(`judge.py`) and ablation runner (`ablate.py`) arrive with the phases that own
them; the retrieval half is complete and GPU-free, which is what makes the large
retrieval matrix affordable on a 30-hour weekly quota (§15, §21).
"""

from raglab.eval.dataset import (
    EvalQuestion,
    EvalSet,
    RunRecord,
    assert_comparable,
    load_questions,
)
from raglab.eval.gen_metrics import (
    AbstentionMetrics,
    CitationMetrics,
    GenerationMetrics,
    abstention_metrics,
    citation_metrics,
    extract_markers,
    segment_claims,
)
from raglab.eval.report import TableRow, render_markdown, render_report
from raglab.eval.retrieval_metrics import (
    RetrievalMetrics,
    aggregate,
    evaluate,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    span_coverage,
)

__all__ = [
    "AbstentionMetrics",
    "CitationMetrics",
    "EvalQuestion",
    "EvalSet",
    "GenerationMetrics",
    "RetrievalMetrics",
    "RunRecord",
    "TableRow",
    "abstention_metrics",
    "aggregate",
    "assert_comparable",
    "citation_metrics",
    "evaluate",
    "extract_markers",
    "hit_rate_at_k",
    "load_questions",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "render_markdown",
    "render_report",
    "segment_claims",
    "span_coverage",
]
