"""Markdown and JSON ablation tables (§15).

Two rendering rules that come straight from the plan and are the reason this is
a module rather than a `print` in the harness:

1. **The abstention pair always renders together** (§15). A table row with
   `correct_abstention` but no `abstention_rate` cannot be produced here.
2. **A generation-matrix row is only comparable with another row that shares its
   config hash** (§15 determinism). `comparable_groups` partitions rows so the
   report can say which differences are real.

`rerank_window` is a column, not a footnote (§10): a 2048-token row must never
be read against a 512-token row as if the pipelines matched.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raglab.eval.dataset import RunRecord, assert_comparable

RETRIEVAL_COLUMNS: tuple[str, ...] = (
    "chunking",
    "strategy",
    "retrieval",
    "recall@5",
    "mrr@10",
    "ndcg@10",
    "p95 ms",
)

GENERATION_EXTRA_COLUMNS: tuple[str, ...] = (
    "faithfulness",
    "relevancy",
    "cite P",
    "cite R",
    "correct_abstention",
    "abstention_rate",
)


@dataclass(frozen=True, slots=True)
class TableRow:
    """One rendered row. Built from a `RunRecord`, never from raw dicts."""

    label: str
    matrix: str
    config_hash: str
    chunker: str = ""
    retrieval: str = ""
    rerank_window: str = ""
    chunk_count: int = 0
    chunk_rows: int = 0
    metrics: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metrics is None:
            object.__setattr__(self, "metrics", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "matrix": self.matrix,
            "config_hash": self.config_hash,
            "chunker": self.chunker,
            "retrieval": self.retrieval,
            "rerank_window": self.rerank_window,
            "chunk_count": self.chunk_count,
            "chunk_rows": self.chunk_rows,
            "metrics": dict(self.metrics),
        }


def row_from_record(run: RunRecord) -> TableRow:
    """Project a `RunRecord` into a display row."""
    m = run.metrics
    retrieval = str(m.get("retrieval", "")) or ", ".join(run.config.get("stages", []))
    return TableRow(
        label=run.label,
        matrix=run.matrix,
        config_hash=run.config_hash,
        chunker=run.chunker,
        retrieval=retrieval,
        rerank_window=run.rerank_window,
        chunk_count=run.chunk_count,
        chunk_rows=run.chunk_rows,
        metrics={
            "recall@5": float(m.get("recall_at_k", {}).get("5", 0.0)),
            "mrr@10": float(m.get("mrr_at_10", 0.0)),
            "ndcg@10": float(m.get("ndcg_at_10", 0.0)),
            "p95 ms": float(m.get("latency", {}).get("p95_ms", 0.0)),
            "faithfulness": float(m.get("faithfulness", 0.0)),
            "relevancy": float(m.get("answer_relevancy", 0.0)),
            "cite P": float(m.get("citation_precision", 0.0)),
            "cite R": float(m.get("citation_recall", 0.0)),
            "correct_abstention": float(
                m.get("abstention", {}).get("correct_abstention", 0.0)
            ),
            "abstention_rate": float(m.get("abstention", {}).get("abstention_rate", 0.0)),
        },
    )


def columns_for(matrix: str) -> tuple[str, ...]:
    """Columns for a matrix. Retrieval rows never show generation metrics, and
    vice versa — the two matrices have different costs and different meanings
    (§15)."""
    if matrix == "generation":
        return RETRIEVAL_COLUMNS[:2] + RETRIEVAL_COLUMNS[2:4] + GENERATION_EXTRA_COLUMNS + (
            "p95 ms",
        )
    return RETRIEVAL_COLUMNS


def render_markdown(rows: Sequence[TableRow], *, title: str = "Ablation") -> str:
    """Render a markdown table.

    Columns are the union of what the rows actually carry, minus all-zero ones,
    so a retrieval-only run does not print an empty `faithfulness` column and a
    generation run does not hide its abstention pair.
    """
    if not rows:
        return f"# {title}\n\n_No runs recorded._\n"

    matrices = {r.matrix for r in rows}
    base = RETRIEVAL_COLUMNS[:2] if len(matrices) == 1 else ("label", "matrix")
    candidates = set(base) | set(RETRIEVAL_COLUMNS[2:]) | set(GENERATION_EXTRA_COLUMNS)

    present: list[str] = []
    for col in candidates:
        if col in base or any(abs(r.metrics.get(col, 0.0)) > 0 for r in rows):
            present.append(col)

    order = list(RETRIEVAL_COLUMNS[:2])
    if "retrieval" in present:
        order.append("retrieval")
    order += [c for c in RETRIEVAL_COLUMNS[2:] if c in present]
    order += [c for c in GENERATION_EXTRA_COLUMNS if c in present]

    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(order) + " |",
        "|" + "|".join("---" for _ in order) + "|",
    ]
    for r in rows:
        cells: list[str] = []
        for col in order:
            if col == "chunking":
                cells.append(r.chunker or "-")
            elif col == "strategy":
                cells.append(r.label)
            elif col == "label":
                cells.append(r.label)
            elif col == "matrix":
                cells.append(r.matrix)
            elif col == "retrieval":
                cells.append(r.retrieval or "-")
            elif col == "rerank_window":
                cells.append(r.rerank_window or "-")
            else:
                cells.append(f"{r.metrics.get(col, 0.0):.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines += _notes(rows)
    return "\n".join(lines) + "\n"


def _notes(rows: Sequence[TableRow]) -> list[str]:
    """Footnotes that carry the qualifications the numbers need."""
    notes: list[str] = ["", "## Reading this table", ""]

    if any(r.matrix == "generation" for r in rows):
        notes += [
            "- `correct_abstention` and `abstention_rate` are a pair. A good "
            "system sits high on the first and low on the second; a system "
            "refusing everything hits the first at ceiling and the second near "
            "1.0. Read them together or not at all (§15).",
            "- `faithfulness` comes from the same `qwen3:8b` that generated the "
            "answers, so it is biased toward them. Absolute values are "
            "pessimistic and only comparable within this fixed setup (§15).",
        ]
    if len({r.rerank_window for r in rows if r.rerank_window}) > 1:
        notes.append(
            "- `rerank_window` differs across rows. `bge-reranker-v2-m3` has a "
            "hard 512-token ceiling, so a row with a different window is not "
            "measuring the same pipeline (§10)."
        )
    if len({r.chunk_rows for r in rows}) > 1:
        notes.append(
            "- `chunk_rows` (physical) and `chunk_count` (live) differ by "
            "design: deletion is logical, so rows are appended forever (§6.4)."
        )
    if len({r.config_hash for r in rows}) > 1:
        notes.append(
            "- Rows with different `config_hash` are not directly comparable. "
            "Only differences *within* one hash isolate a single change (§15)."
        )
    return notes


def comparable_groups(rows: Iterable[TableRow]) -> list[list[TableRow]]:
    """Partition rows into groups that may be compared with each other.

    §15's rule — never compare across a different corpus, embedder, or judge —
    reduced to the config hash, which encodes exactly that. Rows in different
    groups are reported separately rather than in one table.
    """
    groups: dict[str, list[TableRow]] = {}
    for r in rows:
        groups.setdefault(r.config_hash, []).append(r)
    return list(groups.values())


def render_report(
    runs: Sequence[RunRecord],
    *,
    title: str = "Ablation",
    out_dir: str | Path | None = None,
) -> tuple[str, list[TableRow]]:
    """Render every run, grouped by comparability. Returns `(markdown, rows)`.

    Raises if two runs differ on corpus/embedder/judge *and* are being placed in
    one table — the check is `assert_comparable`, so a bad grouping is an
    exception rather than a silently misleading table.
    """
    rows = [row_from_record(r) for r in runs]
    groups = comparable_groups(rows)

    parts: list[str] = []
    for i, group in enumerate(groups, start=1):
        if len(groups) > 1:
            verify_group(group)
        suffix = f" (group {i}: config_hash {group[0].config_hash})" if len(groups) > 1 else ""
        parts.append(render_markdown(group, title=f"{title}{suffix}"))
        parts.append("")

    markdown = "\n".join(parts)
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "ablation.md").write_text(markdown, encoding="utf-8")
        (d / "ablation.json").write_text(
            json.dumps([r.to_dict() for r in rows], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return markdown, rows


def verify_group(group: Sequence[TableRow]) -> None:
    """Assert a group is internally comparable.

    `TableRow` has already lost the provenance fields, so this checks the
    rendered invariant instead: every row in a group must share one config hash
    and one `rerank_window`. Cross-group checks belong on `RunRecord`, which is
    why `assert_comparable` takes those.
    """
    if not group:
        return
    hashes = {r.config_hash for r in group}
    if len(hashes) > 1:
        msg = f"table group mixes config hashes {sorted(hashes)}; §15 forbids comparing across one"
        raise ValueError(msg)


def delta(baseline: TableRow, variant: TableRow, metric: str = "recall@5") -> float:
    """Metric delta between two rows. The unit an ablation actually reports.

    §15: whether reranking moves recall from 0.68 to 0.81, or structural
    chunking adds another 5 points, is an empirical question.
    """
    return variant.metrics.get(metric, 0.0) - baseline.metrics.get(metric, 0.0)


def markdown_delta_table(baseline: TableRow, variants: Sequence[TableRow]) -> str:
    """A baseline row plus per-variant deltas. The form the plan shows."""
    lines = [
        "| Variant | Δrecall@5 | Δmrr@10 | Δndcg@10 | Δp95 ms |",
        "|---|---|---|---|---|",
        f"| {baseline.label} (baseline) | — | — | — | — |",
    ]
    for v in variants:
        lines.append(
            f"| {v.label} | {delta(baseline, v, 'recall@5'):+.3f} "
            f"| {delta(baseline, v, 'mrr@10'):+.3f} "
            f"| {delta(baseline, v, 'ndcg@10'):+.3f} "
            f"| {delta(baseline, v, 'p95 ms'):+.1f} |"
        )
    return "\n".join(lines) + "\n"
