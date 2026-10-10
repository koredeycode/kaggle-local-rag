"""Generation-side metrics: faithfulness, relevancy, citation P/R, abstention (§15).

The abstention pair is the part worth reading carefully, because §15 is emphatic
about it and it is trivially gameable.

Scoring refusals as correct when gold evidence is absent is the right behaviour
to reward. On its own it is also a metric that a system which refuses everything
scores 15/15 on. So `correct_abstention` is *never* reported alone:

| Metric | Reads as |
|---|---|
| `correct_abstention` | of the unanswerable questions, how many were correctly refused |
| `abstention_rate` | of **all** questions, how many were refused at all |

A good system sits high on the first and low on the second. A degenerate one
shows up as `correct_abstention` at ceiling with `abstention_rate` climbing
toward 1.0. `GenerationMetrics.validate` refuses to build a report that carries
one without the other, so the degenerate configuration cannot even be rendered.

Faithfulness and relevancy are produced by the local judge (`eval/judge.py`) and
are therefore only comparable against themselves — §15 documents the self-
preference bias that comes from the judge being the same model as the generator.
Nothing in this module treats those numbers as absolute.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from raglab.errors import RagError

#: `[S1]`, `[S12]`. The citation marker grammar (§13 labels sources this way).
MARKER_RE = re.compile(r"\[S(\d{1,3})\]", re.IGNORECASE)

#: Sentence split for claim extraction. Deliberately simple; the judge's
#: faithfulness verdict is the real signal, this only proposes candidates.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

MIN_CITATION_PRECISION_SENTENCES = 3
"""Below this many sentences, citation precision is not reported.

A precision computed over one sentence is a boolean, and printing `1.0` next to
it invites reading it as a quality number."""


def extract_markers(text: str) -> list[str]:
    """All `[Sn]` markers in order of appearance, including duplicates."""
    return [f"S{m.group(1)}" for m in MARKER_RE.finditer(text)]


def extract_unique_markers(text: str) -> set[str]:
    return {f"S{m.group(1)}" for m in MARKER_RE.finditer(text)}


def split_sentences(text: str) -> list[str]:
    """Split into sentences, dropping empties. Not a linguistics claim."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


@dataclass(frozen=True, slots=True)
class SentenceClaim:
    """One sentence and the marker attached to it, if any."""

    text: str
    marker: str | None
    index: int


def segment_claims(answer: str) -> list[SentenceClaim]:
    """Attribute each sentence to the first marker that follows it.

    Follows rather than precedes, because that is how the prompt asks for it
    (§14: cite every claim) and how the model emits it. A sentence with no
    trailing marker is an unverified claim — the dangerous kind, because it
    reads as correct (§14).
    """
    claims: list[SentenceClaim] = []
    for i, sent in enumerate(split_sentences(answer)):
        m = MARKER_RE.search(sent)
        claims.append(SentenceClaim(sent, f"S{m.group(1)}" if m else None, i))
    return claims


# ---------------------------------------------------------------------------
# Citation precision / recall
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CitationMetrics:
    """How the answer's citations line up with the claims it makes.

    - `precision` = cited sentences / sentences carrying a citation. Penalises
      **over-citation** only weakly, because citing everything looks like rigour
      and is noise; it is caught by `over_citation_ratio` below, which is where
      that failure actually shows up.
    - `recall` = sentences with a citation / sentences making a factual claim.
      Catches **unsupported claims**.
    - `unverified_ratio` = sentences with no citation / all sentences. The
      headline number for the failure mode that matters.
    - `broken_markers` = markers referencing a source that does not exist.
      §17.3 renders these as visibly broken chips rather than dropping them.
    """

    precision: float
    recall: float
    n_sentences: int
    n_cited_sentences: int
    n_unverified: int
    over_citation_ratio: float = 0.0
    broken_markers: tuple[str, ...] = ()
    unverified_texts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "n_sentences": self.n_sentences,
            "n_cited_sentences": self.n_cited_sentences,
            "n_unverified": self.n_unverified,
            "over_citation_ratio": round(self.over_citation_ratio, 4),
            "broken_markers": list(self.broken_markers),
            "unverified_texts": list(self.unverified_texts),
        }


def citation_metrics(
    answer: str,
    *,
    valid_markers: Iterable[str],
    n_sources: int,
) -> CitationMetrics:
    """Score one answer's citation behaviour.

    `valid_markers` is the set of markers the pipeline actually supplied;
    `n_sources` is how many sources existed. A marker outside `valid_markers` is
    a hallucinated citation (§14 failure mode 1) and is counted, not hidden.

    Precision is deliberately narrow. Every marker the model emits that resolves
    is, mechanically, a valid citation — so the only precision failure available
    is *over-citation*, and that is what `precision` measures:

        precision = min(markers_used / markers_supplied,
                        1 / markers_per_cited_sentence)

    The first term punishes spraying every source onto every claim. The second
    punishes citing several sources for one claim, which is the subtler version
    of the same failure. Both are 1.0 in the healthy case: each source used, one
    marker per claim.
    """
    valid = set(valid_markers)
    claims = segment_claims(answer)
    n = len(claims)
    if n == 0:
        return CitationMetrics(0.0, 0.0, 0, 0, 0)

    cited = [c for c in claims if c.marker is not None]
    unverified = [c for c in claims if c.marker is None]
    all_markers = extract_markers(answer)
    distinct_used = {m for m in all_markers if m in valid}
    broken = tuple(sorted({m for m in all_markers if m not in valid}))

    precision = 1.0
    if valid:
        precision = min(1.0, len(distinct_used) / len(valid))
    markers_per_cited = len(all_markers) / len(cited) if cited else 1.0
    if len(cited) >= MIN_CITATION_PRECISION_SENTENCES:
        precision = min(precision, 1.0 / markers_per_cited)

    # Over-citation ratio: markers emitted beyond one per cited sentence.
    excess = max(0, len(all_markers) - len(cited))
    over_ratio = excess / len(all_markers) if all_markers else 0.0

    recall = len(cited) / n
    return CitationMetrics(
        precision=precision,
        recall=recall,
        n_sentences=n,
        n_cited_sentences=len(cited),
        n_unverified=len(unverified),
        over_citation_ratio=over_ratio,
        broken_markers=broken,
        unverified_texts=tuple(c.text for c in unverified),
    )


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AbstentionMetrics:
    """The §15 pair, which is never reported apart."""

    correct_abstention: float
    abstention_rate: float
    n_unanswerable: int
    n_correct_abstentions: int
    n_total: int
    n_abstentions: int
    """Refusals on *all* questions, including answerable ones. The second half
    of the pair: a high value here is the system being useless."""

    false_abstentions: int = 0
    """Answerable questions the system refused. Always reported next to the
    rest, because `correct_abstention` alone hides exactly this."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct_abstention": round(self.correct_abstention, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "n_unanswerable": self.n_unanswerable,
            "n_correct_abstentions": self.n_correct_abstentions,
            "n_total": self.n_total,
            "n_abstentions": self.n_abstentions,
            "false_abstentions": self.false_abstentions,
        }

    def is_degenerate(self, *, threshold: float = 0.8) -> bool:
        """Heuristic flag: near-perfect refusal scoring with a high refusal rate.

        Not a pass/fail gate — a warning that the row needs the human sample
        (§15) before anyone reads it as a quality result.
        """
        return self.correct_abstention >= threshold and self.abstention_rate >= threshold


def abstention_metrics(
    *,
    abstained_by_qid: dict[str, bool],
    answerable_by_qid: dict[str, bool],
) -> AbstentionMetrics:
    """Build the §15 pair.

    `abstained_by_qid` covers every question in the run, answerable or not. A
    missing qid is treated as "did not abstain" rather than skipped, so a run
    that crashed halfway produces a visibly bad score instead of a flattering
    one computed over the survivors.
    """
    qids = set(answerable_by_qid) | set(abstained_by_qid)
    unanswerable = [q for q in qids if not answerable_by_qid.get(q, True)]

    if not unanswerable:
        # Not a warning: an error. `correct_abstention` over a set with no
        # unanswerable questions is a division by zero wearing a percentage
        # costume, and §15 requires the pair precisely because the single
        # number is gameable.
        msg = (
            "abstention_metrics called with no unanswerable questions; §15 "
            "requires correct_abstention to be reported alongside the "
            "abstention rate over a set containing unanswerable cases"
        )
        raise RagError(msg)
    if not qids:
        msg = "abstention_metrics called with no questions at all"
        raise RagError(msg)

    answerable = [q for q in qids if answerable_by_qid.get(q, True)]

    n_correct = sum(1 for q in unanswerable if abstained_by_qid.get(q, False))
    n_abstained = sum(1 for q in qids if abstained_by_qid.get(q, False))
    n_false = sum(1 for q in answerable if abstained_by_qid.get(q, False))

    return AbstentionMetrics(
        correct_abstention=n_correct / len(unanswerable) if unanswerable else 0.0,
        abstention_rate=n_abstained / len(qids),
        n_unanswerable=len(unanswerable),
        n_correct_abstentions=n_correct,
        n_total=len(qids),
        n_abstentions=n_abstained,
        false_abstentions=n_false,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GenerationMetrics:
    """One ablation row's generation-side numbers (§15)."""

    faithfulness: float = 0.0
    """Judge score: is every claim supported by its cited span? Pessimistic in
    absolute terms — an 8B judge is a weak judge, and it is also the generator
    (§15). Comparable only against itself."""

    answer_relevancy: float = 0.0
    """Judge score: did the answer address the question?"""

    citation_precision: float = 0.0
    citation_recall: float = 0.0
    unverified_ratio: float = 0.0
    abstention: AbstentionMetrics | None = None
    n_citation_failures: int = 0
    """Broken markers, summed. §16 counts these as an observability counter too."""

    per_question: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        """Refuse an incomplete row.

        The rule from §15: reporting either abstention number alone is worse than
        reporting neither, because it invites exactly the conclusion the other
        number rules out. So a row carrying one without the other is an error,
        not a table cell to fill in later.
        """
        if self.abstention is None:
            msg = (
                "GenerationMetrics.abstention is required; §15 forbids reporting "
                "correct_abstention without abstention_rate, since a system that "
                "refuses everything scores 15/15 on the unanswerable set"
            )
            raise RagError(msg)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        d: dict[str, Any] = {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "citation_precision": round(self.citation_precision, 4),
            "citation_recall": round(self.citation_recall, 4),
            "unverified_ratio": round(self.unverified_ratio, 4),
            "n_citation_failures": self.n_citation_failures,
            "abstention": self.abstention.to_dict(),
        }
        d["abstention_degenerate"] = self.abstention.is_degenerate()
        return d


def aggregate_generation(
    per_question: Sequence[dict[str, Any]],
    *,
    abstention: AbstentionMetrics,
    faithfulness_values: Sequence[float] = (),
    relevancy_values: Sequence[float] = (),
) -> GenerationMetrics:
    """Fold per-question judge and citation outputs into one row.

    Judge values are passed separately because they come from a model and a
    failed judge call must not be silently averaged in as 0.0 — a judge outage
    would then look like an unfaithful system rather than a broken harness.
    """
    if not faithfulness_values:
        msg = (
            "aggregate_generation received no faithfulness values; an 8B judge "
            "is a weak judge, and reporting 0.0 for 'it did not run' is the "
            "worst of both options (§15)"
        )
        raise RagError(msg)

    cp = [p["citation_precision"] for p in per_question if "citation_precision" in p]
    cr = [p["citation_recall"] for p in per_question if "citation_recall" in p]
    uv = [p["unverified_ratio"] for p in per_question if "unverified_ratio" in p]

    return GenerationMetrics(
        faithfulness=sum(faithfulness_values) / len(faithfulness_values),
        answer_relevancy=(
            sum(relevancy_values) / len(relevancy_values) if relevancy_values else 0.0
        ),
        citation_precision=(sum(cp) / len(cp)) if cp else 0.0,
        citation_recall=(sum(cr) / len(cr)) if cr else 0.0,
        unverified_ratio=(sum(uv) / len(uv)) if uv else 0.0,
        abstention=abstention,
        n_citation_failures=sum(len(p.get("broken_markers", ())) for p in per_question),
        per_question=list(per_question),
    )
