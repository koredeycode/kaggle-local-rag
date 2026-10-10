"""Reciprocal Rank Fusion.

§2.2 / §11: RRF combines ranked lists without needing their scores to be
comparable. For each row, sum `1 / (k + rank)` across every list that contained
it.

The reason this exists, and the reason it is rank-based, is worth stating
plainly because it is the usual source of hybrid-retrieval bugs: a dense cosine
similarity lives in [-1, 1] and a BM25 score is unbounded and corpus-dependent.
Any attempt to combine them directly — min-max normalise, z-score, add — has to
answer "normalise against what population?", and getting that wrong silently
shifts every result. Fusion on *positions* sidesteps the question entirely: rank
1 in both lists is simply better than rank 1 in one and rank 30 in the other,
regardless of what either system's score scale means.

`k = 60` is from the original Cormack et al. paper and is notably insensitive;
it is exposed as a constant rather than a tuning knob for exactly that reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class FusedItem:
    """One fused row."""

    key: str | int
    score: float
    """Sum of `1 / (k + rank)` over every list containing the key."""

    ranks: tuple[int, ...]
    """0-based rank in each list, in list order. Two entries means both stages
    returned it; a single entry means only one did."""

    sources: tuple[str, ...]
    """Names of the lists that returned it, in the same order as `ranks`."""

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "score": self.score,
            "ranks": list(self.ranks),
            "sources": list(self.sources),
        }


def rrf_score(rank: int, k: int = DEFAULT_RRF_K) -> float:
    """`1 / (k + rank)` for a 0-based rank.

    Note the rank is 0-based here, so a first-place result contributes
    `1/(k+0)` = the largest single contribution. The `+k` offset exists to
    damp the influence of the very top ranks relative to the tail — without it,
    a single #1 would dominate a fusion of 40 candidates.
    """
    if rank < 0:
        msg = f"rank must be >= 0, got {rank}"
        raise ValueError(msg)
    if k <= 0:
        msg = f"k must be > 0, got {k}"
        raise ValueError(msg)
    return 1.0 / (k + rank)


def fuse(
    ranked_lists: Mapping[str, Sequence[str | int]],
    *,
    k: int = DEFAULT_RRF_K,
    top_n: int | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[FusedItem]:
    """Fuse ranked lists into one ordering.

    `ranked_lists` maps a source name to an already-ranked sequence of keys, best
    first. The returned order is by descending fused score; ties break on first
    appearance across sources, so the result is deterministic — an ablation table
    has to be diffable, and a tie broken by dict iteration order is not.

    `weights` scales a source's contribution. Default 1.0 for every source,
    which is the correct starting point: introducing weights without a measured
    reason is how a hybrid setup starts encoding a guess.
    """
    if not ranked_lists:
        return []

    contributions: dict[str | int, float] = {}
    ranks: dict[str | int, list[int]] = {}
    sources: dict[str | int, list[str]] = {}
    first_seen: dict[str | int, tuple[int, int]] = {}

    for source_idx, (source, keys) in enumerate(ranked_lists.items()):
        weight = (weights or {}).get(source, 1.0)
        for rank, key in enumerate(keys):
            contributions[key] = contributions.get(key, 0.0) + weight * rrf_score(rank, k)
            ranks.setdefault(key, []).append(rank)
            sources.setdefault(key, []).append(source)
            first_seen.setdefault(key, (source_idx, rank))

    items = [
        FusedItem(
            key=key,
            score=score,
            ranks=tuple(ranks[key]),
            sources=tuple(sources[key]),
        )
        for key, score in contributions.items()
    ]
    items.sort(key=lambda it: (-it.score, first_seen[it.key]))
    return items[:top_n] if top_n is not None else items


def fuse_scores(
    ranked_lists: Mapping[str, Sequence[str | int]],
    *,
    k: int = DEFAULT_RRF_K,
) -> dict[str | int, float]:
    """Just the scores. Convenient for the ablation harness."""
    return {it.key: it.score for it in fuse(ranked_lists, k=k)}


def explain(fused: Iterable[FusedItem]) -> list[str]:
    """Human-readable provenance per item. For `docs/retrieval.md` and debugging.

    Shows where each key came from, which is the fastest way to answer "why is
    this row ranked first" when the answer is not obvious from the scores.
    """
    out: list[str] = []
    for it in fused:
        parts = ", ".join(f"{s}#{r}" for s, r in zip(it.sources, it.ranks, strict=True))
        out.append(f"{it.key} score={it.score:.6f} [{parts}]")
    return out
