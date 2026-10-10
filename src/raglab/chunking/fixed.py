"""Fixed-size chunking with overlap.

§10: N tokens per chunk, overlap tokens repeated between neighbours.

Strength: predictable, no model needed, deterministic, and the only strategy that
behaves identically on every corpus — which is what makes it the baseline the
ablation table is built against.

Weakness: it splits mid-thought. A chunk can begin "…the fee was reduced to" and
be unretrievable, because the fact that makes it matchable is in the previous
chunk. Overlap mitigates this without removing it, and at a cost: index size
grows and near-duplicate candidates start crowding the top-k (§2.1).

This is where the plan's default lives: `fixed` at 512 with 64 overlap, then
`structural`.
"""

from __future__ import annotations

from raglab.chunking.base import BaseChunker, Span, register
from raglab.types import Document

DEFAULT_SIZE = 512
DEFAULT_OVERLAP = 64


@register("fixed")
class FixedChunker(BaseChunker):
    """Token-count windows with a fixed stride.

    Implemented over token spans rather than characters, because `chunk_size` is
    specified in tokens throughout the plan and a character count would drift
    silently on any non-Latin text — the exact failure §13 warns about for
    context packing, avoided here at the source.
    """

    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        tokenizer=None,  # noqa: ANN001
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        if size <= 0:
            msg = f"size must be positive, got {size}"
            raise ValueError(msg)
        if overlap < 0:
            msg = f"overlap must be >= 0, got {overlap}"
            raise ValueError(msg)
        if overlap >= size:
            # stride <= 0 means an infinite loop, not a subtle bug.
            msg = f"overlap ({overlap}) must be < size ({size})"
            raise ValueError(msg)
        self.size = size
        self.overlap = overlap
        self.stride = size - overlap
        self._finalise()

    allows_overlap = True

    def _effective_params(self) -> dict[str, int | float]:
        return {"size": self.size, "overlap": self.overlap}

    def spans(self, doc: Document) -> list[Span]:
        tokens = self.tokenizer.encode_spans(doc.text)
        if not tokens:
            return []
        out: list[Span] = []
        i = 0
        while i < len(tokens):
            window = tokens[i : i + self.size]
            if not window:
                break
            out.append(
                Span(
                    start=window[0].start,
                    end=window[-1].end,
                    token_count=len(window),
                )
            )
            if i + self.size >= len(tokens):
                break
            i += self.stride
        return out
