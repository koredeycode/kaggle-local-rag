"""Recursive chunking.

§10: split on the best available separator, recurse into pieces that are still
too large, stop at one that fits.

The point of the recursion is *preference order*. A paragraph boundary is a
better place to cut than a sentence boundary, which is better than a space,
which is better than mid-word. `fixed` has no such preference and cuts wherever
the token count lands; `recursive` respects the document's own structure without
needing to have parsed it.

This matters most for prose. In Markdown and HTML, `structural` beats both — but
for a flat TXT file with no headings, `structural` is useless and this is the
strategy that recovers natural boundaries (§10).

Weakness, stated plainly: it still ignores meaning. A chunk boundary at a
sentence boundary may fall exactly between two sentences that only make sense
together. That is the gap `semantic` closes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from raglab.chunking.base import BaseChunker, Span, register
from raglab.types import Document

DEFAULT_SIZE = 512
DEFAULT_OVERLAP = 64

#: Most-preferred first. Paragraph > line break > sentence end > clause > word.
SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
)
"""Ordered by preference. `""` is the last resort and splits mid-word, which is
why reaching it means the document is one enormous unbroken token run."""

_WORD_RE = re.compile(r"\S+")


@register("recursive")
class RecursiveChunker(BaseChunker):
    """Separator-preference chunking.

    Merges consecutive small pieces back up to `size`, so a document of short
    paragraphs does not produce 200-token chunks when 512 were requested. That
    merge-back is the difference between this strategy and naive recursive
    splitting, and without it the chunk-size ablation measures the wrong thing.
    """

    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        separators: Sequence[str] | None = None,
        tokenizer=None,  # noqa: ANN001
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        if size <= 0:
            msg = f"size must be positive, got {size}"
            raise ValueError(msg)
        if overlap < 0 or overlap >= size:
            msg = f"overlap must satisfy 0 <= overlap < size ({size}), got {overlap}"
            raise ValueError(msg)
        self.size = size
        self.overlap = overlap
        self.separators = tuple(separators) if separators else SEPARATORS
        self._finalise()

    allows_overlap = True

    def _effective_params(self) -> dict[str, int | float]:
        return {"size": self.size, "overlap": self.overlap}

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _split(self, text: str, base_offset: int, level: int) -> list[tuple[int, int]]:
        """Return `(start, end)` spans of `text`, recursively.

        Offset arithmetic is carried down the recursion so every returned span is
        an absolute position in the *document*, not in a substring. Getting this
        wrong is the classic recursive-chunker bug: chunks that look right and
        cite the wrong words (§9).
        """
        if self.tokenizer.count(text) <= self.size:
            return [(base_offset, base_offset + len(text))]

        if level >= len(self.separators) or self.separators[level] == "":
            return self._hard_split(text, base_offset)

        sep = self.separators[level]
        pieces = text.split(sep)
        if len(pieces) == 1:
            return self._split(text, base_offset, level + 1)

        spans: list[tuple[int, int]] = []
        # `cursor` stays RELATIVE to `base_offset`. Accumulating an absolute
        # `end` into it and then re-adding `base_offset` compounds the offset at
        # every piece, which produces spans that run past the document and land
        # in the wrong place entirely — while still looking plausible.
        cursor = 0
        for piece in pieces:
            rel_start = cursor
            rel_end = cursor + len(piece)
            start, end = base_offset + rel_start, base_offset + rel_end
            if start < end:
                if self.tokenizer.count(piece) <= self.size:
                    spans.append((start, end))
                else:
                    spans.extend(self._split(piece, start, level + 1))
            cursor = rel_end + len(sep)
        return spans

    def _hard_split(self, text: str, base_offset: int) -> list[tuple[int, int]]:
        """Last resort: cut on whitespace, never mid-word."""
        words = list(_WORD_RE.finditer(text))
        if not words:
            return [(base_offset, base_offset + len(text))]
        spans: list[tuple[int, int]] = []
        for i in range(0, len(words), self.size):
            group = words[i : i + self.size]
            spans.append((base_offset + group[0].start(), base_offset + group[-1].end()))
        return spans

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def _merge(self, text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Coalesce adjacent spans up to `size`.

        Measured in tokens with the real tokenizer, not characters: a
        512-token target and a 512-character target are different targets, and
        conflating them is the drift §13 warns about.
        """
        merged: list[tuple[int, int]] = []
        cur_start: int | None = None
        cur_end = 0
        for start, end in spans:
            if end <= start:
                continue
            if cur_start is None:
                cur_start, cur_end = start, end
                continue
            # +1 for the separator that was dropped between the two spans.
            candidate = cur_end + 1
            if self.tokenizer.count(text[candidate:end]) + self._kept_tokens(
                text, cur_start, cur_end
            ) <= self.size:
                cur_end = end
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        if cur_start is not None:
            merged.append((cur_start, cur_end))
        return merged

    def _kept_tokens(self, text: str, start: int, end: int) -> int:
        return self.tokenizer.count(text[start:end])

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def spans(self, doc: Document) -> list[Span]:
        text = doc.text
        if not text.strip():
            return []

        merged = self._merge(text, self._split(text, 0, 0))
        base = [
            Span(start=s, end=e, token_count=self.tokenizer.count(text[s:e]))
            for s, e in merged
            if text[s:e].strip()
        ]
        return self._apply_overlap(text, base)

    def _apply_overlap(self, text: str, spans: list[Span]) -> list[Span]:
        """Extend each span backwards by `overlap` tokens.

        Done on the token list so the overlap is exactly `overlap` tokens rather
        than a whitespace-search approximation, which drifts with punctuation
        and non-Latin scripts.
        """
        if self.overlap == 0 or not spans:
            return spans
        out: list[Span] = []
        for i, s in enumerate(spans):
            if i == 0:
                out.append(s)
                continue
            prev = spans[i - 1]
            gap = self.tokenizer.encode_spans(text[prev.start : s.start])
            new_start = (
                prev.start + gap[-self.overlap].start if len(gap) > self.overlap else prev.start
            )
            out.append(
                Span(
                    start=min(new_start, s.start),
                    end=s.end,
                    token_count=s.token_count + self.overlap,
                    heading_path=s.heading_path,
                    metadata=dict(s.metadata),
                )
            )
        return out
