"""Structural chunking: never cross a heading boundary.

§10: the best strategy for Markdown, DOCX, and HTML, and useless for flat prose.

The implementation follows the document's own heading tree rather than a
heuristic, because the parse already recovered the structure and re-deriving it
from text is strictly worse. Each heading section becomes one chunk (or several,
if it exceeds `size`), and every chunk carries its full breadcrumb — which
§9 already prepends to the embedded text, so a chunk saying only "Term 1 is
₦45,000" becomes "Lagos State > Education > Fee Structure 2026 / Term 1 is
₦45,000" and becomes retrievable.

The weakness is not a subtlety, it is a hard limit: for a document with no
headings at all there is exactly one section, so this degenerates to one giant
chunk unless the size cap is applied. That is handled below rather than left as
a surprise, and it is why the ablation keeps `fixed` as the baseline.
"""

from __future__ import annotations

from raglab.chunking.base import BaseChunker, Span, register
from raglab.types import Document

DEFAULT_SIZE = 1024
DEFAULT_MIN_SIZE = 64
"""Sections below this are merged into the following one.

Without it, every page footer promoted to a heading becomes its own 10-token
chunk, and §9's near-duplicate filter then has to clean up after the chunker
instead of the chunker avoiding the problem."""

DEFAULT_OVERLAP = 0
"""No overlap by default.

Overlap across a heading boundary re-introduces text from the previous section,
which is precisely the boundary this strategy exists to respect."""


@register("structural")
class StructuralChunker(BaseChunker):
    """Heading-bounded sections, size-capped.

    Heading boundaries come from `Document.heading_path_at`, which the loaders
    populate (`ingest/structure.py`). A document with no headings yields one
    implicit root section, which then gets size-capped like any other — so this
    degrades to `fixed` rather than to a single unusable chunk.
    """

    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        min_size: int = DEFAULT_MIN_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        tokenizer=None,  # noqa: ANN001
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        if size <= 0:
            msg = f"size must be positive, got {size}"
            raise ValueError(msg)
        if overlap != 0:
            # Raising rather than clamping. Overlap is not merely unhelpful here,
            # it is contradictory: it re-introduces text from the previous
            # section, which is the boundary this strategy exists to respect, and
            # `allows_overlap = False` means it would fail validation anyway.
            # A silent clamp would leave `Settings.chunk_overlap=64` looking
            # applied to a `structural` build while changing nothing.
            msg = (
                f"structural chunking does not support overlap (got {overlap}); "
                "overlap would reintroduce text across the heading boundary this "
                "strategy preserves (§10)"
            )
            raise ValueError(msg)
        self.size = size
        self.min_size = min_size
        self.overlap = overlap
        self._finalise()

    def _effective_params(self) -> dict[str, int | float]:
        return {"size": self.size, "overlap": self.overlap, "min_size": self.min_size}

    def spans(self, doc: Document) -> list[Span]:
        text = doc.text
        if not text.strip():
            return []

        boundaries = self._section_boundaries(doc)
        raw: list[Span] = []
        for start, end, path in boundaries:
            section = text[start:end]
            if not section.strip():
                continue
            tokens = self.tokenizer.count(section)
            if tokens <= self.size:
                raw.append(
                    Span(start=start, end=end, token_count=tokens, heading_path=path)
                )
                continue
            # Oversized section: split inside it, but every piece keeps the
            # breadcrumb, which is the entire point of the strategy.
            for s_start, s_end in self._cap(section, start):
                raw.append(
                    Span(
                        start=s_start,
                        end=s_end,
                        token_count=self.tokenizer.count(text[s_start:s_end]),
                        heading_path=path,
                        metadata={"split_section": True},
                    )
                )

        raw = self._merge_small(raw, text)
        return raw

    def _section_boundaries(
        self, doc: Document
    ) -> list[tuple[int, int, tuple[str, ...]]]:
        """Split the document into (start, end, heading_path) sections.

        Starts at each heading offset; the section runs to the next heading of
        any level. Content *before* the first heading gets an empty path, which
        is the honest answer rather than a fabricated root.
        """
        text = doc.text
        # Drop empty headings and collapse duplicate offsets. A trailing
        # newline parsed as an empty heading at offset 0 would otherwise emit a
        # second section starting at 0 *after* the first, duplicating the
        # document start and producing out-of-order spans.
        by_offset: dict[int, str] = {}
        for off, h in doc.heading_path_at:
            if not h.strip() or not 0 <= off < len(text):
                continue
            by_offset.setdefault(off, h)
        marks = sorted(by_offset.items())
        if not marks:
            return [(0, len(text), ())]

        sections: list[tuple[int, int, tuple[str, ...]]] = []
        if marks[0][0] > 0:
            sections.append((0, marks[0][0], ()))
        for i, (off, _heading) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            if end <= off:
                continue
            sections.append((off, end, tuple(doc.heading_path_at_offset(off))))
        return sections

    def _cap(self, section: str, base: int) -> list[tuple[int, int]]:
        """Token windows within one oversized section."""
        toks = self.tokenizer.encode_spans(section)
        if not toks:
            return [(base, base + len(section))]
        stride = max(1, self.size - self.overlap)
        out: list[tuple[int, int]] = []
        i = 0
        while i < len(toks):
            window = toks[i : i + self.size]
            out.append((base + window[0].start, base + window[-1].end))
            if i + self.size >= len(toks):
                break
            i += stride
        return out

    def _merge_small(self, spans: list[Span], text: str) -> list[Span]:
        """Fold runs of tiny sections into the section that follows them.

        Lookahead, not backward: a 10-token section is prepended to its
        successor and the merged chunk takes the *successor's* heading path.
        Merging backward instead would leave the chunk labelled with the earlier
        (shallower) heading while its body is mostly the later section's content,
        so the breadcrumb — which §9 prepends to the embedded text — would
        describe the wrong thing.

        Merging is allowed to cross a heading boundary here on purpose: a
        heading whose entire section is two tokens long is not useful context on
        its own. It is bounded by `min_size`, and `size` still caps the result,
        so the strategy's defining property (chunks do not *arbitrarily* span
        sections) survives.
        """
        if self.min_size <= 0 or len(spans) < 2:
            return spans
        out: list[Span] = []
        pending: list[Span] = []
        for i, s in enumerate(spans):
            pending.append(s)
            # `head` is the span whose heading path the merged chunk adopts:
            # the successor when this one is too small, otherwise itself.
            head = spans[i + 1] if (i + 1 < len(spans) and s.token_count < self.min_size) else s
            if s.token_count >= self.min_size or i + 1 == len(spans):
                self._flush(out, pending, text, head=head)
                pending = []
        return out

    def _flush(
        self,
        out: list[Span],
        pending: list[Span],
        text: str,
        *,
        head: Span,
    ) -> None:
        """Emit `pending` as one span led by `head`'s heading path."""
        if not pending:
            return
        start = pending[0].start
        end = pending[-1].end
        out.append(
            Span(
                start=start,
                end=end,
                token_count=self.tokenizer.count(text[start:end]),
                heading_path=head.heading_path,
                metadata={"merged_sections": len(pending)} if len(pending) > 1 else {},
            )
        )
        pending.clear()
