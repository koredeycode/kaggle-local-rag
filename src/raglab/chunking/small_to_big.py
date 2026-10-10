"""Small-to-big / parent-child retrieval.

§10: index small chunks for precise matching, then return the larger parent
region to the generator. Small chunks retrieve sharply; large chunks read
coherently.

This resolves the tension that otherwise forces a compromise. With one chunk size
you must choose: 256-token chunks match precisely but the generator gets
fragments it cannot reason across ("Term 1 is ₦45,000" then "…which is payable
before the resumption date"); 1024-token chunks read well but their embeddings
are diluted by content that does not match the query, so recall drops. Indexing
at 256 and generating from ~1000 gets both.

The parent is a contiguous region of the *same* document, and children record
their parent's start/end. Expansion happens at retrieval time, after ranking, so
it costs nothing when the gate abstains.

Not a `Chunker` in the §10 sense — it wraps one. The child strategy is what
determines the boundaries; this only adds the parent layer on top.
"""

from __future__ import annotations

from dataclasses import dataclass

from raglab.chunking.base import BaseChunker
from raglab.types import Chunk

DEFAULT_CHILD_SIZE = 200
DEFAULT_PARENT_SIZE = 1000
DEFAULT_PARENT_OVERLAP = 0
"""No overlap between parents.

Overlapping parents would return near-duplicate context to the generator, which
is the distraction problem §2.3 describes rather than a solution to it."""


@dataclass(slots=True)
class ParentRegion:
    """A contiguous window of one document, and the children inside it."""

    doc_id: str
    start: int
    end: int
    text: str
    child_ordinals: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.text.split())


@dataclass(slots=True)
class SmallToBigChunker:
    """Child chunks for the index, parent regions for the generator."""

    child: BaseChunker
    parent_size: int = DEFAULT_PARENT_SIZE
    parent_overlap: int = DEFAULT_PARENT_OVERLAP
    tokenizer: object = None

    def __post_init__(self) -> None:
        if self.parent_size <= 0:
            msg = f"parent_size must be positive, got {self.parent_size}"
            raise ValueError(msg)
        if self.parent_overlap < 0 or self.parent_overlap >= self.parent_size:
            msg = (
                f"parent_overlap must satisfy 0 <= overlap < parent_size "
                f"({self.parent_size}), got {self.parent_overlap}"
            )
            raise ValueError(msg)
        self.tokenizer = self.tokenizer or self.child.tokenizer

    @property
    def name(self) -> str:
        return f"small_to_big/{self.child.name}"

    def chunk(self, doc, *, start_ordinal: int = 0) -> list[Chunk]:
        """Produce child chunks, each tagged with its parent region.

        Returns children only. The index must hold children — that is the whole
        premise — and parents are recovered by `expand()` at query time.
        """
        children = self.child.chunk(doc, start_ordinal=start_ordinal)
        regions = self.parent_regions(doc)
        out: list[Chunk] = []
        for c in children:
            parent = self._find_parent(regions, c)
            out.append(
                Chunk(
                    ordinal=c.ordinal,
                    doc_id=c.doc_id,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    clean_start=c.clean_start,
                    clean_end=c.clean_end,
                    orig_start=c.orig_start,
                    orig_end=c.orig_end,
                    page=c.page,
                    heading_path=c.heading_path,
                    token_count=c.token_count,
                    live=c.live,
                    parent_ordinal=None,
                    metadata={
                        **c.metadata,
                        "parent_start": parent.start if parent else None,
                        "parent_end": parent.end if parent else None,
                    },
                )
            )
        return out

    def parent_regions(self, doc) -> list[ParentRegion]:
        """Cover the document with parent windows on the child's boundaries.

        Built from the children's spans rather than by sliding a window over raw
        text, so every parent starts and ends at a real chunk boundary and no
        parent is ever empty.
        """
        children = self.child.chunk(doc, start_ordinal=0)
        if not children:
            return []

        regions: list[ParentRegion] = []
        n = len(doc.text)
        stride = max(1, self.parent_size - self.parent_overlap)

        i = 0
        while i < len(children):
            start = children[i].clean_start
            end = start
            group: list[Chunk] = []
            while i < len(children) and (end - start) < self.parent_size:
                group.append(children[i])
                end = children[i].clean_end
                i += 1
            # Extend forward to at least the stride so consecutive parents do
            # not leave a gap a query's true answer could fall into.
            target = min(n, start + stride)
            if target > end:
                end = target
            regions.append(
                ParentRegion(
                    doc_id=doc.doc_id,
                    start=start,
                    end=end,
                    text=doc.text[start:end],
                    child_ordinals=tuple(c.ordinal for c in group),
                )
            )
        return regions

    @staticmethod
    def _find_parent(
        regions: list[ParentRegion], child: Chunk
    ) -> ParentRegion | None:
        for r in regions:
            if r.start <= child.clean_start < r.end:
                return r
        return None

    @staticmethod
    def expand(
        child: Chunk,
        doc_text: str,
        *,
        max_chars: int = DEFAULT_PARENT_SIZE * 4,
    ) -> str:
        """Return the parent region text for a child chunk.

        Falls back to the child's own text when the chunk carries no parent
        span — which happens for chunks produced by a plain `Chunker` and after a
        manifest built without `small_to_big`. Returning the child text is the
        correct degradation: the answer loses context but stays grounded.
        """
        start = child.metadata.get("parent_start")
        end = child.metadata.get("parent_end")
        if not isinstance(start, int) or not isinstance(end, int):
            return child.text
        # `max_chars` guards against a stale parent span outliving a re-chunk:
        # expanding to a span past the end of the text would raise, and a
        # citation panel must never be the thing that crashes a query.
        start = max(0, min(start, len(doc_text)))
        end = max(start, min(end, start + max_chars, len(doc_text)))
        return doc_text[start:end]
