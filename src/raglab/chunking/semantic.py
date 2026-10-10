"""Semantic chunking: break where embedding similarity dips.

§10: tracks topic shifts rather than formatting. A boundary lands where
consecutive sentences stop being about the same thing.

Strength: the only strategy here whose boundaries correspond to meaning. When a
document switches from fee schedules to refund policy, `fixed` may well place the
boundary inside the fee figure and `recursive` may place it at the paragraph
break after it, but only this one places it at the change of topic.

Cost, and it is real: an embedding pass per document, on top of the embedding
pass for the chunks themselves. §15a notes that if the chunking grid has to be
narrowed, `semantic` is the arm to drop precisely because of this.

It also needs the embedder, which is why `requires_embedder = True` and why it
cannot run in the `mock` profile's default test path without a stub. The
similarity threshold is a genuine tuning parameter with no principled default:
`0.5` is a starting point to be measured, not a constant.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from raglab.profiles import Embedder, MockEmbedder
from raglab.chunking.base import BaseChunker, Span, register
from raglab.types import Document

DEFAULT_SIZE = 512
DEFAULT_THRESHOLD = 0.5
"""Cosine similarity below which a boundary is proposed.

Deliberately a parameter. The right value depends on the corpus and on the
embedder, and a hardcoded default would quietly become an unrecorded config
change the first time somebody tuned it.
"""

DEFAULT_BUFFER = 1
"""Sentences grouped together before comparison.

Comparing sentence *i* to sentence *i-1* alone makes boundaries wherever the
wording shifts, including mid-argument style changes. A buffer of 2 or 3 is more
stable; 1 is the cheap default and is what the ablation starts from."""

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@register("semantic")
class SemanticChunker(BaseChunker):
    """Embedding-similarity boundary detection.

    Sentences are grouped into `buffer`-sized windows, embedded in one batch,
    and a boundary is proposed wherever consecutive window cosine similarity
    falls below `threshold`. Windows are then packed up to `size`, so the output
    count is still governed by `size` and the semantic signal only decides *where
    inside* that budget to cut.

    The one-pass structure matters for the quota: embedding a document is O(text),
    and doing it once here plus once for the resulting chunks is materially less
    than a strategy that re-embedded per candidate.
    """

    requires_embedder = True

    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        threshold: float = DEFAULT_THRESHOLD,
        buffer: int = DEFAULT_BUFFER,
        embedder: Embedder | None = None,
        tokenizer=None,  # noqa: ANN001
    ) -> None:
        super().__init__(tokenizer=tokenizer)
        if size <= 0:
            msg = f"size must be positive, got {size}"
            raise ValueError(msg)
        if not 0.0 <= threshold <= 1.0:
            msg = f"threshold must be in [0,1], got {threshold}"
            raise ValueError(msg)
        if buffer < 1:
            msg = f"buffer must be >= 1, got {buffer}"
            raise ValueError(msg)
        self.size = size
        self.threshold = threshold
        self.buffer = buffer
        # Defaults to the mock embedder so the strategy is exercisable without a
        # GPU. It makes NO semantic claim, and `docs/chunking.md` says so — a
        # mock-derived boundary is a plausible-looking number, which is exactly
        # what makes it dangerous if it leaks into a quality claim.
        self.embedder: Embedder = embedder or MockEmbedder()
        self._finalise()

    def _effective_params(self) -> dict[str, int]:
        # `threshold` and `buffer` shape boundaries, so they are part of the
        # §6.5 signature: changing them against a warm index must 409 rather
        # than append two boundary regimes into one matrix.
        return {"size": self.size, "threshold": self.threshold, "buffer": self.buffer}

    def _sentences(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in _SENT_RE.finditer(text) if m.group().strip()]

    def _windows(
        self, sentences: Sequence[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Group sentences into `buffer`-sized windows as (start, end) spans."""
        if not sentences:
            return []
        return [
            (sentences[i][0], sentences[min(i + self.buffer, len(sentences)) - 1][1])
            for i in range(0, len(sentences), self.buffer)
        ]

    def _similarities(self, text: str, windows: Sequence[tuple[int, int]]) -> list[float]:
        """Cosine similarity between consecutive windows.

        Batch encode of window texts only. The document-level pre-pass some
        semantic chunkers use would embed the whole document once more; since
        every window is embedded here anyway, that pass is redundant.
        """
        from raglab.profiles import cosine

        texts = [text[s:e] for s, e in windows]
        vecs = self.embedder.encode(texts)
        return [cosine(vecs[i - 1], vecs[i]) for i in range(1, len(vecs))]

    def spans(self, doc: Document) -> list[Span]:
        text = doc.text
        if not text.strip():
            return []
        sentences = self._sentences(text)
        if not sentences:
            return [Span(start=0, end=len(text), token_count=self.tokenizer.count(text))]
        windows = self._windows(sentences)
        sims = self._similarities(text, windows)

        # A boundary is proposed at window i when the window *after* it starts a
        # new topic. `sims[i-1]` is the similarity between window i-1 and i.
        cut_after: set[int] = set()
        for i, sim in enumerate(sims, start=1):
            if sim < self.threshold:
                cut_after.add(i - 1)

        out: list[Span] = []
        start = windows[0][0]
        for i, (ws, we) in enumerate(windows):
            if i in cut_after and ws > start:
                out.append(self._span(text, start, ws))
                start = ws
            # Pack: keep extending while under `size`.
            if self.tokenizer.count(text[start:we]) >= self.size:
                out.append(self._span(text, start, we))
                start = we
        if start < len(text):
            out.append(self._span(text, start, len(text)))

        return [s for s in out if text[s.start : s.end].strip()]

    def _span(self, text: str, start: int, end: int) -> Span:
        end = min(end, len(text))
        return Span(start=start, end=end, token_count=self.tokenizer.count(text[start:end]))
