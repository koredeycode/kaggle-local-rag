"""Core value types.

These are the shapes every stage of the pipeline agrees on. They are frozen
dataclasses with explicit `to_dict`/`from_dict` because they are serialised into
`chunks.jsonl`, `documents.jsonl`, and every API response, and because the
eval harness needs to compare gold spans against what retrieval actually
ranked.

The two offset pairs on `Chunk` are load-bearing and are documented in
`docs/ingestion.md`. Conflating them produces citations that point at the wrong
text and an undefined recall@k.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocState(StrEnum):
    """Per-document ingest state.

    §17.2 fixes this set exactly. The client renders the state name literally
    rather than a percentage, because per-document ingest progress is not
    honestly measurable and a fake percentage is worse than none.
    """

    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"


class SessionState(StrEnum):
    """Index build state for a session."""

    UNBUILT = "unbuilt"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PageSpan:
    """A page's worth of text within a parsed document.

    PDFs are the only format that produces these, but keeping the type generic
    lets the structural chunker reason about page boundaries for any format.
    """

    index: int
    """0-based page index as it appears in the source file."""

    start: int
    """Character offset of this page's text within the *original* document text."""

    end: int
    """Exclusive end offset, same space as `start`."""


@dataclass(frozen=True, slots=True)
class Document:
    """One source file after parsing and normalisation.

    Not the raw file. `text` is the *cleaned* text that chunking, retrieval, and
    the generator all operate on. `raw_len` records the original length so the
    clean/original offset map can be sanity-checked against it.
    """

    doc_id: str
    """Content-addressed: `sha256:<hex>` over the raw bytes. Stable across
    re-ingest, so an unchanged file is recognisably unchanged."""

    path: str
    """Path relative to the session's `files/` directory. Never an absolute
    path: a citation's `path` is shown to the operator and must not leak the
    VM's filesystem layout."""

    text: str
    """Normalised (cleaned) document text. This is what chunks are cut from."""

    pages: tuple[PageSpan, ...] = ()
    """Page boundaries in original space. Empty for formats without pages."""

    heading_path_at: tuple[tuple[int, str], ...] = ()
    """Heading breadcrumbs as `(offset, heading)` pairs, sorted by offset.
    Cheaper than a per-character lookup table and sufficient for slicing."""

    raw_len: int = 0
    """Length of the original text before cleaning. 0 when unknown."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Loader-specific extras. Never trusted for anything load-bearing."""

    def page_for_offset(self, orig_offset: int) -> int | None:
        """Return the 0-based page containing `orig_offset`, or None."""
        for page in self.pages:
            if page.start <= orig_offset < page.end:
                return page.index
        return None

    def heading_path_at_offset(self, clean_offset: int) -> list[str]:
        """Return the heading path in effect at `clean_offset`.

        Walks the sorted breadcrumb list backwards, so the innermost heading
        wins. Returns `[]` for flat prose, which is the correct answer rather
        than a placeholder.
        """
        path: list[str] = []
        for offset, heading in self.heading_path_at:
            if offset > clean_offset:
                break
            path.append(heading)
        return path

    def clean_text(self, start: int, end: int) -> str:
        """Slice the cleaned text. Verification always runs through here."""
        return self.text[start:end]


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable slice of a document.

    §6.4: `ordinal` is session-global, monotonically increasing, and never
    reused, and row `i` of `vectors.npy` equals line `i` of `chunks.jsonl`
    equals this ordinal. Deletion is logical — `live` goes False and the rows
    stay — so a deleted chunk's ordinal stays burned.
    """

    ordinal: int
    """Session-global, monotonically increasing, never reused (§6.4)."""

    doc_id: str
    chunk_index: int
    """0-based position within its own document."""

    text: str
    """The chunk body, as cut from the cleaned text."""

    clean_start: int
    """Offset into cleaned document text. What retrieval ranks and what span
    verification runs against (§9)."""

    clean_end: int
    """Exclusive. See `clean_start`."""

    orig_start: int | None = None
    """Offset into the file on disk. For navigation and display. `None` where
    the cleaner rewrote text it cannot map exactly — honestly absent rather
    than guessed."""

    orig_end: int | None = None
    """Exclusive. See `orig_start`."""

    page: int | None = None
    """0-based source page for PDFs; None otherwise."""

    heading_path: tuple[str, ...] = ()
    """Breadcrumb from the document structure, prepended to the embedded text."""

    token_count: int = 0
    """Chunk length in the chunker's tokenizer units."""

    live: bool = True
    """False once the owning document is deleted. Rows are never removed."""

    parent_ordinal: int | None = None
    """For small-to-big: the larger region this chunk belongs to."""

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Text that actually gets embedded.

        The heading breadcrumb is prepended, which costs nothing at inference
        time and reliably beats larger chunk sizes in practice: a chunk reading
        only "Term 1 is ₦45,000" is unretrievable, and one carrying its section
        path is not.
        """
        if not self.heading_path:
            return self.text
        return f"{' > '.join(self.heading_path)}\n\n{self.text}"

    def clean_text(self, doc: Document) -> str:
        """The exact string the model was shown, resolved via the document."""
        return doc.clean_text(self.clean_start, self.clean_end)

    @property
    def span_key(self) -> tuple[str, int, int]:
        """Identity used to match a chunk against a gold span (§15)."""
        return (self.doc_id, self.clean_start, self.clean_end)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["heading_path"] = list(self.heading_path)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Chunk:
        raw = dict(raw)
        raw["heading_path"] = tuple(raw.get("heading_path", ()))
        return cls(**raw)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved chunk with the scores that put it where it is."""

    chunk: Chunk
    score: float
    """Fused score. Meaning depends on the retriever that produced it."""

    rank: int
    """0-based position in the returned list."""

    dense_score: float | None = None
    """Cosine similarity, if a dense stage contributed."""

    lexical_score: float | None = None
    """BM25 score, if a lexical stage contributed. Not comparable to the
    dense score — that is precisely why fusion is rank-based (§2.2)."""

    rrf_score: float | None = None
    """Reciprocal rank fusion contribution, when RRF produced the order."""

    rerank_score: float | None = None
    """Raw cross-encoder logit, roughly -8..+5 for `bge-reranker-v2-m3`. Not a
    probability; apply a sigmoid before display or thresholding, and never
    compare across queries (§2.2)."""

    rerank_prob: float | None = None
    """Sigmoid of `rerank_score`, for display only."""

    windowed: bool = False
    """True when the chunk exceeded `rerank_max_len` and the cross-encoder saw
    a windowed, query-aware head instead of the whole chunk (§10)."""

    retrieved_by: frozenset[str] = frozenset()
    """Which stages returned this chunk: {"dense"}, {"bm25"}, {"dense","bm25"}."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "dense_score": self.dense_score,
            "lexical_score": self.lexical_score,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
            "rerank_prob": self.rerank_prob,
            "windowed": self.windowed,
            "retrieved_by": sorted(self.retrieved_by),
        }


@dataclass(frozen=True, slots=True)
class ScoredSpan:
    """A passage prepared for the generator, labelled `[S1]`..`[Sn]` (§13)."""

    marker: str
    text: str
    hit: Hit
    compression_ratio: float = 1.0
    """1.0 when no compression was applied."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "text": self.text,
            "score": self.hit.score,
            "rerank_prob": self.hit.rerank_prob,
            "compression_ratio": self.compression_ratio,
            "doc_id": self.hit.chunk.doc_id,
            "clean_start": self.hit.chunk.clean_start,
            "clean_end": self.hit.chunk.clean_end,
        }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Citation:
    """A resolved `[Sn]` marker (§14).

    Carries both offset pairs on purpose. "Which span supports this claim" is a
    clean-space question; "where is that in my file" is an original-space one.
    They are different questions and get different answers.
    """

    marker: str
    doc_id: str
    path: str
    heading_path: tuple[str, ...]
    page: int | None
    char_start: int | None
    """Original-file offset, for navigation and display."""

    char_end: int | None
    char_start_clean: int
    """Cleaned-text offset, for verification against what the model saw."""

    char_end_clean: int
    score: float
    rerank_prob: float | None = None
    ordinal: int | None = None
    """The chunk's session-global ordinal, so the evidence panel can re-fetch
    the exact chunk without re-running retrieval."""

    valid: bool = True
    """False for a hallucinated marker or an out-of-range span. A broken
    citation renders as a visibly broken chip rather than being dropped
    (§17.3) — a citation failure is a signal you need to see."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "doc_id": self.doc_id,
            "path": self.path,
            "heading_path": list(self.heading_path),
            "page": self.page,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "char_start_clean": self.char_start_clean,
            "char_end_clean": self.char_end_clean,
            "score": self.score,
            "rerank_prob": self.rerank_prob,
            "ordinal": self.ordinal,
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class Claim:
    """A sentence the model asserted, with whatever citation it attached."""

    text: str
    marker: str | None = None
    supported: bool | None = None
    """None until a judge or validator has ruled on it."""

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "marker": self.marker, "supported": self.supported}


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Per-stage latency in milliseconds.

    `gate_ms` is its own field rather than being folded into `retrieve_ms`:
    the gate is a full LLM prefill over the assembled candidates, plausibly
    the slowest pre-generation stage on a T4, and hiding it makes a
    multi-second pause look like a hang (§11, §17.3).
    """

    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    gate_ms: float = 0.0
    compress_ms: float = 0.0
    generate_ms: float = 0.0
    first_token_ms: float | None = None
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v or v == 0.0}


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated answer plus everything needed to check it (§14)."""

    answer: str
    citations: tuple[Citation, ...] = ()
    abstained: bool = False
    unverified_claims: tuple[Claim, ...] = ()
    """Sentences emitted with no citation. Visible by default in the UI, not
    behind a toggle (§17.3) — these are the dangerous ones, because they read
    as correct."""

    spans: tuple[ScoredSpan, ...] = ()
    stages: StageTimings = field(default_factory=StageTimings)
    question: str = ""
    session_id: str = ""

    def citation_for(self, marker: str) -> Citation | None:
        for c in self.citations:
            if c.marker == marker:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "abstained": self.abstained,
            "unverified_claims": [c.to_dict() for c in self.unverified_claims],
            "stages": self.stages.to_dict(),
            "question": self.question,
            "session_id": self.session_id,
        }


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def content_hash(data: bytes) -> str:
    """`sha256:<hex>` over raw bytes. The document identity (§9 manifest)."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    """`sha256:<hex>` over text, used as the embedding cache key (§20)."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def dumps_line(obj: Any) -> str:
    """One JSONL record. Compact and key-sorted so diffs stay readable."""
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str)


def loads_line(line: str) -> Any:
    return json.loads(line)


def iter_jsonl(path: Any) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file without holding it in memory.

    `chunks.jsonl` reaches hundreds of thousands of lines, and the invariant in
    §6.4 is checked by counting them — which is a streaming operation by
    necessity, not just by preference.
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Any) -> int:
    """Line count of a JSONL file, streamed. Used by the §6.4 invariant test."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n
