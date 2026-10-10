"""The `Chunker` protocol and registry.

§10: four strategies behind one interface so ablations can swap them freely.
That single interface is what makes the chunking grid possible — without it, each
strategy would need its own indexing path and the comparison would not be
controlled.

Ordinal assignment lives here rather than in each strategy, because it is part
of the §6.4 invariant (`row i == line i == chunk.ordinal`) and must be produced
by exactly one code path. A strategy returns spans; the base class numbers them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from raglab.tokens import Tokenizer, get_tokenizer
from raglab.types import Chunk, Document

CHUNKER_VERSION = "1.0.0"
"""Bumped whenever chunk output changes for identical input.

Part of `Settings.index_signature` (§6.5). A bump forces a 409 on re-index
rather than appending two chunking regimes into one matrix."""


@runtime_checkable
class Chunker(Protocol):
    """Cuts a document into retrievable spans."""

    name: str

    def chunk(self, doc: Document) -> list[Chunk]:
        ...


@dataclass(frozen=True, slots=True)
class ChunkerSpec:
    """A chunker plus its parameters. Hashable, so it can key a config hash.

    Built from the chunker's *effective* configuration rather than whatever was
    passed to the constructor. `get_chunker("fixed", size=256, overlap=32)`
    leaves `overlap` at its default 64, and the spec has to record 64 — a
    signature reflecting only the explicitly-passed params would make two
    differently-configured chunkers hash identically, and §6.5 would then let
    them append into one matrix.
    """

    name: str
    params: tuple[tuple[str, int | float], ...] = ()

    def signature(self) -> str:
        if not self.params:
            return f"{self.name}/{CHUNKER_VERSION}"
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params))
        return f"{self.name}[{inner}]/{CHUNKER_VERSION}"

    @classmethod
    def of(cls, name: str, **params: int | float) -> ChunkerSpec:
        return cls(name, tuple(sorted(params.items())))

    def get(self, key: str) -> int | float | None:
        for k, v in self.params:
            if k == key:
                return v
        return None

    @property
    def size(self) -> int | None:
        v = self.get("size")
        return int(v) if v is not None else None

    @property
    def overlap(self) -> int | None:
        v = self.get("overlap")
        return int(v) if v is not None else None


@dataclass(slots=True)
class Span:
    """A half-open character span of cleaned document text.

    Offsets, not text. A strategy decides *where* to cut; the base class resolves
    offsets into text and builds `Chunk` records, so no strategy can forget to
    record `clean_start`/`clean_end` and silently break the citation contract
    (§9).
    """

    start: int
    end: int
    token_count: int = 0
    heading_path: tuple[str, ...] = ()
    parent_index: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class BaseChunker:
    """Shared machinery: validation, numbering, and `Chunk` construction."""

    name: str = "base"
    version: str = CHUNKER_VERSION
    requires_embedder: bool = False
    """True for `semantic`, which needs an embedding pass per document (§10)."""

    allows_overlap: bool = False
    """Whether consecutive spans may share characters.

    `fixed` and `recursive` do — overlap is the point (§2.1: tokens repeated
    between adjacent chunks so a fact spanning a boundary survives in one
    piece). `structural` and `semantic` do not, and for `structural` overlap
    across a heading boundary would undo the one thing the strategy is for.

    `validate` enforces this, so it is a per-strategy declaration rather than a
    convention each strategy has to remember.
    """

    def __init__(self, tokenizer: Tokenizer | None = None, **params: int | float) -> None:
        self.tokenizer: Tokenizer = tokenizer or get_tokenizer("regex")
        self.params: dict[str, int | float] = dict(params)
        self.spec = ChunkerSpec.of(self.name)

    def _finalise(self) -> None:
        """Record effective params into the spec. Called at the end of each
        subclass `__init__`, after defaults are resolved."""
        self.spec = ChunkerSpec.of(self.name, **self._effective_params())

    def _effective_params(self) -> dict[str, int | float]:
        """Params that shape chunk output, for the §6.5 signature.

        Subclasses override. Only parameters that change the produced spans
        belong here: adding a knob that cannot move a boundary would make an
        index rebuild for no reason.
        """
        return dict(self.params)

    # -- to implement --------------------------------------------------

    def spans(self, doc: Document) -> list[Span]:
        """Where to cut. Offsets into `doc.text` (clean space, §9)."""
        raise NotImplementedError

    # -- shared --------------------------------------------------------

    def validate(self, doc: Document, spans: Sequence[Span]) -> None:
        """Reject spans that would corrupt the offset contract.

        Checked here rather than trusted from each strategy because the failure
        is silent: an out-of-range `clean_start` yields a chunk whose text does
        not match its offsets, and a citation pointing at that span points at
        the wrong words with no error anywhere.

        Ordering is always required. Overlap is required to be *absent* unless
        the strategy declares `allows_overlap`.
        """
        n = len(doc.text)
        prev_end = 0
        prev_start = 0
        for i, s in enumerate(spans):
            if not (0 <= s.start < s.end <= n):
                msg = (
                    f"{self.name}: span {i} [{s.start},{s.end}) is outside "
                    f"document text of length {n}"
                )
                raise ValueError(msg)
            if s.start < prev_start:
                msg = (
                    f"{self.name}: span {i} starts at {s.start}, before the "
                    f"previous span started at {prev_start}; spans must be ordered"
                )
                raise ValueError(msg)
            if not self.allows_overlap and s.start < prev_end:
                msg = (
                    f"{self.name}: span {i} starts at {s.start}, overlapping the "
                    f"previous span which ended at {prev_end}; {self.name} does "
                    "not allow overlap"
                )
                raise ValueError(msg)
            prev_start, prev_end = s.start, s.end

    def chunk(self, doc: Document, *, start_ordinal: int = 0) -> list[Chunk]:
        """Number spans into `Chunk` records.

        `start_ordinal` is where this document's first chunk lands in the
        session-global sequence. §6.4 makes that sequence monotonic and never
        reused, so the caller — not the strategy — owns it.
        """
        spans = list(self.spans(doc))
        self.validate(doc, spans)
        chunks: list[Chunk] = []
        for i, s in enumerate(spans):
            text = doc.text[s.start : s.end]
            chunks.append(
                Chunk(
                    ordinal=start_ordinal + i,
                    doc_id=doc.doc_id,
                    chunk_index=i,
                    text=text,
                    clean_start=s.start,
                    clean_end=s.end,
                    page=doc.page_for_offset(s.start),
                    heading_path=s.heading_path or tuple(doc.heading_path_at_offset(s.start)),
                    token_count=s.token_count or self.tokenizer.count(text),
                    metadata=dict(s.metadata),
                )
            )
        return chunks

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.spec.signature()})"


_REGISTRY: dict[str, Callable[..., BaseChunker]] = {}


def register(name: str) -> Callable[[type[BaseChunker]], type[BaseChunker]]:
    """Class decorator adding a chunker to the registry by name."""

    def deco(cls: type[BaseChunker]) -> type[BaseChunker]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_chunker(name: str, **params: int) -> BaseChunker:
    """Resolve a chunker by name.

    Unknown names raise. The registry exists so `Settings.chunker` can be a
    string in config and in the manifest, but a typo must not fall through to a
    default — a silent fallback would mean an ablation row labelled `structural`
    that actually ran `fixed`.
    """
    if name not in _REGISTRY:
        msg = f"unknown chunker {name!r}; available: {available()}"
        raise ValueError(msg)
    return _REGISTRY[name](**params)


def _load_builtins() -> None:
    """Import the built-in chunkers so they register themselves."""
    from raglab.chunking import fixed, recursive, semantic, small_to_big, structural  # noqa: F401


_load_builtins()
