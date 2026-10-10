"""Embedding, reranking, and generation interfaces plus profile resolution.

§8 rule 2 drives the shape of this module: a failure must degrade to a
*visible, recoverable* error, never a silent model swap. Silent substitution is
how you end up with eval numbers that do not mean anything — the run looks fine
and the `mock` hash embedder quietly replaced `bge-m3`.

So `resolve_profile` raises `ModelUnavailable` with the profile and model named,
and never falls back.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from zlib import crc32

from raglab.config import Profile, Settings
from raglab.errors import ModelUnavailable

MOCK_DIM = 64
"""Mock vectors are 64-dim, not 1024. Small enough that a 100k-chunk test
index is a few MB, large enough that dot products are not degenerate."""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Bi-encoder. Embeds query and document *separately* (§2.1)."""

    @property
    def model_id(self) -> str:
        """Recorded in every manifest. Swapping it invalidates stored vectors."""

    @property
    def dim(self) -> int:
        ...

    @property
    def max_length(self) -> int:
        ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch. Returns one vector per input, in input order.

        Order preservation is part of the contract: `vectors.npy` row *i* must
        equal `chunks.jsonl` line *i*, and that chain breaks silently if a
        batch comes back reordered.
        """


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder. Scores query and candidate *together* (§2.2)."""

    @property
    def model_id(self) -> str:
        ...

    @property
    def max_length(self) -> int:
        """Hard ceiling. Chunks longer than this are windowed by the caller
        (§10) — never silently truncated inside the model."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Raw logits. Unbounded and roughly -8..+5; not probabilities."""


@runtime_checkable
class Generator(Protocol):
    """Text generation, optionally streaming."""

    @property
    def model_id(self) -> str:
        ...

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        ...

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        ...


# ---------------------------------------------------------------------------
# Mock implementations — the `mock` profile (§8)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Deterministic across runs and platforms.

    `re` on unicode word characters rather than `\\b\\w+` so accented and
    non-Latin corpus text tokenises rather than collapsing to nothing. §2.1
    notes `bge-m3` covers 100+ languages, and a mock that only saw ASCII would
    make every non-English test look like a retrieval failure.
    """
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


@dataclass(slots=True)
class MockEmbedder:
    """Deterministic hashed bag-of-words embedder.

    Not semantic — it cannot be. What it *is* is deterministic, dependency-free,
    and preserves exact-token overlap, which is enough to make the BM25/dense
    complementarity test meaningful: a string with a shared rare token scores
    higher under the mock embedder than under real `bge-m3` would.

    Unit-test determinism is the point. It must never be mistaken for a quality
    signal, so `model_id` says so loudly.
    """

    dim: int = MOCK_DIM
    model_id: str = "mock/hash-v1"
    max_length: int = 8192

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _tokenize(text)[: self.max_length]
        if not toks:
            return vec
        for tok in toks:
            # NOT `hash()`: Python salts string hashes per process, so a mock
            # embedder built on it would produce different vectors on every run
            # and make every retrieval test non-reproducible.
            h = crc32(tok.encode("utf-8"))
            idx = h % self.dim
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed(self, text: str) -> list[float]:
        return self._vec(text)


@dataclass(slots=True)
class MockReranker:
    """Lexical-overlap scorer standing in for a cross-encoder.

    Shapes like a cross-encoder on one axis that matters: it sees the query and
    the passage *together*, so it can prefer a passage that answers this query
    over one that merely shares a topic. That joint-read property is what Phase
    4 is about, and a stub that scored passages independently would make the
    rerank ablation untestable.
    """

    model_id: str = "mock/lexical-overlap-v1"
    max_length: int = 512

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(_tokenize(query))
        if not q:
            return [0.0] * len(passages)
        out: list[float] = []
        for p in passages:
            ptoks = _tokenize(p)[: self.max_length]
            if not ptoks:
                out.append(-8.0)
                continue
            pset = set(ptoks)
            overlap = len(q & pset)
            # Length-normalised so a long passage is not rewarded for volume.
            out.append(-8.0 + 13.0 * overlap / max(1.0, math.sqrt(len(ptoks))))
        return out


@dataclass(slots=True)
class MockGenerator:
    """Scripted generator for the `mock` profile.

    Emits a citation marker so the citation-resolution path is exercised
    end-to-end in tests. It is not a model and makes no quality claim.
    """

    model_id: str = "mock/scripted-v1"
    answer: str = "Mock answer. [S1]"
    reason: str = "mock profile: no generator is configured"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self.answer

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        for word in self.answer.split(" "):
            yield word + " "


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedModels:
    """What a profile actually resolved to. Reported by `GET /api/profile`."""

    profile: Profile
    embedder: Embedder
    reranker: Reranker
    generator: Generator | None
    embed_dim: int
    """Dimension the *index* is pinned to. For `mock` this is MOCK_DIM, not
    `settings.embed_dim`, and the manifest records it so a later switch to
    `local` produces a clean 409 rather than a dimension mismatch at search
    time (§6.5)."""

    notes: tuple[str, ...] = ()


def resolve_profile(settings: Settings) -> ResolvedModels:
    """Resolve a profile to concrete model objects.

    Raises `ModelUnavailable` rather than degrading. See module docstring.
    """
    match settings.profile:
        case Profile.MOCK:
            embedder = MockEmbedder()
            return ResolvedModels(
                profile=Profile.MOCK,
                embedder=embedder,
                reranker=MockReranker(),
                generator=MockGenerator(),
                embed_dim=embedder.dim,
                notes=(
                    "deterministic hashed embedder; no semantic quality claim",
                    "unit tests and CI only",
                ),
            )

        case Profile.TINY:
            embedder, reranker = _load_tiny(settings)
            return ResolvedModels(
                profile=Profile.TINY,
                embedder=embedder,
                reranker=reranker,
                # §8: `tiny` has no generator. Querying returns a visible
                # abstention naming the reason rather than a fake answer.
                generator=None,
                embed_dim=settings.embed_dim if settings.embed_dim != 1024 else 384,
                notes=(
                    "all-MiniLM-L6-v2, 384-dim",
                    "no generator: retrieval-only local testing",
                    "384 dims differ from bge-m3's 1024, so an index built here "
                    "will 409 against a `local` build (§6.5)",
                ),
            )

        case Profile.LOCAL:
            embedder, reranker = _load_local(settings)
            return ResolvedModels(
                profile=Profile.LOCAL,
                embedder=embedder,
                reranker=reranker,
                generator=None,  # supplied lazily; Ollama is a network client
                embed_dim=settings.embed_dim,
                notes=(
                    "bge-m3 needs no query instruction prefix (model card, §5.2)",
                    "generator is qwen3:8b via Ollama native /api/chat",
                ),
            )

    raise ModelUnavailable(
        f"unknown profile {settings.profile!r}",
        profile=str(settings.profile),
        model="unknown",
    )


def _require(module: str, profile: str, model: str) -> object:
    """Import a module or raise a *named* `ModelUnavailable`.

    The message names the missing package and the profile that wanted it, so the
    operator sees "install the local extra" rather than a stack trace.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ModelUnavailable(
            f"profile {profile!r} needs {module!r} (pip install -e '.[{profile}]'): {exc}",
            profile=profile,
            model=model,
        ) from exc


def _load_tiny(settings: Settings) -> tuple[Embedder, Reranker]:
    _require("sentence_transformers", "tiny", settings.embed_model)
    _require("sentence_transformers", "tiny", settings.rerank_model)
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = _resolve_device(settings.embed_device, "tiny")
    return (
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device),
        CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device),
    )  # type: ignore[return-value]


def _load_local(settings: Settings) -> tuple[Embedder, Reranker]:
    _require("torch", "local", settings.embed_model)
    _require("sentence_transformers", "local", settings.embed_model)
    from sentence_transformers import CrossEncoder, SentenceTransformer

    # §5.5: precision is an open measurement, not an assumption. fp16 on GPU,
    # fp32 on CPU where there is no VRAM to save and fp16 buys nothing. The
    # Phase 0 agreement measurement is what decides the GPU case.
    device = _resolve_device(settings.embed_device, "local")
    embedder = SentenceTransformer(
        settings.embed_model,
        device=device,
        model_kwargs={"torch_dtype": "float16"} if device.startswith("cuda") else None,
    )
    reranker = CrossEncoder(settings.rerank_model, device=device)
    return embedder, reranker  # type: ignore[return-value]


def _resolve_device(requested: str, profile: str) -> str:
    """Map a requested device to one that actually exists, or fail loudly.

    §4 makes the GPU split a hard boundary via `CUDA_VISIBLE_DEVICES`. Silently
    landing on CPU when CUDA is absent would make every latency number a lie, so
    a `cuda:*` request that cannot be honoured raises.
    """
    if not requested.startswith("cuda"):
        return requested
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded by _require
        raise ModelUnavailable(
            f"profile {profile!r} requested {requested!r} but torch is absent",
            profile=profile,
            model="torch",
        ) from exc
    if not torch.cuda.is_available():
        raise ModelUnavailable(
            f"profile {profile!r} requested {requested!r} but no CUDA device is "
            "visible; set RAGLAB_EMBED_DEVICE=cpu to run without a GPU",
            profile=profile,
            model="cuda",
        )
    index = int(requested.split(":")[1])
    if index >= torch.cuda.device_count():
        raise ModelUnavailable(
            f"requested {requested!r} but only {torch.cuda.device_count()} CUDA "
            "device(s) visible; §4 expects T4 x2 with the embedder on cuda:1",
            profile=profile,
            model="cuda",
        )
    return requested


# ---------------------------------------------------------------------------
# Scoring helpers shared by every profile
# ---------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    """Logit to probability.

    §2.2: a reranker score is an unbounded logit, roughly -8..+5 for
    `bge-reranker-v2-m3`. Display and threshold on the sigmoid; never compare
    raw scores across queries.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Zero-norm inputs return 0.0 rather than raising."""
    if len(a) != len(b):
        msg = f"dimension mismatch: {len(a)} vs {len(b)}"
        raise ValueError(msg)
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
