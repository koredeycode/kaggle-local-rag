"""Tokening.

§13 is explicit that context packing must use the generator's own tokenizer and
never a character-count guess, because character heuristics drift silently the
moment the text is not English prose. That rule is why this module has a real
tokenizer implementation rather than a `len(text) // 4` shortcut.

Three implementations:

- `RegexTokenizer` — stdlib only, deterministic, no vocabulary. Default for
  `mock`, and the one used by chunking tests. Its unit is *a word*, which is
  fine for sizing chunks in a test and is **not** a substitute for a real
  BPE tokenizer when packing against a 16GB T4.
- `HFTokenizer` — `AutoTokenizer` for the Qwen family, the production path.
- `IdentityTokenizer` — one token per character, for tests that need an exact
  known count.

The tokenizer identity is recorded alongside every eval run, for the same
reason model revisions are: a chunk-size comparison across two tokenizers is
not a comparison.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer contract. Counts and truncation, nothing more."""

    @property
    def name(self) -> str:
        ...

    @property
    def model_id(self) -> str:
        """Recorded in eval runs and the index manifest."""

    def count(self, text: str) -> int:
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, ids: Sequence[int]) -> str:
        ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Return the leading `max_tokens` worth of text. Must not split a
        multi-byte character or leave a dangling half-token."""


@dataclass(frozen=True, slots=True)
class Token:
    """One token plus its character span, so truncation preserves offsets."""

    id: int
    start: int
    end: int


class BaseTokenizer:
    """Shared truncation logic over a token-span list."""

    name: str = "base"
    model_id: str = "base"

    def _spans(self, text: str) -> list[Token]:  # pragma: no cover - abstract
        raise NotImplementedError

    def encode_spans(self, text: str) -> list[Token]:
        return self._spans(text)

    def count(self, text: str) -> int:
        return len(self._spans(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        spans = self._spans(text)
        if len(spans) <= max_tokens:
            return text
        # Cut at the end of the last retained token so no word is left half-in.
        return text[: spans[max_tokens - 1].end]


@dataclass(slots=True)
class RegexTokenizer(BaseTokenizer):
    """Word-and-punctuation tokenizer. Stdlib only, no vocabulary file.

    Counts differ from a BPE tokenizer, which is fine for chunk sizing in a
    test and wrong for packing a real prompt. `docs/chunking.md` says so, and
    the eval harness records `tokenizer.model_id` so a chunk-size row from this
    tokenizer is never read as comparable to one from `HFTokenizer`.
    """

    name: str = "regex"
    model_id: str = "raglab/regex-v1"

    def _spans(self, text: str) -> list[Token]:
        return [
            Token(id=i, start=m.start(), end=m.end())
            for i, m in enumerate(_WORD_RE.finditer(text))
        ]

    def encode(self, text: str) -> list[int]:
        return [t.id for t in self._spans(text)]

    def decode(self, ids: Sequence[int]) -> str:
        msg = "RegexTokenizer is not invertible; use count() or truncate()"
        raise NotImplementedError(msg)


@dataclass(slots=True)
class IdentityTokenizer(BaseTokenizer):
    """One token per character. For tests that need an exact known count."""

    name: str = "char"
    model_id: str = "raglab/char-v1"

    def _spans(self, text: str) -> list[Token]:
        return [Token(id=ord(c), start=i, end=i + 1) for i, c in enumerate(text)]

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids)


@dataclass(slots=True)
class HFTokenizer(BaseTokenizer):
    """`AutoTokenizer` wrapper. The production path for §13 packing.

    Lazy-imports `transformers` so the `mock` profile stays dependency-free
    (§8 rule 1: a run must never require a model download to be testable).
    """

    model_id: str = "Qwen/Qwen3-8B"
    name: str = "hf"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        self._tok: Any | None = None

    def _ensure(self) -> Any:
        if self._tok is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - guarded by profile
                msg = (
                    "HFTokenizer needs transformers: "
                    "pip install -e '.[local]'"
                )
                raise RuntimeError(msg) from exc
            self._tok = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
        return self._tok

    def _spans(self, text: str) -> list[Token]:
        tok = self._ensure()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        # Some tokenizers emit (0,0) for special tokens added implicitly; those
        # carry no character span, so a span-less token would make `truncate`
        # cut at offset 0 and return an empty string.
        return [
            Token(id=int(i), start=int(s), end=int(e))
            for i, (s, e) in zip(enc["input_ids"], enc["offset_mapping"], strict=True)
            if e > s
        ]

    def encode(self, text: str) -> list[int]:
        return list(self._ensure()(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: Sequence[int]) -> str:
        return str(self._ensure().decode(list(ids), skip_special_tokens=True))


_REGISTRY: dict[str, Tokenizer] = {}


def get_tokenizer(name: str = "regex", model_id: str | None = None) -> Tokenizer:
    """Resolve a tokenizer by name. Cached.

    Unknown names raise rather than defaulting: silently falling back to the
    regex tokenizer in production would mean packing a prompt with a wrong
    unit count and overflowing `gen_ctx` with no error.
    """
    if name in ("hf", "transformers"):
        key = f"hf:{model_id or 'Qwen/Qwen3-8B'}"
        if key not in _REGISTRY:
            _REGISTRY[key] = HFTokenizer(model_id=model_id or "Qwen/Qwen3-8B")
        return _REGISTRY[key]
    if name == "regex":
        _REGISTRY.setdefault("regex", RegexTokenizer())
        return _REGISTRY["regex"]
    if name in ("char", "identity"):
        _REGISTRY.setdefault("char", IdentityTokenizer())
        return _REGISTRY["char"]
    msg = f"unknown tokenizer {name!r}; expected one of: regex, char, hf"
    raise ValueError(msg)


def tokenizer_for_profile(profile: str, gen_model: str = "qwen3:8b") -> Tokenizer:
    """Profile-appropriate tokenizer.

    `mock` gets `regex`: no download, deterministic. `local` gets the real Qwen
    tokenizer, because §13's packing rule only means something against the
    model that will actually read the prompt.
    """
    if profile == "local":
        # Ollama tags are not HF ids; map the two this project ships with.
        hf_id = {"qwen3:8b": "Qwen/Qwen3-8B", "qwen3:14b": "Qwen/Qwen3-14B"}.get(
            gen_model, gen_model
        )
        try:
            return get_tokenizer("hf", hf_id)
        except Exception:  # noqa: BLE001 - fall back loudly, not silently
            return get_tokenizer("regex")
    return get_tokenizer("regex")
