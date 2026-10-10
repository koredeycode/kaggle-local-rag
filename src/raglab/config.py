"""Frozen settings, profile resolution, and environment loading.

Design notes that are easy to lose:

- `Settings` is frozen. The retrieval config hash (§15 determinism) is computed
  from it, so mutability would let a run record a config it did not use.
- Split into `global_` and per-session `SessionConfig` fields. §6.5: settings
  that shape *stored data* are pinned to the index; settings that shape
  *queries* are free. Keeping them in one dataclass makes the 409 check
  (`ComponentMismatch`) a diff between two explicit field sets.
- `store_root` defaults to a repo-local path in dev and `/kaggle/working/ragstore`
  when `/kaggle` exists, because §6.2 makes that the only durable tier on Kaggle.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from raglab.errors import RagError

ENV_PREFIX = "RAGLAB_"


class Profile(StrEnum):
    """§8 tiers."""

    MOCK = "mock"
    TINY = "tiny"
    LOCAL = "local"


PROFILES: tuple[Profile, ...] = tuple(Profile)


def _default_store_root() -> str:
    """Durable store root.

    On Kaggle the store must live under `/kaggle/working` or it dies with the
    session (§6.2). Everywhere else a repo-local directory keeps dev runs off
    the durable tier and out of git (`.gitignore` covers `ragstore/`).
    """
    if Path("/kaggle/working").exists():
        return "/kaggle/working/ragstore"
    return "./ragstore"


@dataclass(frozen=True, slots=True)
class Settings:
    """Whole-system configuration. Frozen; see module docstring."""

    profile: Profile = Profile.MOCK

    # --- storage (§6.3) ---
    store_root: str = field(default_factory=_default_store_root)
    vector_dtype: str = "float16"
    """float32 if §5.5 measures fp16 embedding degradation. Recorded in the
    manifest, never in the filename (§6.4)."""

    # --- embedding (§5.1) ---
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_device: str = "cuda:1"
    embed_batch: int = 32
    embed_max_length: int = 8192

    # --- reranking (§10) ---
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "cuda:1"
    rerank_top_n: int = 20
    rerank_max_len: int = 512
    """Hard ceiling of `bge-reranker-v2-m3`. Longer chunks are windowed, not
    silently truncated — that is the footgun §10 exists to avoid."""

    # --- generation (§5.2) ---
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_api: str = "native"
    """`native` = `/api/chat`. The OpenAI-compat route accepts neither `num_ctx`
    nor `think`, so choosing it silently invalidates the §5.3 KV budget."""

    gen_model: str = "qwen3:8b"
    gen_ctx: int = 8192
    gen_predict: int = 512
    gen_think: bool = False
    ollama_keep_alive: str = "30m"

    # --- chunking (§10) ---
    chunker: str = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- retrieval (§8) ---
    retrieve_top_k: int = 40
    context_top_n: int = 6
    context_token_budget: int = 3000
    rrf_k: int = 60
    """RRF's `k`. 60 is the value from the original Cormack et al. paper and is
    remarkably insensitive; it is a constant, not a tuning knob."""

    # --- gate (§11) ---
    gate_enabled: bool = True
    """Not a §12 transform. It governs whether the system speaks at all, which
    is a correctness property, so it defaults on."""

    gate_max_spans: int = 6
    gate_retry_once: bool = True

    # --- concurrency (§6.7) ---
    embed_concurrency: int = 2
    rerank_batch: int = 16
    open_session_limit: int = 8
    dense_gpu_threshold: int = 50_000
    """Above this many chunks the dense matmul moves to GPU 1; at or below, it
    runs on CPU over a memory-mapped fp32 upcast (§4)."""

    # --- transforms (§12) ---
    transform_rewrite: bool = False
    transform_hyde: bool = False
    transform_decompose: bool = False
    transform_multiquery: bool = False
    small_to_big: bool = False
    compress_enabled: bool = True

    # --- service (§16) ---
    host: str = "127.0.0.1"
    """Loopback is the real boundary (§16). The tunnel is the only ingress."""

    port: int = 8000
    public_mode: bool = False
    runtime_token: str = field(default="", repr=False)
    """Never logged, never persisted to disk (§16). `repr=False` because a
    token in a traceback or a debugger transcript is a leak."""

    cors_origins: tuple[str, ...] = ()

    # --- observability (§16) ---
    metrics_enabled: bool = True
    span_buffer_size: int = 200
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def sessions_dir(self) -> Path:
        return Path(self.store_root) / "sessions"

    @property
    def tmp_dir(self) -> Path:
        """Safe to delete at any time (§6.3)."""
        return Path(self.store_root) / "tmp"

    @property
    def static_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "static"

    def validated_vector_dtype(self) -> str:
        """Return `vector_dtype`, raising if it is not one of the two real ones.

        numpy has more float types than this system stores. A typo like
        `float16 ` or `f16` would otherwise produce a confusing error deep in
        the vector writer, or worse, silently default to fp32 and quietly
        double the store size.
        """
        if self.vector_dtype not in ("float16", "float32"):
            msg = f"vector_dtype must be float16 or float32, got {self.vector_dtype!r}"
            raise RagError(msg)
        return self.vector_dtype

    @property
    def index_signature(self) -> dict[str, str]:
        """Exactly the components that shape stored data (§6.5).

        Anything NOT in this dict is a query-time setting and can change freely
        against a warm index. Anything in it that differs from the manifest is a
        409, because appending across regimes produces a mixed index that still
        passes the §6.4 invariant.
        """
        return {
            "chunker": f"{self.chunker}/{self.chunk_size}-{self.chunk_overlap}",
            "embed_model": self.embed_model,
            "embed_dim": str(self.embed_dim),
            "vector_dtype": self.vector_dtype,
        }

    @property
    def retrieval_signature(self) -> dict[str, str]:
        """Query-time settings, hashed into every eval run (§15 determinism)."""
        return {
            "retrieve_top_k": str(self.retrieve_top_k),
            "context_top_n": str(self.context_top_n),
            "context_token_budget": str(self.context_token_budget),
            "rerank_model": self.rerank_model,
            "rerank_top_n": str(self.rerank_top_n),
            "rerank_max_len": str(self.rerank_max_len),
            "rerank_window": str(self.rerank_window_mode),
            "rrf_k": str(self.rrf_k),
            "gate_enabled": str(self.gate_enabled),
            "gate_max_spans": str(self.gate_max_spans),
            "transform_rewrite": str(self.transform_rewrite),
            "transform_hyde": str(self.transform_hyde),
            "transform_decompose": str(self.transform_decompose),
            "transform_multiquery": str(self.transform_multiquery),
            "small_to_big": str(self.small_to_big),
            "compress_enabled": str(self.compress_enabled),
        }

    @property
    def rerank_window_mode(self) -> str:
        """How chunks longer than `rerank_max_len` are handled (§10).

        Option 3 from the plan: window the chunk to a query-aware head
        (breadcrumb plus the most query-relevant sentences) and score that.
        Costs one embedding pass per candidate rather than several cross-encoder
        passes, and composes with §13 compression instead of duplicating it.
        """
        return "query_aware_head"

    @property
    def retrieval_config_hash(self) -> str:
        """Stable hash of the query-time config. Recorded with every eval run.

        Without it a 3-point "improvement" between two runs is uninterpretable
        (§15 determinism).
        """
        payload = json.dumps(self.retrieval_signature, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def vector_nbytes_per_row(self) -> int:
        """Bytes per stored vector row. Used by the §6.6 size arithmetic.

        1024 dims at fp16 is 2,048 bytes exactly; the plan quotes that figure
        repeatedly, so it is computed from first principles rather than
        hardcoded anywhere.
        """
        width = 2 if self.validated_vector_dtype() == "float16" else 4
        return self.embed_dim * width

    def is_gpu(self) -> bool:
        return self.embed_device.startswith("cuda")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def with_overrides(self, **kwargs: object) -> Settings:
        """Return a copy with fields replaced. The dataclass stays frozen."""
        return replace(self, **kwargs)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["profile"] = str(self.profile)
        d.pop("runtime_token", None)
        return d


#: Field-name sets driving env coercion. Explicit rather than inferred from the
#: annotation: a wrong guess here becomes a confusing `ValueError` at startup
#: instead of a clear config error.
_BOOL_FIELDS = {
    "gate_enabled",
    "gen_think",
    "public_mode",
    "metrics_enabled",
    "compress_enabled",
    "small_to_big",
    "transform_rewrite",
    "transform_hyde",
    "transform_decompose",
    "transform_multiquery",
}
_INT_FIELDS = {
    "embed_dim",
    "rerank_top_n",
    "retrieve_top_k",
    "context_top_n",
    "chunk_size",
    "chunk_overlap",
    "gen_ctx",
    "gen_predict",
    "embed_batch",
    "embed_concurrency",
    "rerank_batch",
    "open_session_limit",
    "dense_gpu_threshold",
    "gate_max_spans",
    "context_token_budget",
    "rerank_max_len",
    "rrf_k",
    "embed_max_length",
    "span_buffer_size",
    "port",
}
_STR_FIELDS = {
    "store_root",
    "vector_dtype",
    "embed_model",
    "embed_device",
    "rerank_model",
    "rerank_device",
    "ollama_url",
    "ollama_api",
    "gen_model",
    "ollama_keep_alive",
    "chunker",
    "host",
    "log_level",
}


def from_env(env: Mapping[str, str] | None = None, **overrides: object) -> Settings:
    """Build settings from `RAGLAB_*` environment variables.

    `runtime_token` is deliberately *not* read from the environment or `.env`
    (§16): it is printed once to the operator and never persisted. It may only
    be supplied programmatically, via the `runtime_token=` keyword, which the
    notebook does after generating it.
    """
    src = dict(os.environ if env is None else env)
    values: dict[str, object] = {}
    for key, raw in src.items():
        if not key.startswith(ENV_PREFIX):
            continue
        name = key[len(ENV_PREFIX) :].lower()
        if name == "runtime_token":
            continue
        if name in _INT_FIELDS:
            values[name] = int(raw)
        elif name in _BOOL_FIELDS:
            values[name] = raw.strip().lower() in ("1", "true", "yes", "on")
        elif name == "profile":
            values[name] = Profile(raw.strip().lower())
        elif name == "cors_origins":
            values[name] = tuple(x.strip() for x in raw.split(",") if x.strip())
        elif name in _STR_FIELDS:
            values[name] = raw
    values.update(overrides)
    known = {f for f in Settings.__dataclass_fields__}
    unknown = set(values) - known
    if unknown:
        msg = f"unknown settings: {sorted(unknown)}"
        raise ValueError(msg)
    return Settings(**values)  # type: ignore[arg-type]


def load_dotenv_if_present(path: str | Path = ".env") -> None:
    """Minimal `.env` loader. No dependency; a full parser is not warranted.

    Existing environment variables win, which is the behaviour every dotenv
    implementation converges on and the one that makes CI overrides work.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)
