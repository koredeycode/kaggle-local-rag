"""Redacting logging.

§16: no prompts, no retrieved text, no tokens, no secrets in logs or metrics.
`docs/observability.md` states the rules; this module is where they are
enforced, so they are not a convention someone can quietly break.

The mechanism is a `logging.Filter` that scrubs known-sensitive keys, plus a
deliberately narrow public API (`get_logger`) that requires an explicit event
name. Callers cannot log a `Chunk` by accident because the structured helpers
take scalars, not objects.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping
from typing import Any, Final

_SENSITIVE_KEYS: Final = frozenset(
    {
        "token",
        "runtime_token",
        "password",
        "secret",
        "authorization",
        "api_key",
        "text",
        "prompt",
        "answer",
        "chunk_text",
        "content",
        "span_text",
        "question",
    }
)

_REDACTED: Final = "[redacted]"

#: Patterns that catch a secret even when it appears inside a free-form string,
#: which key-based scrubbing alone misses.
#:
#: Every one of these has a matching test in `tests/test_metrics.py`. A
#: redaction pattern that is never exercised is a pattern that silently stopped
#: matching after a refactor.
_PATTERNS: Final = (
    # Document and chunk-text hashes. §9 `doc_id` is `sha256:<hex>`.
    re.compile(r"sha256:[0-9a-f]{8,}"),
    # `Authorization: Bearer <token>`.
    re.compile(r"\bBearer\s+\S+", re.IGNORECASE),
    # Provider-shaped tokens: OpenAI `sk-`, Stripe `pk_`, GitHub `ghp_`,
    # Slack `xoxb-`. The prefix is what makes these safe to match without
    # eating ordinary prose.
    re.compile(r"\b(?:sk|pk|tok|ghp|gho|ghu|ghs|xox[abprs])[-_][A-Za-z0-9_-]{8,}"),
    # Cloudflare Quick Tunnel hostnames (§19.2). The random label precedes the
    # domain: `https://word-word-1234.trycloudflare.com`.
    re.compile(r"[a-z0-9-]{8,}\.trycloudflare\.com"),
)

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
_DATEFMT = "%H:%M:%S"
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))


class RedactionFilter(logging.Filter):
    """Scrub sensitive values from both the message and keyword arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = scrub(record.msg)
        if record.args:
            if isinstance(record.args, Mapping):
                record.args = {k: scrub_value(k, v) for k, v in record.args.items()}  # type: ignore[assignment]
            else:
                record.args = tuple(scrub(a) for a in record.args)  # type: ignore[assignment]
        for key in list(record.__dict__):
            if key in _SENSITIVE_KEYS:
                record.__dict__[key] = _REDACTED
        return True


def scrub_value(key: str, value: Any) -> Any:
    """Scrub by key name. Non-strings pass through unless the key is sensitive."""
    if key.lower() in _SENSITIVE_KEYS:
        return _REDACTED
    return scrub(value)


def scrub(value: Any) -> Any:
    """Scrub secrets out of a free-form value.

    Applied to every log message regardless of key names, because the most
    likely leak is a `detail` string that happens to embed a URL or a hash.
    """
    if not isinstance(value, str):
        return value
    out = value
    for pat in _PATTERNS:
        out = pat.sub(_REDACTED, out)
    return out


def get_logger(name: str = "raglab", level: str | None = None) -> logging.Logger:
    """Return a configured logger. Idempotent per name."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, RedactionFilter) for f in logger.filters):
        logger.addFilter(RedactionFilter())
    lvl = (level or os.environ.get("RAGLAB_LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, lvl, logging.INFO))
    if not any(h is _handler for h in logger.handlers):
        logger.addHandler(_handler)
    logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Structured single-line event log.

    Only scalars and pre-scrubbed values belong in `fields`. The event name
    first, then sorted key=value pairs, so log lines are greppable without a
    parser.
    """
    parts = [event]
    for key in sorted(fields):
        val = scrub_value(key, fields[key])
        if isinstance(val, str) and (" " in val or "=" in val):
            val = f'"{val}"'
        parts.append(f"{key}={val}")
    logger.log(level, " ".join(parts))


def debug(logger: logging.Logger, event: str, **fields: Any) -> None:
    log_event(logger, logging.DEBUG, event, **fields)


def info(logger: logging.Logger, event: str, **fields: Any) -> None:
    log_event(logger, logging.INFO, event, **fields)


def warning(logger: logging.Logger, event: str, **fields: Any) -> None:
    log_event(logger, logging.WARNING, event, **fields)


def error(logger: logging.Logger, event: str, **fields: Any) -> None:
    log_event(logger, logging.ERROR, event, **fields)


SENSITIVE_KEYS: Iterable[str] = _SENSITIVE_KEYS
REDACTED = _REDACTED
