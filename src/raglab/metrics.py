"""Dependency-free latency and counter metrics.

§16: dependency-free p50/p95 counters. No Prometheus client, no statsd, no
opentelemetry — the whole point is that this runs inside a Kaggle session with
4 CPU cores and a fixed dependency budget, and because `/metrics` is one of the
authenticated routes rather than a scraped sidecar.

Two decisions worth stating:

- **p95, not mean.** §2.4: averages hide the slow tail, which is what a user
  actually experiences. `percentile` uses the nearest-rank method on a sorted
  sample, which is exact for small n and needs no interpolation guess.
- **Stage timings are separate names, not a bag.** `gate_ms` is its own series
  because the gate is plausibly the slowest pre-generation stage and folding it
  into `retrieve_ms` makes a multi-second pause look like a hang (§11).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final

STAGE_NAMES: Final = (
    "embed_ms",
    "retrieve_ms",
    "rerank_ms",
    "gate_ms",
    "compress_ms",
    "generate_ms",
    "total_ms",
)

COUNTER_NAMES: Final = (
    "queries_total",
    "abstentions_total",
    "empty_retrievals_total",
    "citation_failures_total",
    "oom_retries_total",
    "errors_total",
)

#: Quantile bucket boundaries reported by `GET /metrics`.
LATENCY_SUMMARY_BUCKETS: Final = (10, 25, 50, 75, 90, 95, 99)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. `q` in [0, 100].

    Nearest-rank rather than linear interpolation: with the small sample counts
    a single-request service produces, an interpolated p95 is a value that was
    never observed by any request. Nearest-rank always reports something real.
    """
    if not values:
        return 0.0
    if not 0 <= q <= 100:
        msg = f"q must be in [0,100], got {q}"
        raise ValueError(msg)
    ordered = sorted(values)
    if q <= 0:
        return ordered[0]
    rank = math.ceil(q / 100 * len(ordered))
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]


@dataclass(slots=True)
class Timer:
    """Monotonic stopwatch. Wall-clock time is the wrong clock for latency:
    an NTP step mid-request produces a negative duration."""

    _start: float = field(default_factory=time.monotonic)

    def ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0

    def reset(self) -> None:
        self._start = time.monotonic()


@dataclass(slots=True)
class LatencySeries:
    """Bounded sample buffer for one metric name.

    Bounded because the store is long-lived and unbounded sample lists are a
    memory leak that only shows up on the second day of a 12-hour session.
    """

    name: str
    samples: deque[float]
    total: float = 0.0
    count: int = 0

    def observe(self, value: float) -> None:
        self.samples.append(value)
        self.total += value
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def p50(self) -> float:
        return percentile(list(self.samples), 50)

    @property
    def p95(self) -> float:
        return percentile(list(self.samples), 95)

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 2),
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "max_ms": round(self.max, 2),
        }


@dataclass(frozen=True, slots=True)
class Span:
    """One timed unit of work. Deliberately holds ids and durations only.

    §16 redaction: a span must not be able to carry retrieved text, because a
    span buffer is the most likely place for it to leak — it is a debugging
    feature, and debugging features get pasted into issues.

    That rule is enforced by `__post_init__` rather than by convention: the
    attribute mapping is validated to scalars only, so a caller who tries to
    attach `{"text": chunk.text}` gets an error at the call site instead of a
    leak in `/metrics`.
    """

    name: str
    duration_ms: float
    stage: str | None = None
    session_id: str | None = None
    attributes: dict[str, float | int | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.attributes.items():
            if not isinstance(value, (int, float, bool)) and value is not None:
                msg = (
                    f"span {self.name!r}: attribute {key!r} must be a scalar "
                    f"(int|float|bool|None), got {type(value).__name__}; §16 keeps "
                    "retrieved text and prompts out of metrics and logs"
                )
                raise TypeError(msg)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "duration_ms": round(self.duration_ms, 2)}
        if self.stage:
            d["stage"] = self.stage
        if self.session_id:
            d["session_id"] = self.session_id
        if self.attributes:
            d["attributes"] = self.attributes
        return d


class MetricsRegistry:
    """In-process metrics. One instance per service process."""

    def __init__(self, *, buffer_size: int = 200, enabled: bool = True) -> None:
        self.enabled = enabled
        self.buffer_size = buffer_size
        self._latency: dict[str, LatencySeries] = defaultdict(
            lambda: LatencySeries(name="", samples=deque(maxlen=buffer_size))
        )
        for name in STAGE_NAMES:
            self._latency[name]
        self._counters: dict[str, int] = {name: 0 for name in COUNTER_NAMES}
        self._gauges: dict[str, float] = defaultdict(float)
        self._spans: deque[Span] = deque(maxlen=buffer_size)

    # -- latency -------------------------------------------------------

    def observe(self, name: str, value_ms: float) -> None:
        """Record one duration. No-op when metrics are disabled."""
        if not self.enabled:
            return
        series = self._latency[name]
        series.name = name
        series.observe(value_ms)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        stage: str | None = None,
        session_id: str | None = None,
        **attributes: float | int | bool | None,
    ) -> Iterator[dict[str, Any]]:
        """Time a block and record a span.

        Yields a mutable dict so a caller can attach an outcome attribute
        (`hit_count`, `abstained`) without the timing plumbing. Attributes are
        scalars only, by type — that is the redaction rule enforced in code
        rather than by review.
        """
        t = Timer()
        extra: dict[str, Any] = {}
        try:
            yield extra
        finally:
            duration = t.ms()
            self.observe(name, duration)
            self.record_span(
                Span(name=name, duration_ms=duration, stage=stage, session_id=session_id,
                     attributes={**attributes, **extra})
            )

    def record_span(self, span: Span) -> None:
        if self.enabled:
            self._spans.append(span)

    def latency(self, name: str) -> LatencySeries:
        return self._latency[name]

    @property
    def spans(self) -> tuple[Span, ...]:
        return tuple(self._spans)

    def recent_spans(self, n: int = 20) -> list[dict[str, Any]]:
        """Last `n` spans, newest last."""
        items = list(self._spans)[-n:]
        return [s.to_dict() for s in items]

    # -- counters ------------------------------------------------------

    def incr(self, name: str, amount: int = 1) -> None:
        if self.enabled:
            self._counters[name] = self._counters.get(name, 0) + amount

    def counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    @property
    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    # -- gauges --------------------------------------------------------

    def set_gauge(self, name: str, value: float) -> None:
        """Point-in-time values: chunk counts, index bytes, live/physical rows.

        §6.4 tracks live vs physical chunk rows separately, because their
        divergence is otherwise invisible and reads as storage drift.
        """
        self._gauges[name] = value

    def gauge(self, name: str) -> float:
        return self._gauges.get(name, 0.0)

    @property
    def gauges(self) -> dict[str, float]:
        return dict(self._gauges)

    # -- reporting -----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe snapshot. Everything here is a number or a string."""
        latency = {
            name: series.to_dict()
            for name, series in sorted(self._latency.items())
            if series.count
        }
        return {
            "latency": latency,
            "counters": dict(sorted(self._counters.items())),
            "gauges": dict(sorted(self._gauges.items())),
        }

    def prometheus(self) -> str:
        """Prometheus text exposition, derived from the same numbers.

        Kept in sync with `snapshot()` deliberately rather than as a second
        implementation: two metric paths drift, and then `/metrics` and the API
        disagree about the system's behaviour.
        """
        lines: list[str] = []
        for name in STAGE_NAMES:
            s = self._latency[name]
            if not s.count:
                continue
            lines.append(f"# TYPE {name} summary")
            lines.append(f"{name}_count {s.count}")
            lines.append(f"{name}_sum_ms {s.total:.2f}")
            lines.append(f"{name}_p50_ms {s.p50:.2f}")
            lines.append(f"{name}_p95_ms {s.p95:.2f}")
            lines.append(f"{name}_max_ms {s.max:.2f}")
        for name, value in sorted(self._counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        for name, value in sorted(self._gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Clear all series. Tests only."""
        self.__init__(buffer_size=self.buffer_size, enabled=self.enabled)


#: Process-wide default registry. The service uses this; tests construct their own.
DEFAULT = MetricsRegistry()
