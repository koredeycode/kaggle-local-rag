"""Metrics: p50/p95 on a known sequence, plus redaction (§16, §22)."""

from __future__ import annotations

import logging
import unittest

from tests.base import RagTestCase
from raglab.logging import RedactionFilter, get_logger, scrub
from raglab.metrics import (
    DEFAULT,
    MetricsRegistry,
    Span,
    Timer,
    percentile,
)


class TestPercentile(RagTestCase):
    def test_p50_and_p95_on_known_sequence(self) -> None:
        """1..100: p50 is the 50th value, p95 the 95th, by nearest rank."""
        values = [float(i) for i in range(1, 101)]
        self.assertEqual(percentile(values, 50), 50.0)
        self.assertEqual(percentile(values, 95), 95.0)
        self.assertEqual(percentile(values, 100), 100.0)
        self.assertEqual(percentile(values, 0), 1.0)

    def test_always_reports_an_observed_value(self) -> None:
        """Nearest-rank, not interpolated.

        With three samples [1, 2, 100] an interpolated p95 is 95.6 — a number no
        request ever produced. Nearest-rank reports 100, which happened.
        """
        values = [1.0, 2.0, 100.0]
        self.assertEqual(percentile(values, 95), 100.0)
        self.assertIn(percentile(values, 95), values)

    def test_single_sample(self) -> None:
        self.assertEqual(percentile([42.0], 50), 42.0)
        self.assertEqual(percentile([42.0], 99), 42.0)

    def test_empty_is_zero(self) -> None:
        self.assertEqual(percentile([], 95), 0.0)

    def test_rejects_out_of_range_q(self) -> None:
        for bad in (-1, 101):
            with self.assertRaises(ValueError):
                percentile([1.0], bad)

    def test_unsorted_input(self) -> None:
        self.assertEqual(percentile([5.0, 1.0, 3.0], 50), 3.0)


class TestLatencySeries(RagTestCase):
    def test_summary(self) -> None:
        m = MetricsRegistry()
        for i in range(1, 101):
            m.observe("retrieve_ms", float(i))
        s = m.latency("retrieve_ms")
        self.assertEqual(s.count, 100)
        self.assertEqual(s.p50, 50.0)
        self.assertEqual(s.p95, 95.0)
        self.assertEqual(s.max, 100.0)
        self.assertAlmostEqual(s.mean, 50.5)

    def test_samples_are_bounded(self) -> None:
        """Unbounded sample lists are a leak that only shows on day two of a
        12-hour session."""
        m = MetricsRegistry(buffer_size=10)
        for i in range(100):
            m.observe("generate_ms", float(i))
        self.assertEqual(len(m.latency("generate_ms").samples), 10)
        self.assertEqual(m.latency("generate_ms").count, 100)


class TestRegistry(RagTestCase):
    def test_counters(self) -> None:
        m = MetricsRegistry()
        self.assertEqual(m.counter("queries_total"), 0)
        m.incr("queries_total")
        m.incr("queries_total", 2)
        self.assertEqual(m.counter("queries_total"), 3)

    def test_stage_counters_are_predeclared(self) -> None:
        m = MetricsRegistry()
        m.observe("gate_ms", 940.0)
        self.assertIn("gate_ms", m.snapshot()["latency"])

    def test_gate_is_not_folded_into_retrieve(self) -> None:
        """§11/§17.3: `gate_ms` is its own stage. If it were folded into
        `retrieve_ms`, a multi-second unexplained pause would look like a
        retrieval problem rather than the gate doing its job."""
        m = MetricsRegistry()
        m.observe("retrieve_ms", 41.0)
        m.observe("gate_ms", 940.0)
        lat = m.snapshot()["latency"]
        self.assertEqual(lat["retrieve_ms"]["p95_ms"], 41.0)
        self.assertEqual(lat["gate_ms"]["p95_ms"], 940.0)

    def test_gauges_for_live_and_physical_counts(self) -> None:
        """§6.4: `chunk_count` (live) and `chunk_rows` (physical) are reported
        separately, because their divergence otherwise looks like storage
        drift."""
        m = MetricsRegistry()
        m.set_gauge("chunk_count", 412)
        m.set_gauge("chunk_rows", 500)
        self.assertEqual(m.gauge("chunk_count"), 412)
        self.assertEqual(m.gauge("chunk_rows"), 500)

    def test_disabled_registry_records_nothing(self) -> None:
        m = MetricsRegistry(enabled=False)
        m.observe("retrieve_ms", 5.0)
        m.incr("queries_total")
        self.assertEqual(m.snapshot()["latency"], {})
        self.assertEqual(m.counter("queries_total"), 0)

    def test_prometheus_mirrors_snapshot(self) -> None:
        """Two metric paths drift, and then `/metrics` and the API disagree about
        the system's behaviour."""
        m = MetricsRegistry()
        for i in range(1, 21):
            m.observe("rerank_ms", float(i))
        m.incr("abstentions_total")
        m.set_gauge("chunk_count", 12)
        text = m.prometheus()
        snap = m.snapshot()
        self.assertIn("rerank_ms_count 20", text)
        self.assertIn("abstentions_total 1", text)
        self.assertIn("chunk_count 12", text)
        self.assertEqual(snap["counters"]["abstentions_total"], 1)

    def test_spans_are_bounded_and_ordered(self) -> None:
        m = MetricsRegistry(buffer_size=5)
        for i in range(10):
            m.record_span(Span(name=f"s{i}", duration_ms=float(i)))
        names = [s.name for s in m.spans]
        self.assertEqual(len(names), 5)
        self.assertEqual(names, [f"s{i}" for i in range(5, 10)])

    def test_span_context_manager_records_outcome(self) -> None:
        m = MetricsRegistry()
        with m.span("retrieve_ms", stage="retrieve", session_id="s1") as extra:
            extra["hit_count"] = 6
        spans = m.recent_spans(1)
        self.assertEqual(spans[0]["stage"], "retrieve")
        self.assertEqual(spans[0]["attributes"]["hit_count"], 6)
        self.assertGreaterEqual(spans[0]["duration_ms"], 0.0)


class TestTimer(RagTestCase):
    def test_monotonic_and_non_negative(self) -> None:
        t = Timer()
        first = t.ms()
        second = t.ms()
        self.assertGreaterEqual(second, first)
        self.assertGreaterEqual(first, 0.0)


class TestRedaction(RagTestCase):
    """§16: no prompts, retrieved text, tokens, or secrets in logs."""

    def test_scrubs_provider_tokens(self) -> None:
        for raw in (
            "token sk-abcdefgh12345678",
            "ghp_ABCDEFGHIJKLMNOP",
            "xoxb-1234567890-abcdef",
        ):
            self.assertIn("[redacted]", scrub(raw), raw)

    def test_scrubs_bearer_headers(self) -> None:
        out = scrub("Authorization: Bearer abcdef123456")
        self.assertNotIn("abcdef123456", out)
        self.assertIn("[redacted]", out)

    def test_scrubs_content_hashes(self) -> None:
        """`doc_id` is `sha256:<hex>` (§9). A hash in a log identifies a document,
        which is the first step from "which corpus" to "which passage"."""
        out = scrub("indexed doc sha256:ab12cd34ef56a789")
        self.assertIn("[redacted]", out)
        self.assertNotIn("ab12cd34ef56a789", out)

    def test_scrubs_tunnel_urls(self) -> None:
        """§19.2: the Quick Tunnel URL is the one piece of the system's address
        that must not end up in a shared log."""
        out = scrub("tunnel up: https://foo-bar-baz-qux-1234.trycloudflare.com")
        self.assertIn("[redacted]", out)
        self.assertNotIn("trycloudflare.com/foo", out)
        self.assertNotIn("baz-qux", out)

    def test_leaves_ordinary_prose_alone(self) -> None:
        """Over-scrubbing is its own failure: a log that redacts everything is
        useless for debugging."""
        text = "retrieve_ms=41 rerank_ms=180 stage=retrieve session=abc123"
        self.assertEqual(scrub(text), text)

    def test_filter_redacts_sensitive_attributes(self) -> None:
        """`extra` cannot use a reserved LogRecord key, so the attribute is
        attached the way the filter actually sees it: as a record attribute."""
        log = get_logger("raglab.test.redaction")
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        log.addHandler(Capture())
        record = log.makeRecord(
            "raglab.test", logging.ERROR, "f", 1, "event", None, None,
            func="f",
        )
        record.__dict__["chunk_text"] = "secret passage text"
        record.__dict__["runtime_token"] = "sk-abcdefgh12345678"
        self.assertTrue(log.filter(record))
        self.assertEqual(record.__dict__["chunk_text"], "[redacted]")
        self.assertEqual(record.__dict__["runtime_token"], "[redacted]")

    def test_span_carries_no_text_by_type(self) -> None:
        """§16: spans are the most likely leak, because they are a debugging
        feature and debugging features get pasted into issues. The attribute
        type is scalars-only, which enforces it in code."""
        span = Span(name="retrieve", duration_ms=1.0, attributes={"hit_count": 3})
        for value in span.attributes.values():
            self.assertIsInstance(value, (int, float, bool, type(None)))
        with self.assertRaises((TypeError, ValueError)):
            Span(name="x", duration_ms=0.0, attributes={"text": "leak"})  # type: ignore[dict-item]

    def test_default_registry_exists(self) -> None:
        self.assertIsInstance(DEFAULT, MetricsRegistry)
        self.assertIsInstance(DEFAULT.counter("queries_total"), int)


if __name__ == "__main__":
    unittest.main()
