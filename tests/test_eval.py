"""Evaluation harness (§15, §22).

Two things here are less about arithmetic and more about honesty, which is why
they get more assertions than the formulas do:

- `correct_abstention` is meaningless without `abstention_rate` alongside it.
  A system that refuses everything scores 15/15 on the unanswerable set, so
  `GenerationMetrics.validate` refuses to render a row carrying one alone.
- A question with no gold evidence contributes nothing to a retrieval average.
  Including it as a zero would make an unanswerable question look like a
  retrieval failure, which is the opposite of what it is.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.base import RagTestCase
from raglab.errors import RagError
from raglab.eval.dataset import (
    EvalQuestion,
    EvalSet,
    RunRecord,
    assert_comparable,
    load_questions,
)
from raglab.eval.gen_metrics import (
    AbstentionMetrics,
    GenerationMetrics,
    abstention_metrics,
    citation_metrics,
    extract_markers,
    segment_claims,
)
from raglab.eval.report import TableRow, comparable_groups, render_markdown, render_report
from raglab.eval.retrieval_metrics import (
    aggregate,
    evaluate,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_question,
    span_coverage,
)
from raglab.metrics import percentile
from raglab.types import Chunk

GOLD = ("sha256:doc1", 100, 200)
OTHER = ("sha256:doc2", 500, 600)


def q(qid: str, *, answerable: bool = True, gold=(GOLD,), **kw) -> EvalQuestion:
    return EvalQuestion(
        qid=qid,
        question=kw.pop("question", f"question {qid}"),
        answerable=answerable,
        gold_docs=kw.pop("gold_docs", ("corpus/a.md",)),
        gold_spans=tuple(gold) if answerable else (),
        **kw,
    )


class TestRetrievalMetrics(RagTestCase):
    def test_recall_counts_fraction_of_gold_found(self) -> None:
        gold = [GOLD, OTHER]
        ranked = [GOLD, ("sha256:doc3", 0, 10)]
        self.assertAlmostEqual(recall_at_k(ranked, gold, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranked, gold, 10), 0.5)

    def test_recall_is_one_when_all_gold_is_found(self) -> None:
        self.assertEqual(recall_at_k([GOLD, OTHER], [GOLD, OTHER], 5), 1.0)

    def test_recall_with_no_gold_is_zero_not_an_error(self) -> None:
        self.assertEqual(recall_at_k([GOLD], [], 5), 0.0)

    def test_hit_rate_differs_from_recall_for_multi_gold(self) -> None:
        gold = [GOLD, OTHER]
        ranked = [GOLD]
        self.assertEqual(hit_rate_at_k(ranked, gold, 5), 1.0)
        self.assertAlmostEqual(recall_at_k(ranked, gold, 5), 0.5)

    def test_single_gold_makes_hit_rate_equal_recall(self) -> None:
        """§2.4: for single-gold questions these are the same number, which is why
        reporting both is not redundant for a mixed set."""
        ranked = ["a", "b", GOLD]
        self.assertEqual(
            hit_rate_at_k(ranked, [GOLD], 2), recall_at_k(ranked, [GOLD], 2)
        )

    def test_reciprocal_rank(self) -> None:
        self.assertEqual(reciprocal_rank([GOLD], [GOLD]), 1.0)
        self.assertAlmostEqual(reciprocal_rank(["a", GOLD], [GOLD]), 0.5)
        self.assertAlmostEqual(reciprocal_rank(["a", "b", GOLD], [GOLD]), 1 / 3)
        self.assertEqual(reciprocal_rank(["a"], [GOLD]), 0.0)

    def test_ndcg_rewards_early_hits(self) -> None:
        """§2.4: a relevant hit at rank 1 outweighs the same hit at rank 8."""
        early = ndcg_at_k([GOLD] + [("x", i, i) for i in range(10)], [GOLD], 10)
        late = ndcg_at_k([("x", i, i) for i in range(7)] + [GOLD], [GOLD], 10)
        self.assertGreater(early, late)
        self.assertAlmostEqual(ndcg_at_k([GOLD], [GOLD], 10), 1.0)
        self.assertEqual(ndcg_at_k(["a"], [GOLD], 10), 0.0)

    def test_ndcg_with_no_gold_is_zero(self) -> None:
        self.assertEqual(ndcg_at_k([GOLD], [], 10), 0.0)

    def test_k_zero_is_zero(self) -> None:
        self.assertEqual(recall_at_k([GOLD], [GOLD], 0), 0.0)
        self.assertEqual(ndcg_at_k([GOLD], [GOLD], 0), 0.0)

    def test_span_coverage_is_partial_credit(self) -> None:
        """§9: gold spans are clean-space offsets, and a chunk overlapping 80% of
        one is a different outcome from missing it."""
        spans = {"c1": ("d1", 100, 180), "c2": ("d2", 0, 10)}
        cov = span_coverage(["c1", "c2"], spans, [("d1", 100, 200)], 10)
        self.assertAlmostEqual(cov, 0.8, places=6)

    def test_span_coverage_does_not_double_count_overlap(self) -> None:
        spans = {"c1": ("d1", 100, 150), "c2": ("d1", 150, 200)}
        self.assertAlmostEqual(span_coverage(["c1", "c2"], spans, [("d1", 100, 200)], 10), 1.0)

    def test_span_coverage_respects_k(self) -> None:
        spans = {"c1": ("d2", 0, 10), "c2": ("d1", 100, 200)}
        # At k=1 only the non-covering chunk is in scope.
        self.assertEqual(span_coverage(["c1", "c2"], spans, [("d1", 100, 200)], 1), 0.0)
        self.assertAlmostEqual(span_coverage(["c1", "c2"], spans, [("d1", 100, 200)], 2), 1.0)

    def test_span_coverage_zero_when_gold_absent(self) -> None:
        self.assertEqual(span_coverage(["c1"], {"c1": ("d1", 0, 5)}, [GOLD], 10), 0.0)

    def test_span_coverage_wrong_document_is_not_counted(self) -> None:
        """Clean-space offsets are meaningless across documents: byte 100 of one
        document has nothing to do with byte 100 of another."""
        spans = {"c1": ("other-doc", 100, 200)}
        self.assertEqual(span_coverage(["c1"], spans, [GOLD], 10), 0.0)


class TestAggregate(RagTestCase):
    def test_unanswerable_questions_are_excluded_from_recall(self) -> None:
        """A question with no gold evidence has no recall to score; counting it as
        zero would make it look like a retrieval failure."""
        results = [
            score_question(q("a"), [GOLD]),
            score_question(q("b", answerable=False), []),
        ]
        m = aggregate(results)
        self.assertEqual(m.n_answerable, 1)
        self.assertEqual(m.n_unanswerable, 1)
        self.assertAlmostEqual(m.recall_at_k[10], 1.0)

    def test_missing_result_scores_zero_rather_than_being_dropped(self) -> None:
        """An ablation cell that failed to retrieve for one question should show a
        zero in that row. Skipping it would make a broken run look better than a
        working one."""
        questions = [q("a"), q("b")]
        m = evaluate(questions, {"a": [GOLD]})  # `b` has no entry at all
        self.assertEqual(m.n_questions, 2)
        self.assertAlmostEqual(m.recall_at_k[10], 0.5)

    def test_all_unanswerable_reports_no_averages(self) -> None:
        m = evaluate([q("a", answerable=False), q("b", answerable=False)], {})
        self.assertEqual(m.recall_at_k, {})
        self.assertEqual(m.mrr_at_10, 0.0)
        self.assertEqual(m.n_unanswerable, 2)

    def test_latency_summary(self) -> None:
        results = [score_question(q(str(i)), [GOLD], latency_ms=float(i)) for i in range(1, 101)]
        m = aggregate(results)
        self.assertAlmostEqual(m.latency["p95_ms"], percentile([float(i) for i in range(1, 101)], 95))


class TestCitationMetrics(RagTestCase):
    ANSWER = (
        "Term 1 tuition is N450,000 [S1]. "
        "It is payable before the resumption date [S2]. "
        "Boarding is refunded pro-rata."
    )

    def test_markers_are_extracted_in_order(self) -> None:
        self.assertEqual(extract_markers(self.ANSWER), ["S1", "S2"])

    def test_uncited_sentence_is_reported(self) -> None:
        """§14 failure mode 3: unsupported claims are the dangerous ones, because
        they read as correct."""
        m = citation_metrics(self.ANSWER, valid_markers={"S1", "S2"}, n_sources=2)
        self.assertEqual(m.n_sentences, 3)
        self.assertEqual(m.n_cited_sentences, 2)
        self.assertEqual(m.n_unverified, 1)
        self.assertEqual(len(m.unverified_texts), 1)
        self.assertIn("pro-rata", m.unverified_texts[0])

    def test_hallucinated_marker_is_caught_not_hidden(self) -> None:
        """§14 failure mode 1: `[S7]` when only 2 sources existed."""
        answer = "Term 1 tuition is N450,000 [S1]. Term 2 is due in March [S7]."
        m = citation_metrics(answer, valid_markers={"S1"}, n_sources=1)
        self.assertEqual(m.broken_markers, ("S7",))

    def test_over_citation_is_flagged(self) -> None:
        """§14 failure mode 2: citing everything looks like rigour and is noise.

        `citation_recall` stays 1.0 — every sentence is cited — so only the
        precision terms and the over-citation ratio can see the problem. That is
        the whole reason precision is defined against over-citation rather than
        against unresolvable markers, which are counted separately.
        """
        answer = (
            "Everything here is true [S1] [S2] [S3]. "
            "Term 1 is N450,000 [S1]. "
            "Term 2 is due in March [S2]."
        )
        m = citation_metrics(answer, valid_markers={"S1", "S2", "S3"}, n_sources=3)
        self.assertEqual(m.n_cited_sentences, 3)
        self.assertAlmostEqual(m.recall, 1.0, places=6)
        # 5 markers across 3 cited sentences: 2 excess.
        self.assertAlmostEqual(m.over_citation_ratio, 2 / 5, places=6)
        self.assertLess(m.precision, 1.0)
        self.assertEqual(m.broken_markers, ())

    def test_healthy_answer_scores_full_precision(self) -> None:
        """One marker per claim, every source used: both terms are 1.0."""
        answer = (
            "Term 1 is N450,000 [S1]. "
            "Term 2 is due in March [S2]. "
            "Boarding is refunded pro-rata [S3]."
        )
        m = citation_metrics(answer, valid_markers={"S1", "S2", "S3"}, n_sources=3)
        self.assertAlmostEqual(m.precision, 1.0, places=6)
        self.assertAlmostEqual(m.recall, 1.0, places=6)
        self.assertEqual(m.over_citation_ratio, 0.0)

    def test_unused_source_lowers_precision(self) -> None:
        answer = (
            "Term 1 is N450,000 [S1]. "
            "Term 2 is due in March [S1]. "
            "Boarding is refunded pro-rata [S1]."
        )
        m = citation_metrics(answer, valid_markers={"S1", "S2", "S3"}, n_sources=3)
        self.assertAlmostEqual(m.precision, 1 / 3, places=6)

    def test_marker_follows_its_sentence(self) -> None:
        claims = segment_claims("Fee is N450,000 [S1]. Next sentence.")
        self.assertEqual(claims[0].marker, "S1")
        self.assertIsNone(claims[1].marker)

    def test_empty_answer_scores_zero_without_raising(self) -> None:
        m = citation_metrics("", valid_markers={"S1"}, n_sources=1)
        self.assertEqual(m.n_sentences, 0)

    def test_precision_is_not_reported_on_one_sentence(self) -> None:
        """A precision computed over one sentence is a boolean; printing `1.0`
        invites reading it as a quality number."""
        m = citation_metrics("Only one [S1].", valid_markers={"S1"}, n_sources=1)
        self.assertEqual(m.n_sentences, 1)
        self.assertEqual(m.n_cited_sentences, 1)


class TestAbstentionPair(RagTestCase):
    """§15's pair, and the degenerate case it exists to expose."""

    def test_good_system_is_high_and_low(self) -> None:
        m = abstention_metrics(
            abstained_by_qid={"a": False, "b": False, "u1": True, "u2": True},
            answerable_by_qid={"a": True, "b": True, "u1": False, "u2": False},
        )
        self.assertEqual(m.correct_abstention, 1.0)
        self.assertAlmostEqual(m.abstention_rate, 0.5)
        self.assertFalse(m.is_degenerate())

    def test_refuses_everything_is_flagged_degenerate(self) -> None:
        """The reason the pair is mandatory: 15/15 on the unanswerable set while
        being completely useless."""
        answerable = {f"q{i}": True for i in range(45)}
        unanswerable = {f"u{i}": False for i in range(15)}
        verdicts = {**answerable, **unanswerable}
        refuses_everything = dict.fromkeys(verdicts, True)
        m = abstention_metrics(
            abstained_by_qid=refuses_everything,
            answerable_by_qid=verdicts,
        )
        self.assertEqual(m.correct_abstention, 1.0)
        self.assertAlmostEqual(m.abstention_rate, 1.0)
        self.assertTrue(m.is_degenerate())
        self.assertEqual(m.false_abstentions, 45)

    def test_never_abstains_is_not_degenerate(self) -> None:
        verdicts = {f"q{i}": True for i in range(45)} | {f"u{i}": False for i in range(15)}
        m = abstention_metrics(
            abstained_by_qid=dict.fromkeys(verdicts, False),
            answerable_by_qid=verdicts,
        )
        self.assertEqual(m.correct_abstention, 0.0)
        self.assertEqual(m.abstention_rate, 0.0)
        self.assertFalse(m.is_degenerate())

    def test_false_abstentions_are_reported(self) -> None:
        """Answerable questions the system refused. `correct_abstention` alone
        hides exactly this."""
        m = abstention_metrics(
            abstained_by_qid={"a": True, "u1": True},
            answerable_by_qid={"a": True, "u1": False},
        )
        self.assertEqual(m.correct_abstention, 1.0)
        self.assertEqual(m.false_abstentions, 1)
        self.assertAlmostEqual(m.abstention_rate, 1.0)

    def test_missing_qid_counts_as_not_abstained(self) -> None:
        """A run that crashed halfway must look bad, not flattering."""
        m = abstention_metrics(
            abstained_by_qid={"u1": True}, answerable_by_qid={"u1": False, "a": True}
        )
        self.assertEqual(m.n_total, 2)
        self.assertAlmostEqual(m.abstention_rate, 0.5)

    def test_no_unanswerable_questions_raises(self) -> None:
        """Scoring abstention over a set with no unanswerable cases is the
        situation §15 forbids, so it is an error rather than a flattering 0.0."""
        with self.assertRaises(RagError):
            abstention_metrics(
                abstained_by_qid={"a": False}, answerable_by_qid={"a": True}
            )

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(RagError):
            abstention_metrics(abstained_by_qid={}, answerable_by_qid={})


class TestGenerationMetricsRow(RagTestCase):
    def test_refuses_a_row_without_the_abstention_pair(self) -> None:
        m = GenerationMetrics(faithfulness=0.71)
        with self.assertRaises(RagError):
            m.to_dict()

    def test_renders_both_columns(self) -> None:
        pair = abstention_metrics(
            abstained_by_qid={"u1": True}, answerable_by_qid={"a": True, "u1": False}
        )
        d = GenerationMetrics(faithfulness=0.71, answer_relevancy=0.8, abstention=pair).to_dict()
        self.assertIn("correct_abstention", d["abstention"])
        self.assertIn("abstention_rate", d["abstention"])
        self.assertFalse(d["abstention_degenerate"])

    def test_absent_abstention_metrics_type(self) -> None:
        self.assertIsInstance(AbstentionMetrics(0, 0, 0, 0, 0, 0), AbstentionMetrics)


class TestEvalSetValidation(RagTestCase):
    def test_answerable_without_gold_is_rejected(self) -> None:
        bad = EvalQuestion(qid="q1", question="x", answerable=True)
        with self.assertRaises(RagError):
            bad.validate()

    def test_unanswerable_with_gold_is_a_labelling_error(self) -> None:
        bad = EvalQuestion(qid="q1", question="x", answerable=False, gold_spans=(GOLD,))
        with self.assertRaises(RagError):
            bad.validate()

    def test_empty_span_is_rejected(self) -> None:
        bad = EvalQuestion(qid="q1", question="x", gold_spans=(("d", 10, 10),))
        with self.assertRaises(RagError):
            bad.validate()

    def test_duplicate_qid_is_rejected(self) -> None:
        es = EvalSet(session_id="s", questions=(q("a"), q("a")))
        with self.assertRaises(RagError):
            es.validate()

    def test_session_id_is_required(self) -> None:
        """Evaluation is per session; a set without one cannot be attributed."""
        with self.assertRaises(RagError):
            EvalSet(session_id="", questions=(q("a"),)).validate()

    def test_unknown_tag_is_rejected(self) -> None:
        with self.assertRaises(RagError):
            q("a", tags=("not-a-tag",)).validate()

    def test_composition_counts(self) -> None:
        es = EvalSet(
            session_id="s",
            questions=(
                q("a", tags=("factual", "single-hop")),
                q("b", tags=("factual",)),
                q("u", answerable=False, tags=("unanswerable",)),
            ),
        )
        c = es.composition()
        self.assertEqual((c["total"], c["answerable"], c["unanswerable"]), (3, 2, 1))
        self.assertEqual(c["tags"]["factual"], 2)

    def test_recommended_composition_is_advisory(self) -> None:
        """A 10-question smoke set is legitimate. What is not legitimate is scoring
        abstention with no unanswerable questions, which the metrics refuse."""
        small = EvalSet(session_id="s", questions=(q("a"), q("u", answerable=False)))
        self.assertFalse(small.meets_recommended_composition())
        full = EvalSet(
            session_id="s",
            questions=tuple(
                [q(f"a{i}") for i in range(45)] + [q(f"u{i}", answerable=False) for i in range(15)]
            ),
        )
        self.assertTrue(full.meets_recommended_composition())

    def test_round_trips_through_jsonl_on_disk(self) -> None:
        es = EvalSet(session_id="s1", questions=(q("a"), q("u", answerable=False)))
        with tempfile.TemporaryDirectory() as d:
            path = es.write(Path(d) / "eval" / "questions.jsonl")
            self.assertTrue(path.exists())
            back = EvalSet.load(path, session_id="s1")
            self.assertEqual(len(back), 2)
            self.assertEqual(back.questions[0].qid, "a")

    def test_subset_and_tag_filter(self) -> None:
        es = EvalSet(
            session_id="s",
            questions=(q("a", tags=("factual",)), q("b", tags=("multi-hop",))),
        )
        self.assertEqual(len(es.subset(["a"])), 1)
        self.assertEqual([x.qid for x in es.with_tags("multi-hop")], ["b"])

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(RagError):
            load_questions("/nonexistent/questions.jsonl")


class TestRunComparability(RagTestCase):
    def _run(self, **kw) -> RunRecord:
        base = dict(
            run_id="r1",
            session_id="s1",
            label="512/64 fixed hybrid",
            matrix="retrieval",
            embed_model="BAAI/bge-m3",
            embed_model_revision="abc",
            judge_model="qwen3:8b",
        )
        base.update(kw)
        return RunRecord(**base)

    def test_same_setup_is_comparable(self) -> None:
        assert_comparable(self._run(), self._run(run_id="r2"))

    def test_different_session_is_not_comparable(self) -> None:
        """§15: every metric is only meaningful against a specific corpus."""
        with self.assertRaises(RagError):
            assert_comparable(self._run(), self._run(run_id="r2", session_id="s2"))

    def test_different_embedder_is_not_comparable(self) -> None:
        with self.assertRaises(RagError):
            assert_comparable(self._run(), self._run(run_id="r2", embed_model="other"))

    def test_different_judge_is_not_comparable(self) -> None:
        with self.assertRaises(RagError):
            assert_comparable(self._run(), self._run(run_id="r2", judge_model="qwen3:14b"))

    def test_record_carries_the_provenance_the_plan_names(self) -> None:
        """§15 determinism and §10's rerank_window column."""
        r = self._run(rerank_window="query_aware_head", chunk_count=412, chunk_rows=500)
        d = json.loads(r.to_json())
        for field in (
            "config_hash", "chunker", "embed_model_revision", "rerank_window",
            "chunk_count", "chunk_rows", "seed", "library_versions", "judge_model",
        ):
            self.assertIn(field, d)

    def test_live_and_physical_counts_are_both_recorded(self) -> None:
        """§6.4: their divergence is otherwise invisible and reads as storage drift."""
        d = self._run(chunk_count=412, chunk_rows=500).to_dict()
        self.assertEqual(d["chunk_count"], 412)
        self.assertEqual(d["chunk_rows"], 500)


class TestReport(RagTestCase):
    def _row(self, label: str, config_hash: str = "h1", matrix: str = "retrieval", **metrics) -> TableRow:
        return TableRow(
            label=label,
            matrix=matrix,
            config_hash=config_hash,
            chunker="512/64",
            retrieval="hybrid",
            rerank_window="query_aware_head",
            metrics={"recall@5": 0.68, "mrr@10": 0.61, "ndcg@10": 0.63, "p95 ms": 135.0, **metrics},
        )

    def test_renders_retrieval_columns(self) -> None:
        md = render_markdown([self._row("512/64 fixed hybrid")])
        for col in ("recall@5", "mrr@10", "ndcg@10", "p95 ms"):
            self.assertIn(col, md, col)
        self.assertIn("0.680", md)

    def test_abstention_pair_appears_together_or_not_at_all(self) -> None:
        md = render_markdown([
            self._row(
                "hybrid+rerank", matrix="generation",
                faithfulness=0.81, relevance=0.9,
                correct_abstention=1.0, abstention_rate=0.27,
            )
        ])
        self.assertIn("correct_abstention", md)
        self.assertIn("abstention_rate", md)

    def test_footnotes_state_the_bias(self) -> None:
        md = render_markdown([
            self._row("x", matrix="generation", faithfulness=0.8, correct_abstention=1.0, abstention_rate=0.3)
        ])
        self.assertIn("same `qwen3:8b`", md)
        self.assertIn("bias", md.lower())

    def test_rerank_window_difference_is_called_out(self) -> None:
        a = self._row("512")
        b = TableRow(label="2048", matrix="retrieval", config_hash="h1",
                     rerank_window="windowed_max_pool",
                     metrics={"recall@5": 0.7, "mrr@10": 0.6, "ndcg@10": 0.6, "p95 ms": 200.0})
        self.assertIn("rerank_window", render_markdown([a, b]))

    def test_empty_report_says_so(self) -> None:
        self.assertIn("No runs recorded", render_markdown([]))

    def test_groups_split_on_config_hash(self) -> None:
        groups = comparable_groups([self._row("a", "h1"), self._row("b", "h2"), self._row("c", "h1")])
        self.assertEqual([len(g) for g in groups], [2, 1])

    def test_deltas(self) -> None:
        base = self._row("base")
        var = self._row("variant", **{"recall@5": 0.81})
        from raglab.eval.report import markdown_delta_table

        md = markdown_delta_table(base, [var])
        self.assertIn("+0.130", md)


class TestChunkToGoldKey(RagTestCase):
    def test_chunk_span_key_matches_gold_form(self) -> None:
        """recall@k compares chunk identities against gold; both must be
        `(doc_id, clean_start, clean_end)` in clean space (§9, §15)."""
        c = Chunk(
            ordinal=0, doc_id="sha256:doc1", chunk_index=0, text="x",
            clean_start=100, clean_end=200,
        )
        self.assertEqual(c.span_key, GOLD)


if __name__ == "__main__":
    unittest.main()
