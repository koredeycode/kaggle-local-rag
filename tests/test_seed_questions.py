"""The committed seed eval set must stay loadable and internally consistent (§15).

Two things this pins that nothing else would:

- `gold_spans` is empty at Phase 0, and every gold span remains **clean space**
  (§9). A test that documents this keeps Phase 2's `eval/author.py` from quietly
  filling in original-file offsets, which would make recall@k undefined rather
  than wrong.
- The answerable:unanswerable ratio stays near 3:1, because that ratio is what
  stops `correct_abstention` from being the only number anyone reports (§15).
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.base import RagTestCase
from raglab.eval.dataset import EvalSet, load_questions

SEED = Path(__file__).resolve().parents[1] / "eval" / "questions.sample.jsonl"


class TestSeedQuestionSet(RagTestCase):
    def setUp(self) -> None:
        self.questions = load_questions(SEED)

    def test_loads(self) -> None:
        self.assertGreater(len(self.questions), 20)

    def test_every_question_validates(self) -> None:
        for question in self.questions:
            question.validate()

    def test_constructs_as_an_eval_set(self) -> None:
        es = EvalSet(session_id="synthetic", questions=tuple(self.questions))
        es.validate()
        self.assertEqual(len(es), len(self.questions))

    def test_qids_are_unique(self) -> None:
        qids = [q.qid for q in self.questions]
        self.assertEqual(len(qids), len(set(qids)))

    def test_answerable_ratio_is_roughly_three_to_one(self) -> None:
        """§15: 45/15. A set dominated by unanswerable questions scores
        abstention beautifully and retrieval terribly."""
        answerable = sum(q.answerable for q in self.questions)
        unanswerable = len(self.questions) - answerable
        self.assertGreater(answerable, 0)
        self.assertGreater(unanswerable, 0)
        self.assertGreaterEqual(answerable / unanswerable, 1.5)

    def test_every_question_is_tagged(self) -> None:
        """§15: a single blended recall number hides which case is failing."""
        for q in self.questions:
            self.assertTrue(q.tags, f"{q.qid} has no tags")

    def test_unanswerable_questions_carry_no_gold(self) -> None:
        for q in self.questions:
            if not q.answerable:
                self.assertEqual(q.gold_spans, (), q.qid)
                self.assertEqual(q.gold_docs, (), q.qid)

    def test_answerable_questions_name_a_gold_document(self) -> None:
        for q in self.questions:
            if q.answerable:
                self.assertTrue(q.gold_docs, q.qid)

    def test_gold_spans_are_empty_until_the_authoring_tool_exists(self) -> None:
        """Phase 0 state, stated as a test.

        Phase 2's `eval/author.py` fills these from a chunk click. If one ever
        lands here by hand, this test fails and asks the author to confirm the
        offsets are **clean space** (§9) rather than original-file offsets —
        which is the mistake that makes recall@k undefined.
        """
        with_spans = [q.qid for q in self.questions if q.gold_spans]
        self.assertEqual(
            with_spans, [], f"gold spans added by hand: {with_spans}. Use eval/author.py."
        )

    def test_contains_the_case_types_the_ablation_probes(self) -> None:
        """If the seed set loses its `exact-identifier` or `paraphrased`
        questions, the two ablations §15 cares most about become unreadable."""
        by_tag = {t: len(self.questions_for_tag(t)) for t in
                  ("single-hop", "multi-hop", "paraphrased", "exact-identifier", "unanswerable")}
        for tag, count in by_tag.items():
            self.assertGreater(count, 0, f"no questions tagged {tag}")

    def questions_for_tag(self, tag: str):
        return [q for q in self.questions if tag in q.tags]


if __name__ == "__main__":
    unittest.main()
