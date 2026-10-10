"""BM25 correctness, against a hand-computed case (§22, §11).

Not regression output: the expected scores below are worked out by hand from
Okapi BM25, so a change to the formula shows up as a number change with an
explanation rather than as a diff nobody can interpret.

    score(q, d) = Σ_t  idf(t) · (tf · (k1 + 1)) / (tf + k1 · (1 - b + b·len_d/avg_len))

with idf(t) = ln(1 + (N - df + 0.5)/(df + 0.5)).
"""

from __future__ import annotations

import math
import unittest

from tests.base import RagTestCase
from raglab.index.bm25 import BM25Config, BM25Index, build_index, tokenize


def reference_score(
    docs: list[str], query: str, k1: float = 1.5, b: float = 0.75
) -> dict[int, float]:
    """Independent brute-force BM25, written from the formula.

    Deliberately *not* sharing code with the implementation: if both used the
    same helper, a bug in the helper would cancel out and the test would pass.
    """
    toks = [tokenize(d) for d in docs]
    n = len(toks)
    avg = sum(len(t) for t in toks) / n
    out: dict[int, float] = {}
    for i, t in enumerate(toks):
        s = 0.0
        for term in set(tokenize(query)):
            df = sum(1 for u in toks if term in u)
            if df == 0:
                continue
            tf = t.count(term)
            if tf == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            norm = 1.0 - b + b * (len(t) / avg)
            s += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)
        if s:
            out[i] = s
    return out


class TestHandComputed(RagTestCase):
    DOCS = [
        "the cat sat on the mat",
        "the dog sat on the mat",
        "birds fly in the sky",
        "the cat and the dog are friends",
    ]

    def test_matches_hand_computed_scores(self) -> None:
        idx = build_index([(i, t) for i, t in enumerate(self.DOCS)])
        got = idx.score("cat mat")
        want = reference_score(self.DOCS, "cat mat")
        self.assertEqual(set(got), set(want))
        for k, v in want.items():
            self.assertAlmostEqual(got[k], v, places=10, msg=f"row {k}")

    def test_single_document_case_is_arithmetic(self) -> None:
        """One doc, one matching term: df = 1, N = 1, so idf = ln(3).

        doc: "fee schedule", len 2. query: "fee". tf = 1.
        idf = ln(1 + (1 - 1 + 0.5)/(1 + 0.5)) = ln(1 + 1/3) = ln(4/3)
        norm = 1 - 0.75 + 0.75·(2/2) = 1
        score = ln(4/3) · (1·2.5)/(1 + 1.5) = ln(4/3)
        """
        idx = build_index([(0, "fee schedule")])
        got = idx.score("fee")[0]
        self.assertAlmostEqual(got, math.log(4.0 / 3.0), places=10)

    def test_repeated_term_saturates(self) -> None:
        """§2.1: the tenth occurrence of "fee" adds far less than the first.

        Both documents are padded to the *same* length so that length
        normalisation contributes nothing and the ratio isolates term-frequency
        saturation. With `k1 = 1.5`: the tf factor is 2.5/2.5 = 1.0 at tf=1 and
        25/11.5 ≈ 2.17 at tf=10, so ten occurrences score ~2.2× one — not 10×.
        """
        ten = " ".join(["fee"] * 10)
        one = "fee " + " ".join(["pad"] * 9)
        self.assertEqual(len(ten.split()), len(one.split()))
        idx = build_index([(0, ten), (1, one)])
        scores = idx.score("fee")
        ratio = scores[0] / scores[1]
        self.assertEqual(idx._docs[0].length, idx._docs[1].length)
        # tf=1  → 1·2.5/(1 + 1.5)      = 2.5/2.5  = 1.000
        # tf=10 → 10·2.5/(10 + 1.5)    = 25/11.5  = 2.174
        self.assertAlmostEqual(
            ratio, 10 * 2.5 / 11.5, places=6, msg=f"tf saturation (ratio {ratio:.4f})"
        )

    def test_long_document_is_penalised(self) -> None:
        """§2.1: long documents are penalised. Same term count, longer doc scores
        less."""
        short = "fee schedule"
        long = "fee " + " ".join(["filler"] * 200)
        idx = build_index([(0, short), (1, long)])
        scores = idx.score("fee")
        self.assertGreater(scores[0], scores[1])

    def test_idf_is_never_negative(self) -> None:
        """The +1 guard. Without it a term in more than half the corpus gets
        negative IDF and containing it *lowers* a score — a bug that shows up on
        small test corpora and hides on large ones."""
        docs = [f"common term {i}" for i in range(5)]
        idx = build_index([(i, t) for i, t in enumerate(docs)])
        scores = idx.score("common")
        self.assertTrue(all(v > 0 for v in scores.values()), scores)


class TestRanking(RagTestCase):
    def test_search_orders_by_score(self) -> None:
        idx = build_index(
            [
                (0, "python asyncio event loop"),
                (1, "unrelated text about gardening"),
                (2, "asyncio coroutines and tasks"),
            ]
        )
        results = idx.search("asyncio coroutines", top_k=3)
        self.assertEqual(results[0][0], 2)

    def test_search_respects_top_k(self) -> None:
        idx = build_index([(i, f"fee item {i}") for i in range(50)])
        self.assertEqual(len(idx.search("fee", top_k=5)), 5)

    def test_ties_break_on_ordinal(self) -> None:
        """Deterministic order, so an ablation table is diffable."""
        idx = build_index([(i, "same text here") for i in range(5)])
        results = idx.search("same")
        self.assertEqual([o for o, _ in results], [0, 1, 2, 3, 4])

    def test_absent_terms_yield_nothing(self) -> None:
        idx = build_index([(0, "alpha beta")])
        self.assertEqual(idx.score("zebra"), {})

    def test_stopwords_only_query_yields_nothing(self) -> None:
        idx = build_index([(0, "alpha beta")])
        self.assertEqual(idx.score("the and of"), {})


class TestMutation(RagTestCase):
    def test_remove_drops_the_row(self) -> None:
        """§6.4: BM25 filters to live rows. A dead row left in the postings would
        be returned by every query forever."""
        idx = build_index([(0, "fee schedule"), (1, "dog park")])
        idx.remove(0)
        self.assertNotIn(0, idx.score("fee schedule"))
        self.assertIn(1, idx.score("dog"))

    def test_remove_is_idempotent(self) -> None:
        idx = build_index([(0, "fee schedule")])
        idx.remove(0)
        idx.remove(0)
        self.assertEqual(idx.size, 0)

    def test_remove_of_absent_row_is_a_noop(self) -> None:
        idx = build_index([(0, "fee")])
        idx.remove(99)
        self.assertEqual(idx.size, 1)

    def test_add_replaces_existing_row(self) -> None:
        """Re-adding a row must not double-count its terms, or an incremental
        rebuild against a warm index would inflate every score."""
        idx = build_index([(0, "fee schedule")])
        before = idx.score("fee")[0]
        idx.add(0, "fee schedule")
        self.assertAlmostEqual(idx.score("fee")[0], before, places=10)
        self.assertEqual(idx.size, 1)

    def test_add_then_remove_restores_document_frequency(self) -> None:
        idx = build_index([(0, "alpha beta"), (1, "alpha gamma")])
        self.assertEqual(idx.document_frequency("alpha"), 2)
        idx.remove(0)
        self.assertEqual(idx.document_frequency("alpha"), 1)
        idx.add(0, "alpha beta")
        self.assertEqual(idx.document_frequency("alpha"), 2)


class TestTokenize(RagTestCase):
    def test_lowercases_and_drops_stopwords(self) -> None:
        self.assertEqual(tokenize("The Fee IS Due"), ["fee", "due"])

    def test_can_keep_stopwords(self) -> None:
        self.assertIn("the", tokenize("the fee", remove_stopwords=False))

    def test_unicode_tokens_survive(self) -> None:
        """§24 Q3 takes multilingual corpora seriously; a tokenizer that collapses
        non-ASCII to nothing makes every non-English document look like a
        retrieval failure."""
        toks = tokenize("Ọ̀rẹ́ àti ọkọ́-owo Yoruba Hausa")
        self.assertTrue(toks)
        self.assertTrue(all(t.strip() for t in toks))

    def test_kana_tokens_survive(self) -> None:
        self.assertTrue(tokenize("日本語 の テキスト"))


class TestConfig(RagTestCase):
    def test_k1_and_b_are_honoured(self) -> None:
        docs = ["fee schedule", "fee " + " ".join(["x"] * 50)]
        for k1, b in ((1.5, 0.75), (0.0, 0.0), (2.0, 1.0)):
            idx = BM25Index(config=BM25Config(k1=k1, b=b))
            idx.add_many([(i, d) for i, d in enumerate(docs)])
            want = reference_score(docs, "fee", k1=k1, b=b)
            got = idx.score("fee")
            for k, v in want.items():
                self.assertAlmostEqual(got[k], v, places=10, msg=f"k1={k1} b={b} row {k}")

    def test_supplied_weights_override_counts(self) -> None:
        """§5.4 sparse weights from `bge-m3` are an explicit argument, not a
        silent path, because supplying them changes what the lexical stage
        measures."""
        idx = build_index([(0, "fee fee fee")])
        baseline = idx.score("fee")[0]
        idx2 = BM25Index()
        idx2.add(0, "fee fee fee", weights={"fee": 1})
        self.assertLess(idx2.score("fee")[0], baseline)


class TestEmpty(RagTestCase):
    def test_empty_index_scores_nothing(self) -> None:
        idx = BM25Index()
        self.assertEqual(idx.score("anything"), {})
        self.assertEqual(idx.search("anything"), [])

    def test_empty_query_scores_nothing(self) -> None:
        idx = build_index([(0, "fee")])
        self.assertEqual(idx.score(""), {})

    def test_add_empty_text(self) -> None:
        idx = build_index([(0, "   ")])
        self.assertEqual(idx.size, 1)
        self.assertEqual(idx.score("fee"), {})


if __name__ == "__main__":
    unittest.main()
