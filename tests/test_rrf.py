"""RRF properties (§22: monotone in rank, invariant to score scale).

§11's argument for rank-based fusion is that a dense cosine and a BM25 score are
not comparable, so combining them requires a normalisation step whose failure is
silent. These tests pin the two properties that make fusion safe: positions
decide, and positions alone.
"""

from __future__ import annotations

import unittest

from tests.base import RagTestCase
from raglab.index.rrf import DEFAULT_RRF_K, explain, fuse, fuse_scores, rrf_score


class TestRRFScore(RagTestCase):
    def test_first_place_is_largest_contribution(self) -> None:
        contributions = [rrf_score(r) for r in range(40)]
        self.assertEqual(contributions, sorted(contributions, reverse=True))

    def test_default_k_is_60(self) -> None:
        self.assertEqual(DEFAULT_RRF_K, 60)
        self.assertAlmostEqual(rrf_score(0), 1 / 60)

    def test_k_dampens_top_ranks(self) -> None:
        """The `+k` offset exists so a single #1 does not dominate a fusion of 40
        candidates. Without it, `1/(0+1)` is 60× `1/(0+60)`."""
        self.assertAlmostEqual(rrf_score(0, k=1) / rrf_score(39, k=1), 40.0)
        self.assertLess(rrf_score(0) / rrf_score(39), 3.0)

    def test_rejects_negative_rank(self) -> None:
        with self.assertRaises(ValueError):
            rrf_score(-1)

    def test_rejects_nonpositive_k(self) -> None:
        with self.assertRaises(ValueError):
            rrf_score(0, k=0)


class TestFuse(RagTestCase):
    def test_item_in_both_lists_beats_item_in_one(self) -> None:
        """The property hybrid retrieval depends on: agreement between two
        retrievers beats a single retriever's confidence.

        `a` and `b` are deliberately symmetric — a is rank 0 in `dense` and
        rank 1 in `bm25`, b is the mirror — so they tie exactly, which is what
        makes the tie-break worth asserting separately below.
        """
        fused = fuse({"dense": ["a", "b", "c"], "bm25": ["b", "a", "d"]})
        by_key = {f.key: f.score for f in fused}
        self.assertAlmostEqual(by_key["a"], by_key["b"], places=12)
        self.assertGreater(by_key["a"], by_key["c"], "two lists beat one")
        # `c` (rank 2 in dense) and `d` (rank 2 in bm25) are symmetric, so they
        # tie as well — a single retriever's rank-2 hit is the same either way.
        self.assertAlmostEqual(by_key["c"], by_key["d"], places=12)
        self.assertEqual([f.key for f in fused][:2], ["a", "b"])

    def test_tie_break_is_deterministic_and_prefers_earlier_first_seen(self) -> None:
        """`a` and `b` score identically above. Order must be stable so an
        ablation table is diffable, and it must not depend on dict iteration."""
        fused = fuse({"dense": ["a", "b", "c"], "bm25": ["b", "a", "d"]})
        self.assertEqual([f.key for f in fused], ["a", "b", "c", "d"])
        again = fuse({"dense": ["a", "b", "c"], "bm25": ["b", "a", "d"]})
        self.assertEqual([f.key for f in fused], [f.key for f in again])

    def test_single_list_agreement_at_same_rank_wins_on_total(self) -> None:
        fused = fuse({"a": ["x", "y", "z"], "b": ["y", "x"]})
        by_key = {f.key: f.score for f in fused}
        # y: rank 1 in `a` + rank 0 in `b`. x: rank 0 in `a` + rank 1 in `b`.
        self.assertAlmostEqual(by_key["y"], by_key["x"], places=12)
        self.assertGreater(by_key["x"], by_key["z"])

    def test_score_is_sum_over_lists(self) -> None:
        fused = fuse({"dense": ["a"], "bm25": ["a"]}, k=60)
        self.assertAlmostEqual(fused[0].score, 2 * rrf_score(0, 60))
        self.assertEqual(fused[0].ranks, (0, 0))
        self.assertEqual(fused[0].sources, ("dense", "bm25"))

    def test_provenance_is_recorded(self) -> None:
        fused = fuse({"dense": ["a", "b"], "bm25": ["b"]})
        by_key = {f.key: f for f in fused}
        self.assertEqual(by_key["a"].sources, ("dense",))
        self.assertEqual(by_key["b"].sources, ("dense", "bm25"))
        self.assertEqual(by_key["b"].ranks, (1, 0))

    def test_is_invariant_to_source_scores(self) -> None:
        """The whole point. `fuse` takes *ranked lists*, not scores, so there is
        no score scale for it to get wrong."""
        a = fuse_scores({"dense": ["x", "y"], "bm25": ["y", "x"]})
        b = fuse_scores({"dense": ["x", "y"], "bm25": ["y", "x"]})
        self.assertEqual(a, b)

    def test_deterministic_tie_break(self) -> None:
        first = [f.key for f in fuse({"d": ["a", "b"], "l": ["a", "b"]})]
        second = [f.key for f in fuse({"l": ["a", "b"], "d": ["a", "b"]})]
        self.assertEqual(first, ["a", "b"])
        # With sources renamed but order preserved, the same tie-break applies.
        self.assertEqual(second, ["a", "b"])

    def test_monotone_in_rank(self) -> None:
        """Promoting a key strictly improves its fused score.

        The claim is per-key, not corpus-wide: promoting `z` from rank 2 to rank
        0 raises `z`, while the keys it passed fall by exactly the same amount.
        A fusion that could *lower* a promoted key's score would make the whole
        ordering unexplainable.
        """
        before = dict((f.key, f.score) for f in fuse({"a": ["x", "y", "z"]}))
        after = dict((f.key, f.score) for f in fuse({"a": ["z", "x", "y"]}))
        self.assertGreater(after["z"], before["z"])
        # x took y's old rank, so it holds exactly the score y had.
        self.assertAlmostEqual(after["x"], before["y"], places=12)
        self.assertEqual([f.key for f in fuse({"a": ["z", "x", "y"]})][0], "z")

    def test_promoting_a_key_never_lowers_its_own_score(self) -> None:
        base = ["a", "b", "c", "d"]
        for i, key in enumerate(base):
            promoted = base[:i] + [key] + [k for k in base if k != key]
            moved = dict((f.key, f.score) for f in fuse({"s": promoted}))
            stayed = dict((f.key, f.score) for f in fuse({"s": base}))
            self.assertGreaterEqual(
                moved[key], stayed[key] - 1e-12, msg=f"promoting {key} lowered its score"
            )

    def test_top_n_truncates(self) -> None:
        fused = fuse({"a": [str(i) for i in range(50)]}, top_n=5)
        self.assertEqual(len(fused), 5)

    def test_empty_input(self) -> None:
        self.assertEqual(fuse({}), [])

    def test_empty_lists(self) -> None:
        self.assertEqual(fuse({"a": [], "b": []}), [])

    def test_weights_scale_a_source(self) -> None:
        """A weight multiplies that source's whole contribution, including for
        keys the source shares with another list."""
        lists = {"a": ["x"], "b": ["y", "x"]}
        plain = dict((f.key, f.score) for f in fuse(lists))
        weighted = dict((f.key, f.score) for f in fuse(lists, weights={"b": 3.0}))
        # x appears in both lists, so its `b` contribution triples.
        self.assertAlmostEqual(
            weighted["x"] - plain["x"], 2.0 * rrf_score(1), places=12
        )
        # y appears only in `b`, so all of its contribution triples.
        self.assertAlmostEqual(weighted["y"], 3.0 * plain["y"], places=12)

    def test_default_weights_are_all_one(self) -> None:
        """Introducing weights without a measured reason is how a hybrid setup
        starts encoding a guess (§11)."""
        lists = {"a": ["x", "y"], "b": ["y", "x"]}
        self.assertEqual(
            dict((f.key, f.score) for f in fuse(lists, weights={"a": 1.0, "b": 1.0})),
            dict((f.key, f.score) for f in fuse(lists)),
        )

    def test_integer_keys_supported(self) -> None:
        """Rows are addressed by ordinal (§6.4), so keys are usually ints."""
        fused = fuse({"dense": [0, 1, 2], "bm25": [2, 1, 0]})
        self.assertEqual({f.key for f in fused}, {0, 1, 2})
        self.assertEqual(fuse_scores({"dense": [0, 1]})[0], rrf_score(0))


class TestExplain(RagTestCase):
    def test_names_source_and_rank_per_contribution(self) -> None:
        lines = explain(fuse({"dense": ["a", "b"], "bm25": ["b"]}))
        self.assertEqual(len(lines), 2)
        self.assertTrue(any("dense#0" in l for l in lines))
        self.assertTrue(any("dense#1" in l and "bm25#0" in l for l in lines))


if __name__ == "__main__":
    unittest.main()
