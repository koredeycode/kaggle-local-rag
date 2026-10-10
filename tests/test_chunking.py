"""Chunker invariants (§10, §22).

Per-chunker fixed invariants — size bounds, overlap, no mid-boundary splits for
`structural` — plus a determinism check. The determinism check matters more than
it looks: chunk boundaries feed the §6.4 ordinal invariant and the citation
offsets, so a chunker that returns different spans on identical input makes every
stored offset quietly unreproducible.
"""

from __future__ import annotations

import unittest

from tests.base import RagTestCase, make_document
from raglab.chunking import available, get_chunker
from raglab.chunking.base import Span
from raglab.types import Chunk

PROSE = (
    "Lagos State school fees for the 2026 academic session have been published. "
    "The fee schedule covers tuition, compulsory levies, and a resumption charge. "
    "Term 1 tuition for senior secondary is N450,000 per pupil. "
    "Term 2 tuition is payable by the end of March. "
    "Junior secondary tuition is substantially lower, at N185,000 per term. "
    "\n\n"
    "Refund policy. A pupil who withdraws after the first term is not entitled to "
    "a refund of tuition already paid. Applications must be made in writing to "
    "the principal within fourteen days of withdrawal. "
    "Boarding charges are refundable on a pro-rata basis."
)

LONG_PROSE = PROSE + "\n\n" + (PROSE * 4)


class TestFixedChunker(RagTestCase):
    def test_size_and_overlap_bounds(self) -> None:
        c = get_chunker("fixed", size=32, overlap=8)
        chunks = c.chunk(make_document(LONG_PROSE))
        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertLessEqual(ch.token_count, 32)

    def test_offsets_are_valid_and_ordered(self) -> None:
        c = get_chunker("fixed", size=40, overlap=10)
        doc = make_document(LONG_PROSE)
        prev_end = 0
        for ch in c.chunk(doc):
            self.assertOffsetsInRange(doc.text, ch.clean_start, ch.clean_end)
            self.assertLessEqual(ch.clean_start, prev_end)  # overlap may repeat
            prev_end = ch.clean_end

    def test_chunk_text_matches_offsets(self) -> None:
        """The contract that makes citations verifiable (§9)."""
        c = get_chunker("fixed", size=64, overlap=16)
        doc = make_document(LONG_PROSE)
        for ch in c.chunk(doc):
            self.assertEqual(doc.text[ch.clean_start : ch.clean_end], ch.text)

    def test_overlap_repeats_tokens(self) -> None:
        c = get_chunker("fixed", size=50, overlap=25)
        chunks = c.chunk(make_document(PROSE * 3))
        self.assertGreaterEqual(len(chunks), 2)
        a, b = chunks[0], chunks[1]
        self.assertTrue(a.text[-20:].strip() and b.text[:20].strip())
        # A 25-token overlap of a 50-token chunk means ~half is repeated.
        self.assertLess(b.clean_start, a.clean_end)

    def test_zero_overlap_leaves_only_whitespace_gaps(self) -> None:
        """With overlap=0, consecutive chunks must not share a character.

        They are not necessarily *contiguous*: token boundaries skip the
        whitespace between two words, so the gap between one chunk's end and the
        next chunk's start is the separating whitespace and nothing else.
        """
        c = get_chunker("fixed", size=32, overlap=0)
        doc = make_document(LONG_PROSE)
        chunks = c.chunk(doc)
        for a, b in zip(chunks, chunks[1:], strict=False):
            self.assertLessEqual(b.clean_start, a.clean_end + 1)
            gap = doc.text[a.clean_end : b.clean_start]
            self.assertEqual(gap.strip(), "", f"non-whitespace gap {gap!r}")

    def test_rejects_overlap_at_or_above_size(self) -> None:
        for bad in (50, 51, 100):
            with self.assertRaises(ValueError):
                get_chunker("fixed", size=50, overlap=bad)

    def test_rejects_nonpositive_size(self) -> None:
        with self.assertRaises(ValueError):
            get_chunker("fixed", size=0)

    def test_ordinals_are_sequential_from_offset(self) -> None:
        c = get_chunker("fixed", size=64, overlap=16)
        chunks = c.chunk(make_document(LONG_PROSE), start_ordinal=100)
        self.assertEqual([ch.ordinal for ch in chunks], list(range(100, 100 + len(chunks))))


class TestRecursiveChunker(RagTestCase):
    def test_respects_size(self) -> None:
        c = get_chunker("recursive", size=64, overlap=0)
        for ch in c.chunk(make_document(LONG_PROSE)):
            self.assertLessEqual(ch.token_count, 64 + 1)

    def test_prefers_paragraph_boundaries(self) -> None:
        c = get_chunker("recursive", size=200, overlap=0)
        chunks = c.chunk(make_document(PROSE))
        text = PROSE
        for ch in chunks:
            self.assertEqual(text[ch.clean_start : ch.clean_end], ch.text)

    def test_offsets_valid(self) -> None:
        c = get_chunker("recursive", size=48, overlap=8)
        doc = make_document(LONG_PROSE)
        prev_end = 0
        for ch in c.chunk(doc):
            self.assertOffsetsInRange(doc.text, ch.clean_start, ch.clean_end)
            self.assertLessEqual(ch.clean_start, prev_end)
            prev_end = ch.clean_end

    def test_never_cuts_mid_word_at_paragraph_level(self) -> None:
        """A chunk boundary should land on whitespace, not inside a token."""
        c = get_chunker("recursive", size=40, overlap=0)
        doc = make_document(PROSE)
        for ch in c.chunk(doc):
            body = doc.text[ch.clean_start : ch.clean_end]
            self.assertEqual(body, body.strip())

    def test_single_word_document(self) -> None:
        c = get_chunker("recursive", size=32, overlap=0)
        chunks = c.chunk(make_document("Naira"))
        self.assertEqual(len(chunks), 1)

    def test_empty_document(self) -> None:
        # `semantic` has no `overlap` parameter, so each chunker is built with
        # only the parameters it actually declares.
        for name, params in (
            ("fixed", {"size": 64, "overlap": 8}),
            ("recursive", {"size": 64, "overlap": 8}),
            ("structural", {"size": 64}),
            ("semantic", {"size": 64}),
        ):
            c = get_chunker(name, **params)
            self.assertEqual(c.chunk(make_document("")), [], name)

    def test_whitespace_only_document(self) -> None:
        c = get_chunker("recursive", size=64, overlap=0)
        self.assertEqual(c.chunk(make_document("   \n\n  \t ")), [])


class TestStructuralChunker(RagTestCase):
    DOC = (
        "Lagos State\n"
        "Education\n"
        "Fee Structure 2026\n"
        "Term 1 tuition is N450,000.\n"
        "Term 2 tuition is payable by March.\n"
        "Refund Policy\n"
        "No refund after the first term.\n"
        "Boarding is pro-rata.\n"
    )

    #: `(heading_text, offset)` — the headings a loader would recover. Body
    #: lines are deliberately *not* headings: this is the property being tested.
    HEADINGS = ("Lagos State", "Education", "Fee Structure 2026", "Refund Policy")

    def _doc(self):
        text = self.DOC
        marks = tuple((text.index(h), h) for h in self.HEADINGS)
        return make_document(text, heading_path_at=marks)

    def test_never_crosses_a_heading_boundary(self) -> None:
        """The defining property. A structural chunk must contain whole sections."""
        c = get_chunker("structural", size=1024, min_size=0)
        doc = self._doc()
        heading_offsets = {m[0] for m in doc.heading_path_at}
        chunks = c.chunk(doc)
        self.assertGreaterEqual(len(chunks), 2)
        for ch in chunks:
            self.assertTrue(ch.heading_path)
            # Every chunk starts at a heading offset, or at 0 when the document
            # opens with content before any heading.
            self.assertIn(ch.clean_start, heading_offsets | {0})

    def test_sections_start_at_headings_or_document_start(self) -> None:
        """`min_size` folds tiny sections together, but a chunk's start is always
        a heading offset or 0. Merging across a boundary would undo the one job
        this strategy has."""
        c = get_chunker("structural", size=1024, min_size=1)
        doc = self._doc()
        allowed = {m[0] for m in doc.heading_path_at} | {0}
        for ch in c.chunk(doc):
            self.assertIn(ch.clean_start, allowed)

    def test_chunks_carry_breadcrumbs(self) -> None:
        c = get_chunker("structural", size=1024, min_size=0)
        for ch in c.chunk(self._doc()):
            self.assertIn("Lagos State", ch.heading_path)
            # `embed_text` joins with " > " (§9).
            self.assertIn(" > ".join(ch.heading_path), ch.embed_text)

    def test_no_headings_degrades_to_size_capped(self) -> None:
        c = get_chunker("structural", size=40, min_size=0)
        chunks = c.chunk(make_document(LONG_PROSE))
        self.assertGreater(len(chunks), 1, "flat prose must not become one chunk")
        for ch in chunks:
            self.assertEqual(ch.heading_path, ())

    def test_oversized_section_is_split_but_keeps_breadcrumb(self) -> None:
        c = get_chunker("structural", size=16, min_size=0)
        chunks = c.chunk(self._doc())
        self.assertGreater(len(chunks), 3)
        for ch in chunks:
            self.assertIn("Lagos State", ch.heading_path)

    def test_small_sections_merge_forward(self) -> None:
        c = get_chunker("structural", size=1024, min_size=20)
        merged = c.chunk(self._doc())
        self.assertLess(len(merged), 4)


class TestSemanticChunker(RagTestCase):
    def test_requires_embedder_flag(self) -> None:
        from raglab.chunking.base import BaseChunker

        self.assertTrue(get_chunker("semantic").requires_embedder)
        self.assertFalse(get_chunker("fixed").requires_embedder)
        self.assertIsInstance(get_chunker("semantic"), BaseChunker)

    def test_offsets_valid(self) -> None:
        c = get_chunker("semantic", size=48, threshold=0.5, buffer=1)
        doc = make_document(LONG_PROSE)
        for ch in c.chunk(doc):
            self.assertOffsetsInRange(doc.text, ch.clean_start, ch.clean_end)
            self.assertEqual(doc.text[ch.clean_start : ch.clean_end], ch.text)

    def test_rejects_bad_threshold(self) -> None:
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                get_chunker("semantic", threshold=bad)

    def test_rejects_zero_buffer(self) -> None:
        with self.assertRaises(ValueError):
            get_chunker("semantic", buffer=0)

    def test_threshold_one_yields_few_boundaries(self) -> None:
        """Threshold 1.0 means nothing is 'unlike its neighbour', so cuts come
        only from the size cap. This is the knob, so it must visibly work."""
        loose = get_chunker("semantic", size=64, threshold=0.0, buffer=1)
        tight = get_chunker("semantic", size=64, threshold=1.0, buffer=1)
        doc = make_document(LONG_PROSE)
        self.assertGreaterEqual(
            len(loose.chunk(doc)), len(tight.chunk(doc)) - 1
        )


class TestSmallToBig(RagTestCase):
    def test_children_carry_parent_span(self) -> None:
        from raglab.chunking.small_to_big import SmallToBigChunker

        child = get_chunker("fixed", size=40, overlap=0)
        stb = SmallToBigChunker(child=child, parent_size=200)
        chunks = stb.chunk(make_document(LONG_PROSE))
        self.assertTrue(chunks)
        for ch in chunks:
            ps, pe = ch.metadata["parent_start"], ch.metadata["parent_end"]
            self.assertIsInstance(ps, int)
            self.assertIsInstance(pe, int)
            self.assertLessEqual(ps, ch.clean_start)
            self.assertGreaterEqual(pe, ch.clean_end)

    def test_parent_text_contains_child(self) -> None:
        from raglab.chunking.small_to_big import SmallToBigChunker

        child = get_chunker("fixed", size=40, overlap=0)
        stb = SmallToBigChunker(child=child, parent_size=200)
        doc = make_document(LONG_PROSE)
        ch = stb.chunk(doc)[0]
        expanded = SmallToBigChunker.expand(ch, doc.text)
        self.assertIn(ch.text[:20].strip(), expanded.strip()[:80])

    def test_expand_falls_back_to_child_text(self) -> None:
        from raglab.chunking.small_to_big import SmallToBigChunker

        plain = get_chunker("fixed", size=40, overlap=0).chunk(make_document(PROSE))[0]
        self.assertEqual(SmallToBigChunker.expand(plain, PROSE), plain.text)

    def test_expand_clamps_stale_parent_span(self) -> None:
        """A stale parent span must never raise — a citation panel crashing a
        query is worse than a truncated one."""
        from raglab.chunking.small_to_big import SmallToBigChunker

        ch = Chunk(
            ordinal=0,
            doc_id="d",
            chunk_index=0,
            text="x",
            clean_start=0,
            clean_end=1,
            metadata={"parent_start": 0, "parent_end": 10**9},
        )
        self.assertEqual(SmallToBigChunker.expand(ch, "short"), "short")

    def test_rejects_bad_parent_overlap(self) -> None:
        from raglab.chunking.small_to_big import SmallToBigChunker

        child = get_chunker("fixed", size=40, overlap=0)
        with self.assertRaises(ValueError):
            SmallToBigChunker(child=child, parent_size=100, parent_overlap=100)


class TestRegistry(RagTestCase):
    def test_all_four_strategies_registered(self) -> None:
        self.assertEqual(
            set(available()), {"fixed", "recursive", "semantic", "structural"}
        )

    def test_unknown_chunker_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_chunker("nope")

    def test_spec_signature_includes_params(self) -> None:
        c = get_chunker("fixed", size=256, overlap=32)
        sig = c.spec.signature()
        self.assertIn("256", sig)
        self.assertIn("32", sig)
        self.assertIn("fixed", sig)


class TestDeterminism(RagTestCase):
    def test_same_input_same_output(self) -> None:
        for name, params in (
            ("fixed", {"size": 64, "overlap": 8}),
            ("recursive", {"size": 64, "overlap": 8}),
            ("structural", {"size": 64}),
            ("semantic", {"size": 64}),
        ):
            c = get_chunker(name, **params)
            a = c.chunk(make_document(LONG_PROSE))
            b = c.chunk(make_document(LONG_PROSE))
            self.assertEqual(
                [(x.clean_start, x.clean_end) for x in a],
                [(x.clean_start, x.clean_end) for x in b],
                f"{name} is not deterministic",
            )

    def test_spans_are_validated_against_document(self) -> None:
        """`validate` must reject out-of-range spans rather than trust them.

        This is the test that makes the citation contract safe: a `clean_start`
        past the end of the text yields a chunk whose `text` does not match its
        offsets, and a citation pointing at it names the wrong words with no
        error anywhere (§9).
        """
        c = get_chunker("fixed", size=64, overlap=8)
        doc = make_document(PROSE)
        with self.assertRaises(ValueError):
            c.validate(doc, [Span(start=0, end=len(doc.text) + 100)])
        with self.assertRaises(ValueError):
            c.validate(doc, [Span(start=-1, end=10)])
        with self.assertRaises(ValueError):
            c.validate(doc, [Span(start=10, end=10)])

    def test_overlapping_spans_rejected_when_not_allowed(self) -> None:
        """`structural` forbids overlap, so out-of-order overlapping spans are a
        bug in the strategy and must be caught rather than indexed."""
        c = get_chunker("structural", size=1024, min_size=0)
        # Force the bad shape through the base class's validation.
        with self.assertRaises(ValueError):
            c.validate(
                make_document(PROSE * 5),
                [Span(start=0, end=10), Span(start=5, end=20)],
            )

    def test_overlap_allowed_for_fixed_and_recursive(self) -> None:
        for name in ("fixed", "recursive"):
            c = get_chunker(name, size=64, overlap=8)
            self.assertTrue(c.allows_overlap, name)
            c.validate(make_document(PROSE * 5), [Span(start=0, end=10), Span(start=5, end=20)])

    def test_overlap_forbidden_for_structural_and_semantic(self) -> None:
        self.assertFalse(get_chunker("structural").allows_overlap)
        self.assertFalse(get_chunker("semantic").allows_overlap)

    def test_structural_rejects_nonzero_overlap(self) -> None:
        """Raising rather than clamping.

        A silent clamp would let `chunk_overlap=64` look applied to a
        `structural` build while changing nothing — and the §6.5 index signature
        would record the overlap the operator asked for, not the one in force.
        """
        with self.assertRaises(ValueError):
            get_chunker("structural", size=512, overlap=64)
        self.assertEqual(get_chunker("structural", size=512, overlap=0).overlap, 0)

    def test_out_of_order_spans_rejected_even_with_overlap(self) -> None:
        """Overlap is permitted, but spans must still be ordered. Without the
        ordering check, an overlap of more than one chunk's length produces
        spans running backwards through the document."""
        c = get_chunker("fixed", size=64, overlap=8)
        with self.assertRaises(ValueError):
            c.validate(
                make_document(PROSE * 5),
                [Span(start=0, end=10), Span(start=40, end=50), Span(start=20, end=30)],
            )


if __name__ == "__main__":
    unittest.main()
