"""Core types, config, and the error envelope.

The properties tested here are the ones whose failure is silent: the two offset
spaces (§9), the frozen config that §15's config hash depends on, and the split
between index-shaping and query-shaping settings (§6.5) that decides whether a
re-index is appendable or must be refused with a 409.
"""

from __future__ import annotations

import json
import unittest

from tests.base import RagTestCase, make_document
from raglab.config import Profile, Settings, from_env
from raglab.errors import (
    STATUS_BY_CODE,
    STATE_TO_CODE,
    ComponentMismatch,
    ErrorCode,
    ModelUnavailable,
    RagError,
    envelope,
)
from raglab.types import Chunk, content_hash, count_jsonl, dumps_line, text_hash

TEXT = "Lagos State\nEducation\nTerm 1 tuition is N450,000.\nTerm 2 is due in March.\n"


class TestChunkOffsets(RagTestCase):
    """§9: verification in clean space, navigation in original space.

    Conflating the two produces citations that point at the wrong text and a
    recall@k that is undefined rather than merely wrong.
    """

    def setUp(self) -> None:
        self.chunk = Chunk(
            ordinal=7,
            doc_id="sha256:ab12",
            chunk_index=0,
            text="Term 1 tuition is N450,000.",
            clean_start=23,
            clean_end=51,
            orig_start=101,
            orig_end=129,
            page=4,
            heading_path=("Lagos State", "Education"),
        )

    def test_both_offset_pairs_are_kept(self) -> None:
        self.assertEqual(self.chunk.clean_start, 23)
        self.assertEqual(self.chunk.orig_start, 101)
        self.assertNotEqual(self.chunk.clean_start, self.chunk.orig_start)

    def test_original_offsets_may_be_absent(self) -> None:
        """Honest absence, not a guess.

        Where the cleaner rewrote text it cannot map, the original offset is
        `None` and the evidence panel falls back to page + breadcrumb + the
        retrieved clean text (§9).
        """
        c = Chunk(
            ordinal=0,
            doc_id="d",
            chunk_index=0,
            text="x",
            clean_start=0,
            clean_end=1,
            orig_start=None,
            orig_end=None,
        )
        self.assertIsNone(c.orig_start)
        self.assertEqual(c.clean_start, 0)

    def test_clean_text_resolves_through_the_document(self) -> None:
        doc = make_document(TEXT)
        needle = "Term 1 tuition is N450,000."
        start = TEXT.index(needle)
        self.assertEqual(doc.clean_text(start, start + len(needle)), needle)
        self.assertEqual(
            self.chunk.clean_text(doc),
            TEXT[self.chunk.clean_start : self.chunk.clean_end],
        )

    def test_page_lookup_is_original_space(self) -> None:
        from raglab.types import PageSpan

        doc = make_document(TEXT, pages=((0, 0, 20), (1, 20, 40), (2, 40, len(TEXT))))
        self.assertEqual(doc.page_for_offset(5), 0)
        self.assertEqual(doc.page_for_offset(25), 1)
        self.assertEqual(doc.page_for_offset(999), None)

    def test_heading_path_at_offset(self) -> None:
        doc = make_document(TEXT, heading_path_at=((0, "Lagos State"), (12, "Education")))
        self.assertEqual(doc.heading_path_at_offset(0), ["Lagos State"])
        self.assertEqual(doc.heading_path_at_offset(30), ["Lagos State", "Education"])
        self.assertEqual(doc.heading_path_at_offset(5), ["Lagos State"])

    def test_flat_document_has_empty_heading_path(self) -> None:
        """`[]` is the honest answer for flat prose, not a placeholder."""
        self.assertEqual(make_document(TEXT).heading_path_at_offset(5), [])


class TestChunkEmbedText(RagTestCase):
    """§9: the breadcrumb is prepended to the *embedded* text.

    A chunk reading only "Term 1 is N450,000" is unretrievable; one carrying its
    section path is not. Costs nothing at inference time.
    """

    def test_breadcrumb_is_prepended(self) -> None:
        c = Chunk(
            ordinal=0,
            doc_id="d",
            chunk_index=0,
            text="Term 1 tuition is N450,000.",
            clean_start=0,
            clean_end=1,
            heading_path=("Lagos State", "Education"),
        )
        self.assertTrue(c.embed_text.startswith("Lagos State > Education"))
        self.assertIn(c.text, c.embed_text)

    def test_no_heading_means_no_prefix(self) -> None:
        c = Chunk(
            ordinal=0, doc_id="d", chunk_index=0, text="body", clean_start=0, clean_end=4
        )
        self.assertEqual(c.embed_text, "body")

    def test_embed_text_is_not_what_is_cited(self) -> None:
        """The prefix exists for the embedder. A citation must quote the text the
        reader can find in the document, not the synthetic prefix."""
        c = Chunk(
            ordinal=0,
            doc_id="d",
            chunk_index=0,
            text="body",
            clean_start=0,
            clean_end=4,
            heading_path=("A", "B"),
        )
        self.assertNotEqual(c.embed_text, c.text)


class TestChunkSerialisation(RagTestCase):
    def test_round_trips_through_json(self) -> None:
        c = Chunk(
            ordinal=3,
            doc_id="sha256:ab12",
            chunk_index=1,
            text="body",
            clean_start=0,
            clean_end=4,
            heading_path=("A", "B"),
            token_count=2,
            live=False,
        )
        again = Chunk.from_dict(json.loads(dumps_line(c.to_dict())))
        self.assertEqual(c, again)

    def test_live_flag_survives(self) -> None:
        """§6.4: deletion is logical. Losing `live` would resurrect a deleted
        document's chunks into every query."""
        c = Chunk(
            ordinal=0, doc_id="d", chunk_index=0, text="x", clean_start=0, clean_end=1,
            live=False,
        )
        self.assertFalse(Chunk.from_dict(c.to_dict()).live)

    def test_span_key_identity(self) -> None:
        c = Chunk(
            ordinal=9, doc_id="d", chunk_index=0, text="x", clean_start=4, clean_end=9
        )
        self.assertEqual(c.span_key, ("d", 4, 9))


class TestHashes(RagTestCase):
    def test_content_hash_is_prefixed(self) -> None:
        h = content_hash(b"hello")
        self.assertTrue(h.startswith("sha256:"))
        self.assertEqual(h, content_hash(b"hello"))
        self.assertNotEqual(h, content_hash(b"hellp"))

    def test_text_hash_is_stable_and_content_addressed(self) -> None:
        """§20: the embedding cache is keyed by chunk-text hash, so it is shared
        across chunker settings without leaking between sessions."""
        self.assertEqual(text_hash("abc"), text_hash("abc"))
        self.assertNotEqual(text_hash("abc"), text_hash("abd"))

    def test_count_jsonl(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "chunks.jsonl"
            p.write_text(dumps_line({"a": 1}) + "\n" + dumps_line({"a": 2}) + "\n")
            self.assertEqual(count_jsonl(p), 2)
            p.write_text("")
            self.assertEqual(count_jsonl(p), 0)
            self.assertEqual(count_jsonl(Path(d) / "missing.jsonl"), 0)


class TestSettings(RagTestCase):
    def test_default_profile_is_mock(self) -> None:
        """§8 rule 1: a run must never require a GPU download to be testable."""
        self.assertEqual(Settings().profile, Profile.MOCK)

    def test_is_frozen(self) -> None:
        """The retrieval config hash (§15) is computed from this object, so
        mutability would let a run record a config it did not use."""
        s = Settings()
        with self.assertRaises(Exception):
            s.profile = Profile.LOCAL  # type: ignore[misc]

    def test_with_overrides_returns_a_copy(self) -> None:
        s = Settings()
        t = s.with_overrides(chunker="structural", chunk_size=1024)
        self.assertEqual(s.chunker, "fixed")
        self.assertEqual(t.chunker, "structural")

    def test_to_dict_never_carries_the_token(self) -> None:
        """§16: the runtime token is never written to disk, logged, or persisted."""
        s = Settings(runtime_token="sk-supersecret12345678")
        self.assertNotIn("runtime_token", s.to_dict())
        self.assertNotIn("sk-supersecret", repr(s))

    def test_index_signature_lists_only_stored_data_components(self) -> None:
        """§6.5: settings that shape stored data are pinned to the index."""
        sig = Settings().index_signature
        self.assertEqual(
            set(sig),
            {"chunker", "embed_model", "embed_dim", "vector_dtype"},
        )

    def test_retrieval_settings_are_absent_from_the_index_signature(self) -> None:
        """The distinction that decides append vs 409: changing top_k or a
        transform flag does not touch stored data."""
        a = Settings(retrieve_top_k=40).index_signature
        b = Settings(retrieve_top_k=100).index_signature
        self.assertEqual(a, b)
        c = Settings(chunker="structural").index_signature
        self.assertNotEqual(a, c)

    def test_index_signature_changes_with_dtype(self) -> None:
        """§5.5 may switch to fp32, which changes every stored row."""
        self.assertNotEqual(
            Settings(vector_dtype="float16").index_signature,
            Settings(vector_dtype="float32").index_signature,
        )

    def test_retrieval_config_hash_is_stable_and_sensitive(self) -> None:
        base = Settings()
        self.assertEqual(base.retrieval_config_hash, Settings().retrieval_config_hash)
        self.assertNotEqual(
            base.retrieval_config_hash, Settings(retrieve_top_k=41).retrieval_config_hash
        )
        self.assertNotEqual(
            base.retrieval_config_hash, Settings(gate_enabled=False).retrieval_config_hash
        )

    def test_retrieval_signature_records_rerank_window(self) -> None:
        """§10/§15: a 2048-token row must never be read against a 512-token one."""
        self.assertIn("rerank_window", Settings().retrieval_signature)
        self.assertEqual(Settings().rerank_window_mode, "query_aware_head")

    def test_vector_row_size_matches_the_plan_arithmetic(self) -> None:
        """§6.6 quotes 2,048 bytes per fp16 1024-dim row. Computed, not hardcoded."""
        self.assertEqual(Settings(vector_dtype="float16").vector_nbytes_per_row, 2048)
        self.assertEqual(Settings(vector_dtype="float32").vector_nbytes_per_row, 4096)

    def test_bad_vector_dtype_raises(self) -> None:
        for bad in ("f16", "float16 ", "bfloat16"):
            with self.assertRaises(RagError):
                Settings(vector_dtype=bad).validated_vector_dtype()

    def test_defaults_are_the_plans_defaults(self) -> None:
        """These numbers appear throughout the plan; a silent drift here changes
        what an ablation row means."""
        s = Settings()
        self.assertEqual(s.chunker, "fixed")
        self.assertEqual((s.chunk_size, s.chunk_overlap), (512, 64))
        self.assertEqual((s.retrieve_top_k, s.context_top_n), (40, 6))
        self.assertEqual(s.rerank_top_n, 20)
        self.assertEqual(s.rerank_max_len, 512)
        self.assertEqual(s.gen_ctx, 8192)
        self.assertFalse(s.gen_think)
        self.assertTrue(s.gate_enabled, "the gate is a correctness property")
        self.assertEqual(s.rrf_k, 60)
        self.assertEqual(s.dense_gpu_threshold, 50_000)

    def test_transforms_are_off_by_default(self) -> None:
        """§12: each transform earns its place through an ablation delta."""
        s = Settings()
        self.assertFalse(s.transform_rewrite or s.transform_hyde)
        self.assertFalse(s.transform_decompose or s.transform_multiquery)

    def test_binds_loopback_by_default(self) -> None:
        """§16: binding 127.0.0.1 *is* the security boundary."""
        self.assertEqual(Settings().host, "127.0.0.1")


class TestFromEnv(RagTestCase):
    def test_reads_prefixed_vars(self) -> None:
        s = from_env(
            {
                "RAGLAB_PROFILE": "tiny",
                "RAGLAB_RETRIEVE_TOP_K": "25",
                "RAGLAB_GATE_ENABLED": "false",
                "RAGLAB_CORS_ORIGINS": "https://a.example, https://b.example",
                "UNRELATED": "ignored",
            }
        )
        self.assertEqual(s.profile, Profile.TINY)
        self.assertEqual(s.retrieve_top_k, 25)
        self.assertFalse(s.gate_enabled)
        self.assertEqual(s.cors_origins, ("https://a.example", "https://b.example"))

    def test_ignores_ignored_prefix(self) -> None:
        self.assertEqual(from_env({"RAGLAB_RETRIEVE_TOP_K": "10"}).retrieve_top_k, 10)

    def test_runtime_token_is_not_read_from_env(self) -> None:
        """§16: printed once to the operator, never persisted. Reading it from the
        environment would make `.env` a place the secret lives."""
        self.assertEqual(from_env({"RAGLAB_RUNTIME_TOKEN": "sk-leaked12345678"}).runtime_token, "")

    def test_unknown_prefixed_var_is_ignored(self) -> None:
        """Silently accepted rather than fatal.

        `RAGLAB_LOG_LEVEL` and future additions may be read by another component
        (the notebook, a shell script) with no `Settings` field behind them. A
        typo'd variable would then break a session at import time, which is a
        worse failure than one unused value.
        """
        s = from_env({"RAGLAB_STORE_TIER": "durable", "RAGLAB_PROFILE": "mock"})
        self.assertEqual(s.profile, Profile.MOCK)
        self.assertEqual(s.log_level, "INFO")

    def test_unknown_override_raises(self) -> None:
        """An explicit `from_env(..., not_a_setting=1)` is a programming error,
        and unlike an env var there is no benign explanation for it."""
        with self.assertRaises(ValueError):
            from_env({}, not_a_setting=1)  # type: ignore[call-arg]

    def test_overrides_win(self) -> None:
        s = from_env({"RAGLAB_RETRIEVE_TOP_K": "10"}, retrieve_top_k=99)
        self.assertEqual(s.retrieve_top_k, 99)


class TestErrors(RagTestCase):
    def test_envelope_shape(self) -> None:
        """§16: one envelope everywhere; clients branch on `code`."""
        e = RagError("boom", code=ErrorCode.RETRIEVAL_ERROR, session_id="s1")
        body = e.to_envelope()
        self.assertEqual(set(body), {"error"})
        self.assertEqual(body["error"]["code"], "retrieval_error")
        self.assertEqual(body["error"]["detail"], "boom")
        self.assertEqual(body["error"]["session_id"], "s1")

    def test_status_codes_are_mapped(self) -> None:
        self.assertEqual(STATUS_BY_CODE[ErrorCode.UNAUTHORIZED], 401)
        self.assertEqual(STATUS_BY_CODE[ErrorCode.SESSION_NOT_FOUND], 404)
        # 409, not 400: the request is well formed, the server's state conflicts.
        self.assertEqual(STATUS_BY_CODE[ErrorCode.INDEX_COMPONENT_MISMATCH], 409)
        self.assertEqual(STATUS_BY_CODE[ErrorCode.INDEX_COMPONENT_MISMATCH], 409)

    def test_every_ui_state_maps_to_a_code(self) -> None:
        """§17.4's state table must be complete, or a UI state branches on
        something that does not exist."""
        for state in (
            "index_component_mismatch",
            "token_invalid",
            "no_evidence",
            "citation_broken",
            "retrieval_error",
            "generation_error",
            "session_deleted",
            "session_title_empty",
            "indexing",
            "index_empty",
            "offline",
        ):
            self.assertIn(state, STATE_TO_CODE, state)

    def test_detail_is_never_the_branch_key(self) -> None:
        """Two errors with different codes may share wording; a client that
        matched on `detail` would break the day someone reworded it."""
        a = RagError("not found", code=ErrorCode.SESSION_NOT_FOUND)
        b = RagError("not found", code=ErrorCode.DOCUMENT_NOT_FOUND)
        self.assertNotEqual(a.to_envelope()["error"]["code"], b.to_envelope()["error"]["code"])

    def test_component_mismatch_names_the_differing_field(self) -> None:
        e = ComponentMismatch(
            {"chunker": ("fixed/1.0.0", "structural/1.0.0")}, session_id="s1"
        )
        body = e.to_envelope()["error"]
        self.assertEqual(body["code"], "index_component_mismatch")
        self.assertIn("chunker", body["detail"])
        self.assertIn("fixed/1.0.0", body["detail"])
        self.assertEqual(body["rebuild"], "POST ?rebuild=true")

    def test_model_unavailable_names_profile_and_model(self) -> None:
        """§8 rule 2: a visible, recoverable error, never a silent swap."""
        e = ModelUnavailable("needs sentence_transformers", profile="local", model="bge-m3")
        body = e.to_envelope()["error"]
        self.assertEqual(body["profile"], "local")
        self.assertEqual(body["model"], "bge-m3")

    def test_envelope_helper(self) -> None:
        body = envelope(ErrorCode.NO_EVIDENCE, "nothing retrieved")
        self.assertEqual(body["error"]["code"], "no_evidence")


if __name__ == "__main__":
    unittest.main()
