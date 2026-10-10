"""Error codes and the single error envelope.

§16: the §17.4 state table has fifteen distinct states and every one of them
branches on machine-readable data, so the envelope shape is specified once here
rather than left to `docs/api.md`.

Clients branch on `code`, never on `detail`. `detail` is for humans and may be
reworded freely — which is exactly why it cannot be a branch key.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Closed set. Every §17.4 UI state maps to exactly one of these."""

    # --- auth / session ---
    UNAUTHORIZED = "unauthorized"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_TITLE_REQUIRED = "session_title_required"
    SESSION_ID_INVALID = "session_id_invalid"

    # --- corpus ---
    DOCUMENT_NOT_FOUND = "document_not_found"
    UNSUPPORTED_FORMAT = "unsupported_format"
    EMPTY_UPLOAD = "empty_upload"
    PARSE_FAILED = "parse_failed"

    # --- index ---
    INDEX_COMPONENT_MISMATCH = "index_component_mismatch"
    INDEX_BUSY = "index_busy"
    INDEX_NOT_BUILT = "index_not_built"

    # --- query pipeline ---
    QUERY_EMPTY = "query_empty"
    NO_EVIDENCE = "no_evidence"
    RETRIEVAL_ERROR = "retrieval_error"
    GENERATION_ERROR = "generation_error"
    GATE_ERROR = "gate_error"

    # --- citations ---
    CITATION_UNRESOLVABLE = "citation_unresolvable"
    CITATION_OUT_OF_RANGE = "citation_out_of_range"

    # --- infrastructure ---
    MODEL_UNAVAILABLE = "model_unavailable"
    OLLAMA_UNREACHABLE = "ollama_unreachable"
    DEVICE_UNAVAILABLE = "device_unavailable"
    INTERNAL = "internal"


#: HTTP status for each code. Kept beside the code so a route never invents its
#: own mapping and the UI can rely on the pair.
STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.SESSION_NOT_FOUND: 404,
    ErrorCode.SESSION_TITLE_REQUIRED: 422,
    ErrorCode.SESSION_ID_INVALID: 400,
    ErrorCode.DOCUMENT_NOT_FOUND: 404,
    ErrorCode.UNSUPPORTED_FORMAT: 415,
    ErrorCode.EMPTY_UPLOAD: 422,
    ErrorCode.PARSE_FAILED: 422,
    # 409, not 400: the request is well-formed, the server's state conflicts
    # with it. Detect-then-append would leave a mixed index with the §6.4
    # invariant still green, so this must not be a soft failure (§6.5).
    ErrorCode.INDEX_COMPONENT_MISMATCH: 409,
    ErrorCode.INDEX_BUSY: 409,
    ErrorCode.INDEX_NOT_BUILT: 409,
    ErrorCode.QUERY_EMPTY: 422,
    ErrorCode.NO_EVIDENCE: 200,
    ErrorCode.RETRIEVAL_ERROR: 500,
    ErrorCode.GENERATION_ERROR: 502,
    ErrorCode.GATE_ERROR: 502,
    ErrorCode.CITATION_UNRESOLVABLE: 200,
    ErrorCode.CITATION_OUT_OF_RANGE: 200,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.OLLAMA_UNREACHABLE: 503,
    ErrorCode.DEVICE_UNAVAILABLE: 503,
    ErrorCode.INTERNAL: 500,
}

#: §17.4 state -> code. The UI states are literally spelled out so a missing
#: mapping is a startup assertion failure rather than a runtime guess.
STATE_TO_CODE: dict[str, ErrorCode] = {
    "index_component_mismatch": ErrorCode.INDEX_COMPONENT_MISMATCH,
    "token_invalid": ErrorCode.UNAUTHORIZED,
    "no_evidence": ErrorCode.NO_EVIDENCE,
    "citation_broken": ErrorCode.CITATION_UNRESOLVABLE,
    "retrieval_error": ErrorCode.RETRIEVAL_ERROR,
    "generation_error": ErrorCode.GENERATION_ERROR,
    "session_deleted": ErrorCode.SESSION_NOT_FOUND,
    "session_title_empty": ErrorCode.SESSION_TITLE_REQUIRED,
    "indexing": ErrorCode.INDEX_BUSY,
    "index_empty": ErrorCode.INDEX_NOT_BUILT,
    "offline": ErrorCode.GENERATION_ERROR,
}


class RagError(Exception):
    """Base for every error this system raises deliberately.

    Carries a machine code and a human detail. `to_envelope()` produces the one
    wire shape (§16).
    """

    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(
        self,
        detail: str,
        *,
        code: ErrorCode | None = None,
        session_id: str | None = None,
        doc_id: str | None = None,
        **extra: object,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        if code is not None:
            self.code = code
        self.session_id = session_id
        self.doc_id = doc_id
        self.extra: dict[str, object] = extra

    @property
    def status_code(self) -> int:
        return STATUS_BY_CODE.get(self.code, 500)

    def to_envelope(self) -> dict[str, object]:
        """The single wire shape. Every error response in the system is this."""
        body: dict[str, object] = {"code": str(self.code), "detail": self.detail}
        if self.session_id:
            body["session_id"] = self.session_id
        if self.doc_id:
            body["doc_id"] = self.doc_id
        if self.extra:
            body.update(self.extra)
        return {"error": body}


class ComponentMismatch(RagError):
    """Stored data and requested components disagree (§6.5).

    Raised *instead of* appending, because appending into a matrix built by a
    different chunker produces two chunking regimes in one index: the §6.4
    invariant passes, recall degrades, and nothing errors anywhere.
    """

    def __init__(
        self,
        differing: dict[str, str],
        *,
        session_id: str | None = None,
        rebuild_hint: bool = True,
    ) -> None:
        detail = "; ".join(f"{k} {was} != {now}" for k, (was, now) in differing.items())
        super().__init__(
            detail,
            code=ErrorCode.INDEX_COMPONENT_MISMATCH,
            session_id=session_id,
            differing=differing,
            **({"rebuild": "POST ?rebuild=true"} if rebuild_hint else {}),
        )
        self.differing = differing


class ModelUnavailable(RagError):
    """A profile's model could not be loaded.

    §8 rule 2: failures degrade to a *visible, recoverable* error, never a
    silent model swap. A silent substitution is how eval numbers end up meaning
    nothing.
    """

    def __init__(self, detail: str, *, profile: str, model: str) -> None:
        super().__init__(
            detail,
            code=ErrorCode.MODEL_UNAVAILABLE,
            profile=profile,
            model=model,
        )
        self.profile = profile
        self.model = model


def envelope(code: ErrorCode, detail: str, **extra: object) -> dict[str, object]:
    """Build an envelope without raising. For non-exceptional paths."""
    body: dict[str, object] = {"code": str(code), "detail": detail}
    body.update(extra)
    return {"error": body}
