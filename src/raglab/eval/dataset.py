"""Eval question sets, scoped to a session (§15).

Two rules from the plan that this module enforces rather than documents:

1. **Evaluation is per session.** Every metric is only meaningful against a
   specific corpus, so a question set lives inside its session's `eval/`
   directory and travels with the corpus it describes. `EvalSet.session_id` is
   required and `assert_comparable` refuses to mix two.
2. **`gold_spans` are clean-space offsets** (§9, §15). Recall@k compares gold
   against what retrieval actually ranked, and retrieval ranks clean-space
   offsets. A gold span in original-file coordinates is in a different space and
   the comparison is undefined rather than merely wrong.

Authoring gold spans by hand means counting characters, which nobody does
correctly twice. `eval/author.py` (Phase 2) handles the coordinates from a
chunk click; this module only stores and validates what it is given.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from raglab.errors import RagError

#: §15: 45 answerable across single-hop, multi-hop and paraphrase, plus 15
#: unanswerable. The ratio is 3:1 and that is deliberate — a set dominated by
#: unanswerable questions scores abstention beautifully and retrieval terribly.
RECOMMENDED_TOTAL = 60
RECOMMENDED_ANSWERABLE = 45
RECOMMENDED_UNANSWERABLE = 15

VALID_TAGS: frozenset[str] = frozenset(
    {
        "factual",
        "single-hop",
        "multi-hop",
        "paraphrased",
        "exact-identifier",
        "cross-document",
        "numeric",
        "temporal",
        "negation",
        "unanswerable",
    }
)
"""Tags name the retrieval property a question probes.

§15: a single blended recall number hides which case is failing. `exact-identifier`
is the tag to check first when the hybrid delta is disappointing, and
`paraphrased` is the one to check when the lexical stage is winning alone.
"""


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    """One question with its gold evidence."""

    qid: str
    question: str
    answerable: bool = True
    gold_docs: tuple[str, ...] = ()
    gold_spans: tuple[tuple[str, int, int], ...] = ()
    """`(doc_id, clean_start, clean_end)` triples. Clean space (§9)."""

    tags: tuple[str, ...] = ()
    reference_answer: str | None = None
    """Optional. Used by the judge for relevancy; never used by retrieval
    metrics, which would leak the answer text into the retrieval signal."""

    def validate(self) -> None:
        """Reject a question that would produce an undefined metric."""
        if not self.qid.strip():
            msg = "qid must be non-empty"
            raise RagError(msg)
        if not self.question.strip():
            msg = f"question {self.qid}: question text must be non-empty"
            raise RagError(msg)
        if self.answerable and not self.gold_spans and not self.gold_docs:
            msg = (
                f"question {self.qid}: answerable=True requires gold_spans or "
                "gold_docs; an answerable question with no gold evidence scores "
                "zero everywhere and drags the corpus average"
            )
            raise RagError(msg)
        if not self.answerable and self.gold_spans:
            msg = (
                f"question {self.qid}: answerable=False must not carry gold_spans; "
                "an unanswerable question with gold evidence is a labelling error, "
                "not a hard case"
            )
            raise RagError(msg)
        for doc_id, start, end in self.gold_spans:
            if start < 0 or end <= start:
                msg = (
                    f"question {self.qid}: gold span [{start},{end}) in {doc_id} "
                    "is empty or negative; spans are clean-space character offsets"
                )
                raise RagError(msg)
        unknown = set(self.tags) - VALID_TAGS
        if unknown:
            msg = f"question {self.qid}: unknown tags {sorted(unknown)}"
            raise RagError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "answerable": self.answerable,
            "gold_docs": list(self.gold_docs),
            "gold_spans": [list(s) for s in self.gold_spans],
            "tags": list(self.tags),
            "reference_answer": self.reference_answer,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalQuestion:
        return cls(
            qid=str(raw["qid"]),
            question=str(raw["question"]),
            answerable=bool(raw.get("answerable", True)),
            gold_docs=tuple(raw.get("gold_docs", ())),
            gold_spans=tuple(
                (str(s[0]), int(s[1]), int(s[2])) for s in raw.get("gold_spans", [])
            ),
            tags=tuple(raw.get("tags", ())),
            reference_answer=raw.get("reference_answer"),
        )


@dataclass(frozen=True, slots=True)
class EvalSet:
    """A versioned question set belonging to exactly one session."""

    session_id: str
    questions: tuple[EvalQuestion, ...]
    version: int = 1
    name: str = "default"

    def validate(self) -> None:
        if not self.session_id:
            msg = "EvalSet.session_id is required; results from two sessions are never tabulated together (§15)"
            raise RagError(msg)
        seen: set[str] = set()
        for q in self.questions:
            q.validate()
            if q.qid in seen:
                msg = f"duplicate qid {q.qid!r}; qid is the join key for every result row"
                raise RagError(msg)
            seen.add(q.qid)

    # -- composition ---------------------------------------------------

    @property
    def answerable(self) -> tuple[EvalQuestion, ...]:
        return tuple(q for q in self.questions if q.answerable)

    @property
    def unanswerable(self) -> tuple[EvalQuestion, ...]:
        return tuple(q for q in self.questions if not q.answerable)

    def by_tag(self, tag: str) -> tuple[EvalQuestion, ...]:
        return tuple(q for q in self.questions if tag in q.tags)

    def get(self, qid: str) -> EvalQuestion | None:
        for q in self.questions:
            if q.qid == qid:
                return q
        return None

    def composition(self) -> dict[str, Any]:
        """Counts the report renders, and the §15 gate checks."""
        tags: dict[str, int] = {}
        for q in self.questions:
            for t in q.tags:
                tags[t] = tags.get(t, 0) + 1
        return {
            "total": len(self.questions),
            "answerable": len(self.answerable),
            "unanswerable": len(self.unanswerable),
            "tags": dict(sorted(tags.items())),
        }

    def meets_recommended_composition(self) -> bool:
        """Whether the 45/15 split of §15 is satisfied.

        Advisory, not enforced. A 10-question smoke set is legitimate; what is
        *not* legitimate is scoring abstention on a set with no unanswerable
        questions, which the generation metrics refuse outright.
        """
        return (
            len(self.answerable) >= RECOMMENDED_ANSWERABLE
            and len(self.unanswerable) >= RECOMMENDED_UNANSWERABLE
        )

    def subset(self, qids: Iterable[str]) -> EvalSet:
        wanted = set(qids)
        return EvalSet(
            session_id=self.session_id,
            questions=tuple(q for q in self.questions if q.qid in wanted),
            version=self.version,
            name=f"{self.name}.subset",
        )

    def with_tags(self, *tags: str) -> EvalSet:
        """Filter to questions carrying any of `tags`. The generation-matrix
        subsetting mechanism (§15): a smaller, focused question set."""
        return EvalSet(
            session_id=self.session_id,
            questions=tuple(q for q in self.questions if set(q.tags) & set(tags)),
            version=self.version,
            name=f"{self.name}.tagged",
        )

    # -- serialisation -------------------------------------------------

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(q.to_dict(), sort_keys=True) for q in self.questions)

    def write(self, path: str | Path) -> Path:
        """Write to `path`, creating parents. Returns the path.

        §6.2 write-through: a question set is expensive human labour and the
        session can end without warning, so it is on disk the moment it exists.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_jsonl() + "\n", encoding="utf-8")
        return p

    @classmethod
    def from_jsonl(cls, text: str, *, session_id: str) -> EvalSet:
        qs = tuple(
            EvalQuestion.from_dict(json.loads(line))
            for line in text.splitlines()
            if line.strip()
        )
        es = cls(session_id=session_id, questions=qs)
        es.validate()
        return es

    @classmethod
    def load(cls, path: str | Path, *, session_id: str) -> EvalSet:
        p = Path(path)
        if not p.exists():
            msg = f"eval set not found: {p}"
            raise RagError(msg)
        return cls.from_jsonl(p.read_text(encoding="utf-8"), session_id=session_id)

    def __iter__(self) -> Iterator[EvalQuestion]:
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)


@dataclass(slots=True)
class RunRecord:
    """One ablation row's provenance.

    §15 determinism: seeds, model revisions, library versions, the full
    retrieval config hash, `rerank_window`, and the live/physical chunk counts
    (§6.4). Without these a 3-point "improvement" is uninterpretable — you cannot
    tell a real delta from a changed setting.
    """

    run_id: str
    session_id: str
    label: str
    """Human label for the table row, e.g. "512/64 fixed hybrid+rerank"."""

    matrix: str
    """`retrieval` or `generation` (§15: two matrices, because one grid costs more
    than the quota)."""

    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    chunker: str = ""
    embed_model: str = ""
    embed_model_revision: str = ""
    rerank_model: str = ""
    rerank_model_revision: str = ""
    judge_model: str = ""
    rerank_window: str = ""
    """Recorded per row so a 2048-token chunk is never read as comparable to a
    512-token one (§10)."""

    chunk_count: int = 0
    """Live chunks (§6.4)."""

    chunk_rows: int = 0
    """Physical rows. Divergence from `chunk_count` is reported, not hidden."""

    seed: int = 0
    library_versions: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "label": self.label,
            "matrix": self.matrix,
            "config": self.config,
            "config_hash": self.config_hash,
            "chunker": self.chunker,
            "embed_model": self.embed_model,
            "embed_model_revision": self.embed_model_revision,
            "rerank_model": self.rerank_model,
            "rerank_model_revision": self.rerank_model_revision,
            "judge_model": self.judge_model,
            "rerank_window": self.rerank_window,
            "chunk_count": self.chunk_count,
            "chunk_rows": self.chunk_rows,
            "seed": self.seed,
            "library_versions": self.library_versions,
            "metrics": self.metrics,
            "per_question": self.per_question,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def write(self, directory: str | Path) -> Path:
        p = Path(directory) / f"{self.run_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json() + "\n", encoding="utf-8")
        return p


def assert_comparable(a: RunRecord | EvalSet, b: RunRecord | EvalSet) -> None:
    """Refuse to tabulate two runs that cannot be compared (§15).

    §15's rule: never compare numbers across different corpora, embedders, or
    judges. This makes the violation an exception rather than a footnote.
    """
    if isinstance(a, EvalSet) and isinstance(b, EvalSet):
        if a.session_id != b.session_id:
            msg = (
                f"cannot combine eval sets from sessions {a.session_id!r} and "
                f"{b.session_id!r}: evaluation is per session because every metric "
                "is only meaningful against a specific corpus (§15)"
            )
            raise RagError(msg)
        return

    if isinstance(a, RunRecord) and isinstance(b, RunRecord):
        for attr in ("session_id", "embed_model", "embed_model_revision", "judge_model"):
            av, bv = getattr(a, attr), getattr(b, attr)
            if av != bv:
                msg = (
                    f"runs {a.run_id!r} and {b.run_id!r} differ in {attr} "
                    f"({av!r} vs {bv!r}); §15 forbids comparing across a corpus, "
                    "embedder, or judge change"
                )
                raise RagError(msg)


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """Read a JSONL file into questions, without session validation.

    For the committed seed set (§15), which describes a corpus the operator has
    not ingested yet and therefore has no session id for.
    """
    p = Path(path)
    if not p.exists():
        msg = f"question file not found: {p}"
        raise RagError(msg)
    out: list[EvalQuestion] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(EvalQuestion.from_dict(json.loads(line)))
    return out


def dump_questions(questions: Sequence[EvalQuestion]) -> str:
    return "\n".join(json.dumps(q.to_dict(), sort_keys=True) for q in questions)
