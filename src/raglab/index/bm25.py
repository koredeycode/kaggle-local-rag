"""BM25 (Okapi), in-repo.

§11 specifies roughly 60 lines with tokenisation, document frequencies, and
length normalisation, chosen deliberately: it is core RAG curriculum, it adds no
dependency, and `docs/retrieval.md` teaches it. `rank_bm25` remains an
acceptable substitute if it ever saves time; the ablation table does not care
which implementation produced the numbers, only that they are correct.

Two properties this implementation must have, because they are what make hybrid
retrieval worth the extra work at all:

- **Term frequency saturates.** The tenth occurrence of "fee" adds far less than
  the first (`k1`), so a document cannot win by repeating one word.
- **Long documents are penalised.** `b` normalises for length, so a 5000-token
  chunk does not beat a 300-token chunk purely by containing more.

The index is derived, never persisted (§6.5): a persisted copy can only ever be
stale relative to the live chunk set, and stale is worse than slow to rebuild.

Deliberate non-goals: no stemming, no stopword list beyond a tiny one, no
phrase queries. Those are tuning knobs with real costs and the plan does not
claim them. Adding stemming silently changes what the lexical stage measures,
and an unrecorded stemming change invalidates the hybrid-vs-dense delta.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Deliberately tiny. Not a linguistic stopword list — just the terms that
#: appear in every document in a typical corpus and carry no discriminative
#: signal. A full list would be a large table whose only effect is to make
#: BM25 numbers incomparable to any other implementation.
_STOPWORDS = frozenset({"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                        "in", "is", "it", "of", "on", "or", "that", "the", "this",
                        "to", "was", "were", "will", "with"})

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Lowercase word tokens.

    `\w+` with `re.UNICODE` rather than an ASCII pattern, so Yoruba or Hausa
    text tokenises instead of vanishing. §24 Q3 takes multilingual support
    seriously enough that the test stub should not quietly assume English.
    """
    toks = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        return [t for t in toks if t not in _STOPWORDS]
    return toks


@dataclass(frozen=True, slots=True)
class BM25Config:
    k1: float = DEFAULT_K1
    """Term-frequency saturation. Higher = saturation happens later."""

    b: float = DEFAULT_B
    """Length normalisation. 0 = off, 1 = full."""


@dataclass(slots=True)
class _Doc:
    """Per-document posting data. Counted once at build, reused per query."""

    length: int
    term_freqs: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class BM25Index:
    """Okapi BM25 over live chunks, keyed by row ordinal.

    Rows are addressed by `ordinal`, which is session-global and equals the row
    index in `vectors.npy` (§6.4). Passing ordinals rather than positions means
    a caller cannot accidentally score row 7 of a stale list against row 7 of a
    current one.
    """

    config: BM25Config = field(default_factory=BM25Config)
    _docs: dict[int, _Doc] = field(default_factory=dict)
    _postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    _df: dict[str, int] = field(default_factory=dict)
    _total_len: int = 0
    _idf_cache: dict[str, float] = field(default_factory=dict)

    # -- building ------------------------------------------------------

    def add(self, ordinal: int, text: str, *, weights: Mapping[str, float] | None = None) -> None:
        """Add or replace one row's posting data.

        `weights` lets a caller supply precomputed term frequencies (§5.4 sparse
        weights from `bge-m3`). Supplying them changes what the lexical stage
        measures, so it is an explicit argument rather than a silent path.
        """
        if ordinal in self._docs:
            self.remove(ordinal)
        toks = tokenize(text)
        freqs: dict[str, int] = defaultdict(int)
        for t in toks:
            freqs[t] += 1
        if weights:
            for t, w in weights.items():
                if w > 0:
                    freqs[t] = int(w)
        doc = _Doc(length=max(1, len(toks)), term_freqs=dict(freqs))
        self._docs[ordinal] = doc
        self._total_len += doc.length
        for term in doc.term_freqs:
            self._df[term] = self._df.get(term, 0) + 1
            self._postings.setdefault(term, []).append((ordinal, doc.term_freqs[term]))
        self._idf_cache.clear()

    def add_many(self, rows: Iterable[tuple[int, str]]) -> None:
        for ordinal, text in rows:
            self.add(ordinal, text)

    def remove(self, ordinal: int) -> None:
        """Drop a row from the postings.

        §6.4: deletion is logical for the vector file, but BM25 filters to live
        rows before building postings precisely because this index is derived
        and a dead row in it would be returned by every query forever.
        """
        doc = self._docs.pop(ordinal, None)
        if doc is None:
            return
        self._total_len -= doc.length
        for term in doc.term_freqs:
            lst = self._postings.get(term)
            if lst is None:
                continue
            self._postings[term] = [(o, tf) for o, tf in lst if o != ordinal]
            self._df[term] = self._df.get(term, 0) - 1
            if self._df[term] <= 0:
                del self._df[term]
                del self._postings[term]
        self._idf_cache.clear()

    # -- querying ------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._docs)

    @property
    def avg_length(self) -> float:
        return self._total_len / len(self._docs) if self._docs else 0.0

    def _idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF with the +1 guard.

        The guard matters: with an unguarded formula a term in more than half the
        documents gets a *negative* IDF, so a document containing it is scored
        down for containing it. That is a real bug that shows up on small test
        corpora and not on large ones.
        """
        cached = self._idf_cache.get(term)
        if cached is not None:
            return cached
        n = len(self._docs)
        df = self._df.get(term, 0)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self._idf_cache[term] = idf
        return idf

    def score(self, query: str) -> dict[int, float]:
        """Score every row matching any query term. Rows with no match are absent.

        Absent rather than 0.0, because a zero is indistinguishable from a real
        score when ranking is done downstream, and because it keeps the output
        proportional to the query rather than to the corpus.
        """
        if not self._docs:
            return {}
        q_terms = tokenize(query)
        if not q_terms:
            return {}
        avg = self.avg_length
        k1, b = self.config.k1, self.config.b
        scores: dict[int, float] = defaultdict(float)
        for term in q_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for ordinal, tf in postings:
                doc = self._docs.get(ordinal)
                if doc is None:
                    continue
                norm = 1.0 - b + b * (doc.length / avg) if avg > 0 else 1.0
                scores[ordinal] += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)
        return dict(scores)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Top `top_k` as `(ordinal, score)`, best first.

        Ties break on ascending ordinal so results are deterministic — a stable
        order is what makes an ablation table diffable.
        """
        scores = self.score(query)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

    def terms(self) -> list[str]:
        """Distinct indexed terms. For diagnostics and tests."""
        return sorted(self._postings)

    def document_frequency(self, term: str) -> int:
        return self._df.get(term, 0)

    def stats(self) -> dict[str, float]:
        return {
            "documents": float(len(self._docs)),
            "terms": float(len(self._postings)),
            "total_length": float(self._total_len),
            "avg_length": self.avg_length,
        }


def build_index(
    rows: Iterable[tuple[int, str]],
    *,
    config: BM25Config | None = None,
) -> BM25Index:
    """Build an index from `(ordinal, text)` pairs.

    Cost note (§6.7): pure-Python tokenisation, document frequencies, and length
    normalisation over 100k chunks on 4 cores is plausibly seconds, not
    "well under a second". Phase 0 measures it (`scripts/measure_bm25.py`) and
    the session-open path is moved off the query path if it is felt.
    """
    idx = BM25Index(config=config or BM25Config())
    idx.add_many(rows)
    return idx


def bm25_scores(index: BM25Index, queries: Sequence[str]) -> list[dict[int, float]]:
    """Convenience for the ablation harness: score several queries at once."""
    return [index.score(q) for q in queries]
