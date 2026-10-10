#!/usr/bin/env python3
"""Phase 0 measurement: BM25 build time on a synthetic chunk set (§6.7, §20).

§6.7 is explicit that BM25 build time is *measured in Phase 0, not assumed*:
pure-Python tokenisation, document frequencies and length normalisation over
100k chunks on 4 cores is plausibly seconds, not "well under a second". If it is
slow enough to be felt, session-open moves outside the query path.

Runs with no GPU, no models, and no network. This gate is therefore completable
on a laptop, which is worth knowing before spending Kaggle quota on it.

    python3 scripts/measure_bm25.py                  # 10k, 50k, 100k
    python3 scripts/measure_bm25.py --chunks 100000  # one size
    python3 scripts/measure_bm25.py --json out.json

Exit code is 0 on any successful measurement — this reports, it does not gate.
The interpretation is in the printed verdict, because a threshold here would be
a guess about a machine this has not run on.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.index.bm25 import build_index  # noqa: E402
from raglab.metrics import percentile  # noqa: E402

VOCAB = [
    "fee", "tuition", "term", "senior", "junior", "boarding", "levy", "refund",
    "principal", "admission", "uniform", "textbook", "session", "resumption",
    "schedule", "statute", "section", "chapter", "ordinance", "guideline",
    "payment", "installment", "deadline", "receipt", "bursar", "registrar",
    "naira", "amount", "outstanding", "balance", "waiver", "bursary", "grant",
    "exam", "subject", "credit", "gpa", "promotion", "repetition", "class",
    "attendance", "compulsory", "elective", "vocational", "commercial", "arts",
    "science", "technology", "laboratory", "library", "hostel", "matron",
]

# §11 requires a *representative* vocabulary, not a handful of words. BM25 query
# cost is dominated by posting-list length, and posting length is inversely
# related to vocabulary size: with 52 distinct terms every term appears in ~19%
# of a 10k-row corpus, so every scoring pass walks thousands of postings that a
# real corpus would never produce. Understating this would make the query half of
# the measurement look better than it is.
_SYNONYMS = [
    "policy", "procedure", "regulation", "directive", "circular", "memorandum",
    "bulletin", "gazette", "notification", "advisory", "instruction",
]


def build_vocab() -> list[str]:
    """Vocabulary with realistic per-word variety."""
    out = list(VOCAB)
    for base in list(VOCAB):
        out.append(base + "s")
        out.append(base + "ing")
    return out + _SYNONYMS * 20


SIZES = (10_000, 50_000)


def synthetic_chunks(n: int, *, seed: int = 17, mean_tokens: int = 220) -> list[tuple[int, str]]:
    """`n` `(ordinal, text)` rows of realistic token counts.

    Seeded so a run is reproducible: an unseeded measurement cannot be compared
    with a later one, and the whole point of measuring is comparison (§6.7).
    `mean_tokens` matches the §10 default of ~512 with overlap producing more
    rows per document, so the row count and document count both scale the way
    they would in a real index.
    """
    rng = random.Random(seed)
    vocab = build_vocab()
    rows: list[tuple[int, str]] = []
    for i in range(n):
        length = max(12, int(rng.gauss(mean_tokens, mean_tokens / 4)))
        words = [rng.choice(vocab) for _ in range(length)]
        rows.append((i, " ".join(words)))
    return rows


def measure(n: int, *, seed: int, repeats: int) -> dict[str, float]:
    rows = synthetic_chunks(n, seed=seed)
    chars = sum(len(t) for _, t in rows)

    # Build a couple of times; the first pays import and first-touch costs that
    # a warm session would not.
    build_ms: list[float] = []
    index = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        index = build_index(rows)
        build_ms.append((time.perf_counter() - start) * 1000.0)

    assert index is not None

    queries = [
        "term 1 tuition schedule",
        "boarding levy refund policy",
        "bursary admission deadline",
        "statute section chapter",
    ]
    query_ms: list[float] = []
    picks = random.Random(0)  # noqa: S311 - test data, not crypto
    for _ in range(max(1, repeats * 5)):
        start = time.perf_counter()
        index.search(picks.choice(queries), top_k=40)
        query_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "chunks": float(n),
        "corpus_mb": round(chars / 1_048_576, 3),
        "build_ms_min": round(min(build_ms), 2),
        "build_ms_median": round(percentile(build_ms, 50), 2),
        "build_ms_max": round(max(build_ms), 2),
        "build_us_per_chunk": round(percentile(build_ms, 50) * 1000.0 / n, 1),
        "query_ms_p50": round(percentile(query_ms, 50), 3),
        "query_ms_p95": round(percentile(query_ms, 95), 3),
        "terms": float(len(index.terms())),
        "avg_doc_length": round(index.avg_length, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", type=int, nargs="*", default=list(SIZES))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", type=Path, help="write results to this path")
    args = ap.parse_args()

    rows = [measure(n, seed=args.seed, repeats=args.repeats) for n in args.chunks]

    print("BM25 build and query timing (§6.7)")
    print("Machine: " + ", ".join(f"{k} cores" for k in (4,)) + " assumed; see §3 for Kaggle.")
    print()
    header = (
        f"{'chunks':>9} {'corpus MB':>10} {'build ms':>10} {'us/chunk':>9} "
        f"{'query p50':>10} {'query p95':>10} {'terms':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{int(r['chunks']):>9,} {r['corpus_mb']:>10.1f} "
            f"{r['build_ms_median']:>10.1f} {r['build_us_per_chunk']:>9.1f} "
            f"{r['query_ms_p50']:>10.3f} {r['query_ms_p95']:>10.3f} "
            f"{int(r['terms']):>7,}"
        )

    print()
    print("Verdict (§6.7):")
    per_chunk = max(r["build_us_per_chunk"] for r in rows)
    for r in rows:
        n = int(r["chunks"])
        print(
            f"  {n:>7,} chunks -> {r['build_ms_median']:>9.0f} ms build, "
            f"{r['query_ms_p50']:.2f} ms p50 query"
        )
    print()
    print(
        f"  Marginal cost is {per_chunk:.0f} us/chunk, so a 100k-chunk index costs "
        f"~{per_chunk * 100_000 / 1_000_000:.0f} s"
    )
    print("  to build. That is a linear extrapolation from the measured sizes above,")
    print("  not a measurement at 100k. Run --chunks 100000 --repeats 1 to confirm.")
    print()
    print(
        f"  CAVEAT on the query columns: this synthetic corpus has only "
        f"{int(rows[-1]['terms'])} distinct terms, so every"
    )
    print(
        "  query term walks a posting list of corpus/167 rows. A real corpus has"
    )
    print(
        "  orders of magnitude more vocabulary, so its posting lists are far"
    )
    print(
        "  shorter and real query cost will be much lower than the numbers above."
    )
    print(
        "  Treat these as an UPPER BOUND on query cost, and the build column as"
    )
    print("  the figure that actually needs acting on.")

    largest = max(rows, key=lambda r: r["build_ms_median"])
    if largest["build_ms_median"] > 1000:
        print()
        print("  ACTION (§6.7): build time is felt by a user, so session-open must")
        print("  move OUTSIDE the query path — build on POST /index or on session")
        print("  selection in the UI, and cover index-open with the §6.7 semaphore.")
        print("  The LRU eviction order (BM25 first) matters much more than the plan")
        print("  assumed: BM25 is the expensive structure, not the mmap handle.")
    else:
        print()
        print("  Build time is not felt by a user here, so a synchronous build on")
        print("  first query would be acceptable. Re-measure on the Kaggle VM: 4 cores")
        print("  there is an assumption in this report, not a measurement.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
