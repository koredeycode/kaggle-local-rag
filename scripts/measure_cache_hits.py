#!/usr/bin/env python3
"""Phase 0 measurement: embedding-cache hit rate across chunker settings (§20).

The chunking grid in §15a is 16 fresh index builds, which §21 budgets at 8-16
GPU-hours — more than half the weekly quota for one phase. The plan's stated
lever is the content-addressed embedding cache: if `fixed/512` and
`recursive/512` produce heavily overlapping chunks, the second build is mostly
cache hits instead of 16 hours of GPU work.

**That premise has never been measured.** The plan says "Phase 0 measures the
real hit rate rather than assuming it". This is that measurement, and it is
model-independent: the cache is keyed by chunk-text hash, so whether two chunker
settings produce the same chunk text depends only on the chunkers. It therefore
runs here, on a laptop, with zero GPU and zero model download.

    python3 scripts/measure_cache_hits.py
    python3 scripts/measure_cache_hits.py --sizes 256 512 1024 --json out.json
    python3 scripts/measure_cache_hits.py --strategies fixed recursive structural

What it reports: for each pair of (strategy, size), the fraction of chunks in the
second configuration whose exact text already appeared in the first. Exact match
only — the cache is content-addressed, so a one-character difference is a miss,
and pretending otherwise would overstate the hit rate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.chunking import get_chunker  # noqa: E402
from raglab.tokens import get_tokenizer  # noqa: E402
from raglab.types import Document, text_hash  # noqa: E402

#: A synthetic corpus with the structure real documents have: headings, short
#: paragraphs, tables, and prose. A corpus of uniform lorem ipsum would make every
#: chunker produce identical output and report a 100% hit rate, which is the one
#: number this script must not report.
PARAGRAPHS = [
    "Term 1 tuition for senior secondary is N450,000 per pupil, payable before the "
    "resumption date. Parents may pay in two instalments provided the second is "
    "settled before the close of the term.",
    "Junior secondary tuition is N185,000 per term. The reduction reflects the state "
    "government's subsidy policy announced in the 2025 budget circular.",
    "Boarding charges are N120,000 per term and are refundable on a pro-rata basis "
    "where a pupil withdraws after the first term has commenced.",
    "Refund policy is governed by section 14 of the education ordinance. A pupil who "
    "withdraws after the first term is not entitled to a refund of tuition already "
    "paid. Applications must be made in writing within fourteen days.",
    "Compulsory levies cover the textbook programme, the science laboratory "
    "improvement fund, and the library modernisation levy. Each is itemised "
    "separately on the fee schedule.",
    "Examination and result processing: candidates who sit fewer than four "
    "subjects are not eligible for promotion. The promotion requirement is a grade "
    "point average of 4.0 or above across all subjects examined.",
    "Attendance requirements are strict. A pupil who accumulates more than ten days "
    "of unexplained absence during a term may not be promoted to the next class.",
    "Bursaries are awarded on the basis of household income, school performance, "
    "and community service. Applications close on the fifteenth of January each year.",
]

HEADINGS = [
    "Fee Structure 2026",
    "Payment and Instalments",
    "Refund Policy",
    "Levies",
    "Examinations and Promotion",
    "Attendance",
    "Bursaries and Grants",
]

SCHOOLS = [
    "Lagos State Model Secondary, Ikeja",
    "Lagos State Model Secondary, Badagry",
    "Lagos State Comprehensive Secondary, Epe",
]


def synthetic_document(index: int, *, seed: int = 5) -> str:
    """A document with heading structure, in the style of a real fee schedule."""
    rng = random.Random(seed + index)
    school = rng.choice(SCHOOLS)
    lines = [f"# {school}", "", "## Academic Session 2026", ""]
    n_sections = rng.randint(3, len(HEADINGS))
    for heading in rng.sample(HEADINGS, n_sections):
        lines.append(f"### {heading}")
        lines.append("")
        for _ in range(rng.randint(1, 3)):
            lines.append(rng.choice(PARAGRAPHS))
            lines.append("")
    return "\n".join(lines)


def chunk_hashes(strategy: str, size: int, overlap: int, docs: list, tokenizer) -> set[str]:
    """Content hashes of every chunk this configuration produces."""
    params: dict[str, int] = {"size": size}
    if strategy in ("fixed", "recursive"):
        params["overlap"] = overlap
    if strategy == "structural":
        params["overlap"] = 0
    if strategy == "semantic":
        params["threshold"] = 0.5
    chunker = get_chunker(strategy, tokenizer=tokenizer, **params)
    out: set[str] = set()
    ordinal = 0
    for doc in docs:
        for chunk in chunker.chunk(doc, start_ordinal=ordinal):
            out.add(text_hash(chunk.embed_text))
            ordinal += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents", type=int, default=60)
    ap.add_argument("--strategies", nargs="*", default=["fixed", "recursive", "structural"])
    ap.add_argument("--sizes", nargs="*", type=int, default=[256, 512, 1024])
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--json", type=Path, help="write results to this path")
    args = ap.parse_args()

    tokenizer = get_tokenizer("regex")
    docs = []
    for i in range(args.documents):
        text = synthetic_document(i)
        marks = []
        for line in text.splitlines():
            if line.startswith("#") and line.lstrip("#").strip():
                marks.append((text.index(line), line.lstrip("# ").strip()))
        docs.append(
            Document(
                doc_id=f"sha256:synth{i:04d}",
                path=f"corpus/synthetic/{i:04d}.md",
                text=text,
                heading_path_at=tuple(dict(marks).items()),
                raw_len=len(text),
            )
        )

    print("Embedding-cache hit rate across chunker settings (§20, §15a)")
    print()
    print(f"Corpus: {len(docs)} synthetic documents, "
          f"{sum(len(d.text) for d in docs):,} characters")
    print("Cache key: sha256 of the embedded chunk text (exact match only).")
    print("Model-independent: the cache is content-addressed, so this figure does")
    print("not change when the embedder does.")
    print()

    configs = [(s, z) for s in args.strategies for z in args.sizes]
    results: dict[tuple[str, int], set[str]] = {}
    for strategy, size in configs:
        results[(strategy, size)] = chunk_hashes(
            strategy, size, args.overlap, docs, tokenizer
        )

    print("Distinct chunk texts per configuration")
    print(f"{'strategy':>12} {'size':>6} {'distinct texts':>15}")
    print("-" * 36)
    for key in configs:
        print(f"{key[0]:>12} {key[1]:>6} {len(results[key]):>15,}")

    # Baseline-first hit rate: every other configuration measured against
    # `fixed/512`, which is the §10 default. This is the number that decides
    # whether the grid costs 16 rebuilds or 16 plus a near-free set of cache hits.
    baseline_key = ("fixed", 512)
    if baseline_key in results:
        baseline = results[baseline_key]
        print()
        print(f"Hit rate against the baseline {baseline_key[0]}/{baseline_key[1]} "
              f"({len(baseline):,} chunks)")
        print(f"{'strategy':>12} {'size':>6} {'chunks':>8} {'hits':>8} {'hit rate':>9}")
        print("-" * 46)
        for key in configs:
            if key == baseline_key:
                continue
            cur = results[key]
            hits = len(cur & baseline)
            print(f"{key[0]:>12} {key[1]:>6} {len(cur):>8,} "
                  f"{hits:>8,} {hits / len(cur):>8.1%}")

    print()
    print("All-pairs overlap (the full 16-cell grid's best case)")
    print("Each row is the chance that THIS cell's chunks are already cached")
    print("when a cell that ran before it populated the cache.")
    print(f"{'strategy':>12} {'size':>6} {'max prior hit':>14} {'mean prior hit':>15}")
    print("-" * 50)
    grid_summary = []
    for key in configs:
        cur = results[key]
        priors = [results[p] for p in configs if configs.index(p) < configs.index(key)]
        if not priors:
            continue
        rates = []
        for prior in priors:
            union = prior | cur
            rates.append(len(prior & cur) / len(cur) if cur else 0.0)
        grid_summary.append({
            "strategy": key[0], "size": key[1],
            "max_prior_hit": max(rates), "mean_prior_hit": sum(rates) / len(rates),
        })
        print(f"{key[0]:>12} {key[1]:>6} {max(rates):>13.1%} "
              f"{sum(rates) / len(rates):>14.1%}")

    print()
    print("Interpretation (§15a):")
    mean_prior = [g["mean_prior_hit"] for g in grid_summary]
    if mean_prior:
        overall = sum(mean_prior) / len(mean_prior)
        print(f"  Mean prior-cell hit rate is {overall:.1%}.")
        if overall < 0.20:
            print(
                "  That is LOW. The cache barely helps, so the chunking grid costs\n"
                "  close to 16 full rebuilds. The honest move is to NARROW the grid\n"
                "  per §15a: drop `semantic` (it needs an embedding pass per document\n"
                "  anyway) and ablate sizes only on `fixed` and `structural`."
            )
        elif overall < 0.50:
            print(
                "  That is MODERATE. The cache removes a real fraction of the 8-16\n"
                "  GPU-hours, but not most of it. Re-baseline the §21 budget with\n"
                "  this number rather than the 8-16h figure, and run the grid on a\n"
                "  100-200 document corpus subset as §15a specifies."
            )
        else:
            print(
                "  That is HIGH. Most later cells are mostly cache hits, so the grid\n"
                "  is much cheaper than 16 rebuildes and the §21 budget can be cut.\n"
                "  Verify on the real corpus before relying on it: synthetic documents\n"
                "  share phrasing, which inflates overlap."
            )
    print(
        "  This corpus is synthetic and shares paragraph phrasing across documents,\n"
        "  which INFLATES the hit rate. Re-run against the real corpus before\n"
        "  spending or saving quota on the number."
    )

    if args.json:
        payload = {
            "documents": args.documents,
            "overlap": args.overlap,
            "chunk_counts": {f"{s}/{z}": len(v) for (s, z), v in results.items()},
            "grid": grid_summary,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
