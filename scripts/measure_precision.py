#!/usr/bin/env python3
"""Phase 0 measurement: fp16 vs fp32 embedding agreement (§5.5).

`bge-m3` is published at fp32 and its model card says fp16 "speeds up
computation with a slight performance degradation". "Slight" is not a number, and
silent embedding degradation would corrupt every retrieval metric downstream —
recall, MRR, and nDCG would all shift for a reason that has nothing to do with
the change under test.

So this measures it, on the real model, and the plan gets corrected against the
result rather than trusted:

    python3 scripts/measure_precision.py                  # 200 chunks, the §5.5 sample
    python3 scripts/measure_precision.py --sample 500
    python3 scripts/measure_precision.py --device cuda:1
    python3 scripts/measure_precision.py --json fp16.json

**Requires a GPU (or patience).** It loads `bge-m3` twice, once per dtype. This
is the one Phase 0 gate that cannot be completed on a laptop with 3GB of RAM, and
§5.5 is explicit that it must not be guessed.

What it reports:

- cosine similarity between the fp32 and fp16 embedding matrices, mean and
  minimum. A minimum near 1.0 means some chunk moved; a mean near 1.0 means it
  moved very little *on average*, which is exactly the case where a mean alone
  would be misleading.
- the retrieval-visible consequence: for a sample query set, whether the fp16
  matrix returns the same top-k neighbours, and at what rank the ordering first
  diverges. Cosine agreement does not imply identical rankings, and rankings are
  what recall is computed from.

Verdict follows §5.3: fp16 is fine, otherwise load the embedder at fp32 and note
the VRAM cost.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.config import Settings, from_env  # noqa: E402
from raglab.profiles import cosine  # noqa: E402

DEFAULT_SAMPLE = 200
"""§5.5 specifies 200 chunks. Not tuned: a sample big enough that a rare large
deviation shows up, small enough to fit the comparison in one VRAM budget."""

AGREEMENT_FLOOR = 0.999
"""Below this mean cosine agreement, fp16 is treated as unsafe for the index.

Chosen to be strict. The cost of being wrong here is high and asymmetric: fp32
costs VRAM that §5.3 says fits comfortably anyway, while a silently degraded
index produces metrics that look fine and mean nothing.
"""

PARAGRAPHS = [
    "Term 1 tuition for senior secondary is N450,000 per pupil, payable before "
    "the resumption date. Parents may pay in two instalments provided the second "
    "is settled before the close of the term.",
    "Junior secondary tuition is N185,000 per term. The reduction reflects the "
    "state government's subsidy policy announced in the 2025 budget circular.",
    "Boarding charges are N120,000 per term and are refundable on a pro-rata "
    "basis where a pupil withdraws after the first term has commenced.",
    "Refund policy is governed by section 14 of the education ordinance. A pupil "
    "who withdraws after the first term is not entitled to a refund of tuition "
    "already paid.",
    "Compulsory levies cover the textbook programme, the science laboratory "
    "improvement fund, and the library modernisation levy.",
    "Candidates who sit fewer than four subjects are not eligible for promotion. "
    "The promotion requirement is a grade point average of 4.0 or above.",
    "Bursaries are awarded on the basis of household income, school performance, "
    "and community service. Applications close on the fifteenth of January.",
    "A pupil who accumulates more than ten days of unexplained absence during a "
    "term may not be promoted to the next class.",
]


def sample_texts(n: int, *, seed: int = 3) -> list[str]:
    rng = random.Random(seed)  # noqa: S311 - sample data, not crypto
    out: list[str] = []
    i = 0
    while len(out) < n:
        para = PARAGRAPHS[i % len(PARAGRAPHS)]
        prefix = f"Section {i // len(PARAGRAPHS) + 1}. {rng.choice(['Note', 'Clause', 'Item'])} "
        out.append(prefix + para)
        i += 1
    return out[:n]


QUERIES = [
    "What is Term 1 tuition?",
    "When is Term 2 payable?",
    "Is a refund available after withdrawal?",
    "What does section 14 say?",
    "What are the promotion requirements?",
    "How are bursaries awarded?",
]


def embed(model_id: str, texts: list[str], *, device: str, dtype: str, batch: int) -> list[list[float]]:
    """Embed with an explicit dtype, reporting wall time and peak VRAM."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[dtype]
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = SentenceTransformer(
        model_id,
        device=device,
        model_kwargs={"torch_dtype": torch_dtype},
    )
    start = time.perf_counter()
    vectors = model.encode(texts, batch_size=batch, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.perf_counter() - start
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if device.startswith("cuda") else 0.0
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return [list(map(float, v)) for v in vectors], elapsed, peak_mb


def top_k(matrix: list[list[float]], query: list[float], k: int) -> list[int]:
    scored = [(i, cosine(query, row)) for i, row in enumerate(matrix)]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [i for i, _ in scored[:k]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    ap.add_argument("--device", default=None, help="default: settings.embed_device")
    ap.add_argument("--model", default=None, help="default: settings.embed_model")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    settings: Settings = from_env()
    device = args.device or settings.embed_device
    model_id = args.model or settings.embed_model
    batch = args.batch or settings.embed_batch

    if not device.startswith("cuda"):
        print(
            f"WARNING: running on {device}. §5.5 asks whether fp16 is safe for the\n"
            "stored index, which is a GPU question — on CPU both dtypes go through\n"
            "the same kernels and the comparison is close to meaningless.",
            file=sys.stderr,
        )

    texts = sample_texts(args.sample)
    print("fp16 vs fp32 embedding agreement (§5.5)")
    print(f"  model    {model_id}")
    print(f"  device   {device}")
    print(f"  sample   {len(texts)} chunks")
    print(f"  batch    {batch}")
    print()

    print("Embedding at fp32 (reference)...")
    fp32, fp32_s, fp32_vram = embed(model_id, texts, device=device, dtype="float32", batch=batch)
    print(f"  {len(fp32)} vectors in {fp32_s:.1f}s, peak {fp32_vram:.0f} MB")

    print("Embedding at fp16...")
    fp16, fp16_s, fp16_vram = embed(model_id, texts, device=device, dtype="float16", batch=batch)
    print(f"  {len(fp16)} vectors in {fp16_s:.1f}s, peak {fp16_vram:.0f} MB")
    print()

    agreements = [cosine(a, b) for a, b in zip(fp32, fp16, strict=True)]
    mean_agreement = sum(agreements) / len(agreements)
    worst_i = agreements.index(min(agreements))
    print("Per-vector cosine agreement (fp32 vs fp16)")
    print(f"  mean      {mean_agreement:.6f}")
    print(f"  minimum   {min(agreements):.6f}  (chunk {worst_i})")
    print(f"  below {AGREEMENT_FLOOR}: {sum(1 for a in agreements if a < AGREEMENT_FLOOR)}"
          f" of {len(agreements)}")
    print()

    print(f"Retrieval consequence (top-{args.k} neighbour sets)")
    identical = 0
    first_divergence: list[int] = []
    for q in QUERIES:
        # Embed the query with each dtype rather than reusing a document vector:
        # the query goes through the same arithmetic as the index, so a shared
        # fp32 query would hide exactly the difference being measured.
        q32, _, _ = embed(model_id, [q], device=device, dtype="float32", batch=1)
        q16, _, _ = embed(model_id, [q], device=device, dtype="float16", batch=1)
        a = top_k(fp32, q32[0], args.k)
        b = top_k(fp16, q16[0], args.k)
        if a == b:
            identical += 1
            rank = 0
        else:
            rank = next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), args.k)
        first_divergence.append(rank)
        print(f"  {q[:44]:<44} identical={str(a == b):<5} first diff at rank {rank}")

    identical_set = set(first_divergence) == {0}
    print()
    print(f"  {identical}/{len(QUERIES)} queries returned an identical top-{args.k}.")
    print()

    safe = mean_agreement >= AGREEMENT_FLOOR and identical_set
    print("Verdict (§5.5, §5.3):")
    if safe:
        print(f"  fp16 is safe. Mean agreement {mean_agreement:.6f}, every query returns")
        print("  an identical neighbour set. Keep vector_dtype=float16 and the §6.6")
        print("  size arithmetic as written.")
    else:
        print(f"  fp16 is NOT safe at this tolerance. Mean agreement {mean_agreement:.6f}"
              f" (floor {AGREEMENT_FLOOR});")
        if not identical_set:
            print("  the neighbour ordering also diverges, which is what recall is")
            print("  actually computed from.")
        print("  Set vector_dtype=float32. §5.3 budgets bge-m3 fp32 at ~2.3GB on GPU 1,")
        print("  which fits comfortably, so the cost is index size (§6.6: 4,096 bytes")
        print("  per row instead of 2,048) rather than VRAM pressure.")
        print("  Re-run the Phase 2 baseline recall under fp32 — it is a new index.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "model": model_id,
            "device": device,
            "sample": len(texts),
            "mean_agreement": mean_agreement,
            "min_agreement": min(agreements),
            "worst_chunk": worst_i,
            "queries_identical": identical,
            "queries_total": len(QUERIES),
            "fp32_seconds": fp32_s,
            "fp16_seconds": fp16_s,
            "fp32_peak_mb": fp32_vram,
            "fp16_peak_mb": fp16_vram,
            "fp16_safe": safe,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 0 if safe else 3


if __name__ == "__main__":
    raise SystemExit(main())
