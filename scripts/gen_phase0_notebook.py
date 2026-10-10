#!/usr/bin/env python3
"""Generate kaggle/00_phase0_checks.ipynb.

Written as a generator rather than by hand because a notebook is JSON, and a
hand-edited one with a stray trailing comma fails at the worst moment: on Kaggle,
after the session has already burned GPU quota.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NBFORMAT = 4
NBFORMAT_MINOR = 5


def md(text: str) -> dict[str, Any]:
    lines = text.strip("\n").split("\n")
    src = [f"{line}\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(text: str) -> dict[str, Any]:
    lines = text.strip("\n").split("\n")
    src = [f"{line}\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


CELLS: list[dict[str, Any]] = [
    md(
        r'''
# Kaggle Local RAG — Phase 0 checks

**This is not the launcher.** The system notebook (`00_rag_system.ipynb`) is a
Phase 8 deliverable; there is no service to serve and no index to build yet.

What this notebook does is close the two Phase 0 gates that need real hardware,
and validate the environment claims the rest of the plan rests on:

| Check | Gate | Why now |
|---|---|---|
| GPU identity | §3 | P100 is sm_60 and breaks PyTorch silently |
| fp16 vs fp32 embeddings | §5.5 | **Gates the vector dtype, which gates the index format** |
| numpy fp16 matmul | §4 | A load-bearing claim currently asserted, never measured |
| Ollama `think: false` | §5.2 | Settles which Qwen3 tag behaviour you actually have |

**The fp16 measurement is the reason to do this before building anything.** If
fp16 turns out to be unsafe, `vector_dtype` becomes fp32, the stored rows change
from 2,048 to 4,096 bytes, and every index built beforehand has to be re-embedded.
That is §21's 0.5–1.0h per 1k documents, paid twice.

## Quota

30 GPU-hours per week, 12h per session. This notebook should take roughly **1
hour**, most of it the 2.3GB `bge-m3` download. Shut the session down when you
are done — an idle GPU session burns quota for nothing.

## If a cell fails

`scripts/measure_precision.py` has never been executed. If it throws, the
traceback is the deliverable: send it back rather than patching it blind. The
likely failure is a dtype/`model_kwargs` mismatch in `sentence-transformers`.

Most cells record into `phase0_results.json` instead of raising, so one session
collects everything it can even when part of it breaks.
'''
    ),
    code(
        r'''
# ---------------------------------------------------------------------------
# CONFIG — the only cell you should need to edit (§19.1)
# ---------------------------------------------------------------------------
REPO = "https://github.com/koredeycode/kaggle-local-rag.git"
WORKDIR = "/kaggle/working/raglab"

# §19.2: measuring fp16-vs-fp32 needs the real model on a real GPU.
RUN_PRECISION = True

# Skip this by default. Ollama is ~5GB of download plus a 5.2GB model, and it is
# only needed to settle the §5.2 think:false trap, which Phase 5 does not reach
# for several sessions. Flip it on only if you have quota to spare.
RUN_OLLAMA = False

# `qwen3:14b` does not fit beside anything on one T4 (§5.2), so 8b is the only
# sensible default on this hardware.
GEN_MODEL = "qwen3:8b"

RESULTS = {}          # collected across cells, written out at the end
RESULTS_PATH = "/kaggle/working/phase0_results.json"

import json, os, subprocess, sys, time, traceback
from pathlib import Path


def record(key, value):
    """Collect a result without failing the notebook."""
    RESULTS[key] = value
    print(f"[recorded] {key} = {value}")


def run(cmd, timeout=1800):
    """Shell out, returning (returncode, output). Never raises."""
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
    started = time.perf_counter()
    proc = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-6000:])
    print(f"[exit {proc.returncode} in {time.perf_counter()-started:.1f}s]")
    return proc.returncode, out


def section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
'''
    ),
    md(
        r'''
## 1. Verify the GPU

**Do not skip this, and do not continue past a failure.**

§19.2: P100 is `sm_60`. Modern PyTorch needs `sm_70+`, so on a P100 kernels die
with no error message — you get a session that appears to work and produces
nothing. T4 is `sm_75`.

Selecting `T4 x2` in the Accelerator dropdown is the fix. If you pushed via the
CLI instead, `"machine_shape": "NvidiaTeslaT4"` belongs in
`kernel-metadata.json`; `"enable_gpu": true` alone lands you on P100.
'''
    ),
    code(
        r'''
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), (
    "No CUDA device visible. Select 'T4 x2' in the Accelerator dropdown and "
    "re-run. §3: all GPU work is Kaggle-only."
)

n = torch.cuda.device_count()
print("visible GPUs:", n)

caps = []
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    cap = f"{props.major}.{props.minor}"
    caps.append(cap)
    free, total = torch.cuda.mem_get_info(i)
    print(f"  [{i}] {props.name}  sm_{cap.replace('.', '')}  "
          f"{free//(1024*1024)}MB free of {total//(1024*1024)}MB")

# Hard gates. §19.1: failing loudly here is far cheaper than failing silently
# forty minutes into a measurement.
assert n == 2, (
    f"expected T4 x2 (2 GPUs), found {n}. §4's workload split needs both: "
    "Ollama on GPU 0, the retrieval worker on GPU 1."
)
assert all(float(c) >= 7.0 for c in caps), (
    f"compute capability {caps} is below 7.0 — this is a P100 (sm_60). "
    "Modern PyTorch needs sm_70+ and kernels will die with no error. "
    "Switch to T4 x2 in the Accelerator dropdown. (§19.2)"
)

record("gpu_count", n)
record("compute_capabilities", caps)
record("gpu_names", [torch.cuda.get_device_properties(i).name for i in range(n)])
print("\\nOK: T4 x2 confirmed.")
'''
    ),
    md(
        r'''
## 2. Install

`pyproject.toml` puts torch, sentence-transformers, and FastAPI in the `local`
extra rather than the core, so `pip install -e .` does not drag a multi-GB wheel
onto a machine that only wants the logic (§8 rule 1).

§19.2: **do not pin torch.** Kaggle's image ships a working build and pip will
fight it into a broken one.
'''
    ),
    code(
        r'''
import sys

code, out = run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[local]"], timeout=1800)
record("pip_local_exit", code)
if code != 0:
    print("pip install failed. Try the torch-free path for the numpy check below:")
    print("  !pip install -q numpy")
'''
    ),
    md(
        r'''
## 3. Clone

The repository is public, so a plain clone works. The §19.2 `GIT_ASKPASS`
workaround is only needed for a fork or a private clone.
'''
    ),
    code(
        r'''
import shutil

target = Path(WORKDIR)
if target.exists():
    print(f"{target} already exists — pulling instead of cloning.")
    code, out = run(["git", "-C", str(target), "pull", "--ff-only"])
else:
    code, out = run(["git", "clone", "--depth", "1", REPO, str(target)])

if code != 0:
    print("CLONE FAILED. The rest of this notebook needs the repo. Stop here.")
else:
    code, out = run(["git", "-C", str(target), "log", "--oneline", "-1"])
    os.chdir(target)
    sys.path.insert(0, str(target / "src"))
    print(f"cwd: {os.getcwd()}")
    record("repo_head", out.strip().splitlines()[-1] if out.strip() else "?")
'''
    ),
    md(
        r'''
## 4. Preflight

Reports the environment, per-GPU memory, and — once you pass `--pid` — which
*physical* GPU each named process is on. Counting visible devices is the weaker
check: it passes even when Ollama and the retrieval worker share a card.
'''
    ),
    code(
        r'''
code, out = run([sys.executable, "scripts/preflight.py",
                 "--json", "/kaggle/working/preflight.json",
                 "--require", "python", "cuda_count"], timeout=600)
record("preflight_exit", code)

# The exact device-placement assertion §19.1 wants. This runs after Ollama is
# started in section 7, so it is recorded here as a reminder rather than run.
print(
    "§19.1 device placement is checked once Ollama is running:\\n"
    "  python scripts/preflight.py --pid ollama=$(pgrep -x ollama) --require device_separation"
)
'''
    ),
    md(
        r'''
## 5. numpy fp16 matmul — is §4 right?

The plan says numpy has no fast `float16` BLAS path, so exact dense search must
upcast fp16→fp32 before the matmul, and that skipping this makes exact search
read as "slow" when it is really a dtype bug (§4, §23, and the §22 reference
test).

That is asserted in `docs/architecture.md` and has never been measured. It takes
about thirty seconds to settle, and the answer is a design input rather than a
detail.
'''
    ),
    code(
        r'''
try:
    import numpy as np

    n, d = 50_000, 1024
    rng = np.random.default_rng(0)
    q16 = rng.standard_normal(d).astype(np.float16)
    m16 = rng.standard_normal((n, d)).astype(np.float16)
    m32 = m16.astype(np.float32)

    def bench(fn, label, repeats=3):
        fn()  # warm up: first call pays import and page-fault costs
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t) * 1000)
        best = min(times)
        print(f"  {label:<34} {best:9.1f} ms")
        return best

    print(f"numpy {np.__version__}  |  {n:,} x {d} matrix\n")

    naive = bench(lambda: q16 @ m16.T, "fp16 matmul (naive)")
    upcast = bench(lambda: q16.astype(np.float32) @ m32.T, "fp32 matmul (upcast)")

    ratio = naive / upcast if upcast else 0.0
    print(f"\\nupcast speedup: {ratio:.1f}x")

    # Verify the two paths agree numerically, so the comparison is meaningful.
    a = np.asarray(q16 @ m16.T, dtype=np.float32)
    b = q16.astype(np.float32) @ m32.T
    rel = float(np.abs(a - b).max() / (np.abs(b).max() or 1.0))
    print(f"max relative difference between the two paths: {rel:.2e}")

    record("bm25_fp16_ms", round(naive, 1))
    record("bm25_fp32_ms", round(upcast, 1))
    record("dense_fp16_speedup", round(ratio, 2))
    record("dense_fp16_max_rel_diff", rel)

    if ratio >= 3:
        print("\\nVERDICT: §4 is CONFIRMED. The fp16 path is several times slower,")
        print("so the upcast is required, not an optimisation.")
    elif ratio >= 1.5:
        print("\\nVERDICT: §4 is confirmed but modest. Upcast anyway — it is cheap")
        print("and it removes a failure mode that reads as a design flaw.")
    else:
        print("\\nVERDICT: §4's claim does NOT hold on this machine. numpy appears to")
        print("have a usable fp16 BLAS path here. §4 and docs/architecture.md should")
        print("be corrected against this measurement, and the §22 reference test")
        print("still needs to exist to catch regressions.")

    print("\\nSanity: §6.6 arithmetic, recomputed rather than trusted.")
    row_bytes = 1024 * 2
    for chunks in (10_000, 100_000, 1_000_000):
        print(f"  {chunks:>9,} chunks -> {chunks*row_bytes/1_048_576:>9.1f} MB vectors")

except ImportError:
    print("numpy unavailable — skipping. Install it with `!pip install -q numpy`.")
except Exception:
    traceback.print_exc()
    record("numpy_fp16_check", "failed — see traceback above")
'''
    ),
    md(
        r'''
## 6. fp16 vs fp32 embedding agreement — the priority measurement

§5.5: `bge-m3` is published at fp32 and its card says fp16 gives "a slight
performance degradation". "Slight" is not a number, and silent embedding
degradation would corrupt every retrieval metric downstream — recall, MRR, and
nDCG would all shift for a reason unrelated to whatever change is under test.

The script reports two things, and the second is the one that matters: cosine
agreement between the two matrices, and whether a query set still returns an
identical top-k neighbour set. Cosine agreement does not imply identical
rankings, and rankings are what recall is computed from.

**Exit code 3 means fp16 failed.** The response is `vector_dtype=float32`, and
Phase 2's baseline has to be measured under it.
'''
    ),
    code(
        r'''
if RUN_PRECISION:
    cmd = [sys.executable, "scripts/measure_precision.py",
           "--json", "/kaggle/working/fp16.json", "--device", "cuda:1"]
    code, out = run(cmd, timeout=3600)
    record("measure_precision_exit", code)
    record("fp16_safe", {0: True, 3: False}.get(code, "indeterminate"))
    if code == 3:
        print("\\nACTION: set vector_dtype=float32 in Settings BEFORE building any")
        print("index. §5.3 budgets bge-m3 fp32 at ~2.3GB on GPU 1, which fits;")
        print("the cost is index size, not VRAM pressure.")
    elif code != 0:
        print("\\nmeasure_precision.py failed. Send the traceback back — it has never")
        print("been run and is the most likely thing here to need a fix.")
else:
    print("RUN_PRECISION is False — skipping. Set it True to measure (§5.5).")
'''
    ),
    md(
        r'''
## 7. Ollama — optional

Off by default. It is ~5GB of installer plus a 5.2GB model, and it only settles
one thing: whether your `qwen3:8b` tag honours `think: false` (§5.2). Some Qwen3
tags have been reported to ignore it, and preflight is the only way to find out
which behaviour you have.

Phase 5 does not reach this for several sessions, so it is a quota decision
rather than a correctness one.
'''
    ),
    code(
        r'''
if RUN_OLLAMA:
    # §19.2: zstd first — newer Ollama installers fail without it.
    run(["apt-get", "install", "-y", "-q", "zstd"], timeout=600)
    code, out = run(
        "curl -fsSL https://ollama.com/install.sh | sh", timeout=1800
    )
    record("ollama_install_exit", code)

    if code == 0:
        # §19.2: Popen, never shell '&' — a notebook cell's children die with it.
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=open("/tmp/ollama.log", "w"),
            stderr=subprocess.STDOUT,
            env={**os.environ, "OLLAMA_CONTEXT_LENGTH": "8192"},
        )
        import builtins
        builtins.ollama_process = proc  # pin the handle; GC kills the tunnel otherwise

        # §19.2: not ready immediately. Poll rather than sleep-and-hope.
        ready = False
        for _ in range(5):
            time.sleep(3)
            try:
                import urllib.request
                with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                continue
        record("ollama_ready", ready)
        print(f"Ollama ready: {ready}")

        if ready:
            code, out = run(["ollama", "pull", GEN_MODEL], timeout=3600)
            # §19.2: assert the EXACT name appears afterwards. Guessing a tag is
            # how you get `model not found`.
            code, tags = run(
                ["curl", "-s", "http://127.0.0.1:11434/api/tags"], timeout=60
            )
            present = GEN_MODEL in tags
            record("ollama_model_present", present)
            print(f"{GEN_MODEL} present in /api/tags: {present}")
            if not present:
                print("PULL FAILED — assert the real tag name from /api/tags (§19.2).")

            # The §4 device-placement check, now that Ollama exists.
            code, out = run(
                [sys.executable, "scripts/preflight.py",
                 "--pid", "ollama=" + str(proc.pid),
                 "--require", "device_separation"],
                timeout=300,
            )
            record("device_separation_exit", code)
    else:
        print("Ollama install failed — see output above (§19.2: zstd first).")
else:
    print("RUN_OLLAMA is False — skipping. Set it True to check the §5.2 think:false trap.")
'''
    ),
    md(
        r'''
## 8. Summary

Everything lands in `/kaggle/working/phase0_results.json`. That file is the
deliverable.
'''
    ),
    code(
        r'''
import json as _json

RESULTS["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
RESULTS["gpu_info"] = _json.dumps({
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cpu_count": os.cpu_count(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
})

Path(RESULTS_PATH).write_text(_json.dumps(RESULTS, indent=2, default=str) + "\\n")
print(_json.dumps(RESULTS, indent=2, default=str))
print(f"\\nWritten: {RESULTS_PATH}")

print("\\n" + "=" * 72)
print("  SEND THESE BACK")
print("=" * 72)
print("""
  1. /kaggle/working/phase0_results.json   ← the results dict
  2. /kaggle/working/fp16.json             ← §5.5 precision result (if run)
  3. /kaggle/working/preflight.json        ← §19.1 environment report
  4. The section 5 output                   ← the fp16 matmul speedup
  5. Any traceback, especially from measure_precision.py

  The first two decide vector_dtype and therefore whether Phase 2's index
  format is right on the first attempt.
""")

print("=" * 72)
print("  TEARDOWN — run this before you stop, or the session keeps burning quota")
print("=" * 72)
print("""
  §19.2: Run All fires every cell, so cleanup must NOT be wired to it. Keep
  CLEANUP_ON_RUN_ALL = False and stop things by hand.

  # from the repo directory
  python scripts/cleanup.py --dry-run    # what would be stopped
  python scripts/cleanup.py              # stop it

  Then stop the Kaggle session. An idle GPU session spends the 30h/week quota
  for nothing.
""")
'''
    ),
]


def build() -> dict[str, Any]:
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": NBFORMAT,
        "nbformat_minor": NBFORMAT_MINOR,
    }


def main() -> int:
    out = Path(__file__).resolve().parents[1] / "kaggle" / "00_phase0_checks.ipynb"
    nb = build()
    # Validate before writing: a notebook that fails to parse is discovered on
    # Kaggle, after the quota is already spent.
    json.loads(json.dumps(nb))
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # Re-read and assert structure.
    check = json.loads(out.read_text(encoding="utf-8"))
    assert check["nbformat"] == NBFORMAT
    assert len(check["cells"]) == len(CELLS)
    for i, cell in enumerate(check["cells"]):
        assert cell["cell_type"] in ("markdown", "code"), i
        assert isinstance(cell["source"], list) and cell["source"], i
        if cell["cell_type"] == "code":
            for key in ("execution_count", "outputs", "metadata"):
                assert key in cell, f"cell {i} missing {key}"
    print(f"OK  {out}  ({len(check['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
