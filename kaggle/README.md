# kaggle/

## Notebooks

| Notebook | Purpose | Status |
|---|---|---|
| `00_phase0_checks.ipynb` | Close the two hardware-gated Phase 0 measurements; validate the environment claims | **ready** |
| `00_rag_system.ipynb` | clone, bootstrap, index, tunnel, serve | Phase 8 |
| `01_ablation.ipynb` | run the eval matrix against a warm index | Phase 8 |

`00_phase0_checks.ipynb` is **not** the launcher. There is no service to serve and
no index to build at Phase 0. What it does:

| Cell | Gate | Why it matters |
|---|---|---|
| 1 | §3 | P100 is sm_60 and breaks PyTorch silently; T4 is sm_75 |
| 4 | §19.1 | Preflight report |
| 5 | §4 | Settles whether numpy's fp16 matmul really is a slow path — asserted in the docs, never measured |
| 6 | §5.5 | **Gates the vector dtype, which gates the index format** |
| 7 | §5.2 | Optional. The `think: false` trap, and the real §4 device-placement check |

### Running it

Copy the cells into a Kaggle notebook, **Run All**, with **T4 x2** selected in the
Accelerator dropdown. Budget roughly 1 hour, mostly the 2.3GB `bge-m3` download.

Everything lands in `/kaggle/working/phase0_results.json`, plus `fp16.json` and
`preflight.json` beside it. Those are the deliverable — send them back.

Two toggles at the top of cell 2:

- `RUN_PRECISION = True` — the §5.5 measurement. Leave it on.
- `RUN_OLLAMA = False` — ~10GB of downloads for one question. Flip it only if you
  have quota to spare; Phase 5 does not reach it for several sessions.

### After it runs

```bash
python scripts/cleanup.py --dry-run    # what would be stopped
python scripts/cleanup.py              # stop it
```

Then stop the Kaggle session. §19.2: **Run All fires every cell**, so cleanup must
not be wired to it — an idle GPU session burns the 30h/week quota for nothing.

### Why this is generated

`scripts/gen_phase0_notebook.py` produces the `.ipynb`, so the JSON is valid by
construction. A notebook with a stray trailing comma fails at the worst possible
moment: on Kaggle, after the session has already spent quota. Regenerate with
`make notebook`.

## Scripts these notebooks use

```bash
python scripts/preflight.py       # §19.1 environment, device and model report
python scripts/cleanup.py         # §19.1 teardown
python scripts/measure_precision.py  # §5.5 fp16 vs fp32 (needs bge-m3)
```

All three run outside Kaggle too, which is how the two laptop-completable
Phase 0 gates were run. See [`../docs/measurements.md`](../docs/measurements.md).