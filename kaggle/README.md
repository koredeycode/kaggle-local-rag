# kaggle-local-rag

Placeholder for the launcher notebook (Phase 8): clone, bootstrap, index, tunnel,
serve.

Two notebooks land here:

| Notebook | Purpose |
|---|---|
| `00_rag_system.ipynb` | clone, bootstrap, index, tunnel, serve |
| `01_ablation.ipynb` | run the eval matrix against a warm index |

The cell-by-cell structure and the Kaggle workaround catalogue are in
[`../docs/notebook-runbook.md`](../docs/notebook-runbook.md). The two scripts the
notebook depends on ship now:

```bash
python3 scripts/preflight.py    # §19.1 environment, device and model report
python3 scripts/cleanup.py      # §19.1 teardown
```

Both are usable outside Kaggle, which is how the local Phase 0 gates were run.

Nothing here works end to end yet: the service is Phase 7 and the index is
Phase 2.
