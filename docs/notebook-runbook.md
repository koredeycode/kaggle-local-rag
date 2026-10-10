# Notebook runbook

Companion to §19. Cell structure and the Kaggle workaround catalogue.

Phase 0 ships `scripts/preflight.py` and `scripts/cleanup.py`. The notebooks
themselves arrive in Phase 8, when there is a service to launch.

## Two notebooks

| Notebook | Purpose |
|---|---|
| `kaggle/00_rag_system.ipynb` | clone, bootstrap, index, tunnel, serve |
| `kaggle/01_ablation.ipynb` | run the eval matrix against a warm index |

The notebook is the deployment target, so its structure is a design artefact
rather than an afterthought: a single CONFIG cell, assert-based verification at
every boundary, and explicit teardown.

## Cell plan

| # | Type | Purpose |
|---|---|---|
| 0 | md | Title, one-time setup, quota warning, what to expect |
| 1 | code | **CONFIG** — the only cell edited per run |
| 2 | md | §1 Verify GPU |
| 3 | code | `nvidia-smi` + torch device count; assert 2 visible GPUs |
| 4 | md | §2 Clone repository |
| 5 | code | Clone or pull, `GIT_ASKPASS` if private, `pip install -e .` |
| 6 | md | §3 Preflight |
| 7 | code | `scripts/preflight.py`, assert device separation |
| 8 | md | §4 Install cloudflared |
| 9 | code | `zstd` first, then Ollama installer, then cloudflared `.deb` |
| 10 | md | §5 Start Ollama |
| 11 | code | `Popen`, `pkill` stale, readiness poll, assert `/api/tags` responds 200 |
| 12 | md | §6 Pull generator. Note `num_ctx`/`think` need `/api/chat` |
| 13 | code | `ollama pull`; **then** assert the exact name appears in `/api/tags` |
| 14 | md | §7 Warm the retrieval stack |
| 15 | code | Load models, assert both on `cuda:1` **and** Ollama on GPU 0; report VRAM |
| 16 | md | §8 Build the index |
| 17 | code | Ingest and embed, or reuse an existing index |
| 18 | md | §9 Smoke test |
| 19 | code | `scripts/smoke_test.py`, assert one grounded answer with a citation |
| 20 | md | §10 Start the API |
| 21 | code | `Popen` uvicorn, log to `/tmp`, detect early exit |
| 22 | md | §11 Cloudflare Quick Tunnel |
| 23 | code | `Popen` cloudflared, scrape URL by regex, pin handle, write to `/tmp` |
| 24 | md | §12 Verify the public path |
| 25 | code | Retry loop through the tunnel for DNS lag, assert `/healthz` 200 |
| 26 | md | §13 Operator instructions |
| 27 | code | Print token, URL, and a **QR code of the URL**; print the SSE caveat; define `cleanup()` |
| 28 | md | Troubleshooting and quota recovery |

Cells 3, 7, 11, 13, 15, 19, and 25 all **assert**. A Kaggle session that fails
silently is far more expensive than one that fails loudly in cell 3.

### Two ordering details that matter

**Cells 11 → 13.** `/api/tags` is asserted *reachable* before the pull, and the
exact model name is asserted *after* it. Asserting a non-empty model list at cell
11 would fail on a clean run, because nothing has been pulled yet.

**Cell 15's assertion is about placement, not count.** Both models must land on
`cuda:1` while Ollama holds GPU 0. `preflight.py` cross-checks
`nvidia-smi --query-compute-apps` — resolving PIDs to physical GPU indices via
GPU UUIDs — so the assertion covers *which* GPU each process is on. Counting
visible devices is the weaker check and would pass even if both processes landed
on GPU 0.

## Workaround catalogue

Every entry is a specific failure already hit on Kaggle. Each looks like
unexplained flakiness until you know the cause.

| Symptom | Cause | Workaround |
|---|---|---|
| Ollama installer fails | Newer installers need `zstd` | `apt-get install -y zstd` **before** `curl ... \| sh` |
| Server dies when the cell ends | Shell `&` in a notebook cell is killed with the cell | `subprocess.Popen`, never `!cmd &` |
| Port already in use on re-run | Stale process from a previous cell | `pkill ollama` then `sleep(2)` before `Popen` |
| Connection refused right after start | Ollama not ready yet | Poll `GET /` up to 5 times with `sleep(3)`, break on 200 |
| `model not found` | Guessed tag rather than the real one | `ollama pull` first, **then** assert the name appears in `/api/tags` |
| `num_ctx` seems to do nothing | Used Ollama's OpenAI-compat route | `/v1/chat/completions` accepts neither `num_ctx` nor `think`; use `/api/chat`, and set `OLLAMA_CONTEXT_LENGTH` too |
| Reasoning output despite `think: false` | Some Qwen3 tags ignore the flag | `preflight.py` asks the model for a trivial reply and asserts no reasoning block; fall back to a `-no-think` variant |
| Dead kernels, no error, GPU shows P100 | **P100 is sm_60; modern PyTorch needs sm_70+** | Select T4 x2 in the UI, or set `"machine_shape": "NvidiaTeslaT4"` in `kernel-metadata.json`. `enable_gpu: true` alone lands on P100. |
| Exact search feels slow at 100k chunks | fp16 matmul fell back to a slow numpy loop | Upcast fp16→fp32 before the multiply; numpy has no fast `float16` BLAS path |
| Session switch feels like a hang | BM25 rebuilt synchronously on open | **Measured: ~0.7–0.9 ms/chunk, so 8.6s at 10k chunks.** Move session-open outside the query path — see [`measurements.md`](measurements.md) |
| Tunnel handle vanishes mid-session | `Popen` object garbage collected | `builtins.tunnel_process = p` |
| Tunnel URL not printed yet | cloudflared takes 5–15s | Regex the log line, 60 iterations, `bufsize=1` |
| URL detection failed entirely | Flaky scrape | Write to `/tmp`, allow manual paste, assert `https://` prefix |
| First request through tunnel times out | Quick Tunnel DNS propagates in 20–60s | 6 attempts, `sleep(15)` between |
| `403` on the tunnel root | Meaningless response from the edge | Only ever assert on a real route |
| App exited silently | Crash before first request | `poll()` check, then dump the `/tmp` log |
| Private clone prompts for credentials | No TTY in a notebook | `GIT_ASKPASS` script from a Kaggle Secret, `unlink` after |
| Optional model bootstrap fails | Partial download, OOM, bad asset | `try/except`, terminate Ollama, fall back to `mock` **visibly** |
| cloudflared version drifts mid-session | Auto-update between cells | `--no-autoupdate`, optional pinned URL + SHA256 |
| Run All kills the demo you wanted to use | Cleanup fired too eagerly | `CLEANUP_ON_RUN_ALL = False`, call `cleanup()` manually |
| Index gone next session | Session-scoped RAM or temp disk | Everything durable in `/kaggle/working` |
| `pip install` fights Kaggle's preinstalled torch | Version conflict | Do not pin torch; let the image's version stand |
| **Answer streams all at once** | **Quick Tunnel buffers GET responses** | **POST + `fetch` + `ReadableStream`, never `EventSource`** |

## Teardown

`scripts/cleanup.py` targets PID **files**, not process names, and escalates
SIGTERM → wait → SIGKILL on the process **group**. Both details are deliberate:

- `pkill ollama` kills the wrong thing when two sessions share a VM, and matches
  this script's own command line.
- Signalling only the parent leaves forked children holding the port, which then
  looks like "port already in use" on the next run.
- SIGTERM first means a process that flushes its write-through state (§6.2) gets
  to do so. SIGKILL alone means the log is empty exactly when you need it.

It is idempotent, and it removes a PID file only after successfully stopping the
process: a stale file claiming a dead PID is harmless, but a removed file for a
live PID means the next cleanup silently skips it.

```bash
python3 scripts/cleanup.py --dry-run   # what would be stopped
python3 scripts/cleanup.py             # stop it
```

## Session cost discipline

The index is built once and reused, so a steady-state session is short. Every
model-loading cell is gated on the CONFIG cell's profile, so a code-editing pass
never touches the network.

Order matters and is fixed: preflight and install before any download, models
before the index, index before the smoke test, smoke test before the tunnel.
Every later stage depends on everything above it, so failing early is cheaper.

Rules from §21: shut the session down when idle (an idle tunnel burns quota); use
the `None` accelerator for code editing; prefer retrieval-only tuning over
full-generation tuning; split the retrieval and generation matrices so phase gates
do not implicitly require generation.
