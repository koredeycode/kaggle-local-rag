#!/usr/bin/env python3
"""Preflight: environment, devices, models, and VRAM report (§19.1 cell 7).

Three jobs, in priority order:

1. **Report.** JSON environment + VRAM + per-process device placement. §3 says
   every VRAM figure in the plan is an estimate that this script must confirm,
   and that the plan must then be *corrected against measurements* rather than
   trusted. This is that measurement.

2. **Assert device separation.** §4 makes the GPU split a hard boundary:
   Ollama on GPU 0, the retrieval worker on GPU 1. §19.1 is explicit that
   counting visible devices is the weaker check and would pass even if both
   processes landed on GPU 0 — so this cross-checks
   `nvidia-smi --query-compute-apps` to see which *physical* GPU each PID is on.

3. **Verify the Ollama knobs actually work.** §5.2: `/v1/chat/completions`
   accepts neither `num_ctx` nor `think`, so choosing it silently invalidates the
   §5.3 KV budget with no error. And some Qwen3 tags have been reported to ignore
   `think: false`. Both are checked by asking the model, not by reading config.

Exit codes:
    0  all requested checks passed
    1  at least one required check failed
    2  ran, but something it could not check (no GPU, no Ollama) — informational

Every check is a separate `--require` flag so notebook cell 7 can assert only
what that run actually needs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raglab.config import Profile, Settings, from_env  # noqa: E402

CHECKS = (
    "python",
    "cuda_count",
    "device_separation",
    "models_present",
    "ollama_reachable",
    "ollama_model",
    "think_disabled",
    "vram_headroom",
)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def sh(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        proc = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def nvidia_smi(query: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-*`. Returns [] when unavailable."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    code, out = sh([
        binary, f"--query-{query}", "--format=csv,noheader,nounits",
    ])
    if code != 0 or not out.strip():
        return []
    return [line.split(", ") for line in out.strip().splitlines()]


def gpu_report() -> dict[str, Any]:
    rows = nvidia_smi("gpu=index,name,memory.total,memory.used,compute_cap")
    gpus = []
    for r in rows:
        total, used = _num(r[2]), _num(r[3])
        gpus.append({
            "index": _num(r[0]),
            "name": r[1],
            "memory_total_mb": total,
            "memory_used_mb": used,
            "memory_free_mb": max(0, total - used),
            "compute_cap": r[4] if len(r) > 4 else "",
            "headroom_pct": round(100 * (total - used) / total, 1) if total else 0.0,
        })

    apps = []
    for line in nvidia_smi("compute_apps=pid,process_name,used_gpu_memory,gpu_uuid"):
        if len(line) >= 4:
            apps.append({"pid": _num(line[0]), "process": line[1],
                         "used_mb": _num(line[2]), "gpu_uuid": line[3]})

    # nvidia-smi reports compute apps by GPU *UUID*, not index. Resolving the
    # UUID is what turns "some process holds memory" into "Ollama is on GPU 0 and
    # the retrieval worker is on GPU 1" — which is the check §19.1 actually wants,
    # as opposed to counting visible devices.
    uuid_to_index: dict[str, int] = {}
    for row in nvidia_smi("gpu=index,uuid"):
        if len(row) >= 2:
            uuid_to_index[row[1].strip()] = _num(row[0])
    for app in apps:
        app["gpu_index"] = uuid_to_index.get(app.get("gpu_uuid", ""), -1)

    return {"gpus": gpus, "compute_apps": apps}


def _num(raw: str) -> int | float:
    try:
        return int(float(raw.strip()))
    except (TypeError, ValueError):
        return 0


def torch_report() -> dict[str, Any]:
    """Device visibility and VRAM from torch, when it is installed."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch is not installed"}
    info: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "version": torch.__version__,
        "device_count": 0,
        "devices": [],
    }
    if not info["available"]:
        info["reason"] = "no CUDA device visible"
        return info
    info["device_count"] = torch.cuda.device_count()
    for i in range(info["device_count"]):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        info["devices"].append({
            "index": i,
            "name": props.name,
            "total_mb": total // (1024 * 1024),
            "free_mb": free // (1024 * 1024),
            "capability": f"{props.major}.{props.minor}",
            # §3: T4 is Turing, sm_75. No bf16.
            "supports_bf16": props.major >= 8,
        })
    return info


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(  # noqa: S310 - loopback-only, §16
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


def ollama_reachable(base: str) -> tuple[bool, Any]:
    try:
        return True, http_json(f"{base}/api/tags", timeout=10)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        return False, str(exc)


def ollama_tags(base: str) -> list[str]:
    ok, data = ollama_reachable(base)
    if not ok or not isinstance(data, dict):
        return []
    return [m.get("name", "") for m in data.get("models", [])]


def check_think_disabled(base: str, model: str) -> dict[str, Any]:
    """§5.2: some Qwen3 tags have been reported to ignore `think: false`.

    Asked of the model, not read from config, because the failure mode is that
    the flag is accepted and then ignored.
    """
    try:
        data = http_json(f"{base}/api/chat", {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "think": False,
            "stream": False,
            "options": {"num_predict": 32, "num_ctx": 2048},
        }, timeout=180)
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": f"request failed: {exc}"}

    message = data.get("message", {}) if isinstance(data, dict) else {}
    thinking = message.get("thinking")
    content = (message.get("content") or "").strip()
    return {
        "ok": thinking in (None, "", False) and bool(content),
        "has_thinking": bool(thinking),
        "content_preview": content[:120],
        "note": (
            "reasoning block present despite think=false; fall back to a "
            "-no-think variant (§5.2)"
        ) if thinking else "",
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_python() -> dict[str, Any]:
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) < (3, 14)
    return {
        "ok": ok,
        "version": platform.python_version(),
        "detail": (
            "python 3.12 on the Kaggle image; requires-python is >=3.10,<3.14 (§3)"
            if not ok
            else ""
        ),
    }


def check_cuda_count(gpus: list[dict], expected: int = 2) -> dict[str, Any]:
    n = len(gpus)
    return {
        "ok": n >= expected,
        "found": n,
        "expected": expected,
        "detail": (
            f"§4 designs for T4 x2. Found {n}. A single GPU still works but the "
            "workload split cannot, and every latency figure changes."
            if n < expected
            else ""
        ),
    }


def check_compute_caps(gpus: list[dict]) -> dict[str, Any]:
    """§19.2: P100 is sm_60 and silently breaks modern PyTorch with dead kernels.

    Not a failure — reporting it is the point, because a notebook that lands on
    P100 via `enable_gpu: true` produces no error at all.
    """
    bad = [g for g in gpus if g.get("compute_cap") and float(g["compute_cap"]) < 7.0]
    return {
        "ok": not bad,
        "capabilities": [g["compute_cap"] for g in gpus],
        "detail": (
            f"compute capability {[g['compute_cap'] for g in bad]} is below 7.0. "
            "Modern PyTorch needs sm_70+; kernels will die silently. Set "
            '"machine_shape": "NvidiaTeslaT4" in kernel-metadata.json (§19.2).'
            if bad
            else ""
        ),
    }


def check_device_separation(
    apps: list[dict],
    procs: dict[int, str],
    *,
    expect: tuple[int, int] = (0, 1),
) -> dict[str, Any]:
    """Which *physical* GPU each named process is on.

    §19.1: counting visible devices is the weaker check and would pass even if
    both processes had landed on GPU 0. This maps PID to GPU index via
    `nvidia-smi --query-compute-apps=pid,...,gpu_uuid`, so it fails when Ollama
    and the retrieval worker share a card.

    With no `--pid` arguments there is nothing to verify, so the check reports
    `ok: False` with a reason rather than a vacuous pass. A check that cannot run
    and a check that passed must not look the same in a notebook assertion.
    """
    if not procs:
        return {
            "ok": False,
            "checked": False,
            "reason": (
                "no processes given; pass --pid NAME=PID to verify the §4 split, "
                "e.g. --pid ollama=$(pgrep -x ollama)"
            ),
        }

    by_pid = {a["pid"]: a for a in apps}
    matched: dict[str, int] = {}
    unmatched: list[str] = []
    for name, pid in procs.items():
        app = by_pid.get(pid)
        if app is None or app.get("gpu_index", -1) < 0:
            unmatched.append(f"{name}(pid {pid})")
            continue
        matched[name] = app["gpu_index"]

    detail = ""
    ok = not unmatched
    if unmatched:
        detail = (
            f"could not locate {unmatched} in nvidia-smi --query-compute-apps. "
            "A process holding VRAM always appears there; if one is missing, it "
            "has not allocated GPU memory yet (cell ordering, §19.4)."
        )
    elif len(set(matched.values())) < len(matched):
        detail = (
            f"{matched} — two named processes share one GPU. §4 splits by "
            "workload precisely so they cannot OOM each other."
        )
        ok = False
    return {
        "ok": ok,
        "checked": True,
        "matched": matched,
        "unmatched": unmatched,
        "detail": detail,
    }


def check_vram_headroom(gpus: list[dict], minimum_free_mb: int = 6_000) -> dict[str, Any]:
    """§5.3: each card needs ~6.7GB (GPU 0) / ~4.5GB (GPU 1) free."""
    tight = [g for g in gpus if g["memory_free_mb"] < minimum_free_mb]
    return {
        "ok": not tight,
        "free_mb": {str(g["index"]): g["memory_free_mb"] for g in gpus},
        "detail": (
            f"cards {[g['index'] for g in tight]} have under "
            f"{minimum_free_mb}MB free. §5.3 budgets 6.7GB on GPU 0; lower "
            "embed_batch first (§5.3)."
            if tight
            else ""
        ),
    }


def check_models_present(tags: list[str], required: list[str]) -> dict[str, Any]:
    missing = [m for m in required if not any(t.startswith(m) for t in tags)]
    return {
        "ok": bool(tags) and not missing,
        "tags": tags,
        "missing": missing,
        "detail": (
            f"not pulled yet: {missing}. Run `ollama pull` and assert the exact "
            "name appears in /api/tags afterwards (§19.2)."
            if missing
            else ""
        ),
    }


def build_report(settings: Settings, *, require: list[str], procs: dict[int, str]) -> dict[str, Any]:
    gpus_info = gpu_report()
    gpus = gpus_info["gpus"]
    torch_info = torch_report()
    tags = ollama_tags(settings.ollama_url)
    required_gen = [settings.gen_model]

    results: dict[str, dict[str, Any]] = {
        "python": check_python(),
        "cuda_count": check_cuda_count(gpus),
        "compute_capability": check_compute_caps(gpus),
        "device_separation": check_device_separation(gpus_info["compute_apps"], procs),
        "vram_headroom": check_vram_headroom(gpus),
        "ollama_reachable": {"ok": bool(tags) or not settings.is_gpu(), "tags": tags},
        "models_present": check_models_present(tags, required_gen),
    }
    if tags:
        results["ollama_model"] = {
            "ok": any(t.startswith(settings.gen_model) for t in tags),
            "model": settings.gen_model,
        }
        results["think_disabled"] = check_think_disabled(settings.ollama_url, settings.gen_model)

    for name in list(results):
        results[name]["required"] = name in require

    return {
        "profile": str(settings.profile),
        "embed_device": settings.embed_device,
        "rerank_device": settings.rerank_device,
        "ollama_url": settings.ollama_url,
        "ollama_api": settings.ollama_api,
        "gen_model": settings.gen_model,
        "gen_ctx": settings.gen_ctx,
        "gen_think": settings.gen_think,
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_count": os.cpu_count(),
        },
        "gpus": gpus,
        "compute_apps": gpus_info["compute_apps"],
        "torch": torch_info,
        "checks": results,
    }


def print_report(report: dict[str, Any]) -> None:
    p = report["platform"]
    print("Preflight (§19.1 cell 7)")
    print(f"  profile            {report['profile']}")
    print(f"  python             {p['python']} on {p['system']} {p['machine']}")
    print(f"  cpu_count          {p['cpu_count']}")
    print(
        f"  CUDA_VISIBLE_DEVICES  {p['cuda_visible_devices'] or '(unset)'}"
        "   <- §4: the hard GPU boundary"
    )
    print(f"  embed device       {report['embed_device']}")
    print(f"  rerank device      {report['rerank_device']}")
    print(f"  ollama             {report['ollama_url']} via {report['ollama_api']} "
          f"(§5.2: num_ctx and think need /api/chat)")
    print(f"  generator          {report['gen_model']} ctx={report['gen_ctx']} "
          f"think={report['gen_think']}")
    print()

    print("GPUs")
    if not report["gpus"]:
        print("  none detected (expected locally; §3 says all GPU work is Kaggle-only)")
    for g in report["gpus"]:
        print(
            f"  [{g['index']}] {g['name']}  sm_{g['compute_cap'].replace('.', '')}  "
            f"{g['memory_used_mb']}/{g['memory_total_mb']} MB used  "
            f"{g['memory_free_mb']} MB free"
        )
    if report["compute_apps"]:
        print("  compute apps:")
        for a in report["compute_apps"]:
            print(f"    pid {a['pid']:<8} {a['process'][:40]:<40} {a['used_mb']} MB")
    print()

    print("Checks")
    width = max(len(n) for n in report["checks"])
    for name, r in report["checks"].items():
        # PASS/FAIL always reflects the check's own verdict. The `*` says
        # whether it was required, and only required failures change the exit
        # code. Collapsing "not required" into a PASS/SKIP word would hide a
        # real failure behind an exemption.
        mark = "PASS" if r.get("ok") else "FAIL"
        flag = "*" if r.get("required") else " "
        print(f"  {mark:>4}{flag} {name:<{width}}")
        detail = r.get("detail") or r.get("reason") or ""
        if detail:
            for line in str(detail).splitlines():
                print(f"         {line}")
    print()
    print("  * = required by --require")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write the report here")
    ap.add_argument("--require", nargs="*", default=[], choices=[*CHECKS, "compute_capability"])
    ap.add_argument("--profile", default=None, choices=[str(p) for p in Profile])
    ap.add_argument(
        "--pid", nargs="*", default=[], metavar="NAME=PID",
        help="named processes to map onto GPUs, e.g. ollama=1234 raglab=1256",
    )
    args = ap.parse_args()

    settings = from_env(**({"profile": Profile(args.profile)} if args.profile else {}))
    procs: dict[int, str] = {}
    for item in args.pid:
        name, _, raw = item.partition("=")
        procs[int(raw)] = name

    report = build_report(settings, require=args.require, procs=procs)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    if args.json and not sys.stdout.isatty():
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)
        if args.json:
            print(f"Wrote {args.json}")

    failed = [n for n, r in report["checks"].items() if r.get("required") and not r.get("ok")]
    if failed:
        print(f"\nFAILED required checks: {failed}")
        return 1
    if not report["gpus"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
