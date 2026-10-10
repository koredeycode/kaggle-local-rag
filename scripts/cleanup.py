#!/usr/bin/env python3
"""Cleanup: stop the processes a Kaggle session started (§19.1 cells 21/23/27).

The §19.2 workaround catalogue is blunt about this: "run All kills the demo you
wanted to use — cleanup fired too eagerly." So cleanup is a *manual* command with
`CLEANUP_ON_RUN_ALL = False` as the notebook default, not a cell that fires on
every Run All.

What it has to get right, because each of these is a documented Kaggle failure:

- **PID files, not name matching.** `pkill ollama` kills the wrong thing when two
  sessions share a VM, and it also matches this script's own command line.
- **`killpg`, not `kill`.** Ollama and uvicorn fork; signalling only the parent
  leaves the children holding the port, which then looks like "port already in
  use" on the next run.
- **SIGTERM, wait, then SIGKILL.** §19.2 logs are what you need when something
  refuses to die, and killing instantly means the log is empty.
- **Idempotent.** Cleaning up twice is normal, and a missing PID file is success.

    python3 scripts/cleanup.py                      # everything in /tmp
    python3 scripts/cleanup.py --pid-file /tmp/uvicorn.pid
    python3 scripts/cleanup.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

DEFAULT_PID_DIR = Path("/tmp")

#: Written by notebook cells 21 (uvicorn) and 23 (cloudflared), and by cell 11
#: for Ollama. One file per process so cleanup can target one without guessing.
KNOWN_PID_FILES = ("ollama.pid", "uvicorn.pid", "cloudflared.pid", "raglab.pid")

#: SIGTERM grace period. Long enough for a process to flush its write-through
#: state (§6.2), short enough not to eat the session.
GRACE_SECONDS = 5.0


def read_pid(path: Path) -> tuple[int | None, str]:
    """Return `(pid, note)`. A malformed file is reported, not fatal."""
    if not path.exists():
        return None, "no pid file"
    raw = path.read_text(encoding="utf-8").strip()
    try:
        return int(raw.split()[0]), ""
    except (ValueError, IndexError):
        return None, f"unparseable pid file: {raw[:40]!r}"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def process_group(pid: int) -> int | None:
    """The process group of `pid`, or None when it is its own group leader.

    `killpg` is the point of the exercise: a process started with `Popen` in a
    notebook is frequently a group leader, and its children would otherwise
    survive to hold the port.
    """
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return None
    return pgid if pgid != pid else None


def terminate(pid: int, *, grace: float, dry_run: bool) -> dict[str, object]:
    """SIGTERM the group, wait, then SIGKILL what is left."""
    result: dict[str, object] = {"pid": pid, "initial": "alive", "escalated": False}
    if dry_run:
        result["action"] = "dry-run, nothing signalled"
        return result

    pgid = process_group(pid)
    target_desc = f"pgid {pgid}" if pgid else "pid (no separate group)"

    try:
        if pgid:
            os.killpg(pgid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        result["term_sent_to"] = target_desc
    except ProcessLookupError:
        result["final"] = "already gone"
        return result
    except PermissionError:
        result["final"] = "permission denied; not ours to kill"
        result["ok"] = False
        return result

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not alive(pid):
            result["final"] = "terminated"
            return result
        time.sleep(0.1)

    result["escalated"] = True
    try:
        if pgid:
            os.killpg(pgid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
        result["final"] = "killed" if not alive(pid) else "SIGKILL did not take effect"
    except ProcessLookupError:
        result["final"] = "terminated during escalation"
    except PermissionError:
        result["final"] = "permission denied on SIGKILL"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid-dir", type=Path, default=DEFAULT_PID_DIR)
    ap.add_argument("--pid-file", type=Path, action="append", default=[])
    ap.add_argument("--grace", type=float, default=GRACE_SECONDS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    files = list(args.pid_file)
    if not files:
        files = [args.pid_dir / name for name in KNOWN_PID_FILES]

    report: list[dict[str, object]] = []
    for path in files:
        pid, note = read_pid(path)
        entry: dict[str, object] = {"file": str(path)}
        if pid is None:
            entry.update({"ok": True, "note": note, "skipped": True})
        elif not alive(pid):
            entry.update({"ok": True, "note": f"pid {pid} not running", "skipped": True})
        else:
            entry.update(terminate(pid, grace=args.grace, dry_run=args.dry_run))
            entry.setdefault("ok", not alive(pid))
        report.append(entry)
        if not args.dry_run and entry.get("ok") and not entry.get("skipped"):
            # Remove the pid file only on success: a stale file that claims a
            # dead pid is harmless, but a removed file for a live pid means the
            # next cleanup silently skips it.
            path.unlink(missing_ok=True)

    failures = [e for e in report if not e.get("ok")]
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("Cleanup (§19.1)")
        if args.dry_run:
            print("  DRY RUN — nothing was signalled")
        for entry in report:
            if entry.get("skipped"):
                print(f"  -    {entry['file']}: {entry.get('note')}")
                continue
            mark = "ok" if entry.get("ok") else "!!"
            extra = f" (escalated to SIGKILL)" if entry.get("escalated") else ""
            print(
                f"  {mark:>4} {entry['file']}: pid {entry.get('pid')} -> "
                f"{entry.get('final', entry.get('action', '?'))}{extra}"
            )
        print()
        if failures:
            print(f"  {len(failures)} process(es) could not be stopped.")
        elif args.dry_run:
            print("  Nothing to do.")
        else:
            print("  All known processes stopped.")

    if args.dry_run and not args.json:
        print("\nNOTE (§19.2): Run All fires every cell, so cleanup must NOT be wired")
        print("to it. Keep CLEANUP_ON_RUN_ALL = False and call this by hand.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
