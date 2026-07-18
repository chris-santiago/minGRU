"""Shared bench-environment helpers (CPU info, git commit).

Used by ``scripts/bench_delta.py`` and ``scripts/scaling_probe.py`` to
record run metadata in their artifacts. Stdlib-only, no torch dependency.

Evidence-pin idiom, by script (one place to look -- each script enforces
``torch==2.5.1`` differently, since each has a different call shape):

- ``experiments/hetero_lab.py``: opt-in ``--require-torch <version>`` CLI
  flag on ``run_arm`` -- exact-string match against ``torch.__version__``,
  ``SystemExit`` before any training step on mismatch, no row written.
  Opt-in (not a permanent hardcoded assert) so existing/future invocations
  without the flag are unaffected.
- ``scripts/scaling_probe.py``: hard assert inside each child subprocess
  (``_run_child``), before constructing any model -- the parent never
  imports torch at all, so the pin can only be enforced where torch is
  actually loaded.
- ``scripts/bench_delta.py``: hard assert in-process, at the top of
  ``run()``, before any timing -- this script is single-process (no
  child subprocess boundary to enforce the pin across).
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def cpu_info() -> str:
    """Best-effort CPU model string; falls back to ``platform.processor()``.

    ``platform.processor()`` alone is often uninformative (empty string,
    or a bare architecture tag like ``arm``) -- ``sysctl``/``/proc/cpuinfo``
    give the actual model name where available. Never raises: this is
    metadata for the artifact, not something the bench should depend on.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return out.stdout.strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def git_commit_sha() -> str | None:
    """Return the current git HEAD SHA, or None if unavailable.

    Best-effort: never raises. Used for artifact metadata / run
    identification.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None
