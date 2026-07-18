"""Shared bench-environment helpers (CPU info, git commit).

Used by ``scripts/bench_delta.py`` and ``scripts/scaling_probe.py`` to
record run metadata in their artifacts. Stdlib-only, no torch dependency.
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
