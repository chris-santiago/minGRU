"""Triton kernels for `min_gru.py`'s four scan functions.

Lazily imported by `min_gru._dispatch_scan` only (never at `min_gru.py`
import time); see that function's docstring and the design spec
(`.claude/output/specs/2026-07-16-triton-scan-kernels-design.md`) for the
full dispatch contract. Development is GPU-blind locally (this module
requires a CUDA device and Triton to do anything useful); correctness
and benchmarks land on a cloud GPU.

`min_gru.py`'s recorded lab evidence stays pinned to `torch==2.5.1` CPU
and never imports this module. This module targets `torch>=2.8` (mature
`torch.library.triton_op`/`wrap_triton` registration) and raises a clear
`ImportError` below that floor, rather than failing confusingly deep
inside kernel registration.

Task 1 (this stub): no kernels registered yet -- `SCAN_IMPLS` is empty,
so `min_gru._dispatch_scan` always falls back to eager under
`MINGRU_SCAN=auto`, and raises under `MINGRU_SCAN=triton` (no kernel to
run, never a silent downgrade). Later kernel tasks populate `SCAN_IMPLS`
with `torch.library.triton_op`-registered implementations.
"""

import torch

_MIN_TORCH = (2, 8)


def _torch_version_at_least(version: str, minimum: tuple[int, int]) -> bool:
    """Compare a ``torch.__version__`` string against a ``(major, minor)`` floor.

    Parameters
    ----------
    version : str
        A version string such as ``"2.5.1"`` or ``"2.8.0.dev20260101"``.
    minimum : tuple of int
        The ``(major, minor)`` floor to compare against.

    Returns
    -------
    bool
        Whether ``version``'s ``(major, minor)`` is ``>= minimum``.
    """
    major_minor = version.split("+")[0].split(".")[:2]
    return tuple(int(p) for p in major_minor) >= minimum


if not _torch_version_at_least(torch.__version__, _MIN_TORCH):
    raise ImportError(
        f"triton_scans requires torch>={'.'.join(map(str, _MIN_TORCH))} "
        f"(found {torch.__version__}); the recorded lab evidence pin "
        "(torch==2.5.1) runs the eager scan path only and never imports "
        "this module -- see min_gru._dispatch_scan."
    )


# Populated by later kernel tasks: scan-function name -> Triton-backed
# callable with the same signature/contract as its eager counterpart in
# min_gru.py (parallel_scan_log, linear_scan, matrix_scan,
# matrix_affine_scan). Empty in this stub -- no kernels registered yet.
SCAN_IMPLS: dict = {}


def available() -> bool | str:
    """Whether Triton scan kernels can run in this process.

    Returns
    -------
    bool or str
        ``True`` if a CUDA device is present and ``triton`` is
        importable. Otherwise a human-readable reason string naming why
        not (e.g. ``"CUDA not available"`` or ``"triton not importable:
        ..."``) -- callers should treat anything other than ``True`` as
        unavailable and use the string as the fallback/error reason (see
        ``min_gru._dispatch_scan``).
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        return f"triton not importable: {exc}"
    return True


if __name__ == "__main__":
    _status = available()
    if _status is not True:
        print(f"triton_scans selftest SKIPPED (loud): {_status}")
        raise SystemExit(0)
    print(
        "triton_scans: CUDA + Triton available, but no kernels are "
        "registered yet (Task 1 stub; SCAN_IMPLS is empty) -- nothing to "
        "test yet."
    )
    raise SystemExit(0)
