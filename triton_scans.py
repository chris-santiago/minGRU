"""Evidence driver for the packaged :mod:`mingru.triton_scans` module.

The Triton scan kernels live in ``src/mingru/triton_scans.py`` (import name
``mingru.triton_scans``). This repo-root file is a thin *evidence driver*,
not the library. It exists so the recorded GPU-conformance command keeps
working verbatim from a repo checkout with no package install:

* it puts ``src/`` on ``sys.path`` and re-exports the packaged module's
  attributes with object identity, so ``import triton_scans`` (in
  ``scripts/bench_scans.py``) resolves here unchanged; and
* its ``__main__`` block is the module's relocated selftest suite, so
  ``python triton_scans.py`` preserves the loud-skip / GPU-parity behavior.

Never shipped in the wheel (src-layout excludes the repo root). See
``src/mingru/triton_scans.py`` for the library itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mingru import triton_scans as _triton_scans

# Re-export the packaged module's attributes with object identity: attribute
# access on this driver delegates to the packaged module, so both the public
# API and the private parity runners reached by name (e.g. by
# scripts/bench_scans.py) resolve to the packaged objects.
__all__ = list(_triton_scans.__all__)


def __getattr__(name):
    return getattr(_triton_scans, name)


if __name__ == "__main__":
    # Import header (driver-contract adaptation (a)): bind every free name the
    # relocated selftest block uses -- the availability probe, the module-level
    # parity runners and the CPU-lockstep reference, plus torch/F -- from the
    # packaged module. (The block's own ``import min_gru`` resolves to this
    # repo-root evidence driver, which re-exports ``log_g`` and
    # ``parallel_scan_log``.)
    import torch
    import torch.nn.functional as F

    from mingru.triton_scans import (
        available,
        parallel_scan_log_recompute,
        _run_angle_fused_parity,
        _run_forward_parity,
        _run_grad_parity,
    )

    # =========================================================================
    # CPU lockstep guard: `parallel_scan_log_recompute` (the backward's
    # autograd-through-recomputation formula, module-level and Triton-free)
    # vs `min_gru.parallel_scan_log` (the frozen eager reference). Runs
    # ALWAYS, before the CUDA/Triton availability check below -- unlike the
    # rest of this file's selftest suite, this section needs no GPU/Triton,
    # so it is no longer fully vacuous on a CPU-only/no-Triton machine: it
    # catches head-math drift between the backward's recompute path and the
    # eager reference on ordinary CI (and the GPU-less Phase-4 wheel CI),
    # not only the GPU-only grad-parity selftest (`_run_grad_parity`, which
    # additionally exercises the actual Triton-dispatched backward).
    # =========================================================================
    import min_gru as _min_gru_lockstep

    torch.manual_seed(2026)
    _B, _T, _D = 3, 17, 8
    _k_pre = torch.randn(_B, _T, _D)
    _log_coeffs = -F.softplus(_k_pre)
    _log_z = -F.softplus(-_k_pre)
    _log_tilde_h = _min_gru_lockstep.log_g(torch.randn(_B, _T, _D))
    _log_h0 = torch.randn(_B, 1, _D) * 0.1
    _log_values = torch.cat([_log_h0, _log_z + _log_tilde_h], dim=1)

    _h_recompute = parallel_scan_log_recompute(_log_coeffs, _log_values)
    _h_eager = _min_gru_lockstep.parallel_scan_log(_log_coeffs, _log_values)
    _lockstep_err = (_h_recompute - _h_eager).abs().max().item()
    assert _lockstep_err < 1e-6, (
        "parallel_scan_log_recompute diverges from min_gru.parallel_scan_log "
        f"(max_abs={_lockstep_err:.3e}) -- MAINTENANCE lockstep broken"
    )
    print(
        "CPU lockstep: parallel_scan_log_recompute vs min_gru.parallel_scan_log "
        f"(max_abs={_lockstep_err:.3e}): ok"
    )

    _status = available()
    if _status is not True:
        print(f"triton_scans selftest SKIPPED (loud): {_status}")
        raise SystemExit(0)
    _fwd = _run_forward_parity()
    print()
    _bwd = _run_grad_parity()
    print()
    _angle = _run_angle_fused_parity()
    raise SystemExit(_fwd or _bwd or _angle)
