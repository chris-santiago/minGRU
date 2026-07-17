"""Equivalence + benchmark driver for the Triton scan kernels (Task 6).

Consumes the selftest suite already implemented in ``triton_scans.py``
(Tasks 2-5) and the four scan functions dispatched from ``min_gru.py``.
This script does not invent any new parity logic or tolerance -- it only
orchestrates what already exists and adds timing/memory measurement:

``--check``
    Runs the full equivalence suite (``triton_scans._run_forward_parity``,
    ``_run_grad_parity``, ``_run_angle_fused_parity``) by calling those
    module-level runners directly. Skips LOUDLY (prints the reason, exits
    0) when CUDA/Triton are unavailable, or when the installed torch is
    below ``triton_scans``'s ``torch>=2.8`` floor -- a vacuous pass would
    be a bug. This is the parity half of the design spec's "done bar"
    (``.claude/output/specs/2026-07-16-triton-scan-kernels-design.md``
    §9.1-9.2). Also persists a parity-conformance artifact (one row per
    GATED case -- op/mixer-case, shape, dtype, direction, gate, measured
    max abs/rel error, pass) to ``experiments/bench/scan_parity.json`` /
    ``.md``, by passing an accumulator list into each runner's optional
    ``collect`` parameter -- the runners already measure this per case
    (printed on failure); this only persists what they already measured,
    including for passing cases.

``--bench``
    Times the four scan functions (``linear_scan``, ``matrix_scan``,
    ``matrix_affine_scan``, ``parallel_scan_log``) across three impls --
    ``triton`` (``MINGRU_SCAN=triton``), ``eager`` (``MINGRU_SCAN=eager``
    on CUDA tensors), and ``torch_compile`` (``torch.compile`` over the
    same eager path) -- at two shapes (lab: ``B=128, T=64, d=64, k=8``;
    long-T: ``B=16, T=1024, d=64, k=8``) and two directions (``fwd``,
    ``fwdbwd``). Emits JSON rows matching the spec §6 schema exactly
    (``{op, impl, shape, dtype, direction, us_per_iter}``) plus a markdown
    table, both under ``experiments/bench/`` (see ``_OUT_DIR``).

``--memory``
    Records ``torch.cuda.max_memory_allocated`` for one forward+backward
    of ``GivensMinGRU`` at the lab shape, comparing the angle-fused
    Triton path (``MINGRU_SCAN=triton``) against the Phase-1 generic
    ``matrix_affine_scan`` path (``MINGRU_SCAN=eager``) -- spec §9.2's
    memory acceptance criterion (the angle-fused backward never
    materializes the k x k transitions the generic path does).

Every mode degrades gracefully without CUDA (a clear, non-crashing
message) rather than raising a bare traceback; see ``_gpu_status``.

Input generation reuses ``triton_scans``'s own parity-sweep builders
(``_linear_scan_inputs``, ``_matrix_scan_inputs``,
``_matrix_affine_scan_inputs``, ``_log_space_inputs``) rather than
reimplementing shape construction -- those are the same tensors the
equivalence suite already validates against, so the benchmark measures
the same inputs whose correctness ``--check`` establishes. They are
private (underscore-prefixed) helpers in ``triton_scans.py``; reaching
into them from this script is the sanctioned reuse path (this task's
brief), not an ad hoc import of internals.

Import-path note: running ``python scripts/bench_scans.py`` puts this
file's own directory (``scripts/``), not the caller's CWD, on
``sys.path[0]`` -- a plain ``import min_gru`` fails even when invoked
from the repo root, purely as an artifact of Python's script-mode import
resolution. This script therefore inserts its own repo root (derived
from ``__file__``, not the CWD) onto ``sys.path`` before importing --
the one minimal, documented ``sys.path`` insertion the brief authorizes
for a script entry point. If the import still fails after that (e.g. this
file was copied out of a minGRU checkout), it prints a clear message
instead of a bare ``ModuleNotFoundError`` traceback. That allowance binds
this script only -- ``min_gru.py``/``triton_scans.py`` remain free of any
``sys.path`` handling per the design spec's library-code constraint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

# `python scripts/bench_scans.py` puts THIS file's directory (scripts/), not
# the caller's CWD, on sys.path[0] -- a plain `import min_gru` fails even
# when invoked from the repo root, purely as an artifact of how Python
# resolves script-mode imports. The brief's "no sys.path manipulation"
# constraint binds library code (min_gru.py/triton_scans.py); it explicitly
# permits a script entry point to resolve its own repo root when a plain
# import is insufficient, which this is -- so this is the one minimal,
# documented insertion, not a general-purpose sys.path hack.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import min_gru
except ImportError as exc:  # pragma: no cover - exercised only from a bad repo layout
    print(
        "bench_scans.py: `import min_gru` failed even after adding the repo "
        f"root ({_REPO_ROOT}) to sys.path -- is this script still inside a "
        f"minGRU checkout, next to min_gru.py? ({exc})",
        file=sys.stderr,
    )
    raise SystemExit(2)

# triton_scans.py raises ImportError below its own torch>=2.8 floor (see its
# module docstring); it does NOT require CUDA or an installed `triton`
# package to import successfully -- `_HAS_TRITON` degrades gracefully
# inside it. So this import can fail only on the torch-version floor; every
# CUDA/Triton-availability check happens later via `triton_scans.available()`.
try:
    import triton_scans

    _TRITON_SCANS_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    triton_scans = None  # type: ignore[assignment]
    _TRITON_SCANS_IMPORT_ERROR = exc


# Benchmark artifacts land here (deterministic, stable filenames so Task 7
# can commit them without inventing paths). Chosen over `scripts/out/`
# because `experiments/` is already this repo's home for recorded
# evidence/results files (`experiments/lab_results.jsonl`, `EXPERIMENTS.md`);
# `bench/` keeps these kernel-throughput artifacts visibly distinct from the
# probe-evidence ledger those files hold. Anchored to `_REPO_ROOT` (not a
# bare relative path) so the artifact location stays deterministic
# regardless of the invocation CWD -- the same CWD-independence the
# `_REPO_ROOT` sys.path insertion above already gives the imports.
_OUT_DIR = _REPO_ROOT / "experiments" / "bench"
_BENCH_JSON = _OUT_DIR / "scan_bench.json"
_BENCH_MD = _OUT_DIR / "scan_bench.md"
_MEMORY_JSON = _OUT_DIR / "scan_memory.json"
_MEMORY_MD = _OUT_DIR / "scan_memory.md"
_PARITY_JSON = _OUT_DIR / "scan_parity.json"
_PARITY_MD = _OUT_DIR / "scan_parity.md"

# Two bench shapes (spec §9.3 / this task's brief): lab and long-T. Only
# B, T vary between them; d (elementwise channel width / block-mixer hidden
# size) and k (matrix_affine_scan's block size) are held at the lab values
# for both, per the brief's shape statement ("lab B=128,T=64,d=64,k=8 and
# long-T B=16,T=1024"). `warmup`/`reps` are tuned down for the long-T shape
# since each iteration there is far more expensive (T=1024 vs T=64).
_LAB_SHAPE = {"label": "lab", "B": 128, "T": 64, "d": 64, "k": 8, "warmup": 10, "reps": 30}
_LONGT_SHAPE = {"label": "long-T", "B": 16, "T": 1024, "d": 64, "k": 8, "warmup": 5, "reps": 10}
_BENCH_SHAPES = (_LAB_SHAPE, _LONGT_SHAPE)

_OPS = ("linear_scan", "matrix_scan", "matrix_affine_scan", "parallel_scan_log")
_IMPLS = ("triton", "eager", "torch_compile")
_DIRECTIONS = ("fwd", "fwdbwd")
_DTYPE_LABEL = "float32"  # every op accumulates in fp32 (spec §7); bench inputs match


@dataclass(frozen=True)
class _OpConfig:
    """One scan op's input builder and JSON-schema shape-dict builder.

    ``build`` mirrors the op's own ``triton_scans`` parity-sweep helper
    (same tensors the equivalence suite validates); ``shape`` renders the
    spec §6 schema's ``shape`` field for this op ("elementwise" ops key on
    ``D``, block ops key on ``n``/``k``/``v``).
    """

    build: Callable[[int, int, str], tuple[torch.Tensor, ...]]
    shape: Callable[[int, int], dict]


def _op_configs(d: int, k: int) -> dict[str, _OpConfig]:
    """Build the per-op input/shape config dict for one (d, k) bench shape.

    ``n``/``v`` for the block ops mirror how each op is actually invoked by
    a real mixer in ``min_gru.py`` (not an arbitrary synthetic width):
    ``matrix_scan`` is ``RotationMinGRU``'s k=2 path (``n = d // 2``
    blocks), and ``matrix_affine_scan`` is ``GivensMinGRU``'s path with
    ``v = 1`` (the state viewed as a k x 1 column, exactly
    ``b.unsqueeze(-1)`` in ``GivensMinGRU._coeffs``'s caller).
    """
    return {
        "parallel_scan_log": _OpConfig(
            build=lambda B, T, device: triton_scans._log_space_inputs(B, T, d, device),
            shape=lambda B, T: {"B": B, "T": T, "D": d},
        ),
        "linear_scan": _OpConfig(
            build=lambda B, T, device: triton_scans._linear_scan_inputs(B, T, d, device),
            shape=lambda B, T: {"B": B, "T": T, "D": d},
        ),
        "matrix_scan": _OpConfig(
            build=lambda B, T, device: triton_scans._matrix_scan_inputs(B, T, d // 2, device),
            shape=lambda B, T: {"B": B, "T": T, "n": d // 2, "k": 2, "v": 1},
        ),
        "matrix_affine_scan": _OpConfig(
            build=lambda B, T, device: triton_scans._matrix_affine_scan_inputs(
                B, T, d // k, k, 1, device
            ),
            shape=lambda B, T: {"B": B, "T": T, "n": d // k, "k": k, "v": 1},
        ),
    }


def _gpu_status() -> bool | str:
    """Whether this process can run any CUDA-only mode; else a reason string."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    return True


def _loud_skip(reason: str) -> int:
    """Print a loud, non-vacuous skip message and return the exit code 0."""
    print(f"bench_scans.py SKIPPED (loud): {reason}")
    return 0


# --- --check ------------------------------------------------------------


def _nvidia_driver_version() -> str | None:
    """Best-effort NVIDIA driver version string, or ``None`` if unavailable.

    No torch API exposes this (``torch.version.cuda`` is the CUDA
    runtime/toolkit version, not the driver), so this shells out to
    ``nvidia-smi``. Never raises: a driver string is convenience metadata
    for the parity artifact, not something ``--check`` should depend on to
    run -- a missing/failing ``nvidia-smi`` degrades to ``None``.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return None


def _triton_version() -> str | None:
    """The installed ``triton`` package version, or ``None`` if not importable."""
    try:
        import triton

        return triton.__version__
    except ImportError:
        return None


def _git_commit_sha() -> str | None:
    """The current commit SHA of this checkout, or ``None`` if unavailable."""
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


def _parity_counts(rows: list[dict]) -> dict:
    """Pass/fail counts overall and per ``direction`` (fwd/grad)."""
    by_direction: dict[str, dict] = {}
    for direction in ("fwd", "grad"):
        d_rows = [r for r in rows if r["direction"] == direction]
        by_direction[direction] = {
            "total": len(d_rows),
            "passed": sum(1 for r in d_rows if r["pass"]),
            "failed": sum(1 for r in d_rows if not r["pass"]),
        }
    return {
        "total": len(rows),
        "passed": sum(1 for r in rows if r["pass"]),
        "failed": sum(1 for r in rows if not r["pass"]),
        "by_direction": by_direction,
    }


def _render_parity_markdown(rows: list[dict], meta: dict) -> str:
    """One markdown table: one row per gated parity case, plus the meta block."""
    lines = [
        "# Scan kernel parity conformance",
        "",
        f"torch {meta['torch_version']}, triton {meta['triton_version']}, "
        f"device {meta['gpu_name']}, driver {meta['driver_version']}, "
        f"commit {meta['git_commit']}, generated {meta['timestamp']}.",
        "",
        "| op | shape | dtype | direction | gate atol | gate rtol | max abs err | max rel err | pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        abs_str = f"{row['max_abs_err']:.2e}" if row["max_abs_err"] is not None else "n/a"
        rel_str = f"{row['max_rel_err']:.2e}" if row["max_rel_err"] is not None else "n/a"
        lines.append(
            f"| {row['op']} | {row['shape']} | {row['dtype']} | {row['direction']} | "
            f"{row['gate_atol']:.0e} | {row['gate_rtol']:.0e} | {abs_str} | {rel_str} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append(f"Counts: {json.dumps(meta['counts'])}")
    return "\n".join(lines) + "\n"


def _write_parity_artifacts(rows: list[dict]) -> None:
    """Write the parity-conformance JSON + markdown artifacts (Task 7 follow-up).

    Always writes (even if some/all cases failed) -- the conformance
    artifact's job is to document exactly what was measured, gate, and
    outcome, for every gated case, not only the passing ones.
    """
    meta = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver_version": _nvidia_driver_version(),
        "torch_version": torch.__version__,
        "triton_version": _triton_version(),
        "git_commit": _git_commit_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": _parity_counts(rows),
    }
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _PARITY_JSON.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    _PARITY_MD.write_text(_render_parity_markdown(rows, meta))
    print(f"wrote {_PARITY_JSON} and {_PARITY_MD} ({len(rows)} row(s))")


def run_check() -> int:
    """Run the full equivalence suite via ``triton_scans``'s own runners.

    Mirrors ``triton_scans.py``'s own ``__main__`` block exactly (same
    skip convention, same three runners, same exit-code combination) --
    this does not reimplement any parity sweep or tolerance. Additionally
    passes a shared ``collect`` accumulator list into each runner (an
    optional parameter on all three, defaulting to ``None`` so
    ``triton_scans.py``'s own ``__main__`` invocation is unaffected) and
    persists the collected rows as the parity-conformance artifact.
    """
    if triton_scans is None:
        return _loud_skip(
            f"triton_scans not importable: {_TRITON_SCANS_IMPORT_ERROR}"
        )
    status = triton_scans.available()
    if status is not True:
        return _loud_skip(str(status))

    parity_rows: list[dict] = []
    fwd = triton_scans._run_forward_parity(collect=parity_rows)
    print()
    grad = triton_scans._run_grad_parity(collect=parity_rows)
    print()
    angle = triton_scans._run_angle_fused_parity(collect=parity_rows)

    _write_parity_artifacts(parity_rows)

    return fwd or grad or angle


# --- --bench --------------------------------------------------------------


def _get_impl_fn(
    impl: str, op_name: str, shape_label: str, compiled_cache: dict
) -> Callable:
    """Resolve the callable for one impl, setting ``MINGRU_SCAN`` as a side effect.

    ``triton``/``eager`` call the scan function itself (its own
    ``MINGRU_SCAN`` dispatch seam picks the path); ``torch_compile`` wraps
    the same eager-dispatched function in ``torch.compile`` -- so all three
    impls exercise the exact same public function, never a hand-forked
    copy of the scan math.

    ``compiled_cache`` is keyed by ``(op_name, shape_label)``, not
    ``op_name`` alone: the two bench shapes have different (B, T, ...)
    tensor shapes, and reusing one shape's compiled graph for the other
    would force a recompile mid-sweep that dynamo's automatic-dynamic
    heuristic is likely to promote to a dynamic-shape guard -- silently
    turning the long-T ``torch_compile`` row into a measurement of a
    dynamic-guarded graph instead of a shape-specialized one. A fresh
    compile (+ warmup) per shape avoids that.
    """
    base = getattr(min_gru, op_name)
    if impl == "triton":
        os.environ["MINGRU_SCAN"] = "triton"
        return base
    if impl == "eager":
        os.environ["MINGRU_SCAN"] = "eager"
        return base
    if impl == "torch_compile":
        os.environ["MINGRU_SCAN"] = "eager"
        cache_key = (op_name, shape_label)
        if cache_key not in compiled_cache:
            compiled_cache[cache_key] = torch.compile(base)
        return compiled_cache[cache_key]
    raise ValueError(f"unknown impl {impl!r}")


def _time_forward(fn: Callable, args: tuple, warmup: int, reps: int) -> float:
    """CUDA-event-timed forward-only pass; returns microseconds per iteration.

    Runs under ``torch.no_grad()`` -- a forward-only throughput number
    should not carry autograd bookkeeping overhead, matching how an
    inference-only caller would run this same function.
    """
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(reps):
            fn(*args)
        end_evt.record()
        torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) * 1000.0 / reps  # ms -> us


def _fwd_bwd_once(fn: Callable, leaves: tuple[torch.Tensor, ...]) -> None:
    """One forward+backward iteration: zero grads, forward, sum-loss, backward.

    ``leaves`` are reused across iterations (zeroing ``.grad`` before each
    backward, the same zero_grad-forward-backward shape a real training
    step takes) rather than rebuilding fresh random tensors every
    iteration, so the timed region measures only the scan op's own
    forward+backward cost, not tensor allocation/randn noise.
    """
    for leaf in leaves:
        leaf.grad = None
    out = fn(*leaves)
    outs = out if isinstance(out, tuple) else (out,)
    loss = sum(o.float().sum() for o in outs)
    loss.backward()


def _time_forward_backward(
    fn: Callable, leaves: tuple[torch.Tensor, ...], warmup: int, reps: int
) -> float:
    """CUDA-event-timed forward+backward pass; returns microseconds per iteration."""
    for _ in range(warmup):
        _fwd_bwd_once(fn, leaves)
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(reps):
        _fwd_bwd_once(fn, leaves)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) * 1000.0 / reps


def _compile_backend_exceptions() -> tuple[type[BaseException], ...]:
    """The exception types a failed ``torch.compile`` backend legitimately raises.

    Imported lazily (only reached from ``--bench``, already gated on
    ``torch>=2.8`` via the ``triton_scans`` import) so a torch build
    without ``torch._dynamo`` degrades to an empty tuple rather than
    breaking the whole module at import time.
    """
    try:
        import torch._dynamo.exc as dynamo_exc

        return (dynamo_exc.TorchDynamoException,)
    except ImportError:
        return ()


def _bench_one_case(
    op_name: str,
    cfg: _OpConfig,
    shape: dict,
    impl: str,
    direction: str,
    compiled_cache: dict,
) -> dict | None:
    """Time one (op, shape, impl, direction) case; return a schema row or None.

    Returns ``None`` (and prints a ``[skip]`` line) only for the
    legitimately-expected unavailability types: ``triton_scans.ScanFallback``
    (the kernel declined this shape), ``torch.cuda.OutOfMemoryError``
    (transient GPU memory exhaustion), and a ``torch.compile`` backend
    failure (``torch._dynamo``'s exception hierarchy). Any other exception
    -- a real programming defect (e.g. a future signature mismatch raising
    ``TypeError``) -- propagates and aborts the sweep instead of being
    silently absorbed into a skip line; the caller (``run_bench``) counts
    this case as "attempted" either way, so a caught skip here still
    contributes to the run's overall pass/fail verdict.
    """
    device = "cuda"
    label = f"{op_name}/{shape['label']}/{impl}/{direction}"
    expected_unavailable = (torch.cuda.OutOfMemoryError,) + _compile_backend_exceptions()
    try:
        fn = _get_impl_fn(impl, op_name, shape["label"], compiled_cache)
        # Same seed for every impl on this (op, shape, direction) case, so
        # all three impls time the identical input tensors -- an
        # apples-to-apples comparison, not one incidentally on easier data.
        torch.manual_seed(0)
        base_tensors = cfg.build(shape["B"], shape["T"], device)
        if direction == "fwd":
            us = _time_forward(fn, base_tensors, shape["warmup"], shape["reps"])
        else:
            leaves = tuple(t.detach().clone().requires_grad_(True) for t in base_tensors)
            us = _time_forward_backward(fn, leaves, shape["warmup"], shape["reps"])
    except triton_scans.ScanFallback as exc:
        print(f"  [skip] {label}: kernel declined this shape: {exc}")
        return None
    except expected_unavailable as exc:
        print(f"  [skip] {label}: {type(exc).__name__}: {exc}")
        return None
    finally:
        os.environ["MINGRU_SCAN"] = "eager"

    print(f"  [ok]   {label}: {us:.2f} us/iter")
    return {
        "op": op_name,
        "impl": impl,
        "shape": _shape_for(cfg, shape),
        "dtype": _DTYPE_LABEL,
        "direction": direction,
        "us_per_iter": us,
    }


def _shape_for(cfg: _OpConfig, shape: dict) -> dict:
    """Render one bench shape's spec §6 schema ``shape`` dict for ``cfg``."""
    return cfg.shape(shape["B"], shape["T"])


def _fmt_us(value: float | None) -> str:
    """Format a microseconds-per-iter value for the markdown table, or "n/a"."""
    return f"{value:.2f}" if value is not None else "n/a"


def _render_bench_markdown(rows: list[dict], meta: dict) -> str:
    """One markdown table: op/shape/direction rows, one column per impl."""
    lines = [
        "# Scan kernel benchmark",
        "",
        f"torch {meta['torch_version']}, device {meta['device_name']}. "
        f"Warmup/reps per shape: {meta['warmup_reps']}.",
        "",
        "| op | shape | direction | triton (us) | eager (us) | torch_compile (us) | triton speedup vs eager |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    by_key: dict[tuple, dict] = {}
    for row in rows:
        key = (row["op"], json.dumps(row["shape"], sort_keys=True), row["direction"])
        by_key.setdefault(key, {})[row["impl"]] = row["us_per_iter"]

    for (op_name, shape_json, direction), impls in by_key.items():
        shape_str = json.dumps(json.loads(shape_json))
        tri = impls.get("triton")
        eag = impls.get("eager")
        cmp_ = impls.get("torch_compile")
        speedup = f"{eag / tri:.2f}x" if tri and eag else "n/a"
        lines.append(
            f"| {op_name} | {shape_str} | {direction} | {_fmt_us(tri)} | {_fmt_us(eag)} | "
            f"{_fmt_us(cmp_)} | {speedup} |"
        )
    return "\n".join(lines) + "\n"


def run_bench() -> int:
    """Run the full bench matrix (ops x shapes x impls x directions).

    ``triton_scans.available()`` is checked once up front: if the Triton
    kernel path itself is unavailable on this CUDA machine (present but no
    usable Triton), the ``triton`` impl is excluded from the sweep
    entirely (never attempted, so its absence can't count as a failure --
    see the exit-code contract below) while ``eager``/``torch_compile``
    still run. This also means ``_bench_one_case`` should structurally
    never see a "Triton unavailable" ``RuntimeError`` from
    ``min_gru._dispatch_scan``: that path is pre-empted here rather than
    caught per-case, keeping the per-case exception filter narrow.

    Exit-code contract: every case not proactively excluded above is
    "attempted"; a fully-excluded run (impossible in practice once CUDA
    gates this function, since eager/torch_compile are always attempted)
    would fall back to the loud-skip path. Any attempted case that raises
    (caught in ``_bench_one_case`` as a legitimate skip, or left uncaught
    as a real bug) makes this function return nonzero -- a benchmark run
    with holes in it is not a clean pass.
    """
    status = _gpu_status()
    if status is not True:
        return _loud_skip(f"--bench requires CUDA: {status}")
    if triton_scans is None:
        return _loud_skip(
            f"--bench requires triton_scans (torch>=2.8): {_TRITON_SCANS_IMPORT_ERROR}"
        )

    triton_status = triton_scans.available()
    triton_ok = triton_status is True
    if not triton_ok:
        print(
            f"note: Triton unavailable ({triton_status}) -- 'triton' impl cases "
            "will be excluded (not attempted), eager/torch_compile still run"
        )

    rows: list[dict] = []
    compiled_cache: dict = {}
    attempted = succeeded = failed = excluded_unavailable = 0
    saved_scan_env = os.environ.get("MINGRU_SCAN")
    try:
        for shape in _BENCH_SHAPES:
            print(f"shape {shape['label']} (B={shape['B']} T={shape['T']} "
                  f"d={shape['d']} k={shape['k']}):")
            cfgs = _op_configs(shape["d"], shape["k"])
            for op_name in _OPS:
                cfg = cfgs[op_name]
                for direction in _DIRECTIONS:
                    for impl in _IMPLS:
                        if impl == "triton" and not triton_ok:
                            excluded_unavailable += 1
                            continue
                        attempted += 1
                        row = _bench_one_case(
                            op_name, cfg, shape, impl, direction, compiled_cache
                        )
                        if row is not None:
                            succeeded += 1
                            rows.append(row)
                        else:
                            failed += 1
    finally:
        if saved_scan_env is None:
            os.environ.pop("MINGRU_SCAN", None)
        else:
            os.environ["MINGRU_SCAN"] = saved_scan_env

    meta = {
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "warmup_reps": {s["label"]: {"warmup": s["warmup"], "reps": s["reps"]} for s in _BENCH_SHAPES},
        "counts": {
            "attempted": attempted,
            "succeeded": succeeded,
            "failed": failed,
            "excluded_unavailable": excluded_unavailable,
        },
    }
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _BENCH_JSON.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    _BENCH_MD.write_text(_render_bench_markdown(rows, meta))
    print(f"\nwrote {_BENCH_JSON} and {_BENCH_MD} ({len(rows)} row(s))")
    summary = f"{succeeded}/{attempted} attempted case(s) succeeded"
    if excluded_unavailable:
        summary += f", {excluded_unavailable} excluded (Triton unavailable)"
    print(summary)
    if attempted == 0:
        return _loud_skip("no bench case was attempted (nothing to run)")
    if failed:
        print(f"FAILED: {failed} attempted case(s) did not produce a row")
        return 1
    return 0


# --- --memory ---------------------------------------------------------------


def _warmup_givens(mixer: torch.nn.Module, x: torch.Tensor, mode: str) -> None:
    """One discarded forward+backward under ``MINGRU_SCAN=mode``, no stats recorded.

    Primes whatever one-time allocation a fresh forward+backward under
    this mode would otherwise incur on its first call -- CUDA context /
    cuBLAS workspace for ``eager`` (the mixer's ``nn.Linear`` heads), the
    Triton kernel's first-launch JIT compile for ``triton`` -- so that
    cost can't land inside (and inflate) whichever measurement runs
    first. Must run, for both modes, before any
    ``torch.cuda.reset_peak_memory_stats()`` call.
    """
    os.environ["MINGRU_SCAN"] = mode
    mixer.zero_grad(set_to_none=True)
    out = mixer(x)
    out.float().sum().backward()
    mixer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def _measure_givens_peak_bytes(mixer: torch.nn.Module, x: torch.Tensor, mode: str) -> int:
    """One forward+backward of ``mixer`` under ``MINGRU_SCAN=mode``; peak bytes allocated."""
    os.environ["MINGRU_SCAN"] = mode
    mixer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    out = mixer(x)
    out.float().sum().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def run_memory() -> int:
    """Peak-memory comparison: angle-fused vs Phase-1 generic backward (spec §9.2).

    Builds one ``GivensMinGRU`` at the lab shape (``d=64, k=8`` ->
    ``block_size=8``, ``n_blocks=8``) and measures
    ``torch.cuda.max_memory_allocated`` for one forward+backward under
    ``MINGRU_SCAN=eager`` (the Phase-1 generic ``matrix_affine_scan``
    path, which materializes the full ``(B, T, n, k, k)`` transition
    tensors for autograd) against ``MINGRU_SCAN=triton`` (the angle-fused
    exact stored-state backward, which never materializes them). Stats are
    reset between measurements so each number is that run's own peak, not
    a running maximum across both. A discarded warmup forward+backward
    runs in both modes before either measurement window opens (mirroring
    ``--bench``'s warmup discipline) -- otherwise the one-time CUDA
    context / cuBLAS workspace allocation (and, for ``triton``, the
    kernel's first-launch JIT compile) would land inside whichever mode
    is measured first (``eager``, here), inflating that number and
    flattering the angle-fused memory win.
    """
    status = _gpu_status()
    if status is not True:
        return _loud_skip(f"--memory requires CUDA: {status}")
    if triton_scans is None:
        return _loud_skip(
            f"--memory requires triton_scans (torch>=2.8): {_TRITON_SCANS_IMPORT_ERROR}"
        )
    tri_status = triton_scans.available()
    if tri_status is not True:
        return _loud_skip(f"--memory requires a usable Triton kernel: {tri_status}")

    device = "cuda"
    B, T, d, k = _LAB_SHAPE["B"], _LAB_SHAPE["T"], _LAB_SHAPE["d"], _LAB_SHAPE["k"]
    saved_scan_env = os.environ.get("MINGRU_SCAN")
    try:
        torch.manual_seed(0)
        mixer = min_gru.GivensMinGRU(d, d, block_size=k, rounds=3).to(device)
        x = torch.randn(B, T, d, device=device)

        _warmup_givens(mixer, x, "eager")
        _warmup_givens(mixer, x, "triton")
        try:
            eager_bytes = _measure_givens_peak_bytes(mixer, x, "eager")
            triton_bytes = _measure_givens_peak_bytes(mixer, x, "triton")
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  [FAIL] GivensMinGRU peak-memory measurement: {type(exc).__name__}: {exc}")
            return 1
    finally:
        if saved_scan_env is None:
            os.environ.pop("MINGRU_SCAN", None)
        else:
            os.environ["MINGRU_SCAN"] = saved_scan_env

    shape = {"B": B, "T": T, "d": d, "k": k}
    rows = [
        {
            "impl": "eager_phase1_generic",
            "op": "GivensMinGRU.forward+backward",
            "shape": shape,
            "peak_bytes": eager_bytes,
            "peak_mb": eager_bytes / 2**20,
        },
        {
            "impl": "angle_fused_triton",
            "op": "GivensMinGRU.forward+backward",
            "shape": shape,
            "peak_bytes": triton_bytes,
            "peak_mb": triton_bytes / 2**20,
        },
    ]
    print(f"  eager (Phase-1 generic):  {eager_bytes / 2**20:.2f} MB peak")
    print(f"  triton (angle-fused):     {triton_bytes / 2**20:.2f} MB peak")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _MEMORY_JSON.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    _MEMORY_MD.write_text(
        "# GivensMinGRU backward peak memory (angle-fused vs Phase-1 generic)\n\n"
        f"Shape: {shape}\n\n"
        "| impl | peak MB |\n| --- | --- |\n"
        f"| eager_phase1_generic | {eager_bytes / 2**20:.2f} |\n"
        f"| angle_fused_triton | {triton_bytes / 2**20:.2f} |\n"
    )
    print(f"\nwrote {_MEMORY_JSON} and {_MEMORY_MD}")
    return 0


# --- entry point --------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="run the full equivalence suite")
    parser.add_argument("--bench", action="store_true", help="run the timing benchmark matrix")
    parser.add_argument(
        "--memory", action="store_true", help="run the angle-fused vs Phase-1 memory comparison"
    )
    args = parser.parse_args(argv)
    if not (args.check or args.bench or args.memory):
        parser.error("at least one of --check, --bench, --memory is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    exit_code = 0
    if args.check:
        print("=== --check: equivalence suite ===")
        exit_code = run_check() or exit_code
    if args.bench:
        print("\n=== --bench: timing matrix ===")
        exit_code = run_bench() or exit_code
    if args.memory:
        print("\n=== --memory: angle-fused vs Phase-1 peak memory ===")
        exit_code = run_memory() or exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
