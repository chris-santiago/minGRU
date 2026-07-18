"""CUDA fusion-headroom probe for ``DeltaMinGRU``'s chunked-WY forward (Task 6).

Runs INSIDE a Lightning GPU job (submitted by ``scripts/gpu_check.py --job
delta-probe``, see that script's ``build_delta_probe_command``); never
invoked directly by a developer machine without CUDA. Answers the
speedup-worth-it question opened by ``.claude/next-round-matched-state.md``
Phase B: profile eager chunked-WY ``DeltaMinGRU`` on CUDA and measure the
gap to its matmul-FLOP floor (the maximum a fused Triton kernel could win),
with a ``torch.compile`` arm showing how much of that gap Inductor already
closes for free. If compile sits near the floor, a hand-written kernel has
nothing left to win; see ``.git/sdd/task-6-brief.md`` for the full task.

Grid (five shapes, B=128 throughout):

- ``pd1024`` -- ``DeltaMinGRU(64, 64, n_heads=4, nh=2, d_k=16, d_v=16)``
  (state 1024, the packaged default's "pd1024" training-arm shape from the
  matched-state round) at T in {64, 256, 1024}.
- ``stepup`` -- ``DeltaMinGRU(256, 256, n_heads=4, nh=2)`` (``d_k``/``d_v``
  default to ``hidden_size // n_heads = 64``; state 16384) at T in {256,
  1024}.

``chunk_size`` is left at ``DeltaMinGRU``'s default (64) for every shape.

Three arms per shape:

``eager``
    Forward+backward (``.sum().backward()``), CUDA-event timed
    (``torch.cuda.synchronize()`` around each timed step), 3 warmup + 10
    timed steps (median reported, full list kept for transparency); peak
    memory via ``torch.cuda.reset_peak_memory_stats()`` /
    ``max_memory_allocated()`` reset AFTER warmup so the reported peak
    reflects steady-state allocator behavior, not one-time warmup growth.

``floor`` (approximate, explicitly labeled)
    Standalone batched-GEMM/triangular-solve ops replicating
    ``DeltaMinGRU._forward_chunked``'s dominant per-chunk contractions
    (``src/mingru/min_gru.py``) at the shape's ACTUAL per-chunk tensor
    sizes -- see ``_floor_forward_ops`` for the seven-op inventory (six
    batched matmuls plus the unit-lower-triangular UT-transform solve).
    Forward-only, same CUDA-event timing protocol as ``eager``, then
    scaled by the standard 3x fwd-GEMM convention (fwd:bwd:total ~ 1:2:3)
    to estimate fwd+bwd cost. This is a floor on the DOMINANT contractions
    only: it omits the elementwise beta/mask scaling, the query/key/value
    projections, and concat/reshape overhead the real forward also pays,
    and the 3x rule is a convention, not a measured backward -- never
    presented as exact. The 3x convention is uniform across all seven ops
    but is a rougher approximation for the triangular solve than for the
    six GEMMs specifically (``solve_triangular``'s backward is a second
    triangular solve plus one outer-product matmul, not a simple 2-matmul,
    with a different GPU-parallelism profile than the GEMMs' fully
    data-parallel backward); see ``_FLOOR_METHOD`` for the full
    disclosure. ``floor_method`` in the result states all of this inline
    so a reader of the artifact never has to cross-reference this
    docstring to know the approximation is in play. Each row also carries
    ``floor_suspect`` (``True`` when ``floor_step_secs > eager_step_secs_
    median`` for that shape -- the floor cannot legitimately exceed what
    it lower-bounds, so this flags a broken floor estimate for that shape
    rather than leaving a reader to infer it from
    ``headroom_eager_over_floor < 1``).

``compile``
    ``torch.compile(layer)``, same timing protocol as ``eager`` (warmup
    absorbs graph compilation). If compilation or a compiled step raises,
    the arm records ``compile_status: "failed"`` with the exception text
    and the probe continues to the next shape -- a compile failure on one
    shape must never crash the whole grid (the broad ``except Exception``
    in ``_compile_arm`` is this script's one deliberately-disclosed
    boundary catch, required by the task brief, not a silently swallowed
    error: the message is threaded into the artifact verbatim). After
    each shape's compile arm, ``_run_shape`` best-effort calls
    ``torch._dynamo.reset()`` so this shape's guard cache does not persist
    into the next shape's ``eager`` timing/peak-memory baseline.

Output protocol (mirrors ``scripts/scaling_probe.py``'s
``MINGRU_PROBE_HEADER``/``MINGRU_PROBE_RESULT`` line-marker pattern):
human-readable progress lines to stdout, then exactly ONE final line
``MINGRU_GPU_PROBE_RESULT <json>`` carrying the whole result (env block +
every shape's row). This is the sole transport: the job's container
filesystem dies with the job, so nothing is written to disk here --
``scripts/gpu_check.py --job delta-probe`` extracts this line from the
job's fetched logs on the submitting machine and writes the local
artifact (``experiments/bench/gpu_delta_probe.json``/``.md``).

Evidence-pin discipline: this probe deliberately does NOT assert a pinned
torch version (contrast ``scripts/scaling_probe.py``/``bench_delta.py``,
which pin ``torch==2.5.1`` for CPU comparability against recorded rows).
It runs under whatever CUDA-capable torch the job's container image
provides and records that version, the CUDA device name/capability, the
triton version (if importable), and the matmul-precision flags
(``torch.backends.cuda.matmul.allow_tf32``, ``torch.backends.cudnn.
allow_tf32``, ``torch.get_float32_matmul_precision()`` -- unmodified from
the container's defaults, this script never sets them) in the artifact's
env block. The precision flags matter because TF32 eligibility is not
guaranteed to apply identically across the op families this probe times
(``torch.bmm``, which the floor/eager arms' GEMMs use, vs.
``torch.linalg.solve_triangular``'s cuBLAS TRSM path) -- recording them
lets a reader distinguish real fusion headroom from a TF32-eligibility
asymmetry between GEMM and TRSM. Nothing in this artifact is presented as
comparable to the pinned-CPU rows in
``experiments/lab_results.jsonl``/``EXPERIMENTS.md``.

CUDA-only: ``main()`` asserts ``torch.cuda.is_available()`` before
constructing any model or importing ``mingru``, and fails fast with a
clear message (nonzero exit, no traceback from deep inside) when it is
not -- this is what lets a CPU-side smoke test verify the guard without a
GPU.

Import-path note: like ``scripts/bench_delta.py``/``scripts/
scaling_probe.py``, this script must import ``mingru`` even when the
project isn't pip/uv-installed in the job's container image. Unlike
those two, the ``src/``-onto-``sys.path`` bootstrap and the ``mingru``
import live inside ``_import_delta_mingru`` (function-scoped, called once
from ``main`` after the CUDA guard) rather than at module level -- this
keeps every module-level import a plain, unconditional stdlib/torch
import (torch itself needs no path surgery; it resolves via the
container's site-packages regardless of CWD), so this file needs no
``ruff`` ``E402`` exemption anywhere, module-level or per-file.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch

_RESULT_PREFIX = "MINGRU_GPU_PROBE_RESULT "

_BATCH_SIZE = 128
_WARMUP_STEPS = 3
_TIMED_STEPS = 10
_SEED = 0

# Standard fwd-GEMM convention: backward costs ~2x forward's matmul FLOPs
# (one gradient matmul w.r.t. the input, one w.r.t. the weight), so
# fwd+bwd ~ 3x a forward-only measurement. Applied to the floor arm's
# forward-only timing to estimate its fwd+bwd cost without needing an
# actual backward graph through the standalone ops (see the module
# docstring's "floor" arm section).
_FLOOR_FWD_BWD_CONVENTION = 3.0

_FLOOR_METHOD = (
    "Standalone batched GEMM/triangular-solve ops replicating "
    "DeltaMinGRU._forward_chunked's dominant per-chunk contractions "
    "(KK=K@K^T, KH=K@H, the unit-lower-triangular UT-transform solve, "
    "R=Q@K^T, QH=Q@H, RU=(R*mask)@U, KU=K^T@U) at the shape's actual "
    "per-chunk tensor sizes, forward-only CUDA-event timing "
    f"(warmup={_WARMUP_STEPS}, median of {_TIMED_STEPS}), scaled by "
    f"{_FLOOR_FWD_BWD_CONVENTION:g}x for the standard fwd+bwd convention "
    "(fwd:bwd:total ~ 1:2:3). Approximate and explicitly disclosed: omits "
    "the elementwise beta/mask scaling, the q/k/v/beta linear "
    "projections, and concat/reshape overhead the real forward also "
    "pays; the 3x rule is a convention, not a measured backward. The "
    "blanket 3x convention is a rougher approximation for the solve "
    "component than for the six GEMM components specifically: "
    "solve_triangular's backward is NOT a simple 2-matmul the way a "
    "plain matmul's backward is -- it is a second triangular solve (with "
    "the transposed system) plus one outer-product matmul against the "
    "forward solution, a different GPU-parallelism profile (sequential "
    "forward-substitution dependency chain vs. the six ops' fully "
    "data-parallel GEMMs) than the 1:2 fwd:bwd FLOP ratio the 3x rule "
    "assumes. The 3x convention is still applied uniformly across all "
    "seven ops (this probe does not attempt a per-op-family bwd "
    "estimate), so treat floor_step_secs as a looser bound wherever the "
    "solve dominates a shape's per-chunk cost. Never presented as an "
    "exact floor -- see the module docstring."
)

_PD1024_KWARGS: dict[str, Any] = {
    "input_size": 64,
    "hidden_size": 64,
    "n_heads": 4,
    "nh": 2,
    "d_k": 16,
    "d_v": 16,
}
_STEPUP_KWARGS: dict[str, Any] = {
    "input_size": 256,
    "hidden_size": 256,
    "n_heads": 4,
    "nh": 2,
}


@dataclass(frozen=True)
class _ShapeConfig:
    label: str
    config_name: str  # "pd1024" | "stepup"
    build_kwargs: dict[str, Any]
    batch_size: int
    seq_len: int


def _grid() -> list[_ShapeConfig]:
    grid = [
        _ShapeConfig(
            label=f"pd1024_T{t}",
            config_name="pd1024",
            build_kwargs=_PD1024_KWARGS,
            batch_size=_BATCH_SIZE,
            seq_len=t,
        )
        for t in (64, 256, 1024)
    ]
    grid += [
        _ShapeConfig(
            label=f"stepup_T{t}",
            config_name="stepup",
            build_kwargs=_STEPUP_KWARGS,
            batch_size=_BATCH_SIZE,
            seq_len=t,
        )
        for t in (256, 1024)
    ]
    return grid


def _import_delta_mingru() -> type:
    """Import and return ``DeltaMinGRU``, bootstrapping ``src/`` onto ``sys.path``.

    See the module docstring's "Import-path note": function-scoped so no
    module-level import in this file needs a ``sys.path`` insertion ahead
    of it (avoids an ``E402`` exemption). Called once from ``main`` after
    the CUDA guard.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from mingru import DeltaMinGRU

    return DeltaMinGRU


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _time_cuda_steps(step_fn, warmup: int, timed: int) -> list[float]:
    """CUDA-event-timed step_fn, ``warmup`` untimed calls then ``timed`` timed ones."""
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    secs: list[float] = []
    for _ in range(timed):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        step_fn()
        end_evt.record()
        torch.cuda.synchronize()
        secs.append(start_evt.elapsed_time(end_evt) / 1000.0)
    return secs


def _eager_arm(layer: torch.nn.Module, x: torch.Tensor) -> tuple[list[float], int]:
    """Forward+backward, CUDA-event timed; returns (timed secs, peak mem bytes).

    Peak memory is reset AFTER the untimed warmup (see the module
    docstring's ``eager`` arm) so it reflects the timed loop's
    steady-state allocations, not one-time warmup-only growth (e.g. the
    caching allocator's first block reservations).
    """

    def step() -> None:
        layer.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        loss = layer(x).sum()
        loss.backward()

    for _ in range(_WARMUP_STEPS):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    secs = _time_cuda_steps(step, warmup=0, timed=_TIMED_STEPS)
    peak_mem = torch.cuda.max_memory_allocated()
    return secs, peak_mem


def _floor_forward_ops(
    layer: torch.nn.Module, batch_size: int, seq_len: int, device: torch.device
) -> tuple[Any, list[dict[str, Any]]]:
    """Build the per-chunk GEMM/solve op sequence at this shape's actual sizes.

    Reads ``n_heads``/``nh``/``d_k``/``d_v``/``chunk_size`` live off the
    constructed ``layer`` (not the caller's ``build_kwargs``) so the op
    shapes can never drift from what the module actually uses (mirrors
    ``scripts/scaling_probe.py``'s "read live off the constructed module"
    convention). Values are random standins with the right shape/
    triangular structure, not the true k.k/beta contraction -- this arm
    times op cost, not the numerically exact algorithm (see
    ``_FLOOR_METHOD``).

    Returns ``(run_once, op_inventory)``: ``run_once()`` executes every
    chunk's op sequence once, forward-only, no autograd; ``op_inventory``
    is a JSON-able per-chunk shape listing for the artifact.
    """
    n_heads, nh = layer.n_heads, layer.nh
    d_k, d_v, chunk_size = layer.d_k, layer.d_v, layer.chunk_size
    bh = batch_size * n_heads
    dtype = next(layer.parameters()).dtype

    chunks: list[tuple[torch.Tensor, ...]] = []
    inventory: list[dict[str, Any]] = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        c = end - start
        m = nh * c
        k = torch.randn(bh, m, d_k, device=device, dtype=dtype)
        v = torch.randn(bh, m, d_v, device=device, dtype=dtype)
        q = torch.randn(bh, c, d_k, device=device, dtype=dtype)
        h = torch.randn(bh, d_k, d_v, device=device, dtype=dtype)
        # Unit-lower-triangular UT-transform matrix: identity diagonal plus
        # random strictly-lower entries -- matches _forward_chunked's
        # `T_mat` shape/structure for the triangular solve's op cost
        # without claiming numerical fidelity (see _FLOOR_METHOD).
        strict_lower = torch.tril(torch.rand(bh, m, m, device=device, dtype=dtype), diagonal=-1)
        t_mat = strict_lower + torch.eye(m, device=device, dtype=dtype)
        read_mask = torch.rand(bh, c, m, device=device, dtype=dtype)
        chunks.append((k, v, q, h, t_mat, read_mask))
        inventory.append(
            {
                "chunk_C": c,
                "M": m,
                "ops": [
                    {"name": "KK = K @ K^T", "out_shape": [bh, m, m]},
                    {"name": "KH = K @ H (rhs term)", "out_shape": [bh, m, d_v]},
                    {
                        "name": "U = solve_triangular(T_mat, V - KH, unitriangular=True)",
                        "out_shape": [bh, m, d_v],
                    },
                    {"name": "R = Q @ K^T", "out_shape": [bh, c, m]},
                    {"name": "QH = Q @ H", "out_shape": [bh, c, d_v]},
                    {"name": "RU = (R * read_mask) @ U", "out_shape": [bh, c, d_v]},
                    {"name": "KU = K^T @ U (state update)", "out_shape": [bh, d_k, d_v]},
                ],
            }
        )

    def run_once() -> None:
        for k, v, q, h, t_mat, read_mask in chunks:
            kk = torch.bmm(k, k.transpose(-1, -2))
            kh = torch.bmm(k, h)
            rhs = v - kh
            u = torch.linalg.solve_triangular(t_mat, rhs, upper=False, unitriangular=True)
            r = torch.bmm(q, k.transpose(-1, -2))
            qh = torch.bmm(q, h)
            ru = torch.bmm(r * read_mask, u)
            ku = torch.bmm(k.transpose(-1, -2), u)
            del kk, qh, ru, ku  # forward-only op cost; nothing consumed downstream

    return run_once, inventory


def _floor_arm(
    layer: torch.nn.Module, batch_size: int, seq_len: int, device: torch.device
) -> tuple[float | None, list[float], list[dict[str, Any]]]:
    run_once, inventory = _floor_forward_ops(layer, batch_size, seq_len, device)
    fwd_secs = _time_cuda_steps(run_once, warmup=_WARMUP_STEPS, timed=_TIMED_STEPS)
    fwd_median = _median(fwd_secs)
    floor_step_secs = _FLOOR_FWD_BWD_CONVENTION * fwd_median if fwd_median is not None else None
    return floor_step_secs, fwd_secs, inventory


def _compile_arm(
    layer: torch.nn.Module, x: torch.Tensor
) -> tuple[list[float] | None, str, str | None]:
    """``torch.compile`` arm; never raises -- returns ``(secs, status, error)``.

    The broad ``except Exception`` here is this script's one deliberately
    -disclosed boundary catch (required by the task brief: "If compile
    fails, record compile_status: failed with the error string and
    continue -- never crash the probe"). Catches both compilation
    failures and failures during the first compiled step (Inductor lazily
    traces on first call), so either failure mode degrades to
    ``status="failed"`` with the exception text rather than aborting the
    whole grid.
    """
    try:
        compiled = torch.compile(layer)

        def step() -> None:
            layer.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            loss = compiled(x).sum()
            loss.backward()

        secs = _time_cuda_steps(step, warmup=_WARMUP_STEPS, timed=_TIMED_STEPS)
        return secs, "ok", None
    except Exception as exc:  # disclosed boundary catch, see docstring above
        return None, "failed", f"{type(exc).__name__}: {exc}"


def _run_shape(cfg: _ShapeConfig, delta_mingru_cls: type, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(_SEED)
    layer = delta_mingru_cls(**cfg.build_kwargs).to(device)
    x = torch.randn(
        cfg.batch_size,
        cfg.seq_len,
        cfg.build_kwargs["input_size"],
        device=device,
        requires_grad=True,
    )

    eager_secs, eager_peak_mem = _eager_arm(layer, x)
    eager_median = _median(eager_secs)

    floor_step_secs, floor_fwd_secs, op_inventory = _floor_arm(
        layer, cfg.batch_size, cfg.seq_len, device
    )

    compile_secs, compile_status, compile_error = _compile_arm(layer, x)
    compile_median = _median(compile_secs) if compile_secs else None

    # torch._dynamo.reset() between shapes: without it, Dynamo's guard cache
    # from this shape's compile arm can persist into the next shape's eager
    # arm timing (recompilation checks/guard-cache lookups) and inflate its
    # peak-memory baseline. Best-effort -- torch always ships _dynamo
    # alongside torch.compile, but this must never crash the probe over a
    # cleanup step.
    try:
        torch._dynamo.reset()
    except Exception as exc:
        print(f"[warn] torch._dynamo.reset() failed: {exc!r}", file=sys.stderr)

    headroom = None
    if eager_median is not None and floor_step_secs:
        headroom = eager_median / floor_step_secs

    # Sanity flag: a floor estimate slower than the measured eager step is a
    # broken floor for this shape (the "floor" cannot legitimately exceed
    # what it's meant to lower-bound), not evidence of negative headroom --
    # surfaced explicitly so a reader doesn't have to notice
    # headroom_eager_over_floor < 1 on their own.
    floor_suspect = (
        eager_median is not None and floor_step_secs is not None and floor_step_secs > eager_median
    )

    compile_recovered = None
    if compile_median is not None and eager_median is not None and floor_step_secs is not None:
        denom = eager_median - floor_step_secs
        if denom != 0:
            compile_recovered = (eager_median - compile_median) / denom

    return {
        "label": cfg.label,
        "config_name": cfg.config_name,
        "config": dict(cfg.build_kwargs),
        "live_config": {
            "n_heads": layer.n_heads,
            "nh": layer.nh,
            "d_k": layer.d_k,
            "d_v": layer.d_v,
            "chunk_size": layer.chunk_size,
            "state_elements": layer.n_heads * layer.d_k * layer.d_v,
        },
        "B": cfg.batch_size,
        "T": cfg.seq_len,
        "eager_step_secs_median": eager_median,
        "eager_step_secs_all": eager_secs,
        "eager_peak_mem_bytes": eager_peak_mem,
        "floor_step_secs": floor_step_secs,
        "floor_forward_only_secs_all": floor_fwd_secs,
        "floor_method": _FLOOR_METHOD,
        "floor_op_inventory": op_inventory,
        "floor_suspect": floor_suspect,
        "compile_step_secs_median": compile_median,
        "compile_step_secs_all": compile_secs or [],
        "compile_status": compile_status,
        "compile_error": compile_error,
        "headroom_eager_over_floor": headroom,
        "compile_recovered_fraction": compile_recovered,
    }


def _env_block() -> dict[str, Any]:
    env: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "platform": platform.platform(),
        "batch_size": _BATCH_SIZE,
        "warmup_steps": _WARMUP_STEPS,
        "timed_steps": _TIMED_STEPS,
        # Matmul-precision flags: on Ampere/Ada these materially affect GEMM
        # wall-clock (TF32 vs fp32), and are not guaranteed to apply
        # identically to every op family this probe times -- torch.bmm (the
        # floor/eager arms' GEMMs) reads torch.backends.cuda.matmul.allow_tf32
        # / get_float32_matmul_precision(), while torch.linalg.solve_triangular
        # (cuBLAS TRSM) has its own internal precision path that is not
        # controlled the same way. Recorded here, unmodified from whatever
        # the container's torch defaults are (this script never sets them),
        # so a reader can tell real fusion headroom apart from a TF32-
        # eligibility asymmetry between the GEMM and TRSM components instead
        # of having to guess at the run's precision configuration.
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import triton

        env["triton_version"] = triton.__version__
    except ImportError:
        env["triton_version"] = None
    return env


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)

    # CUDA-only guard: must run before any model construction or `mingru`
    # import so a CPU-only invocation fails fast with a clear message, not
    # a traceback from deep inside DeltaMinGRU/torch.linalg.solve_triangular
    # (see the module docstring's "CUDA-only" section).
    if not torch.cuda.is_available():
        print(
            "FAILED: scripts/gpu_delta_probe.py requires CUDA "
            "(torch.cuda.is_available() is False) -- this probe measures "
            "CUDA fusion headroom for DeltaMinGRU's chunked-WY forward and "
            "cannot run on CPU. Run it inside the Lightning GPU job via "
            "`python scripts/gpu_check.py --job delta-probe`.",
            file=sys.stderr,
        )
        return 1

    delta_mingru_cls = _import_delta_mingru()
    device = torch.device("cuda")

    env = _env_block()
    print("gpu_delta_probe: env " + json.dumps(env))

    shapes: list[dict[str, Any]] = []
    for cfg in _grid():
        print(f"  running {cfg.label} (B={cfg.batch_size}, T={cfg.seq_len})...", flush=True)
        row = _run_shape(cfg, delta_mingru_cls, device)
        eager_s = row["eager_step_secs_median"]
        floor_s = row["floor_step_secs"]
        compile_s = row["compile_step_secs_median"]
        print(
            f"    eager={eager_s:.4f}s floor={floor_s:.4f}s "
            f"compile={compile_s if compile_s is None else f'{compile_s:.4f}s'} "
            f"({row['compile_status']}) headroom={row['headroom_eager_over_floor']}",
            flush=True,
        )
        shapes.append(row)

    result = {"env": env, "shapes": shapes}
    print(_RESULT_PREFIX + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
