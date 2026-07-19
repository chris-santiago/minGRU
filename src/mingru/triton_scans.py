"""Triton kernels for `min_gru.py`'s four scan functions.

Lazily imported by `min_gru._dispatch_scan` only (never at `min_gru.py`
import time); see that function's docstring for the full dispatch
contract. Development is GPU-blind locally (this module requires a CUDA
device and Triton to do anything useful); correctness and benchmarks land
on a cloud GPU.

`min_gru.py`'s recorded lab evidence stays pinned to `torch==2.5.1` CPU
and never imports this module. This module targets `torch>=2.8` (mature
`torch.library.triton_op`/`wrap_triton` registration) and raises a clear
`ImportError` below that floor, rather than failing confusingly deep
inside kernel registration.

Contents
--------
Forward cores (Kernel 1, Kernel 2). The generic affine-scan FORWARD core
(Kernel 1) plus wrappers mapping three of the four scan ops onto it --
`linear_scan` (k=1, channel-tiled), `matrix_scan` (k=2, v=1), and
`matrix_affine_scan` (generic k, v) -- and the fused log-space FORWARD
scan (Kernel 2) for `parallel_scan_log` (channel-tiled lanes carrying a
running log-coefficient cumsum plus an online max-shifted log-sum-exp
accumulator, writing h = exp(.) directly). All four scan ops therefore
have an entry in `SCAN_IMPLS`. Kernels accumulate in fp32 regardless of
input dtype and are registered via `torch.library.triton_op` so
`torch.compile` sees them without graph breaks.

Backward cores (Kernel 3, adjoint recurrences). Every Triton path is
trainable. The generic core's adjoint is one reverse-direction scan
(`_affine_scan_bwd_kernel`) that reads ONLY the forward's inputs and
outputs -- it reverse-scans the incoming output grads with A_{t+1}^T,
then forms dL/dA_t from the forward outputs Abar_{t-1}/Bbar_{t-1} (seeded
with I/0 at t=1) and dL/dB_t directly. That zero-extra-saved-tensors
property is the point of the reversed-scan design. `linear_scan` gets the
channel-tiled k=1 specialization (`_linear_scan_bwd_kernel`);
`matrix_scan`/`matrix_affine_scan` share the generic kernel through the
same unsqueeze/`affine_scan_fwd` seam as the forward. `parallel_scan_log`'s
backward is autograd-through-recomputation: it saves only its two forward
INPUTS (log_coeffs, log_values) and re-derives the grad through the eager
log-space math (exact-to-eager, no hand-derived log-space adjoint kernel
to get wrong blind) -- also zero extra saved tensors beyond the forward
inputs. All four ops are wired via `torch.library.register_autograd` on
their forward `triton_op` (the torch 2.8-idiomatic route for a
`triton_op`: the backward composes with `torch.compile`'s tracing and
dispatcher, which an `autograd.Function` wrapper around the op would not).

Angle-fused path (Kernel 4). `angle_scan_fwd`/`angle_scan_bwd`, a
module-level fast path (not one of the four scan ops) that `GivensMinGRU`
and `RotationMinGRU` route their forwards through on CUDA. The forward
carries the state vector in registers and applies factored plane
rotations from angles directly, never materializing or scanning the k x k
transition matrices; the backward is an EXACT stored-state reversible
recomputation (every `h_{t-1}` read from the forward output, never
reconstructed by inverting the forward step) accumulating the grads of
the angles, scale channel, injection, decay, and h0. It is generic over
(block size, rounds, scale channel): Givens (k=8, `rounds`, no per-block
scale) and Rotation (k=2, one plane, post-snap angles, tanh(u) scale).
Division-based reversal across a multi-step checkpoint chunk (dividing the
backward by `gamma`/`tanh(u)`) is deliberately NOT used: a blind CPU probe
showed that division amplifies roundoff by roughly `sigma_min^{-chunklen}`,
blowing past the grad tolerance even at ordinary decay/init strengths
(including `GivensMinGRU`'s class-default `decay_rate=1.0`), so BOTH mixers
use exact per-step recompute and no interval parameter exists to re-enable
division-based reversal. Registered via `torch.library.register_autograd`
like the four scan ops.

DeltaMinGRU chunked-WY forward trio (Kernel 5, Kernel 6a, Kernel 6b).
`delta_scan_impl` drives three `@triton.jit` kernels implementing the
two-stage WY decomposition of `DeltaMinGRU._forward_chunked` (the eager
oracle). The unit-lower-triangular system `T = I + (K K^T (.) beta) (.)
strict-lower` depends only on `K`/`beta`, never on the carried state `H`, so
its solves parallelize across chunks: the pre-pass (`_delta_prepass_kernel`,
Kernel 5, grid over `(batch*head, chunk)`) builds each chunk's `T` in
registers and forward-substitutes it against the two H-independent right-hand
sides, producing `T^-1 V` and `T^-1 K` for the whole sequence in one launch.
The state pass (`_delta_state_kernel`, Kernel 6a, grid over `(batch*head)`)
then loops over chunks in-kernel carrying `H` (d_k x d_v fp32), forming
`U = T^-1 V - (T^-1 K) H` and the state update `H += K^T (beta U)`; it writes
ONLY the per-chunk boundary states (`Hbound`, the start-of-chunk `H`) and the
final `H_T`. The readout (`_delta_readout_kernel`, Kernel 6b, grid over
`(batch*head, chunk)`) then runs one program per chunk -- reading `Hbound[c]`,
recomputing `U`, and writing the block-causal masked readout `y = Q H +
(Q K^T (.) beta (.) read_mask) U` -- so the expensive readout parallelizes
across chunks instead of being serialized in the `H`-chain, and each program
carries a small tile set (no `C x M` readout tile in the serial pass, no
`H`-carry in the parallel pass). `Hbound[c]` is exactly the `H` the readout
sees at chunk `c`, so the split is numerically identical to a single fused
sequential pass. Unlike the four scan ops and the angle-fused path, this trio
is plain `@triton.jit` launched directly via
`_delta_forward_launch` (not `triton_op`/`register_autograd`): the delta
path's backward is a hand-derived `torch.autograd.Function` (`_DeltaScanFn`,
which `delta_scan_impl` routes through) seeded from `Hbound`, and its target
user is eager-only, so the compile-tracing machinery the four scan ops use is
deliberately not in this path. That backward is itself a fused Triton trio
(Kernel 7a/7b/7c, `_delta_backward_launch`) MIRRORING the forward
decomposition: `_delta_bwd_prepass_kernel` (chunk-parallel) hoists the
`dH`-independent transpose solve into `TinvTBK`/`dHconst`,
`_delta_bwd_state_kernel` (serial per `batch*head`) carries the reverse `dH`
chain writing `dHbound`/`dH0`, and `_delta_bwd_grad_kernel` (chunk-parallel)
recomputes `T`/`U`/`G` per chunk and writes `dQ`/`dK`/`dV`/`dbeta` -- so the
whole training step is kernel-resident (a torch-op backward would serialize
~12 ops per chunk and forfeit the fwd+bwd speed bar). The verified torch
reverse-chunk loop (`_delta_backward_torch`) is retained as the documented
fallback if the fused launch raises (a resource/compile failure): exact,
slower, warned, never a silent wrong grad. Every tile is padded to
`max(16, next_pow2(...))`
(`_delta_block_sizes`) so all `tl.dot`s route through
`input_precision="ieee"` and stay bit-exact fp32, matching Kernel 5/6's
`d_k == d_v`, fp32-only envelope (`_delta_validate_envelope`/
`_delta_envelope_reason`), which is separate from the four-op envelope
below but raises `ScanFallback` the same way.

Envelope: k, v in {1, 2, 4, 8, 16}, any T >= 1 including non-power-of-two.
Out-of-envelope shapes raise `ScanFallback`, which `min_gru._dispatch_scan`
turns into a loud eager fallback under `auto` (and a raised error under
`triton`) -- never a wrong result, never a crash.
"""

import contextlib
import copy
import os
import types
import warnings

import torch
import torch.nn.functional as F

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


class ScanFallback(Exception):
    """Signal that the Triton path declines this call; fall back to eager.

    Raised by a ``SCAN_IMPLS`` entry when the inputs are outside the
    kernel envelope (unsupported ``k``/``v``) or when a kernel launch
    fails. ``min_gru._dispatch_scan`` catches this: under
    ``MINGRU_SCAN=auto`` it warns once and runs the eager reference;
    under ``MINGRU_SCAN=triton`` it re-raises as a ``RuntimeError`` (never
    a silent downgrade). This is distinct from "no kernel registered for
    this op" -- an op simply absent from ``SCAN_IMPLS`` is handled by
    ``_dispatch_scan`` before any call is attempted.
    """


# Kernel envelope for the state block size ``k`` and injection width ``v``.
# All members are powers of two, so the kernels tile ``k``/``v`` exactly
# with no masking; a shape outside this set raises ``ScanFallback``.
_ENVELOPE = frozenset({1, 2, 4, 8, 16})

# Channel-tile width for the two elementwise-in-D kernels (`linear_scan`
# and `parallel_scan_log`): each program walks the full T sequence for
# BLOCK_D channels at once, keeping lanes wide. A safe default;
# the cloud benchmark phase may autotune it.
_LINEAR_BLOCK_D = 128


@contextlib.contextmanager
def _scan_env(mode: str | None = None):
    """Save ``MINGRU_SCAN``'s current value, optionally force ``mode`` for
    the duration of the block, then restore whatever value (or absence)
    preceded it.

    Matches the repo-root ``min_gru.py`` evidence driver's own ``__main__``
    selftest discipline (its ``_set_scan_env``/``finally:
    _set_scan_env(_saved_scan_env)`` pattern).
    Without this, each parity runner (``_run_forward_parity``,
    ``_run_grad_parity``, ``_run_angle_fused_parity``) set
    ``os.environ["MINGRU_SCAN"] = "eager"`` (or ``"triton"``) to force one
    path for its own sweep and never restored it, leaking ``eager`` into the
    rest of the process (e.g. a subsequent ``--bench``/``--memory`` run
    invoked in the same process as ``--check``) once the runner returned.

    ``mode=None`` (the default) means "don't force a value on entry, just
    guarantee restoration on exit" -- for callers (``_run_angle_fused_parity``)
    whose own body toggles ``MINGRU_SCAN`` internally (e.g. alternating
    ``eager``/``triton`` per case) rather than needing one fixed value for
    the whole block.
    """
    saved = os.environ.get("MINGRU_SCAN")
    if mode is not None:
        os.environ["MINGRU_SCAN"] = mode
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("MINGRU_SCAN", None)
        else:
            os.environ["MINGRU_SCAN"] = saved


def parallel_scan_log_recompute(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
    """Pure-torch re-derivation of ``min_gru.parallel_scan_log``'s eager math.

    Defined at module level, OUTSIDE the ``if _HAS_TRITON:`` block below (no
    Triton import needed -- plain ``torch``/``torch.nn.functional`` only), so
    it is always importable, even on a CPU-only/no-Triton install. Two
    callers: ``_parallel_scan_log_backward``'s autograd-through-recomputation
    (inside ``_HAS_TRITON`` -- differentiates through this via
    ``torch.autograd.grad``), and the repo-root ``triton_scans.py`` evidence
    driver's own ``__main__`` CPU lockstep selftest (run via ``python
    triton_scans.py`` from a checkout), which asserts this function matches
    ``min_gru.parallel_scan_log`` on random CPU tensors WITHOUT needing a
    GPU/Triton -- catching drift between this and the eager reference on
    ordinary CI (and the GPU-less Phase-4 wheel CI), not only the GPU-only
    grad-parity selftest.

    Cost note: because the backward differentiates through this full eager
    recompute (under ``enable_grad``), the Triton route's forward+backward
    for ``parallel_scan_log`` is measured SLOWER than eager -- 0.70-0.80x at
    benchmarked shapes (``experiments/bench/scan_bench.md`` in the
    repository); ``MINGRU_SCAN=eager`` is the faster training choice for
    this op today.

    MAINTENANCE: this formula must be kept byte-identical to
    ``min_gru.parallel_scan_log``'s eager body (``a_star = pad(cumsum(log_coeffs));
    log_h = a_star + logcumsumexp(log_values - a_star); h = exp(log_h)[:, 1:]``).
    Replicated here (rather than calling ``min_gru.parallel_scan_log`` directly)
    so the backward neither re-enters the ``MINGRU_SCAN`` dispatcher nor
    depends on ``min_gru``'s env state; the CPU lockstep selftest is what
    keeps the two copies from silently drifting.

    Parameters
    ----------
    log_coeffs : torch.Tensor
        Shape ``(B, T, D)``. ``log(a_t)`` for ``t = 1..T``.
    log_values : torch.Tensor
        Shape ``(B, T + 1, D)``. Slot 0 is ``log(h_0)``; slots ``1..T`` are
        ``log(b_t)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, T, D)``. The states ``h_1..h_T``.
    """
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h = a_star + torch.logcumsumexp(log_values - a_star, dim=1)
    return torch.exp(log_h)[:, 1:]


# --- DeltaMinGRU chunked-WY kernel envelope -----------------------------
#
# The delta path is its own envelope (distinct from the four scan ops'
# ``_ENVELOPE`` above): the eager oracle is ``DeltaMinGRU._forward_chunked``,
# whose per-head state is a ``d_k x d_v`` associative-memory matrix updated by
# ``nh`` rank-1 corrections per token, processed ``chunk_size`` tokens at a
# time through a unit-lower-triangular UT-transform solve. The kernel path
# (three Triton kernels launched from ``_delta_forward_launch``) is gated to the
# recorded probe-grid shapes. These constants and the pure-Python validators
# below live OUTSIDE the ``if _HAS_TRITON:`` block (like
# ``parallel_scan_log_recompute``) so the envelope reasons are importable and
# unit-testable on a CPU-only/no-Triton install -- the reasons carry the whole
# gate contract and must be checkable without a GPU.

# Per-head key/query/value dimension: the WY solve tiles ``d_k`` exactly and
# the readout/state contractions assume ``d_k == d_v``. Restricted to this set
# (the probe grid) so every kernel contraction has all three matmul dims >= 16
# after padding (see ``_delta_block_sizes``), keeping every ``tl.dot`` both
# legal and bit-exact fp32 on the target arch.
_DELTA_DK_ENVELOPE = frozenset({4, 8, 16, 32, 64})
# Upper bound on ``chunk_size`` (tokens per UT-transform chunk).
_DELTA_MAX_CHUNK_SIZE = 64
# Upper bound on ``nh * chunk_size`` -- the micro-step count ``M`` of a full
# chunk, which is the side length of the in-kernel triangular system ``T`` and
# of the ``(M, M)`` shared-memory tile the pre-pass builds.
_DELTA_MAX_MICROSTEPS = 128


def _next_pow2(n: int) -> int:
    """Smallest power of two ``>= n`` (``>= 1``). Pure Python; no Triton needed."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _delta_block_sizes(nh: int, chunk_size: int, d_k: int, d_v: int) -> tuple[int, int, int, int]:
    """Constexpr tile sizes ``(BLOCK_M, BLOCK_C, BLOCK_K, BLOCK_V)`` for the delta kernels.

    Every tile is padded to a power of two AND to a floor of 16. The floor
    matters for correctness, not just occupancy: on the target arch a
    Triton contraction auto-lowers to a tensor-core MMA (defaulting to
    lossy TF32, or hard-crashing when a dim < 16) whenever both output free
    dims are >= 16. Padding the micro-step axis ``M``, the token axis ``C``,
    and the ``d_k``/``d_v`` axes all to >= 16 guarantees every kernel
    ``tl.dot`` has all three matmul dims >= 16, so each is routed through an
    explicit ``input_precision="ieee"`` dot that is bit-exact fp32 -- no
    per-dim TF32/legality guard combinatorics are needed inside the kernels.
    Padded rows/cols are masked to zero on load, so the widened contractions
    contribute exactly zero and the result is identical to the unpadded math.

    Pure Python (no Triton import), so it is callable for unit tests on a
    CPU-only install.
    """
    block_m = max(16, _next_pow2(nh * chunk_size))
    block_c = max(16, _next_pow2(chunk_size))
    block_k = max(16, _next_pow2(d_k))
    block_v = max(16, _next_pow2(d_v))
    return block_m, block_c, block_k, block_v


def _delta_envelope_reason(
    *,
    is_cuda: bool,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    beta_dtype: torch.dtype,
    h0_dtype: torch.dtype,
    d_k: int,
    d_v: int,
    chunk_size: int,
    nh: int,
) -> str | None:
    """Return a human-readable rejection reason, or ``None`` if in-envelope.

    Operates on plain descriptors (not tensors), so every branch is
    reachable from a CPU-only unit test by passing the descriptors that
    satisfy the earlier checks (e.g. ``is_cuda=True`` with all five dtypes
    ``float32`` and ``d_k==d_v`` to reach the ``d_k`` membership branch) --
    the whole gate contract is testable without a GPU. Checks are ordered so
    each violation yields one distinct string: non-CUDA, dtype (checked
    across all five of ``Q``, ``K``, ``V``, ``beta``, ``H0``, in that order),
    ``d_k != d_v``, ``d_k`` membership, ``chunk_size`` bound,
    ``nh * chunk_size`` bound.
    """
    if not is_cuda:
        return "DeltaMinGRU Triton kernel requires CUDA tensors (got a non-CUDA tensor)"
    if q_dtype != torch.float32:
        return f"DeltaMinGRU Triton kernel is fp32-only (got Q dtype {q_dtype})"
    if k_dtype != torch.float32:
        return f"DeltaMinGRU Triton kernel is fp32-only (got K dtype {k_dtype})"
    if v_dtype != torch.float32:
        return f"DeltaMinGRU Triton kernel is fp32-only (got V dtype {v_dtype})"
    if beta_dtype != torch.float32:
        return f"DeltaMinGRU Triton kernel is fp32-only (got beta dtype {beta_dtype})"
    if h0_dtype != torch.float32:
        return f"DeltaMinGRU Triton kernel is fp32-only (got H0 dtype {h0_dtype})"
    if d_k != d_v:
        return f"DeltaMinGRU Triton kernel requires d_k == d_v (got d_k={d_k}, d_v={d_v})"
    if d_k not in _DELTA_DK_ENVELOPE:
        return (
            f"DeltaMinGRU Triton kernel d_k must be in {sorted(_DELTA_DK_ENVELOPE)} (got d_k={d_k})"
        )
    if chunk_size > _DELTA_MAX_CHUNK_SIZE:
        return (
            f"DeltaMinGRU Triton kernel chunk_size must be <= {_DELTA_MAX_CHUNK_SIZE} "
            f"(got chunk_size={chunk_size})"
        )
    if nh * chunk_size > _DELTA_MAX_MICROSTEPS:
        return (
            "DeltaMinGRU Triton kernel requires nh * chunk_size <= "
            f"{_DELTA_MAX_MICROSTEPS} (got nh={nh}, chunk_size={chunk_size}, "
            f"product={nh * chunk_size})"
        )
    return None


def _delta_validate_envelope(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    beta: torch.Tensor,
    H0: torch.Tensor,
    chunk_size: int,
) -> None:
    """Raise ``ScanFallback`` unless ``(Q, K, V, beta, H0)`` are kernel-eligible.

    Guards the kernels' pointer arithmetic (rank + shape agreement of the
    post-permute assemblies ``DeltaMinGRU._forward_chunked`` builds) before
    delegating the value-envelope checks (CUDA/fp32/``d_k``/``chunk_size``/
    ``nh``) to ``_delta_envelope_reason``. Every rejection carries a distinct
    reason string. Pure torch/Python (no Triton), so ``min_gru``'s delta
    dispatch seam only ever needs to catch ``ScanFallback`` and the reasons
    are unit-testable without a GPU.

    Expected layouts (spec section 6): ``Q`` ``(B, n_heads, T, d_k)``; ``K``
    ``(B, n_heads, T, nh, d_k)``; ``V`` ``(B, n_heads, T, nh, d_v)``;
    ``beta`` ``(B, n_heads, T, nh)``; ``H0`` ``(B, n_heads, d_k, d_v)``.
    """
    if Q.ndim != 4 or K.ndim != 5 or V.ndim != 5 or beta.ndim != 4 or H0.ndim != 4:
        raise ScanFallback(
            "DeltaMinGRU Triton kernel needs 4-D Q (B, n_heads, T, d_k), 5-D K "
            "(B, n_heads, T, nh, d_k), 5-D V (B, n_heads, T, nh, d_v), 4-D beta "
            "(B, n_heads, T, nh), and 4-D H0 (B, n_heads, d_k, d_v); got ranks "
            f"Q={Q.ndim}, K={K.ndim}, V={V.ndim}, beta={beta.ndim}, H0={H0.ndim}"
        )
    B, n_heads, T, d_k = Q.shape
    nh = K.shape[3]
    d_v = V.shape[4]
    if (
        K.shape[:3] != (B, n_heads, T)
        or K.shape[4] != d_k
        or V.shape[:3] != (B, n_heads, T)
        or V.shape[3] != nh
        or tuple(beta.shape) != (B, n_heads, T, nh)
        or tuple(H0.shape) != (B, n_heads, d_k, d_v)
    ):
        raise ScanFallback(
            "DeltaMinGRU Triton kernel shapes disagree: from Q=(B, n_heads, T, "
            f"d_k)={tuple(Q.shape)} expected K={(B, n_heads, T, nh, d_k)}, "
            f"V={(B, n_heads, T, nh, d_v)}, beta={(B, n_heads, T, nh)}, "
            f"H0={(B, n_heads, d_k, d_v)}; got K={tuple(K.shape)}, "
            f"V={tuple(V.shape)}, beta={tuple(beta.shape)}, H0={tuple(H0.shape)}"
        )
    reason = _delta_envelope_reason(
        is_cuda=Q.is_cuda and K.is_cuda and V.is_cuda and beta.is_cuda and H0.is_cuda,
        q_dtype=Q.dtype,
        k_dtype=K.dtype,
        v_dtype=V.dtype,
        beta_dtype=beta.dtype,
        h0_dtype=H0.dtype,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
        nh=nh,
    )
    if reason is not None:
        raise ScanFallback(reason)


# Populated below only when Triton is importable (see ``_HAS_TRITON``).
# Maps a scan-function name to a Triton-backed callable with the same
# signature/contract as its eager counterpart in min_gru.py. Absent on a
# CPU-only install (or any machine without Triton), so ``_dispatch_scan``
# falls back to eager under ``auto`` and raises under ``triton``.
SCAN_IMPLS: dict = {}


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError as _exc:  # pragma: no cover - exercised only off-GPU
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = _exc


if _HAS_TRITON:
    from torch.library import register_autograd, triton_op, wrap_triton

    # fp32 contraction / auto-MMA guard (validated on an NVIDIA L4, sm_89,
    # torch 2.8 Triton). Triton auto-lowers a
    # ``tl.sum(a[:, :, None] * b[None, :, :], axis=...)`` contraction to a
    # tensor-core MMA whenever BOTH output (free) dims are >= 16. That MMA
    # (a) defaults to TF32 -- ~2^-10 relative error, ~2e-4 absolute at the
    # scaled inputs here, which breaks the flat 1e-5 fp32 gate -- when the
    # contraction dim is also >= 16, and (b) HARD-CRASHES the compile with
    # ``LLVM ERROR: Unsupported DotOp`` when the contraction dim is < 16.
    # Both bite only at the envelope's ``k = 16`` (and, for the injection /
    # ``v``-dim contractions, only when ``v = 16`` as well). Fix: route the
    # MMA-shaped contractions (all three matmul dims >= 16, so the dot is
    # both needed and legal) through an explicit ``tl.dot`` with
    # ``input_precision="ieee"`` (bit-exact fp32 on this arch), and keep the
    # exact elementwise-sum idiom for every case with a dim < 16, where it
    # stays a correct FMA reduction and never triggers the MMA path. The
    # constexpr ``k``/``v`` guards compile to a single branch per kernel
    # specialization -- no runtime cost.

    @triton.jit
    def _affine_scan_fwd_kernel(
        A_ptr,
        B_ptr,
        Abar_ptr,
        Bbar_ptr,
        T,
        n,
        k: tl.constexpr,
        v: tl.constexpr,
    ):
        """Sequential-in-T forward prefix scan for one (batch, block) lane.

        Contiguous layout: ``A`` is ``(B, T, n, k, k)`` and ``B`` is
        ``(B, T, n, k, v)``. One program owns a single ``(b, n)`` lane and
        carries the running prefix ``(Abar, Bbar)`` in registers across
        the whole T sequence, writing both per-step outputs. All arithmetic
        is fp32; stores cast to the output dtype. Because every envelope
        ``k``/``v`` is a power of two, the ``k``/``v`` tiles are exact and
        need no masking.
        """
        sA_t = n * k * k  # elements between consecutive timesteps in A / Abar
        sB_t = n * k * v  # elements between consecutive timesteps in B / Bbar

        lane = tl.program_id(0)
        b = lane // n
        ni = lane % n

        i = tl.arange(0, k)
        j = tl.arange(0, k)
        w = tl.arange(0, v)

        a_off = i[:, None] * k + j[None, :]  # (k, k), element [i, j] = i*k + j
        b_off = i[:, None] * v + w[None, :]  # (k, v), element [i, j] = i*v + j

        a_base = b * (T * n * k * k) + ni * (k * k)
        bb_base = b * (T * n * k * v) + ni * (k * v)

        A_ptrs = A_ptr + a_base + a_off
        B_ptrs = B_ptr + bb_base + b_off
        Abar_ptrs = Abar_ptr + a_base + a_off
        Bbar_ptrs = Bbar_ptr + bb_base + b_off

        # Prefix init: Abar_0 = I (k x k identity), Bbar_0 = 0.
        Abar = tl.where(i[:, None] == j[None, :], 1.0, 0.0)
        Bbar = tl.zeros((k, v), dtype=tl.float32)

        for _ in range(T):
            a = tl.load(A_ptrs).to(tl.float32)  # (k, k) = A_t, indexed [i, p]
            bt = tl.load(B_ptrs).to(tl.float32)  # (k, v) = B_t

            # Composition order A_current @ A_earlier:
            #   Abar_t[i, j] = sum_p A_t[i, p] * Abar_{t-1}[p, j]
            #   Bbar_t[i, j] = sum_p A_t[i, p] * Bbar_{t-1}[p, j] + B_t[i, j]
            # See the auto-MMA / TF32 note above: the (k, k, k) Abar product
            # auto-MMAs at k=16; the (k, v, k) Bbar product only when v=16 too.
            if k >= 16:
                Abar = tl.dot(a, Abar, input_precision="ieee")
            else:
                Abar = tl.sum(a[:, :, None] * Abar[None, :, :], axis=1)
            if k >= 16 and v >= 16:
                Bbar = tl.dot(a, Bbar, input_precision="ieee") + bt
            else:
                Bbar = tl.sum(a[:, :, None] * Bbar[None, :, :], axis=1) + bt

            tl.store(Abar_ptrs, Abar)
            tl.store(Bbar_ptrs, Bbar)

            A_ptrs += sA_t
            B_ptrs += sB_t
            Abar_ptrs += sA_t
            Bbar_ptrs += sB_t

    @triton.jit
    def _linear_scan_fwd_kernel(
        a_ptr,
        b_ptr,
        A_ptr,
        Bc_ptr,
        T,
        D,
        BLOCK_D: tl.constexpr,
    ):
        """Channel-tiled k=1 forward scan for ``h_t = a_t * h_{t-1} + b_t``.

        Contiguous layout ``(B, T, D)``. Grid is ``(B, ceil(D/BLOCK_D))``;
        each program walks the full T sequence for one batch row and one
        BLOCK_D-wide channel tile, carrying the scalar prefix per channel.
        This is the ``k = v = 1`` special case of the generic kernel, kept
        separate so the lanes stay wide (one program handles BLOCK_D
        channels instead of one channel). fp32 accumulation; stores cast to
        the output dtype.
        """
        b = tl.program_id(0)
        dblk = tl.program_id(1)

        offs = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D

        base = b * (T * D) + offs
        a_ptrs = a_ptr + base
        b_ptrs = b_ptr + base
        A_ptrs = A_ptr + base
        Bc_ptrs = Bc_ptr + base

        # Prefix init: A_0 = 1 (1x1 identity), Bc_0 = 0.
        Aacc = tl.full((BLOCK_D,), 1.0, tl.float32)
        Bacc = tl.zeros((BLOCK_D,), tl.float32)

        for _ in range(T):
            av = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
            bv = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)

            Aacc = av * Aacc
            Bacc = av * Bacc + bv

            tl.store(A_ptrs, Aacc, mask=mask)
            tl.store(Bc_ptrs, Bacc, mask=mask)

            a_ptrs += D
            b_ptrs += D
            A_ptrs += D
            Bc_ptrs += D

    @triton.jit
    def _parallel_scan_log_fwd_kernel(
        lc_ptr,
        lv_ptr,
        h_ptr,
        T,
        D,
        BLOCK_D: tl.constexpr,
    ):
        """Channel-tiled fused log-space scan (Kernel 2).

        Solves ``h_t = a_t * h_{t-1} + b_t`` in log space (Heinsen 2023),
        given ``log_coeffs = log(a_t)`` ``(B, T, D)`` and ``log_values``
        ``(B, T+1, D)`` whose slot 0 is ``log(h_0)`` and slots ``1..T`` are
        ``log(b_t)``. Mirrors the eager reference
        (``min_gru.parallel_scan_log``)::

            a_star = pad(cumsum(log_coeffs))          # (B, T+1, D), a_star_0 = 0
            log_h  = a_star + logcumsumexp(log_values - a_star)
            h      = exp(log_h)[:, 1:]

        but never materializes the ``(B, T+1, D)`` intermediates: one
        program owns one batch row and one BLOCK_D-wide channel tile and
        walks ``m = 0..T``, carrying the running cumsum ``a_star`` and an
        online max-shifted log-sum-exp accumulator ``(m_run, s_run)`` over
        the shifted terms ``e_m = log_values[m] - a_star_m``. The max shift
        matches the eager path's numerical structure so the fp32 outputs
        stay parity-tight. ``m = 0`` seeds the accumulator (that term is
        the ``h_0`` contribution and is never emitted); steps ``m = 1..T``
        write ``h_{m} = exp(a_star_m + m_run + log(s_run))`` to output
        index ``m - 1``. fp32 accumulation; the store casts to the output
        dtype. Every non-power-of-two ``T`` works: the loop is exact and
        needs no padding.
        """
        b = tl.program_id(0)
        dblk = tl.program_id(1)

        offs = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D

        lc_ptrs = lc_ptr + b * (T * D) + offs
        lv_ptrs = lv_ptr + b * ((T + 1) * D) + offs
        h_ptrs = h_ptr + b * (T * D) + offs

        # m = 0: a_star_0 = 0, e_0 = log_values[0] - 0. Seed the online
        # log-sum-exp with this single term (m_run = e_0, s_run = 1). This
        # term carries h_0 into the scan; it is never emitted.
        a_star = tl.zeros((BLOCK_D,), tl.float32)
        m_run = tl.load(lv_ptrs, mask=mask, other=0.0).to(tl.float32)
        s_run = tl.full((BLOCK_D,), 1.0, tl.float32)
        lv_ptrs += D

        for _ in range(T):
            lc = tl.load(lc_ptrs, mask=mask, other=0.0).to(tl.float32)
            a_star = a_star + lc  # a_star_m = a_star_{m-1} + log_coeffs[m-1]
            lv = tl.load(lv_ptrs, mask=mask, other=0.0).to(tl.float32)
            e = lv - a_star  # e_m = log_values[m] - a_star_m

            new_m = tl.maximum(m_run, e)
            s_run = s_run * tl.exp(m_run - new_m) + tl.exp(e - new_m)
            m_run = new_m

            # log_h_m = a_star_m + logsumexp_{i<=m}(e_i) = a_star_m + m_run + log(s_run)
            log_h = a_star + m_run + tl.log(s_run)
            tl.store(h_ptrs, tl.exp(log_h), mask=mask)

            lc_ptrs += D
            lv_ptrs += D
            h_ptrs += D

    @triton.jit
    def _affine_scan_bwd_kernel(
        A_ptr,
        Abar_ptr,
        Bbar_ptr,
        gA_ptr,
        gB_ptr,
        dA_ptr,
        dB_ptr,
        T,
        n,
        k: tl.constexpr,
        v: tl.constexpr,
    ):
        """Reverse-direction adjoint of the generic affine prefix scan.

        The exact backward of ``_affine_scan_fwd_kernel``. For the forward
        recurrence ``Abar_t = A_t @ Abar_{t-1}`` (``Abar_0 = I``) and
        ``Bbar_t = A_t @ Bbar_{t-1} + B_t`` (``Bbar_0 = 0``), with incoming
        output grads ``G^A_t = dL/dAbar_t`` and ``G^B_t = dL/dBbar_t``, the
        adjoint recurrence is::

            Ghat^A_t = G^A_t + A_{t+1}^T @ Ghat^A_{t+1}   (reverse scan)
            Ghat^B_t = G^B_t + A_{t+1}^T @ Ghat^B_{t+1}   (Ghat_{T+1} = 0)
            dL/dA_t  = Ghat^A_t @ Abar_{t-1}^T + Ghat^B_t @ Bbar_{t-1}^T
            dL/dB_t  = Ghat^B_t

        with the same ``Abar_0 = I``, ``Bbar_0 = 0`` seed used at ``t = 1``.
        One program owns a single ``(b, n)`` lane and walks ``t = T .. 1``,
        carrying the running adjoint ``(Ghat^A, Ghat^B)`` and the previous
        step's transition ``A_{t+1}`` in registers -- so it reads ONLY the
        forward's inputs (``A``) and outputs (``Abar``, ``Bbar``); it never
        touches ``B`` and saves no extra tensors (the reversed-scan design's
        whole point). All arithmetic is fp32; stores cast to the output
        dtype. Contractions mirror the forward kernel's broadcast idiom:
        ``(A_next^T @ Ghat)[p, j] = sum_i A_next[i, p] * Ghat[i, j]`` and
        ``(Ghat @ Abar_prev^T)[i, j] = sum_p Ghat[i, p] * Abar_prev[j, p]``.
        """
        sA_t = n * k * k  # elements between consecutive timesteps in A / Abar
        sB_t = n * k * v  # elements between consecutive timesteps in B / Bbar

        lane = tl.program_id(0)
        b = lane // n
        ni = lane % n

        i = tl.arange(0, k)
        j = tl.arange(0, k)
        w = tl.arange(0, v)

        a_off = i[:, None] * k + j[None, :]  # (k, k)
        b_off = i[:, None] * v + w[None, :]  # (k, v)

        a_base = b * (T * n * k * k) + ni * (k * k)
        bb_base = b * (T * n * k * v) + ni * (k * v)

        # Start at the last timestep t = T (0-based index T-1). The "prev"
        # pointers lag one step behind (index t-2 == c-1); at c == 0 they
        # address index -1 and are never dereferenced (masked load below).
        last_A = a_base + (T - 1) * sA_t
        last_B = bb_base + (T - 1) * sB_t
        A_ptrs = A_ptr + last_A + a_off
        gA_ptrs = gA_ptr + last_A + a_off
        gB_ptrs = gB_ptr + last_B + b_off
        dA_ptrs = dA_ptr + last_A + a_off
        dB_ptrs = dB_ptr + last_B + b_off
        Abar_prev_ptrs = Abar_ptr + a_base + (T - 2) * sA_t + a_off
        Bbar_prev_ptrs = Bbar_ptr + bb_base + (T - 2) * sB_t + b_off

        ident = tl.where(i[:, None] == j[None, :], 1.0, 0.0)
        GhatA = tl.zeros((k, k), dtype=tl.float32)
        GhatB = tl.zeros((k, v), dtype=tl.float32)
        # A_{t+1}, cached from the previous (higher-t) iteration. Seeded to 0
        # so the first iteration's transition term (Ghat carry is also 0)
        # vanishes -- i.e. Ghat_T = G^A_T with no A_{T+1}.
        A_next = tl.zeros((k, k), dtype=tl.float32)
        c = T - 1

        for _ in range(T):
            gA = tl.load(gA_ptrs).to(tl.float32)  # G^A_t (k, k)
            gB = tl.load(gB_ptrs).to(tl.float32)  # G^B_t (k, v)

            # Reverse-scan transition: Ghat_t = G_t + A_{t+1}^T @ Ghat_{t+1}.
            # See the auto-MMA / TF32 note above: the (k, k, k) A^T@GhatA
            # product auto-MMAs at k=16; the (k, v, k) A^T@GhatB only when
            # v=16 too. `tl.dot` has no transpose flag, so A^T is materialized
            # with `tl.trans`; the else-branch keeps the exact axis-0 reduce.
            if k >= 16:
                AtGA = tl.dot(tl.trans(A_next), GhatA, input_precision="ieee")
            else:
                AtGA = tl.sum(A_next[:, :, None] * GhatA[:, None, :], axis=0)  # (k, k)
            if k >= 16 and v >= 16:
                AtGB = tl.dot(tl.trans(A_next), GhatB, input_precision="ieee")
            else:
                AtGB = tl.sum(A_next[:, :, None] * GhatB[:, None, :], axis=0)  # (k, v)
            GhatA = gA + AtGA
            GhatB = gB + AtGB

            # Abar_{t-1}, Bbar_{t-1}: forward outputs shifted by one, seeded
            # I / 0 at t = 1 (c == 0). `prev` keeps the c == 0 load off the
            # out-of-bounds index -1; where() then installs the identity seed
            # (Bbar's seed is 0, already the masked `other`). No d-tile bounds
            # mask is needed here (unlike the k=1 kernel's channel-tile mask):
            # every envelope k/v is a power of two, so the (k, k)/(k, v) tiles
            # this program addresses are always exact -- `prev` is the only
            # real condition, not a conjunction with an always-true bound.
            prev = c > 0
            Abar_prev = tl.load(Abar_prev_ptrs, mask=prev, other=0.0).to(tl.float32)
            Abar_prev = tl.where(prev, Abar_prev, ident)
            Bbar_prev = tl.load(Bbar_prev_ptrs, mask=prev, other=0.0).to(tl.float32)

            # dL/dA_t = Ghat^A_t @ Abar_{t-1}^T + Ghat^B_t @ Bbar_{t-1}^T.
            # See the auto-MMA / TF32 note above: the first term (k, k, k)
            # auto-MMAs at k=16; the second contracts over v (K=v), so it
            # only auto-MMAs when v=16 too. `tl.dot` needs the ^T explicit.
            if k >= 16:
                dA_A = tl.dot(GhatA, tl.trans(Abar_prev), input_precision="ieee")
            else:
                dA_A = tl.sum(GhatA[:, None, :] * Abar_prev[None, :, :], axis=2)
            if k >= 16 and v >= 16:
                dA_B = tl.dot(GhatB, tl.trans(Bbar_prev), input_precision="ieee")
            else:
                dA_B = tl.sum(GhatB[:, None, :] * Bbar_prev[None, :, :], axis=2)
            dA = dA_A + dA_B
            tl.store(dA_ptrs, dA)
            tl.store(dB_ptrs, GhatB)  # dL/dB_t = Ghat^B_t

            # Cache A_t as A_{t+1} for the next (lower) index t-1.
            A_next = tl.load(A_ptrs).to(tl.float32)

            c -= 1
            A_ptrs -= sA_t
            gA_ptrs -= sA_t
            gB_ptrs -= sB_t
            dA_ptrs -= sA_t
            dB_ptrs -= sB_t
            Abar_prev_ptrs -= sA_t
            Bbar_prev_ptrs -= sB_t

    @triton.jit
    def _linear_scan_bwd_kernel(
        a_ptr,
        A_ptr,
        Bc_ptr,
        gA_ptr,
        gB_ptr,
        da_ptr,
        db_ptr,
        T,
        D,
        BLOCK_D: tl.constexpr,
    ):
        """Channel-tiled k=1 adjoint of ``_linear_scan_fwd_kernel``.

        The scalar (k = v = 1) specialization of ``_affine_scan_bwd_kernel``,
        kept separate for the same wide-lane reason the forward is: one
        program walks the full T sequence for one batch row and one BLOCK_D
        channel tile. For ``A_t = a_t * A_{t-1}`` (``A_0 = 1``) and
        ``Bc_t = a_t * Bc_{t-1} + b_t`` (``Bc_0 = 0``) with incoming grads
        ``gA_t``, ``gB_t``::

            Ghat^A_t = gA_t + a_{t+1} * Ghat^A_{t+1}   (reverse scan)
            Ghat^B_t = gB_t + a_{t+1} * Ghat^B_{t+1}
            da_t     = Ghat^A_t * A_{t-1} + Ghat^B_t * Bc_{t-1}   (A_0=1, Bc_0=0)
            db_t     = Ghat^B_t

        Reads only the forward input ``a`` and outputs ``A``, ``Bc``. fp32
        accumulation; stores cast to the output dtype.
        """
        b = tl.program_id(0)
        dblk = tl.program_id(1)

        offs = dblk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D

        base = b * (T * D) + offs
        last = base + (T - 1) * D
        a_ptrs = a_ptr + last
        gA_ptrs = gA_ptr + last
        gB_ptrs = gB_ptr + last
        da_ptrs = da_ptr + last
        db_ptrs = db_ptr + last
        Aprev_ptrs = A_ptr + base + (T - 2) * D  # A_{t-1}
        Bcprev_ptrs = Bc_ptr + base + (T - 2) * D  # Bc_{t-1}

        GhatA = tl.zeros((BLOCK_D,), tl.float32)
        GhatB = tl.zeros((BLOCK_D,), tl.float32)
        a_next = tl.zeros((BLOCK_D,), tl.float32)  # a_{t+1}
        c = T - 1

        for _ in range(T):
            gA = tl.load(gA_ptrs, mask=mask, other=0.0).to(tl.float32)
            gB = tl.load(gB_ptrs, mask=mask, other=0.0).to(tl.float32)
            GhatA = gA + a_next * GhatA
            GhatB = gB + a_next * GhatB

            prev = c > 0
            Aprev = tl.load(Aprev_ptrs, mask=mask & prev, other=0.0).to(tl.float32)
            Aprev = tl.where(prev, Aprev, 1.0)  # A_0 = 1 seed at t = 1
            Bcprev = tl.load(Bcprev_ptrs, mask=mask & prev, other=0.0).to(tl.float32)

            da = GhatA * Aprev + GhatB * Bcprev
            tl.store(da_ptrs, da, mask=mask)
            tl.store(db_ptrs, GhatB, mask=mask)

            a_next = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)

            c -= 1
            a_ptrs -= D
            gA_ptrs -= D
            gB_ptrs -= D
            da_ptrs -= D
            db_ptrs -= D
            Aprev_ptrs -= D
            Bcprev_ptrs -= D

    @triton.jit
    def _angle_scan_fwd_kernel(
        theta_ptr,
        scale_ptr,
        gamma_ptr,
        b_ptr,
        h0_ptr,
        perm_ptr,
        sgn_ptr,
        p2p_ptr,
        out_ptr,
        T,
        n,
        k: tl.constexpr,
        R: tl.constexpr,
        half: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        HAS_DECAY: tl.constexpr,
    ):
        """Angle-fused forward recurrence for one (batch, block) lane (Kernel 4).

        Carries the state ``h`` (a ``k``-vector) in registers across the whole
        T sequence, applying per step the per-block scale (rotation: multiply
        dim 1 by ``tanh(u)``; Givens: none), then ``rounds`` of factored
        brick-wall plane rotations, then the decay ``gamma`` and injection
        ``b`` -- exactly ``h_t = gamma_t * R(theta_t) * S_t * h_{t-1} + b_t``.
        It NEVER materializes or scans the ``k x k`` transition matrices the
        eager ``matrix_affine_scan`` path is defined over: each round applies a
        structured plane rotation directly to the state vector.

        Layouts (contiguous): ``theta`` ``(B,T,n,R,half)``; ``scale``/``gamma``
        ``(B,T,n)``; ``b``/``out`` ``(B,T,n,k)``; ``h0`` ``(B,n,k)``;
        ``perm``/``sgn``/``p2p`` ``(R,k)``. ``perm[r]`` is the partner index of
        each position within round ``r``'s planes, ``sgn[r]`` the ``-1``/``+1``
        first/second-member sign, ``p2p[r]`` each position's angle index among
        the round's ``half`` angles. A round applies, per position ``p``,
        ``v'[p] = cos_p * v[p] + sgn[p] * sin_p * v[perm[p]]`` (the ``i``/``j``
        pair rotation), with ``cos_p``/``sin_p`` gathered from the ``half``
        per-plane angles via the ``p2p`` selection. fp32 throughout; the store
        casts to the output dtype. All of ``k``, ``R``, ``half`` are powers-of-
        two-friendly constexprs, so every tile is exact and needs no masking.
        """
        lane = tl.program_id(0)
        b = lane // n
        ni = lane % n

        ar_k = tl.arange(0, k)
        ar_h = tl.arange(0, half)

        sc_base = b * (T * n) + ni
        bk_base = b * (T * n * k) + ni * k
        th_base = b * (T * n * R * half) + ni * (R * half)
        h0_base = b * (n * k) + ni * k

        v = tl.load(h0_ptr + h0_base + ar_k).to(tl.float32)  # h_0 (k,)

        sc_t = sc_base
        bk_t = bk_base
        th_t = th_base
        for _t in range(T):
            if HAS_SCALE:  # k == 2: scale dim 1 by d = tanh(u), dim 0 unchanged
                d = tl.load(scale_ptr + sc_t).to(tl.float32)
                v = tl.where(ar_k == 1, d * v, v)
            for r in range(R):
                th = tl.load(theta_ptr + th_t + r * half + ar_h).to(tl.float32)
                ch = tl.cos(th)  # (half,)
                sh = tl.sin(th)
                p2p = tl.load(p2p_ptr + r * k + ar_k)  # (k,)
                sel = (p2p[:, None] == ar_h[None, :]).to(tl.float32)  # (k, half)
                cos_pos = tl.sum(sel * ch[None, :], axis=1)  # (k,)
                sin_pos = tl.sum(sel * sh[None, :], axis=1)
                perm_r = tl.load(perm_ptr + r * k + ar_k)  # (k,)
                sgn_r = tl.load(sgn_ptr + r * k + ar_k).to(tl.float32)
                Pm = (perm_r[:, None] == ar_k[None, :]).to(tl.float32)  # (k, k)
                vp = tl.sum(Pm * v[None, :], axis=1)  # v[perm_r] (k,)
                v = cos_pos * v + sgn_r * sin_pos * vp
            if HAS_DECAY:
                g = tl.load(gamma_ptr + sc_t).to(tl.float32)
                v = g * v
            bt = tl.load(b_ptr + bk_t + ar_k).to(tl.float32)
            v = v + bt  # h_t
            tl.store(out_ptr + bk_t + ar_k, v)
            sc_t += n
            bk_t += n * k
            th_t += n * R * half

    @triton.jit
    def _angle_scan_bwd_kernel(
        theta_ptr,
        scale_ptr,
        gamma_ptr,
        b_ptr,
        h0_ptr,
        out_ptr,
        gout_ptr,
        perm_ptr,
        sgn_ptr,
        p2p_ptr,
        gtheta_ptr,
        gscale_ptr,
        ggamma_ptr,
        gb_ptr,
        gh0_ptr,
        T,
        n,
        k: tl.constexpr,
        R: tl.constexpr,
        half: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        HAS_DECAY: tl.constexpr,
    ):
        """Exact stored-state reversible backward of ``_angle_scan_fwd_kernel``.

        One program owns a ``(b, n)`` lane and walks ``t = T .. 1``, carrying
        the state adjoint ``ghat`` (the future steps' contribution to
        ``dL/dh_t``) in registers. ``h_{t-1}`` is read EXACTLY every step from
        the forward output ``out`` (all states, which the module returns
        anyway) -- or from ``h0`` at ``t = 1`` -- never reconstructed by
        inverting the forward step. An earlier design divided by ``gamma``/
        ``tanh(u)`` to reverse across a multi-step checkpoint chunk; a blind
        CPU probe showed that division amplifies roundoff by roughly
        ``sigma_min^{-chunklen}`` (sigma_min = gamma * min(1, |scale|)), blowing
        past the grad tolerance even at ordinary decay/init strengths (e.g. the
        ``GivensMinGRU`` class-default ``decay_rate=1.0``). The user's ruling
        (see the Task-5 report, "Fix round 2") rejected division-based reversal
        entirely: every ``h_{t-1}`` is the stored forward state, so no such
        amplification can occur, at the cost of the forward output being the
        only "checkpoint" (already required as the module's return value, so
        no extra memory).

        Given ``h_{t-1}`` it recomputes the per-round forward states, forms the
        total adjoint ``gbar = grad_out_t + ghat`` on ``h_t``, and accumulates
        ``dL/db`` (= ``gbar``), ``dL/dgamma`` (= ``sum(gbar * rot)``), the
        per-plane ``dL/dtheta`` (reduced from per-position grads via the same
        ``p2p`` selection), and ``dL/d(scale)`` (rotation only). The adjoint
        propagated to ``h_{t-1}`` (= ``M_t^T gbar``) becomes the next step's
        ``ghat``; after ``t = 1`` it is ``dL/dh_0``. fp32 throughout.
        """
        lane = tl.program_id(0)
        b = lane // n
        ni = lane % n
        ar_k = tl.arange(0, k)
        ar_h = tl.arange(0, half)

        sc_base = b * (T * n) + ni
        bk_base = b * (T * n * k) + ni * k
        th_base = b * (T * n * R * half) + ni * (R * half)
        h0_base = b * (n * k) + ni * k

        ghat = tl.zeros((k,), tl.float32)

        # Walk t = T .. 1 with a forward loop + computed reverse index (the
        # proven Kernel-3 idiom; avoids a runtime negative-step range).
        for _step in range(T):
            it = T - 1 - _step

            # h_{t-1}: the stored forward output at it-1, or h0 at t = 1 --
            # always an exact read, never an inverted reconstruction.
            if it == 0:
                hprev = tl.load(h0_ptr + h0_base + ar_k).to(tl.float32)
            else:
                hprev = tl.load(out_ptr + bk_base + (it - 1) * (n * k) + ar_k).to(tl.float32)

            # --- recompute forward per-round states vv[0..R] from h_{t-1} ---
            if HAS_SCALE:
                d = tl.load(scale_ptr + sc_base + it * n).to(tl.float32)
                v0s = tl.where(ar_k == 1, d * hprev, hprev)
            else:
                v0s = hprev
            vv = [v0s]
            # `tl.static_range`, NOT `range`: this loop indexes the Python
            # list `vv` by the loop variable (`vv[r]`) and grows it per round,
            # both of which require `r` to be a compile-time int. A plain
            # `range(R)` (even with constexpr R) is lowered to a runtime
            # `scf.for`, making `r` a runtime value and `vv[r]` a Python-list
            # index by a Triton value -- a frontend AssertionError.
            # `static_range` unrolls at compile time so `r` is a Python int.
            # The list is grown by CONCATENATION (`vv = vv + [..]`), not
            # `vv.append(..)`: Triton's static-unrolled list type rejects
            # in-place `.append` ("'append' is not in list") but supports
            # `+`. (The forward kernel's R-loop carries a single state and
            # needs no list, so it can stay a plain `range`.)
            for r in tl.static_range(R):
                th = tl.load(theta_ptr + th_base + it * (n * R * half) + r * half + ar_h).to(
                    tl.float32
                )
                ch = tl.cos(th)
                sh = tl.sin(th)
                p2p = tl.load(p2p_ptr + r * k + ar_k)
                sel = (p2p[:, None] == ar_h[None, :]).to(tl.float32)
                cos_pos = tl.sum(sel * ch[None, :], axis=1)
                sin_pos = tl.sum(sel * sh[None, :], axis=1)
                perm_r = tl.load(perm_ptr + r * k + ar_k)
                sgn_r = tl.load(sgn_ptr + r * k + ar_k).to(tl.float32)
                Pm = (perm_r[:, None] == ar_k[None, :]).to(tl.float32)
                vr = vv[r]
                vp = tl.sum(Pm * vr[None, :], axis=1)
                vv = vv + [cos_pos * vr + sgn_r * sin_pos * vp]
            rot = vv[R]

            # --- adjoints ---
            gout = tl.load(gout_ptr + bk_base + it * (n * k) + ar_k).to(tl.float32)
            gbar = gout + ghat  # dL/dh_t
            tl.store(gb_ptr + bk_base + it * (n * k) + ar_k, gbar)
            if HAS_DECAY:
                g = tl.load(gamma_ptr + sc_base + it * n).to(tl.float32)
                tl.store(ggamma_ptr + sc_base + it * n, tl.sum(gbar * rot))
                a = g * gbar  # adjoint on rot = vv[R]
            else:
                a = gbar

            # `tl.static_range` with a DOWN-counting loop variable so `rr`
            # itself is the bare loop var (R-1 .. 0). Indexing the Python list
            # `vv[rr]` with the bare loop variable is the only list-subscript
            # form Triton accepts inside this runtime `for _step` loop body:
            # a COMPUTED index such as `vv[R - 1 - r]` (R is a constexpr) is
            # rejected with a frontend AssertionError here, even though it
            # compiles fine outside a runtime loop. Down-counting the range
            # sidesteps that entirely and doubles as the pointer/load index.
            for rr in tl.static_range(R - 1, -1, -1):
                th = tl.load(theta_ptr + th_base + it * (n * R * half) + rr * half + ar_h).to(
                    tl.float32
                )
                ch = tl.cos(th)
                sh = tl.sin(th)
                p2p = tl.load(p2p_ptr + rr * k + ar_k)
                sel = (p2p[:, None] == ar_h[None, :]).to(tl.float32)
                cos_pos = tl.sum(sel * ch[None, :], axis=1)
                sin_pos = tl.sum(sel * sh[None, :], axis=1)
                perm_r = tl.load(perm_ptr + rr * k + ar_k)
                sgn_r = tl.load(sgn_ptr + rr * k + ar_k).to(tl.float32)
                Pm = (perm_r[:, None] == ar_k[None, :]).to(tl.float32)
                vr = vv[rr]
                vp = tl.sum(Pm * vr[None, :], axis=1)
                # dL/dtheta per position, then reduce to the round's `half` angles
                gtp = a * (-sin_pos * vr + sgn_r * cos_pos * vp)  # (k,)
                gth_half = tl.sum(sel * gtp[:, None], axis=0)  # (half,)
                tl.store(
                    gtheta_ptr + th_base + it * (n * R * half) + rr * half + ar_h,
                    gth_half,
                )
                # adjoint on vv[rr]: cos_pos * a + gather(sgn * sin_pos * a, perm)
                tmp = sgn_r * sin_pos * a
                a = cos_pos * a + tl.sum(Pm * tmp[None, :], axis=1)

            if HAS_SCALE:
                d = tl.load(scale_ptr + sc_base + it * n).to(tl.float32)
                # v0s = (hprev0, d*hprev1): dL/dd = a1 * hprev1; adjoint on hprev
                gd = tl.sum(tl.where(ar_k == 1, a * hprev, 0.0))
                tl.store(gscale_ptr + sc_base + it * n, gd)
                a = tl.where(ar_k == 1, a * d, a)
            ghat = a

        tl.store(gh0_ptr + h0_base + ar_k, ghat)  # dL/dh_0

    @triton_op("mingru_scans::affine_scan_fwd", mutates_args={})
    def affine_scan_fwd(A: torch.Tensor, Bm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generic k x k affine prefix scan (Kernel 1).

        ``A`` is ``(B, T, n, k, k)``, ``Bm`` is ``(B, T, n, k, v)``. Returns
        ``(Abar, Bbar)`` with the same shapes, where ``Abar_t`` is the
        running matrix product of ``A`` (composition ``A_current @
        A_earlier``) and ``Bbar_t`` the ``H_0 = 0`` solution of
        ``H_t = A_t @ H_{t-1} + B_t``.

        The functional body (``.contiguous`` + ``empty_like`` + a single
        ``wrap_triton`` launch) is what ``triton_op`` traces for the
        fake-tensor meta, so ``torch.compile`` gets correct output shapes
        with no graph break.
        """
        A = A.contiguous()
        Bm = Bm.contiguous()
        b, t, n, k, _ = A.shape
        v = Bm.shape[-1]
        Abar = torch.empty_like(A)
        Bbar = torch.empty_like(Bm)
        grid = (b * n,)
        wrap_triton(_affine_scan_fwd_kernel)[grid](A, Bm, Abar, Bbar, t, n, k, v)
        return Abar, Bbar

    @triton_op("mingru_scans::linear_scan_fwd", mutates_args={})
    def linear_scan_fwd(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Channel-tiled k=1 affine prefix scan (Kernel 1).

        ``a`` and ``b`` are ``(B, T, D)``. Returns ``(A, Bc)`` of the same
        shape, where ``A_t`` is the running product of ``a`` and ``Bc_t``
        the ``h_0 = 0`` solution of ``h_t = a_t * h_{t-1} + b_t``. Same
        traced-body fake-meta contract as ``affine_scan_fwd``.
        """
        a = a.contiguous()
        b = b.contiguous()
        bsz, t, d = a.shape
        A = torch.empty_like(a)
        Bc = torch.empty_like(b)
        grid = (bsz, triton.cdiv(d, _LINEAR_BLOCK_D))
        wrap_triton(_linear_scan_fwd_kernel)[grid](a, b, A, Bc, t, d, _LINEAR_BLOCK_D)
        return A, Bc

    @triton_op("mingru_scans::parallel_scan_log_fwd", mutates_args={})
    def parallel_scan_log_fwd(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
        """Fused log-space forward scan (Kernel 2).

        ``log_coeffs`` is ``(B, T, D)`` and ``log_values`` is ``(B, T+1,
        D)`` (slot 0 = ``log(h_0)``, slots ``1..T`` = ``log(b_t)``).
        Returns the states ``h_1..h_T`` as ``(B, T, D)`` -- a single
        tensor, matching ``min_gru.parallel_scan_log`` (unlike the affine
        ops, which return a pair). Same traced-body fake-meta contract as
        ``linear_scan_fwd``: ``.contiguous`` + ``empty_like`` + one
        ``wrap_triton`` launch, so ``torch.compile`` sees the ``(B, T, D)``
        output shape with no graph break.
        """
        log_coeffs = log_coeffs.contiguous()
        log_values = log_values.contiguous()
        bsz, t, d = log_coeffs.shape
        h = torch.empty_like(log_coeffs)
        grid = (bsz, triton.cdiv(d, _LINEAR_BLOCK_D))
        wrap_triton(_parallel_scan_log_fwd_kernel)[grid](
            log_coeffs, log_values, h, t, d, _LINEAR_BLOCK_D
        )
        return h

    @triton_op("mingru_scans::affine_scan_bwd", mutates_args={})
    def affine_scan_bwd(
        A: torch.Tensor,
        Abar: torch.Tensor,
        Bbar: torch.Tensor,
        gAbar: torch.Tensor,
        gBbar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generic k x k affine-scan adjoint (Kernel 3).

        Reads the forward input ``A`` ``(B, T, n, k, k)`` and the forward
        outputs ``Abar`` ``(B, T, n, k, k)`` / ``Bbar`` ``(B, T, n, k, v)``,
        plus the incoming output grads ``gAbar`` / ``gBbar`` (same shapes as
        ``Abar`` / ``Bbar``). Returns ``(dA, dB)`` -- the grads of the loss
        w.r.t. the forward inputs ``A`` and ``Bm`` -- with shapes matching
        ``A`` and ``Bbar``. Same traced-body fake-meta contract as the
        forward ops (``.contiguous`` + ``empty_like`` + one ``wrap_triton``
        launch), so ``torch.compile`` sees correct grad shapes with no graph
        break.
        """
        A = A.contiguous()
        Abar = Abar.contiguous()
        Bbar = Bbar.contiguous()
        gAbar = gAbar.contiguous()
        gBbar = gBbar.contiguous()
        b, t, n, k, _ = A.shape
        v = Bbar.shape[-1]
        dA = torch.empty_like(A)
        dB = torch.empty_like(Bbar)
        grid = (b * n,)
        wrap_triton(_affine_scan_bwd_kernel)[grid](A, Abar, Bbar, gAbar, gBbar, dA, dB, t, n, k, v)
        return dA, dB

    @triton_op("mingru_scans::linear_scan_bwd", mutates_args={})
    def linear_scan_bwd(
        a: torch.Tensor,
        A: torch.Tensor,
        Bc: torch.Tensor,
        gA: torch.Tensor,
        gBc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Channel-tiled k=1 affine-scan adjoint.

        Reads the forward input ``a`` ``(B, T, D)`` and outputs ``A`` /
        ``Bc`` ``(B, T, D)``, plus incoming grads ``gA`` / ``gBc``. Returns
        ``(da, db)``, the grads w.r.t. the forward inputs ``a`` and ``b``.
        Same traced-body fake-meta contract as ``linear_scan_fwd``.
        """
        a = a.contiguous()
        A = A.contiguous()
        Bc = Bc.contiguous()
        gA = gA.contiguous()
        gBc = gBc.contiguous()
        bsz, t, d = a.shape
        da = torch.empty_like(a)
        db = torch.empty_like(Bc)
        grid = (bsz, triton.cdiv(d, _LINEAR_BLOCK_D))
        wrap_triton(_linear_scan_bwd_kernel)[grid](a, A, Bc, gA, gBc, da, db, t, d, _LINEAR_BLOCK_D)
        return da, db

    # --- Autograd registration (Kernel 3; torch 2.8 route) ----------
    #
    # `torch.library.register_autograd` on each forward `triton_op` is the
    # idiomatic torch 2.8 way to make a `triton_op` differentiable: the
    # backward participates in the dispatcher and `torch.compile`'s tracing
    # (an `autograd.Function` wrapping the op would sit outside both). The
    # setup_context saves only forward inputs/outputs; every backward routes
    # to a Triton adjoint kernel except `parallel_scan_log`, which recomputes
    # through the eager log-space math (see below).
    #
    # Second-order autograd (`create_graph=True`) is unsupported for all four
    # ops. For `affine_scan_fwd`/`linear_scan_fwd`, the generated backward
    # differentiates through `affine_scan_bwd`/`linear_scan_bwd` -- Triton
    # ops with no `register_autograd` formula of their own -- so a
    # create_graph=True outer call raises loudly ("no autograd formula was
    # registered", from `torch.library`'s generated backward). The
    # recompute-based `parallel_scan_log` backward has no such op boundary to
    # trip that check (its body is plain differentiable torch ops), so it
    # needs an explicit guard below to fail the same way instead of silently
    # dropping the second-order term -- see `_parallel_scan_log_backward`.
    #
    # None-grad handling: `register_autograd` materializes zero tensors for
    # any output with no downstream consumer (the default
    # `autograd.Function` behavior, since none of these ops calls
    # `ctx.set_materialize_grads(False)` -- verified against torch 2.8:
    # a two-output custom op driven through only one output still receives
    # a concrete zero tensor, never `None`, in the unused slot). The affine
    # and linear backwards below therefore take `grad_Abar`/`grad_Bbar` and
    # `grad_A`/`grad_Bc` as already-materialized tensors with no `is None`
    # branch.

    def _affine_setup_context(ctx, inputs, output):
        """``register_autograd`` setup hook: save what ``_affine_backward`` needs."""
        A, _Bm = inputs
        Abar, Bbar = output
        # Saves the forward input A and forward outputs Abar/Bbar only --
        # the adjoint needs no other tensor (not even Bm), so this reuses
        # already-materialized activations with zero extra memory.
        ctx.save_for_backward(A, Abar, Bbar)

    def _affine_backward(ctx, grad_Abar, grad_Bbar):
        """``register_autograd`` backward for ``affine_scan_fwd``: dispatches to ``affine_scan_bwd``."""
        A, Abar, Bbar = ctx.saved_tensors
        dA, dB = affine_scan_bwd(A, Abar, Bbar, grad_Abar, grad_Bbar)
        return dA, dB

    register_autograd(
        "mingru_scans::affine_scan_fwd",
        _affine_backward,
        setup_context=_affine_setup_context,
    )

    def _linear_setup_context(ctx, inputs, output):
        """``register_autograd`` setup hook: save what ``_linear_backward`` needs."""
        a, _b = inputs
        A, Bc = output
        ctx.save_for_backward(a, A, Bc)

    def _linear_backward(ctx, grad_A, grad_Bc):
        """``register_autograd`` backward for ``linear_scan_fwd``: dispatches to ``linear_scan_bwd``."""
        a, A, Bc = ctx.saved_tensors
        da, db = linear_scan_bwd(a, A, Bc, grad_A, grad_Bc)
        return da, db

    register_autograd(
        "mingru_scans::linear_scan_fwd",
        _linear_backward,
        setup_context=_linear_setup_context,
    )

    def _parallel_scan_log_setup_context(ctx, inputs, output):
        """``register_autograd`` setup hook: save what ``_parallel_scan_log_backward`` needs."""
        log_coeffs, log_values = inputs
        # Autograd-through-recomputation: the log op has no hand-derived
        # adjoint kernel (nothing to get wrong blind). Saving the two forward
        # INPUTS is enough to re-derive the grad exactly through the eager
        # log-space math -- zero extra saved tensors beyond forward inputs
        # (the forward output h is not even retained).
        ctx.save_for_backward(log_coeffs, log_values)

    def _parallel_scan_log_backward(ctx, grad_h):
        """``register_autograd`` backward for ``parallel_scan_log_fwd`` via autograd-through-recomputation."""
        # No `grad_h is None` guard: per the "None-grad handling" note
        # above (this op has a single output, so the same
        # always-materialized-tensor guarantee applies), `grad_h` is never
        # `None` here -- consistent with `_affine_backward`/`_linear_backward`
        # above, which take their materialized grad tensors with no such
        # branch either.
        # Second-order autograd guard: `torch.is_grad_enabled()` here reads
        # the AMBIENT grad mode this backward was entered under (checked
        # before this function's own `torch.enable_grad()` below, which
        # would otherwise mask it). PyTorch runs a backward pass under
        # `no_grad` unless the outer call used `create_graph=True` -- the
        # standard tell for "am I being asked to build a second-order graph"
        # inside any `autograd.Function`/`register_autograd` backward
        # (verified empirically: `is_grad_enabled()` is False for a plain
        # `.backward()`/`torch.autograd.grad(...)` call and True under
        # `create_graph=True`). Without this guard, `torch.autograd.grad`
        # below would run with `create_graph=False` regardless of the outer
        # request and SILENTLY drop the second-order contribution -- unlike
        # the other three ops, which already raise loudly in this scenario
        # because their backwards differentiate through an un-registered
        # Triton op. This guard makes all four ops fail the same way:
        # loudly, never a silent undercount.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "parallel_scan_log's Triton-dispatched backward does not "
                "support second-order autograd (create_graph=True): its "
                "recompute-through-eager-math backward would silently drop "
                "the double-backward term if it proceeded. Use "
                "MINGRU_SCAN=eager (or torch.autograd.grad(..., "
                "create_graph=True) against min_gru.parallel_scan_log "
                "directly) for double-backward through this op."
            )
        log_coeffs, log_values = ctx.saved_tensors
        with torch.enable_grad():
            lc = log_coeffs.detach().requires_grad_(True)
            lv = log_values.detach().requires_grad_(True)
            # Calls the module-level `parallel_scan_log_recompute` (defined
            # above, outside this `if _HAS_TRITON:` block) rather than
            # inlining the formula here -- the CPU-runnable lockstep
            # selftest in the repo-root `triton_scans.py` evidence driver's
            # `__main__` cross-checks THAT function against
            # `min_gru.parallel_scan_log`, so this call site inherits that
            # guarantee instead of maintaining its own unchecked copy.
            h = parallel_scan_log_recompute(lc, lv)
        dlc, dlv = torch.autograd.grad(h, (lc, lv), grad_outputs=grad_h)
        return dlc, dlv

    register_autograd(
        "mingru_scans::parallel_scan_log_fwd",
        _parallel_scan_log_backward,
        setup_context=_parallel_scan_log_setup_context,
    )

    # --- Kernel 4: angle-fused forward + exact stored-state reversible backward
    #
    # A module-level fast path (not one of the four scan ops): `GivensMinGRU`
    # and `RotationMinGRU` route their forwards here on CUDA, passing the raw
    # rotation angles / scale channel / injection / decay / h0 directly, so the
    # k x k transition matrices `matrix_affine_scan` is defined over are never
    # materialized or scanned. `theta` `(B,T,n,R,half)`, `scale`/`gamma`
    # `(B,T,n)`, `b` `(B,T,n,k)`, `h0` `(B,n,k)`, and the plane metadata
    # `perm`/`sgn`/`p2p` `(R,k)`. `scale`/`gamma` are always passed as concrete
    # tensors (ones when the feature is off); the `has_scale`/`has_decay`
    # constexprs gate whether they are read, so the disabled path is exact
    # (no spurious multiply/divide).
    #
    # The backward reads every ``h_{t-1}`` exactly from the forward output
    # (never reconstructs it by dividing out ``gamma``/``tanh(u)``): an earlier
    # design let the caller pick a reversal checkpoint interval C (recomputing
    # C-1 states per chunk via inverse rotation + division), but a blind CPU
    # probe showed that division amplifies roundoff by roughly
    # ``sigma_min^{-chunklen}``, which blew past the grad tolerance even at
    # ordinary decay/init strengths (including ``GivensMinGRU``'s class-default
    # ``decay_rate=1.0``). The user's ruling (Task-5 report, "Fix round 2")
    # rejected division-based reversal entirely, so there is no interval
    # parameter left in this op's signature -- reversal cannot be re-enabled by
    # any caller.

    @triton_op("mingru_scans::angle_scan_fwd", mutates_args={})
    def angle_scan_fwd(
        theta: torch.Tensor,
        scale: torch.Tensor,
        gamma: torch.Tensor,
        b: torch.Tensor,
        h0: torch.Tensor,
        perm: torch.Tensor,
        sgn: torch.Tensor,
        p2p: torch.Tensor,
        has_scale: int,
        has_decay: int,
    ) -> torch.Tensor:
        """Angle-fused forward (Kernel 4). Returns states ``h`` ``(B,T,n,k)``.

        Same traced-body fake-meta contract as the scan ops (``.contiguous`` +
        ``empty_like`` + one ``wrap_triton`` launch), so ``torch.compile`` sees
        the output shape with no graph break.
        """
        theta = theta.contiguous()
        scale = scale.contiguous()
        gamma = gamma.contiguous()
        b = b.contiguous()
        h0 = h0.contiguous()
        perm = perm.contiguous()
        sgn = sgn.contiguous()
        p2p = p2p.contiguous()
        B, T, n, R, half = theta.shape
        k = b.shape[-1]
        out = torch.empty_like(b)
        grid = (B * n,)
        wrap_triton(_angle_scan_fwd_kernel)[grid](
            theta,
            scale,
            gamma,
            b,
            h0,
            perm,
            sgn,
            p2p,
            out,
            T,
            n,
            k,
            R,
            half,
            int(has_scale),
            int(has_decay),
        )
        return out

    @triton_op("mingru_scans::angle_scan_bwd", mutates_args={})
    def angle_scan_bwd(
        theta: torch.Tensor,
        scale: torch.Tensor,
        gamma: torch.Tensor,
        b: torch.Tensor,
        h0: torch.Tensor,
        out: torch.Tensor,
        grad_out: torch.Tensor,
        perm: torch.Tensor,
        sgn: torch.Tensor,
        p2p: torch.Tensor,
        has_scale: int,
        has_decay: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Angle-fused exact stored-state reversible backward.

        Reads the forward inputs, the forward output ``out`` (the source of
        every ``h_{t-1}``), and the incoming state grads ``grad_out``. Returns
        ``(dtheta, dscale, dgamma, db, dh0)`` shaped like ``theta``, ``scale``,
        ``gamma``, ``b``, ``h0``. The module's own autograd then backprops
        ``dscale``/``dgamma`` through ``tanh(u)`` / ``exp(-lambda dt)`` to the
        real parameters, so this op handles only the recurrence.

        ``dscale``/``dgamma`` use ``zeros_like`` rather than ``empty_like``:
        the kernel only stores into them when ``HAS_SCALE``/``HAS_DECAY`` is
        set, so the disabled-feature output would otherwise be uninitialized
        memory. The disabled-feature grads are never consumed (the module
        never differentiates through the "ones" placeholder it passed in that
        case), but zero-init makes the returned tensor correct on its own
        terms rather than relying on that caller invariant.
        """
        theta = theta.contiguous()
        scale = scale.contiguous()
        gamma = gamma.contiguous()
        b = b.contiguous()
        h0 = h0.contiguous()
        out = out.contiguous()
        grad_out = grad_out.contiguous()
        perm = perm.contiguous()
        sgn = sgn.contiguous()
        p2p = p2p.contiguous()
        B, T, n, R, half = theta.shape
        k = b.shape[-1]
        dtheta = torch.empty_like(theta)
        dscale = torch.zeros_like(scale)
        dgamma = torch.zeros_like(gamma)
        db = torch.empty_like(b)
        dh0 = torch.empty_like(h0)
        grid = (B * n,)
        wrap_triton(_angle_scan_bwd_kernel)[grid](
            theta,
            scale,
            gamma,
            b,
            h0,
            out,
            grad_out,
            perm,
            sgn,
            p2p,
            dtheta,
            dscale,
            dgamma,
            db,
            dh0,
            T,
            n,
            k,
            R,
            half,
            int(has_scale),
            int(has_decay),
        )
        return dtheta, dscale, dgamma, db, dh0

    def _angle_setup_context(ctx, inputs, output):
        """``register_autograd`` setup hook: save what ``_angle_backward`` needs."""
        (theta, scale, gamma, b, h0, perm, sgn, p2p, has_scale, has_decay) = inputs
        # The forward output (all states) is what the backward reads every
        # h_{t-1} from; it is the module's return value, so saving it adds no
        # allocation.
        ctx.save_for_backward(theta, scale, gamma, b, h0, output, perm, sgn, p2p)
        ctx.has_scale = has_scale
        ctx.has_decay = has_decay

    def _angle_backward(ctx, grad_out):
        """``register_autograd`` backward for ``angle_scan_fwd``: dispatches to ``angle_scan_bwd``."""
        theta, scale, gamma, b, h0, out, perm, sgn, p2p = ctx.saved_tensors
        dtheta, dscale, dgamma, db, dh0 = angle_scan_bwd(
            theta,
            scale,
            gamma,
            b,
            h0,
            out,
            grad_out,
            perm,
            sgn,
            p2p,
            ctx.has_scale,
            ctx.has_decay,
        )
        # One grad per forward input, in order; None for the non-tensor args and
        # for perm/sgn/p2p (constant plane metadata, never differentiated).
        return dtheta, dscale, dgamma, db, dh0, None, None, None, None, None

    register_autograd(
        "mingru_scans::angle_scan_fwd",
        _angle_backward,
        setup_context=_angle_setup_context,
    )

    def angle_scan_impl(
        theta: torch.Tensor,
        scale: torch.Tensor,
        gamma: torch.Tensor,
        b: torch.Tensor,
        h0: torch.Tensor,
        perm: torch.Tensor,
        sgn: torch.Tensor,
        p2p: torch.Tensor,
        *,
        has_scale: int,
        has_decay: int,
    ) -> torch.Tensor:
        """Envelope-guarded entry point for the angle-fused kernel (module callers).

        Mirrors the ``SCAN_IMPLS``-style wrappers (e.g.
        ``_matrix_affine_scan_impl``): validates every shape-agreement
        invariant ``_angle_scan_fwd_kernel``'s pointer arithmetic assumes
        BEFORE any launch, then funnels any launch/compile failure into
        ``ScanFallback`` -- so ``min_gru._dispatch_angle_scan`` only ever needs
        to catch that one exception type, exactly the contract ``_dispatch_scan``
        gets from each ``SCAN_IMPLS`` entry. Not itself a ``SCAN_IMPLS`` entry
        (the angle-fused path is a module-forward fast path, not one of the
        four scan ops), so it is looked up by ``hasattr(triton_scans,
        "angle_scan_impl")`` instead.

        ``has_scale``/``has_decay`` are keyword-only here (this function's
        public-ish entry point, called from ``min_gru._dispatch_angle_scan``)
        -- self-documenting 0/1 flags at the call site instead of a bare
        trailing ``1, 0``. The underlying ``angle_scan_fwd`` ``triton_op``
        this function calls keeps its flat positional-int schema unchanged
        (a ``torch.library`` op signature, not a Python call-site
        convenience).

        Parameters mirror ``angle_scan_fwd``'s: ``theta`` ``(B, T, n, R,
        half)``; ``scale``/``gamma`` ``(B, T, n)``; ``b`` ``(B, T, n, k)``;
        ``h0`` ``(B, n, k)``; ``perm``/``sgn``/``p2p`` ``(R, k)``. ``k`` must
        be in the module's block-size envelope.
        """
        if theta.ndim != 5 or b.ndim != 4:
            raise ScanFallback(
                f"angle-fused kernel (theta={tuple(theta.shape)}, "
                f"b={tuple(b.shape)}) outside the envelope: need 5-D "
                "(B, T, n, R, half) theta and 4-D (B, T, n, k) b"
            )
        B, T, n, R, half = theta.shape
        k = b.shape[-1]
        if (
            k not in _ENVELOPE
            or b.shape[:3] != (B, T, n)
            or tuple(scale.shape) != (B, T, n)
            or tuple(gamma.shape) != (B, T, n)
            or tuple(h0.shape) != (B, n, k)
            or tuple(perm.shape) != (R, k)
            or tuple(sgn.shape) != (R, k)
            or tuple(p2p.shape) != (R, k)
        ):
            raise ScanFallback(
                f"angle-fused kernel (theta={tuple(theta.shape)}, "
                f"scale={tuple(scale.shape)}, gamma={tuple(gamma.shape)}, "
                f"b={tuple(b.shape)}, h0={tuple(h0.shape)}, "
                f"perm={tuple(perm.shape)}, sgn={tuple(sgn.shape)}, "
                f"p2p={tuple(p2p.shape)}) outside the envelope: need k in "
                f"{sorted(_ENVELOPE)} and scale/gamma/b/h0/perm/sgn/p2p shapes "
                "agreeing with theta's (B, T, n, R, half) on (B, T, n, R, k)"
            )
        try:
            return angle_scan_fwd(theta, scale, gamma, b, h0, perm, sgn, p2p, has_scale, has_decay)
        except Exception as exc:  # kernel launch/compile failure -> loud fallback
            raise ScanFallback(f"angle-fused Triton kernel failed: {exc}") from exc

    # --- DeltaMinGRU chunked-WY forward (two kernels) --------------------
    #
    # Two-stage WY decomposition of `DeltaMinGRU._forward_chunked` (the eager
    # oracle). Exploits that the unit-lower-triangular system `T = I + (K K^T
    # * beta) (.) strict-lower` depends only on `K`/`beta`, never on the
    # carried state `H`, so its solves parallelize across chunks:
    #
    #   Pre-pass (_delta_prepass_kernel): grid over (batch*head, chunk). Each
    #   program builds its chunk's `T` and forward-substitutes to the two
    #   H-independent solve products `T^-1 V` and `T^-1 K`. One launch covers
    #   the whole sequence.
    #
    #   State pass (_delta_state_kernel): grid over (batch*head). Each program
    #   loops over chunks carrying `H` (d_k x d_v fp32), forming `U = T^-1 V -
    #   (T^-1 K) H` and the state update `H += K^T (beta U)`; it writes ONLY the
    #   per-chunk boundary states (start-of-chunk `H`, for both the readout pass
    #   and the Task-2 recompute backward) and the final `H_T`. The readout is
    #   NOT computed here.
    #
    #   Readout pass (_delta_readout_kernel): grid over (batch*head, chunk). One
    #   program per chunk reads `Hbound[c]`, recomputes `U`, and writes the
    #   block-causal readout `y = Q H + (Q K^T (.) beta (.) mask) U`. Splitting
    #   the readout out of the serial `H`-chain restores parallelism
    #   proportional to num_chunks and shrinks per-program shared memory; the
    #   result is bit-identical to a single fused sequential pass because
    #   `Hbound[c]` is exactly the `H` the fused readout used at chunk `c`.
    #
    # Both are plain `@triton.jit` kernels launched directly (not
    # `triton_op`/`register_autograd`): the delta path is differentiated by a
    # hand-derived `torch.autograd.Function` (Task 2), and its target user is
    # eager-only ("compiling isn't always an option"), so the compile-tracing
    # machinery the four scan ops use is deliberately not in this path. The
    # raw launch (`_delta_forward_launch`) is factored apart from the envelope
    # validation so the autograd Function can call it directly (it also needs
    # the boundary states `delta_scan_impl` drops).
    #
    # Every contraction uses `input_precision="ieee"` and all three matmul
    # dims are >= 16 by construction (see `_delta_block_sizes`), so no per-dim
    # TF32/legality guard is needed -- unlike the generic affine kernels,
    # whose small `k`/`v` cases must branch between `tl.dot` and the
    # elementwise-sum idiom. fp32 throughout; the only division is the
    # implicit unit diagonal of the triangular substitution (the repo's one
    # permitted division), matching the eager `solve_triangular(...,
    # unitriangular=True)`.

    @triton.jit
    def _delta_prepass_kernel(
        K_ptr,
        V_ptr,
        beta_ptr,
        TinvV_ptr,
        TinvK_ptr,
        T,
        nh,
        chunk_size,
        d_k: tl.constexpr,
        d_v: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Pre-pass: per (batch*head, chunk), build ``T`` and solve ``T^-1 V``, ``T^-1 K``.

        One program owns a single chunk of a single ``(batch, head)`` lane.
        The chunk's ``M = nh * C`` micro-steps (``C`` tokens, token-major /
        micro-step-minor) are the contiguous block ``[start_tok*nh :
        start_tok*nh + M)`` of the flattened ``(T*nh, d)`` view of ``K``/``V``
        -- the same reshape convention ``_forward_chunked`` uses. Builds the
        unit-lower-triangular ``T[i, l] = 1{i==l} + (k_i . k_l) beta_l
        1{l<i}`` in registers, then forward-substitutes both right-hand sides
        (``V`` and ``K``) row by row. Padded rows (``m >= M``) load as zero
        and get an identity ``T`` row, so their solution is zero and never
        corrupts the real rows. Writes ``T^-1 V`` and ``T^-1 K`` back into the
        ``V``/``K``-shaped output tensors at the same chunk offset.
        """
        bh = tl.program_id(0)
        c = tl.program_id(1)

        start_tok = c * chunk_size
        rem = T - start_tok
        C = tl.where(rem < chunk_size, rem, chunk_size)  # ragged final chunk
        M = nh * C
        row0 = start_tok * nh  # first micro-step row in the (T*nh, d) view

        m = tl.arange(0, BLOCK_M)
        kk = tl.arange(0, BLOCK_K)
        vv = tl.arange(0, BLOCK_V)
        row_valid = m < M
        kcol = kk < d_k
        vcol = vv < d_v

        K_base = bh * (T * nh * d_k) + row0 * d_k
        V_base = bh * (T * nh * d_v) + row0 * d_v
        beta_base = bh * (T * nh) + row0

        Ktile = tl.load(
            K_ptr + K_base + m[:, None] * d_k + kk[None, :],
            mask=row_valid[:, None] & kcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_M, BLOCK_K)
        Vtile = tl.load(
            V_ptr + V_base + m[:, None] * d_v + vv[None, :],
            mask=row_valid[:, None] & vcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_M, BLOCK_V)
        beta_tile = tl.load(beta_ptr + beta_base + m, mask=row_valid, other=0.0).to(tl.float32)

        # T = I + (K K^T) (.) beta_l (.) strict-lower. K K^T is MMA-shaped
        # (all dims >= 16 after padding) -> exact ieee dot.
        KK = tl.dot(Ktile, tl.trans(Ktile), input_precision="ieee")  # (BLOCK_M, BLOCK_M)
        strict = (m[:, None] > m[None, :]).to(tl.float32)
        eye = (m[:, None] == m[None, :]).to(tl.float32)
        Tmat = eye + KK * beta_tile[None, :] * strict

        # Forward substitution (unit lower triangular): for real row i,
        # x_i = rhs_i - sum_{l<i} T[i, l] x_l. Summing T[i, :] @ X over ALL
        # rows is equivalent -- T[i, i]=1 hits the not-yet-written zero row i,
        # and T[i, l]=0 for l>i -- so no strict-lower re-masking is needed.
        XV = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
        XK = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for i in range(M):
            sel = m == i
            T_row = tl.sum(tl.where(sel[:, None], Tmat, 0.0), axis=0)  # (BLOCK_M,)
            rhs_v = tl.sum(tl.where(sel[:, None], Vtile, 0.0), axis=0)  # (BLOCK_V,)
            rhs_k = tl.sum(tl.where(sel[:, None], Ktile, 0.0), axis=0)  # (BLOCK_K,)
            xv = rhs_v - tl.sum(T_row[:, None] * XV, axis=0)
            xk = rhs_k - tl.sum(T_row[:, None] * XK, axis=0)
            XV = tl.where(sel[:, None], xv[None, :], XV)
            XK = tl.where(sel[:, None], xk[None, :], XK)

        tl.store(
            TinvV_ptr + V_base + m[:, None] * d_v + vv[None, :],
            XV,
            mask=row_valid[:, None] & vcol[None, :],
        )
        tl.store(
            TinvK_ptr + K_base + m[:, None] * d_k + kk[None, :],
            XK,
            mask=row_valid[:, None] & kcol[None, :],
        )

    @triton.jit
    def _delta_state_kernel(
        K_ptr,
        beta_ptr,
        TinvV_ptr,
        TinvK_ptr,
        H0_ptr,
        Hbound_ptr,
        HT_ptr,
        T,
        nh,
        num_chunks,
        chunk_size,
        d_k: tl.constexpr,
        d_v: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Serial state pass (Kernel 6a): per (batch*head), carry ``H`` across chunks.

        One program owns a ``(batch, head)`` lane and walks chunks
        ``0..num_chunks-1``, carrying the ``d_k x d_v`` state ``H`` (seeded
        from ``H0``). Per chunk, using the pre-pass products ``T^-1 V`` /
        ``T^-1 K``: it stores the start-of-chunk ``H`` as the chunk boundary
        state ``Hbound[c]`` (chunk 0's is ``H0``) BEFORE updating, forms
        ``U = T^-1 V - (T^-1 K) H`` and applies the state update
        ``H += K^T (beta U)``; after the loop it writes the final ``H_T``.

        This kernel carries ONLY the state recurrence -- it computes no readout
        (no ``Q``, no ``Q K^T`` term, no ``y``). That work moves to the
        chunk-parallel ``_delta_readout_kernel``, which consumes the
        ``Hbound[c]`` this pass produces. Splitting the readout out of the
        serial ``H``-chain (a) restores parallelism proportional to
        ``num_chunks`` for the expensive readout, and (b) shrinks this pass's
        per-program tiles to only ``M x d_k`` / ``M x d_v`` / ``d_k x d_v``
        (no ``C x M`` readout tile), so the serial grid (one program per
        ``batch*head``) is light. ``num_stages=1`` on the launch: the ``H``
        carry is a strict serial dependency the pipeliner cannot overlap.
        Padded rows/cols load as zero and never contribute.
        """
        bh = tl.program_id(0)

        m = tl.arange(0, BLOCK_M)
        kk = tl.arange(0, BLOCK_K)
        vv = tl.arange(0, BLOCK_V)
        kcol = kk < d_k
        vcol = vv < d_v

        H0_base = bh * (d_k * d_v)
        H = tl.load(
            H0_ptr + H0_base + kk[:, None] * d_v + vv[None, :],
            mask=kcol[:, None] & vcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_K, BLOCK_V)

        for c in range(num_chunks):
            start_tok = c * chunk_size
            rem = T - start_tok
            C = tl.where(rem < chunk_size, rem, chunk_size)
            M = nh * C
            row0 = start_tok * nh

            row_valid = m < M

            K_base = bh * (T * nh * d_k) + row0 * d_k
            V_base = bh * (T * nh * d_v) + row0 * d_v
            beta_base = bh * (T * nh) + row0
            Hb_base = bh * (num_chunks * d_k * d_v) + c * (d_k * d_v)

            # Boundary state = start-of-chunk H (chunk 0's is H0), stored
            # before the update for the reverse-chunk backward AND for the
            # chunk-parallel readout pass.
            tl.store(
                Hbound_ptr + Hb_base + kk[:, None] * d_v + vv[None, :],
                H,
                mask=kcol[:, None] & vcol[None, :],
            )

            Ktile = tl.load(
                K_ptr + K_base + m[:, None] * d_k + kk[None, :],
                mask=row_valid[:, None] & kcol[None, :],
                other=0.0,
            ).to(tl.float32)  # (BLOCK_M, BLOCK_K)
            TinvK = tl.load(
                TinvK_ptr + K_base + m[:, None] * d_k + kk[None, :],
                mask=row_valid[:, None] & kcol[None, :],
                other=0.0,
            ).to(tl.float32)  # (BLOCK_M, BLOCK_K)
            TinvV = tl.load(
                TinvV_ptr + V_base + m[:, None] * d_v + vv[None, :],
                mask=row_valid[:, None] & vcol[None, :],
                other=0.0,
            ).to(tl.float32)  # (BLOCK_M, BLOCK_V)
            beta_tile = tl.load(beta_ptr + beta_base + m, mask=row_valid, other=0.0).to(tl.float32)

            # U = T^-1 V - (T^-1 K) H, then H += K^T (beta U).
            U = TinvV - tl.dot(TinvK, H, input_precision="ieee")  # (BLOCK_M, BLOCK_V)
            bU = beta_tile[:, None] * U  # (BLOCK_M, BLOCK_V)
            H = H + tl.dot(tl.trans(Ktile), bU, input_precision="ieee")  # (BLOCK_K, BLOCK_V)

        tl.store(
            HT_ptr + H0_base + kk[:, None] * d_v + vv[None, :],
            H,
            mask=kcol[:, None] & vcol[None, :],
        )

    @triton.jit
    def _delta_readout_kernel(
        Q_ptr,
        K_ptr,
        beta_ptr,
        TinvV_ptr,
        TinvK_ptr,
        Hbound_ptr,
        y_ptr,
        T,
        nh,
        num_chunks,
        chunk_size,
        d_k: tl.constexpr,
        d_v: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Chunk-parallel readout (Kernel 6b): per (batch*head, chunk), write ``y``.

        Grid is ``(batch*head, num_chunks)`` -- one independent program per
        chunk, so the whole readout runs in parallel across chunks instead of
        being serialized inside the state pass's ``H``-chain. Each program
        reads its chunk's start-of-chunk state ``H = Hbound[c]`` (produced by
        ``_delta_state_kernel``) and recomputes ``U = T^-1 V - (T^-1 K) H``
        from the pre-pass products (cheaper in memory than persisting a whole-
        sequence ``U`` buffer -- one ``(M, d_k) x (d_k, d_v)`` dot per chunk vs.
        a ``V``-sized workspace), then forms the block-causal masked readout
        ``y = Q H + (Q K^T (.) beta (.) read_mask) U`` (``read_mask[t, m] =
        1{token(m) <= t}``, token-granularity, inclusive of a token's own
        ``nh`` micro-steps).

        Because ``H`` is read (not carried), there is no cross-chunk
        dependency and no in-kernel chunk loop, so the launch multi-buffers
        nothing (``num_stages`` is irrelevant -- single-buffer footprint) and
        latency is hidden by occupancy across the ``batch*head * num_chunks``
        grid rather than by software pipelining. The per-program tile set is a
        strict subset of the pre-split fused kernel's single-chunk tiles, so
        its shared-memory footprint fits wherever that single-chunk case fit.
        The ``y`` this writes is bit-identical to the pre-split fused readout:
        ``Hbound[c]`` is exactly the ``H`` the fused kernel held at chunk
        ``c``'s readout. Padded rows/cols load as zero and never contribute.
        """
        bh = tl.program_id(0)
        c = tl.program_id(1)

        m = tl.arange(0, BLOCK_M)
        cc = tl.arange(0, BLOCK_C)
        kk = tl.arange(0, BLOCK_K)
        vv = tl.arange(0, BLOCK_V)
        kcol = kk < d_k
        vcol = vv < d_v

        start_tok = c * chunk_size
        rem = T - start_tok
        C = tl.where(rem < chunk_size, rem, chunk_size)
        M = nh * C
        row0 = start_tok * nh

        row_valid = m < M
        tok_valid = cc < C

        K_base = bh * (T * nh * d_k) + row0 * d_k
        V_base = bh * (T * nh * d_v) + row0 * d_v
        beta_base = bh * (T * nh) + row0
        Q_base = bh * (T * d_k) + start_tok * d_k
        y_base = bh * (T * d_v) + start_tok * d_v
        Hb_base = bh * (num_chunks * d_k * d_v) + c * (d_k * d_v)

        H = tl.load(
            Hbound_ptr + Hb_base + kk[:, None] * d_v + vv[None, :],
            mask=kcol[:, None] & vcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_K, BLOCK_V) -- start-of-chunk state

        Ktile = tl.load(
            K_ptr + K_base + m[:, None] * d_k + kk[None, :],
            mask=row_valid[:, None] & kcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_M, BLOCK_K)
        TinvK = tl.load(
            TinvK_ptr + K_base + m[:, None] * d_k + kk[None, :],
            mask=row_valid[:, None] & kcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_M, BLOCK_K)
        TinvV = tl.load(
            TinvV_ptr + V_base + m[:, None] * d_v + vv[None, :],
            mask=row_valid[:, None] & vcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_M, BLOCK_V)
        beta_tile = tl.load(beta_ptr + beta_base + m, mask=row_valid, other=0.0).to(tl.float32)
        Qtile = tl.load(
            Q_ptr + Q_base + cc[:, None] * d_k + kk[None, :],
            mask=tok_valid[:, None] & kcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_C, BLOCK_K)

        # Recompute U = T^-1 V - (T^-1 K) H (same math the state pass used).
        U = TinvV - tl.dot(TinvK, H, input_precision="ieee")  # (BLOCK_M, BLOCK_V)

        # Block-causal masked readout. read_mask[t, m] = 1 iff the token owning
        # micro-step m (m // nh) is <= t; padded rows/cols zeroed.
        R = tl.dot(Qtile, tl.trans(Ktile), input_precision="ieee")  # (BLOCK_C, BLOCK_M)
        tok_of_m = m // nh
        read_mask = (
            (tok_of_m[None, :] <= cc[:, None]) & row_valid[None, :] & tok_valid[:, None]
        ).to(tl.float32)  # (BLOCK_C, BLOCK_M)
        masked = R * beta_tile[None, :] * read_mask
        y = tl.dot(Qtile, H, input_precision="ieee") + tl.dot(
            masked, U, input_precision="ieee"
        )  # (BLOCK_C, BLOCK_V)
        tl.store(
            y_ptr + y_base + cc[:, None] * d_v + vv[None, :],
            y,
            mask=tok_valid[:, None] & vcol[None, :],
        )

    def _delta_forward_launch(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        beta: torch.Tensor,
        H0: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raw three-kernel forward launch (no envelope validation).

        Assumes the inputs are already in-envelope (the caller --
        ``delta_scan_impl`` or the Task-2 ``autograd.Function`` -- validates
        first via ``_delta_validate_envelope``). Factored apart from that
        validation so the autograd Function can reuse the exact launch without
        rewrite; it returns the boundary states the recompute backward needs
        in addition to the public ``(y, H_T)`` pair.

        Parameters mirror spec section 6: ``Q`` ``(B, n_heads, T, d_k)``;
        ``K`` ``(B, n_heads, T, nh, d_k)``; ``V`` ``(B, n_heads, T, nh,
        d_v)``; ``beta`` ``(B, n_heads, T, nh)``; ``H0`` ``(B, n_heads, d_k,
        d_v)``.

        Returns
        -------
        tuple of torch.Tensor
            ``y`` ``(B, n_heads, T, d_v)``; ``H_T`` ``(B, n_heads, d_k,
            d_v)``; ``Hbound`` ``(B, n_heads, num_chunks, d_k, d_v)`` -- the
            start-of-chunk states (chunk 0's is ``H0``).
        """
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        beta = beta.contiguous()
        H0 = H0.contiguous()

        B, n_heads, T, d_k = Q.shape
        nh = K.shape[3]
        d_v = V.shape[4]
        bh = B * n_heads
        num_chunks = triton.cdiv(T, chunk_size)
        block_m, block_c, block_k, block_v = _delta_block_sizes(nh, chunk_size, d_k, d_v)

        # Pre-pass: T^-1 V / T^-1 K, laid out exactly like V / K so the
        # state and readout passes read each chunk at the same offset.
        TinvV = torch.empty_like(V)
        TinvK = torch.empty_like(K)
        _delta_prepass_kernel[(bh, num_chunks)](
            K,
            V,
            beta,
            TinvV,
            TinvK,
            T,
            nh,
            chunk_size,
            d_k,
            d_v,
            block_m,
            block_k,
            block_v,
        )

        y = torch.empty(B, n_heads, T, d_v, device=Q.device, dtype=torch.float32)
        Hbound = torch.empty(B, n_heads, num_chunks, d_k, d_v, device=Q.device, dtype=torch.float32)
        H_T = torch.empty(B, n_heads, d_k, d_v, device=Q.device, dtype=torch.float32)

        # Serial state pass (Kernel 6a): grid over (batch*head), carries H
        # across chunks and writes Hbound / H_T. num_stages=1 (single-buffer,
        # pipelining OFF): the in-kernel `for c in range(num_chunks)` loop
        # carries H (iteration c reads the H iteration c-1 wrote), a strict
        # serial dependency the software pipeliner cannot overlap, so the
        # multi-buffered prefetch a higher num_stages would allocate buys
        # nothing here (the dependent H-chain, not load latency, bounds the
        # loop) while enlarging shared memory. Its per-program tiles are now
        # only M x d_k / M x d_v / d_k x d_v (the readout's C x M tile moved to
        # the parallel pass), a strict subset of the pre-split fused kernel's
        # single-chunk footprint, so num_stages=1 fits with room to spare.
        _delta_state_kernel[(bh,)](
            K,
            beta,
            TinvV,
            TinvK,
            H0,
            Hbound,
            H_T,
            T,
            nh,
            num_chunks,
            chunk_size,
            d_k,
            d_v,
            block_m,
            block_k,
            block_v,
            num_stages=1,
        )

        # Chunk-parallel readout (Kernel 6b): grid over (batch*head, chunk),
        # so the expensive readout runs one program per chunk instead of being
        # serialized inside the state pass. num_warps is widened for the wide
        # (d_k >= 32) tiles -- their C x M / M x d_v dots have enough
        # per-program work to use 8 warps, where the narrow tiles are saturated
        # by the default 4. The kernel has no in-kernel chunk loop, so it
        # multi-buffers nothing (num_stages irrelevant, single-buffer footprint
        # = a subset of the fused single-chunk footprint that already fit);
        # latency is hidden by occupancy across the batch*head * num_chunks grid.
        readout_warps = 8 if d_k >= 32 else 4
        _delta_readout_kernel[(bh, num_chunks)](
            Q,
            K,
            beta,
            TinvV,
            TinvK,
            Hbound,
            y,
            T,
            nh,
            num_chunks,
            chunk_size,
            d_k,
            d_v,
            block_m,
            block_c,
            block_k,
            block_v,
            num_warps=readout_warps,
        )
        return y, H_T, Hbound

    # --- DeltaMinGRU fused backward trio (Kernel 7a/7b/7c) ------------------
    #
    # The reverse-chunk gradient of the chunked-WY forward, decomposed to
    # MIRROR the forward trio: the forward's ONLY serial dependency is the `H`
    # chain (everything H-independent -- T-build, T^-1 V, T^-1 K -- parallelizes
    # across chunks in the pre-pass); the backward's ONLY serial dependency is
    # the `dH` chain carried last-to-first (everything dH-independent
    # parallelizes across chunks).
    #
    # Serial-vs-parallel analysis. Per chunk the exact torch backward forms
    # `G = T^-T @ dU` with `dU = dU_read + beta (.) (K dH)`, where
    # `dU_read = P^T dy` is dH-independent and `beta (.) (K dH) = BK @ dH`
    # (`BK = beta (.) K`). `T^-T` is `H`/`dH`-independent (built from `K`/`beta`
    # only), so `G` splits LINEARLY in `dH`:
    #     G = T^-T dU_read + T^-T (BK @ dH) = Gr + (T^-T BK) @ dH,
    # exactly the forward's `U = T^-1 V - (T^-1 K) H` trick transposed. The dH
    # recurrence `dHc = Q^T dy + dH - K^T G` therefore becomes an AFFINE map
    #     dHc = (Q^T dy - K^T Gr) + dH - K^T ((T^-T BK) @ dH)
    #         = dHconst + dH - K^T (TinvTBK @ dH),
    # so the transpose SOLVE is hoisted OUT of the serial pass (into the
    # chunk-parallel B1), leaving the serial pass (B2) only small GEMMs on the
    # `dH` chain -- the difference between a fast and a slow backward. The final
    # per-chunk grads need the full `G`/`U` (which depend on the resolved
    # `dH_c`), so they run in a third chunk-parallel pass (B3) AFTER the serial
    # `dH` chain is known, seeded from the per-chunk `dHbound[c]` B2 writes --
    # the transposed analogue of the forward readout reading `Hbound[c]`.
    #
    # Memory. The torch backward materialized ~4 concurrent `(B, n_heads, M, M)`
    # global tensors (`KK`, `T_mat`, `dT`, `dKK`), the peak that overran eager at
    # `pd1024_T64`. Here every `(M, M)` tile is per-program register/SMEM, NEVER
    # a global batched tensor; the only extra global workspace is `TinvTBK`
    # (K-shaped, B1->B2) plus the two cheap boundary tensors `dHconst`/`dHbound`
    # (`num_chunks * d_k * d_v` per head) and `dH0` -- strictly leaner than the
    # torch peak, and B3 RECOMPUTES `T`/`U`/`G` per chunk rather than persisting
    # any sequence-sized `U`/`Gr` (the same recompute-over-persist call the
    # forward readout makes for `U`).
    #
    # Every contraction is `input_precision="ieee"` with all three matmul dims
    # >= 16 by construction (`_delta_block_sizes`); the only division is the
    # implicit unit diagonal of the triangular substitutions -- forward
    # substitution for `U` (mirrors the pre-pass), BACKWARD substitution for the
    # transpose solves `T^-T (.)` (down-counting micro-step index, unit diagonal,
    # no division), transcribed from the verified torch backward, not re-derived.

    @triton.jit
    def _delta_bwd_prepass_kernel(
        Q_ptr,
        K_ptr,
        beta_ptr,
        dy_ptr,
        TinvTBK_ptr,
        dHconst_ptr,
        T,
        nh,
        num_chunks,
        chunk_size,
        d_k: tl.constexpr,
        d_v: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Backward pre-pass (Kernel 7a): per (batch*head, chunk), hoist the transpose solve.

        Rebuilds the chunk's unit-lower-triangular ``T`` (register, exactly the
        forward pre-pass build) and produces the two dH-independent products the
        serial ``dH`` pass consumes:

        * ``TinvTBK = T^-T (beta (.) K)`` ``(M, d_k)`` -- the linear operator
          the serial recurrence applies to ``dH`` (written K-shaped so B2 reads
          it at the same chunk offset as ``K``).
        * ``dHconst = Q^T dy - K^T Gr`` ``(d_k, d_v)`` where ``Gr = T^-T (P^T dy)``
          is the readout half of ``G`` -- the constant term of the affine ``dH``
          recurrence (written boundary-shaped, ``num_chunks * d_k * d_v``).

        Both ``T^-T`` actions are BACKWARD substitutions (unit upper ``T^T``,
        down-counting micro-step ``i = M-1..0``): ``x_i = rhs_i - sum_l T[l,i] x_l``,
        unit diagonal so no division. Padded rows/cols load as zero and get an
        identity ``T`` row/col, so they never contribute. No ``U`` and no ``H``
        are needed here (the readout ``dU_read`` and ``BK`` are both
        ``H``-independent), so this pass is a pure function of ``K``/``beta``/
        ``Q``/``dy``.
        """
        bh = tl.program_id(0)
        c = tl.program_id(1)

        start_tok = c * chunk_size
        rem = T - start_tok
        C = tl.where(rem < chunk_size, rem, chunk_size)
        M = nh * C
        row0 = start_tok * nh

        m = tl.arange(0, BLOCK_M)
        cc = tl.arange(0, BLOCK_C)
        kk = tl.arange(0, BLOCK_K)
        vv = tl.arange(0, BLOCK_V)
        row_valid = m < M
        tok_valid = cc < C
        kcol = kk < d_k
        vcol = vv < d_v

        K_base = bh * (T * nh * d_k) + row0 * d_k
        beta_base = bh * (T * nh) + row0
        Q_base = bh * (T * d_k) + start_tok * d_k
        dy_base = bh * (T * d_v) + start_tok * d_v
        Hb_base = bh * (num_chunks * d_k * d_v) + c * (d_k * d_v)

        Ktile = tl.load(
            K_ptr + K_base + m[:, None] * d_k + kk[None, :],
            mask=row_valid[:, None] & kcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_M, BLOCK_K)
        beta_tile = tl.load(beta_ptr + beta_base + m, mask=row_valid, other=0.0).to(tl.float32)
        Qtile = tl.load(
            Q_ptr + Q_base + cc[:, None] * d_k + kk[None, :],
            mask=tok_valid[:, None] & kcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_C, BLOCK_K)
        dyc = tl.load(
            dy_ptr + dy_base + cc[:, None] * d_v + vv[None, :],
            mask=tok_valid[:, None] & vcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_C, BLOCK_V)

        # T = I + (K K^T) (.) beta_l (.) strict-lower (same build as forward).
        KK = tl.dot(Ktile, tl.trans(Ktile), input_precision="ieee")  # (BLOCK_M, BLOCK_M)
        strict = (m[:, None] > m[None, :]).to(tl.float32)
        eye = (m[:, None] == m[None, :]).to(tl.float32)
        Tmat = eye + KK * beta_tile[None, :] * strict

        # Readout half of dU: P = (Q K^T) (.) beta (.) read_mask; dU_read = P^T dy.
        S = tl.dot(Qtile, tl.trans(Ktile), input_precision="ieee")  # (BLOCK_C, BLOCK_M)
        tok_of_m = m // nh
        read_mask = (
            (tok_of_m[None, :] <= cc[:, None]) & row_valid[None, :] & tok_valid[:, None]
        ).to(tl.float32)  # (BLOCK_C, BLOCK_M)
        P = S * beta_tile[None, :] * read_mask
        dU_read = tl.dot(tl.trans(P), dyc, input_precision="ieee")  # (BLOCK_M, BLOCK_V)

        # BK = beta (.) K (row-scaled); pad rows already zero (beta/K masked).
        BK = beta_tile[:, None] * Ktile  # (BLOCK_M, BLOCK_K)

        # Backward substitution (unit upper T^T), down-counting i = M-1..0:
        #   XA = T^-T BK, XG = T^-T dU_read. For real row i,
        #   x_i = rhs_i - sum_l T[l, i] x_l (l>i already written; l<i -> T[l,i]=0;
        #   l=i -> x_i not yet written = 0), so summing over ALL rows is exact.
        XA = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        XG = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
        for step in range(M):
            i = M - 1 - step
            sel = m == i
            T_col = tl.sum(tl.where(sel[None, :], Tmat, 0.0), axis=1)  # Tmat[:, i] (BLOCK_M,)
            rhs_a = tl.sum(tl.where(sel[:, None], BK, 0.0), axis=0)  # (BLOCK_K,)
            rhs_g = tl.sum(tl.where(sel[:, None], dU_read, 0.0), axis=0)  # (BLOCK_V,)
            xa = rhs_a - tl.sum(T_col[:, None] * XA, axis=0)
            xg = rhs_g - tl.sum(T_col[:, None] * XG, axis=0)
            XA = tl.where(sel[:, None], xa[None, :], XA)
            XG = tl.where(sel[:, None], xg[None, :], XG)

        tl.store(
            TinvTBK_ptr + K_base + m[:, None] * d_k + kk[None, :],
            XA,
            mask=row_valid[:, None] & kcol[None, :],
        )

        # dHconst = Q^T dy - K^T Gr, Gr = XG.
        QtY = tl.dot(tl.trans(Qtile), dyc, input_precision="ieee")  # (BLOCK_K, BLOCK_V)
        KtGr = tl.dot(tl.trans(Ktile), XG, input_precision="ieee")  # (BLOCK_K, BLOCK_V)
        tl.store(
            dHconst_ptr + Hb_base + kk[:, None] * d_v + vv[None, :],
            QtY - KtGr,
            mask=kcol[:, None] & vcol[None, :],
        )

    @triton.jit
    def _delta_bwd_state_kernel(
        K_ptr,
        TinvTBK_ptr,
        dHconst_ptr,
        dHT_ptr,
        dHbound_ptr,
        dH0_ptr,
        T,
        nh,
        num_chunks,
        chunk_size,
        d_k: tl.constexpr,
        d_v: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Serial reverse ``dH`` pass (Kernel 7b): per (batch*head), carry ``dH`` last-to-first.

        One program owns a ``(batch, head)`` lane and walks chunks
        ``num_chunks-1 .. 0``, carrying ``dH`` (grad w.r.t. the chunk's OUTPUT
        state, seeded from ``dH_T``). Per chunk it stores the incoming ``dH`` as
        ``dHbound[c]`` (what the B3 grad pass needs as this chunk's output-state
        grad) BEFORE updating, then applies the affine recurrence
        ``dHc = dHconst_c + dH - K^T (TinvTBK_c @ dH)`` using the two B1 products
        -- only small GEMMs, no solve (the transpose solve was hoisted to B1).
        After the loop ``dH`` is the grad w.r.t. ``H0`` and is written to
        ``dH0``. ``num_stages=1`` on the launch: the ``dH`` carry is a strict
        serial dependency the pipeliner cannot overlap (exactly the forward
        state pass). Padded rows load as zero and never contribute.
        """
        bh = tl.program_id(0)

        m = tl.arange(0, BLOCK_M)
        kk = tl.arange(0, BLOCK_K)
        vv = tl.arange(0, BLOCK_V)
        kcol = kk < d_k
        vcol = vv < d_v

        H0_base = bh * (d_k * d_v)
        dH = tl.load(
            dHT_ptr + H0_base + kk[:, None] * d_v + vv[None, :],
            mask=kcol[:, None] & vcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_K, BLOCK_V) -- grad w.r.t. final state

        for step in range(num_chunks):
            c = num_chunks - 1 - step
            start_tok = c * chunk_size
            rem = T - start_tok
            C = tl.where(rem < chunk_size, rem, chunk_size)
            M = nh * C
            row0 = start_tok * nh
            row_valid = m < M

            K_base = bh * (T * nh * d_k) + row0 * d_k
            Hb_base = bh * (num_chunks * d_k * d_v) + c * (d_k * d_v)

            # dHbound[c] = incoming dH (grad w.r.t. this chunk's output state).
            tl.store(
                dHbound_ptr + Hb_base + kk[:, None] * d_v + vv[None, :],
                dH,
                mask=kcol[:, None] & vcol[None, :],
            )

            TinvTBK = tl.load(
                TinvTBK_ptr + K_base + m[:, None] * d_k + kk[None, :],
                mask=row_valid[:, None] & kcol[None, :],
                other=0.0,
            ).to(tl.float32)  # (BLOCK_M, BLOCK_K)
            Ktile = tl.load(
                K_ptr + K_base + m[:, None] * d_k + kk[None, :],
                mask=row_valid[:, None] & kcol[None, :],
                other=0.0,
            ).to(tl.float32)  # (BLOCK_M, BLOCK_K)
            dHconst = tl.load(
                dHconst_ptr + Hb_base + kk[:, None] * d_v + vv[None, :],
                mask=kcol[:, None] & vcol[None, :],
                other=0.0,
            ).to(tl.float32)  # (BLOCK_K, BLOCK_V)

            tmp = tl.dot(TinvTBK, dH, input_precision="ieee")  # (BLOCK_M, BLOCK_V)
            dH = dHconst + dH - tl.dot(tl.trans(Ktile), tmp, input_precision="ieee")

        tl.store(
            dH0_ptr + H0_base + kk[:, None] * d_v + vv[None, :],
            dH,
            mask=kcol[:, None] & vcol[None, :],
        )

    @triton.jit
    def _delta_bwd_grad_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        beta_ptr,
        Hbound_ptr,
        dHbound_ptr,
        dy_ptr,
        dQ_ptr,
        dK_ptr,
        dK2_ptr,
        dV_ptr,
        dbeta_ptr,
        T,
        nh,
        num_chunks,
        chunk_size,
        d_k: tl.constexpr,
        d_v: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """Chunk-parallel grad pass (Kernel 7c): per (batch*head, chunk), write dQ/dK/dV/dbeta.

        Runs AFTER the serial ``dH`` chain (B2) has resolved every chunk's
        output-state grad. One independent program per chunk reads its
        start-of-chunk state ``Hc = Hbound[c]`` (forward-saved) and its
        output-state grad ``dH_c = dHbound[c]`` (from B2), recomputes ``T`` and
        the intra-chunk intermediates the torch backward built (``U`` via
        forward substitution; the full ``G = T^-T dU`` via backward substitution
        with ``dU = P^T dy + beta (.) (K dH_c)``), then transcribes the exact
        torch grad accumulations:

            dQ = dy Hc^T + dS K
            dK = (beta (.) U) dH_c^T + dS^T Q - G Hc^T + dKK K + dKK^T K
            dV = G
            dbeta = sum_v(U (.) K dH_c) + sum_t(dP (.) S (.) mask) + sum_i(dT (.) KK (.) strict)

        with ``dP = dy U^T``, ``dS = dP (.) beta (.) mask``, ``dT = -(G U^T)``,
        ``dKK = dT (.) beta (.) strict`` -- the reference torch loop's math,
        just tiled. Recomputing ``T``/``U``/``G`` per chunk (rather than
        persisting sequence-sized ``U``/``Gr`` from B1) keeps the backward
        workspace lean, mirroring the forward readout's recompute-of-``U`` call.
        Every ``(M, M)`` tile is per-program (never global). Padded rows/cols
        load as zero and are dropped on store.

        Shared-memory discipline (the L4 launch-fit invariant). Every packed
        multi-``tl.dot`` accumulation (``dQ``, ``dK``) is written as SEPARATE
        sequential ``+=`` statements: a single summed expression lets Triton
        co-stage several dots' operand tiles at once (the over-concurrency that
        put the earlier one-expression ``dK`` at ~110 KB, over the 101376 B L4
        limit), whereas sequential statements reuse the scratch so the peak is
        one dot's operand pair. The two ``(M, M)``-operand terms ``dKK K`` and
        ``dKK^T K`` (dominant: the ``(M, M)`` ``dKK`` tile is 65536 B at
        ``M = 128``) are additionally BLOCKED over the ``d_k`` free dimension
        into 16-wide sub-tiles loaded from global (fla's feature-blocking), so
        ``dKK`` pairs only with an ``(M, 16)`` slice of ``K`` (<= 73728 B at
        ``M = 128``, every ``d_k``) and are written to a separate ``dK2`` buffer
        the launch folds into ``dK``. Output-column blocking is exact (matmul
        columns are independent), so each block equals the matching columns of
        the full ``(M, M)``-operand product. (tiling scheme after fla's
        chunk_delta_rule wy_fast backward, MIT.)
        """
        bh = tl.program_id(0)
        c = tl.program_id(1)

        start_tok = c * chunk_size
        rem = T - start_tok
        C = tl.where(rem < chunk_size, rem, chunk_size)
        M = nh * C
        row0 = start_tok * nh

        m = tl.arange(0, BLOCK_M)
        cc = tl.arange(0, BLOCK_C)
        kk = tl.arange(0, BLOCK_K)
        vv = tl.arange(0, BLOCK_V)
        row_valid = m < M
        tok_valid = cc < C
        kcol = kk < d_k
        vcol = vv < d_v

        K_base = bh * (T * nh * d_k) + row0 * d_k
        V_base = bh * (T * nh * d_v) + row0 * d_v
        beta_base = bh * (T * nh) + row0
        Q_base = bh * (T * d_k) + start_tok * d_k
        dy_base = bh * (T * d_v) + start_tok * d_v
        Hb_base = bh * (num_chunks * d_k * d_v) + c * (d_k * d_v)

        Ktile = tl.load(
            K_ptr + K_base + m[:, None] * d_k + kk[None, :],
            mask=row_valid[:, None] & kcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_M, BLOCK_K)
        Vtile = tl.load(
            V_ptr + V_base + m[:, None] * d_v + vv[None, :],
            mask=row_valid[:, None] & vcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_M, BLOCK_V)
        beta_tile = tl.load(beta_ptr + beta_base + m, mask=row_valid, other=0.0).to(tl.float32)
        Qtile = tl.load(
            Q_ptr + Q_base + cc[:, None] * d_k + kk[None, :],
            mask=tok_valid[:, None] & kcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_C, BLOCK_K)
        dyc = tl.load(
            dy_ptr + dy_base + cc[:, None] * d_v + vv[None, :],
            mask=tok_valid[:, None] & vcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_C, BLOCK_V)
        Hc = tl.load(
            Hbound_ptr + Hb_base + kk[:, None] * d_v + vv[None, :],
            mask=kcol[:, None] & vcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_K, BLOCK_V) -- start-of-chunk state
        dH_c = tl.load(
            dHbound_ptr + Hb_base + kk[:, None] * d_v + vv[None, :],
            mask=kcol[:, None] & vcol[None, :],
            other=0.0,
        ).to(tl.float32)  # (BLOCK_K, BLOCK_V) -- output-state grad from B2

        # Rebuild T (register).
        KK = tl.dot(Ktile, tl.trans(Ktile), input_precision="ieee")  # (BLOCK_M, BLOCK_M)
        strict = (m[:, None] > m[None, :]).to(tl.float32)
        eye = (m[:, None] == m[None, :]).to(tl.float32)
        Tmat = eye + KK * beta_tile[None, :] * strict

        # U = T^-1 (V - K Hc) by forward substitution (mirrors the pre-pass).
        rhsU = Vtile - tl.dot(Ktile, Hc, input_precision="ieee")  # (BLOCK_M, BLOCK_V)
        U = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
        for i in range(M):
            sel = m == i
            T_row = tl.sum(tl.where(sel[:, None], Tmat, 0.0), axis=0)  # Tmat[i, :] (BLOCK_M,)
            rhs_i = tl.sum(tl.where(sel[:, None], rhsU, 0.0), axis=0)  # (BLOCK_V,)
            xu = rhs_i - tl.sum(T_row[:, None] * U, axis=0)
            U = tl.where(sel[:, None], xu[None, :], U)

        # Readout mask / S / P (same build as the forward readout).
        S = tl.dot(Qtile, tl.trans(Ktile), input_precision="ieee")  # (BLOCK_C, BLOCK_M)
        tok_of_m = m // nh
        read_mask = (
            (tok_of_m[None, :] <= cc[:, None]) & row_valid[None, :] & tok_valid[:, None]
        ).to(tl.float32)  # (BLOCK_C, BLOCK_M)
        P = S * beta_tile[None, :] * read_mask

        # dU = P^T dy + beta (.) (K dH_c); dbU = K dH_c reused for dbeta.
        dbU = tl.dot(Ktile, dH_c, input_precision="ieee")  # (BLOCK_M, BLOCK_V)
        dU = tl.dot(tl.trans(P), dyc, input_precision="ieee") + beta_tile[:, None] * dbU

        # G = T^-T dU by backward substitution (down-counting i = M-1..0).
        G = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
        for step in range(M):
            i = M - 1 - step
            sel = m == i
            T_col = tl.sum(tl.where(sel[None, :], Tmat, 0.0), axis=1)  # Tmat[:, i] (BLOCK_M,)
            rhs_g = tl.sum(tl.where(sel[:, None], dU, 0.0), axis=0)  # (BLOCK_V,)
            xg = rhs_g - tl.sum(T_col[:, None] * G, axis=0)
            G = tl.where(sel[:, None], xg[None, :], G)

        # --- transcribe torch grad accumulations -----------------------------
        dP = tl.dot(dyc, tl.trans(U), input_precision="ieee")  # (BLOCK_C, BLOCK_M)
        dS = dP * beta_tile[None, :] * read_mask  # (BLOCK_C, BLOCK_M)
        dT = -tl.dot(G, tl.trans(U), input_precision="ieee")  # (BLOCK_M, BLOCK_M)

        # dbeta: state + readout(P) + T-build terms (register reductions, no dot).
        dbeta_local = (
            tl.sum(U * dbU, axis=1)
            + tl.sum(dP * S * read_mask, axis=0)
            + tl.sum(dT * KK * strict, axis=0)
        )  # (BLOCK_M,)

        # dQ = dy Hc^T + dS K. Sequential += (one dot's operands staged at a
        # time; the summed form let Triton co-stage both pairs -- see docstring).
        dQc = tl.dot(dyc, tl.trans(Hc), input_precision="ieee")  # (BLOCK_C, BLOCK_K)
        dQc += tl.dot(dS, Ktile, input_precision="ieee")
        tl.store(
            dQ_ptr + Q_base + cc[:, None] * d_k + kk[None, :],
            dQc,
            mask=tok_valid[:, None] & kcol[None, :],
        )

        # dK direct terms = (beta (.) U) dH_c^T + dS^T Q - G Hc^T. Sequential +=;
        # the two (M, M)-operand terms dKK K + dKK^T K are added below (blocked).
        bU = beta_tile[:, None] * U  # (BLOCK_M, BLOCK_V)
        dK_local = tl.dot(bU, tl.trans(dH_c), input_precision="ieee")  # (BLOCK_M, BLOCK_K)
        dK_local += tl.dot(tl.trans(dS), Qtile, input_precision="ieee")
        dK_local -= tl.dot(G, tl.trans(Hc), input_precision="ieee")
        tl.store(
            dK_ptr + K_base + m[:, None] * d_k + kk[None, :],
            dK_local,
            mask=row_valid[:, None] & kcol[None, :],
        )

        # dKK K + dKK^T K -> dK2 (folded into dK by the launch). dKK is (M, M)
        # (65536 B at M=128); block the d_k free dim into 16-wide sub-tiles so it
        # only ever pairs with an (M, 16) slice of K. Output-column blocking is
        # exact (matmul columns are independent). BLOCK_K is a multiple of 16.
        # (tiling scheme after fla's chunk_delta_rule wy_fast bwd, MIT.)
        dKK = dT * beta_tile[None, :] * strict  # (BLOCK_M, BLOCK_M)
        for kb in tl.static_range(BLOCK_K // 16):
            kcols = kb * 16 + tl.arange(0, 16)
            kcmask = kcols < d_k
            Kblk = tl.load(
                K_ptr + K_base + m[:, None] * d_k + kcols[None, :],
                mask=row_valid[:, None] & kcmask[None, :],
                other=0.0,
            ).to(tl.float32)  # (BLOCK_M, 16)
            dKKblk = tl.dot(dKK, Kblk, input_precision="ieee")  # (BLOCK_M, 16)
            dKKblk += tl.dot(tl.trans(dKK), Kblk, input_precision="ieee")
            tl.store(
                dK2_ptr + K_base + m[:, None] * d_k + kcols[None, :],
                dKKblk,
                mask=row_valid[:, None] & kcmask[None, :],
            )

        # dV = G.
        tl.store(
            dV_ptr + V_base + m[:, None] * d_v + vv[None, :],
            G,
            mask=row_valid[:, None] & vcol[None, :],
        )
        tl.store(
            dbeta_ptr + beta_base + m,
            dbeta_local,
            mask=row_valid,
        )

    def _delta_backward_launch(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        beta: torch.Tensor,
        Hbound: torch.Tensor,
        dy: torch.Tensor,
        dH_T: torch.Tensor,
        chunk_size: int,
        needs_input_grad: tuple,
    ) -> tuple:
        """Fused three-kernel backward launch; returns the positional grad tuple.

        Mirrors ``_delta_forward_launch``: B1 (``_delta_bwd_prepass_kernel``,
        chunk-parallel) hoists the transpose solve into ``TinvTBK``/``dHconst``;
        B2 (``_delta_bwd_state_kernel``, serial per ``batch*head``,
        ``num_stages=1``) carries ``dH`` last-to-first writing ``dHbound``/``dH0``;
        B3 (``_delta_bwd_grad_kernel``, chunk-parallel) writes
        ``dQ``/``dK``/``dV``/``dbeta``. ``needs_input_grad`` gates the returned
        tuple exactly like the torch fallback -- an input whose grad is not
        needed gets ``None`` (its kernel output is still computed but dropped;
        the wide grid makes per-input gating not worth a kernel specialization),
        and ``chunk_size`` (non-tensor) always gets ``None``.
        """
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        beta = beta.contiguous()
        Hbound = Hbound.contiguous()
        dy = dy.contiguous()
        dH_T = dH_T.contiguous()

        B, n_heads, T, d_k = Q.shape
        nh = K.shape[3]
        d_v = V.shape[4]
        bh = B * n_heads
        num_chunks = triton.cdiv(T, chunk_size)
        block_m, block_c, block_k, block_v = _delta_block_sizes(nh, chunk_size, d_k, d_v)

        TinvTBK = torch.empty_like(K)
        dHconst = torch.empty(
            B, n_heads, num_chunks, d_k, d_v, device=Q.device, dtype=torch.float32
        )
        dHbound = torch.empty_like(dHconst)
        dH0 = torch.empty(B, n_heads, d_k, d_v, device=Q.device, dtype=torch.float32)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        # dKK K + dKK^T K, written d_k-blocked by B3 and folded into dK below;
        # kept separate so the (M, M) dKK operand never stages against a wide K
        # tile (the launch-fit SMEM fix -- see _delta_bwd_grad_kernel).
        dK2 = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dbeta = torch.zeros_like(beta)

        _delta_bwd_prepass_kernel[(bh, num_chunks)](
            Q,
            K,
            beta,
            dy,
            TinvTBK,
            dHconst,
            T,
            nh,
            num_chunks,
            chunk_size,
            d_k,
            d_v,
            block_m,
            block_c,
            block_k,
            block_v,
        )

        # Serial reverse dH chain; num_stages=1 (strict serial dependency, same
        # as the forward state pass -- prefetch buys nothing, only enlarges SMEM).
        _delta_bwd_state_kernel[(bh,)](
            K,
            TinvTBK,
            dHconst,
            dH_T,
            dHbound,
            dH0,
            T,
            nh,
            num_chunks,
            chunk_size,
            d_k,
            d_v,
            block_m,
            block_k,
            block_v,
            num_stages=1,
        )

        # Chunk-parallel grads; widen warps for the wide (d_k >= 32) tiles, as
        # the readout does. num_stages=1: the two in-kernel substitution loops
        # carry a register accumulator with NO in-loop global load (nothing to
        # prefetch), and the d_k-blocked dKK sub-tile loop DOES load from
        # global in-loop -- multi-buffering it would re-stage the (M, M) dKK
        # operand per stage, re-inflating exactly the SMEM footprint the
        # 16-wide blocking exists to bound (73728 B uniform across the
        # envelope), so single-buffer is the safe choice.
        grad_warps = 8 if d_k >= 32 else 4
        _delta_bwd_grad_kernel[(bh, num_chunks)](
            Q,
            K,
            V,
            beta,
            Hbound,
            dHbound,
            dy,
            dQ,
            dK,
            dK2,
            dV,
            dbeta,
            T,
            nh,
            num_chunks,
            chunk_size,
            d_k,
            d_v,
            block_m,
            block_c,
            block_k,
            block_v,
            num_warps=grad_warps,
            num_stages=1,
        )

        need_Q, need_K, need_V, need_beta, need_H0, _ = needs_input_grad
        return (
            dQ if need_Q else None,
            # Fold the d_k-blocked dKK K + dKK^T K terms into the direct dK.
            (dK + dK2) if need_K else None,
            dV if need_V else None,
            dbeta if need_beta else None,
            dH0 if need_H0 else None,
            None,  # chunk_size (non-tensor)
        )

    def _delta_backward_torch(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        beta: torch.Tensor,
        Hbound: torch.Tensor,
        dy: torch.Tensor,
        dH_T: torch.Tensor,
        chunk_size: int,
        needs_input_grad: tuple,
    ) -> tuple:
        """Reference reverse-chunk backward in torch (the fused kernels' fallback).

        The verified, exact-to-eager hand-derived reverse-chunk recurrence (two
        independent reviewers + the L4 grad-parity suite). Retained as the
        DOCUMENTED fallback path for ``_DeltaScanFn.backward``: if the fused
        three-kernel launch raises (a resource/compile failure at launch), the
        Function warns and runs THIS loop -- safe and exact, only slower, never
        a silent wrong grad. It is also this file's second and last copy of the
        chunked-WY backward math (the fused kernels transcribe from it).

        Returns grads positionally for ``(Q, K, V, beta, H0, chunk_size)``;
        ``chunk_size`` gets ``None`` and any input whose ``needs_input_grad`` is
        ``False`` also gets ``None``.
        """
        need_Q, need_K, need_V, need_beta, need_H0, _ = needs_input_grad

        B, n_heads, T, d_k = Q.shape
        nh = K.shape[3]
        d_v = V.shape[4]

        def mT(x):
            return x.transpose(-1, -2)

        dH = dH_T

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dbeta = torch.zeros_like(beta)

        num_chunks = Hbound.shape[2]
        for c in range(num_chunks - 1, -1, -1):
            start = c * chunk_size
            end = min(start + chunk_size, T)
            C = end - start
            M = nh * C

            Kc = K[:, :, start:end].reshape(B, n_heads, M, d_k)
            Vc = V[:, :, start:end].reshape(B, n_heads, M, d_v)
            betac = beta[:, :, start:end].reshape(B, n_heads, M)
            Qc = Q[:, :, start:end]  # (B, n_heads, C, d_k)
            Hc = Hbound[:, :, c]  # (B, n_heads, d_k, d_v) -- start-of-chunk state
            dyc = dy[:, :, start:end]  # (B, n_heads, C, d_v)

            # --- recompute intra-chunk intermediates (eager math) --------
            KK = Kc @ mT(Kc)  # (B, n_heads, M, M)
            strict = torch.tril(Kc.new_ones(M, M), diagonal=-1)
            eye_M = torch.eye(M, dtype=Kc.dtype, device=Kc.device)
            betac_col = betac.unsqueeze(-2)  # (B, n_heads, 1, M)
            T_mat = eye_M + KK * betac_col * strict  # unit lower-triangular
            U = torch.linalg.solve_triangular(
                T_mat, Vc - Kc @ Hc, upper=False, unitriangular=True
            )  # (B, n_heads, M, d_v)
            del Vc  # only feeds U's rhs; free the M x d_v recompute copy
            S = Qc @ mT(Kc)  # (B, n_heads, C, M)
            read_mask = torch.tril(Qc.new_ones(C, C)).repeat_interleave(nh, dim=1)  # (C, M)
            P = S * betac_col * read_mask  # (B, n_heads, C, M)

            # --- backward through y = Q H + P U --------------------------
            # Large intra-chunk intermediates (the batched (M, M) and
            # (C, M) tensors: KK, T_mat, S, P, dP, dS, dT, dKK) are freed
            # with `del` at their last use so the recompute backward's
            # instantaneous peak stays at a couple of them at once rather
            # than the whole set -- the spec section 7 kernel-peak <= eager
            # -peak invariant is tightest at the widest single-chunk shape
            # (M = nh * chunk_size = 128), where each (M, M) batched tensor
            # is the dominant allocation. Pure memory hygiene; the math is
            # unchanged.
            dQc = dyc @ mT(Hc)  # (B, n_heads, C, d_k)
            dHc = mT(Qc) @ dyc  # (B, n_heads, d_k, d_v)
            dP = dyc @ mT(U)  # (B, n_heads, C, M)
            dU = mT(P) @ dyc  # (B, n_heads, M, d_v)
            del P  # last use above

            # --- backward through H_out = H + K^T (beta * U) -------------
            dK_local = (betac.unsqueeze(-1) * U) @ mT(dH)  # from K^T (beta U)
            dbU = Kc @ dH  # (B, n_heads, M, d_v)
            dU = dU + betac.unsqueeze(-1) * dbU
            dbeta_local = (U * dbU).sum(-1)  # (B, n_heads, M)
            del dbU  # last use above
            dHc = dHc + dH  # identity term of H_out = H + ...

            # --- backward through P = (Q K^T) * beta * read_mask ---------
            dS = dP * betac_col * read_mask  # (B, n_heads, C, M)
            dbeta_local = dbeta_local + (dP * S * read_mask).sum(-2)
            del dP, S  # last use above (read_mask already consumed into dS)
            dQc = dQc + dS @ Kc
            dK_local = dK_local + mT(dS) @ Qc
            del dS  # last use above

            # --- backward through U = T^-1 (V - K H): transpose solve ----
            G = torch.linalg.solve_triangular(
                mT(T_mat), dU, upper=True, unitriangular=True
            )  # T^-T dU, (B, n_heads, M, d_v)
            del T_mat, dU  # both consumed by the transpose solve
            dT = -(G @ mT(U))  # (B, n_heads, M, M)
            dV_local = G  # alias -- G stays live via dV_local until the scatter
            dK_local = dK_local - G @ mT(Hc)
            dHc = dHc - mT(Kc) @ G

            # --- backward through T = I + (K K^T) * beta * strict --------
            dKK = dT * betac_col * strict  # (B, n_heads, M, M)
            dbeta_local = dbeta_local + (dT * KK * strict).sum(-2)
            del dT, KK  # last use above
            dK_local = dK_local + dKK @ Kc + mT(dKK) @ Kc
            del dKK  # last use above

            # --- scatter chunk grads; carry dH to the previous chunk -----
            dQ[:, :, start:end] = dQc
            dK[:, :, start:end] = dK_local.reshape(B, n_heads, C, nh, d_k)
            dV[:, :, start:end] = dV_local.reshape(B, n_heads, C, nh, d_v)
            dbeta[:, :, start:end] = dbeta_local.reshape(B, n_heads, C, nh)
            dH = dHc

        return (
            dQ if need_Q else None,
            dK if need_K else None,
            dV if need_V else None,
            dbeta if need_beta else None,
            dH if need_H0 else None,
            None,  # chunk_size (non-tensor)
        )

    class _DeltaBackwardFallbackWarning(RuntimeWarning):
        """Raised when ``_DeltaScanFn.backward`` falls back to the torch loop.

        A distinct category (not a bare ``RuntimeWarning``) so the
        grad-parity suite (``_run_delta_grad_parity_body``'s ``check_raw``/
        ``check_module``) can assert, per case, that the fused Kernel 7 trio
        actually executed -- by recording warnings under
        ``warnings.catch_warnings(record=True)`` around the triton-path
        backward and filtering on THIS category, not a message string.
        Because ``_delta_backward_torch`` is numerically EXACT, a fallback
        that fires silently would let grad parity stay green forever even if
        the fused kernels never once executed correctly, silently defeating
        the round's speed goal -- the whole reason the parity oracle treats
        this warning itself as a failure signal, not informational noise.
        """

    def _delta_backward_fallback_exceptions() -> tuple[type[BaseException], ...]:
        """Genuine Triton launch/resource/compile failure types, resolved lazily.

        ``_DeltaScanFn.backward`` narrows its ``except`` clause to EXACTLY
        this tuple -- never a bare ``except Exception`` -- so only a failure
        that could not have executed the kernel at all falls back to the
        exact torch loop: shared-memory/register overflow
        (``triton.runtime.errors.OutOfResources``), a Triton compile error
        (``triton.compiler.errors.CompilationError``), or a CUDA allocation
        failure building a backward workspace tensor
        (``torch.cuda.OutOfMemoryError`` -- the same scoped resource-failure
        precedent ``scripts/bench_scans.py``'s ``_compile_backend_exceptions``
        uses for the compile path). A bug in the launch's own Python glue
        (wrong shape, wrong dtype, an index error, or a ``RuntimeError`` from
        a genuine kernel correctness bug such as an illegal memory access)
        must propagate loudly instead: the torch fallback is numerically
        EXACT, so silently swallowing a correctness bug there would mask it
        forever -- grad parity would stay green without the fused kernels
        ever having run correctly, the exact failure mode the commit gate
        flagged.

        Resolved lazily and defensively at import time (Triton's exception
        module paths have moved across versions): each candidate type is
        added only if importable under this Triton install; an unresolvable
        type is simply omitted (never breaks this module's import). If
        nothing resolves beyond the always-present ``OutOfMemoryError``, the
        catch stays scoped to that one type rather than silently widening.
        """
        types: list[type[BaseException]] = [torch.cuda.OutOfMemoryError]
        try:
            from triton.runtime.errors import OutOfResources

            types.append(OutOfResources)
        except ImportError:
            pass
        try:
            from triton.compiler.errors import CompilationError

            types.append(CompilationError)
        except ImportError:
            pass
        return tuple(types)

    _DELTA_BWD_KERNEL_EXCEPTIONS = _delta_backward_fallback_exceptions()

    class _DeltaScanFn(torch.autograd.Function):
        """Autograd wrapper making the three-kernel delta forward differentiable.

        The forward is the raw Triton launch (``_delta_forward_launch``); the
        backward is a hand-derived reverse-chunk recurrence (no autograd
        through the kernels -- they are plain ``@triton.jit`` launches, opaque
        to autograd). This is the seam that lets the eager-only target user
        ("compiling isn't always an option") get a full training-step speedup:
        a forward-only kernel would capture only the forward share, so the
        round's fwd+bwd speed bar is met here.

        Differentiates w.r.t. the five assembled sequence inputs ``Q``, ``K``,
        ``V``, ``beta`` and the initial state ``H0`` (all ``needs_input_grad``
        combinations, including the ``H0`` grad state-carrying callers need);
        ``chunk_size`` is a non-tensor and gets a ``None`` grad. Parameter
        grads (``_coeffs``/``out_proj``) flow through these five in ordinary
        autograd outside the Function, exactly as on the eager path.

        Recompute-based, per spec section 5: forward saves only the assembled
        inputs plus the per-chunk boundary states ``Hbound`` (cheap:
        ``num_chunks * d_k * d_v`` per batch-head); the intra-chunk
        intermediates (``T``, ``U``, the masks) are recomputed in the
        backward, so kernel-path peak training memory stays at or below the
        eager path's. The backward is a fused three-kernel Triton trio
        (``_delta_backward_launch``) mirroring the forward trio -- a
        chunk-parallel transpose-solve hoist, a serial ``dH`` chain, and a
        chunk-parallel grad pass -- so the round's fwd+bwd speed bar is met
        without a torch-op backward that serializes ~12 ops per chunk. If that
        launch raises one of ``_DELTA_BWD_KERNEL_EXCEPTIONS`` (a genuine
        launch/resource/compile failure -- SMEM/register overflow, a Triton
        compile error, a CUDA allocation failure), it falls back to the
        verified torch reverse-chunk loop (``_delta_backward_torch``) with a
        ``_DeltaBackwardFallbackWarning``: safe and exact, only slower, never
        a silent wrong grad. Any OTHER exception (a Python-glue bug: wrong
        shape, wrong dtype, an index error) propagates loudly instead of
        falling back -- the exact-fallback would otherwise mask a real
        correctness bug forever, since grad parity would stay green without
        the fused kernels ever having run correctly. ``@once_differentiable``
        runs the backward under ``no_grad`` so a second ``.backward()``
        raises rather than silently mis-differentiating (double-backward is
        unsupported, the standard custom-Function limitation).
        """

        @staticmethod
        def forward(ctx, Q, K, V, beta, H0, chunk_size):
            """Run the three-kernel launch; save inputs + boundary states for backward.

            Returns the public ``(y, H_T)`` pair; the boundary states
            ``Hbound`` are stashed on the tape (not returned) so the reverse
            chunk loop can seed each chunk's start-of-chunk state without
            re-running the forward passes.
            """
            y, H_T, Hbound = _delta_forward_launch(Q, K, V, beta, H0, chunk_size)
            ctx.save_for_backward(Q, K, V, beta, H0, Hbound)
            ctx.chunk_size = chunk_size
            return y, H_T

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, dy, dH_T):
            """Fused reverse-chunk backward (Kernel 7 trio); torch loop fallback.

            Launches ``_delta_backward_launch`` (B1 transpose-solve hoist, B2
            serial ``dH`` chain, B3 chunk-parallel grads) seeded from the saved
            inputs and the saved start-of-chunk states ``Hbound``. If that
            launch raises one of ``_DELTA_BWD_KERNEL_EXCEPTIONS`` (a genuine
            launch/resource/compile failure -- SMEM/register overflow, a
            Triton compile error, or a CUDA allocation failure; see
            ``_delta_backward_fallback_exceptions``'s docstring), warns with
            ``_DeltaBackwardFallbackWarning`` and runs the verified torch
            reverse-chunk loop (``_delta_backward_torch``) -- safe, exact,
            slower, never a silent wrong grad. Any OTHER exception (a bug in
            the launch's own Python glue -- wrong shape, wrong dtype, an
            index error) is NOT caught here and propagates loudly: catching
            it would let the numerically-exact torch fallback mask the bug
            forever, since grad parity would stay green without the fused
            kernels ever having run correctly. Returns grads positionally for
            ``(Q, K, V, beta, H0, chunk_size)``; ``chunk_size`` (non-tensor)
            gets ``None``, and any input whose ``needs_input_grad`` is
            ``False`` also gets ``None``.

            ``dy`` / ``dH_T`` always arrive as materialized (possibly all-zero)
            tensors here, never ``None``: forward never calls
            ``ctx.set_materialize_grads(False)``, so an unused output still gets
            a concrete zero grad -- see the ``register_autograd``
            None-grad-handling note above (~line 1277) for the same
            verified-against-torch-2.8 behavior on the sibling backwards.
            """
            Q, K, V, beta, H0, Hbound = ctx.saved_tensors
            chunk_size = ctx.chunk_size
            try:
                return _delta_backward_launch(
                    Q, K, V, beta, Hbound, dy, dH_T, chunk_size, ctx.needs_input_grad
                )
            except _DELTA_BWD_KERNEL_EXCEPTIONS as exc:  # launch/resource/compile failure only
                warnings.warn(
                    "DeltaMinGRU fused backward launch failed "
                    f"({type(exc).__name__}: {exc}); falling back to the exact "
                    "torch reverse-chunk loop (correct, slower).",
                    _DeltaBackwardFallbackWarning,
                    stacklevel=2,
                )
                return _delta_backward_torch(
                    Q, K, V, beta, Hbound, dy, dH_T, chunk_size, ctx.needs_input_grad
                )

    def delta_scan_impl(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        beta: torch.Tensor,
        H0: torch.Tensor,
        *,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Envelope-guarded entry point for the DeltaMinGRU chunked-WY kernel.

        Mirrors ``angle_scan_impl``: validates the envelope (raising
        ``ScanFallback`` with a distinct per-violation reason) and funnels any
        launch/compile failure into ``ScanFallback`` too, so
        ``min_gru``'s delta dispatch seam (Task 3) only ever catches that one
        exception type. Not a ``SCAN_IMPLS`` entry (the delta path is a
        module-forward fast path, not one of the four scan ops); discovered by
        ``hasattr(triton_scans, "delta_scan_impl")``, exactly like
        ``angle_scan_impl``.

        Parameters (spec section 6): ``Q`` ``(B, n_heads, T, d_k)``; ``K``
        ``(B, n_heads, T, nh, d_k)``; ``V`` ``(B, n_heads, T, nh, d_v)``;
        ``beta`` ``(B, n_heads, T, nh)``; ``H0`` ``(B, n_heads, d_k, d_v)`` --
        the eager ``_forward_chunked`` post-permute assemblies. ``chunk_size``
        is keyword-only (self-documenting at the call site).

        Returns
        -------
        tuple of torch.Tensor
            ``y`` ``(B, n_heads, T, d_v)`` and ``H_T`` ``(B, n_heads, d_k,
            d_v)``. The caller owns the final ``(B, T, n_heads, d_v)`` restore
            and ``out_proj``.
        """
        _delta_validate_envelope(Q, K, V, beta, H0, chunk_size)
        try:
            # Route through the autograd Function (not the raw launch) so the
            # registered entry point is differentiable: `.backward()` through
            # it yields grads for Q/K/V/beta/H0, and parameter grads flow via
            # `_coeffs`/`out_proj` exactly as on the eager path. The Function
            # runs `_delta_forward_launch` internally and stashes the boundary
            # states its recompute backward needs.
            y, H_T = _DeltaScanFn.apply(Q, K, V, beta, H0, chunk_size)
        except Exception as exc:  # kernel launch/compile failure -> loud fallback
            raise ScanFallback(f"DeltaMinGRU Triton kernel failed: {exc}") from exc
        return y, H_T

    def _linear_scan_impl(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SCAN_IMPLS entry for ``linear_scan`` (k = v = 1, always in envelope).

        Guards the ``(B, T, D)`` rank and the ``a``/``b`` shape agreement
        (the kernel's per-channel pointer walk assumes both operands share
        the layout) before any launch, so a mismatched input raises
        ``ScanFallback`` rather than a bare error deep in the kernel.
        """
        if a.ndim != 3 or b.ndim != 3 or a.shape != b.shape:
            raise ScanFallback(
                f"linear_scan (a={tuple(a.shape)}, b={tuple(b.shape)}) "
                "outside the Triton envelope: need matching (B, T, D) a and b"
            )
        try:
            return linear_scan_fwd(a, b)
        except Exception as exc:  # kernel launch/compile failure -> loud fallback
            raise ScanFallback(f"linear_scan Triton kernel failed: {exc}") from exc

    def _parallel_scan_log_impl(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
        """SCAN_IMPLS entry for ``parallel_scan_log`` (elementwise in D, always in envelope).

        Guards the ``log_coeffs`` ``(B, T, D)`` rank and the ``log_values``
        ``(B, T+1, D)`` shape (one extra leading time slot for ``log(h_0)``)
        before any launch: the kernel strides ``log_values`` by ``T+1``
        timesteps, so a wrong rank or a missing/extra value column would
        otherwise mis-address memory instead of falling back cleanly.
        """
        if (
            log_coeffs.ndim != 3
            or log_values.ndim != 3
            or log_values.shape
            != (log_coeffs.shape[0], log_coeffs.shape[1] + 1, log_coeffs.shape[2])
        ):
            raise ScanFallback(
                f"parallel_scan_log (log_coeffs={tuple(log_coeffs.shape)}, "
                f"log_values={tuple(log_values.shape)}) outside the Triton "
                "envelope: need (B, T, D) log_coeffs and (B, T+1, D) log_values"
            )
        try:
            return parallel_scan_log_fwd(log_coeffs, log_values)
        except Exception as exc:  # kernel launch/compile failure -> loud fallback
            raise ScanFallback(f"parallel_scan_log Triton kernel failed: {exc}") from exc

    def _matrix_scan_impl(M: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SCAN_IMPLS entry for ``matrix_scan`` (k = 2, v = 1, fixed in envelope).

        Maps onto the generic kernel by viewing the ``(B, T, n, 2)``
        injection ``b`` as a ``(B, T, n, 2, 1)`` matrix, then squeezing the
        trailing ``v = 1`` axis off the returned ``Bbar``. Guards the
        ``M`` ``(B, T, n, 2, 2)`` / ``b`` ``(B, T, n, 2)`` ranks, the 2x2
        block shape, and the shared ``(B, T, n)`` lane grid before any
        launch (same gap class as the generic op's up-front check).
        """
        if (
            M.ndim != 5
            or b.ndim != 4
            or M.shape[-2:] != (2, 2)
            or b.shape[-1] != 2
            or M.shape[:3] != b.shape[:3]
        ):
            raise ScanFallback(
                f"matrix_scan (M={tuple(M.shape)}, b={tuple(b.shape)}) "
                "outside the Triton envelope: need (B, T, n, 2, 2) M and "
                "(B, T, n, 2) b sharing the (B, T, n) lane grid"
            )
        Bm = b.unsqueeze(-1)
        try:
            Abar, Bbar = affine_scan_fwd(M, Bm)
            # Squeeze inside the try so the ScanFallback invariant (every
            # exit is either a clean result or a ScanFallback) is enforced
            # by construction, even if the squeeze itself were to fail.
            return Abar, Bbar.squeeze(-1)
        except Exception as exc:  # kernel launch/compile failure -> loud fallback
            raise ScanFallback(f"matrix_scan Triton kernel failed: {exc}") from exc

    def _matrix_affine_scan_impl(
        A: torch.Tensor, Bm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SCAN_IMPLS entry for ``matrix_affine_scan`` (generic k, v).

        Enforces the kernel envelope up front: a wrong rank, an unsupported
        ``k``/``v``, a non-square transition, or an ``A``/``Bm`` shape
        disagreement raises ``ScanFallback`` so the caller loudly falls back
        to eager rather than launching a kernel the tile shapes cannot
        represent (or indexing ``.shape[-2]``/``.shape[:3]`` on a
        wrong-rank tensor). The ``A``/``Bm`` agreement checks (matching
        ``(B, T, n)`` lane grid and ``Bm``'s ``k`` rows) guard the kernel's
        pointer arithmetic, which assumes both operands share the lane
        layout.
        """
        if A.ndim != 5 or Bm.ndim != 5:
            raise ScanFallback(
                f"matrix_affine_scan (A={tuple(A.shape)}, Bm={tuple(Bm.shape)}) "
                "outside the Triton envelope: need 5-D (B, T, n, k, k) A and "
                "(B, T, n, k, v) Bm"
            )
        k = A.shape[-1]
        v = Bm.shape[-1]
        if (
            k not in _ENVELOPE
            or v not in _ENVELOPE
            or A.shape[-2] != k
            or Bm.shape[-2] != k
            or A.shape[:3] != Bm.shape[:3]
        ):
            raise ScanFallback(
                f"matrix_affine_scan (A={tuple(A.shape)}, Bm={tuple(Bm.shape)}) "
                f"outside the Triton envelope: need k,v in {sorted(_ENVELOPE)}, "
                f"a square A (A.shape[-2:] == (k, k)), Bm with k rows "
                f"(Bm.shape[-2] == k), and A/Bm sharing the (B, T, n) lane grid"
            )
        try:
            return affine_scan_fwd(A, Bm)
        except Exception as exc:  # kernel launch/compile failure -> loud fallback
            raise ScanFallback(f"matrix_affine_scan Triton kernel failed: {exc}") from exc

    SCAN_IMPLS = {
        "parallel_scan_log": _parallel_scan_log_impl,
        "linear_scan": _linear_scan_impl,
        "matrix_scan": _matrix_scan_impl,
        "matrix_affine_scan": _matrix_affine_scan_impl,
    }


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
    if not _HAS_TRITON:
        return f"triton not importable: {_TRITON_IMPORT_ERROR}"
    return True


# Public surface (additive metadata, Phase-4 packaging prep): the
# availability probe, the fallback-signal exception, the scan-op impl
# registry, and the angle-fused entry point + the raw triton_op wrappers.
# Built conditionally because the Triton-gated names only exist in the
# namespace when `_HAS_TRITON` (a CPU-only/no-Triton install still imports
# this module successfully -- see the module docstring -- so `__all__`
# must not name an attribute that doesn't exist in that case). Everything
# else (the `_`-prefixed kernels, dispatch helpers, and selftest runners)
# is implementation detail, not part of the documented public API.
__all__ = ["available", "ScanFallback", "SCAN_IMPLS"]
if _HAS_TRITON:
    __all__ += [
        "angle_scan_impl",
        "delta_scan_impl",
        "affine_scan_fwd",
        "linear_scan_fwd",
        "parallel_scan_log_fwd",
        "affine_scan_bwd",
        "linear_scan_bwd",
        "angle_scan_fwd",
        "angle_scan_bwd",
    ]


# --- Forward-parity selftest -------------------------------
#
# Runs only when CUDA + Triton are present; otherwise skips LOUDLY (prints
# the reason, exits 0) -- a vacuous pass is a bug. The reference is always
# the eager scan on identical inputs, obtained by forcing MINGRU_SCAN=eager
# so min_gru's own dispatch guard stays on the eager path.


def _max_abs_rel(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Max absolute and max relative deviation between two tensors (fp64)."""
    out = out.double()
    ref = ref.double()
    abs_err = (out - ref).abs()
    rel_err = abs_err / ref.abs().clamp_min(1e-12)
    return abs_err.max().item(), rel_err.max().item()


def _parity_row(
    op: str,
    shape: str,
    dtype_sample: torch.Tensor,
    direction: str,
    gate_atol: float,
    gate_rtol: float,
    max_abs_err: float | None,
    max_rel_err: float | None,
    passed: bool,
    *,
    ref_fp64_dev: float | None = None,
) -> dict:
    """Build one parity-conformance artifact row (shared by every runner).

    One dict per GATED case (never the informational bf16 rows -- those have
    no pass/fail verdict and are not part of the persisted parity-conformance
    artifact). ``shape`` is the same descriptive tag string each runner
    already prints (e.g. ``"B=2 T=64 k=16 v=16"``) -- reusing it rather than
    re-deriving a separate structured shape dict avoids a second shape
    representation to keep in sync with the console output. ``dtype_sample``
    is any tensor from this case's inputs; only its dtype is read.

    ``ref_fp64_dev``, keyword-only, defaults to ``None`` so the three
    pre-existing runners (whose gates are fixed constants, not derived from a
    per-case fp64 reference) are unaffected. The DeltaMinGRU parity runners
    (``_run_delta_forward_parity``, ``_run_delta_grad_parity``) pass the
    eager path's own fp32-vs-fp64 deviation on this case's inputs here --
    the tolerance-justification rule's ``max_abs_err <= 10 * ref_fp64_dev``
    reference value (spec 9.2/9.3) -- alongside ``gate_atol`` (the gate
    actually applied, ``max(10 * ref_fp64_dev, a flat floor)``) and
    ``max_abs_err`` (the gated kernel-vs-eager deviation), so both
    deviations this rule compares land in the persisted artifact.
    """
    return {
        "op": op,
        "shape": shape,
        "dtype": str(dtype_sample.dtype).replace("torch.", ""),
        "direction": direction,
        "gate_atol": gate_atol,
        "gate_rtol": gate_rtol,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "pass": passed,
        "ref_fp64_dev": ref_fp64_dev,
    }


def _rand_contractive_matrix(shape: tuple[int, ...], k: int, device) -> torch.Tensor:
    """Random square-transition tensor scaled to keep the T-product bounded.

    Scaling by ``0.5 / sqrt(k)`` holds each block's spectral norm safely
    below 1, so the running product over T <= 1024 steps neither vanishes
    to the fp32 floor nor overflows -- keeping the parity comparison
    meaningful at the long-T end of the shape matrix.
    """
    return torch.randn(shape, device=device, dtype=torch.float32) * (0.5 / (k**0.5))


def _log_space_inputs(B: int, T: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(log_coeffs, log_values)`` in the shape ``MinGRU.forward`` produces.

    The ``log_coeffs = -softplus(k)`` and ``log_z + log_tilde_h`` value
    columns follow ``min_gru.py``'s construction exactly, so the parity
    sweeps exercise the kernel on the log-magnitude regime the model
    actually produces (decay coeffs < 0, a ``T+1`` value column). The
    leading ``log(h_0)`` slot, by contrast, is a synthetic ``randn * 0.1``
    stand-in -- ``MinGRU.forward`` derives it from the caller's initial
    state, which the parity tests do not model -- so this term mimics the
    slot's magnitude/placement, not its exact provenance. Shared by the
    forward- and gradient-parity sweeps.
    """
    from mingru import min_gru

    k_pre = torch.randn(B, T, D, device=device)
    log_coeffs = -F.softplus(k_pre)  # log(1 - sigmoid(k)) < 0
    log_z = -F.softplus(-k_pre)  # log(sigmoid(k))
    log_tilde_h = min_gru.log_g(torch.randn(B, T, D, device=device))
    log_h0 = torch.randn(B, 1, D, device=device) * 0.1
    log_values = torch.cat([log_h0, log_z + log_tilde_h], dim=1)
    return log_coeffs, log_values


def _linear_scan_inputs(B: int, T: int, D: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(a, b)`` for ``linear_scan`` (k=1): ``a`` in ``(-0.9, 0.9)``.

    Shared by the forward- and gradient-parity sweeps.
    """
    a = torch.rand(B, T, D, device=device) * 1.8 - 0.9
    b = torch.randn(B, T, D, device=device) * 0.1
    return a, b


def _matrix_scan_inputs(B: int, T: int, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(M, b)`` for ``matrix_scan`` (k=2, v=1): contractive 2x2 blocks.

    Shared by the forward- and gradient-parity sweeps.
    """
    M = _rand_contractive_matrix((B, T, n, 2, 2), 2, device)
    b = torch.randn(B, T, n, 2, device=device) * 0.1
    return M, b


def _matrix_affine_scan_inputs(
    B: int, T: int, n: int, k: int, v: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(A, Bm)`` for ``matrix_affine_scan`` (generic k, v): contractive A.

    Shared by the forward- and gradient-parity sweeps.
    """
    A = _rand_contractive_matrix((B, T, n, k, k), k, device)
    Bm = torch.randn(B, T, n, k, v, device=device) * 0.1
    return A, Bm


# Shape matrix shared by both parity sweeps: T=1 exercises a
# program that never traverses the loop-increment path (a single timestep);
# the rest covers non-power-of-two lengths and the long-T end. Defined once
# so ``_run_forward_parity`` and ``_run_grad_parity`` cannot silently diverge.
_PARITY_TS = (1, 13, 64, 128, 1024)
_PARITY_BS = (2, 128)


def _run_forward_parity(collect: list[dict] | None = None) -> int:
    """Run the forward-parity matrix; return process exit code.

    ``collect``, when not ``None``, is a caller-owned list that this sweep
    APPENDS one row dict to per GATED case (never the informational bf16
    rows, which have no pass/fail verdict) -- the parity-conformance
    artifact seam. Defaults to ``None`` so
    ``python triton_scans.py``'s own selftest invocation (and any other
    caller that doesn't pass it) is byte-identical to before this seam
    existed: nothing is collected, nothing else changes.

    A thin wrapper: ``_scan_env("eager")`` forces the eager reference path
    for the whole sweep and restores ``MINGRU_SCAN`` to whatever it was
    before this call on return, so `--check` never leaks ``eager`` into
    the rest of the process; the actual sweep is ``_run_forward_parity_body``.
    """
    with _scan_env("eager"):
        return _run_forward_parity_body(collect)


def _run_forward_parity_body(collect: list[dict] | None = None) -> int:
    """The forward-parity sweep itself; see ``_run_forward_parity``."""
    from mingru import min_gru

    device = "cuda"
    torch.manual_seed(0)
    # Flat conformance bound: outputs <= 1e-5 for fp32. rtol=0 so this is
    # exactly max_abs_err <= atol, not a looser relative gate -- do not
    # reintroduce a nonzero rtol here.
    atol, rtol = 1e-5, 0.0
    Ts, Bs = _PARITY_TS, _PARITY_BS

    failures: list[str] = []
    n_pass = 0

    def check(
        name: str,
        tag: str,
        *inputs: torch.Tensor,
        informational: bool = False,
        case_atol: float = atol,
        case_rtol: float = rtol,
    ) -> None:
        """Run the Triton impl and eager reference on ``inputs`` and compare.

        The Triton call is guarded so a ``ScanFallback`` or launch/compile
        error is reported as a per-case failure (loud, but not aborting the
        whole sweep) rather than a bare traceback.

        ``case_atol``/``case_rtol`` default to the flat conformance bound
        (``atol=1e-5, rtol=0``) that governs the affine ops; a case may
        override them where that flat bound is not physically meetable in
        fp32 (see the ``parallel_scan_log`` section for the log-space
        exception and its justification).
        """
        nonlocal n_pass
        label = f"{name} {tag}"
        try:
            out = SCAN_IMPLS[name](*inputs)
        except Exception as exc:  # ScanFallback or a kernel launch/compile error
            failures.append(f"{label}: Triton path raised {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
            if collect is not None and not informational:
                collect.append(
                    _parity_row(
                        name, tag, inputs[0], "fwd", case_atol, case_rtol, None, None, False
                    )
                )
            return
        ref = getattr(min_gru, name)(*inputs)
        # Ops return either a pair (the affine scans) or a single tensor
        # (parallel_scan_log). Normalize to a tuple so one comparison loop
        # handles both -- one harness, no fork.
        outs = out if isinstance(out, tuple) else (out,)
        refs = ref if isinstance(ref, tuple) else (ref,)
        abs_err = 0.0
        rels: list[float] = []
        ok = True
        for out_i, ref_i in zip(outs, refs):
            abs_i, rel_i = _max_abs_rel(out_i, ref_i)
            abs_err = max(abs_err, abs_i)
            rels.append(rel_i)
            ok = ok and torch.allclose(out_i.float(), ref_i.float(), atol=case_atol, rtol=case_rtol)
        if informational:
            print(f"  [info] {label}: max_abs={abs_err:.2e} (not gated)")
            return
        if ok:
            n_pass += 1
        else:
            rel_str = ", ".join(f"{r:.2e}" for r in rels)
            failures.append(f"{label}: max_abs={abs_err:.2e} (rel={rel_str})")
            print(f"  [FAIL] {label}: max_abs={abs_err:.2e}")
        if collect is not None:
            collect.append(
                _parity_row(
                    name,
                    tag,
                    inputs[0],
                    "fwd",
                    case_atol,
                    case_rtol,
                    abs_err,
                    max(rels) if rels else None,
                    ok,
                )
            )

    # parallel_scan_log tolerance is the log-space exception to the flat
    # atol=1e-5 fp32 gate, and it is NOT a loosening of that gate (which
    # still governs the affine ops verbatim). The eager reference itself
    # runs torch.logcumsumexp over T+1 terms in fp32; accumulating a
    # log-sum-exp that long carries an intrinsic fp32 error near 1e-4 at
    # T=1024 (empirically the eager fp32 path is already ~1.4e-4 from its
    # own fp64 evaluation, and the kernel's online max-shifted accumulator
    # is in fact CLOSER to the fp64 truth than eager fp32 is). No correct
    # fp32 kernel can match a co-noisy fp32 reference to 1e-5 at that
    # length. The gate therefore uses the mixed (atol=1e-4, rtol=1e-4)
    # bound -- the same tolerance min_gru.py's own parallel-scan parity
    # self-tests use, matching those existing selftest standards -- so it
    # still catches any real kernel defect (off by
    # O(1), a shifted index, a dropped term) while honoring fp32 physics.
    _LOG_SCAN_ATOL, _LOG_SCAN_RTOL = 1e-4, 1e-4
    print("parallel_scan_log (log-space, elementwise in D):")
    for B in Bs:
        for T in Ts:
            D = 64
            log_coeffs, log_values = _log_space_inputs(B, T, D, device)
            check(
                "parallel_scan_log",
                f"B={B} T={T} D={D}",
                log_coeffs,
                log_values,
                case_atol=_LOG_SCAN_ATOL,
                case_rtol=_LOG_SCAN_RTOL,
            )

    print("linear_scan (k=1):")
    for B in Bs:
        for T in Ts:
            D = 64
            a, b = _linear_scan_inputs(B, T, D, device)
            check("linear_scan", f"B={B} T={T} D={D}", a, b)

    print("matrix_scan (k=2, v=1):")
    for B in Bs:
        for T in Ts:
            n = 4
            M, b = _matrix_scan_inputs(B, T, n, device)
            check("matrix_scan", f"B={B} T={T} n={n}", M, b)

    print("matrix_affine_scan (generic k, v):")
    for B in Bs:
        for T in Ts:
            n = 1
            for k in sorted(_ENVELOPE):
                for v in sorted(_ENVELOPE):
                    A, Bm = _matrix_affine_scan_inputs(B, T, n, k, v, device)
                    check("matrix_affine_scan", f"B={B} T={T} k={k} v={v}", A, Bm)

    print("matrix_affine_scan (n>1 lane-grid exercise):")
    for B in Bs:
        for T in Ts:
            n = 3
            k, v = 8, 4
            A, Bm = _matrix_affine_scan_inputs(B, T, n, k, v, device)
            check("matrix_affine_scan", f"B={B} T={T} n={n} k={k} v={v}", A, Bm)

    print("bf16-input rows (informational):")
    B, T = 2, 64
    lc32, lv32 = _log_space_inputs(B, T, 64, device)
    check(
        "parallel_scan_log",
        "bf16 B=2 T=64 D=64",
        lc32.bfloat16(),
        lv32.bfloat16(),
        informational=True,
    )
    a, b = _linear_scan_inputs(B, T, 64, device)
    check("linear_scan", "bf16 B=2 T=64 D=64", a.bfloat16(), b.bfloat16(), informational=True)
    M, b2 = _matrix_scan_inputs(B, T, 2, device)
    check("matrix_scan", "bf16 B=2 T=64 n=2", M.bfloat16(), b2.bfloat16(), informational=True)
    A, Bm = _matrix_affine_scan_inputs(B, T, 1, 8, 4, device)
    check(
        "matrix_affine_scan",
        "bf16 B=2 T=64 k=8 v=4",
        A.bfloat16(),
        Bm.bfloat16(),
        informational=True,
    )

    print()
    if failures:
        print(f"FORWARD PARITY FAILED: {len(failures)} case(s), {n_pass} passed")
        for line in failures:
            print(f"  {line}")
        return 1
    print(
        f"FORWARD PARITY PASSED: {n_pass} fp32 case(s) -- affine ops within "
        f"the flat atol={atol:.0e} rtol={rtol:.0e} gate; parallel_scan_log "
        f"within its documented log-space bound atol={_LOG_SCAN_ATOL:.0e} "
        f"rtol={_LOG_SCAN_RTOL:.0e}"
    )
    return 0


def _run_grad_parity(collect: list[dict] | None = None) -> int:
    """Run the gradient-parity matrix; return process exit code.

    For every op the Triton path is differentiable
    (``register_autograd`` on each forward ``triton_op``). This sweep drives
    a scalar loss ``sum(cotangent * output)`` through both the Triton path
    and the eager reference on identical inputs and identical cotangents,
    then compares the resulting INPUT gradients. The gate is the
    parameter-grad bound (``atol=1e-3``); it exercises the reversed-scan
    adjoint kernels (affine, k=1) and the log op's recompute backward.

    ``collect``: see ``_run_forward_parity`` -- same optional append-only
    parity-conformance-artifact seam, same ``None``-default no-op contract.

    A thin wrapper: ``_scan_env("eager")`` forces the eager reference path
    for the whole sweep and restores ``MINGRU_SCAN`` on return -- see
    ``_run_forward_parity``. The actual sweep is ``_run_grad_parity_body``.
    """
    with _scan_env("eager"):
        return _run_grad_parity_body(collect)


def _run_grad_parity_body(collect: list[dict] | None = None) -> int:
    """The gradient-parity sweep itself; see ``_run_grad_parity``."""
    from mingru import min_gru

    device = "cuda"
    torch.manual_seed(0)
    # Plan constraint: parameter-grad parity <= 1e-3. rtol left at 0 so this
    # is exactly max_abs_grad_err <= atol against the eager autograd grads.
    atol, rtol = 1e-3, 0.0
    Ts, Bs = _PARITY_TS, _PARITY_BS

    failures: list[str] = []
    n_pass = 0

    def check_grad(name: str, tag: str, *inputs: torch.Tensor) -> None:
        """Compare Triton-path vs eager input grads for one case.

        ``inputs`` are grad-free leaves; each is cloned into two independent
        leaves (one per path) so the two backward passes never share a
        graph. Cotangents are drawn once from a fixed seed and reused for
        both paths, so any grad difference is the backward's, not the
        loss's.
        """
        nonlocal n_pass
        label = f"{name} {tag}"
        tri_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
        eager_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
        try:
            out_t = SCAN_IMPLS[name](*tri_inputs)
        except Exception as exc:  # ScanFallback or a kernel launch/compile error
            failures.append(f"{label}: Triton path raised {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
            if collect is not None:
                collect.append(
                    _parity_row(name, tag, inputs[0], "grad", atol, rtol, None, None, False)
                )
            return
        out_e = getattr(min_gru, name)(*eager_inputs)
        outs_t = out_t if isinstance(out_t, tuple) else (out_t,)
        outs_e = out_e if isinstance(out_e, tuple) else (out_e,)

        # Identical cotangents for both paths (same seed, same shapes).
        torch.manual_seed(1234)
        cots = [torch.randn_like(o) for o in outs_e]
        loss_t = sum((c * o).sum() for c, o in zip(cots, outs_t))
        loss_e = sum((c * o).sum() for c, o in zip(cots, outs_e))
        loss_t.backward()
        loss_e.backward()

        max_err = 0.0
        max_rel = 0.0
        ok = True
        for tri_in, eager_in in zip(tri_inputs, eager_inputs):
            gt = tri_in.grad
            ge = eager_in.grad
            if gt is None or ge is None:
                # Both must agree on whether a grad exists for this input.
                if gt is not ge:
                    ok = False
                    failures.append(f"{label}: grad presence mismatch (triton={gt is not None})")
                continue
            err, rel = _max_abs_rel(gt, ge)
            max_err = max(max_err, err)
            max_rel = max(max_rel, rel)
            ok = ok and torch.allclose(gt.float(), ge.float(), atol=atol, rtol=rtol)
        if ok:
            n_pass += 1
        else:
            failures.append(f"{label}: max_grad_abs={max_err:.2e}")
            print(f"  [FAIL] {label}: max_grad_abs={max_err:.2e}")
        if collect is not None:
            collect.append(
                _parity_row(name, tag, inputs[0], "grad", atol, rtol, max_err, max_rel, ok)
            )

    print("grad parallel_scan_log (recompute backward):")
    for B in Bs:
        for T in Ts:
            D = 64
            log_coeffs, log_values = _log_space_inputs(B, T, D, device)
            check_grad("parallel_scan_log", f"B={B} T={T} D={D}", log_coeffs, log_values)

    print("grad linear_scan (k=1 reverse-scan adjoint):")
    for B in Bs:
        for T in Ts:
            D = 64
            a, b = _linear_scan_inputs(B, T, D, device)
            check_grad("linear_scan", f"B={B} T={T} D={D}", a, b)

    print("grad matrix_scan (k=2, v=1 via generic adjoint):")
    for B in Bs:
        for T in Ts:
            n = 4
            M, b = _matrix_scan_inputs(B, T, n, device)
            check_grad("matrix_scan", f"B={B} T={T} n={n}", M, b)

    print("grad matrix_affine_scan (generic k, v reverse-scan adjoint):")
    for B in Bs:
        for T in Ts:
            n = 1
            for k in sorted(_ENVELOPE):
                for v in sorted(_ENVELOPE):
                    A, Bm = _matrix_affine_scan_inputs(B, T, n, k, v, device)
                    check_grad("matrix_affine_scan", f"B={B} T={T} k={k} v={v}", A, Bm)

    print("grad matrix_affine_scan (n>1 lane-grid exercise):")
    for B in Bs:
        for T in Ts:
            n = 3
            k, v = 8, 4
            A, Bm = _matrix_affine_scan_inputs(B, T, n, k, v, device)
            check_grad("matrix_affine_scan", f"B={B} T={T} n={n} k={k} v={v}", A, Bm)

    print()
    if failures:
        print(f"GRAD PARITY FAILED: {len(failures)} case(s), {n_pass} passed")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"GRAD PARITY PASSED: {n_pass} case(s) within atol={atol:.0e} rtol={rtol:.0e}")
    return 0


def _run_angle_fused_parity(collect: list[dict] | None = None) -> int:
    """Module-level output+grad parity for the angle-fused path.

    ``collect``: see ``_run_forward_parity`` -- same optional append-only
    parity-conformance-artifact seam, same ``None``-default no-op contract.
    Each mixer case appends TWO rows (this runner checks both output and
    grad parity per case, unlike the single-direction sweeps above): one
    ``direction="fwd"`` row for the output comparison, one
    ``direction="grad"`` row for the parameter-grad comparison, both keyed
    by the same ``op`` (mixer case name) and ``shape``.

    Builds each rotation-family mixer on CUDA and compares its forward output
    and every parameter gradient between the eager path (``MINGRU_SCAN=eager``,
    the ``_coeffs`` -> ``matrix_scan``/``matrix_affine_scan`` reference) and the
    angle-fused Triton path (``MINGRU_SCAN=triton``) on identical weights,
    inputs, and cotangents. Covers ``GivensMinGRU`` (decay off, and decay on AT
    THE CLASS DEFAULT ``decay_rate=1.0``) and ``RotationMinGRU`` (snap off/on,
    decay off and on). Both mixers now back their angle-fused reversal with
    EXACT stored-state recompute (no division, no checkpoint interval -- see
    the Task-5 report, "Fix round 2"), so realistic decay strengths are used
    throughout; the earlier "gentle decay" Givens case existed only to dodge
    the since-rejected division-based reversal's roundoff blowup and would be
    misleading to keep now that that dodge is unnecessary.

    A thin wrapper: ``check_mixer`` (below, in the body) toggles
    ``MINGRU_SCAN`` between ``"eager"``/``"triton"`` itself per case, so
    ``_scan_env()`` (no forced mode) only guarantees ``MINGRU_SCAN`` is
    restored to whatever it was before this call once the sweep returns --
    see ``_run_forward_parity``. The actual sweep is
    ``_run_angle_fused_parity_body``.
    """
    with _scan_env():
        return _run_angle_fused_parity_body(collect)


def _run_angle_fused_parity_body(collect: list[dict] | None = None) -> int:
    """The angle-fused parity sweep itself; see ``_run_angle_fused_parity``."""
    from mingru import min_gru

    device = "cuda"
    # Module-output bound: the affine flat fp32 gate (unchanged). Parameter
    # grads: mixed (atol=1e-3, rtol=1e-3), NOT the flat atol=1e-3/rtol=0 bound
    # the affine-op grad sweep uses -- this is the log-op-forward precedent
    # applied to a second physically-unmeetable-at-flat-atol case, not a
    # general loosening. Cause: the no-decay Givens case is norm-preserving
    # (orthogonal), so its theta gradients accumulate unchecked over T and
    # reach |grad| ~ 1704 at this lab shape/seed; a flat 1e-3 ABSOLUTE gate
    # there is below fp32 ULP resolution at that magnitude (~10 ULP), so two
    # numerically-independent-but-both-correct implementations (eager
    # materializes transition matrices; Triton applies factored rotations)
    # cannot agree to it. Measured graze was max_abs=1.98e-3 at
    # rel=1.8e-4 -- comfortably inside the mixed bound. Every other
    # angle-fused case (including the class-default decay=learnable Givens
    # and all RotationMinGRU cases) already passes far inside 1e-3 flat, so
    # this bound still catches a real kernel defect (an O(1) sign/index
    # error, a dropped term) while honoring fp32 physics for the one
    # genuinely unbounded-magnitude gradient. User-ratified 2026-07-16
    # (cloud validation), same precedent as parallel_scan_log's
    # forward gate (see ``_run_forward_parity``).
    out_atol = 1e-5
    grad_atol, grad_rtol = 1e-3, 1e-3
    failures: list[str] = []
    n_pass = 0

    def check_mixer(name: str, mixer_fn, delta_scale: float | None) -> None:
        """Build and check one mixer. ``mixer_fn`` is seeded THEN called, so
        its parameter init is reproducible (unlike passing an already-built
        mixer, whose weights would depend on whatever RNG state preceded this
        call site)."""
        nonlocal n_pass
        torch.manual_seed(0)
        mixer = mixer_fn().to(device)
        B, T = 4, 96
        shape = f"B={B} T={T} input_size={mixer.input_size} hidden_size={mixer.hidden_size}"
        x = torch.randn(B, T, mixer.input_size, device=device)
        dt = None
        if delta_scale is not None:
            dt = torch.rand(B, T, device=device) * delta_scale

        os.environ["MINGRU_SCAN"] = "eager"
        mixer.zero_grad(set_to_none=True)
        out_e = mixer(x, delta_t=dt)
        torch.manual_seed(1234)
        cot = torch.randn_like(out_e)
        (cot * out_e).sum().backward()
        grads_e = {nm: p.grad.detach().clone() for nm, p in mixer.named_parameters()}

        os.environ["MINGRU_SCAN"] = "triton"
        mixer.zero_grad(set_to_none=True)
        try:
            out_t = mixer(x, delta_t=dt)
        except Exception as exc:
            failures.append(f"{name}: fused forward raised {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            os.environ["MINGRU_SCAN"] = "eager"
            if collect is not None:
                collect.append(_parity_row(name, shape, x, "fwd", out_atol, 0.0, None, None, False))
                collect.append(
                    _parity_row(name, shape, x, "grad", grad_atol, grad_rtol, None, None, False)
                )
            return
        (cot * out_t).sum().backward()
        grads_t = {nm: p.grad.detach().clone() for nm, p in mixer.named_parameters()}
        os.environ["MINGRU_SCAN"] = "eager"

        out_err, out_rel = _max_abs_rel(out_t, out_e)
        out_ok = torch.allclose(out_t.float(), out_e.float(), atol=out_atol, rtol=0.0)
        grad_err = 0.0
        grad_rel = 0.0
        grad_ok = True
        for nm in grads_e:
            ge, gt = grads_e[nm], grads_t.get(nm)
            if gt is None:
                grad_ok = False
                failures.append(f"{name}: missing fused grad for {nm}")
                continue
            err, rel = _max_abs_rel(gt, ge)
            grad_err = max(grad_err, err)
            grad_rel = max(grad_rel, rel)
            grad_ok = grad_ok and torch.allclose(
                gt.float(), ge.float(), atol=grad_atol, rtol=grad_rtol
            )
        ok = out_ok and grad_ok
        if ok:
            n_pass += 1
            print(f"  [ok]   {name}: out={out_err:.2e} grad={grad_err:.2e}")
        else:
            failures.append(f"{name}: out={out_err:.2e} grad={grad_err:.2e}")
            print(f"  [FAIL] {name}: out={out_err:.2e} grad={grad_err:.2e}")
        if collect is not None:
            collect.append(
                _parity_row(name, shape, x, "fwd", out_atol, 0.0, out_err, out_rel, out_ok)
            )
            collect.append(
                _parity_row(
                    name, shape, x, "grad", grad_atol, grad_rtol, grad_err, grad_rel, grad_ok
                )
            )

    print("angle-fused GivensMinGRU (k=8, rounds=3, exact stored-state backward):")
    check_mixer(
        "givens decay=None",
        lambda: min_gru.GivensMinGRU(32, 64, block_size=8, rounds=3),
        None,
    )
    check_mixer(
        "givens decay=learnable (class-default decay_rate=1.0)",
        lambda: min_gru.GivensMinGRU(32, 64, block_size=8, rounds=3, decay="learnable"),
        1.0,
    )
    print("angle-fused RotationMinGRU (k=2, tanh-u scale, exact stored-state backward):")
    check_mixer(
        "rotation snap=None decay=None",
        lambda: min_gru.RotationMinGRU(32, 64, snap=None),
        None,
    )
    check_mixer(
        "rotation snap=(2,3,4,6) decay=None",
        lambda: min_gru.RotationMinGRU(32, 64, snap=(2, 3, 4, 6)),
        None,
    )
    check_mixer(
        "rotation snap=(2,3,4,6) decay=learnable",
        lambda: min_gru.RotationMinGRU(
            32, 64, snap=(2, 3, 4, 6), decay="learnable", decay_rate=1.0
        ),
        1.0,
    )

    print()
    if failures:
        print(f"ANGLE-FUSED PARITY FAILED: {len(failures)} case(s), {n_pass} passed")
        for line in failures:
            print(f"  {line}")
        return 1
    print(
        f"ANGLE-FUSED PARITY PASSED: {n_pass} case(s) -- outputs within "
        f"atol={out_atol:.0e}, parameter grads within "
        f"atol={grad_atol:.0e} rtol={grad_rtol:.0e}"
    )
    return 0


# --- DeltaMinGRU chunked-WY parity selftest ----------------------------
#
# The forward/grad-parity conformance evidence for Task 4 (intent ledger
# statement 5): "kernel outputs and gradients match the eager chunked path
# within stated parity tolerances, verified by conformance rows in the
# existing parity harness." Two grids: a raw ``delta_scan_impl``-level
# sweep spanning the full kernel envelope (every ``d_k``, every ``nh``,
# ragged/T=1/T<chunk_size/multi-chunk ``T``), checked against
# ``_run_delta_forward_parity``/``_run_delta_grad_parity``; and a
# module-level end-to-end case (parameters through ``_coeffs``/
# ``out_proj``, ``MINGRU_SCAN=triton`` vs ``eager``) folded into the grad
# sweep, mirroring ``_run_angle_fused_parity``'s ``check_mixer``.
#
# Tolerance-justification rule (spec 9.2/9.3, binding): the gate must be
# justified against the eager path's OWN fp32-vs-fp64 deviation on the same
# inputs (kernel-vs-eager tolerance <= 10x that reference deviation). Both
# runners therefore compute an in-runner fp64 eager reference per case (via
# ``_delta_ref_forward``, dtype-polymorphic) and record both deviations on
# the collected row: ``max_abs_err`` (kernel vs eager fp32, the gated
# value) and ``ref_fp64_dev`` (eager fp32 vs eager fp64, the reference this
# rule scales by) -- see ``_parity_row``.


# Flat floors matching this file's existing affine-op gates (forward
# atol=1e-5, grad atol=1e-3, both rtol=0 -- see ``_run_forward_parity_body``
# / ``_run_grad_parity_body``). ``_delta_gate`` never drops BELOW these; it
# only tightens above them when a case's own eager fp32-vs-fp64 noise is
# larger than the floor implies.
_DELTA_GATE_FLOOR_FWD = 1e-5
_DELTA_GATE_FLOOR_GRAD = 1e-3


def _delta_gate(own_dev: float, floor: float) -> float:
    """Tolerance-justification gate (spec 9.2/9.3).

    ``>= 10x`` the eager path's own fp32-vs-fp64 deviation (``own_dev``) on
    this case's inputs, floored at ``floor`` so a near-zero ``own_dev``
    (e.g. a tiny T=1 case, where fp32 and fp64 barely differ) never yields
    an unreasonably tight gate below the other kernels' flat baseline.
    """
    return max(10.0 * own_dev, floor)


_DELTA_PARITY_DK = tuple(sorted(_DELTA_DK_ENVELOPE))  # (4, 8, 16, 32, 64)
_DELTA_PARITY_NH = (1, 2, 4)


def _delta_chunk_size_for_nh(nh: int) -> int:
    """Largest sweep ``chunk_size`` for ``nh`` keeping ``nh * chunk_size <= 128``.

    ``nh`` in ``{1, 2}`` use the class-default ``64``; ``nh=4`` is halved to
    ``32`` (task brief: "nh=4 requires chunk_size <= 32") -- both land at or
    exactly on ``_DELTA_MAX_MICROSTEPS``, exercising that envelope edge
    rather than staying comfortably inside it.
    """
    return 64 if nh <= 2 else 32


def _delta_T_grid(chunk_size: int) -> tuple[int, int, int, int, int]:
    """T values spanning spec 9.2's forward-parity shape bullets, in order:
    T=1, T < chunk_size, exactly one full chunk, a clean multi-chunk
    sequence, and a ragged multi-chunk sequence.
    """
    partial = max(1, chunk_size // 8)
    return (1, partial, chunk_size, 2 * chunk_size, 2 * chunk_size + chunk_size // 4)


def _delta_inputs(
    B: int, n_heads: int, T: int, nh: int, d_k: int, d_v: int, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random ``(Q, K, V, beta, H0)`` at ``delta_scan_impl``'s spec-section-6 layout.

    ``K`` is L2-normalized (last dim) and ``beta`` drawn as
    ``2 * sigmoid(randn)`` (in ``(0, 2)``) -- the same ranges
    ``DeltaMinGRU._coeffs`` actually produces -- so the UT-transform's
    unit-triangular system stays as well-conditioned as real training
    inputs, not an adversarial worst case. ``Q``/``V`` are scaled down
    (``* 0.3``) and ``H0`` further (``* 0.1``), mirroring this file's
    existing ``_rand_contractive_matrix``/``* 0.1`` scaling convention, so a
    flat absolute tolerance stays meaningful across the whole ``d_k``/``T``
    grid. ``H0`` is always nonzero/randomized here -- ``H0 = 0`` is
    exercised separately by the module-level end-to-end case's default
    ``h_0=None``. Shared by the forward- and gradient-parity raw-tensor
    sweeps.
    """
    Q = torch.randn(B, n_heads, T, d_k, device=device) * 0.3
    K = F.normalize(torch.randn(B, n_heads, T, nh, d_k, device=device), dim=-1)
    V = torch.randn(B, n_heads, T, nh, d_v, device=device) * 0.3
    beta = 2 * torch.sigmoid(torch.randn(B, n_heads, T, nh, device=device))
    H0 = torch.randn(B, n_heads, d_k, d_v, device=device) * 0.1
    return Q, K, V, beta, H0


def _delta_ref_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    beta: torch.Tensor,
    H0: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager chunked-WY reference at ``delta_scan_impl``'s raw tensor layout.

    Delegates to ``DeltaMinGRU._forward_chunked`` -- the certified eager
    oracle, unchanged by this task -- through a duck-typed shim exposing
    only the attributes that method reads (``nh``, ``n_heads``, ``d_k``,
    ``d_v``, ``chunk_size``, ``_coeffs``), rather than re-deriving the
    chunked-WY math a second time in this file: ``_coeffs`` is stubbed to
    hand back ``(Q, K, V, beta)`` un-projected, via the exact inverse of the
    ``stack``/``permute`` assembly ``_dispatch_delta_scan._call`` builds in
    ``min_gru.py`` (``K = torch.stack(ks, dim=2).permute(0, 3, 1, 2, 4)``
    and so on) -- so this reference and the kernel's real dispatch seam
    consume/produce identical layouts. Fully differentiable: the shim's
    ``_coeffs`` closure closes over VIEWS (``.permute``/``.unbind``, never
    copies) of ``Q``/``K``/``V``/``beta``, and ``H0`` is passed straight
    through as the initial state, so gradients flow back to all five
    inputs exactly like the kernel path's ``_DeltaScanFn`` does.
    dtype-polymorphic (whatever dtype the inputs carry) so this one
    function serves as both the fp32 gate reference and the fp64
    tolerance-justification reference (spec 9.2/9.3).

    DUPLICATION-PENDING note: this shim technique exists specifically to
    AVOID a third copy of the chunked-WY forward math (a first copy already
    lives in ``DeltaMinGRU._forward_chunked``, a second -- of the
    backward's recompute half only -- in this file's ``_DeltaScanFn.backward``).
    If a future task needs a plain-function raw-tensor eager reference
    (not routed through a shim), that would be the point to hoist one --
    but it could only live in ``min_gru.py`` (the file that owns the
    oracle), which this task's brief places out of scope ("do not touch
    min_gru.py").

    Parameters mirror ``delta_scan_impl``'s spec-section-6 layout: ``Q``
    ``(B, n_heads, T, d_k)``; ``K`` ``(B, n_heads, T, nh, d_k)``; ``V``
    ``(B, n_heads, T, nh, d_v)``; ``beta`` ``(B, n_heads, T, nh)``; ``H0``
    ``(B, n_heads, d_k, d_v)``.

    Returns
    -------
    tuple of torch.Tensor
        ``(y, H_T)``, exactly ``_forward_chunked``'s return convention:
        ``y`` shape ``(B, T, n_heads, d_v)``, ``H_T`` shape ``(B, n_heads,
        d_k, d_v)``.
    """
    from mingru import min_gru

    B, n_heads, T, d_k = Q.shape
    nh = K.shape[3]
    d_v = V.shape[4]
    q = Q.permute(0, 2, 1, 3)  # (B, T, n_heads, d_k)
    ks = list(K.permute(0, 2, 3, 1, 4).unbind(2))  # nh x (B, T, n_heads, d_k)
    vs = list(V.permute(0, 2, 3, 1, 4).unbind(2))  # nh x (B, T, n_heads, d_v)
    betas = list(beta.permute(0, 2, 3, 1).unbind(2))  # nh x (B, T, n_heads)
    shim = types.SimpleNamespace(
        nh=nh,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
        _coeffs=lambda x: (q, ks, vs, betas),
    )
    x_dummy = Q.new_empty(B, T, 1)
    return min_gru.DeltaMinGRU._forward_chunked(shim, x_dummy, H0)


def _run_delta_forward_parity(collect: list[dict] | None = None) -> int:
    """Run the DeltaMinGRU forward-parity matrix (``y``, ``H_T``); return exit code.

    ``collect``: see ``_run_forward_parity`` -- same optional append-only
    parity-conformance-artifact seam, same ``None``-default no-op contract.

    ``delta_scan_impl`` is called directly (not through ``DeltaMinGRU``'s
    ``MINGRU_SCAN`` module dispatch), so unlike the four-scan-op sweeps
    this runner does not actually need ``MINGRU_SCAN`` forced to
    ``"eager"`` for correctness -- ``_scan_env("eager")`` is used anyway,
    purely so this runner still restores ``MINGRU_SCAN`` on exit exactly
    like every sibling runner in this file (``--check`` runs all of them in
    one process; see ``_scan_env``'s docstring). The actual sweep is
    ``_run_delta_forward_parity_body``.
    """
    with _scan_env("eager"):
        return _run_delta_forward_parity_body(collect)


def _run_delta_forward_parity_body(collect: list[dict] | None = None) -> int:
    """The DeltaMinGRU forward-parity sweep; see ``_run_delta_forward_parity``.

    Grid: every envelope ``d_k`` (``== d_v``, spec 9.2), ``nh`` in
    ``{1, 2, 4}`` (``chunk_size`` chosen per ``_delta_chunk_size_for_nh`` so
    ``nh * chunk_size <= 128`` always holds -- ``nh=4`` lands exactly on
    that bound), and T spanning ``_delta_T_grid``'s five shapes (T=1, T <
    chunk_size, one full chunk, a clean and a ragged multi-chunk sequence).
    ``H0`` is always randomized/nonzero (see ``_delta_inputs``).

    Each case computes THREE forward passes on identical inputs: the
    Triton kernel (fp32, via ``delta_scan_impl``), the eager reference
    (fp32), and the eager reference again in fp64 -- both eager passes via
    ``_delta_ref_forward``, so they can never independently drift from each
    other -- the tolerance-justification rule (spec 9.2). The gate is
    ``_delta_gate(own_dev, 1e-5)`` where ``own_dev`` is the eager path's
    own fp32-vs-fp64 deviation on that case's inputs; both deviations are
    recorded on the collected row (``max_abs_err`` = kernel-vs-eager,
    ``ref_fp64_dev`` = eager's own fp32-vs-fp64).
    """
    device = "cuda"
    torch.manual_seed(0)
    B, n_heads = 2, 2

    failures: list[str] = []
    n_pass = 0

    def check(tag: str, d_k: int, nh: int, chunk_size: int, T: int) -> None:
        nonlocal n_pass
        label = f"delta_forward {tag}"
        Q, K, V, beta, H0 = _delta_inputs(B, n_heads, T, nh, d_k, d_k, device)
        try:
            y_t, HT_t = delta_scan_impl(Q, K, V, beta, H0, chunk_size=chunk_size)
        except Exception as exc:  # ScanFallback or a kernel launch/compile error
            failures.append(f"{label}: Triton path raised {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
            if collect is not None:
                collect.append(
                    _parity_row(
                        "delta_forward",
                        tag,
                        Q,
                        "fwd",
                        _DELTA_GATE_FLOOR_FWD,
                        0.0,
                        None,
                        None,
                        False,
                    )
                )
            return
        # delta_scan_impl returns y as (B, n_heads, T, d_v) (spec section 6);
        # _delta_ref_forward returns y as (B, T, n_heads, d_v)
        # (_forward_chunked's own convention). Restore the kernel's y to that
        # convention here -- exactly the `y.permute(0, 2, 1, 3)` the real
        # module seam applies in `_dispatch_delta_scan._call` (min_gru.py) --
        # BEFORE any comparison, so this sweep compares like-for-like instead
        # of silently transposing the n_heads/T axes against each other.
        y_t = y_t.permute(0, 2, 1, 3)  # (B, T, n_heads, d_v)

        y_e, HT_e = _delta_ref_forward(Q, K, V, beta, H0, chunk_size)
        assert y_t.shape == y_e.shape, (
            f"{label}: y layout mismatch after restore -- kernel {tuple(y_t.shape)} vs "
            f"eager {tuple(y_e.shape)} (B, T, n_heads, d_v expected for both); a future "
            "delta_scan_impl/_forward_chunked layout change broke this sweep's assumption"
        )
        Q64, K64, V64, beta64, H064 = (t.double() for t in (Q, K, V, beta, H0))
        y_e64, HT_e64 = _delta_ref_forward(Q64, K64, V64, beta64, H064, chunk_size)

        kernel_abs_y, kernel_rel_y = _max_abs_rel(y_t, y_e)
        kernel_abs_H, kernel_rel_H = _max_abs_rel(HT_t, HT_e)
        kernel_abs = max(kernel_abs_y, kernel_abs_H)
        kernel_rel = max(kernel_rel_y, kernel_rel_H)

        own_abs_y, _ = _max_abs_rel(y_e, y_e64)
        own_abs_H, _ = _max_abs_rel(HT_e, HT_e64)
        own_dev = max(own_abs_y, own_abs_H)

        gate = _delta_gate(own_dev, _DELTA_GATE_FLOOR_FWD)
        ok = kernel_abs <= gate
        if ok:
            n_pass += 1
        else:
            failures.append(
                f"{label}: max_abs={kernel_abs:.2e} gate={gate:.2e} own_dev={own_dev:.2e}"
            )
            print(
                f"  [FAIL] {label}: max_abs={kernel_abs:.2e} gate={gate:.2e} own_dev={own_dev:.2e}"
            )
        if collect is not None:
            collect.append(
                _parity_row(
                    "delta_forward",
                    tag,
                    Q,
                    "fwd",
                    gate,
                    0.0,
                    kernel_abs,
                    kernel_rel,
                    ok,
                    ref_fp64_dev=own_dev,
                )
            )

    print(
        "delta_scan_impl forward parity (y, H_T; gate = max(10x eager "
        f"fp32-vs-fp64 dev, {_DELTA_GATE_FLOOR_FWD:.0e})):"
    )
    for d_k in _DELTA_PARITY_DK:
        for nh in _DELTA_PARITY_NH:
            chunk_size = _delta_chunk_size_for_nh(nh)
            for T in _delta_T_grid(chunk_size):
                check(f"d_k={d_k} nh={nh} chunk_size={chunk_size} T={T}", d_k, nh, chunk_size, T)

    print()
    if failures:
        print(f"DELTA FORWARD PARITY FAILED: {len(failures)} case(s), {n_pass} passed")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"DELTA FORWARD PARITY PASSED: {n_pass} case(s)")
    return 0


def _run_delta_grad_parity(collect: list[dict] | None = None) -> int:
    """Run the DeltaMinGRU gradient-parity matrix; return process exit code.

    ``collect``: see ``_run_forward_parity`` -- same optional append-only
    parity-conformance-artifact seam, same ``None``-default no-op contract.

    Covers TWO things (task brief): raw ``Q``/``K``/``V``/``beta``/``H0``
    grads driven directly through ``delta_scan_impl`` (no ``MINGRU_SCAN``
    involvement, same rationale as ``_run_delta_forward_parity``), and an
    end-to-end module-level case that DOES toggle ``MINGRU_SCAN`` between
    ``"eager"``/``"triton"`` itself (mirroring ``_run_angle_fused_parity``'s
    ``check_mixer``) -- so ``_scan_env()`` (no forced mode) only guarantees
    ``MINGRU_SCAN`` is restored once this call returns, exactly like
    ``_run_angle_fused_parity``. The actual sweep is
    ``_run_delta_grad_parity_body``.
    """
    with _scan_env():
        return _run_delta_grad_parity_body(collect)


def _run_delta_grad_parity_body(collect: list[dict] | None = None) -> int:
    """The DeltaMinGRU gradient-parity sweep; see ``_run_delta_grad_parity``.

    Both ``check_raw`` and ``check_module`` run their triton-path backward
    under ``warnings.catch_warnings(record=True)`` and fail the case if a
    ``_DeltaBackwardFallbackWarning`` fired -- i.e. this suite proves the
    fused Kernel 7 backward trio actually EXECUTED, not merely that the
    (numerically exact) torch fallback would have produced the right grads
    if the fused kernels had silently never run.
    """
    from mingru import min_gru

    device = "cuda"
    torch.manual_seed(0)
    B, n_heads = 2, 2

    failures: list[str] = []
    n_pass = 0

    def check_raw(tag: str, d_k: int, nh: int, chunk_size: int, T: int) -> None:
        """Compare kernel-path vs eager Q/K/V/beta/H0 grads for one raw case.

        Same three-pass structure as ``_run_delta_forward_parity_body``'s
        ``check`` (kernel fp32, eager fp32, eager fp64), but comparing the
        INPUT grads of a scalar loss ``sum(cot_y * y) + sum(cot_H * H_T)``
        (identical cotangents across all three passes) instead of the
        outputs themselves.
        """
        nonlocal n_pass
        label = f"delta_grad {tag}"
        base = _delta_inputs(B, n_heads, T, nh, d_k, d_k, device)
        tri_in = [t.detach().clone().requires_grad_(True) for t in base]
        eager_in = [t.detach().clone().requires_grad_(True) for t in base]
        eager64_in = [t.detach().clone().double().requires_grad_(True) for t in base]

        try:
            y_t, HT_t = delta_scan_impl(*tri_in, chunk_size=chunk_size)
        except Exception as exc:  # ScanFallback or a kernel launch/compile error
            failures.append(f"{label}: Triton path raised {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
            if collect is not None:
                collect.append(
                    _parity_row(
                        "delta_grad",
                        tag,
                        base[0],
                        "grad",
                        _DELTA_GATE_FLOOR_GRAD,
                        0.0,
                        None,
                        None,
                        False,
                    )
                )
            return
        # Same layout restore as _run_delta_forward_parity_body's `check` --
        # delta_scan_impl's y is (B, n_heads, T, d_v); _delta_ref_forward's
        # (via _forward_chunked) is (B, T, n_heads, d_v). Must happen before
        # `loss_t` is built below, or the cotangent (shaped from y_e) would
        # silently broadcast against the wrong axes instead of raising.
        y_t = y_t.permute(0, 2, 1, 3)  # (B, T, n_heads, d_v)

        y_e, HT_e = _delta_ref_forward(*eager_in, chunk_size)
        assert y_t.shape == y_e.shape, (
            f"{label}: y layout mismatch after restore -- kernel {tuple(y_t.shape)} vs "
            f"eager {tuple(y_e.shape)} (B, T, n_heads, d_v expected for both); a future "
            "delta_scan_impl/_forward_chunked layout change broke this sweep's assumption"
        )
        y_e64, HT_e64 = _delta_ref_forward(*eager64_in, chunk_size)

        torch.manual_seed(1234)
        cot_y = torch.randn_like(y_e)
        cot_H = torch.randn_like(HT_e)
        loss_t = (cot_y * y_t).sum() + (cot_H * HT_t).sum()
        # Record warnings raised during the triton-path backward specifically,
        # so a fused-kernel-launch fallback (`_DeltaBackwardFallbackWarning`)
        # is caught HERE as a parity failure -- not merely inferred from
        # coincidentally-correct grads. `_delta_backward_torch` is
        # numerically exact, so a silent fallback would let this suite stay
        # green forever without the fused Kernel 7 trio ever having executed,
        # defeating the round's speed goal.
        with warnings.catch_warnings(record=True) as caught_t:
            warnings.simplefilter("always")
            loss_t.backward()
        fallback_msgs = [
            str(w.message)
            for w in caught_t
            if issubclass(w.category, _DeltaBackwardFallbackWarning)
        ]
        loss_e = (cot_y * y_e).sum() + (cot_H * HT_e).sum()
        loss_e.backward()
        loss_e64 = (cot_y.double() * y_e64).sum() + (cot_H.double() * HT_e64).sum()
        loss_e64.backward()

        max_err = 0.0
        max_rel = 0.0
        own_dev = 0.0
        ok = True
        for ti, ei, ei64 in zip(tri_in, eager_in, eager64_in):
            gt, ge, ge64 = ti.grad, ei.grad, ei64.grad
            if gt is None or ge is None or ge64 is None:
                ok = False
                failures.append(
                    f"{label}: grad presence mismatch (triton={gt is not None}, "
                    f"eager={ge is not None}, eager64={ge64 is not None})"
                )
                continue
            ka, kr = _max_abs_rel(gt, ge)
            oa, _ = _max_abs_rel(ge, ge64)
            max_err = max(max_err, ka)
            max_rel = max(max_rel, kr)
            own_dev = max(own_dev, oa)

        gate = _delta_gate(own_dev, _DELTA_GATE_FLOOR_GRAD)
        if fallback_msgs:
            ok = False
            failures.append(f"{label}: fused backward did not engage: {fallback_msgs[0]}")
            print(f"  [FAIL] {label}: fused backward did not engage: {fallback_msgs[0]}")
        ok = ok and max_err <= gate
        if ok:
            n_pass += 1
        else:
            failures.append(
                f"{label}: max_grad_abs={max_err:.2e} gate={gate:.2e} own_dev={own_dev:.2e}"
            )
            print(
                f"  [FAIL] {label}: max_grad_abs={max_err:.2e} gate={gate:.2e} "
                f"own_dev={own_dev:.2e}"
            )
        if collect is not None:
            collect.append(
                _parity_row(
                    "delta_grad",
                    tag,
                    base[0],
                    "grad",
                    gate,
                    0.0,
                    max_err,
                    max_rel,
                    ok,
                    ref_fp64_dev=own_dev,
                )
            )

    print(
        "delta_scan_impl grad parity (Q/K/V/beta/H0; gate = max(10x eager "
        f"fp32-vs-fp64 dev, {_DELTA_GATE_FLOOR_GRAD:.0e})):"
    )
    for d_k in _DELTA_PARITY_DK:
        for nh in _DELTA_PARITY_NH:
            chunk_size = _delta_chunk_size_for_nh(nh)
            for T in _delta_T_grid(chunk_size):
                check_raw(
                    f"d_k={d_k} nh={nh} chunk_size={chunk_size} T={T}", d_k, nh, chunk_size, T
                )

    print()
    print(
        "DeltaMinGRU module end-to-end (params through _coeffs/out_proj, "
        "MINGRU_SCAN=triton vs eager):"
    )

    def check_module(name: str, mixer_fn) -> None:
        """Build and check one DeltaMinGRU config; mirrors ``check_mixer`` in
        ``_run_angle_fused_parity_body``, plus the fp64 tolerance-justification
        pass (``mixer64``, a deep-copied double-precision clone run only under
        ``MINGRU_SCAN=eager`` -- the kernel is fp32-only, so it is never asked
        to run in fp64). Exercises the real, non-contiguous module seam
        ``_dispatch_delta_scan._call`` builds (permuted views straight into
        ``delta_scan_impl``, whose launch calls ``.contiguous()``)."""
        nonlocal n_pass
        torch.manual_seed(0)
        mixer = mixer_fn().to(device)
        B_m = 3
        T_m = 2 * mixer.chunk_size + max(1, mixer.chunk_size // 3)
        shape = (
            f"B={B_m} T={T_m} n_heads={mixer.n_heads} nh={mixer.nh} "
            f"d_k={mixer.d_k} chunk_size={mixer.chunk_size}"
        )
        x = torch.randn(B_m, T_m, mixer.input_size, device=device)

        os.environ["MINGRU_SCAN"] = "eager"
        mixer.zero_grad(set_to_none=True)
        out_e = mixer(x)
        torch.manual_seed(1234)
        cot = torch.randn_like(out_e)
        (cot * out_e).sum().backward()
        grads_e = {nm: p.grad.detach().clone() for nm, p in mixer.named_parameters()}

        mixer64 = copy.deepcopy(mixer).double()
        mixer64.zero_grad(set_to_none=True)
        out_e64 = mixer64(x.double())
        (cot.double() * out_e64).sum().backward()
        grads_e64 = {nm: p.grad.detach().clone() for nm, p in mixer64.named_parameters()}

        os.environ["MINGRU_SCAN"] = "triton"
        mixer.zero_grad(set_to_none=True)
        try:
            out_t = mixer(x)
        except Exception as exc:
            failures.append(f"{name}: fused forward raised {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
            os.environ["MINGRU_SCAN"] = "eager"
            if collect is not None:
                collect.append(
                    _parity_row(
                        name, shape, x, "fwd", _DELTA_GATE_FLOOR_FWD, 0.0, None, None, False
                    )
                )
                collect.append(
                    _parity_row(
                        name, shape, x, "grad", _DELTA_GATE_FLOOR_GRAD, 0.0, None, None, False
                    )
                )
            return
        # Same discriminating-fallback rationale as check_raw above: record
        # warnings around the triton-path backward and treat
        # `_DeltaBackwardFallbackWarning` as a parity failure, not just
        # informational noise -- otherwise a silently-falling-back fused
        # backward would leave this end-to-end module case green forever.
        with warnings.catch_warnings(record=True) as caught_t:
            warnings.simplefilter("always")
            (cot * out_t).sum().backward()
        fallback_msgs = [
            str(w.message)
            for w in caught_t
            if issubclass(w.category, _DeltaBackwardFallbackWarning)
        ]
        grads_t = {nm: p.grad.detach().clone() for nm, p in mixer.named_parameters()}
        os.environ["MINGRU_SCAN"] = "eager"

        out_abs, out_rel = _max_abs_rel(out_t, out_e)
        out_own, _ = _max_abs_rel(out_e, out_e64)
        out_gate = _delta_gate(out_own, _DELTA_GATE_FLOOR_FWD)
        out_ok = out_abs <= out_gate

        grad_abs = 0.0
        grad_rel = 0.0
        grad_own = 0.0
        grad_ok = True
        for nm in grads_e:
            gt = grads_t.get(nm)
            if gt is None:
                grad_ok = False
                failures.append(f"{name}: missing fused grad for {nm}")
                continue
            a, r = _max_abs_rel(gt, grads_e[nm])
            oa, _ = _max_abs_rel(grads_e[nm], grads_e64[nm])
            grad_abs = max(grad_abs, a)
            grad_rel = max(grad_rel, r)
            grad_own = max(grad_own, oa)
        grad_gate = _delta_gate(grad_own, _DELTA_GATE_FLOOR_GRAD)
        grad_ok = grad_ok and grad_abs <= grad_gate

        if fallback_msgs:
            grad_ok = False
            failures.append(f"{name}: fused backward did not engage: {fallback_msgs[0]}")
            print(f"  [FAIL] {name}: fused backward did not engage: {fallback_msgs[0]}")

        ok = out_ok and grad_ok
        if ok:
            n_pass += 1
            print(f"  [ok]   {name}: out={out_abs:.2e} grad={grad_abs:.2e}")
        else:
            failures.append(f"{name}: out={out_abs:.2e} grad={grad_abs:.2e}")
            print(f"  [FAIL] {name}: out={out_abs:.2e} grad={grad_abs:.2e}")
        if collect is not None:
            collect.append(
                _parity_row(
                    name,
                    shape,
                    x,
                    "fwd",
                    out_gate,
                    0.0,
                    out_abs,
                    out_rel,
                    out_ok,
                    ref_fp64_dev=out_own,
                )
            )
            collect.append(
                _parity_row(
                    name,
                    shape,
                    x,
                    "grad",
                    grad_gate,
                    0.0,
                    grad_abs,
                    grad_rel,
                    grad_ok,
                    ref_fp64_dev=grad_own,
                )
            )

    check_module(
        "delta module nh=1 (DeltaNet)",
        lambda: min_gru.DeltaMinGRU(32, 128, n_heads=4, nh=1, chunk_size=16),
    )
    check_module(
        "delta module nh=2 (DeltaProduct) explicit d_k/d_v",
        lambda: min_gru.DeltaMinGRU(32, 64, n_heads=2, nh=2, d_k=16, d_v=16, chunk_size=8),
    )
    check_module(
        "delta module nh=4 chunk boundary (nh*chunk_size=128)",
        lambda: min_gru.DeltaMinGRU(32, 64, n_heads=2, nh=4, d_k=8, d_v=8, chunk_size=32),
    )

    print()
    if failures:
        print(f"DELTA GRAD PARITY FAILED: {len(failures)} case(s), {n_pass} passed")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"DELTA GRAD PARITY PASSED: {n_pass} case(s)")
    return 0
