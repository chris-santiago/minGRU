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

Tasks 2-3 (this module): the generic affine-scan FORWARD core (spec §5,
Kernel 1) plus wrappers mapping three of the four scan ops onto it --
`linear_scan` (k=1, channel-tiled), `matrix_scan` (k=2, v=1), and
`matrix_affine_scan` (generic k, v) -- and the fused log-space FORWARD
scan (spec §5, Kernel 2) for `parallel_scan_log` (channel-tiled lanes
carrying a running log-coefficient cumsum plus an online max-shifted
log-sum-exp accumulator, writing h = exp(.) directly). All four scan
ops therefore have an entry in `SCAN_IMPLS`. Kernels accumulate in fp32
regardless of input dtype and are registered via
`torch.library.triton_op` so `torch.compile` sees them without graph
breaks.

Task 4 (this module): BACKWARD, making every Triton path trainable
(spec §5 Kernel 3, §6 adjoint recurrences). The generic core's adjoint
is one reverse-direction scan (`_affine_scan_bwd_kernel`) that reads
ONLY the forward's inputs and outputs -- it reverse-scans the incoming
output grads with A_{t+1}^T, then forms dL/dA_t from the forward outputs
Abar_{t-1}/Bbar_{t-1} (seeded with I/0 at t=1) and dL/dB_t directly.
That zero-extra-saved-tensors property is the point of the reversed-scan
design. `linear_scan` gets the channel-tiled k=1 specialization
(`_linear_scan_bwd_kernel`); `matrix_scan`/`matrix_affine_scan` share
the generic kernel through the same unsqueeze/`affine_scan_fwd` seam as
the forward. `parallel_scan_log`'s backward is autograd-through-
recomputation: it saves only its two forward INPUTS (log_coeffs,
log_values) and re-derives the grad through the eager log-space math
(exact-to-eager, no hand-derived log-space adjoint kernel to get wrong
blind) -- also zero extra saved tensors beyond the forward inputs. All
four ops are wired via `torch.library.register_autograd` on their
forward `triton_op` (the torch 2.8-idiomatic route for a `triton_op`:
the backward composes with `torch.compile`'s tracing and dispatcher,
which an `autograd.Function` wrapper around the op would not). The
angle-fused Givens path (Phase 2) lands later.

Envelope (spec §4): k, v in {1, 2, 4, 8, 16}, any T >= 1 including
non-power-of-two. Out-of-envelope shapes raise `ScanFallback`, which
`min_gru._dispatch_scan` turns into a loud eager fallback under `auto`
(and a raised error under `triton`) -- never a wrong result, never a
crash.
"""

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


# Kernel envelope for the state block size ``k`` and injection width ``v``
# (spec §4). All members are powers of two, so the kernels tile ``k``/``v``
# exactly with no masking; a shape outside this set raises ``ScanFallback``.
_ENVELOPE = frozenset({1, 2, 4, 8, 16})

# Channel-tile width for the two elementwise-in-D kernels (`linear_scan`
# and `parallel_scan_log`): each program walks the full T sequence for
# BLOCK_D channels at once, keeping lanes wide (spec §5). A safe default;
# the cloud benchmark phase may autotune it.
_LINEAR_BLOCK_D = 128


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

            # Composition order A_current @ A_earlier (spec §6):
            #   Abar_t[i, j] = sum_p A_t[i, p] * Abar_{t-1}[p, j]
            #   Bbar_t[i, j] = sum_p A_t[i, p] * Bbar_{t-1}[p, j] + B_t[i, j]
            Abar = tl.sum(a[:, :, None] * Abar[None, :, :], axis=1)
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
        """Channel-tiled fused log-space scan (spec §5, Kernel 2).

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
        """Reverse-direction adjoint of the generic affine prefix scan (spec §6).

        The exact backward of ``_affine_scan_fwd_kernel``. For the forward
        recurrence ``Abar_t = A_t @ Abar_{t-1}`` (``Abar_0 = I``) and
        ``Bbar_t = A_t @ Bbar_{t-1} + B_t`` (``Bbar_0 = 0``), with incoming
        output grads ``G^A_t = dL/dAbar_t`` and ``G^B_t = dL/dBbar_t``, the
        adjoint is (spec §6)::

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
            AtGA = tl.sum(A_next[:, :, None] * GhatA[:, None, :], axis=0)  # (k, k)
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
            dA = tl.sum(GhatA[:, None, :] * Abar_prev[None, :, :], axis=2) + tl.sum(
                GhatB[:, None, :] * Bbar_prev[None, :, :], axis=2
            )
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

    @triton_op("mingru_scans::affine_scan_fwd", mutates_args={})
    def affine_scan_fwd(
        A: torch.Tensor, Bm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generic k x k affine prefix scan (spec §5, Kernel 1).

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
    def linear_scan_fwd(
        a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Channel-tiled k=1 affine prefix scan (spec §5, Kernel 1).

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
    def parallel_scan_log_fwd(
        log_coeffs: torch.Tensor, log_values: torch.Tensor
    ) -> torch.Tensor:
        """Fused log-space forward scan (spec §5, Kernel 2).

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
        """Generic k x k affine-scan adjoint (spec §5 Kernel 3, §6).

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
        wrap_triton(_affine_scan_bwd_kernel)[grid](
            A, Abar, Bbar, gAbar, gBbar, dA, dB, t, n, k, v
        )
        return dA, dB

    @triton_op("mingru_scans::linear_scan_bwd", mutates_args={})
    def linear_scan_bwd(
        a: torch.Tensor,
        A: torch.Tensor,
        Bc: torch.Tensor,
        gA: torch.Tensor,
        gBc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Channel-tiled k=1 affine-scan adjoint (spec §6).

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
        wrap_triton(_linear_scan_bwd_kernel)[grid](
            a, A, Bc, gA, gBc, da, db, t, d, _LINEAR_BLOCK_D
        )
        return da, db

    # --- Autograd registration (spec §5 Kernel 3; torch 2.8 route) ----------
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
        A, _Bm = inputs
        Abar, Bbar = output
        # Saves the forward input A and forward outputs Abar/Bbar only --
        # the adjoint needs no other tensor (not even Bm), so this reuses
        # already-materialized activations with zero extra memory.
        ctx.save_for_backward(A, Abar, Bbar)

    def _affine_backward(ctx, grad_Abar, grad_Bbar):
        A, Abar, Bbar = ctx.saved_tensors
        dA, dB = affine_scan_bwd(A, Abar, Bbar, grad_Abar, grad_Bbar)
        return dA, dB

    register_autograd(
        "mingru_scans::affine_scan_fwd",
        _affine_backward,
        setup_context=_affine_setup_context,
    )

    def _linear_setup_context(ctx, inputs, output):
        a, _b = inputs
        A, Bc = output
        ctx.save_for_backward(a, A, Bc)

    def _linear_backward(ctx, grad_A, grad_Bc):
        a, A, Bc = ctx.saved_tensors
        da, db = linear_scan_bwd(a, A, Bc, grad_A, grad_Bc)
        return da, db

    register_autograd(
        "mingru_scans::linear_scan_fwd",
        _linear_backward,
        setup_context=_linear_setup_context,
    )

    def _parallel_scan_log_setup_context(ctx, inputs, output):
        log_coeffs, log_values = inputs
        # Autograd-through-recomputation: the log op has no hand-derived
        # adjoint kernel (nothing to get wrong blind). Saving the two forward
        # INPUTS is enough to re-derive the grad exactly through the eager
        # log-space math -- zero extra saved tensors beyond forward inputs
        # (the forward output h is not even retained).
        ctx.save_for_backward(log_coeffs, log_values)

    def _parallel_scan_log_backward(ctx, grad_h):
        if grad_h is None:
            return None, None
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
            # Byte-for-byte the eager math of min_gru.parallel_scan_log
            # (a_star = pad(cumsum(log_coeffs)); log_h = a_star +
            # logcumsumexp(log_values - a_star); h = exp(log_h)[:, 1:]).
            # Replicated here rather than calling the eager function so the
            # backward neither re-enters the MINGRU_SCAN dispatcher nor
            # depends on min_gru's env state. MAINTENANCE: this formula must
            # be kept byte-identical to min_gru.parallel_scan_log's eager
            # body -- no static check enforces that; only the grad-parity
            # selftest (_run_grad_parity, below) catches divergence.
            a_star = F.pad(torch.cumsum(lc, dim=1), (0, 0, 1, 0))
            log_h = a_star + torch.logcumsumexp(lv - a_star, dim=1)
            h = torch.exp(log_h)[:, 1:]
        dlc, dlv = torch.autograd.grad(h, (lc, lv), grad_outputs=grad_h)
        return dlc, dlv

    register_autograd(
        "mingru_scans::parallel_scan_log_fwd",
        _parallel_scan_log_backward,
        setup_context=_parallel_scan_log_setup_context,
    )

    def _linear_scan_impl(
        a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _parallel_scan_log_impl(
        log_coeffs: torch.Tensor, log_values: torch.Tensor
    ) -> torch.Tensor:
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
            or log_values.shape != (log_coeffs.shape[0], log_coeffs.shape[1] + 1, log_coeffs.shape[2])
        ):
            raise ScanFallback(
                f"parallel_scan_log (log_coeffs={tuple(log_coeffs.shape)}, "
                f"log_values={tuple(log_values.shape)}) outside the Triton "
                "envelope: need (B, T, D) log_coeffs and (B, T+1, D) log_values"
            )
        try:
            return parallel_scan_log_fwd(log_coeffs, log_values)
        except Exception as exc:  # kernel launch/compile failure -> loud fallback
            raise ScanFallback(
                f"parallel_scan_log Triton kernel failed: {exc}"
            ) from exc

    def _matrix_scan_impl(
        M: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            raise ScanFallback(
                f"matrix_affine_scan Triton kernel failed: {exc}"
            ) from exc

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


# --- Forward-parity selftest (spec §9.1, §10) -------------------------------
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
    import min_gru

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


# Shape matrix shared by both parity sweeps (spec §9.1): T=1 exercises a
# program that never traverses the loop-increment path (a single timestep);
# the rest covers non-power-of-two lengths and the long-T end. Defined once
# so ``_run_forward_parity`` and ``_run_grad_parity`` cannot silently diverge.
_PARITY_TS = (1, 13, 64, 128, 1024)
_PARITY_BS = (2, 128)


def _run_forward_parity() -> int:
    """Run the §9.1 forward-parity matrix; return process exit code."""
    import os

    os.environ["MINGRU_SCAN"] = "eager"  # force the eager reference path
    import min_gru

    device = "cuda"
    torch.manual_seed(0)
    # Flat spec bound (design spec §7 / plan constraints): outputs <= 1e-5
    # for fp32. rtol=0 so this is exactly max_abs_err <= atol, not a looser
    # relative gate -- do not reintroduce a nonzero rtol here.
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

        ``case_atol``/``case_rtol`` default to the flat spec bound
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
            ok = ok and torch.allclose(
                out_i.float(), ref_i.float(), atol=case_atol, rtol=case_rtol
            )
        if informational:
            print(f"  [info] {label}: max_abs={abs_err:.2e} (not gated)")
            return
        if ok:
            n_pass += 1
        else:
            rel_str = ", ".join(f"{r:.2e}" for r in rels)
            failures.append(f"{label}: max_abs={abs_err:.2e} (rel={rel_str})")
            print(f"  [FAIL] {label}: max_abs={abs_err:.2e}")

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
    # self-tests use, which spec §7 cites as "matching existing selftest
    # standards" -- so it still catches any real kernel defect (off by
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

    print("bf16-input rows (informational, spec §9.1):")
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
    check(
        "matrix_scan", "bf16 B=2 T=64 n=2", M.bfloat16(), b2.bfloat16(), informational=True
    )
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


def _run_grad_parity() -> int:
    """Run the §9.1 gradient-parity matrix; return process exit code.

    For every op the Triton path is now differentiable (Task 4:
    ``register_autograd`` on each forward ``triton_op``). This sweep drives
    a scalar loss ``sum(cotangent * output)`` through both the Triton path
    and the eager reference on identical inputs and identical cotangents,
    then compares the resulting INPUT gradients. The gate is the plan's
    parameter-grad bound (``atol=1e-3``); it exercises the reversed-scan
    adjoint kernels (affine, k=1) and the log op's recompute backward.
    """
    import os

    os.environ["MINGRU_SCAN"] = "eager"  # force the eager reference path
    import min_gru

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
            err, _ = _max_abs_rel(gt, ge)
            max_err = max(max_err, err)
            ok = ok and torch.allclose(gt.float(), ge.float(), atol=atol, rtol=rtol)
        if ok:
            n_pass += 1
        else:
            failures.append(f"{label}: max_grad_abs={max_err:.2e}")
            print(f"  [FAIL] {label}: max_grad_abs={max_err:.2e}")

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


if __name__ == "__main__":
    _status = available()
    if _status is not True:
        print(f"triton_scans selftest SKIPPED (loud): {_status}")
        raise SystemExit(0)
    _fwd = _run_forward_parity()
    print()
    _bwd = _run_grad_parity()
    raise SystemExit(_fwd or _bwd)
