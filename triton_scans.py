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

Task 2 (this module): the generic affine-scan FORWARD core (spec §5,
Kernel 1) plus wrappers mapping three of the four scan ops onto it --
`linear_scan` (k=1, channel-tiled), `matrix_scan` (k=2, v=1), and
`matrix_affine_scan` (generic k, v). Kernels accumulate in fp32
regardless of input dtype and are registered via
`torch.library.triton_op` so `torch.compile` sees them without graph
breaks. Backward (Task 4), the fused log-space scan for
`parallel_scan_log` (Task 3), and the angle-fused Givens path (Phase 2)
land later; `parallel_scan_log` therefore has no entry in `SCAN_IMPLS`
yet and falls back to eager under `auto`.

Envelope (spec §4): k, v in {1, 2, 4, 8, 16}, any T >= 1 including
non-power-of-two. Out-of-envelope shapes raise `ScanFallback`, which
`min_gru._dispatch_scan` turns into a loud eager fallback under `auto`
(and a raised error under `triton`) -- never a wrong result, never a
crash.
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

# Channel-tile width for the k=1 (`linear_scan`) kernel: each program walks
# the full T sequence for BLOCK_D channels at once, keeping lanes wide
# (spec §5). A safe default; the cloud benchmark phase may autotune it.
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
    from torch.library import triton_op, wrap_triton

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

    def _linear_scan_impl(
        a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SCAN_IMPLS entry for ``linear_scan`` (k = v = 1, always in envelope)."""
        try:
            return linear_scan_fwd(a, b)
        except Exception as exc:  # kernel launch/compile failure -> loud fallback
            raise ScanFallback(f"linear_scan Triton kernel failed: {exc}") from exc

    def _matrix_scan_impl(
        M: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SCAN_IMPLS entry for ``matrix_scan`` (k = 2, v = 1, fixed in envelope).

        Maps onto the generic kernel by viewing the ``(B, T, n, 2)``
        injection ``b`` as a ``(B, T, n, 2, 1)`` matrix, then squeezing the
        trailing ``v = 1`` axis off the returned ``Bbar``.
        """
        Bm = b.unsqueeze(-1)
        try:
            Abar, Bbar = affine_scan_fwd(M, Bm)
        except Exception as exc:  # kernel launch/compile failure -> loud fallback
            raise ScanFallback(f"matrix_scan Triton kernel failed: {exc}") from exc
        return Abar, Bbar.squeeze(-1)

    def _matrix_affine_scan_impl(
        A: torch.Tensor, Bm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SCAN_IMPLS entry for ``matrix_affine_scan`` (generic k, v).

        Enforces the kernel envelope up front: an unsupported ``k``/``v``
        (or a non-square transition) raises ``ScanFallback`` so the caller
        loudly falls back to eager rather than launching a kernel the tile
        shapes cannot represent.
        """
        k = A.shape[-1]
        v = Bm.shape[-1]
        if k not in _ENVELOPE or v not in _ENVELOPE or A.shape[-2] != k:
            raise ScanFallback(
                f"matrix_affine_scan (k={k}, v={v}, rows={A.shape[-2]}) outside "
                f"the Triton envelope k,v in {sorted(_ENVELOPE)} with square A"
            )
        try:
            return affine_scan_fwd(A, Bm)
        except Exception as exc:  # kernel launch/compile failure -> loud fallback
            raise ScanFallback(
                f"matrix_affine_scan Triton kernel failed: {exc}"
            ) from exc

    SCAN_IMPLS = {
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
    Ts = (13, 64, 128, 1024)
    Bs = (2, 128)

    failures: list[str] = []
    n_pass = 0

    def check(
        name: str,
        tag: str,
        *inputs: torch.Tensor,
        informational: bool = False,
    ) -> None:
        """Run the Triton impl and eager reference on ``inputs`` and compare.

        The Triton call is guarded so a ``ScanFallback`` or launch/compile
        error is reported as a per-case failure (loud, but not aborting the
        whole sweep) rather than a bare traceback.
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
        out_a, ref_a = out[0], ref[0]
        out_b, ref_b = out[1], ref[1]
        abs_a, rel_a = _max_abs_rel(out_a, ref_a)
        abs_b, rel_b = _max_abs_rel(out_b, ref_b)
        abs_err = max(abs_a, abs_b)
        ok = torch.allclose(
            out_a.float(), ref_a.float(), atol=atol, rtol=rtol
        ) and torch.allclose(out_b.float(), ref_b.float(), atol=atol, rtol=rtol)
        if informational:
            print(f"  [info] {label}: max_abs={abs_err:.2e} (not gated)")
            return
        if ok:
            n_pass += 1
        else:
            failures.append(
                f"{label}: max_abs={abs_err:.2e} "
                f"(A rel={rel_a:.2e}, B rel={rel_b:.2e})"
            )
            print(f"  [FAIL] {label}: max_abs={abs_err:.2e}")

    print("linear_scan (k=1):")
    for B in Bs:
        for T in Ts:
            D = 64
            a = torch.rand(B, T, D, device=device) * 1.8 - 0.9
            b = torch.randn(B, T, D, device=device) * 0.1
            check("linear_scan", f"B={B} T={T} D={D}", a, b)

    print("matrix_scan (k=2, v=1):")
    for B in Bs:
        for T in Ts:
            n = 4
            M = _rand_contractive_matrix((B, T, n, 2, 2), 2, device)
            b = torch.randn(B, T, n, 2, device=device) * 0.1
            check("matrix_scan", f"B={B} T={T} n={n}", M, b)

    print("matrix_affine_scan (generic k, v):")
    for B in Bs:
        for T in Ts:
            n = 1
            for k in sorted(_ENVELOPE):
                for v in sorted(_ENVELOPE):
                    A = _rand_contractive_matrix((B, T, n, k, k), k, device)
                    Bm = torch.randn(B, T, n, k, v, device=device) * 0.1
                    check("matrix_affine_scan", f"B={B} T={T} k={k} v={v}", A, Bm)

    print("matrix_affine_scan (n>1 lane-grid exercise):")
    for B in Bs:
        for T in Ts:
            n = 3
            k, v = 8, 4
            A = _rand_contractive_matrix((B, T, n, k, k), k, device)
            Bm = torch.randn(B, T, n, k, v, device=device) * 0.1
            check("matrix_affine_scan", f"B={B} T={T} n={n} k={k} v={v}", A, Bm)

    print("bf16-input rows (informational, spec §9.1):")
    B, T = 2, 64
    a = (torch.rand(B, T, 64, device=device) * 1.8 - 0.9).bfloat16()
    b = (torch.randn(B, T, 64, device=device) * 0.1).bfloat16()
    check("linear_scan", "bf16 B=2 T=64 D=64", a, b, informational=True)
    M = _rand_contractive_matrix((B, T, 2, 2, 2), 2, device).bfloat16()
    b2 = (torch.randn(B, T, 2, 2, device=device) * 0.1).bfloat16()
    check("matrix_scan", "bf16 B=2 T=64 n=2", M, b2, informational=True)
    A = _rand_contractive_matrix((B, T, 1, 8, 8), 8, device).bfloat16()
    Bm = (torch.randn(B, T, 1, 8, 4, device=device) * 0.1).bfloat16()
    check("matrix_affine_scan", "bf16 B=2 T=64 k=8 v=4", A, Bm, informational=True)

    print()
    if failures:
        print(f"FORWARD PARITY FAILED: {len(failures)} case(s), {n_pass} passed")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"FORWARD PARITY PASSED: {n_pass} fp32 case(s) within "
          f"atol={atol:.0e} rtol={rtol:.0e}")
    return 0


if __name__ == "__main__":
    _status = available()
    if _status is not True:
        print(f"triton_scans selftest SKIPPED (loud): {_status}")
        raise SystemExit(0)
    raise SystemExit(_run_forward_parity())
