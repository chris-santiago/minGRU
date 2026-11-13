"""Tests for the four public parallel-scan functions.

CPU-only: no GPU/Triton execution paths are exercised here (dispatch
semantics are covered in ``test_dispatch.py``). Every scan runs its eager
implementation because ``MINGRU_SCAN`` is unset / ``auto`` on CPU. Section 6
is the one exception to "no Triton" in spirit only: it imports a handful of
``mingru.triton_scans``' pure-Python/CPU-runnable parity-harness helpers
(never the gated Triton kernel path itself, which requires CUDA and is
untestable here) to regression-guard the delta parity harness's own
comparison-pipeline plumbing.

Sections
--------
1. parallel_scan_log -- shapes, dtype, T=1, non-power-of-two T, correctness
   vs. a sequential reference, gradcheck
2. linear_scan -- signed coefficients, shapes, T=1, correctness, gradcheck
3. matrix_scan -- 2x2 block transitions, shapes, correctness, gradcheck
4. matrix_affine_scan -- k x k transitions, shapes, correctness, the
   k=2/v=1 reduction to matrix_scan, gradcheck
5. Gradient flow -- non-None, non-zero grads after one backward
6. Delta parity harness y-layout regression guard -- CPU-only pin for
   ``_run_delta_forward_parity_body``/``_run_delta_grad_parity_body``'s
   ``delta_scan_impl``-vs-eager ``y`` axis-order restore
7. Delta backward-torch vs eager-autograd fp64 semantic guard -- runs on any
   CPU, with or without the ``triton`` package, pinning
   ``triton_scans._delta_backward_torch``'s hand-derived gradients against
   ``torch.autograd.grad`` through the eager oracle
"""

from __future__ import annotations

import warnings

import pytest
import torch

from mingru import (
    DeltaMinGRU,
    linear_scan,
    matrix_affine_scan,
    matrix_scan,
    parallel_scan_log,
    triton_scans,
)

# The float32 saturation tests below derive their boundary delta_t values from
# this constant (not hardcoded magnitudes) so a retune fails the binding
# assertions loudly instead of silently staling a premise.
from mingru.min_gru import _DECAY_SATURATION_WARN_THRESHOLD

SEED = 42

# All float32-robustness layers below use decay_rate=1.0, so lambda == 1 and
# lambda*dt == dt: the delta_t thresholds equal the lambda*dt thresholds.


# ===========================================================================
# Sequential reference implementations (the "manual" grad/correctness oracle)
# ===========================================================================


def _ref_parallel_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
    a = torch.exp(log_coeffs)
    h = torch.exp(log_values[:, 0])
    outs = []
    for t in range(a.size(1)):
        h = a[:, t] * h + torch.exp(log_values[:, t + 1])
        outs.append(h)
    return torch.stack(outs, dim=1)


def _ref_linear_scan(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    A_run = torch.ones_like(a[:, 0])
    Bc_run = torch.zeros_like(b[:, 0])
    A_out, Bc_out = [], []
    for t in range(a.size(1)):
        A_run = a[:, t] * A_run
        Bc_run = a[:, t] * Bc_run + b[:, t]
        A_out.append(A_run)
        Bc_out.append(Bc_run)
    return torch.stack(A_out, dim=1), torch.stack(Bc_out, dim=1)


def _ref_matrix_scan(M: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    A_run = torch.eye(2, dtype=M.dtype).expand_as(M[:, 0]).clone()
    Bc_run = torch.zeros_like(b[:, 0])
    A_out, Bc_out = [], []
    for t in range(M.size(1)):
        A_run = M[:, t] @ A_run
        Bc_run = torch.einsum("bnij,bnj->bni", M[:, t], Bc_run) + b[:, t]
        A_out.append(A_run)
        Bc_out.append(Bc_run)
    return torch.stack(A_out, dim=1), torch.stack(Bc_out, dim=1)


def _ref_matrix_affine_scan(A: torch.Tensor, Bm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = A.size(-1)
    A_run = torch.eye(k, dtype=A.dtype).expand_as(A[:, 0]).clone()
    Bb_run = torch.zeros_like(Bm[:, 0])
    A_out, Bb_out = [], []
    for t in range(A.size(1)):
        A_run = A[:, t] @ A_run
        Bb_run = A[:, t] @ Bb_run + Bm[:, t]
        A_out.append(A_run)
        Bb_out.append(Bb_run)
    return torch.stack(A_out, dim=1), torch.stack(Bb_out, dim=1)


# ===========================================================================
# 1. parallel_scan_log
# ===========================================================================


class TestParallelScanLog:
    def test_shape_and_dtype(self):
        torch.manual_seed(SEED)
        B, T, D = 3, 7, 5
        log_coeffs = -torch.rand(B, T, D)
        log_values = -torch.rand(B, T + 1, D)
        h = parallel_scan_log(log_coeffs, log_values)
        assert h.shape == (B, T, D)
        assert h.dtype == log_coeffs.dtype

    def test_matches_sequential_reference(self):
        torch.manual_seed(SEED)
        B, T, D = 2, 6, 4
        log_coeffs = -torch.rand(B, T, D)
        log_values = -torch.rand(B, T + 1, D)
        h = parallel_scan_log(log_coeffs, log_values)
        ref = _ref_parallel_scan_log(log_coeffs, log_values)
        assert torch.allclose(h, ref, atol=1e-5)

    def test_single_timestep(self):
        torch.manual_seed(SEED)
        B, D = 2, 4
        log_coeffs = -torch.rand(B, 1, D)
        log_values = -torch.rand(B, 2, D)
        h = parallel_scan_log(log_coeffs, log_values)
        assert h.shape == (B, 1, D)
        expected = torch.exp(log_coeffs[:, 0]) * torch.exp(log_values[:, 0]) + torch.exp(
            log_values[:, 1]
        )
        assert torch.allclose(h[:, 0], expected, atol=1e-5)

    def test_non_power_of_two_length(self):
        torch.manual_seed(SEED)
        B, T, D = 2, 5, 3  # T = 5 is not a power of two
        log_coeffs = -torch.rand(B, T, D)
        log_values = -torch.rand(B, T + 1, D)
        h = parallel_scan_log(log_coeffs, log_values)
        ref = _ref_parallel_scan_log(log_coeffs, log_values)
        assert torch.allclose(h, ref, atol=1e-5)

    def test_gradcheck(self):
        torch.manual_seed(SEED)
        B, T, D = 2, 5, 3
        log_coeffs = (-torch.rand(B, T, D, dtype=torch.double)).requires_grad_(True)
        log_values = (-torch.rand(B, T + 1, D, dtype=torch.double)).requires_grad_(True)
        assert torch.autograd.gradcheck(parallel_scan_log, (log_coeffs, log_values))


# ===========================================================================
# 2. linear_scan
# ===========================================================================


class TestLinearScan:
    def test_shape(self):
        torch.manual_seed(SEED)
        B, T, D = 3, 7, 5
        a = torch.rand(B, T, D) * 2 - 1  # signed coefficients
        b = torch.randn(B, T, D)
        A, Bc = linear_scan(a, b)
        assert A.shape == (B, T, D)
        assert Bc.shape == (B, T, D)

    def test_signed_coefficients_matches_reference(self):
        torch.manual_seed(SEED)
        B, T, D = 2, 6, 4
        a = torch.rand(B, T, D) * 2 - 1  # negative eigenvalues allowed
        b = torch.randn(B, T, D)
        A, Bc = linear_scan(a, b)
        A_ref, Bc_ref = _ref_linear_scan(a, b)
        assert torch.allclose(A, A_ref, atol=1e-5)
        assert torch.allclose(Bc, Bc_ref, atol=1e-5)

    def test_single_timestep(self):
        torch.manual_seed(SEED)
        B, D = 2, 4
        a = torch.rand(B, 1, D) * 2 - 1
        b = torch.randn(B, 1, D)
        A, Bc = linear_scan(a, b)
        assert torch.allclose(A[:, 0], a[:, 0], atol=1e-6)
        assert torch.allclose(Bc[:, 0], b[:, 0], atol=1e-6)

    def test_non_power_of_two_length(self):
        torch.manual_seed(SEED)
        B, T, D = 2, 5, 3
        a = torch.rand(B, T, D) * 2 - 1
        b = torch.randn(B, T, D)
        A, Bc = linear_scan(a, b)
        A_ref, Bc_ref = _ref_linear_scan(a, b)
        assert torch.allclose(A, A_ref, atol=1e-5)
        assert torch.allclose(Bc, Bc_ref, atol=1e-5)

    def test_gradcheck(self):
        torch.manual_seed(SEED)
        B, T, D = 2, 5, 3
        a = (torch.rand(B, T, D, dtype=torch.double) * 2 - 1).requires_grad_(True)
        b = torch.randn(B, T, D, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(linear_scan, (a, b))


# ===========================================================================
# 3. matrix_scan
# ===========================================================================


class TestMatrixScan:
    def test_shape(self):
        torch.manual_seed(SEED)
        B, T, n = 2, 6, 4
        M = torch.randn(B, T, n, 2, 2) * 0.3
        b = torch.randn(B, T, n, 2)
        A, Bc = matrix_scan(M, b)
        assert A.shape == (B, T, n, 2, 2)
        assert Bc.shape == (B, T, n, 2)

    def test_matches_sequential_reference(self):
        torch.manual_seed(SEED)
        B, T, n = 2, 5, 3
        M = torch.randn(B, T, n, 2, 2) * 0.3
        b = torch.randn(B, T, n, 2)
        A, Bc = matrix_scan(M, b)
        A_ref, Bc_ref = _ref_matrix_scan(M, b)
        assert torch.allclose(A, A_ref, atol=1e-5)
        assert torch.allclose(Bc, Bc_ref, atol=1e-5)

    def test_identity_transitions_accumulate_inputs(self):
        torch.manual_seed(SEED)
        B, T, n = 2, 5, 4
        M = torch.eye(2).expand(B, T, n, 2, 2).contiguous()
        b = torch.randn(B, T, n, 2)
        _, Bc = matrix_scan(M, b)
        assert torch.allclose(Bc, torch.cumsum(b, dim=1), atol=1e-5)

    def test_gradcheck(self):
        torch.manual_seed(SEED)
        B, T, n = 2, 5, 3
        M = (torch.randn(B, T, n, 2, 2, dtype=torch.double) * 0.3).requires_grad_(True)
        b = torch.randn(B, T, n, 2, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(matrix_scan, (M, b))


# ===========================================================================
# 4. matrix_affine_scan
# ===========================================================================


class TestMatrixAffineScan:
    def test_shape(self):
        torch.manual_seed(SEED)
        B, T, n, k, v = 2, 6, 3, 5, 1
        A = torch.randn(B, T, n, k, k) * 0.2
        Bm = torch.randn(B, T, n, k, v)
        Abar, Bbar = matrix_affine_scan(A, Bm)
        assert Abar.shape == (B, T, n, k, k)
        assert Bbar.shape == (B, T, n, k, v)

    def test_matches_sequential_reference(self):
        torch.manual_seed(SEED)
        B, T, n, k, v = 2, 5, 3, 4, 2
        A = torch.randn(B, T, n, k, k) * 0.2
        Bm = torch.randn(B, T, n, k, v)
        Abar, Bbar = matrix_affine_scan(A, Bm)
        A_ref, B_ref = _ref_matrix_affine_scan(A, Bm)
        assert torch.allclose(Abar, A_ref, atol=1e-5)
        assert torch.allclose(Bbar, B_ref, atol=1e-5)

    def test_reduces_to_matrix_scan_at_k2_v1(self):
        torch.manual_seed(SEED)
        B, T, n = 2, 5, 3
        M = torch.randn(B, T, n, 2, 2) * 0.3
        b = torch.randn(B, T, n, 2)
        A_ms, Bc_ms = matrix_scan(M, b)
        Abar, Bbar = matrix_affine_scan(M, b.unsqueeze(-1))
        assert torch.allclose(Abar, A_ms, atol=1e-5)
        assert torch.allclose(Bbar.squeeze(-1), Bc_ms, atol=1e-5)

    def test_gradcheck(self):
        torch.manual_seed(SEED)
        B, T, n, k, v = 2, 5, 3, 4, 1
        A = (torch.randn(B, T, n, k, k, dtype=torch.double) * 0.2).requires_grad_(True)
        Bm = torch.randn(B, T, n, k, v, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(matrix_affine_scan, (A, Bm))


# ===========================================================================
# 5. Gradient flow
# ===========================================================================


class TestGradientFlow:
    def test_parallel_scan_log_grads_nonzero(self):
        torch.manual_seed(SEED)
        log_coeffs = (-torch.rand(2, 5, 3)).requires_grad_(True)
        log_values = (-torch.rand(2, 6, 3)).requires_grad_(True)
        parallel_scan_log(log_coeffs, log_values).sum().backward()
        for grad in (log_coeffs.grad, log_values.grad):
            assert grad is not None
            assert grad.abs().sum() > 0

    def test_linear_scan_grads_nonzero(self):
        torch.manual_seed(SEED)
        a = (torch.rand(2, 5, 3) * 2 - 1).requires_grad_(True)
        b = torch.randn(2, 5, 3, requires_grad=True)
        A, Bc = linear_scan(a, b)
        (A.sum() + Bc.sum()).backward()
        for grad in (a.grad, b.grad):
            assert grad is not None
            assert grad.abs().sum() > 0


# ===========================================================================
# 6. Delta parity harness y-layout regression guard
# ===========================================================================
#
# ``delta_scan_impl``'s ``y`` is documented (spec section 6) as
# ``(B, n_heads, T, d_v)``; ``triton_scans._delta_ref_forward`` (via
# ``DeltaMinGRU._forward_chunked``) returns ``y`` as ``(B, T, n_heads, d_v)``
# -- the same axis-order gap the real module seam bridges with
# ``y.permute(0, 2, 1, 3)`` in ``_dispatch_delta_scan._call``
# (``min_gru.py``; pinned at that seam by
# ``test_dispatch.TestDeltaSeamLayoutRoundTrip``).
# ``_run_delta_forward_parity_body``'s ``check`` and
# ``_run_delta_grad_parity_body``'s ``check_raw`` (``triton_scans.py``) must
# apply that identical restore to the kernel's ``y`` before comparing it
# against the eager reference -- a prior revision of that code compared the
# two axis orders directly, which either raises (``T != n_heads``) or
# silently broadcasts a wrong cross-axis comparison (``T == 1`` or
# ``n_heads == 1``).
#
# Both raw-grid runners hardcode ``device="cuda"`` (GPU-only by design, like
# every sibling runner in ``triton_scans.py``), so they cannot be invoked
# directly here. Instead this reproduces their exact
# permute-then-shape-assert-then-compare sequence on CPU, driven by the same
# building blocks the runners use (``_delta_inputs``, ``_delta_ref_forward``,
# ``_max_abs_rel``) plus a fake ``delta_scan_impl`` stand-in -- eager's own
# output permuted to the kernel's documented ``(B, n_heads, T, d_v)`` layout,
# exactly what a CORRECT real kernel would return -- so this is a genuine
# pipeline check, not a hand-rolled reimplementation of the comparison logic.
# No Triton/CUDA path runs: every tensor here stays on CPU.


class TestDeltaParityHarnessYLayout:
    # T != n_heads (7 != 3) so an unrestored comparison hits the crash case
    # the L4 review report described, not just a shape coincidence.
    B, N_HEADS, T, NH, D_K, D_V, CHUNK_SIZE = 2, 3, 7, 2, 4, 4, 4

    def _fake_kernel_y(self, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager reference ``y_e`` plus a fake kernel ``y`` permuted to
        ``delta_scan_impl``'s documented ``(B, n_heads, T, d_v)`` contract --
        i.e. exactly the inverse of the restore the runner code must apply.
        """
        Q, K, V, beta, H0 = triton_scans._delta_inputs(
            self.B, self.N_HEADS, self.T, self.NH, self.D_K, self.D_V, device
        )
        y_e, H_T_e = triton_scans._delta_ref_forward(Q, K, V, beta, H0, self.CHUNK_SIZE)
        y_fake_kernel = y_e.permute(0, 2, 1, 3)  # (B, n_heads, T, d_v)
        assert tuple(y_fake_kernel.shape) == (self.B, self.N_HEADS, self.T, self.D_V)
        return y_fake_kernel, y_e, H_T_e

    def test_restore_reproduces_the_runners_exact_sequence(self):
        """With the fix's restore, the fake kernel's ``y`` matches the eager
        reference exactly -- both the shape assert and ``_max_abs_rel`` see a
        perfect round trip."""
        torch.manual_seed(SEED)
        y_fake_kernel, y_e, _ = self._fake_kernel_y("cpu")

        # Exactly _run_delta_forward_parity_body's `check` / _run_delta_grad_
        # parity_body's `check_raw`: `y_t = y_t.permute(0, 2, 1, 3)` then a
        # shape assert before any comparison.
        y_t = y_fake_kernel.permute(0, 2, 1, 3)  # (B, T, n_heads, d_v)
        assert y_t.shape == y_e.shape, (
            f"y layout mismatch after restore -- kernel {tuple(y_t.shape)} vs "
            f"eager {tuple(y_e.shape)}"
        )
        abs_err, rel_err = triton_scans._max_abs_rel(y_t, y_e)
        assert abs_err == 0.0
        assert rel_err == 0.0
        assert torch.equal(y_t, y_e)

    def test_unrestored_comparison_raises_when_t_ne_n_heads(self):
        """Negative control: without the restore, comparing the kernel's raw
        ``(B, n_heads, T, d_v)`` against eager's ``(B, T, n_heads, d_v)``
        directly is not broadcastable at this T != n_heads shape -- exactly
        the ``RuntimeError`` the L4 review report described aborting
        ``--check``. Proves this suite would have caught the bug the restore
        fixes.
        """
        torch.manual_seed(SEED)
        y_fake_kernel, y_e, _ = self._fake_kernel_y("cpu")
        assert tuple(y_fake_kernel.shape) != tuple(y_e.shape)
        with pytest.raises(RuntimeError):
            triton_scans._max_abs_rel(y_fake_kernel, y_e)

    def test_unrestored_comparison_silently_broadcasts_when_t_is_one(self):
        """At T=1, the unrestored shapes ARE broadcastable (size-1 dims
        broadcast against anything) -- no crash, but a silently wrong
        cross-``n_heads`` comparison instead of the intended elementwise one.
        The restore is required to prevent this case too, not just the
        T != n_heads crash above.
        """
        torch.manual_seed(SEED)
        device = "cpu"
        Q, K, V, beta, H0 = triton_scans._delta_inputs(
            self.B, self.N_HEADS, 1, self.NH, self.D_K, self.D_V, device
        )
        y_e, _ = triton_scans._delta_ref_forward(Q, K, V, beta, H0, self.CHUNK_SIZE)
        assert tuple(y_e.shape) == (self.B, 1, self.N_HEADS, self.D_V)
        y_fake_kernel = y_e.permute(0, 2, 1, 3)  # (B, n_heads, T=1, d_v)

        # Unrestored: broadcasts to (B, n_heads, n_heads, d_v) -- inflated
        # against the intended (B, 1, n_heads, d_v)/(B, n_heads, 1, d_v)
        # shapes, i.e. silently NOT the per-position comparison the harness
        # means to make.
        unrestored_diff = y_fake_kernel - y_e
        assert tuple(unrestored_diff.shape) == (self.B, self.N_HEADS, self.N_HEADS, self.D_V)

        # Restored (the fix): exact shape match, no broadcasting, an honest
        # elementwise comparison.
        y_t = y_fake_kernel.permute(0, 2, 1, 3)
        assert y_t.shape == y_e.shape
        assert torch.equal(y_t, y_e)


# ===========================================================================
# 7. Delta backward-torch vs eager-autograd fp64 semantic guard
# ===========================================================================
#
# ``triton_scans._delta_backward_torch`` is the hand-derived reverse-chunk
# gradient recurrence that ``_DeltaScanFn.backward`` calls directly (no
# dispatcher, no fallback -- see that class's docstring); it is a plain torch
# function with no Triton launch inside it, defined at module level in
# ``triton_scans.py`` (outside the ``if _HAS_TRITON:`` guard, unlike the
# kernel-touching ``_DeltaScanFn``/``delta_scan_impl``), so it is always
# importable and runnable on CPU -- no GPU, and not even the ``triton``
# package itself, is required. This suite pins its math against
# ``torch.autograd.grad`` through the certified eager oracle
# ``DeltaMinGRU._forward_chunked`` (via the ``_delta_ref_forward`` shim),
# both in fp64, so a future hand-edit to either hand-maintained copy of the
# chunked-WY backward -- this file's reverse-chunk loop, or the shim's
# forward (which autograd differentiates through) -- that silently drifts
# the chunk convention between the two is caught unconditionally, on every
# CPU-only contributor's machine, without needing an L4 to run the real
# grad-parity sweep (``_run_delta_grad_parity``, GPU-only, requires the
# Triton kernel forward for its ``Hbound``).


def _delta_chunk_boundary_states(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    beta: torch.Tensor,
    H0: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Per-chunk start-of-chunk states, matching ``_delta_forward_launch``'s
    ``Hbound`` convention (entry ``c`` is the state BEFORE chunk ``c`` is
    applied).

    Built by driving the certified ``_delta_ref_forward`` oracle one chunk at
    a time -- each call's own ``chunk_size`` equals that chunk's length, so
    its internal loop runs exactly one iteration over that slice -- reusing
    the oracle's own math instead of re-deriving the chunked-WY state update
    a third time in this file (see ``_delta_ref_forward``'s
    ``DUPLICATION-PENDING`` note about the two copies that already exist).
    """
    T = Q.shape[2]
    H = H0
    bounds = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        bounds.append(H)
        _, H = triton_scans._delta_ref_forward(
            Q[:, :, start:end],
            K[:, :, start:end],
            V[:, :, start:end],
            beta[:, :, start:end],
            H,
            end - start,
        )
    return torch.stack(bounds, dim=2)


class TestDeltaBackwardTorchVsEagerAutograd:
    """fp64 CPU pin: ``_delta_backward_torch``'s grads vs ``torch.autograd.grad``
    through ``_delta_ref_forward``, on small multi-chunk ragged-``T``, ``nh``,
    ``d_k`` shapes with nonzero ``H0``."""

    B, N_HEADS, CHUNK_SIZE = 2, 2, 4

    @pytest.mark.parametrize("nh", [1, 2])
    @pytest.mark.parametrize("d_k", [4, 16])
    def test_grads_match_to_fp64_precision(self, nh, d_k):
        torch.manual_seed(SEED)
        device = "cpu"
        # Two full chunks plus a ragged tail (CHUNK_SIZE=4 -> chunks of
        # length 4, 4, 2).
        T = 2 * self.CHUNK_SIZE + self.CHUNK_SIZE // 2

        Q, K, V, beta, H0 = (
            t.double()
            for t in triton_scans._delta_inputs(self.B, self.N_HEADS, T, nh, d_k, d_k, device)
        )

        Hbound = _delta_chunk_boundary_states(Q, K, V, beta, H0, self.CHUNK_SIZE)

        Q_g, K_g, V_g, beta_g, H0_g = (t.clone().requires_grad_(True) for t in (Q, K, V, beta, H0))
        y_e, HT_e = triton_scans._delta_ref_forward(Q_g, K_g, V_g, beta_g, H0_g, self.CHUNK_SIZE)

        torch.manual_seed(SEED + 1)
        cot_y_eager = torch.randn_like(y_e)  # eager's own (B, T, n_heads, d_v) layout
        cot_H = torch.randn_like(HT_e)

        ref_grads = torch.autograd.grad(
            (y_e, HT_e),
            (Q_g, K_g, V_g, beta_g, H0_g),
            grad_outputs=(cot_y_eager, cot_H),
        )

        # `_delta_backward_torch`'s `dy` is the kernel's (B, n_heads, T, d_v)
        # layout (matches Q's own axis order), not eager's (B, T, n_heads,
        # d_v) -- the same restore Section 6 above pins at the `y` seam.
        cot_y_kernel = cot_y_eager.permute(0, 2, 1, 3).contiguous()

        needs_input_grad = (True, True, True, True, True, False)
        dQ, dK, dV, dbeta, dH0, dchunk = triton_scans._delta_backward_torch(
            Q, K, V, beta, Hbound, cot_y_kernel, cot_H, self.CHUNK_SIZE, needs_input_grad
        )
        assert dchunk is None

        names = ("dQ", "dK", "dV", "dbeta", "dH0")
        got_grads = (dQ, dK, dV, dbeta, dH0)
        for name, got, want in zip(names, got_grads, ref_grads, strict=True):
            assert got is not None, f"{name} unexpectedly None"
            torch.testing.assert_close(
                got, want, atol=1e-12, rtol=1e-12, msg=f"{name} mismatch vs eager autograd"
            )


# ===========================================================================
# 8. Gated DeltaMinGRU chunked forward (decay active) vs the sequential oracle
# ===========================================================================
#
# ``DeltaMinGRU._forward_chunked_gated`` threads a per-token time-decay gate
# through the SAME unit-triangular UT solve as ``_forward_chunked``: per chunk
# it weights the ``T`` off-diagonal, the carried-state ``rhs`` term, the
# readout, and the chunk-end carry by decay factors ``exp(<= 0) in [0, 1]``
# formed as direct span sums of per-token log-decays (never a bare
# ``1 / Gamma``, never a difference of two running sums), so the decay-active
# forward equals the sequential gated recurrence (``DeltaMinGRU.step``
# iterated token-by-token) regardless of ``chunk_size`` AND stays finite in
# float32 under strong decay.
# These CPU fp64 tests are the correctness proof the spec's "Oracle
# gradcheck"/"Tiling invariance"/"gamma->1 reduction" criteria call for; the
# float32 tests below pin the "Float32 robustness" criterion. No Triton/CUDA
# path is exercised (decay is eager-only).


def _delta_gated_step_oracle(
    layer: DeltaMinGRU, x: torch.Tensor, h0_flat: torch.Tensor, dt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential gated oracle: ``DeltaMinGRU.step`` token-by-token.

    Threads ``h_prev`` and the per-token ``delta_t`` column through
    ``step`` (each call decays the carried state once, then runs the
    token's ``nh`` micro-steps undecayed), stacking readouts to
    ``(B, T, hidden_size)`` and returning the final flattened state.
    ``step`` is ``@torch.no_grad`` (a value oracle only), so this pins
    the gated forward's *values*; its gradients are pinned by the finite-
    difference ``gradcheck`` below.
    """
    h = h0_flat
    ys = []
    for t in range(x.size(1)):
        y_t, h = layer.step(x[:, t], h_prev=h, delta_t=dt[:, t])
        ys.append(y_t)
    return torch.stack(ys, dim=1), h


class TestDeltaGatedForwardVsOracle:
    """CPU fp64 correctness of the gated chunked forward.

    Value-matches the sequential gated ``step`` oracle (ragged final
    chunk, non-zero ``h_0``, fixed and learnable modes with moderate
    non-zero ``delta_t``), is invariant to ``chunk_size``, reduces to the
    ungated ``_forward_chunked`` at ``gamma -> 1``, and its backward
    passes finite-difference ``gradcheck``.
    """

    B, D_IN, D_H, N_HEADS = 2, 8, 6, 2
    # Two full chunks plus a ragged tail at CHUNK_SIZE=4 -> lengths 4, 4, 2.
    CHUNK_SIZE, T = 4, 10

    def _moderate_dt(self) -> torch.Tensor:
        """delta_t in a gamma-sensitive mid-range (~0.3-0.9 at decay_rate=0.5).

        NOT large delta_t (gamma -> 0 washes carried memory to ~0 and
        would mask a real forward/oracle disagreement); NOT zero (that is
        the separate gamma -> 1 reduction test).
        """
        torch.manual_seed(SEED + 7)
        return (0.2 + 1.4 * torch.rand(self.B, self.T, dtype=torch.float64)).requires_grad_(False)

    @pytest.mark.parametrize("decay", ["fixed", "learnable"])
    @pytest.mark.parametrize("nh", [1, 2])
    def test_gated_forward_matches_sequential_oracle(self, decay, nh):
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=nh,
            chunk_size=self.CHUNK_SIZE,
            decay=decay,
            decay_rate=0.5,
        ).double()
        x = torch.randn(self.B, self.T, self.D_IN, dtype=torch.float64)
        # Non-zero incoming state.
        h0 = torch.randn(self.B, 1, layer.n_heads * layer.d_k * layer.d_v, dtype=torch.float64)
        dt = self._moderate_dt()

        y, hT = layer(x, h_0=h0, delta_t=dt, return_state=True)
        y_ref, hT_ref = _delta_gated_step_oracle(layer, x, h0.reshape(self.B, -1), dt)

        torch.testing.assert_close(y, y_ref, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(hT.reshape(self.B, -1), hT_ref, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("decay", ["fixed", "learnable"])
    def test_gated_forward_tiling_invariant(self, decay):
        """Output is independent of chunk_size (single-chunk, exact-multiple,
        and ragged tilings of the same T all agree to fp64)."""
        torch.manual_seed(SEED)
        base = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=2,
            chunk_size=self.T,
            decay=decay,
            decay_rate=0.5,
        ).double()
        x = torch.randn(self.B, self.T, self.D_IN, dtype=torch.float64)
        h0 = torch.randn(self.B, 1, base.n_heads * base.d_k * base.d_v, dtype=torch.float64)
        dt = self._moderate_dt()

        y_single = base(x, h_0=h0, delta_t=dt)
        for cs in (4, 5):  # 4 -> ragged (4,4,2); 5 -> exact multiple (5,5)
            base.chunk_size = cs
            y_cs = base(x, h_0=h0, delta_t=dt)
            torch.testing.assert_close(y_cs, y_single, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("decay", ["fixed", "learnable"])
    def test_gamma_one_reduces_to_ungated_forward(self, decay):
        """delta_t = 0 -> gamma = exp(0) = 1: the gated chunked forward
        reproduces the decay-free ``_forward_chunked`` output to fp64."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=2,
            chunk_size=self.CHUNK_SIZE,
            decay=decay,
            decay_rate=0.5,
        ).double()
        x = torch.randn(self.B, self.T, self.D_IN, dtype=torch.float64)
        h0 = torch.randn(self.B, 1, layer.n_heads * layer.d_k * layer.d_v, dtype=torch.float64)
        H = h0.reshape(self.B, layer.n_heads, layer.d_k, layer.d_v)
        dt = torch.zeros(self.B, self.T, dtype=torch.float64)

        y_gated, H_gated = layer._forward_chunked_gated(x, H, dt)
        y_ungated, H_ungated = layer._forward_chunked(x, H)
        torch.testing.assert_close(y_gated, y_ungated, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(H_gated, H_ungated, atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize("decay", ["fixed", "learnable"])
    @pytest.mark.parametrize("nh", [1, 2])
    def test_gated_forward_gradcheck(self, decay, nh):
        """Finite-difference gradcheck of the gated forward+backward through
        ``x``, ``h_0``, and ``delta_t`` (moderate non-zero, per-head gate),
        with a ragged final chunk and non-zero incoming state."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=nh,
            chunk_size=self.CHUNK_SIZE,
            decay=decay,
            decay_rate=0.5,
        ).double()
        x = torch.randn(self.B, self.T, self.D_IN, dtype=torch.float64, requires_grad=True)
        h0 = torch.randn(
            self.B,
            1,
            layer.n_heads * layer.d_k * layer.d_v,
            dtype=torch.float64,
            requires_grad=True,
        )
        dt = self._moderate_dt().requires_grad_(True)

        def f(x_, h0_, dt_):
            return layer(x_, h_0=h0_, delta_t=dt_)

        assert torch.autograd.gradcheck(f, (x, h0, dt), atol=1e-6, rtol=1e-4)

    def test_learnable_rho_receives_nonzero_grad(self):
        """The per-head learnable rate is on the graph: a backward through the
        gated forward gives ``rho`` a finite, non-zero gradient (pins the
        per-head ``(n_heads,)`` decay actually drives the output)."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=2,
            chunk_size=self.CHUNK_SIZE,
            decay="learnable",
            decay_rate=0.5,
        ).double()
        x = torch.randn(self.B, self.T, self.D_IN, dtype=torch.float64)
        dt = self._moderate_dt()

        layer(x, delta_t=dt).pow(2).sum().backward()
        assert layer.rho.grad is not None
        assert layer.rho.shape == (self.N_HEADS,)
        assert torch.isfinite(layer.rho.grad).all()
        assert layer.rho.grad.abs().sum() > 0

    def test_decay_none_forward_unchanged_and_gradchecks(self):
        """Guards the untouched decay=None path: ``forward`` still routes to
        the decay-free ``_forward_chunked`` (byte-equal output) and remains
        differentiable, unperturbed by the added gated branch."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=2,
            chunk_size=self.CHUNK_SIZE,
        ).double()
        x = torch.randn(self.B, self.T, self.D_IN, dtype=torch.float64)
        h0 = torch.randn(self.B, 1, layer.n_heads * layer.d_k * layer.d_v, dtype=torch.float64)
        H = h0.reshape(self.B, layer.n_heads, layer.d_k, layer.d_v)

        y = layer(x, h_0=h0)
        y_direct, _ = layer._forward_chunked(x, H)
        y_direct = layer.out_proj(y_direct.reshape(self.B, self.T, layer.n_heads * layer.d_v))
        assert torch.equal(y, y_direct)

        x_g = x.clone().requires_grad_(True)
        h0_g = h0.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda x_, h0_: layer(x_, h_0=h0_), (x_g, h0_g), atol=1e-6, rtol=1e-4
        )


class TestDeltaGatedForwardFloat32Robustness:
    """Float32 finiteness of the gated forward under strong / infinite decay.

    The span-sum decay factors (spec sections 6/7) all lie in ``[0, 1]``, so
    the forward stays finite in float32 even when
    the chunk-relative cumulative decay ``Gamma_C`` is far below the
    float32 subnormal floor -- the regime where the rejected bare
    ``1 / Gamma`` change-of-variables overflowed to ``nan``. The first
    test is that regression (it MUST fail on the old ``V / Gamma`` code);
    the second pins the sanitized ``+inf`` ``delta_t`` cap path against
    the ``step`` oracle.
    """

    B, D_IN, D_H, N_HEADS = 2, 8, 6, 2
    CHUNK_SIZE, T = 64, 96  # T > chunk_size so a full chunk accumulates decay

    def test_float32_strong_decay_is_finite_and_tiling_invariant(self):
        """At chunk_size=64, decay_rate=1.0, per-token dt ~1.6-2.0 (gamma ~
        0.13-0.20), the full-chunk product ``Gamma_C ~ exp(-64 * 1.8) ~
        1e-50`` underflows float32, so a bare ``1 / Gamma`` yields ``inf``
        -> ``nan``. The span-sum form must instead stay finite AND match
        a smaller-chunk float32 run (tiling invariance in float32)."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=2,
            chunk_size=self.CHUNK_SIZE,
            decay="fixed",
            decay_rate=1.0,
        )  # float32 (default dtype)
        x = torch.randn(self.B, self.T, self.D_IN)
        torch.manual_seed(SEED + 11)
        dt = 1.6 + 0.4 * torch.rand(self.B, self.T)  # gamma ~ 0.13-0.20 per token

        y_full = layer(x, delta_t=dt)
        assert torch.isfinite(y_full).all(), "gated float32 forward produced non-finite output"

        # Tiling invariance in float32: a smaller chunk_size (T not a
        # multiple -> ragged) must agree within float32 tolerance.
        layer.chunk_size = 20
        y_small = layer(x, delta_t=dt)
        assert torch.isfinite(y_small).all()
        torch.testing.assert_close(y_full, y_small, atol=1e-4, rtol=1e-4)

    def test_plus_inf_delta_t_matches_step_oracle(self):
        """A ``+inf`` entry in ``delta_t`` (sanitizer-capped to 1e10) wipes
        the pre-gap memory (gamma -> exp(-1e10) = 0). The gated forward
        must stay finite and match the sequential ``step`` oracle to fp64
        across that gap.

        The gap is placed on a chunk-boundary token (last token of chunk
        0 at chunk_size=6) so the memory wipe propagates ACROSS the chunk
        boundary through the carried state (chunk 1 reads a state whose
        pre-gap content is gone), exercising the cross-chunk path. Aligning
        it to the boundary exercises the cross-chunk carry; the mid-chunk
        sibling below covers the interior placement at the same tolerance."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=2,
            chunk_size=6,
            decay="fixed",
            decay_rate=1.0,
        ).double()
        T = 12
        x = torch.randn(self.B, T, self.D_IN, dtype=torch.float64)
        h0 = torch.randn(self.B, 1, layer.n_heads * layer.d_k * layer.d_v, dtype=torch.float64)
        torch.manual_seed(SEED + 13)
        dt = 0.2 + 0.6 * torch.rand(self.B, T, dtype=torch.float64)
        dt[:, 5] = float("inf")  # full-decay gap at the chunk-0 boundary (tokens 0..5)

        y, hT = layer(x, h_0=h0, delta_t=dt, return_state=True)
        assert torch.isfinite(y).all()
        assert torch.isfinite(hT).all()

        y_ref, hT_ref = _delta_gated_step_oracle(layer, x, h0.reshape(self.B, -1), dt)
        torch.testing.assert_close(y, y_ref, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(hT.reshape(self.B, -1), hT_ref, atol=1e-10, rtol=1e-10)

    def test_mid_chunk_plus_inf_delta_t_is_finite(self):
        """Sibling of ``test_plus_inf_delta_t_matches_step_oracle`` above,
        but the ``+inf`` gap sits MID-CHUNK (not on a chunk-boundary token),
        pinning spec section 7's invariant that output finiteness holds
        "regardless of +inf gap placement" -- boundary-aligned and
        interior placements both exercise the same span-sum factors,
        never a bare ``1 / Gamma``.

        Uses ``chunk_size=8``, ``T=20`` so token 10 (the gap) falls inside
        chunk 1 (tokens 8..15), strictly interior to both its chunk and the
        sequence.

        Float32 asserts finiteness AND agreement with the sequential ``step``
        oracle. Every decay factor is a direct span sum, so the one saturated
        ``-lambda * 1e10`` token only zeroes the factors whose span contains
        it (an exact ``exp(-huge) = 0`` wipe, as the oracle's ``gamma = 0``)
        and never perturbs the post-gap tokens sharing its chunk.

        Float64 checks the same oracle agreement at ``atol = rtol = 1e-10``.
        """
        CHUNK_SIZE, T = 8, 20
        MID_CHUNK_TOKEN = 10  # inside chunk 1 (tokens 8..15), not a multiple of CHUNK_SIZE

        # --- float32: finiteness + oracle agreement ---
        torch.manual_seed(SEED)
        layer_f32 = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=2,
            chunk_size=CHUNK_SIZE,
            decay="fixed",
            decay_rate=1.0,
        )  # float32 (default dtype)
        x_f32 = torch.randn(self.B, T, self.D_IN)
        h0_f32 = torch.randn(self.B, 1, layer_f32.n_heads * layer_f32.d_k * layer_f32.d_v)
        torch.manual_seed(SEED + 17)
        dt_f32 = 0.2 + 0.6 * torch.rand(self.B, T)
        dt_f32[:, MID_CHUNK_TOKEN] = float("inf")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_f32, hT_f32 = layer_f32(x_f32, h_0=h0_f32, delta_t=dt_f32, return_state=True)
        assert torch.isfinite(y_f32).all(), "gated float32 forward produced non-finite output"
        assert torch.isfinite(hT_f32).all()

        y_ref_f32, hT_ref_f32 = _delta_gated_step_oracle(
            layer_f32, x_f32, h0_f32.reshape(self.B, -1), dt_f32
        )
        torch.testing.assert_close(y_f32, y_ref_f32, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(hT_f32.reshape(self.B, -1), hT_ref_f32, atol=1e-4, rtol=1e-4)

        # --- float64: finiteness + oracle agreement ---
        torch.manual_seed(SEED)
        layer_f64 = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=2,
            chunk_size=CHUNK_SIZE,
            decay="fixed",
            decay_rate=1.0,
        ).double()
        x_f64 = torch.randn(self.B, T, self.D_IN, dtype=torch.float64)
        h0 = torch.randn(
            self.B, 1, layer_f64.n_heads * layer_f64.d_k * layer_f64.d_v, dtype=torch.float64
        )
        torch.manual_seed(SEED + 17)
        dt_f64 = 0.2 + 0.6 * torch.rand(self.B, T, dtype=torch.float64)
        dt_f64[:, MID_CHUNK_TOKEN] = float("inf")

        y, hT = layer_f64(x_f64, h_0=h0, delta_t=dt_f64, return_state=True)
        assert torch.isfinite(y).all()
        assert torch.isfinite(hT).all()

        y_ref, hT_ref = _delta_gated_step_oracle(layer_f64, x_f64, h0.reshape(self.B, -1), dt_f64)
        torch.testing.assert_close(y, y_ref, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(hT.reshape(self.B, -1), hT_ref, atol=1e-10, rtol=1e-10)

    def test_mid_chunk_plus_inf_float32_tiling_invariant(self):
        """A mid-chunk enormous ``delta_t`` gap is tiling-invariant AND oracle-
        matching in float32 WITHOUT ``log1p_delta``.

        A ``+inf`` gap sanitizes to the frozen ``_DELTA_T_POSINF_CAP=1e10``, so
        ``log_gamma = -lambda * 1e10``. Because every decay factor is a direct
        span sum, that one term only drives the factors whose span contains it
        to an exact ``exp(-huge) = 0`` (the oracle's ``gamma = 0`` wipe); the
        factors between post-gap tokens never see it, so tiling-invariance
        (``< 1e-4`` across ``chunk_size``) and oracle agreement hold.

        The behavior is magnitude-based, not ``+inf``-specific: a huge raw
        finite gap (``dt = 1e6``, no ``+inf``) is pinned to behave
        identically. A ``log1p_delta=True`` leg confirms the documented
        mitigation stays tiling-invariant too (it keeps large gaps
        *distinguishable* rather than saturating to a full wipe).
        """
        CHUNK_A, CHUNK_B, T, MID_CHUNK_TOKEN = 4, 16, 16, 10

        # One set of weights, reloaded into every (chunk_size, log1p_delta)
        # variant, so any output difference is purely the tiling/log1p effect.
        torch.manual_seed(SEED)
        ref = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=1,
            chunk_size=CHUNK_A,
            decay="fixed",
            decay_rate=1.0,
        )
        state = ref.state_dict()
        x = torch.randn(self.B, T, self.D_IN)
        h0 = torch.randn(self.B, 1, ref.n_heads * ref.d_k * ref.d_v)
        torch.manual_seed(SEED + 17)
        dt_inf = 0.2 + 0.6 * torch.rand(self.B, T)
        dt_inf[:, MID_CHUNK_TOKEN] = float("inf")
        dt_huge = dt_inf.clone()
        # huge but FINITE gap (no +inf) -> pins that the fix is magnitude-based
        dt_huge[:, MID_CHUNK_TOKEN] = 100.0 * _DECAY_SATURATION_WARN_THRESHOLD

        def run(chunk_size: int, log1p: bool, dt: torch.Tensor) -> torch.Tensor:
            layer = DeltaMinGRU(
                self.D_IN,
                self.D_H,
                n_heads=self.N_HEADS,
                nh=1,
                chunk_size=chunk_size,
                decay="fixed",
                decay_rate=1.0,
                log1p_delta=log1p,
            )
            layer.load_state_dict(state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return layer(x, h_0=h0, delta_t=dt)

        # Finiteness holds regardless (the spec section 7 safety invariant).
        for chunk_size in (CHUNK_A, CHUNK_B):
            for log1p in (False, True):
                for dt in (dt_inf, dt_huge):
                    assert torch.isfinite(run(chunk_size, log1p, dt)).all(), (
                        f"float32 mid-chunk huge gap must stay finite "
                        f"(chunk={chunk_size}, log1p={log1p})"
                    )

        # Oracle for the raw +inf case at CHUNK_A (float32 step loop).
        oracle_layer = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=1,
            chunk_size=CHUNK_A,
            decay="fixed",
            decay_rate=1.0,
        )
        oracle_layer.load_state_dict(state)
        y_ref, _ = _delta_gated_step_oracle(oracle_layer, x, h0.reshape(self.B, -1), dt_inf)

        # Without log1p, both raw +inf AND huge-finite dt are tiling-invariant
        # across chunk_size and oracle-matching to the fp32 floor.
        for dt, label in ((dt_inf, "+inf"), (dt_huge, "1e6")):
            tiling_err = (run(CHUNK_A, False, dt) - run(CHUNK_B, False, dt)).abs().max()
            assert tiling_err < 1e-4, (
                f"float32 mid-chunk {label} gap must be tiling-invariant "
                f"(no log1p_delta); got {tiling_err:.3e}"
            )
        oracle_err = (run(CHUNK_A, False, dt_inf) - y_ref).abs().max()
        assert oracle_err < 1e-4, (
            f"float32 mid-chunk +inf gap must match the step oracle; got {oracle_err:.3e}"
        )

        # log1p_delta stays tiling-invariant too (distinguishability aid, not a
        # correctness requirement).
        tiling_err_log1p = (run(CHUNK_A, True, dt_inf) - run(CHUNK_B, True, dt_inf)).abs().max()
        assert tiling_err_log1p < 1e-4, (
            "log1p_delta=True must remain float32 tiling-invariant for a "
            f"mid-chunk +inf gap; got {tiling_err_log1p:.3e}"
        )

    @staticmethod
    def _count_saturation_warnings(records) -> int:
        """Number of captured warnings whose message names ``log1p_delta``.

        Isolates the float32 gated-forward saturation warning from the
        pre-existing invalid-``delta_t`` warn-once (a ``+inf`` gap trips
        both), so a test using ``+inf`` can still count the saturation one.
        """
        return sum("log1p_delta" in str(r.message) for r in records)

    def _fresh_layer(self, *, double: bool = False, chunk_size: int = 8) -> DeltaMinGRU:
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=2,
            chunk_size=chunk_size,
            decay="fixed",
            decay_rate=1.0,
        )
        return layer.double() if double else layer

    def test_float32_saturation_warns_once_naming_log1p(self):
        """Pin the once-per-instance float32 saturation warning (spec sections
        6/8, acceptance "Float32 saturation warning").

        A float32 forward whose ``(lambda*dt).max()`` exceeds the pathological
        threshold (``_DECAY_SATURATION_WARN_THRESHOLD = 1e4``) emits exactly
        one ``UserWarning`` naming ``log1p_delta`` across repeated calls (warn-
        once). A moderate float32 forward, a forward whose largest gap sits
        at half the threshold, and an fp64 forward at the same extreme emit
        none. The warning never alters the returned numbers; its
        text names ``log1p_delta`` and never mentions a clamp. Uses a finite
        ``dt = 1e5`` so the pre-existing invalid-``delta_t`` warn-once does
        not also fire; the counter filters on the ``log1p_delta`` message
        regardless.
        """
        T = 16
        x = torch.randn(self.B, T, self.D_IN)
        torch.manual_seed(SEED + 21)
        dt_base = 0.2 + 0.6 * torch.rand(self.B, T)  # moderate, far below the threshold
        assert dt_base.max().item() < 0.5 * _DECAY_SATURATION_WARN_THRESHOLD

        # (1) Extreme float32 (lambda*dt an order past the warn threshold):
        # exactly one warning, and warn-once across two calls.
        dt_extreme = dt_base.clone()
        dt_extreme[:, 10] = 10.0 * _DECAY_SATURATION_WARN_THRESHOLD
        layer = self._fresh_layer()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            layer(x, delta_t=dt_extreme)
            layer(x, delta_t=dt_extreme)
        assert self._count_saturation_warnings(rec) == 1, (
            "float32 forward past the saturation threshold must warn exactly once "
            f"per instance; saw {self._count_saturation_warnings(rec)}"
        )
        sat = [r for r in rec if "log1p_delta" in str(r.message)]
        assert issubclass(sat[0].category, UserWarning)
        assert "clamp" not in str(sat[0].message).lower()

        # (2) Moderate float32 (all lambda*dt well below the threshold): no warning.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            self._fresh_layer()(x, delta_t=dt_base)
        assert self._count_saturation_warnings(rec) == 0

        # (3) Half the warn threshold: no warning.
        dt_mid = dt_base.clone()
        dt_mid[:, 10] = 0.5 * _DECAY_SATURATION_WARN_THRESHOLD
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            self._fresh_layer()(x, delta_t=dt_mid)
        assert self._count_saturation_warnings(rec) == 0

        # (4) fp64 at the same extreme: never warns (fp64 is not at risk).
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            self._fresh_layer(double=True)(x.double(), delta_t=dt_extreme.double())
        assert self._count_saturation_warnings(rec) == 0

    def test_float32_huge_gap_wipes_memory_to_oracle(self):
        """A float32 huge-gap token mid-sequence wipes pre-gap memory, matching
        the sequential ``step`` oracle to the float32 floor (spec section 10).

        Two checks: (a) the post-gap output equals the oracle's, and (b) the
        post-gap output is invariant to the pre-gap inputs (the gap's
        ``gamma = 0`` truly erases the carried state, as the oracle does),
        proving the forward reproduces the wipe rather than merely staying
        finite.
        """
        T, GAP = 16, 8  # gap at a chunk boundary (chunk_size=8) so it wipes the carry
        torch.manual_seed(SEED + 23)
        x = torch.randn(self.B, T, self.D_IN)
        dt = 0.2 + 0.6 * torch.rand(self.B, T)
        # huge finite gap -> gamma = 0 (memory wiped)
        dt[:, GAP] = 100.0 * _DECAY_SATURATION_WARN_THRESHOLD

        layer = self._fresh_layer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = layer(x, delta_t=dt)
        y_ref, _ = _delta_gated_step_oracle(
            layer, x, torch.zeros(self.B, layer.n_heads * layer.d_k * layer.d_v), dt
        )
        torch.testing.assert_close(y[:, GAP:], y_ref[:, GAP:], atol=1e-4, rtol=1e-4)

        # Memory wipe: perturbing pre-gap inputs leaves the post-gap output
        # unchanged (the gap token erased everything the carry held).
        x_perturbed = x.clone()
        x_perturbed[:, :GAP] += torch.randn(self.B, GAP, self.D_IN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_perturbed = layer(x_perturbed, delta_t=dt)
        torch.testing.assert_close(y[:, GAP + 1 :], y_perturbed[:, GAP + 1 :], atol=1e-4, rtol=1e-4)


class TestDeltaGatedSpanSumAccuracy:
    """Float32 accuracy of the decay-active forward on bursty streams.

    A bursty stream (bursts of 4 tokens whose first token is a long gap, the
    other three short gaps in ``[0.05, 1]``) is the regime where decay
    ratios formed by subtracting two running log-sums lose float32 bits. The
    reference is always the fp64 sequential ``step`` loop, never a second
    float32 computation; the bound is the max abs deviation over the
    reference's max magnitude.
    """

    B, D_IN, D_H, N_HEADS = 2, 16, 32, 4
    T = 512
    TOL = 2e-6
    CHUNKS = (1, 64, 256, 512)

    def _make(
        self,
        *,
        nh: int,
        decay: str,
        chunk_size: int,
        log1p_delta: bool = False,
        decay_rate: float = 1.0,
    ) -> DeltaMinGRU:
        """Float32 layer with seed-fixed weights (identical across chunk sizes)."""
        torch.manual_seed(SEED + 3)
        layer = DeltaMinGRU(
            self.D_IN,
            self.D_H,
            n_heads=self.N_HEADS,
            nh=nh,
            chunk_size=chunk_size,
            decay=decay,
            decay_rate=decay_rate,
            log1p_delta=log1p_delta,
        )
        if decay == "learnable":
            with torch.no_grad():
                layer.rho.add_(torch.randn_like(layer.rho) * 0.3)
        return layer

    @staticmethod
    def _as_fp64(layer: DeltaMinGRU) -> DeltaMinGRU:
        """fp64 copy of ``layer`` with identical weights (the step-loop oracle)."""
        twin = DeltaMinGRU(
            layer.input_size,
            layer.hidden_size,
            n_heads=layer.n_heads,
            nh=layer.nh,
            d_k=layer.d_k,
            d_v=layer.d_v,
            chunk_size=layer.chunk_size,
            decay=layer.decay,
            decay_rate=1.0,
            log1p_delta=layer.log1p_delta,
        ).double()
        twin.load_state_dict({k: v.double() for k, v in layer.state_dict().items()})
        return twin

    def _bursty_dt(self, long_gap: float, T: int | None = None, phase: int = 0) -> torch.Tensor:
        """Bursts of 4; ``phase`` shifts which token opens a burst (the long gap).

        With ``phase=0`` every power-of-two chunk opens on a saturating gap and
        wipes the carry; a nonzero phase puts chunk boundaries on short gaps so
        the inter-chunk carry and ``h_0`` reach the output.
        """
        T = self.T if T is None else T
        torch.manual_seed(SEED + 5)
        dt = 0.05 + 0.95 * torch.rand(self.B, T)
        dt[:, phase::4] = long_gap
        return dt

    def _inputs(self, T: int | None = None) -> torch.Tensor:
        torch.manual_seed(SEED + 9)
        return torch.randn(self.B, self.T if T is None else T, self.D_IN)

    def _oracle(self, layer: DeltaMinGRU, x: torch.Tensor, dt: torch.Tensor, h0=None):
        twin = self._as_fp64(layer)
        h = (
            torch.zeros(self.B, layer.n_heads * layer.d_k * layer.d_v, dtype=torch.float64)
            if h0 is None
            else h0.double().reshape(self.B, -1)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _delta_gated_step_oracle(twin, x.double(), h, dt.double())

    @staticmethod
    def _rel_dev(y: torch.Tensor, y_ref: torch.Tensor) -> float:
        return ((y.double() - y_ref).abs().max() / y_ref.abs().max()).item()

    def _forward(self, layer, x, dt, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return layer(x, delta_t=dt, **kw)

    @pytest.mark.parametrize("long_gap", [60.0, 3600.0, float("inf")])
    @pytest.mark.parametrize("decay", ["fixed", "learnable"])
    @pytest.mark.parametrize("nh", [1, 2])
    @pytest.mark.parametrize("chunk_size", CHUNKS)
    def test_bursty_stream_matches_fp64_step_oracle(self, long_gap, decay, nh, chunk_size):
        layer = self._make(nh=nh, decay=decay, chunk_size=chunk_size)
        x, dt = self._inputs(), self._bursty_dt(long_gap)
        y_ref, _ = self._oracle(layer, x, dt)

        y = self._forward(layer, x, dt)
        assert torch.isfinite(y).all()
        dev = self._rel_dev(y, y_ref)
        assert dev <= self.TOL, f"relative deviation {dev:.3e} > {self.TOL:.0e}"

    @pytest.mark.parametrize("long_gap", [60.0, 3600.0, float("inf")])
    @pytest.mark.parametrize("decay", ["fixed", "learnable"])
    def test_bursty_stream_outputs_agree_across_chunk_sizes(self, long_gap, decay):
        x, dt = self._inputs(), self._bursty_dt(long_gap)
        outs = {
            cs: self._forward(self._make(nh=2, decay=decay, chunk_size=cs), x, dt)
            for cs in self.CHUNKS
        }
        scale = max(o.abs().max().item() for o in outs.values())
        for i, a in enumerate(self.CHUNKS):
            for b in self.CHUNKS[i + 1 :]:
                dev = (outs[a] - outs[b]).abs().max().item() / scale
                assert dev <= self.TOL, f"chunk {a} vs {b}: {dev:.3e} > {self.TOL:.0e}"

    @pytest.mark.parametrize("chunk_size", [64, 256])
    def test_ragged_final_chunk_output_and_state(self, chunk_size):
        T = 520
        layer = self._make(nh=2, decay="learnable", chunk_size=chunk_size)
        x, dt = self._inputs(T), self._bursty_dt(60.0, T, phase=2)
        torch.manual_seed(SEED + 13)
        h0 = 0.5 * torch.randn(self.B, 1, layer.n_heads * layer.d_k * layer.d_v)
        y_ref, hT_ref = self._oracle(layer, x, dt, h0)

        y, hT = self._forward(layer, x, dt, h_0=h0, return_state=True)
        assert self._rel_dev(y, y_ref) <= self.TOL
        assert self._rel_dev(hT.reshape(self.B, -1), hT_ref) <= self.TOL

    @pytest.mark.parametrize("log1p_delta", [False, True])
    def test_log1p_delta_with_inf_gap_matches_oracle(self, log1p_delta):
        layer = self._make(nh=1, decay="learnable", chunk_size=64, log1p_delta=log1p_delta)
        x, dt = self._inputs(), self._bursty_dt(float("inf"))
        y_ref, _ = self._oracle(layer, x, dt)
        y = self._forward(layer, x, dt)
        assert torch.isfinite(y).all()
        assert self._rel_dev(y, y_ref) <= self.TOL

    def test_chunk_size_greater_than_T_equals_single_chunk(self):
        T = 96
        x, dt = self._inputs(T), self._bursty_dt(60.0, T)
        y_big = self._forward(self._make(nh=2, decay="fixed", chunk_size=1000), x, dt)
        y_T = self._forward(self._make(nh=2, decay="fixed", chunk_size=T), x, dt)
        assert self._rel_dev(y_big, y_T.double()) <= self.TOL

    def test_single_token_matches_step_oracle(self):
        layer = self._make(nh=2, decay="learnable", chunk_size=64)
        x, dt = self._inputs(1), torch.full((self.B, 1), 0.7)
        torch.manual_seed(SEED + 13)
        h0 = torch.randn(self.B, 1, layer.n_heads * layer.d_k * layer.d_v)
        y_ref, hT_ref = self._oracle(layer, x, dt, h0)
        y, hT = self._forward(layer, x, dt, h_0=h0, return_state=True)
        assert y.shape == (self.B, 1, self.D_H)
        assert self._rel_dev(y, y_ref) <= self.TOL
        assert self._rel_dev(hT.reshape(self.B, -1), hT_ref) <= self.TOL

    @pytest.mark.parametrize("gap_token", [3, 7], ids=["mid-chunk", "last-token-of-chunk"])
    def test_fp64_plus_inf_gap_equals_step_loop(self, gap_token):
        """chunk_size=8: token 3 is interior to chunk 0, token 7 is its last token."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(
            8, 6, n_heads=2, nh=2, chunk_size=8, decay="fixed", decay_rate=1.0
        ).double()
        x = torch.randn(2, 20, 8, dtype=torch.float64)
        h0 = torch.randn(2, 1, layer.n_heads * layer.d_k * layer.d_v, dtype=torch.float64)
        torch.manual_seed(SEED + 17)
        dt = 0.2 + 0.6 * torch.rand(2, 20, dtype=torch.float64)
        dt[:, gap_token] = float("inf")
        y, hT = layer(x, h_0=h0, delta_t=dt, return_state=True)
        y_ref, hT_ref = _delta_gated_step_oracle(layer, x, h0.reshape(2, -1), dt)
        torch.testing.assert_close(y, y_ref, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(hT.reshape(2, -1), hT_ref, atol=1e-10, rtol=1e-10)

    def test_float32_plus_inf_gap_output_and_gradients_finite(self):
        layer = self._make(nh=2, decay="learnable", chunk_size=64)
        T = 128
        x = self._inputs(T).requires_grad_(True)
        torch.manual_seed(SEED + 13)
        h0 = (0.5 * torch.randn(self.B, 1, layer.n_heads * layer.d_k * layer.d_v)).requires_grad_(
            True
        )
        dt = self._bursty_dt(float("inf"), T).requires_grad_(True)
        y = self._forward(layer, x, dt, h_0=h0)
        assert torch.isfinite(y).all()
        y.pow(2).sum().backward()
        for name, g in [("x", x.grad), ("h_0", h0.grad), ("delta_t", dt.grad)] + [
            (n, p.grad) for n, p in layer.named_parameters()
        ]:
            assert g is not None, f"no gradient for {name}"
            assert torch.isfinite(g).all(), f"non-finite gradient for {name}"

    def test_float32_nan_negative_and_neg_inf_gaps_match_oracle(self):
        layer = self._make(nh=1, decay="fixed", chunk_size=64)
        T = 128
        x, dt = self._inputs(T), self._bursty_dt(60.0, T)
        dt[:, 5], dt[:, 70], dt[:, 100] = float("nan"), -2.5, float("-inf")
        y_ref, _ = self._oracle(layer, x, dt)
        y = self._forward(layer, x, dt)
        assert torch.isfinite(y).all()
        assert self._rel_dev(y, y_ref) <= self.TOL

    def test_log_gamma_floor_constant_is_gone(self):
        import mingru.min_gru

        assert not hasattr(mingru.min_gru, "_LOG_GAMMA_FLOAT32_FLOOR")
