"""Tests for the four public parallel-scan functions.

CPU-only: no GPU/Triton execution paths are exercised here (dispatch
semantics are covered in ``test_dispatch.py``). Every scan runs its eager
implementation because ``MINGRU_SCAN`` is unset / ``auto`` on CPU.

Sections
--------
1. parallel_scan_log -- shapes, dtype, T=1, non-power-of-two T, correctness
   vs. a sequential reference, gradcheck
2. linear_scan -- signed coefficients, shapes, T=1, correctness, gradcheck
3. matrix_scan -- 2x2 block transitions, shapes, correctness, gradcheck
4. matrix_affine_scan -- k x k transitions, shapes, correctness, the
   k=2/v=1 reduction to matrix_scan, gradcheck
5. Gradient flow -- non-None, non-zero grads after one backward
"""

from __future__ import annotations

import torch

from mingru import linear_scan, matrix_affine_scan, matrix_scan, parallel_scan_log

SEED = 42


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
