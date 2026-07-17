"""Tests for the four mixers and the block/stack wrappers.

CPU-only. Every mixer runs its eager forward (no Triton on CPU).

Sections
--------
1. MinGRU -- shapes, forward/step equivalence, learnable_h0, h_0 validation
2. SignedMinGRU -- coupled/decoupled shapes, forward/step equivalence
3. RotationMinGRU -- even-hidden constraint, snap on/off, snap STE gradient
4. GivensMinGRU -- construction constraints, orthogonal transition, shapes
5. Time decay -- decay=None bit-identity, fixed decay with delta_t=0,
   the decay/delta_t pairing contract
6. MinGRUBlock / MinGRUStack -- (output, state) contract, list mixers,
   gradient flow
7. Independent numeric oracles -- hand-rolled recurrences that recompute the
   expected output from the layer's linear submodules alone (never _coeffs/
   forward/step), closing the shared-path mutation survivors M5/M7/M8
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from mingru import (
    GivensMinGRU,
    MinGRU,
    MinGRUBlock,
    MinGRUStack,
    RotationMinGRU,
    SignedMinGRU,
)

SEED = 42
B, T, D_IN, D_H = 2, 5, 8, 6


def _step_sequence(mixer, x):
    """Run ``mixer.step`` token-by-token, stacking states to ``(B, T, D)``."""
    h = None
    outs = []
    for t in range(x.size(1)):
        h = mixer.step(x[:, t], h)
        outs.append(h)
    return torch.stack(outs, dim=1)


# ===========================================================================
# 1. MinGRU
# ===========================================================================


class TestMinGRU:
    def test_forward_shape(self):
        torch.manual_seed(SEED)
        layer = MinGRU(D_IN, D_H)
        x = torch.randn(B, T, D_IN)
        assert layer(x).shape == (B, T, D_H)

    def test_step_shape(self):
        torch.manual_seed(SEED)
        layer = MinGRU(D_IN, D_H)
        x_t = torch.randn(B, D_IN)
        assert layer.step(x_t).shape == (B, D_H)

    def test_forward_matches_step(self):
        torch.manual_seed(SEED)
        layer = MinGRU(D_IN, D_H)
        x = torch.randn(B, T, D_IN)
        assert torch.allclose(layer(x), _step_sequence(layer, x), atol=1e-5)

    def test_learnable_h0_used_as_default(self):
        """h0_pre must actually feed forward()'s default h_0, not sit unused.

        Proven two ways: (1) a backward pass gives h0_pre a nonzero gradient,
        so it is on the graph; (2) mutating h0_pre changes the output given
        the same input, so it is not a dead parameter shadowed by a fixed
        default.
        """
        torch.manual_seed(SEED)
        layer = MinGRU(D_IN, D_H, learnable_h0=True)
        assert layer.h0_pre is not None
        x = torch.randn(B, T, D_IN)

        out = layer(x)
        assert out.shape == (B, T, D_H)
        out.sum().backward()
        assert layer.h0_pre.grad is not None
        assert layer.h0_pre.grad.abs().sum() > 0

        with torch.no_grad():
            out_before = layer(x)
            layer.h0_pre.add_(1.0)
            out_after = layer(x)
        assert not torch.allclose(out_before, out_after)

    def test_negative_h0_rejected(self):
        torch.manual_seed(SEED)
        layer = MinGRU(D_IN, D_H)
        x = torch.randn(B, T, D_IN)
        bad_h0 = -torch.ones(B, 1, D_H)
        with pytest.raises(RuntimeError):
            layer(x, h_0=bad_h0)


# ===========================================================================
# 2. SignedMinGRU
# ===========================================================================


class TestSignedMinGRU:
    def test_decoupled_default_shape(self):
        torch.manual_seed(SEED)
        layer = SignedMinGRU(D_IN, D_H)
        assert layer.coupled is False
        x = torch.randn(B, T, D_IN)
        assert layer(x).shape == (B, T, D_H)

    def test_coupled_shape(self):
        torch.manual_seed(SEED)
        layer = SignedMinGRU(D_IN, D_H, coupled=True)
        x = torch.randn(B, T, D_IN)
        assert layer(x).shape == (B, T, D_H)

    def test_forward_matches_step(self):
        torch.manual_seed(SEED)
        layer = SignedMinGRU(D_IN, D_H)
        x = torch.randn(B, T, D_IN)
        assert torch.allclose(layer(x), _step_sequence(layer, x), atol=1e-5)

    def test_coupled_forward_matches_step(self):
        torch.manual_seed(SEED)
        layer = SignedMinGRU(D_IN, D_H, coupled=True)
        x = torch.randn(B, T, D_IN)
        assert torch.allclose(layer(x), _step_sequence(layer, x), atol=1e-5)


# ===========================================================================
# 3. RotationMinGRU
# ===========================================================================


class TestRotationMinGRU:
    def test_odd_hidden_size_rejected(self):
        with pytest.raises(ValueError):
            RotationMinGRU(D_IN, 7)

    def test_forward_shape_snapped(self):
        torch.manual_seed(SEED)
        layer = RotationMinGRU(D_IN, D_H, snap=(2, 3, 4, 6))
        x = torch.randn(B, T, D_IN)
        assert layer(x).shape == (B, T, D_H)

    def test_forward_shape_unsnapped(self):
        torch.manual_seed(SEED)
        layer = RotationMinGRU(D_IN, D_H, snap=None)
        x = torch.randn(B, T, D_IN)
        assert layer(x).shape == (B, T, D_H)

    def test_forward_matches_step(self):
        torch.manual_seed(SEED)
        layer = RotationMinGRU(D_IN, D_H, snap=None)
        x = torch.randn(B, T, D_IN)
        assert torch.allclose(layer(x), _step_sequence(layer, x), atol=1e-5)

    def test_snap_quantizes_angle(self):
        """With snap=(4,), every block's transition angle is a multiple of pi/2."""
        torch.manual_seed(SEED)
        layer = RotationMinGRU(D_IN, D_H, snap=(4,))
        x_t = torch.randn(B, D_IN)
        M, _ = layer._coeffs(x_t)
        # atan2(sin, cos) from M's first column recovers the snapped angle.
        angle = torch.atan2(M[..., 1, 0], M[..., 0, 0])
        step = 2 * math.pi / 4
        remainder = torch.remainder(angle + step / 2, step) - step / 2
        assert remainder.abs().max() < 1e-5

    def test_snap_ste_passes_gradient(self):
        """The straight-through estimator gives a non-zero angle-head grad.

        torch.round alone has zero gradient everywhere; a non-zero grad on
        ``linear_theta`` proves the pre-snap angle carries the gradient.
        """
        torch.manual_seed(SEED)
        layer = RotationMinGRU(D_IN, D_H, snap=(4,))
        x = torch.randn(B, T, D_IN)
        layer(x).sum().backward()
        assert layer.linear_theta.weight.grad is not None
        assert layer.linear_theta.weight.grad.abs().sum() > 0


# ===========================================================================
# 4. GivensMinGRU
# ===========================================================================


class TestGivensMinGRU:
    def test_odd_block_size_rejected(self):
        with pytest.raises(ValueError):
            GivensMinGRU(D_IN, 12, block_size=3)

    def test_hidden_not_multiple_of_block_rejected(self):
        with pytest.raises(ValueError):
            GivensMinGRU(D_IN, 10, block_size=4)

    def test_zero_rounds_rejected(self):
        with pytest.raises(ValueError):
            GivensMinGRU(D_IN, 12, block_size=4, rounds=0)

    def test_forward_shape(self):
        torch.manual_seed(SEED)
        layer = GivensMinGRU(D_IN, 12, block_size=6, rounds=3)
        x = torch.randn(B, T, D_IN)
        assert layer(x).shape == (B, T, 12)

    def test_transition_is_special_orthogonal(self):
        """Each per-block transition (decay off) is orthogonal with det +1."""
        torch.manual_seed(SEED)
        layer = GivensMinGRU(D_IN, 12, block_size=6, rounds=3)
        M, _ = layer._coeffs(torch.randn(B, D_IN))
        eye = torch.eye(6)
        assert torch.allclose(M @ M.transpose(-1, -2), eye.expand_as(M), atol=1e-5)
        assert torch.allclose(torch.linalg.det(M), torch.ones_like(M[..., 0, 0]), atol=1e-4)

    def test_forward_matches_step(self):
        torch.manual_seed(SEED)
        layer = GivensMinGRU(D_IN, 12, block_size=6, rounds=2)
        x = torch.randn(B, T, D_IN)
        assert torch.allclose(layer(x), _step_sequence(layer, x), atol=1e-5)


# ===========================================================================
# 5. Time decay
# ===========================================================================


class TestDecay:
    def test_decay_none_is_default(self):
        torch.manual_seed(SEED)
        layer = MinGRU(D_IN, D_H)
        assert layer.decay is None

    def test_fixed_decay_with_zero_delta_matches_no_decay(self):
        """gamma = exp(0) = 1, so delta_t=0 must reproduce the no-decay output.

        _init_decay draws no RNG and runs last, so both modules share weights
        under the same seed -- the outputs must match bit-for-bit.
        """
        torch.manual_seed(SEED)
        decayed = MinGRU(D_IN, D_H, decay="fixed", decay_rate=1.0)
        torch.manual_seed(SEED)
        plain = MinGRU(D_IN, D_H)
        x = torch.randn(B, T, D_IN)
        delta_t = torch.zeros(B, T)
        assert torch.equal(decayed(x, delta_t=delta_t), plain(x))

    def test_positive_delta_changes_output(self):
        torch.manual_seed(SEED)
        layer = MinGRU(D_IN, D_H, decay="fixed", decay_rate=1.0)
        x = torch.randn(B, T, D_IN)
        out_zero = layer(x, delta_t=torch.zeros(B, T))
        out_gap = layer(x, delta_t=torch.ones(B, T))
        assert not torch.allclose(out_zero, out_gap)

    def test_learnable_decay_registers_rho(self):
        torch.manual_seed(SEED)
        layer = SignedMinGRU(D_IN, D_H, decay="learnable", decay_rate=1.0)
        assert layer.rho.shape == (D_H,)
        assert layer.rho.requires_grad

    def test_decay_enabled_requires_delta_t(self):
        torch.manual_seed(SEED)
        layer = MinGRU(D_IN, D_H, decay="fixed")
        with pytest.raises(ValueError):
            layer(torch.randn(B, T, D_IN))

    def test_delta_t_without_decay_rejected(self):
        torch.manual_seed(SEED)
        layer = MinGRU(D_IN, D_H)
        with pytest.raises(ValueError):
            layer(torch.randn(B, T, D_IN), delta_t=torch.zeros(B, T))

    def test_invalid_decay_mode_rejected(self):
        with pytest.raises(ValueError):
            MinGRU(D_IN, D_H, decay="sometimes")


# ===========================================================================
# 6. MinGRUBlock / MinGRUStack
# ===========================================================================


class TestBlockAndStack:
    def test_block_returns_output_and_state(self):
        torch.manual_seed(SEED)
        block = MinGRUBlock(d_model=D_H, mixer="signed")
        x = torch.randn(B, T, D_H)
        out, state = block(x)
        assert out.shape == (B, T, D_H)
        assert state.shape == (B, 1, D_H)

    def test_unknown_mixer_rejected(self):
        with pytest.raises(ValueError):
            MinGRUBlock(d_model=D_H, mixer="nope")

    def test_stack_shapes_and_state_length(self):
        torch.manual_seed(SEED)
        model = MinGRUStack(input_size=D_IN, d_model=D_H, n_layers=3, mixer="log")
        x = torch.randn(B, T, D_IN)
        out, state = model(x)
        assert out.shape == (B, T, D_H)
        assert len(state) == 3

    def test_stack_list_mixer(self):
        torch.manual_seed(SEED)
        model = MinGRUStack(
            input_size=D_IN,
            d_model=D_H,
            n_layers=2,
            mixer=["log", "signed"],
        )
        x = torch.randn(B, T, D_IN)
        out, state = model(x)
        assert out.shape == (B, T, D_H)
        assert len(state) == 2

    def test_stack_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            MinGRUStack(
                input_size=D_IN,
                d_model=D_H,
                n_layers=3,
                mixer=["log", "signed"],
            )

    def test_multi_rotation_stack_warns_once(self):
        """pytest.warns alone only proves >=1; count records to enforce "once"."""
        torch.manual_seed(SEED)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            MinGRUStack(
                input_size=D_IN,
                d_model=D_H,
                n_layers=2,
                mixer=["rotation", "rotation"],
            )
        user_warnings = [w for w in records if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1

    def test_stack_forward_matches_step(self):
        torch.manual_seed(SEED)
        model = MinGRUStack(input_size=D_IN, d_model=D_H, n_layers=2, mixer="log")
        x = torch.randn(B, T, D_IN)
        out_par, _ = model(x)
        state = model.init_state()
        outs = []
        for t in range(T):
            y, state = model.step(x[:, t], state)
            outs.append(y)
        out_seq = torch.stack(outs, dim=1)
        assert torch.allclose(out_par, out_seq, atol=1e-5)

    def test_stack_gradient_flow(self):
        torch.manual_seed(SEED)
        model = MinGRUStack(input_size=D_IN, d_model=D_H, n_layers=2, mixer="signed")
        x = torch.randn(B, T, D_IN)
        out, _ = model(x)
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


# ===========================================================================
# 7. Independent numeric oracles (mutation-hardening)
#
# The forward-vs-step tests in sections 1-4 assert `forward == step`, but both
# sides call the mixer's own `_coeffs`. A bug in code SHARED by both paths --
# the signed eigenvalue's sign, the decay direction -- moves both outputs
# together and stays invisible to an equivalence check. These tests instead
# recompute the expected output from a hand-rolled recurrence built only from
# the layer's `nn.Linear` submodules and the documented math in the class
# docstrings; they never call `_coeffs`/`forward`/`step` to form the
# expectation, so they see the shared-path bugs the equivalence tests cannot.
#
# Each closes one survivor from the 2026-07-17 mutation audit
# (.claude/output/reviews/2026-07-17-test-suite-audit.md):
#   M8 -- SignedMinGRU._coeffs decoupled eigenvalue sign (tanh_s -> -tanh_s)
#   M5 -- DecayMixin._decay_gamma decay direction (exp(-lambda*dt) -> exp(+.))
#   M7 -- MinGRU.forward log-additive decay direction (-lambda*dt -> +lambda*dt)
# ===========================================================================


def _g(x: torch.Tensor) -> torch.Tensor:
    """Reimplementation of ``min_gru.g`` (Appendix-B activation).

    Kept independent of the module under test so the MinGRU oracle below
    shares no code with the implementation it validates.
    """
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def _ref_signed_decoupled(
    layer, x: torch.Tensor, h_0: torch.Tensor, delta_t: torch.Tensor | None = None
) -> torch.Tensor:
    """Hand-rolled decoupled ``SignedMinGRU`` recurrence ``h_t = a_t h_{t-1} + b_t``.

    From the class docstring's documented math (decoupled form):
    ``a_t = tanh(linear_s(x_t))``, optionally ``* exp(-lambda * delta_t)``
    for fixed decay; ``b_t = sigmoid(linear_z(x_t)) * linear_h(x_t)``. Uses
    only the layer's ``nn.Linear`` submodules -- never ``_coeffs``/``forward``/
    ``step`` -- so the sign of ``a_t`` and the decay direction are checked
    against an oracle that cannot inherit either bug.
    """
    B, T, _ = x.shape
    a = torch.tanh(layer.linear_s(x))
    if delta_t is not None:
        a = a * torch.exp(-layer.decay_rate * delta_t).unsqueeze(-1)
    b = torch.sigmoid(layer.linear_z(x)) * layer.linear_h(x)
    h = h_0.reshape(B, layer.hidden_size)
    outs = []
    for t in range(T):
        h = a[:, t] * h + b[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


def _ref_mingru_forward(
    layer, x: torch.Tensor, delta_t: torch.Tensor | None = None
) -> torch.Tensor:
    """Hand-rolled log-space ``MinGRU`` recurrence, evaluated in linear space.

    ``parallel_scan_log`` solves ``h_t = a_t h_{t-1} + b_t`` with, from the
    forward's documented math, ``a_t = (1 - sigmoid(linear_z(x_t)))`` scaled
    by ``exp(-lambda * delta_t)`` under fixed decay (the log-additive term),
    ``b_t = sigmoid(linear_z(x_t)) * g(linear_h(x_t))``, and default
    ``h_0 = g(0) = 0.5``. Recomputed here in plain linear space with a local
    ``g`` -- shares no code with the log-space implementation.
    """
    B, T, _ = x.shape
    z = torch.sigmoid(layer.linear_z(x))
    a = 1 - z
    if delta_t is not None:
        a = a * torch.exp(-layer.decay_rate * delta_t).unsqueeze(-1)
    b = z * _g(layer.linear_h(x))
    h = torch.full((B, layer.hidden_size), 0.5)
    outs = []
    for t in range(T):
        h = a[:, t] * h + b[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1)


class TestMixerOracles:
    # Tolerance rationale (all three tests): the parallel scans (linear_scan,
    # parallel_scan_log) reorder the recurrence's additions under Hillis-Steele
    # doubling / logcumsumexp, so they differ from the sequential oracle only by
    # float32 rounding (order 1e-6 at these tiny sizes). atol=1e-5 covers that
    # margin; a sign- or direction-flip mutation shifts affected terms by O(1)
    # (a doubled contribution, or exp(+x) vs exp(-x)), orders of magnitude above
    # the tolerance, so the oracle stays a sharp discriminator.

    def test_signed_decoupled_matches_hand_oracle(self):
        """Kills M8: negating ``tanh_s`` flips ``a_t``'s sign vs the oracle.

        No decay, so ``_decay_gamma`` (M5) is not on this path -- this isolates
        the eigenvalue sign. ``h_0`` is nonzero and ``T >= 2`` so the
        ``a_t * h_{t-1}`` term (the only place the eigenvalue enters) materially
        drives the output; a flipped sign diverges by O(|a_t * h_{t-1}|).
        """
        torch.manual_seed(SEED)
        layer = SignedMinGRU(D_IN, D_H)  # decoupled default
        x = torch.randn(B, T, D_IN)
        h_0 = torch.randn(B, 1, D_H)
        ref = _ref_signed_decoupled(layer, x, h_0)
        assert torch.allclose(layer(x, h_0=h_0), ref, atol=1e-5)

    def test_signed_fixed_decay_matches_analytic_gamma(self):
        """Kills M5: ``_decay_gamma`` must contract by ``exp(-lambda*dt)``.

        A fixed ``decay_rate = 0.5 < 1`` with a known positive ``delta_t = 2``
        makes ``gamma = exp(-1) ~ 0.368`` analytically checkable through the
        ``_coeffs`` -> ``_decay_gamma`` site (shared with ``MinGRU.step`` and
        the rotation mixers). Inverting the sign to ``exp(+1) ~ 2.718``
        amplifies ``a_t`` and diverges from the oracle. ``h_0`` is nonzero so
        the decayed ``a_t * h_{t-1}`` term is exercised.
        """
        torch.manual_seed(SEED)
        layer = SignedMinGRU(D_IN, D_H, decay="fixed", decay_rate=0.5)
        x = torch.randn(B, T, D_IN)
        h_0 = torch.randn(B, 1, D_H)
        delta_t = torch.full((B, T), 2.0)
        ref = _ref_signed_decoupled(layer, x, h_0, delta_t=delta_t)
        assert torch.allclose(layer(x, h_0=h_0, delta_t=delta_t), ref, atol=1e-5)

    def test_mingru_forward_fixed_decay_matches_analytic_gamma(self):
        """Kills M7: the log-additive decay must subtract ``lambda*dt``.

        ``MinGRU.forward`` folds decay as ``log_coeffs - lambda*dt`` (a separate
        code path from ``_decay_gamma``), so ``a_t = (1 - z_t) * exp(-lambda*dt)``.
        A fixed ``decay_rate = 0.5`` with ``delta_t = 2`` gives an analytic
        ``exp(-1)`` contraction; flipping the sign to ``+ lambda*dt`` amplifies
        ``a_t`` and diverges from the linear-space oracle.
        """
        torch.manual_seed(SEED)
        layer = MinGRU(D_IN, D_H, decay="fixed", decay_rate=0.5)
        x = torch.randn(B, T, D_IN)
        delta_t = torch.full((B, T), 2.0)
        ref = _ref_mingru_forward(layer, x, delta_t=delta_t)
        assert torch.allclose(layer(x, delta_t=delta_t), ref, atol=1e-5)
