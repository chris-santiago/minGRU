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
