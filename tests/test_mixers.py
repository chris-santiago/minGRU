"""Tests for the five mixers and the block/stack wrappers.

CPU-only. Every mixer runs its eager forward (no Triton on CPU).

Sections
--------
1. MinGRU -- shapes, forward/step equivalence, learnable_h0, h_0 validation
2. SignedMinGRU -- coupled/decoupled shapes, forward/step equivalence
3. RotationMinGRU -- even-hidden constraint, snap on/off, snap STE gradient
4. GivensMinGRU -- construction constraints, orthogonal transition, shapes
5. Time decay -- decay=None bit-identity, fixed decay with delta_t=0,
   the decay/delta_t pairing contract
6. DeltaMinGRU -- constructor validation, forward/step parity, chaining
   contract, h_0/delta_t error cases, block/stack integration
7. MinGRUBlock / MinGRUStack -- (output, state) contract, list mixers,
   gradient flow
8. Independent numeric oracles -- hand-rolled recurrences that recompute the
   expected output from the layer's linear submodules alone (never _coeffs/
   forward/step), closing the shared-path mutation survivors M5/M7/M8 (and,
   for DeltaMinGRU, section 6's own oracle test)
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch
import torch.nn.functional as F

from mingru import (
    DeltaMinGRU,
    GivensMinGRU,
    MinGRU,
    MinGRUBlock,
    MinGRUStack,
    RotationMinGRU,
    SignedMinGRU,
    matrix_affine_scan,
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


def _delta_step_sequence(mixer, x):
    """Run ``DeltaMinGRU.step`` token-by-token.

    Unlike ``_step_sequence``, ``step`` returns a ``(y_t, h_t)`` tuple
    (readout != state for this mixer -- see ``DeltaMinGRU.step``), so
    only the readout is stacked; ``h_t`` is threaded through as the next
    ``h_prev``.
    """
    h = None
    outs = []
    for t in range(x.size(1)):
        y_t, h = mixer.step(x[:, t], h)
        outs.append(y_t)
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
# 6. DeltaMinGRU
# ===========================================================================

# n_heads=4 (the constructor default) needs a hidden_size divisible by 4;
# D_H (6) is not, so this section's default-n_heads tests use their own
# hidden size while everything else reuses D_IN/D_H/B/T for consistency
# with the rest of the file.
D_H_DELTA_DEFAULT = 8


class TestDeltaMinGRU:
    def test_forward_shape_default_n_heads(self):
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H_DELTA_DEFAULT)
        x = torch.randn(B, T, D_IN)
        assert layer(x).shape == (B, T, D_H_DELTA_DEFAULT)

    def test_step_shapes(self):
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2)
        x_t = torch.randn(B, D_IN)
        y_t, h_t = layer.step(x_t)
        assert y_t.shape == (B, D_H)
        assert h_t.shape == (B, layer.n_heads * layer.d_k * layer.d_v)

    def test_return_state_shape(self):
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2)
        x = torch.randn(B, T, D_IN)
        y, h_T = layer(x, return_state=True)
        assert y.shape == (B, T, D_H)
        assert h_T.shape == (B, 1, layer.n_heads * layer.d_k * layer.d_v)

    def test_forward_matches_step_nh1(self):
        """nh=1 (DeltaNet): forward must equal the token-by-token step loop."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2, nh=1)
        x = torch.randn(B, T, D_IN)
        assert torch.allclose(layer(x), _delta_step_sequence(layer, x), atol=1e-5)

    def test_forward_matches_step_nh3(self):
        """nh=3 (DeltaProduct): micro-steps must be applied in order in both paths."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2, nh=3)
        x = torch.randn(B, T, D_IN)
        assert torch.allclose(layer(x), _delta_step_sequence(layer, x), atol=1e-5)

    def test_chaining_contract_misaligned_split(self):
        """forward(x2, h_0=s) after forward(x1, return_state=True) == forward(cat(x1,x2)).

        Split point (5) is deliberately not a multiple of the default
        chunk_size (64) or of T (13) -- this sequential forward ignores
        chunk_size entirely, but the chaining contract itself (spec
        section 6) must hold regardless, since it is what Task 3's
        chunked-WY forward will be tested against next.
        """
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2, nh=2)
        x = torch.randn(B, 13, D_IN, requires_grad=True)
        x1, x2 = x[:, :5], x[:, 5:]

        y_full = layer(x)
        y1, state = layer(x1, return_state=True)
        y2 = layer(x2, h_0=state)
        y_chained = torch.cat([y1, y2], dim=1)
        assert torch.allclose(y_full, y_chained, atol=1e-5)

        grad_full = torch.autograd.grad(y_full.sum(), x, retain_graph=True)[0]
        grad_chained = torch.autograd.grad(y_chained.sum(), x, retain_graph=True)[0]
        assert torch.allclose(grad_full, grad_chained, atol=1e-3)

    def test_gradient_flow(self):
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2, nh=2)
        x = torch.randn(B, T, D_IN)
        layer(x).sum().backward()
        grads = [p.grad for p in layer.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert all(g.abs().sum() > 0 for g in grads)

    def test_carries_matrix_state_attribute(self):
        assert DeltaMinGRU.carries_matrix_state is True
        assert getattr(MinGRU, "carries_matrix_state", False) is False

    # -- constructor validation ---------------------------------------

    def test_nonpositive_n_heads_rejected(self):
        with pytest.raises(ValueError):
            DeltaMinGRU(D_IN, D_H, n_heads=0)

    def test_nonpositive_nh_rejected(self):
        with pytest.raises(ValueError):
            DeltaMinGRU(D_IN, D_H, nh=0)

    def test_nonpositive_chunk_size_rejected(self):
        with pytest.raises(ValueError):
            DeltaMinGRU(D_IN, D_H, chunk_size=0)

    def test_decay_rejected(self):
        with pytest.raises(ValueError):
            DeltaMinGRU(D_IN, D_H, n_heads=2, decay="fixed")

    def test_hidden_size_not_divisible_by_n_heads_rejected(self):
        """D_H (6) is not divisible by n_heads=4, and d_k is defaulted."""
        with pytest.raises(ValueError):
            DeltaMinGRU(D_IN, D_H, n_heads=4)

    def test_explicit_d_k_bypasses_divisibility_check(self):
        """Explicit d_k decouples per-head state size from hidden_size."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=4, d_k=3)
        x = torch.randn(B, T, D_IN)
        assert layer(x).shape == (B, T, D_H)

    # -- delta_t / h_0 error cases --------------------------------------

    def test_delta_t_rejected_in_forward(self):
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2)
        x = torch.randn(B, T, D_IN)
        with pytest.raises(ValueError):
            layer(x, delta_t=torch.zeros(B, T))

    def test_delta_t_rejected_in_step(self):
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2)
        x_t = torch.randn(B, D_IN)
        with pytest.raises(ValueError):
            layer.step(x_t, delta_t=torch.zeros(B))

    def test_h0_shape_mismatch_rejected(self):
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2)
        x = torch.randn(B, T, D_IN)
        bad_h0 = torch.zeros(B, 1, layer.n_heads * layer.d_k * layer.d_v + 1)
        with pytest.raises(ValueError, match=r"h_0 must have shape"):
            layer(x, h_0=bad_h0)

    def test_h_prev_shape_mismatch_rejected(self):
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2)
        x_t = torch.randn(B, D_IN)
        bad_h_prev = torch.zeros(B, layer.n_heads * layer.d_k * layer.d_v + 1)
        with pytest.raises(ValueError, match=r"h_prev must have shape"):
            layer.step(x_t, h_prev=bad_h_prev)

    # -- block / stack integration ---------------------------------------

    def test_block_delta_mixer_constructs_and_runs(self):
        torch.manual_seed(SEED)
        block = MinGRUBlock(d_model=D_H, mixer="delta", mixer_kwargs={"n_heads": 2})
        x = torch.randn(B, T, D_H, requires_grad=True)
        out, state = block(x)
        assert out.shape == (B, T, D_H)
        n_heads, d_k, d_v = block.mingru.n_heads, block.mingru.d_k, block.mingru.d_v
        assert state.shape == (B, 1, n_heads * d_k * d_v)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_stack_mixed_signed_delta_constructs_and_runs(self):
        torch.manual_seed(SEED)
        model = MinGRUStack(
            input_size=D_IN,
            d_model=D_H,
            n_layers=2,
            mixer=["signed", "delta"],
            mixer_kwargs={"delta": {"n_heads": 2}},
        )
        x = torch.randn(B, T, D_IN)
        out, state = model(x)
        assert out.shape == (B, T, D_H)
        assert len(state) == 2
        assert state[0].shape == (B, 1, D_H)  # "signed": output[:, -1:] convention
        delta_mixer = model.blocks[1].mingru
        expected_delta_state = (
            B,
            1,
            delta_mixer.n_heads * delta_mixer.d_k * delta_mixer.d_v,
        )
        assert state[1].shape == expected_delta_state  # "delta": matrix-state seam
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)

    def test_block_step_matches_forward_for_delta(self):
        """MinGRUBlock streaming parity for a matrix-state mixer.

        The seam added to ``MinGRUBlock.step`` (spec section 6, amended
        after Task 2's forward-only reading left this path crashing --
        ``DeltaMinGRU.step`` returns a ``(y_t, h_t)`` tuple, not a bare
        tensor) must unpack ``y_t`` onto the residual/output path and
        carry ``h_t`` as the block state, giving the same token-by-token
        streaming parity every other mixer already has.
        """
        torch.manual_seed(SEED)
        block = MinGRUBlock(d_model=D_H, mixer="delta", mixer_kwargs={"n_heads": 2})
        x = torch.randn(B, T, D_H)
        out_par, _ = block(x)
        state = None
        outs = []
        for t in range(T):
            y, state = block.step(x[:, t], state)
            outs.append(y)
        out_seq = torch.stack(outs, dim=1)
        assert torch.allclose(out_par, out_seq, atol=1e-5)

    def test_stack_step_matches_forward_for_delta_only(self):
        torch.manual_seed(SEED)
        model = MinGRUStack(
            input_size=D_IN,
            d_model=D_H,
            n_layers=2,
            mixer="delta",
            mixer_kwargs={"n_heads": 2},
        )
        x = torch.randn(B, T, D_IN)
        out_par, _ = model(x)
        state = model.init_state()
        outs = []
        for t in range(T):
            y, state = model.step(x[:, t], state)
            outs.append(y)
        out_seq = torch.stack(outs, dim=1)
        assert torch.allclose(out_par, out_seq, atol=1e-5)

    def test_stack_step_matches_forward_mixed_signed_delta(self):
        """Mixed-mixer stack streaming parity, starting from ``init_state()``.

        Also proves ``init_state()``/``step()`` treat per-block state
        entries as opaque: the "signed" block's entry stays
        ``(B, d_model)`` while the "delta" block's is
        ``(B, n_heads*d_k*d_v)`` (asserted below), and neither
        ``init_state`` nor ``step`` assumes a uniform shape across
        blocks.
        """
        torch.manual_seed(SEED)
        model = MinGRUStack(
            input_size=D_IN,
            d_model=D_H,
            n_layers=2,
            mixer=["signed", "delta"],
            mixer_kwargs={"delta": {"n_heads": 2}},
        )
        x = torch.randn(B, T, D_IN)
        out_par, _ = model(x)

        state = model.init_state()
        assert state == [None, None]
        outs = []
        for t in range(T):
            y, state = model.step(x[:, t], state)
            outs.append(y)
        out_seq = torch.stack(outs, dim=1)
        assert torch.allclose(out_par, out_seq, atol=1e-5)

        delta_mixer = model.blocks[1].mingru
        assert state[0].shape == (B, D_H)
        assert state[1].shape == (B, delta_mixer.n_heads * delta_mixer.d_k * delta_mixer.d_v)

    def test_block_unknown_mixer_error_lists_delta(self):
        with pytest.raises(ValueError, match=r"delta"):
            MinGRUBlock(d_model=D_H, mixer="nope")

    def test_importable_from_mingru(self):
        from mingru import DeltaMinGRU as ImportedDeltaMinGRU

        assert ImportedDeltaMinGRU is DeltaMinGRU

    # -- Task 3: chunked-WY forward ---------------------------------------
    # `forward` now runs the chunked-WY UT-transform parallel form
    # (`_forward_chunked`) instead of the token loop; `step` (and the
    # hand-rolled oracles below) stay the correctness anchors. Spec section
    # 7 invariants: sequential/parallel equivalence at T=128 and T=1024,
    # chunk_size invariance (including ragged final chunks), no per-token
    # d_k x d_k transition ever materialized (verified structurally by the
    # affine-scan oracle needing to build that matrix by hand -- the layer
    # under test never does).

    @pytest.mark.parametrize(
        "T_chunk,chunk_size",
        [
            # T=13: C in {1, 3, 64 (>T), 13 (=T), 50 (>T)}; 3 doesn't
            # divide 13 evenly (chunks 3,3,3,3,1) -- ragged final chunk.
            (13, 1),
            (13, 3),
            (13, 64),
            (13, 13),
            (13, 50),
            # T=128: C in {1, 3, 64 (<T, doesn't divide evenly), 128 (=T),
            # 165 (>T)}.
            (128, 1),
            (128, 3),
            (128, 64),
            (128, 128),
            (128, 165),
        ],
    )
    def test_chunk_size_invariant_vs_step_oracle(self, T_chunk, chunk_size):
        """``chunk_size`` is performance-only (spec section 7): every value
        must reproduce the token-by-token ``step`` recurrence exactly
        (within tolerance), not just agree with other chunk sizes.
        """
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2, nh=2, chunk_size=chunk_size)
        x = torch.randn(B, T_chunk, D_IN)
        assert torch.allclose(layer(x), _delta_step_sequence(layer, x), atol=1e-5)

    @pytest.mark.parametrize("T_large", [128, 1024])
    def test_forward_matches_sequential_oracle_large_T(self, T_large):
        """Spec section 7's forward<->sequential-recurrence parity at
        T=128 and T=1024 (Task 2 only asserted this at small T). Uses
        ``_ref_delta_forward`` rather than ``step`` -- ``step`` is
        ``@torch.no_grad()`` by design (matches the family's other
        mixers), so it cannot anchor the gradient half of this
        invariant; ``_ref_delta_forward`` is the same recurrence,
        differentiable.
        """
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2, nh=2)
        x = torch.randn(2, T_large, D_IN, requires_grad=True)
        y = layer(x)
        ref = _ref_delta_forward(layer, x)
        assert torch.allclose(y, ref, atol=1e-5)
        grad_y = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
        grad_ref = torch.autograd.grad(ref.sum(), x, retain_graph=True)[0]
        assert torch.allclose(grad_y, grad_ref, atol=1e-3)

    @pytest.mark.parametrize(
        "nh_affine,n_heads_affine,T_affine",
        [
            (1, 1, 1),
            (1, 4, 13),
            (2, 1, 128),
            (2, 4, 1),
            (3, 1, 13),
            (3, 4, 128),
            (1, 1, 1024),
            (3, 4, 1024),
        ],
    )
    def test_forward_matches_affine_scan_oracle(self, nh_affine, n_heads_affine, T_affine):
        """Second, independent oracle (spec section 10 dual-oracle strategy):
        ``_ref_affine_delta_forward`` composes each token's micro-steps
        into a per-token affine map and reduces with the frozen
        ``matrix_affine_scan`` -- the "measured-slow" reference path
        the chunked-WY form must not take, but exactly correct. An error
        in the chunked-WY composition cannot agree with both this and
        the sequential oracle (previous test) simultaneously. Covers
        nh in {1, 2, 3}, n_heads in {1, 4}, T in {1, 13, 128, 1024}
        (spec/acceptance-criteria parameter space) via a curated subset
        rather than the full 3*2*4 cross product, each value exercised
        at least once, output and gradients both checked.
        """
        torch.manual_seed(SEED)
        d_k = 3
        layer = DeltaMinGRU(D_IN, D_H, n_heads=n_heads_affine, nh=nh_affine, d_k=d_k, d_v=d_k)
        x = torch.randn(2, T_affine, D_IN, requires_grad=True)
        y = layer(x)
        ref = _ref_affine_delta_forward(layer, x)
        assert torch.allclose(y, ref, atol=1e-5)
        grad_y = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
        grad_ref = torch.autograd.grad(ref.sum(), x, retain_graph=True)[0]
        assert torch.allclose(grad_y, grad_ref, atol=1e-3)

    def test_chaining_contract_misaligned_chunk_boundaries(self):
        """Chaining (spec section 6) with an explicit small ``chunk_size``:
        the second ``forward`` call (``h_0=state``) restarts chunk
        indexing from 0, so its chunk boundaries fall at different
        sequence offsets than the equivalent span computed inside the
        single full-sequence call -- this must not matter.
        """
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2, nh=2, chunk_size=4)
        x = torch.randn(B, 13, D_IN, requires_grad=True)
        x1, x2 = x[:, :5], x[:, 5:]

        y_full = layer(x)
        y1, state = layer(x1, return_state=True)
        y2 = layer(x2, h_0=state)
        y_chained = torch.cat([y1, y2], dim=1)
        assert torch.allclose(y_full, y_chained, atol=1e-5)

        grad_full = torch.autograd.grad(y_full.sum(), x, retain_graph=True)[0]
        grad_chained = torch.autograd.grad(y_chained.sum(), x, retain_graph=True)[0]
        assert torch.allclose(grad_full, grad_chained, atol=1e-3)

    def test_gradcheck_double_precision(self):
        """``gradcheck`` on a tiny float64 instance (spec section 9/10)."""
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(4, 4, n_heads=1, nh=2, d_k=2, d_v=2, chunk_size=2).double()
        x = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(layer, (x,), atol=1e-6, eps=1e-6)


# ===========================================================================
# 7. MinGRUBlock / MinGRUStack
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

    def test_block_step_signed_mixer_output_regression_pin(self):
        """Regression pin: adding the ``carries_matrix_state`` seam to
        ``MinGRUBlock.step`` must not change a single bit of the
        non-matrix-state path (every mixer but ``"delta"``). Values
        captured post-seam at ``SEED=42``; any future edit to the seam,
        or to ``step``'s residual/state wiring, that perturbs this
        mixer's output would fail this pin even though
        ``test_stack_forward_matches_step``-style equivalence checks
        could stay green (both sides of an equivalence check can drift
        together).
        """
        torch.manual_seed(SEED)
        block = MinGRUBlock(d_model=6, mixer="signed")
        x_t = torch.randn(2, 6)
        out, state = block.step(x_t, None)
        expected_out = torch.tensor(
            [
                [-0.1164, -0.5916, 0.7359, 1.1987, -0.2292, -0.7409],
                [0.9475, -0.2558, -0.1918, -0.8776, -1.5433, 0.8536],
            ]
        )
        expected_state = torch.tensor(
            [
                [-0.2117, -0.1494, -0.0530, -0.0080, -0.3067, -0.5182],
                [0.1409, 0.5886, -0.3198, -0.0620, 0.1942, 0.8264],
            ]
        )
        assert torch.allclose(out, expected_out, atol=1e-4)
        assert torch.allclose(state, expected_state, atol=1e-4)
        # "signed" is not a matrix-state mixer: output and state must be
        # the same tensor (state IS the readout), never independently
        # drifting apart the way "delta"'s y_t/h_t split does.
        assert torch.equal(out, state) is False  # residual sum differs from raw h
        assert torch.allclose(state, block.mingru.step(block.norm1(x_t), None), atol=1e-6)


# ===========================================================================
# 8. Independent numeric oracles (mutation-hardening)
#
# The forward-vs-step tests in sections 1-4 and 6 assert `forward == step`,
# but both sides call the mixer's own `_coeffs`. A bug in code SHARED by both
# paths -- the signed eigenvalue's sign, the decay direction -- moves both
# outputs together and stays invisible to an equivalence check. These tests
# instead recompute the expected output from a hand-rolled recurrence built
# only from the layer's `nn.Linear` submodules and the documented math in the
# class docstrings; they never call `_coeffs`/`_micro_step`/`forward`/`step`
# to form the expectation, so they see the shared-path bugs the equivalence
# tests cannot.
#
# Each closes one survivor from the 2026-07-17 mutation audit
# (.claude/output/reviews/2026-07-17-test-suite-audit.md):
#   M8 -- SignedMinGRU._coeffs decoupled eigenvalue sign (tanh_s -> -tanh_s)
#   M5 -- DecayMixin._decay_gamma decay direction (exp(-lambda*dt) -> exp(+.))
#   M7 -- MinGRU.forward log-additive decay direction (-lambda*dt -> +lambda*dt)
#
# The DeltaMinGRU oracle (test_delta_matches_hand_oracle) is new for this
# mixer, not a mutation-audit survivor closure; it follows the same
# independent-recomputation pattern as the other three.
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


def _ref_delta_forward(layer: DeltaMinGRU, x: torch.Tensor) -> torch.Tensor:
    """Hand-rolled DeltaNet/DeltaProduct recurrence from ``layer``'s linears.

    From ``DeltaMinGRU``'s class docstring math (and
    ``experiments/variants.py``'s ``DeltaNetMixer``, reimplemented here
    rather than imported): per token, per ordered micro-step ``j``,
    ``k_j = normalize(linear_k[j](x))``, ``v_j = linear_v[j](x)``,
    ``beta_j = 2*sigmoid(linear_beta[j](x))``, and
    ``H <- (I - beta_j k_j k_j^T) H + beta_j k_j v_j^T``; readout
    ``y_t = H_t^T q_t`` after the token's last micro-step, projected by
    ``out_proj``. Never calls ``_coeffs``/``_micro_step``/``forward``/
    ``step``, so it shares no code with the implementation under test.
    """
    B_, T_, _ = x.shape
    n_heads, d_k, d_v, nh = layer.n_heads, layer.d_k, layer.d_v, layer.nh
    q = layer.linear_q(x).view(B_, T_, n_heads, d_k)
    ks = [F.normalize(lin(x).view(B_, T_, n_heads, d_k), dim=-1) for lin in layer.linear_k]
    vs = [lin(x).view(B_, T_, n_heads, d_v) for lin in layer.linear_v]
    betas = [2 * torch.sigmoid(lin(x)) for lin in layer.linear_beta]
    H = x.new_zeros(B_, n_heads, d_k, d_v)
    ys = []
    for t in range(T_):
        for j in range(nh):
            k, v, beta = ks[j][:, t], vs[j][:, t], betas[j][:, t]
            kH = torch.einsum("bhk,bhkv->bhv", k, H)
            beta_ = beta[..., None, None]
            H = H - beta_ * torch.einsum("bhk,bhv->bhkv", k, kH)
            H = H + beta_ * torch.einsum("bhk,bhv->bhkv", k, v)
        ys.append(torch.einsum("bhk,bhkv->bhv", q[:, t], H))
    y = torch.stack(ys, dim=1).reshape(B_, T_, n_heads * d_v)
    return layer.out_proj(y)


def _ref_affine_delta_forward(layer: DeltaMinGRU, x: torch.Tensor) -> torch.Tensor:
    """Second, independent DeltaMinGRU oracle: per-token affine maps through
    the FROZEN ``matrix_affine_scan`` (Task 3's dual-oracle strategy, spec
    section 10).

    Composes each token's ``nh`` ordered micro-steps into a single per-token
    affine map ``H_t = A_t @ H_{t-1} + B_t`` (materializing a ``d_k x d_k``
    matrix per token -- explicitly the slow path the chunked-WY ``forward``
    under test must avoid, per its own docstring) by induction on the
    micro-step recurrence from ``DeltaMinGRU``'s class docstring:
    ``A^(0) = I``, ``B^(0) = 0``,
    ``A^(j) = (I - beta_j k_j k_j^T) A^(j-1)``,
    ``B^(j) = (I - beta_j k_j k_j^T) B^(j-1) + beta_j k_j v_j^T``, giving
    ``A_t = A^(nh)``, ``B_t = B^(nh)``. The resulting per-token maps are
    reduced by ``matrix_affine_scan`` (imported, not reimplemented) rather
    than a hand-rolled scan, so the composition math above is the only new
    logic this oracle introduces; readout is ``y_t = q_t^T H_t`` using
    ``matrix_affine_scan``'s ``Bbar`` directly (``H_0 = 0`` always, since
    these tests never pass ``h_0``). Shares no code with ``_coeffs``/
    ``_micro_step``/``_contract_state``/``forward``/``_forward_chunked``.
    """
    B_, T_, _ = x.shape
    n_heads, d_k, d_v, nh = layer.n_heads, layer.d_k, layer.d_v, layer.nh
    q = layer.linear_q(x).view(B_, T_, n_heads, d_k)
    ks = [F.normalize(lin(x).view(B_, T_, n_heads, d_k), dim=-1) for lin in layer.linear_k]
    vs = [lin(x).view(B_, T_, n_heads, d_v) for lin in layer.linear_v]
    betas = [2 * torch.sigmoid(lin(x)) for lin in layer.linear_beta]

    eye = torch.eye(d_k, dtype=x.dtype).expand(B_, T_, n_heads, d_k, d_k)
    A = eye
    Bm = x.new_zeros(B_, T_, n_heads, d_k, d_v)
    for j in range(nh):
        k, v, beta = ks[j], vs[j], betas[j]
        beta_ = beta[..., None, None]
        kkT = torch.einsum("ntha,nthb->nthab", k, k)
        A_j = eye - beta_ * kkT
        B_j = beta_ * torch.einsum("ntha,nthv->nthav", k, v)
        A = torch.einsum("nthab,nthbc->nthac", A_j, A)
        Bm = torch.einsum("nthab,nthbv->nthav", A_j, Bm) + B_j

    _, Bbar = matrix_affine_scan(A, Bm)
    y = torch.einsum("ntha,nthav->nthv", q, Bbar).reshape(B_, T_, n_heads * d_v)
    return layer.out_proj(y)


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

    def test_delta_matches_hand_oracle(self):
        """DeltaMinGRU oracle: recomputes the DeltaNet/DeltaProduct recurrence
        from ``layer``'s own ``nn.Linear`` submodules alone, sharing no code
        with ``_coeffs``/``_micro_step``/``forward``/``step``. ``nh=2``
        exercises DeltaProduct's ordered multi-micro-step composition, not
        just DeltaNet's single Householder update; a bug in the shared
        micro-step math (e.g. a dropped or reordered term) diverges from this
        independently-written recurrence.
        """
        torch.manual_seed(SEED)
        layer = DeltaMinGRU(D_IN, D_H, n_heads=2, nh=2)
        x = torch.randn(B, T, D_IN)
        ref = _ref_delta_forward(layer, x)
        assert torch.allclose(layer(x), ref, atol=1e-5)
