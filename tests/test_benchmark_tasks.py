"""Tests for `experiments/benchmark_tasks.py`: the `TaskSpec` contract and
the four tasks' data generators/loader -- S5, MQAR (Task 1), psMNIST and
pendulum (Task 2) -- spec
`.claude/output/specs/2026-07-19-benchmark-round-design.md` §9 acceptance
criteria.

Sections
--------
1. S5 Cayley table -- group of order 120 (closure, identity, inverses,
   non-abelian)
2. make_s5 -- labels match brute-force running left-composition, shapes
3. make_group_word -- generic constructor reproduces make_s5, seeded
   reproducibility
4. MQAR -- every query answerable with correct labels, mask marks exactly
   the query positions, validation errors, shapes/dtypes across eval configs
5. TaskSpec / Budget / EvalConfig -- contract field shapes
6. Pendulum -- reference-integrator correctness (analytic + self-
   convergence), strictly-increasing/irregular timestamps, shapes/dtypes,
   seeded reproducibility
7. psMNIST -- permutation stability, split/batch logic on synthetic
   tensors (no torchvision, no MNIST download), and a full
   `PsMNISTLoader` end-to-end pass via monkeypatched torchvision
"""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch
from experiments.benchmark_tasks import (
    MQAR_KEY_VOCAB,
    MQAR_TRAIN_PAIRS,
    MQAR_VALUE_VOCAB,
    PENDULUM_GRAVITY,
    PENDULUM_LENGTH,
    PENDULUM_SUBSTEPS,
    PSMNIST_T,
    S5_COMPOSE,
    S5_ELEMENTS,
    Budget,
    EvalConfig,
    PsMNISTLoader,
    TaskSpec,
    _integrate_pendulum,
    _ordered_batches,
    _prepare_split,
    _shuffled_batches,
    _split_train_val,
    make_group_word,
    make_mqar,
    make_pendulum,
    make_psmnist_permutation,
    make_s5,
)

SEED = 42


# ===========================================================================
# 1. S5 Cayley table -- group of order 120
# ===========================================================================


def test_s5_has_120_elements():
    assert S5_ELEMENTS.shape == (120, 5)
    assert S5_COMPOSE.shape == (120, 120)


def test_s5_identity_is_index_zero():
    assert S5_ELEMENTS[0].tolist() == [0, 1, 2, 3, 4]


def test_s5_compose_closure():
    """Every table entry is a valid element index (in-range) -- closure
    under composition, i.e. every composed permutation is itself one of
    the 120 rows."""
    assert S5_COMPOSE.min().item() >= 0
    assert S5_COMPOSE.max().item() <= 119


def test_s5_identity_is_two_sided():
    ident = 0
    row = torch.arange(120)
    assert torch.equal(S5_COMPOSE[ident, :], row)
    assert torch.equal(S5_COMPOSE[:, ident], row)


def test_s5_every_element_has_inverse():
    ident = 0
    for i in range(120):
        row = S5_COMPOSE[i]
        col = S5_COMPOSE[:, i]
        assert ident in row.tolist(), f"element {i} has no right inverse"
        assert ident in col.tolist(), f"element {i} has no left inverse"


def test_s5_is_non_abelian():
    mismatches = [
        (i, j)
        for i in range(120)
        for j in range(120)
        if S5_COMPOSE[i, j].item() != S5_COMPOSE[j, i].item()
    ]
    assert mismatches, "S5 must be non-abelian"


def test_s5_compose_matches_brute_force_permutation_composition():
    """table[i, j] must equal the index of the ACTUAL composed permutation
    elements[i][elements[j]], independently recomputed here (not just
    self-consistency of the table-construction code)."""
    index_of = {tuple(p.tolist()): k for k, p in enumerate(S5_ELEMENTS)}
    for i in range(0, 120, 7):  # stride: full 120x120 brute force is cheap
        # but keep the loop body's independent recomputation obviously
        # distinct from _compose_table's own implementation.
        for j in range(0, 120, 11):
            composed = tuple(S5_ELEMENTS[i][S5_ELEMENTS[j]].tolist())
            assert S5_COMPOSE[i, j].item() == index_of[composed]


# ===========================================================================
# 2. make_s5 -- labels match brute-force running composition
# ===========================================================================


def test_make_s5_shapes_and_dtype():
    gen = torch.Generator().manual_seed(SEED)
    x, y = make_s5(4, 16, gen)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert x.min().item() >= 0 and x.max().item() <= 119
    assert y.min().item() >= 0 and y.max().item() <= 119


def test_make_s5_labels_match_brute_force_composition():
    gen = torch.Generator().manual_seed(SEED)
    batch, T = 6, 20
    x, y = make_s5(batch, T, gen)
    elements = S5_ELEMENTS.tolist()
    for b in range(batch):
        state = list(range(5))  # identity permutation
        for t in range(T):
            g = elements[x[b, t].item()]
            state = [g[state[k]] for k in range(5)]  # g o state
            expected = elements.index(state)
            assert y[b, t].item() == expected, f"batch={b} t={t}"


def test_make_s5_seeded_reproducibility():
    gen1 = torch.Generator().manual_seed(SEED)
    x1, y1 = make_s5(4, 8, gen1)
    gen2 = torch.Generator().manual_seed(SEED)
    x2, y2 = make_s5(4, 8, gen2)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


# ===========================================================================
# 3. make_group_word -- generic constructor
# ===========================================================================


def test_make_group_word_reproduces_make_s5():
    make = make_group_word(S5_COMPOSE)
    gen1 = torch.Generator().manual_seed(SEED)
    x1, y1 = make(4, 10, gen1)
    gen2 = torch.Generator().manual_seed(SEED)
    x2, y2 = make_s5(4, 10, gen2)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_make_group_word_on_small_group():
    """A 2-element group (Z2 under addition) sanity-checks the generic
    constructor independently of S5's size."""
    z2 = torch.tensor([[0, 1], [1, 0]])
    make = make_group_word(z2)
    gen = torch.Generator().manual_seed(SEED)
    x, y = make(3, 5, gen)
    assert x.shape == (3, 5)
    assert y.shape == (3, 5)
    for b in range(3):
        expected = 0
        for t in range(5):
            expected ^= x[b, t].item()
            assert y[b, t].item() == expected


# ===========================================================================
# 4. MQAR
# ===========================================================================


def test_make_mqar_shapes_and_dtype():
    gen = torch.Generator().manual_seed(SEED)
    x, y, mask = make_mqar(8, 64, gen)
    assert x.shape == (8, 64)
    assert y.shape == (8, 64)
    assert mask.shape == (8, 64)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert mask.dtype == torch.bool


def test_make_mqar_mask_marks_exactly_query_positions():
    gen = torch.Generator().manual_seed(SEED)
    num_pairs = MQAR_TRAIN_PAIRS
    T = 64
    _, _, mask = make_mqar(8, T, gen, num_pairs=num_pairs)
    expected_mask = torch.zeros(T, dtype=torch.bool)
    expected_mask[T - num_pairs : T] = True
    for b in range(8):
        assert torch.equal(mask[b], expected_mask)
    assert mask.sum().item() == 8 * num_pairs


def test_make_mqar_every_query_answerable_and_correct():
    """Every key presented in the query block must be one of the keys
    shown during presentation, with y at that position equal to the value
    paired with that exact key during presentation."""
    gen = torch.Generator().manual_seed(SEED)
    num_pairs = MQAR_TRAIN_PAIRS
    T = 64
    x, y, mask = make_mqar(16, T, gen, num_pairs=num_pairs)
    for b in range(16):
        presented_keys = x[b, 0 : 2 * num_pairs : 2].tolist()
        presented_values = x[b, 1 : 2 * num_pairs : 2].tolist()
        key_to_value = dict(zip(presented_keys, presented_values, strict=True))
        assert len(key_to_value) == num_pairs, "presented keys must be distinct"
        query_positions = mask[b].nonzero(as_tuple=True)[0].tolist()
        assert len(query_positions) == num_pairs
        for pos in query_positions:
            queried_key = x[b, pos].item()
            assert queried_key in key_to_value, (
                f"batch={b} position={pos} queries key {queried_key} never presented"
            )
            assert y[b, pos].item() == key_to_value[queried_key]


def test_make_mqar_query_order_is_a_permutation_of_presented_keys():
    gen = torch.Generator().manual_seed(SEED)
    num_pairs = MQAR_TRAIN_PAIRS
    T = 64
    x, _, mask = make_mqar(8, T, gen, num_pairs=num_pairs)
    for b in range(8):
        presented_keys = sorted(x[b, 0 : 2 * num_pairs : 2].tolist())
        query_positions = mask[b].nonzero(as_tuple=True)[0].tolist()
        queried_keys = sorted(x[b, pos].item() for pos in query_positions)
        assert queried_keys == presented_keys


def test_make_mqar_keys_and_values_in_disjoint_ranges():
    gen = torch.Generator().manual_seed(SEED)
    num_pairs = MQAR_TRAIN_PAIRS
    T = 64
    x, y, _ = make_mqar(8, T, gen, num_pairs=num_pairs)
    presented_values = x[:, 1 : 2 * num_pairs : 2]
    assert presented_values.min().item() >= MQAR_KEY_VOCAB
    assert presented_values.max().item() < MQAR_KEY_VOCAB + MQAR_VALUE_VOCAB
    presented_keys = x[:, 0 : 2 * num_pairs : 2]
    assert presented_keys.min().item() >= 0
    assert presented_keys.max().item() < MQAR_KEY_VOCAB


@pytest.mark.parametrize("T,num_pairs", [(256, 16), (256, 32)])
def test_make_mqar_eval_configs(T, num_pairs):
    """The two length/capacity generalization eval configs (spec §4)."""
    gen = torch.Generator().manual_seed(SEED)
    x, y, mask = make_mqar(4, T, gen, num_pairs=num_pairs)
    assert x.shape == (4, T)
    assert mask.sum().item() == 4 * num_pairs
    for b in range(4):
        presented_keys = x[b, 0 : 2 * num_pairs : 2].tolist()
        presented_values = x[b, 1 : 2 * num_pairs : 2].tolist()
        key_to_value = dict(zip(presented_keys, presented_values, strict=True))
        assert len(key_to_value) == num_pairs
        for pos in mask[b].nonzero(as_tuple=True)[0].tolist():
            assert y[b, pos].item() == key_to_value[x[b, pos].item()]


def test_make_mqar_rejects_too_many_pairs_for_key_vocab():
    gen = torch.Generator().manual_seed(SEED)
    with pytest.raises(ValueError, match="exceeds key_vocab"):
        make_mqar(4, 256, gen, num_pairs=33, key_vocab=32)


def test_make_mqar_rejects_span_exceeding_T():
    gen = torch.Generator().manual_seed(SEED)
    with pytest.raises(ValueError, match="exceeding T"):
        make_mqar(4, 20, gen, num_pairs=8)  # needs 24 > 20


def test_make_mqar_seeded_reproducibility():
    gen1 = torch.Generator().manual_seed(SEED)
    x1, y1, m1 = make_mqar(8, 64, gen1)
    gen2 = torch.Generator().manual_seed(SEED)
    x2, y2, m2 = make_mqar(8, 64, gen2)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
    assert torch.equal(m1, m2)


# ===========================================================================
# 5. TaskSpec / Budget / EvalConfig -- contract field shapes
# ===========================================================================


def test_task_spec_construction_dense_s5():
    spec = TaskSpec(
        name="s5",
        loss_mode="dense",
        data=make_s5,
        fit_metric="val128",
        fit_threshold=0.99,
        fit_direction="ge",
        robustness=(0.98, 0.99, 0.995),
        eval_protocol=(EvalConfig(T=256), EvalConfig(T=512), EvalConfig(T=1024)),
        budget=Budget(lr=3e-3, batch_size=128, steps=1600, eval_every=100),
        seeds=36,
    )
    assert spec.loss_mode == "dense"
    assert spec.data is make_s5
    x, y = spec.data(2, 8, torch.Generator().manual_seed(SEED))
    assert x.shape == (2, 8)
    assert spec.eval_protocol[0].num_pairs is None


def test_task_spec_construction_masked_query_mqar():
    spec = TaskSpec(
        name="mqar",
        loss_mode="masked_query",
        data=make_mqar,
        fit_metric="val_qacc",
        fit_threshold=0.99,
        fit_direction="ge",
        robustness=(0.98, 0.99, 0.995),
        eval_protocol=(EvalConfig(T=256, num_pairs=16), EvalConfig(T=256, num_pairs=32)),
        budget=Budget(lr=3e-3, batch_size=128, steps=1600, eval_every=100),
        seeds=36,
    )
    assert spec.loss_mode == "masked_query"
    cfg = spec.eval_protocol[1]
    x, y, mask = spec.data(2, cfg.T, torch.Generator().manual_seed(SEED), num_pairs=cfg.num_pairs)
    assert x.shape == (2, cfg.T)
    assert mask.sum().item() == 2 * cfg.num_pairs


def test_budget_exactly_one_of_steps_epochs_is_the_contract_shape():
    """Not an enforced invariant (no validation in Budget itself) -- pins
    that a step-based instance leaves epochs None and vice versa, matching
    the documented "exactly one is non-None" convention."""
    step_budget = Budget(lr=3e-3, batch_size=128, steps=1600)
    assert step_budget.epochs is None
    epoch_budget = Budget(lr=1e-3, batch_size=256, epochs=10)
    assert epoch_budget.steps is None


# ===========================================================================
# 6. Pendulum
# ===========================================================================


def test_make_pendulum_shapes_and_dtype():
    gen = torch.Generator().manual_seed(SEED)
    batch, T = 4, 16
    x, dt, y = make_pendulum(batch, T, gen)
    assert x.shape == (batch, T, 2)
    assert dt.shape == (batch, T)
    assert y.shape == (batch, T, 2)
    assert x.dtype == torch.float32
    assert dt.dtype == torch.float32
    assert y.dtype == torch.float32


def test_make_pendulum_first_gap_is_zero():
    """Position 0 is the initial condition -- nothing precedes it to decay
    from (probes.py's no-t=0-exemption convention)."""
    gen = torch.Generator().manual_seed(SEED)
    _, dt, _ = make_pendulum(4, 16, gen)
    assert torch.equal(dt[:, 0], torch.zeros(4))


def test_make_pendulum_timestamps_strictly_increasing_and_irregular():
    gen = torch.Generator().manual_seed(SEED)
    batch, T = 8, 32
    _, dt, _ = make_pendulum(batch, T, gen)
    timestamps = torch.cumsum(dt, dim=1)
    assert (timestamps[:, 1:] > timestamps[:, :-1]).all(), "timestamps must strictly increase"
    # Irregular: not every gap is identical (a constant-step grid would fail
    # the "irregularly spaced" protocol requirement, spec §4).
    assert dt[:, 1:].std().item() > 0.0


def test_make_pendulum_label_is_noiseless_sin_cos_of_trajectory():
    """`y` must be exactly `(sin theta, cos theta)` of the integrated
    trajectory, independently recomputed here from `_integrate_pendulum`
    (not just self-consistency of `make_pendulum`'s own implementation)."""
    gen = torch.Generator().manual_seed(SEED)
    batch, T = 4, 12
    # gen2 mirrors make_pendulum's internal draw order (theta0, omega0, dt)
    # under the same seed, so it reproduces the same random numbers.
    gen2 = torch.Generator().manual_seed(SEED)
    theta0 = torch.empty(batch).uniform_(-1.0, 1.0, generator=gen2)
    omega0 = torch.empty(batch).uniform_(-1.0, 1.0, generator=gen2)
    dt = torch.empty(batch, T).uniform_(0.05, 0.3, generator=gen2)
    dt[:, 0] = 0.0
    theta, _ = _integrate_pendulum(theta0, omega0, dt)
    expected_y = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)

    x, dt_out, y = make_pendulum(batch, T, gen)
    assert torch.equal(dt_out, dt)
    assert torch.allclose(y, expected_y)
    assert not torch.allclose(x, y), "x must carry observation noise, y must not"


def test_integrate_pendulum_matches_small_angle_analytic_solution():
    """The frictionless pendulum reduces to simple harmonic motion at small
    angles: theta(t) = theta0*cos(w*t) (zero initial angular velocity),
    w = sqrt(g/L). This is an independent analytic reference, not internal
    self-consistency of the RK4 stepper."""
    batch, T = 4, 10
    theta0 = torch.full((batch,), 0.01)  # rad: small-angle regime
    omega0 = torch.zeros(batch)
    dt = torch.full((batch, T), 0.1)
    dt[:, 0] = 0.0

    theta_traj, _ = _integrate_pendulum(theta0, omega0, dt)

    w = math.sqrt(PENDULUM_GRAVITY / PENDULUM_LENGTH)
    t = torch.cumsum(dt, dim=1)
    theta_analytic = theta0.unsqueeze(1) * torch.cos(w * t)
    assert torch.allclose(theta_traj, theta_analytic, atol=1e-3)


def test_integrate_pendulum_substeps_convergence():
    """The small-angle analytic check above cannot discriminate a wrong
    (e.g. under-substepped) integrator at theta0=0.01: both fine RK4 and a
    single-step-per-gap integration pass its atol=1e-3. This test instead
    compares against a much finer in-integrator reference (n_substeps=200)
    at a sizable angle and a coarse dt, where discretization error is
    large enough to matter: a single-substep pass (n_substeps=1) must show
    order(s)-of-magnitude more error than the production substep count
    (`PENDULUM_SUBSTEPS`) -- the discriminating signal a fixed-tolerance
    comparison at small angles cannot provide."""
    batch, T = 4, 8
    theta0 = torch.tensor([0.8, -0.6, 1.0, -0.9])  # rad: sizable, nonlinear regime
    omega0 = torch.tensor([0.2, -0.3, 0.0, 0.5])
    dt = torch.full((batch, T), 0.3)
    dt[:, 0] = 0.0

    reference_theta, _ = _integrate_pendulum(theta0, omega0, dt, n_substeps=200)
    coarse_theta, _ = _integrate_pendulum(theta0, omega0, dt, n_substeps=1)
    production_theta, _ = _integrate_pendulum(theta0, omega0, dt, n_substeps=PENDULUM_SUBSTEPS)

    coarse_error = (coarse_theta - reference_theta).abs().max().item()
    production_error = (production_theta - reference_theta).abs().max().item()

    assert coarse_error > 1e-2, "n_substeps=1 must show a real discretization error at this dt"
    assert production_error < 1e-4, "production substep count must track the fine reference tightly"
    assert production_error < coarse_error / 10, "finer substepping must materially reduce error"


def test_integrate_pendulum_position_zero_is_initial_condition():
    batch, T = 3, 5
    theta0 = torch.tensor([0.1, -0.2, 0.3])
    omega0 = torch.tensor([0.0, 0.5, -0.5])
    dt = torch.full((batch, T), 0.1)
    dt[:, 0] = 0.0
    theta_traj, omega_traj = _integrate_pendulum(theta0, omega0, dt)
    assert torch.equal(theta_traj[:, 0], theta0)
    assert torch.equal(omega_traj[:, 0], omega0)


def test_make_pendulum_seeded_reproducibility():
    gen1 = torch.Generator().manual_seed(SEED)
    x1, dt1, y1 = make_pendulum(4, 16, gen1)
    gen2 = torch.Generator().manual_seed(SEED)
    x2, dt2, y2 = make_pendulum(4, 16, gen2)
    assert torch.equal(x1, x2)
    assert torch.equal(dt1, dt2)
    assert torch.equal(y1, y2)


def test_task_spec_construction_regression_pendulum():
    spec = TaskSpec(
        name="pendulum",
        loss_mode="regression",
        data=make_pendulum,
        fit_metric="val_mse",
        fit_threshold=0.05,
        fit_direction="le",
        robustness=(0.0625, 0.05, 0.04),
        eval_protocol=(EvalConfig(T=64),),
        budget=Budget(lr=3e-3, batch_size=64, steps=2000, eval_every=100),
        seeds=36,
    )
    assert spec.loss_mode == "regression"
    assert spec.fit_direction == "le"
    x, dt, y = spec.data(2, 16, torch.Generator().manual_seed(SEED))
    assert x.shape == (2, 16, 2)
    assert dt.shape == (2, 16)
    assert y.shape == (2, 16, 2)


# ===========================================================================
# 7. psMNIST
# ===========================================================================


def test_make_psmnist_permutation_is_a_valid_permutation():
    perm = make_psmnist_permutation(SEED)
    assert perm.shape == (PSMNIST_T,)
    assert torch.equal(perm.sort().values, torch.arange(PSMNIST_T))


def test_make_psmnist_permutation_identical_across_instantiations_given_same_seed():
    perm1 = make_psmnist_permutation(SEED)
    perm2 = make_psmnist_permutation(SEED)
    assert torch.equal(perm1, perm2)


def test_make_psmnist_permutation_differs_across_seeds():
    perm1 = make_psmnist_permutation(SEED)
    perm2 = make_psmnist_permutation(SEED + 1)
    assert not torch.equal(perm1, perm2)


def test_prepare_split_applies_permutation_and_scales_to_unit_range():
    """Pure tensor logic, tested with synthetic images -- no torchvision,
    no MNIST download (spec/plan requirement)."""
    n = 6
    images = torch.randint(0, 256, (n, 28, 28), dtype=torch.uint8)
    labels = torch.randint(0, 10, (n,))
    permutation = make_psmnist_permutation(SEED)

    x, y = _prepare_split(images, labels, permutation)

    assert x.shape == (n, PSMNIST_T, 1)
    assert x.dtype == torch.float32
    assert x.min().item() >= 0.0 and x.max().item() <= 1.0
    assert y.dtype == torch.int64
    assert torch.equal(y, labels.long())

    flat = images.reshape(n, PSMNIST_T).float() / 255.0
    expected_x = flat[:, permutation].unsqueeze(-1)
    assert torch.equal(x, expected_x)


def test_split_train_val_sizes_and_disjointness():
    """Synthetic dataset of MNIST's real 60k row count, split with the
    spec's 50k/10k sizes -- exercised without any real MNIST data."""
    n = 60_000
    images = torch.arange(n).reshape(n, 1, 1).expand(n, 28, 28)
    labels = torch.arange(n) % 10

    train_images, train_labels, val_images, val_labels = _split_train_val(images, labels)

    assert train_images.shape[0] == 50_000
    assert val_images.shape[0] == 10_000
    # Disjoint row ranges: train is [0, 50000), val is [50000, 60000).
    assert train_images[:, 0, 0].max().item() < val_images[:, 0, 0].min().item()
    assert torch.equal(train_labels, labels[:50_000])
    assert torch.equal(val_labels, labels[50_000:60_000])


def test_ordered_batches_covers_dataset_in_fixed_order():
    n, batch_size = 10, 4
    x = torch.arange(n).reshape(n, 1).float()
    y = torch.arange(n)
    batches = list(_ordered_batches(x, y, batch_size))
    assert [b[0].shape[0] for b in batches] == [4, 4, 2]  # last batch smaller
    recovered_x = torch.cat([b[0] for b in batches], dim=0)
    assert torch.equal(recovered_x, x), "ordered batches must not shuffle"


def test_ordered_batches_is_reproducible_across_calls():
    n, batch_size = 10, 4
    x = torch.arange(n).reshape(n, 1).float()
    y = torch.arange(n)
    batches1 = list(_ordered_batches(x, y, batch_size))
    batches2 = list(_ordered_batches(x, y, batch_size))
    for (x1, y1), (x2, y2) in zip(batches1, batches2, strict=True):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


def test_shuffled_batches_covers_dataset_but_reorders():
    n, batch_size = 10, 4
    x = torch.arange(n).reshape(n, 1).float()
    y = torch.arange(n)
    gen = torch.Generator().manual_seed(SEED)
    batches = list(_shuffled_batches(x, y, batch_size, gen))
    recovered_y = torch.cat([b[1] for b in batches], dim=0)
    assert torch.equal(recovered_y.sort().values, y), "shuffle must cover every row exactly once"
    assert not torch.equal(recovered_y, y), "shuffle must actually reorder"


def test_shuffled_batches_seeded_reproducibility():
    n, batch_size = 10, 4
    x = torch.arange(n).reshape(n, 1).float()
    y = torch.arange(n)
    gen1 = torch.Generator().manual_seed(SEED)
    batches1 = list(_shuffled_batches(x, y, batch_size, gen1))
    gen2 = torch.Generator().manual_seed(SEED)
    batches2 = list(_shuffled_batches(x, y, batch_size, gen2))
    for (x1, y1), (x2, y2) in zip(batches1, batches2, strict=True):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


def test_task_spec_construction_last_step_psmnist_with_synthetic_loader():
    """A minimal stand-in `Loader` (no torchvision) exercises the
    `last_step` contract through `TaskSpec.data` without a real dataset."""

    class _FakeLoader:
        def __init__(self):
            self.x = torch.rand(20, PSMNIST_T, 1)
            self.y = torch.randint(0, 10, (20,))

        def train_epoch(self, gen):
            yield from _shuffled_batches(self.x, self.y, 8, gen)

        def val(self):
            yield from _ordered_batches(self.x, self.y, 8)

        def test(self):
            yield from _ordered_batches(self.x, self.y, 8)

    loader = _FakeLoader()
    spec = TaskSpec(
        name="psmnist",
        loss_mode="last_step",
        data=loader,
        fit_metric="val_acc",
        fit_threshold=0.90,
        fit_direction="ge",
        robustness=(0.88, 0.90, 0.92),
        eval_protocol=(EvalConfig(T=PSMNIST_T),),
        budget=Budget(lr=1e-3, batch_size=8, epochs=10),
        seeds=12,
    )
    assert spec.loss_mode == "last_step"
    gen = torch.Generator().manual_seed(SEED)
    batches = list(spec.data.train_epoch(gen))
    assert batches, "train_epoch must yield at least one batch"
    x0, y0 = batches[0]
    assert x0.shape[1:] == (PSMNIST_T, 1)
    assert y0.dtype == torch.int64


class _FakeMNIST:
    """Stand-in for `torchvision.datasets.MNIST`: same constructor
    signature (`root`, `train`, `download`) and the two attributes
    `PsMNISTLoader` reads (`.data`, `.targets`), but synthetic and
    deterministic (fixed seeds per split) so an independent
    reconstruction in the test below reproduces identical raw tensors
    without touching torchvision or the network."""

    def __init__(self, root, train, download):
        n = 60_000 if train else 10_000  # matches real MNIST's train/test sizes
        gen = torch.Generator().manual_seed(100 if train else 200)
        self.data = torch.randint(0, 256, (n, 28, 28), dtype=torch.uint8, generator=gen)
        self.targets = torch.randint(0, 10, (n,), generator=gen)


def test_psmnist_loader_end_to_end_with_monkeypatched_torchvision(monkeypatch):
    """Exercises `PsMNISTLoader.__init__`/`train_epoch`/`val`/`test` for
    real (not `_FakeLoader`) via a monkeypatched `torchvision` module
    (`sys.modules['torchvision']` shim, no real torchvision installed, no
    MNIST download): verifies the production 50k/10k/10k split sizes, that
    the same fixed pixel permutation reaches every split, and the
    `last_step` batch shape/dtype contract."""
    fake_torchvision = types.SimpleNamespace(datasets=types.SimpleNamespace(MNIST=_FakeMNIST))
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)

    loader = PsMNISTLoader(permutation_seed=SEED, batch_size=32, root="unused", download=False)

    # Independent reconstruction (same fixed seeds as _FakeMNIST -> bit-
    # identical raw tensors) of the expected splits, without reaching into
    # the loader's internals.
    train_full = _FakeMNIST(root="unused", train=True, download=False)
    test_full = _FakeMNIST(root="unused", train=False, download=False)
    permutation = make_psmnist_permutation(SEED)
    train_images, train_labels, val_images, val_labels = _split_train_val(
        train_full.data, train_full.targets
    )
    expected_train_x, expected_train_y = _prepare_split(train_images, train_labels, permutation)
    expected_val_x, expected_val_y = _prepare_split(val_images, val_labels, permutation)
    expected_test_x, expected_test_y = _prepare_split(
        test_full.data, test_full.targets, permutation
    )

    val_x = torch.cat([b[0] for b in loader.val()], dim=0)
    val_y = torch.cat([b[1] for b in loader.val()], dim=0)
    test_x = torch.cat([b[0] for b in loader.test()], dim=0)
    test_y = torch.cat([b[1] for b in loader.test()], dim=0)
    train_batches = list(loader.train_epoch(torch.Generator().manual_seed(0)))
    train_x = torch.cat([b[0] for b in train_batches], dim=0)
    train_y = torch.cat([b[1] for b in train_batches], dim=0)

    # -- 50k/10k/10k split sizes (spec §4) --
    assert train_x.shape[0] == 50_000
    assert val_x.shape[0] == 10_000
    assert test_x.shape[0] == 10_000

    # -- last_step batch shape/dtype contract (spec §6) --
    assert val_x.shape == (10_000, PSMNIST_T, 1)
    assert val_x.dtype == torch.float32
    assert val_y.dtype == torch.int64

    # -- same fixed permutation reaches every split: val/test iterate in a
    # fixed (non-shuffled) order, so they must match the independently
    # recomputed expectation bit-for-bit.
    assert torch.equal(val_x, expected_val_x)
    assert torch.equal(val_y, expected_val_y)
    assert torch.equal(test_x, expected_test_x)
    assert torch.equal(test_y, expected_test_y)

    # -- train is shuffled, so check permutation application via a
    # permutation-invariant identity: an arbitrary expected train row must
    # appear, verbatim (same pixel permutation and label), among the
    # actual (reshuffled) train rows exactly once.
    target_x, target_y = expected_train_x[0], expected_train_y[0]
    row_matches = (train_x == target_x).all(dim=(1, 2))
    assert row_matches.sum().item() == 1, "expected train row must survive shuffling exactly once"
    matched_index = row_matches.nonzero(as_tuple=True)[0].item()
    assert train_y[matched_index].item() == target_y.item()
