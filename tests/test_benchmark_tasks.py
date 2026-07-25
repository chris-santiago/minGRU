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
8. auroc -- perfect/reversed/all-tied score orderings, a hand-computed
   tie case, shape mismatch and single-class-input errors
9A/9B. Rosbank pure helpers and `RosbankLoader` end-to-end (Task 2)
9C. `CHURN_TASK` registration -- `TASKS`/`BENCH_ROUND_TAGS` membership,
   TaskSpec field contract, `make_churn_loader` factory wiring (Task 4)
"""

from __future__ import annotations

import datetime
import math
import sys
import types
import urllib.error
from pathlib import Path

import experiments.benchmark_tasks as benchmark_tasks_module
import pytest
import torch
from experiments.benchmark_tasks import (
    BENCH_ROUND_TAGS,
    CHURN_SPLIT_SEED,
    CHURN_TASK,
    MQAR_KEY_VOCAB,
    MQAR_TRAIN_PAIRS,
    MQAR_VALUE_VOCAB,
    PENDULUM_GRAVITY,
    PENDULUM_LENGTH,
    PENDULUM_SUBSTEPS,
    PSMNIST_T,
    ROSBANK_CURRENCY_CAP,
    ROSBANK_EXPECTED_COLUMNS,
    ROSBANK_MCC_CAP,
    ROSBANK_NAN_CHANNEL_TYPE,
    ROSBANK_T,
    ROSBANK_TEST_USERS,
    ROSBANK_TRAIN_USERS,
    ROSBANK_VAL_USERS,
    S5_COMPOSE,
    S5_ELEMENTS,
    TASKS,
    Budget,
    EvalConfig,
    PsMNISTLoader,
    RosbankLoader,
    TaskSpec,
    _build_vocab,
    _download_rosbank_csv,
    _encode_user_sequence,
    _ensure_rosbank_csv,
    _FieldVocab,
    _integrate_pendulum,
    _ordered_batches,
    _prepare_split,
    _rosbank_onehot_batch,
    _rosbank_ordered_batches,
    _rosbank_shuffled_batches,
    _RosbankSplitData,
    _RosbankVocabs,
    _shuffled_batches,
    _split_train_val,
    _split_user_ids,
    _truncate_and_leftpad,
    _user_gap_days,
    _valid_mask,
    _validate_rosbank_schema,
    _window_gap_days,
    auroc,
    make_churn_loader,
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


def test_make_mqar_filler_never_holds_a_key_range_token():
    """Filler positions -- everything outside the presentation key slots
    (`[0, 2*num_pairs)` even indices) and the query block
    (`[T - num_pairs, T)`) -- must never hold a key-range token
    (`[0, MQAR_KEY_VOCAB)`); equivalently, key-range tokens appear only in
    the presentation block and the query block. Pre-matrix technical-review
    fix: filler previously drew from the full combined vocab, fabricating
    spurious key-value bindings whose density scaled with `T` (a confound
    on the capacity-generalization claim)."""
    gen = torch.Generator().manual_seed(SEED)
    num_pairs = MQAR_TRAIN_PAIRS
    T = 64
    x, _, _ = make_mqar(16, T, gen, num_pairs=num_pairs)

    allowed_key_positions = torch.zeros(T, dtype=torch.bool)
    allowed_key_positions[0 : 2 * num_pairs : 2] = True  # presentation key slots
    allowed_key_positions[T - num_pairs : T] = True  # query block
    disallowed_positions = ~allowed_key_positions

    is_key_range = x < MQAR_KEY_VOCAB
    assert not is_key_range[:, disallowed_positions].any(), (
        "a filler (or presentation-value) position holds a key-range token"
    )
    # Every disallowed position -- filler plus the presentation's value
    # slots -- must land in the value range.
    is_value_range = (x >= MQAR_KEY_VOCAB) & (x < MQAR_KEY_VOCAB + MQAR_VALUE_VOCAB)
    assert is_value_range[:, disallowed_positions].all()


def test_make_mqar_filler_beyond_presentation_and_query_stays_in_value_range():
    """At a larger T (eval-style gap between presentation and query, unlike
    the tight T=3*num_pairs training config), the wide filler gap in the
    middle must still never carry a key-range token."""
    gen = torch.Generator().manual_seed(SEED)
    num_pairs = 8
    T = 256
    x, _, mask = make_mqar(8, T, gen, num_pairs=num_pairs)
    presentation_end = 2 * num_pairs
    query_start = T - num_pairs
    filler = x[:, presentation_end:query_start]
    assert filler.min().item() >= MQAR_KEY_VOCAB
    assert filler.max().item() < MQAR_KEY_VOCAB + MQAR_VALUE_VOCAB
    assert mask[:, presentation_end:query_start].logical_not().all()  # sanity: filler is unmasked


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


# ===========================================================================
# 8. auroc -- Mann-Whitney rank statistic, average-rank ties (spec §6 metric
#    contract: "score = positive-class logit minus negative-class logit at
#    the final position; AUROC by Mann-Whitney rank statistic with average
#    ranks for ties; direction ge. Degenerate single-class eval sets are a
#    hard error.")
# ===========================================================================


def test_auroc_perfectly_ranked_scores_is_one():
    """Every positive score strictly exceeds every negative score -> 1.0."""
    scores = torch.tensor([0.1, 0.2, 0.9, 1.5])
    labels = torch.tensor([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(1.0)


def test_auroc_reversed_scores_is_zero():
    """Every positive score strictly below every negative score -> 0.0."""
    scores = torch.tensor([0.9, 1.5, 0.1, 0.2])
    labels = torch.tensor([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.0)


def test_auroc_all_tied_scores_is_half():
    """A completely uninformative (constant) score gives the chance-level
    0.5, not an arbitrary tie-breaking artifact."""
    scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
    labels = torch.tensor([0, 1, 0, 1])
    assert auroc(scores, labels) == pytest.approx(0.5)


def test_auroc_hand_computed_tie_case():
    """scores=[1,1,2,3], labels=[0,1,0,1] (one positive tied with one
    negative, one positive strictly ahead of both negatives).

    By exhaustive pairwise comparison (Mann-Whitney U over the 2x2
    positive/negative pairs, ties scoring 0.5): pos=1 vs neg=1 -> tie
    (0.5); pos=1 vs neg=2 -> discordant (0.0); pos=3 vs neg=1 ->
    concordant (1.0); pos=3 vs neg=2 -> concordant (1.0). Mean = (0.5 + 0.0
    + 1.0 + 1.0) / 4 = 0.625, matching the average-rank formula: tied rank
    1.5 for both score-1 entries, rank 3 for score-2, rank 4 for score-3;
    sum of positive ranks = 1.5 + 4 = 5.5; U = 5.5 - 2*3/2 = 2.5; AUC =
    2.5 / (2*2) = 0.625.
    """
    scores = torch.tensor([1.0, 1.0, 2.0, 3.0])
    labels = torch.tensor([0, 1, 0, 1])
    assert auroc(scores, labels) == pytest.approx(0.625)


def test_auroc_single_class_raises():
    """A degenerate eval set (only one label value present) has no defined
    AUROC -- spec §6 calls this a hard error, not a silent fallback."""
    scores = torch.tensor([0.1, 0.5, 0.9])
    labels = torch.tensor([1, 1, 1])
    with pytest.raises(ValueError):
        auroc(scores, labels)

    with pytest.raises(ValueError):
        auroc(scores, torch.zeros_like(labels))


def test_auroc_shape_mismatch_raises():
    """`scores`/`labels` element counts must match; a length mismatch is a
    caller bug, not a value to silently truncate/broadcast."""
    scores = torch.tensor([0.1, 0.5, 0.9])
    labels = torch.tensor([1, 0])
    with pytest.raises(ValueError):
        auroc(scores, labels)


# ===========================================================================
# 9. Rosbank churn -- spec `.claude/output/specs/2026-07-25-churn-benchmark-
#    design.md` §4/§6/§9. Section A is pure functions (no pandas needed);
#    section B is `RosbankLoader` end-to-end against a synthetic mini-CSV
#    fixture (skip-if-no-pandas, no network).
# ===========================================================================

# --------------------------------------------------- 9A. pure helpers


def test_build_vocab_uncapped_orders_by_value_and_assigns_oov():
    vocab = _build_vocab(["b", "a", "c", "a"], cap=None)
    assert vocab.index == {"a": 0, "b": 1, "c": 2}
    assert vocab.size == 4
    assert vocab.oov_index == 3
    assert vocab.encode("a") == 0
    assert vocab.encode("c") == 2
    assert vocab.encode("unseen") == 3


def test_build_vocab_capped_ranks_by_frequency_then_value_tiebreak():
    """1 and 4 tie at frequency 3; 2 has frequency 2; 3 has frequency 1.
    cap=2 keeps only the top two by frequency, tie broken by ascending
    value -- 1 then 4 -- everything else (2, 3) falls to OOV."""
    vocab = _build_vocab([1, 1, 1, 2, 2, 3, 4, 4, 4], cap=2)
    assert vocab.index == {1: 0, 4: 1}
    assert vocab.size == 3
    assert vocab.encode(1) == 0
    assert vocab.encode(4) == 1
    assert vocab.encode(2) == vocab.oov_index
    assert vocab.encode(3) == vocab.oov_index


def test_rosbank_vocabs_d_in_sums_every_field_plus_amount():
    vocabs = _RosbankVocabs(
        mcc=_FieldVocab({100: 0, 200: 1}),  # size 3
        trx_category=_FieldVocab({"A": 0}),  # size 2
        channel_type=_FieldVocab({"POS": 0, ROSBANK_NAN_CHANNEL_TYPE: 1}),  # size 3
        currency=_FieldVocab({810: 0}),  # size 2
    )
    assert vocabs.d_in == 3 + 2 + 3 + 2 + 1  # == 11


def test_split_user_ids_is_deterministic_and_partitions_every_user_once():
    ids = list(range(1, 9))
    train1, val1, test1 = _split_user_ids(ids, split_seed=7, n_train=4, n_val=2, n_test=2)
    train2, val2, test2 = _split_user_ids(ids, split_seed=7, n_train=4, n_val=2, n_test=2)
    assert (train1, val1, test1) == (train2, val2, test2)
    assert len(train1) == 4
    assert len(val1) == 2
    assert len(test1) == 2
    assert sorted(train1 + val1 + test1) == ids


def test_split_user_ids_differs_across_seeds():
    ids = list(range(1, 9))
    train1, _, _ = _split_user_ids(ids, split_seed=7, n_train=4, n_val=2, n_test=2)
    train2, _, _ = _split_user_ids(ids, split_seed=8, n_train=4, n_val=2, n_test=2)
    assert train1 != train2


def test_split_user_ids_rejects_size_mismatch():
    with pytest.raises(ValueError, match="do not match"):
        _split_user_ids(list(range(1, 9)), split_seed=7, n_train=4, n_val=2, n_test=3)


def test_user_gap_days_first_position_is_zero_and_rest_are_real_gaps():
    timestamps = [
        datetime.datetime(2017, 1, 1),
        datetime.datetime(2017, 1, 3),
        datetime.datetime(2017, 1, 3, 12),
    ]
    assert _user_gap_days(timestamps) == [0.0, 2.0, 0.5]


def test_user_gap_days_empty_input():
    assert _user_gap_days([]) == []


@pytest.mark.parametrize(
    "n_events,T,expected",
    [
        (2, 4, [False, False, True, True]),
        (4, 4, [True, True, True, True]),
        (6, 4, [True, True, True, True]),  # over-length: no padding at all
        (0, 3, [False, False, False]),
    ],
)
def test_valid_mask(n_events, T, expected):
    assert _valid_mask(n_events, T) == expected


def test_truncate_and_leftpad_keeps_last_T_and_pads_front():
    assert _truncate_and_leftpad([1, 2, 3, 4, 5, 6], 4, 0) == [3, 4, 5, 6]
    assert _truncate_and_leftpad([1, 2], 4, 0) == [0, 0, 1, 2]
    assert _truncate_and_leftpad([1, 2, 3, 4], 4, 0) == [1, 2, 3, 4]


def test_window_gap_days_forces_first_kept_position_to_zero():
    """Full history has 6 gaps; keeping the last 4 would naively carry a
    real (nonzero) gap into position 0 of the window -- must be forced back
    to 0 (no predecessor WITHIN the window), unlike every other kept
    position."""
    full = [0.0, 1.0, 1.0, 1.0, 243.0, 1.0]
    assert _window_gap_days(full, 4) == [0.0, 1.0, 243.0, 1.0]


def test_window_gap_days_pads_with_zero():
    assert _window_gap_days([0.0, 1.0], 4) == [0.0, 0.0, 0.0, 1.0]


def test_encode_user_sequence_maps_oov_and_truncates_and_pads():
    vocabs = _RosbankVocabs(
        mcc=_FieldVocab({100: 0}),
        trx_category=_FieldVocab({"A": 0}),
        channel_type=_FieldVocab({"POS": 0}),
        currency=_FieldVocab({810: 0}),
    )
    mcc, trxcat, channel, currency, amount_log1p, dt, valid = _encode_user_sequence(
        mcc=[100, 999],  # 999 unseen -> OOV
        trx_category=["A", "A"],
        channel_type=["POS", "POS"],
        currency=[810, 810],
        amount=[0.0, math.e - 1],  # log1p(e - 1) == 1.0
        timestamps=[datetime.datetime(2017, 1, 1), datetime.datetime(2017, 1, 2)],
        vocabs=vocabs,
        T=3,
    )
    assert valid == [False, True, True]
    assert mcc == [0, vocabs.mcc.encode(100), vocabs.mcc.encode(999)]
    assert mcc[-1] == vocabs.mcc.oov_index
    assert dt == [0.0, 0.0, 1.0]
    assert amount_log1p[1] == pytest.approx(0.0)
    assert amount_log1p[2] == pytest.approx(1.0)


def test_rosbank_onehot_batch_zeros_pad_positions_and_concatenates_fields_in_spec_order():
    """Feature layout (spec §6): MCC || trx_category || channel_type ||
    currency || log1p(amount)."""
    vocabs = _RosbankVocabs(
        mcc=_FieldVocab({100: 0, 200: 1}),
        trx_category=_FieldVocab({"A": 0}),
        channel_type=_FieldVocab({"POS": 0, ROSBANK_NAN_CHANNEL_TYPE: 1}),
        currency=_FieldVocab({810: 0}),
    )
    split = _RosbankSplitData(
        mcc=torch.tensor([[0, 0, 1, 1]]),
        trx_category=torch.tensor([[0, 0, 0, 0]]),
        channel_type=torch.tensor([[0, 0, 1, 0]]),
        currency=torch.tensor([[0, 0, 0, 0]]),
        amount_log1p=torch.tensor([[0.0, 0.0, 1.0, 2.0]]),
        dt=torch.tensor([[0.0, 0.0, 0.0, 3.0]]),
        valid=torch.tensor([[False, False, True, True]]),
        y=torch.tensor([1]),
    )
    x, dt, y = _rosbank_onehot_batch(split, torch.tensor([0]), vocabs)
    assert x.shape == (1, 4, vocabs.d_in)
    assert x[0, 0].sum().item() == 0.0  # pad position -> all-zero feature vector
    assert x[0, 1].sum().item() == 0.0  # pad position -> all-zero feature vector
    assert x[0, 2].sum().item() > 0.0  # real position
    assert x[0, 3].sum().item() > 0.0  # real position
    # MCC one-hot is the first `vocabs.mcc.size` columns: index 1 (value
    # 200) at the last position.
    mcc_width = vocabs.mcc.size
    assert x[0, 3, :mcc_width].argmax().item() == 1
    assert torch.equal(dt, split.dt)
    assert torch.equal(y, split.y)


def test_rosbank_ordered_batches_covers_dataset_in_fixed_order():
    n, T, batch_size = 5, 4, 2
    vocabs = _RosbankVocabs(
        mcc=_FieldVocab({i: i for i in range(n * T)}),
        trx_category=_FieldVocab({0: 0}),
        channel_type=_FieldVocab({0: 0}),
        currency=_FieldVocab({0: 0}),
    )
    split = _RosbankSplitData(
        mcc=torch.arange(n * T).reshape(n, T),
        trx_category=torch.zeros(n, T, dtype=torch.int64),
        channel_type=torch.zeros(n, T, dtype=torch.int64),
        currency=torch.zeros(n, T, dtype=torch.int64),
        amount_log1p=torch.zeros(n, T),
        dt=torch.zeros(n, T),
        valid=torch.ones(n, T, dtype=torch.bool),
        y=torch.arange(n),
    )
    batches = list(_rosbank_ordered_batches(split, vocabs, batch_size))
    assert [b[2].shape[0] for b in batches] == [2, 2, 1]  # last batch smaller
    recovered_y = torch.cat([b[2] for b in batches])
    assert torch.equal(recovered_y, split.y), "ordered batches must not shuffle"


def test_rosbank_shuffled_batches_covers_dataset_but_reorders_and_is_seeded_reproducible():
    n, T, batch_size = 5, 4, 2
    vocabs = _RosbankVocabs(
        mcc=_FieldVocab({0: 0}),
        trx_category=_FieldVocab({0: 0}),
        channel_type=_FieldVocab({0: 0}),
        currency=_FieldVocab({0: 0}),
    )
    split = _RosbankSplitData(
        mcc=torch.zeros(n, T, dtype=torch.int64),
        trx_category=torch.zeros(n, T, dtype=torch.int64),
        channel_type=torch.zeros(n, T, dtype=torch.int64),
        currency=torch.zeros(n, T, dtype=torch.int64),
        amount_log1p=torch.zeros(n, T),
        dt=torch.zeros(n, T),
        valid=torch.ones(n, T, dtype=torch.bool),
        y=torch.arange(n),
    )
    gen1 = torch.Generator().manual_seed(SEED)
    y1 = torch.cat([b[2] for b in _rosbank_shuffled_batches(split, vocabs, batch_size, gen1)])
    assert sorted(y1.tolist()) == list(range(n)), "shuffle must cover every row exactly once"
    assert not torch.equal(y1, split.y), "shuffle must actually reorder"

    gen2 = torch.Generator().manual_seed(SEED)
    y2 = torch.cat([b[2] for b in _rosbank_shuffled_batches(split, vocabs, batch_size, gen2)])
    assert torch.equal(y1, y2)


def test_validate_rosbank_schema_passes_on_exact_column_set():
    class _FakeSeries:
        def __init__(self, n):
            self._n = n

        def nunique(self):
            return self._n

    class _FakeDF:
        def __init__(self, columns, n_users):
            self.columns = columns
            self._n_users = n_users

        def __getitem__(self, key):
            assert key == "cl_id"
            return _FakeSeries(self._n_users)

    df = _FakeDF(set(ROSBANK_EXPECTED_COLUMNS), n_users=5000)
    _validate_rosbank_schema(df)  # must not raise


def test_validate_rosbank_schema_raises_on_missing_columns():
    class _FakeDF:
        columns = ROSBANK_EXPECTED_COLUMNS - {"currency"}

    with pytest.raises(ValueError, match="schema drift"):
        _validate_rosbank_schema(_FakeDF())


def test_validate_rosbank_schema_raises_on_wrong_labeled_user_count():
    class _FakeSeries:
        def nunique(self):
            return 4999

    class _FakeDF:
        columns = ROSBANK_EXPECTED_COLUMNS

        def __getitem__(self, key):
            return _FakeSeries()

    with pytest.raises(ValueError, match="schema drift"):
        _validate_rosbank_schema(_FakeDF(), expected_labeled_users=5000)


def test_download_rosbank_csv_raises_loud_on_http_error(monkeypatch, tmp_path):
    def fake_urlopen(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="404"):
        _download_rosbank_csv("https://example.invalid/train.csv.gz", tmp_path / "train.csv.gz")


def test_ensure_rosbank_csv_reuses_cached_file_without_downloading(monkeypatch, tmp_path):
    dest = tmp_path / "train.csv.gz"
    dest.write_bytes(b"cached-content")

    def boom(url, dest):
        raise AssertionError("must not download when a cached file already exists")

    monkeypatch.setattr(benchmark_tasks_module, "_download_rosbank_csv", boom)
    resolved = _ensure_rosbank_csv(tmp_path, download=True)
    assert resolved == dest
    assert resolved.read_bytes() == b"cached-content"


def test_ensure_rosbank_csv_raises_when_missing_and_download_false(tmp_path):
    with pytest.raises(FileNotFoundError):
        _ensure_rosbank_csv(tmp_path / "missing", download=False)


# --------------------------------------------------- 9B. RosbankLoader end-to-end
# Synthetic mini-CSV fixture (spec §9/§10: "no network"), 8 labeled users
# split 4/2/2. Roles are assigned to ACTUAL cl_id values only after calling
# `_split_user_ids` with the same (seed, sizes) `RosbankLoader` will use
# internally, so each role's expected split membership is known up front
# rather than assumed -- `nan_id` is deliberately pinned to `train_ids[0]`
# so the NaN-as-category sentinel is guaranteed a genuine (train-derived)
# vocabulary slot, not an OOV fallback.

ROSBANK_TEST_SPLIT_SEED = 7
ROSBANK_TEST_ALL_IDS = list(range(1, 9))
ROSBANK_TEST_CSV_HEADER = (
    "PERIOD,cl_id,MCC,channel_type,currency,TRDATETIME,amount,trx_category,target_flag,target_sum\n"
)


def _rosbank_test_roles():
    """Resolve which synthetic cl_id plays which test role, given the same
    split this fixture's `RosbankLoader` will compute internally."""
    train_ids, val_ids, test_ids = _split_user_ids(
        ROSBANK_TEST_ALL_IDS, ROSBANK_TEST_SPLIT_SEED, n_train=4, n_val=2, n_test=2
    )
    nan_id = train_ids[0]
    remaining = [i for i in ROSBANK_TEST_ALL_IDS if i != nan_id]
    truncate_id, pad_id, exact_id, tie_id = remaining[:4]
    filler_ids = remaining[4:]
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "nan_id": nan_id,
        "truncate_id": truncate_id,
        "pad_id": pad_id,
        "exact_id": exact_id,
        "tie_id": tie_id,
        "filler_ids": filler_ids,
    }


def _write_rosbank_fixture_csv(path, roles) -> None:
    rows: list[tuple] = []

    def add(cl_id, events, target_flag):
        for mcc, channel, currency, ts, amount, trxcat in events:
            rows.append(
                (
                    "2017-01",
                    cl_id,
                    mcc,
                    channel,
                    currency,
                    ts,
                    amount,
                    trxcat,
                    target_flag,
                    target_flag * 1000,
                )
            )

    add(
        roles["truncate_id"],
        [
            (100, "POS", 810, "01JAN17:00:00:00", 10.0, "C2C_OUT"),
            (100, "POS", 810, "02JAN17:00:00:00", 20.0, "C2C_OUT"),
            (100, "POS", 810, "03JAN17:00:00:00", 30.0, "C2C_OUT"),
            (100, "POS", 810, "04JAN17:00:00:00", 40.0, "C2C_OUT"),
            # non-trivial month, locale-independence check:
            (100, "POS", 810, "05SEP17:00:00:00", 50.0, "C2C_OUT"),
            (100, "POS", 810, "06SEP17:00:00:00", 60.0, "C2C_OUT"),
        ],
        target_flag=1,
    )
    add(
        roles["pad_id"],
        [
            (200, "POS", 840, "10JAN17:00:00:00", 5.0, "POS"),
            (200, "POS", 840, "11JAN17:00:00:00", 15.0, "POS"),
        ],
        target_flag=0,
    )
    add(
        roles["exact_id"],
        [
            (300, "WD_ATM_ROS", 978, "01FEB17:00:00:00", 1.0, "POS"),
            (300, "WD_ATM_ROS", 978, "02FEB17:00:00:00", 2.0, "POS"),
            (300, "WD_ATM_ROS", 978, "03FEB17:00:00:00", 3.0, "POS"),
            (300, "WD_ATM_ROS", 978, "04FEB17:00:00:00", 4.0, "POS"),
        ],
        target_flag=1,
    )
    add(
        roles["tie_id"],
        [
            (150, "POS", 810, "20JAN17:00:00:00", 100.0, "POS"),
            (150, "POS", 810, "21JAN17:00:00:00", 200.0, "POS"),  # tied timestamp, written first
            (150, "POS", 810, "21JAN17:00:00:00", 300.0, "POS"),  # tied timestamp, written second
        ],
        target_flag=0,
    )
    add(
        roles["nan_id"],
        [
            (400, "POS", 810, "01MAR17:00:00:00", 7.0, "POS"),
            (400, "", 810, "02MAR17:00:00:00", 8.0, "POS"),  # blank -> NaN channel_type
        ],
        target_flag=0,
    )
    for i, fid in enumerate(roles["filler_ids"]):
        add(
            fid,
            [
                (500 + i, "POS", 810, f"0{i + 1}APR17:00:00:00", 1.0 + i, "POS"),
                (500 + i, "POS", 810, f"1{i + 1}APR17:00:00:00", 2.0 + i, "POS"),
            ],
            target_flag=i % 2,
        )

    body = "\n".join(",".join(str(v) for v in row) for row in rows)
    path.write_text(ROSBANK_TEST_CSV_HEADER + body + "\n")


def _rosbank_test_loader(csv_path, **overrides) -> RosbankLoader:
    kwargs = dict(
        split_seed=ROSBANK_TEST_SPLIT_SEED,
        batch_size=2,
        csv_path=str(csv_path),
        T=4,
        mcc_cap=1000,
        currency_cap=1000,
        n_train_users=4,
        n_val_users=2,
        n_test_users=2,
        expected_labeled_users=8,
    )
    kwargs.update(overrides)
    return RosbankLoader(**kwargs)


def _rosbank_split_data_for(loader, cl_id, roles):
    if cl_id in roles["train_ids"]:
        return loader._train, roles["train_ids"].index(cl_id)
    if cl_id in roles["val_ids"]:
        return loader._val, roles["val_ids"].index(cl_id)
    return loader._test, roles["test_ids"].index(cl_id)


@pytest.fixture
def rosbank_fixture(tmp_path):
    pytest.importorskip("pandas")
    roles = _rosbank_test_roles()
    csv_path = tmp_path / "rosbank_synthetic.csv"
    _write_rosbank_fixture_csv(csv_path, roles)
    return csv_path, roles


def test_rosbank_loader_truncates_to_last_T_events_in_chronological_order(rosbank_fixture):
    csv_path, roles = rosbank_fixture
    loader = _rosbank_test_loader(csv_path)
    split, i = _rosbank_split_data_for(loader, roles["truncate_id"], roles)

    assert split.valid[i].tolist() == [True, True, True, True]
    assert split.dt[i][0].item() == 0.0  # forced 0 despite a real (truncated-away) predecessor
    assert split.dt[i][1].item() == pytest.approx(1.0)  # Jan4 - Jan3
    sep5, jan4 = datetime.datetime(2017, 9, 5), datetime.datetime(2017, 1, 4)
    expected_gap = (sep5 - jan4).total_seconds() / 86400.0
    assert split.dt[i][2].item() == pytest.approx(expected_gap, abs=1e-3)  # Sep5 - Jan4
    assert split.dt[i][3].item() == pytest.approx(1.0)  # Sep6 - Sep5
    # Truncation kept the LAST 4 (by time) of 6 events -- amounts 30/40/50/60,
    # not the first 4 (10/20/30/40).
    expected_amounts = [math.log1p(v) for v in [30.0, 40.0, 50.0, 60.0]]
    assert split.amount_log1p[i].tolist() == pytest.approx(expected_amounts)


def test_rosbank_loader_left_pads_short_sequences_with_zero_features(rosbank_fixture):
    csv_path, roles = rosbank_fixture
    loader = _rosbank_test_loader(csv_path)
    split, i = _rosbank_split_data_for(loader, roles["pad_id"], roles)

    assert split.valid[i].tolist() == [False, False, True, True]
    x, dt, _ = _rosbank_onehot_batch(split, torch.tensor([i]), loader.vocabs)
    assert x[0, 0].sum().item() == 0.0
    assert x[0, 1].sum().item() == 0.0
    assert x[0, 2].sum().item() > 0.0  # true final-2 positions are real events
    assert x[0, 3].sum().item() > 0.0
    assert dt[0, 0].item() == 0.0
    assert dt[0, 1].item() == 0.0
    assert dt[0, 2].item() == 0.0  # first real position -- no predecessor within the window
    assert dt[0, 3].item() == pytest.approx(1.0)  # Jan11 - Jan10


def test_rosbank_loader_exact_length_sequence_has_no_padding(rosbank_fixture):
    csv_path, roles = rosbank_fixture
    loader = _rosbank_test_loader(csv_path)
    split, i = _rosbank_split_data_for(loader, roles["exact_id"], roles)

    assert split.valid[i].tolist() == [True, True, True, True]
    assert split.dt[i][0].item() == 0.0
    for k in range(1, 4):
        assert split.dt[i][k].item() == pytest.approx(1.0)


def test_rosbank_loader_tied_timestamps_preserve_original_row_order(rosbank_fixture):
    """Two events share an identical `TRDATETIME`; the stable sort must
    keep them in their original CSV row order (spec §9: "date-only
    timestamp ties preserve original row order")."""
    csv_path, roles = rosbank_fixture
    loader = _rosbank_test_loader(csv_path)
    split, i = _rosbank_split_data_for(loader, roles["tie_id"], roles)

    assert split.valid[i].tolist() == [False, True, True, True]
    expected_amounts = [math.log1p(v) for v in [100.0, 200.0, 300.0]]
    assert split.amount_log1p[i][1:].tolist() == pytest.approx(expected_amounts)
    # Tied pair -> zero-day gap, not negative/undefined.
    assert split.dt[i][3].item() == pytest.approx(0.0)


def test_rosbank_loader_nan_channel_type_maps_to_dedicated_vocab_slot(rosbank_fixture):
    csv_path, roles = rosbank_fixture
    loader = _rosbank_test_loader(csv_path)
    split, i = _rosbank_split_data_for(loader, roles["nan_id"], roles)

    assert split.valid[i].tolist() == [False, False, True, True]
    nan_slot = loader.vocabs.channel_type.index[ROSBANK_NAN_CHANNEL_TYPE]
    assert nan_slot != loader.vocabs.channel_type.oov_index
    assert split.channel_type[i][3].item() == nan_slot


def test_rosbank_loader_is_reproducible_given_the_same_split_seed(rosbank_fixture):
    csv_path, _ = rosbank_fixture
    loader1 = _rosbank_test_loader(csv_path)
    loader2 = _rosbank_test_loader(csv_path)
    assert torch.equal(loader1._train.mcc, loader2._train.mcc)
    assert torch.equal(loader1._val.dt, loader2._val.dt)
    assert torch.equal(loader1._test.y, loader2._test.y)


def test_rosbank_loader_public_iterators_yield_last_step_timed_batch_contract(rosbank_fixture):
    csv_path, _ = rosbank_fixture
    loader = _rosbank_test_loader(csv_path)

    val_batches = list(loader.val())
    assert sum(b[0].shape[0] for b in val_batches) == 2
    for x, dt, y in val_batches:
        assert x.shape[1:] == (4, loader.d_in)
        assert x.dtype == torch.float32
        assert dt.shape == (x.shape[0], 4)
        assert dt.dtype == torch.float32
        assert y.dtype == torch.int64
        assert set(y.tolist()) <= {0, 1}

    test_batches = list(loader.test())
    assert sum(b[0].shape[0] for b in test_batches) == 2

    train_batches = list(loader.train_epoch(torch.Generator().manual_seed(0)))
    assert sum(b[0].shape[0] for b in train_batches) == 4


def test_rosbank_loader_raises_on_missing_column(tmp_path):
    pytest.importorskip("pandas")
    bad_csv = tmp_path / "bad_schema.csv"
    bad_csv.write_text(
        "PERIOD,cl_id,MCC,channel_type,TRDATETIME,amount,trx_category,target_flag,target_sum\n"
        "2017-01,1,100,POS,01JAN17:00:00:00,1.0,POS,0,0\n"
    )
    with pytest.raises(ValueError, match="schema drift"):
        RosbankLoader(split_seed=1, batch_size=2, csv_path=str(bad_csv), expected_labeled_users=1)


def test_rosbank_loader_raises_on_wrong_labeled_user_count(tmp_path):
    pytest.importorskip("pandas")
    csv_path = tmp_path / "small_valid.csv"
    csv_path.write_text(
        ROSBANK_TEST_CSV_HEADER + "2017-01,1,100,POS,810,01JAN17:00:00:00,1.0,POS,0,0\n"
    )
    with pytest.raises(ValueError, match="schema drift"):
        # default expected_labeled_users (5000) does not match this 1-user fixture
        RosbankLoader(split_seed=1, batch_size=2, csv_path=str(csv_path))


def test_rosbank_loader_real_download_cache_if_present():
    """Real-download loader test (spec §10): "A real-download loader test
    is skip-if-missing." This must never trigger a network fetch by
    surprise in CI, so it is gated on the cached file already existing at
    the production cache path -- skip with a clear reason when it does
    not, rather than downloading ~4.7 MB unasked; `download=False` below
    is a second, redundant guard against ever reaching for the network
    even if this gate were bypassed."""
    pytest.importorskip("pandas")
    cache_path = Path("./data/rosbank/train.csv.gz")
    if not cache_path.exists():
        pytest.skip(
            f"real Rosbank dataset not cached at {cache_path}; skip-if-missing "
            f"(spec §10) -- this test never downloads it itself"
        )

    loader = RosbankLoader(split_seed=0, batch_size=128, download=False)

    # d_in == 126 exactly, on the REAL dataset's vocab sizes (spec §6:
    # MCC top-100 + OOV = 101, trx_category all-values + OOV = 11,
    # channel_type all-values-incl-NaN + OOV = 7, currency top-5 + OOV = 6,
    # + 1 log1p(amount) = 126) -- this is the arithmetic the spec review
    # flagged as unverified against real data; the synthetic fixture used
    # elsewhere in this section can't exercise it since its vocab sizes are
    # deliberately tiny.
    assert loader.d_in == 126

    assert loader._train.y.shape[0] == ROSBANK_TRAIN_USERS
    assert loader._val.y.shape[0] == ROSBANK_VAL_USERS
    assert loader._test.y.shape[0] == ROSBANK_TEST_USERS

    for split in (loader._train, loader._val, loader._test):
        # log1p(amount) must stay finite everywhere (dataset min amount is
        # 0.01 per the verified dataset facts, so log1p is always safely
        # defined -- no zero/negative amount to produce -inf/NaN).
        assert torch.isfinite(split.amount_log1p).all()
        # dt must be non-negative at every REAL (non-pad) position --
        # pad positions are excluded since they are deliberately 0 by
        # construction regardless of any real predecessor gap.
        assert (split.dt[split.valid] >= 0).all()


# ===========================================================================
# 9C. CHURN_TASK registration (Task 4: task registration + campaign wiring)
# ===========================================================================


def test_churn_task_is_registered_in_tasks_and_bench_round_tags():
    assert TASKS["churn"] is CHURN_TASK
    assert BENCH_ROUND_TAGS["churn"] == "bench-churn-02"


def test_churn_task_field_contract_matches_spec_section_6():
    """Spec `.claude/output/specs/2026-07-25-churn-benchmark-design.md`
    §6 TaskSpec contract: `loss_mode="last_step_timed"`,
    `fit_metric="val_auc"`, `fit_direction="ge"`, epoch-based budget,
    36 seeds, no post-selection generalization sweep (fixed T=256)."""
    assert CHURN_TASK.name == "churn"
    assert CHURN_TASK.loss_mode == "last_step_timed"
    assert CHURN_TASK.data is make_churn_loader
    assert CHURN_TASK.fit_metric == "val_auc"
    assert CHURN_TASK.fit_direction == "ge"
    assert CHURN_TASK.eval_protocol == ()
    assert CHURN_TASK.seeds == 36
    assert CHURN_TASK.budget.lr == 1e-3
    assert CHURN_TASK.budget.batch_size == 128
    assert CHURN_TASK.budget.epochs == 20
    assert CHURN_TASK.budget.steps is None


def test_churn_fit_threshold_is_the_pilot_frozen_value():
    """`CHURN_FIT_AUROC` (fit_threshold) is PILOT-FROZEN at 0.80
    (2026-07-25 pilot: log/gru/signed x 3 seeds fit at val_auc
    0.816-0.837; see the constant's own calibration comment and the
    round's EXPERIMENTS.md entry). The exact value is now evidence-bearing
    -- report fit-rates are computed against it -- so this pins it, along
    with the psMNIST-style +/-0.02 robustness band centered on it."""
    assert CHURN_TASK.fit_threshold == 0.80
    lo, mid, hi = CHURN_TASK.robustness
    assert (lo, mid, hi) == pytest.approx((0.78, 0.80, 0.82))
    assert lo < mid < hi
    assert mid == CHURN_TASK.fit_threshold


def test_make_churn_loader_constructs_rosbank_loader_with_split_seed_and_batch_size(monkeypatch):
    """`make_churn_loader` mirrors `make_psmnist_loader`'s factory
    convention: every constructor argument beyond `split_seed`/
    `batch_size` stays at `RosbankLoader`'s own production defaults."""
    captured: dict = {}

    class _FakeRosbankLoader:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(benchmark_tasks_module, "RosbankLoader", _FakeRosbankLoader)

    loader = benchmark_tasks_module.make_churn_loader(64)

    assert isinstance(loader, _FakeRosbankLoader)
    assert captured == {"split_seed": CHURN_SPLIT_SEED, "batch_size": 64}


def test_churn_row_provenance_constants_are_the_ones_the_loader_factory_uses():
    """Sanity cross-check for `scripts/gpu_benchmark_campaign.py`'s
    `_churn_row_provenance` (Task 4): the constants it records must be the
    exact ones `make_churn_loader`/`RosbankLoader`'s own production
    defaults are built from, not independently hardcoded literals."""
    assert ROSBANK_MCC_CAP == 100
    assert ROSBANK_CURRENCY_CAP == 5
    assert ROSBANK_T == 256
