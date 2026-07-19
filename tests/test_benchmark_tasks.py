"""Tests for `experiments/benchmark_tasks.py`: the `TaskSpec` contract and
the S5/MQAR data generators (this benchmark round's Task 1 -- spec
`.claude/output/specs/2026-07-19-benchmark-round-design.md` §9 acceptance
criteria).

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
"""

from __future__ import annotations

import pytest
import torch
from experiments.benchmark_tasks import (
    MQAR_KEY_VOCAB,
    MQAR_TRAIN_PAIRS,
    MQAR_VALUE_VOCAB,
    S5_COMPOSE,
    S5_ELEMENTS,
    Budget,
    EvalConfig,
    TaskSpec,
    make_group_word,
    make_mqar,
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
