"""Benchmark-round task definitions (spec: `.claude/output/specs/2026-07-19-
benchmark-round-design.md`, §5/§6): the `TaskSpec` contract that binds every
task's data, loss mode, and fit/eval protocol to the (later) task-agnostic
benchmark lab driver, plus the data generators for the two group-structured
tasks -- S5 (the symmetric-group word problem) and MQAR (multi-query
associative recall).

This module owns everything task-specific: data synthesis, loss mode, fit
metric, eval protocol, training budget shape. It does not train anything --
that is the lab driver's job (a later task in this round), and it does not
build models -- that is the packaged mixer registry's job.

`TaskSpec` fields fixed by the round's Global Constraints (frozen before any
seed matrix runs, spec §4/§7) are typed here; budget fields still awaiting
pilot calibration (S5's training step count; psMNIST's epoch count and the
pendulum's tau, added by a later task) are represented by `Budget`'s
optional fields rather than guessed numbers -- this task ships the contract,
not the calibrated values.

Generator interface convention (`probes.py`, matched here): every generator
is `make(batch, T, gen) -> tensors`, seeded by a caller-supplied
`torch.Generator` so training/eval streams stay reproducible and disjoint
(spec §7 seeding convention). Return shape follows the batch contract for
the task's `loss_mode` (spec §6):

- `dense` (S5): `(x, y)`, both `(batch, T)` int64.
- `masked_query` (MQAR): `(x, y, mask)`, all `(batch, T)` (`x`/`y` int64,
  `mask` bool); loss/accuracy apply only where `mask` is true.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

# ------------------------------------------------------------- TaskSpec

LossMode = Literal["dense", "masked_query", "last_step", "regression"]
FitMetric = Literal["val128", "val_qacc", "val_acc", "val_mse"]
FitDirection = Literal["ge", "le"]


@dataclass(frozen=True)
class EvalConfig:
    """One generalization-eval point in a `TaskSpec.eval_protocol` tuple.

    `T` is the sequence length every task's eval protocol varies. `num_pairs`
    is MQAR-specific (the number of key-value pairs presented before the
    query block, spec §4: "T=256 with 16 and 32 pairs") and stays `None` for
    tasks with no pair-count axis (S5, and the later psMNIST/pendulum
    specs).
    """

    T: int
    num_pairs: int | None = None


@dataclass(frozen=True)
class Budget:
    """Training budget + optimizer settings for one `TaskSpec` (spec §6:
    "steps or epochs; optimizer settings").

    Step-based tasks (S5, MQAR, pendulum) set `steps`; psMNIST's epoch-based
    loop sets `epochs` instead -- exactly one of the two is non-`None` for
    any one instance, and the driver dispatches on whichever is set.
    `eval_every` is the checkpoint-selection cadence for step-based tasks
    (mirrors `probes.py`'s `EVAL_EVERY`); left `None` for epoch-based tasks,
    which checkpoint once per epoch instead.

    Values here are pilot-calibrated (spec §4) before any seed matrix runs
    and frozen afterward; this module does not hardcode calibrated numbers
    for S5 (not yet piloted) -- concrete `TaskSpec` instances are built by
    the pilot/lab-driver task once numbers are frozen.
    """

    lr: float
    batch_size: int
    steps: int | None = None
    epochs: int | None = None
    eval_every: int | None = None


@dataclass(frozen=True)
class TaskSpec:
    """Binds one benchmark task's data, loss mode, and fit/eval protocol to
    the task-agnostic lab driver (spec §5: "Owns everything task-specific:
    data synthesis or loading, loss mode, fit metric, eval protocol,
    training budget.").

    Parameters
    ----------
    name : str
        `"s5" | "mqar" | "psmnist" | "pendulum"`.
    loss_mode : LossMode
        Selects the batch contract (module docstring / spec §6) the driver
        applies to `data`'s output.
    data : Callable
        `make(batch, T, gen) -> tensors` for the synthetic tasks (S5, MQAR,
        pendulum); an epoch loader for psMNIST.
    fit_metric : FitMetric
        The ledger `ckpt` key a trained seed is selected/judged on.
    fit_threshold : float
        The value `fit_metric` must clear (per `fit_direction`) for a seed
        to count as fit.
    fit_direction : FitDirection
        `"ge"`: fit iff `metric >= fit_threshold` (S5, MQAR, psMNIST).
        `"le"`: fit iff `metric <= fit_threshold` (pendulum MSE).
    robustness : tuple[float, float, float]
        The threshold-robustness triple (spec §4) tested alongside
        `fit_threshold`.
    eval_protocol : tuple[EvalConfig, ...]
        Generalization-eval configurations (lengths, and pair counts where
        applicable) run after checkpoint selection.
    budget : Budget
        Training budget + optimizer settings.
    seeds : int
        Seed-matrix size for this task (36 or 12, spec §2).
    """

    name: str
    loss_mode: LossMode
    data: Callable[..., tuple[torch.Tensor, ...]]
    fit_metric: FitMetric
    fit_threshold: float
    fit_direction: FitDirection
    robustness: tuple[float, float, float]
    eval_protocol: tuple[EvalConfig, ...]
    budget: Budget
    seeds: int


# ----------------------------------------------------------------- S5
# Generic group-word generator (probes.py's `_compose_table`/`make_s3`
# pattern, probes.py:196-214, generalized from the fixed S3 element list to
# any permutation-group Cayley table), instantiated below for S5. S3-hier's
# Latin-square pair-function front-end is deliberately NOT reused here --
# S5 is the plain running-product word problem the spec calls for (spec §4:
# "the label at each position is the running left-composition product").


def _permutation_group(n: int) -> torch.Tensor:
    """All `n!` permutations of `range(n)` as rows, `itertools.permutations`
    order (lexicographic over a sorted input) -- row 0 is therefore always
    the identity permutation `(0, 1, ..., n-1)`, matching `probes.py`'s S3
    element list convention (identity at index 0)."""
    perms = list(itertools.permutations(range(n)))
    return torch.tensor(perms, dtype=torch.long)


def _compose_table(elements: torch.Tensor) -> torch.Tensor:
    """Cayley table for left-composition over `elements` (each row a
    permutation): `table[i, j]` is the index within `elements` of
    `elements[i][elements[j]]` (i.e. `p_i o p_j`).

    Generalizes `probes.py`'s S3-specific `_compose_table` (probes.py:196)
    from the fixed 6-element S3 list to any permutation set closed under
    composition -- `elements` must already be closed (true for
    `_permutation_group(n)`, the full symmetric group).
    """
    index_of = {tuple(p.tolist()): k for k, p in enumerate(elements)}
    n = elements.shape[0]
    table = torch.zeros(n, n, dtype=torch.long)
    for i in range(n):
        for j in range(n):
            composed = elements[i][elements[j]]
            table[i, j] = index_of[tuple(composed.tolist())]
    return table


S5_ELEMENTS = _permutation_group(5)  # (120, 5); row 0 is the identity.
S5_COMPOSE = _compose_table(S5_ELEMENTS)  # (120, 120) Cayley table.


def make_group_word(table: torch.Tensor) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Build a `dense`-loss-mode group-word generator from a Cayley `table`
    (`probes.py`'s `make_s3`, probes.py:207-214, generalized to any group
    table): tokens are group-element ids; the label at each position is the
    running left-composition product `g_t o r_{t-1}` (identity before the
    first token).

    Returns
    -------
    Callable
        `make(batch, T, gen) -> (x, y)`, both `(batch, T)` int64.
    """
    n = table.shape[0]

    def make(batch: int, T: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(0, n, (batch, T), generator=gen)
        y = torch.zeros_like(x)
        state = torch.zeros(batch, dtype=torch.long)  # identity
        for t in range(T):
            state = table[x[:, t], state]  # g_t o r_{t-1}
            y[:, t] = state
        return x, y

    return make


make_s5 = make_group_word(S5_COMPOSE)


# ---------------------------------------------------------------- MQAR
# Multi-query associative recall (spec §4): a sequence presents `num_pairs`
# key-value pairs (interleaved key, value tokens) followed, later in the
# same sequence, by the same keys in permuted order -- the query block.
# Loss/accuracy apply only at the query positions (masked_query loss mode).
# Presentation occupies the first `2*num_pairs` positions; the query block
# occupies the last `num_pairs` positions; everything between is random
# filler the model must ignore, giving eval configs (T=256, num_pairs in
# {16, 32}) a much longer presentation-to-query gap than the T=64/8-pair
# training configuration -- the recall-distance stress the eval protocol is
# meant to exercise.

MQAR_KEY_VOCAB = 32
MQAR_VALUE_VOCAB = 32
MQAR_TRAIN_PAIRS = 8


def make_mqar(
    batch: int,
    T: int,
    gen: torch.Generator,
    num_pairs: int = MQAR_TRAIN_PAIRS,
    key_vocab: int = MQAR_KEY_VOCAB,
    value_vocab: int = MQAR_VALUE_VOCAB,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MQAR generator (`masked_query` loss mode): key ids occupy
    `[0, key_vocab)`, value ids occupy `[key_vocab, key_vocab + value_vocab)`
    (disjoint ranges -- a token's range alone identifies whether it is a key
    or a value).

    Per row: `num_pairs` distinct keys are drawn without replacement (so
    every presented key is unambiguous), each paired with an independently
    drawn value; the presentation block interleaves `(key, value)` at
    positions `[0, 2*num_pairs)`. The query block at positions
    `[T - num_pairs, T)` replays the same keys in an independently permuted
    order, with `y`/`mask` set only there. Positions in between are
    uniform random filler over the combined vocab (unmasked, so their
    exact value never affects the task).

    Parameters
    ----------
    batch, T : int
        Batch size and sequence length.
    gen : torch.Generator
        Seeds every random draw (key/value sampling, permutations, filler).
    num_pairs : int
        Number of key-value pairs (8 at training; 16 or 32 at eval per the
        eval protocol).
    key_vocab, value_vocab : int
        Sizes of the disjoint key/value id ranges (32 each per spec §4).

    Returns
    -------
    tuple of torch.Tensor
        `(x, y, mask)`, each `(batch, T)` (`x`/`y` int64, `mask` bool).
        `mask` is true only at the trailing `num_pairs` query positions.

    Raises
    ------
    ValueError
        If `num_pairs` exceeds `key_vocab` (not enough distinct keys) or if
        the presentation-plus-query span (`3 * num_pairs`) does not fit
        within `T`.
    """
    if num_pairs > key_vocab:
        raise ValueError(
            f"num_pairs={num_pairs} exceeds key_vocab={key_vocab}: cannot draw "
            f"that many distinct keys without replacement"
        )
    needed = 3 * num_pairs
    if needed > T:
        raise ValueError(
            f"presentation ({2 * num_pairs}) + query ({num_pairs}) block needs "
            f"{needed} positions, exceeding T={T}"
        )

    total_vocab = key_vocab + value_vocab

    # Per-row random permutation of key ids (torch.randperm has no batched
    # form under a shared generator): argsort of iid uniform noise gives an
    # independent permutation per row.
    key_perm = torch.rand(batch, key_vocab, generator=gen).argsort(dim=-1)
    keys = key_perm[:, :num_pairs]  # (batch, num_pairs), distinct per row
    values = torch.randint(0, value_vocab, (batch, num_pairs), generator=gen) + key_vocab

    query_order = torch.rand(batch, num_pairs, generator=gen).argsort(dim=-1)
    query_keys = torch.gather(keys, 1, query_order)
    query_values = torch.gather(values, 1, query_order)

    x = torch.randint(0, total_vocab, (batch, T), generator=gen)
    y = torch.zeros(batch, T, dtype=torch.long)
    mask = torch.zeros(batch, T, dtype=torch.bool)

    x[:, 0 : 2 * num_pairs : 2] = keys
    x[:, 1 : 2 * num_pairs : 2] = values
    x[:, T - num_pairs : T] = query_keys
    y[:, T - num_pairs : T] = query_values
    mask[:, T - num_pairs : T] = True
    return x, y, mask
