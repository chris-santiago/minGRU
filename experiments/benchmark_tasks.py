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

Generator interface convention (`probes.py`, matched here): every synthetic
task's generator is `make(batch, T, gen) -> tensors`, seeded by a
caller-supplied `torch.Generator` so training/eval streams stay
reproducible and disjoint (spec §7 seeding convention). psMNIST is the one
task whose `data` is not this kind of generator -- an epoch dataset owns a
fixed split and a fixed pixel permutation that must persist across calls,
so its `data` value is a `Loader` instance instead (spec §6: `data:
Callable | Loader`). Return/yield shape follows the batch contract for the
task's `loss_mode` (spec §6):

- `dense` (S5): `(x, y)`, both `(batch, T)` int64.
- `masked_query` (MQAR): `(x, y, mask)`, all `(batch, T)` (`x`/`y` int64,
  `mask` bool); loss/accuracy apply only where `mask` is true.
- `last_step` (psMNIST): `(x, y)`, `x` float `(batch, T, d_in)`, `y` int64
  `(batch,)`; loss/accuracy apply only at the final position.
- `regression` (pendulum): `(x, dt, y)`, `x`/`y` float `(batch, T, d_in)`/
  `(batch, T, d_out)`, `dt` float `(batch, T)` feeding the decay channel via
  the existing timestamped-input path (`probes.py`'s `TIMESTAMPED_TASKS`
  convention: `dt[:, 0] == 0`, nothing precedes the first position).
- `last_step_timed` (churn, spec `.claude/output/specs/2026-07-25-churn-
  benchmark-design.md` §6): `(x, dt, y)`, `x` float `(batch, 256, 126)`, `dt`
  float `(batch, 256)` (day-gaps between consecutive real events; position 0
  of the window and every pad position are 0, same no-predecessor
  convention as `regression`), `y` int64 `(batch,)` in `{0, 1}`; loss/AUROC
  apply only at the final position, which left-padding guarantees is always
  a real event (see `RosbankLoader`).
"""

from __future__ import annotations

import itertools
import math
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch
import torch.nn.functional as F

# ------------------------------------------------------------- TaskSpec

LossMode = Literal["dense", "masked_query", "last_step", "regression", "last_step_timed"]
FitMetric = Literal["val128", "val_qacc", "val_acc", "val_mse", "val_auc"]
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


class Loader(Protocol):
    """Epoch-based counterpart to the synthetic tasks' `make(batch, T, gen)`
    generators (spec §6: `data: Callable | Loader`). psMNIST's `data` is an
    instance of this protocol rather than a stateless per-call callable,
    since an epoch dataset owns a fixed split and a fixed pixel permutation
    that must not be regenerated on every batch (module docstring).

    Every method yields `last_step` batches (module docstring): `x` float
    `(batch, T, d_in)`, `y` int64 `(batch,)`.

    `RosbankLoader` (churn, `last_step_timed`) shares this exact
    `train_epoch`/`val`/`test` method shape and the same epoch-based-
    dataset-owns-its-own-fixed-preprocessing rationale, but yields 3-tuple
    `(x, dt, y)` batches instead -- documented on its own class docstring
    rather than a second literal `Protocol` here, since that would just
    duplicate this docstring's rationale for a different tuple arity.
    """

    def train_epoch(self, gen: torch.Generator) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """One shuffled pass over the training split, reshuffled fresh each
        call via `gen`."""
        ...

    def val(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """The full validation split, in a fixed (non-shuffled) order."""
        ...

    def test(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """The full test split, in a fixed (non-shuffled) order."""
        ...


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
    data : Callable | Loader
        `make(batch, T, gen) -> tensors` for the synthetic tasks (S5, MQAR,
        pendulum); a `Loader` instance (epoch-based) for psMNIST.
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
    data: Callable[..., tuple[torch.Tensor, ...]] | Loader
    fit_metric: FitMetric
    fit_threshold: float
    fit_direction: FitDirection
    robustness: tuple[float, float, float]
    eval_protocol: tuple[EvalConfig, ...]
    budget: Budget
    seeds: int


# ------------------------------------------------------------- metrics
# `val_auc` fit metric (churn round, spec §6 metric contract): a pure-torch
# rank-based AUROC, no sklearn dependency (spec §8 key-decisions). Kept
# task-agnostic (not folded into any one `TaskSpec`) since a later task in
# this round is the first and only consumer, wiring it into the lab
# driver's checkpoint-selection/eval path for `loss_mode="last_step_timed"`.


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the ROC curve via the Mann-Whitney U rank statistic,
    average ranks for ties (spec §6: "AUROC by Mann-Whitney rank statistic
    with average ranks for ties; direction ge").

    Equivalent to the pairwise definition (fraction of positive/negative
    pairs the positive outranks, ties counting as half a win) but computed
    in O(n log n) via one sort instead of the O(n_pos * n_neg) pairwise
    comparison: `U = sum(rank(positives)) - n_pos*(n_pos+1)/2`, `AUC = U /
    (n_pos * n_neg)`. Tied scores (including an entire constant-score input)
    receive the mean of the ranks their tied group spans, via
    `torch.unique_consecutive` over the sorted scores -- the same
    tie-averaging `scipy.stats.rankdata(method="average")` performs, kept
    here as batched tensor ops rather than a Python loop over positions.

    Parameters
    ----------
    scores : torch.Tensor
        Real-valued scores, one per example, any shape (flattened via
        `reshape(-1)`) (spec §6: "positive-class logit minus negative-class
        logit at the final position").
    labels : torch.Tensor
        Same element count as `scores`, values in `{0, 1}`.

    Returns
    -------
    float
        AUROC in `[0, 1]`; 1.0 is perfect ranking, 0.5 is chance
        (including the fully-tied degenerate case), 0.0 is perfectly
        inverted.

    Raises
    ------
    ValueError
        If `scores` and `labels` have a different number of elements, or if
        `labels` contains only one class (spec §6: "Degenerate single-class
        eval sets are a hard error" -- AUROC is undefined with no
        positive/negative pair to rank).
    """
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError(
            f"scores and labels must have the same number of elements, got "
            f"{scores.shape[0]} and {labels.shape[0]}"
        )

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"auroc is undefined for single-class input: got {n_pos} positive and "
            f"{n_neg} negative labels, need at least one of each"
        )

    sorted_scores, order = torch.sort(scores)
    n = sorted_scores.shape[0]

    # Average-rank ties: `unique_consecutive` groups runs of equal sorted
    # values; every position in a group is assigned the mean of the
    # (1-based) ranks that group spans -- e.g. two tied entries at sorted
    # positions 0,1 (ranks 1,2) both get rank 1.5. Untied, this reduces to
    # the plain 1..n rank sequence.
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    group_ends = torch.cumsum(counts, dim=0)
    group_starts = group_ends - counts
    group_avg_rank = (group_starts + group_ends + 1).to(torch.float64) / 2.0
    avg_ranks = torch.repeat_interleave(group_avg_rank, counts)

    ranks_by_original_position = torch.empty(n, dtype=torch.float64, device=scores.device)
    ranks_by_original_position[order] = avg_ranks

    sum_ranks_pos = ranks_by_original_position[labels == 1].sum()
    u_stat = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return (u_stat / (n_pos * n_neg)).item()


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
# filler drawn from the value range only (never the key range -- see
# `make_mqar`'s docstring) that the model must ignore, giving eval configs
# (T=256, num_pairs in {16, 32}) a much longer presentation-to-query gap
# than the T=64/8-pair training configuration -- the recall-distance
# stress the eval protocol is meant to exercise.

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
    uniform random filler drawn from the VALUE range only, `[key_vocab,
    key_vocab + value_vocab)` -- never the key range -- so a filler token
    can never open a spurious key-value binding (pre-matrix technical
    review fix: filler previously drew from the full combined vocab,
    fabricating spurious key-range fillers whose density scaled with `T`
    -- see `x`'s assignment below). Filler is still unmasked, so its exact
    value never reaches the loss; only its RANGE is now constrained.

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

    # Per-row random permutation of key ids (torch.randperm has no batched
    # form under a shared generator): argsort of iid uniform noise gives an
    # independent permutation per row.
    key_perm = torch.rand(batch, key_vocab, generator=gen).argsort(dim=-1)
    keys = key_perm[:, :num_pairs]  # (batch, num_pairs), distinct per row
    values = torch.randint(0, value_vocab, (batch, num_pairs), generator=gen) + key_vocab

    query_order = torch.rand(batch, num_pairs, generator=gen).argsort(dim=-1)
    query_keys = torch.gather(keys, 1, query_order)
    query_values = torch.gather(values, 1, query_order)

    # Filler baseline drawn from the VALUE range only ([key_vocab,
    # key_vocab + value_vocab)) -- a value token can never open a spurious
    # key-value binding, unlike drawing from the full combined vocab (the
    # pre-review behavior this replaces). Presentation/query assignments
    # below overwrite this baseline at their own positions; every position
    # this function does NOT explicitly assign a key to therefore stays a
    # value-range token.
    x = torch.randint(0, value_vocab, (batch, T), generator=gen) + key_vocab
    y = torch.zeros(batch, T, dtype=torch.long)
    mask = torch.zeros(batch, T, dtype=torch.bool)

    x[:, 0 : 2 * num_pairs : 2] = keys
    x[:, 1 : 2 * num_pairs : 2] = values
    x[:, T - num_pairs : T] = query_keys
    y[:, T - num_pairs : T] = query_values
    mask[:, T - num_pairs : T] = True
    return x, y, mask


# ------------------------------------------------------------- Pendulum
# Angle-observation pendulum regression (spec §4, "CRU-protocol-inspired"
# per the design spec's key-decisions section): a frictionless simple
# pendulum is ODE-simulated and observed as noisy (sin theta, cos theta)
# pairs at irregularly spaced timestamps; the model predicts the true
# (noiseless) (sin theta, cos theta) pair at each observation -- a
# denoising target at the same 2-d observation space as `x`, so `d_out ==
# d_in == 2` (the spec's "true state at each observation" read as the
# state the observation itself encodes, position only -- angular velocity
# is never observed and is not part of the regression target). `dt` feeds
# the decay channel via the existing timestamped-input path (`probes.py`'s
# TIMESTAMPED_TASKS convention): `regression` loss mode, MSE over all
# positions (spec §6).

PENDULUM_GRAVITY = 9.81  # m/s^2
PENDULUM_LENGTH = 1.0  # m
PENDULUM_THETA0_RANGE = (-1.0, 1.0)  # rad; single pendulum is non-chaotic at any amplitude
PENDULUM_OMEGA0_RANGE = (-1.0, 1.0)  # rad/s
PENDULUM_DT_RANGE = (0.05, 0.3)  # s; strictly positive -> strictly increasing timestamps
PENDULUM_OBS_NOISE_STD = 0.05
PENDULUM_SUBSTEPS = 20  # RK4 micro-steps per observation gap (integration accuracy)


def _pendulum_derivative(
    theta: torch.Tensor, omega: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frictionless simple-pendulum ODE: theta' = omega, omega' =
    -(g/L) sin(theta)."""
    theta_dot = omega
    omega_dot = -(PENDULUM_GRAVITY / PENDULUM_LENGTH) * torch.sin(theta)
    return theta_dot, omega_dot


def _rk4_step(
    theta: torch.Tensor, omega: torch.Tensor, h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One classical RK4 step of step size `h` (elementwise per batch row,
    so rows may advance by different gaps)."""
    k1_th, k1_om = _pendulum_derivative(theta, omega)
    k2_th, k2_om = _pendulum_derivative(theta + 0.5 * h * k1_th, omega + 0.5 * h * k1_om)
    k3_th, k3_om = _pendulum_derivative(theta + 0.5 * h * k2_th, omega + 0.5 * h * k2_om)
    k4_th, k4_om = _pendulum_derivative(theta + h * k3_th, omega + h * k3_om)
    theta_new = theta + (h / 6.0) * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th)
    omega_new = omega + (h / 6.0) * (k1_om + 2.0 * k2_om + 2.0 * k3_om + k4_om)
    return theta_new, omega_new


def _integrate_pendulum(
    theta0: torch.Tensor,
    omega0: torch.Tensor,
    dt: torch.Tensor,
    n_substeps: int = PENDULUM_SUBSTEPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate the pendulum ODE forward from `(theta0, omega0)` across
    each row's per-position gap `dt` (batch, T): position 0 is the initial
    condition (`dt[:, 0]` is ignored -- nothing precedes the first
    position, matching `probes.py`'s no-t=0-exemption convention); each
    subsequent gap is substepped into `n_substeps` fixed micro-steps of
    RK4 for integration accuracy.

    Returns
    -------
    tuple of torch.Tensor
        `(theta, omega)`, each `(batch, T)`: the state trajectory.
    """
    batch, T = dt.shape
    theta = theta0.clone()
    omega = omega0.clone()
    theta_traj = torch.empty(batch, T, dtype=theta0.dtype)
    omega_traj = torch.empty(batch, T, dtype=omega0.dtype)
    theta_traj[:, 0] = theta
    omega_traj[:, 0] = omega
    for t in range(1, T):
        h = dt[:, t] / n_substeps
        for _ in range(n_substeps):
            theta, omega = _rk4_step(theta, omega, h)
        theta_traj[:, t] = theta
        omega_traj[:, t] = omega
    return theta_traj, omega_traj


def make_pendulum(
    batch: int,
    T: int,
    gen: torch.Generator,
    dt_range: tuple[float, float] = PENDULUM_DT_RANGE,
    obs_noise_std: float = PENDULUM_OBS_NOISE_STD,
    n_substeps: int = PENDULUM_SUBSTEPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`regression`-loss-mode pendulum generator (spec §4): per row, an
    independent initial condition is drawn and the frictionless pendulum
    ODE is integrated across independently drawn, strictly positive gaps
    (`dt_range`) -- irregular and strictly increasing once accumulated
    into timestamps. Observations are the true `(sin theta, cos theta)`
    pair plus iid Gaussian noise (`obs_noise_std`); the label is the
    noiseless pair at that same position.

    Parameters
    ----------
    batch, T : int
        Batch size and sequence length.
    gen : torch.Generator
        Seeds every random draw (initial conditions, gaps, observation
        noise).
    dt_range : tuple[float, float]
        Per-position gap range (seconds); position 0's gap is forced to 0
        (no preceding position).
    obs_noise_std : float
        Standard deviation of the additive Gaussian observation noise.
    n_substeps : int
        RK4 micro-steps per observation gap (integration accuracy).

    Returns
    -------
    tuple of torch.Tensor
        `(x, dt, y)`: `x`/`y` float `(batch, T, 2)`, `dt` float
        `(batch, T)`.
    """
    theta0 = torch.empty(batch).uniform_(*PENDULUM_THETA0_RANGE, generator=gen)
    omega0 = torch.empty(batch).uniform_(*PENDULUM_OMEGA0_RANGE, generator=gen)

    dt = torch.empty(batch, T).uniform_(*dt_range, generator=gen)
    dt[:, 0] = 0.0  # position 0 is the initial condition (probes.py convention)

    theta, _ = _integrate_pendulum(theta0, omega0, dt, n_substeps)

    true_state = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)  # (batch, T, 2)
    noise = torch.empty(batch, T, 2).normal_(mean=0.0, std=obs_noise_std, generator=gen)
    x = true_state + noise
    y = true_state
    return x, dt, y


# ----------------------------------------------------------------- psMNIST
# Permuted-sequential MNIST (spec §4): MNIST pixels flattened to a
# length-784 scalar sequence under one fixed permutation, single 10-class
# label read at the final position (`last_step` loss mode). torchvision is
# an optional import -- only `PsMNISTLoader.__init__` (which downloads and
# reads the dataset) imports it; the permutation/split/batch helpers below
# are pure tensor functions with no torchvision dependency, so they (and
# this module) are testable/importable without it installed.

PSMNIST_T = 28 * 28  # 784: MNIST pixels flattened (spec §4)
PSMNIST_TRAIN_SIZE = 50_000
PSMNIST_VAL_SIZE = 10_000
# Test split is torchvision's MNIST test set as-is (10k rows, spec §4);
# there is no size constant for it since nothing slices it.


def make_psmnist_permutation(seed: int) -> torch.Tensor:
    """The one fixed pixel permutation for psMNIST (spec §4: "one fixed
    permutation from a recorded seed identical across arms and seeds"). A
    pure function of `seed` -- reproducible across process runs, testable
    without any dataset dependency.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(PSMNIST_T, generator=gen)


def _prepare_split(
    images: torch.Tensor, labels: torch.Tensor, permutation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten `images` (N, 28, 28) uint8 to (N, 784) scalar pixels, scale
    to [0, 1], apply the fixed pixel `permutation`, and add the trailing
    feature dim the `last_step` contract expects (spec §6: `x` float
    `(batch, T, d_in)`, here `d_in == 1`).

    Pure tensor transform (no torchvision dependency), so split/batch
    logic is testable with synthetic images.
    """
    n = images.shape[0]
    flat = images.reshape(n, PSMNIST_T).float() / 255.0
    x = flat[:, permutation].unsqueeze(-1)  # (N, T, 1)
    y = labels.long()
    return x, y


def _split_train_val(
    images: torch.Tensor,
    labels: torch.Tensor,
    train_size: int = PSMNIST_TRAIN_SIZE,
    val_size: int = PSMNIST_VAL_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split MNIST's 60k training rows into the 50k/10k train/val split
    (spec §4): the first `train_size` rows are train, the next `val_size`
    are val. torchvision's MNIST ships in a fixed (non-shuffled) order, so
    no reshuffle happens here -- shuffling is `train_epoch`'s job, applied
    fresh every epoch.
    """
    train_images, train_labels = images[:train_size], labels[:train_size]
    val_images = images[train_size : train_size + val_size]
    val_labels = labels[train_size : train_size + val_size]
    return train_images, train_labels, val_images, val_labels


def _ordered_batches(
    x: torch.Tensor, y: torch.Tensor, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield `(x_batch, y_batch)` slices of `batch_size` in dataset order
    (last batch may be smaller) -- val/test, so repeated evaluation sees a
    stable, unshuffled order."""
    n = x.shape[0]
    for start in range(0, n, batch_size):
        yield x[start : start + batch_size], y[start : start + batch_size]


def _shuffled_batches(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, gen: torch.Generator
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """`_ordered_batches` over a fresh per-call shuffle of `x`/`y`, seeded
    by `gen` -- training, reshuffled every epoch."""
    order = torch.randperm(x.shape[0], generator=gen)
    yield from _ordered_batches(x[order], y[order], batch_size)


class PsMNISTLoader:
    """`Loader` (spec §6) for psMNIST (spec §4): downloads MNIST via
    torchvision, applies the fixed pixel permutation recorded by
    `permutation_seed`, and splits into the 50k/10k/10k train/val/test used
    everywhere in this round.

    torchvision is imported lazily inside `__init__`, not at module level,
    so constructing this loader is the only thing that requires it
    installed (module docstring / spec-and-plan requirement).
    """

    def __init__(
        self,
        permutation_seed: int,
        batch_size: int,
        root: str = "./data",
        download: bool = True,
    ) -> None:
        import torchvision  # local import: optional dependency (module docstring)

        self.permutation_seed = permutation_seed
        self.batch_size = batch_size
        self.permutation = make_psmnist_permutation(permutation_seed)

        train_full = torchvision.datasets.MNIST(root=root, train=True, download=download)
        test_set = torchvision.datasets.MNIST(root=root, train=False, download=download)

        train_images, train_labels, val_images, val_labels = _split_train_val(
            train_full.data, train_full.targets
        )
        self._train_x, self._train_y = _prepare_split(train_images, train_labels, self.permutation)
        self._val_x, self._val_y = _prepare_split(val_images, val_labels, self.permutation)
        self._test_x, self._test_y = _prepare_split(
            test_set.data, test_set.targets, self.permutation
        )

    def train_epoch(self, gen: torch.Generator) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        yield from _shuffled_batches(self._train_x, self._train_y, self.batch_size, gen)

    def val(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        yield from _ordered_batches(self._val_x, self._val_y, self.batch_size)

    def test(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        yield from _ordered_batches(self._test_x, self._test_y, self.batch_size)


PSMNIST_PERMUTATION_SEED = 20260719  # spec §4: one fixed permutation, identical across every arm and seed this round; recorded in each row's `config`.


def make_psmnist_loader(batch_size: int) -> PsMNISTLoader:
    """`PsMNISTLoader` factory keyed only by `batch_size` (the lab driver's
    "wiring site", quality-review carry-over from Task 2: `PsMNISTLoader`
    owns its own `batch_size` with no enforced link to a `TaskSpec.budget`'s
    `batch_size` -- this factory lets the driver construct the loader AFTER
    resolving CLI budget overrides, then assert the two values match, rather
    than baking a batch size into `PSMNIST_TASK` at import time).

    `PSMNIST_TASK.data` holds this factory (not a `Loader` instance and not
    the `make(batch, T, gen)` generator shape every other task's `data`
    uses) -- the one documented exception the module docstring already
    calls out for psMNIST; the driver's `loss_mode == "last_step"` dispatch
    is what knows to call it this way.
    """
    return PsMNISTLoader(permutation_seed=PSMNIST_PERMUTATION_SEED, batch_size=batch_size)


# ----------------------------------------------------------------- Rosbank
# Rosbank churn (spec `.claude/output/specs/2026-07-25-churn-benchmark-
# design.md` §4/§6): per-user binary churn prediction from a bank's
# transaction event stream (`pytorch-lifestream/rosbank-churn` on the HF
# Hub). `RosbankLoader` owns download, parsing, per-user chronological
# sorting, the seeded 4000/500/500 user split, train-split-derived
# categorical vocabularies (with OOV/NaN buckets), and last-256-event
# truncation with left-padding -- everything up to but not including the
# per-batch one-hot expansion (spec §5 architecture: "integer-encoded
# sequences are precomputed once, one-hot expansion happens per batch").
#
# pandas is imported lazily inside `RosbankLoader.__init__` only (the same
# torchvision convention `PsMNISTLoader` already uses above): every helper
# function below either touches no pandas object at all (pure Python/torch),
# or receives an already-constructed pandas `DataFrame`/`Series` and calls
# only ITS methods (`.groupby`, `.sort_values`, `.isin`, `.fillna`, `.tolist`
# -- none of which require the `pandas` module itself in scope), so this
# module stays importable without pandas installed; only actually
# CONSTRUCTING a `RosbankLoader` requires it.

ROSBANK_DATASET_ID = "pytorch-lifestream/rosbank-churn"
ROSBANK_REVISION = "aad1b512deaee7ffea4c289e496785bc9e17bd80"  # pinned HF revision (spec §4)
ROSBANK_RESOLVE_URL = (
    f"https://huggingface.co/datasets/{ROSBANK_DATASET_ID}/resolve/{ROSBANK_REVISION}/train.csv.gz"
)
ROSBANK_EXPECTED_COLUMNS = frozenset(
    {
        "PERIOD",
        "cl_id",
        "MCC",
        "channel_type",
        "currency",
        "TRDATETIME",
        "amount",
        "trx_category",
        "target_flag",
        "target_sum",
    }
)
ROSBANK_EXPECTED_LABELED_USERS = 5_000
ROSBANK_TRAIN_USERS = 4_000
ROSBANK_VAL_USERS = 500
ROSBANK_TEST_USERS = 500
ROSBANK_T = 256  # last-256-event truncation window (spec §4/§6)
ROSBANK_MCC_CAP = 100  # top-100-by-train-frequency + OOV (spec §6 feature layout)
ROSBANK_CURRENCY_CAP = 5  # top-5-by-train-frequency + OOV (spec §6 feature layout)
ROSBANK_TRDATETIME_FORMAT = "%d%b%y:%H:%M:%S"  # e.g. "05SEP17:00:00:00" (spec §4)
# NaN-as-category sentinel (spec §4/§6); never a real HF channel_type string.
ROSBANK_NAN_CHANNEL_TYPE = "<rosbank-nan-channel-type>"


def _download_rosbank_csv(url: str, dest: Path) -> None:
    """Fetch `url` and write it to `dest` (spec §4: "download failure ...
    fails loud; there is no silent fallback"). Downloads to a `.part`
    sidecar first and atomically renames into place, so a failed/killed
    download never leaves a corrupt file at `dest` for a later run to treat
    as a valid cache hit.

    Raises
    ------
    RuntimeError
        On any HTTP or network failure (404, other HTTP error, DNS/connect
        failure), chained from the underlying `urllib.error` exception.
    """
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 -- fixed https:// HF resolve URL, not user input
            payload = response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Rosbank dataset download failed ({e.code} {e.reason}) from {url}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Rosbank dataset download failed ({e.reason}) from {url}") from e
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)


def _ensure_rosbank_csv(root: Path, download: bool) -> Path:
    """Return the path to a cached `train.csv.gz` under `root`, downloading
    it first (pinned revision, spec §4) if not already cached and `download`
    is true; raises if missing and `download` is false."""
    dest = root / "train.csv.gz"
    if dest.exists():
        return dest
    if not download:
        raise FileNotFoundError(
            f"Rosbank dataset not found at {dest} and download=False; pass "
            f"download=True or place train.csv.gz there manually"
        )
    root.mkdir(parents=True, exist_ok=True)
    _download_rosbank_csv(ROSBANK_RESOLVE_URL, dest)
    return dest


def _validate_rosbank_schema(
    df, expected_labeled_users: int = ROSBANK_EXPECTED_LABELED_USERS
) -> None:
    """Fail loud on schema drift (spec §4: "column set, labeled-user
    count"). Called before any parsing, so a drifted source raises a clear
    message instead of an obscure `KeyError`/mis-shaped result downstream.
    """
    actual_columns = set(df.columns)
    if actual_columns != ROSBANK_EXPECTED_COLUMNS:
        missing = sorted(ROSBANK_EXPECTED_COLUMNS - actual_columns)
        extra = sorted(actual_columns - ROSBANK_EXPECTED_COLUMNS)
        raise ValueError(
            f"Rosbank schema drift: missing columns {missing}, unexpected columns {extra}"
        )
    n_users = df["cl_id"].nunique()
    if n_users != expected_labeled_users:
        raise ValueError(
            f"Rosbank schema drift: expected {expected_labeled_users} labeled users, "
            f"found {n_users}"
        )


@dataclass(frozen=True)
class _FieldVocab:
    """One categorical field's train-split-derived vocabulary (spec §4/§6):
    `index` maps a raw value to its one-hot slot `0..len(index)-1`; any
    value NOT in `index` (unseen in train, or capped out of the top-N)
    encodes to the trailing OOV slot at index `len(index)`."""

    index: dict

    @property
    def size(self) -> int:
        """One-hot width: every mapped value plus one trailing OOV slot."""
        return len(self.index) + 1

    @property
    def oov_index(self) -> int:
        return len(self.index)

    def encode(self, value) -> int:
        return self.index.get(value, self.oov_index)


def _build_vocab(values: Sequence, cap: int | None) -> _FieldVocab:
    """Build a `_FieldVocab` from train-split `values` (spec §6: "Vocab caps
    from train split only").

    `cap=None` (trx_category, channel_type): every distinct value gets a
    slot, ordered by the value itself (ascending) -- deterministic and
    independent of frequency, since an uncapped field has no OOV-selection
    decision to make; only the one-hot slot ORDER needs to be fixed, and
    sorted-by-value is the simplest reproducible choice.

    `cap=N` (MCC, currency): the top-`N` values by train-split frequency get
    a slot (spec §6: "top-100-by-train-frequency" / "top-5"), ranked
    highest-frequency-first; ties broken by ascending value for a fully
    deterministic cutoff (the spec does not prescribe a tie-break, but the
    cutoff must still be reproducible given `split_seed`). Everything else
    falls to OOV.
    """
    counts = Counter(values)
    if cap is None:
        ordered = sorted(counts.keys())
    else:
        ordered = sorted(counts.keys(), key=lambda v: (-counts[v], v))[:cap]
    return _FieldVocab(index={v: i for i, v in enumerate(ordered)})


@dataclass(frozen=True)
class _RosbankVocabs:
    """The four categorical field vocabularies (spec §6 feature layout),
    bundled so every call site threads one object instead of four."""

    mcc: _FieldVocab
    trx_category: _FieldVocab
    channel_type: _FieldVocab
    currency: _FieldVocab

    @property
    def d_in(self) -> int:
        """Total one-hot-concat-plus-amount feature width (spec §6: 126 for
        the real dataset's vocab sizes; the caps fix MCC/currency at
        `cap + 1` exactly, while trx_category/channel_type vary with
        whatever data built this bundle -- e.g. a small synthetic test
        fixture)."""
        return (
            self.mcc.size
            + self.trx_category.size
            + self.channel_type.size
            + self.currency.size
            + 1  # log1p(amount)
        )


def _split_user_ids(
    user_ids: Sequence[int],
    split_seed: int,
    n_train: int = ROSBANK_TRAIN_USERS,
    n_val: int = ROSBANK_VAL_USERS,
    n_test: int = ROSBANK_TEST_USERS,
) -> tuple[list[int], list[int], list[int]]:
    """Seeded 4000/500/500 user split (spec §4/§6): `user_ids` is first
    canonicalized to ascending-sorted order (so the split does not depend on
    whatever order the source CSV happened to list users in), then permuted
    by a `split_seed`-seeded generator and sliced train/val/test in that
    fixed order -- reproducible given `split_seed` (spec §9: "identical
    `split_seed` reproduces identical splits").

    Raises
    ------
    ValueError
        If `len(user_ids)` does not equal `n_train + n_val + n_test` -- a
        silent size mismatch would otherwise either drop users or crash
        deep inside slicing with a confusing message.
    """
    total_needed = n_train + n_val + n_test
    if len(user_ids) != total_needed:
        raise ValueError(
            f"Rosbank split sizes (train={n_train} + val={n_val} + test={n_test} = "
            f"{total_needed}) do not match the {len(user_ids)} labeled users available"
        )
    canonical = sorted(user_ids)
    gen = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(canonical), generator=gen).tolist()
    shuffled = [canonical[i] for i in perm]
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val : total_needed]
    return train_ids, val_ids, test_ids


def _user_gap_days(timestamps: Sequence) -> list[float]:
    """Per-position day-gaps for one user's ALREADY chronologically sorted
    `timestamps` (any subtractable, `.total_seconds()`-yielding-delta type
    -- `pandas.Timestamp` and `datetime.datetime` both qualify, so this
    needs no pandas import): position 0 is 0.0 (nothing precedes the first
    observed event); each later position is the gap since the previous
    element of `timestamps`, in fractional days."""
    if not timestamps:
        return []
    gaps = [0.0]
    # pairwise consecutive elements: RHS is deliberately one shorter, so
    # `strict=False` (not the usual `strict=True`) is the correct choice here.
    for prev, curr in zip(timestamps, timestamps[1:], strict=False):
        gaps.append((curr - prev).total_seconds() / 86400.0)
    return gaps


def _valid_mask(n_events: int, T: int) -> list[bool]:
    """`True` at the trailing (real-event) positions of a length-`T`
    left-padded window, `False` at the leading pad positions -- `n_events`
    may exceed `T` (truncation, no padding at all) or fall short of it
    (left-padding fills the remainder)."""
    kept = min(n_events, T)
    return [False] * (T - kept) + [True] * kept


def _truncate_and_leftpad(seq: Sequence, T: int, pad_value):
    """Keep the LAST `T` elements of `seq` (spec §4: "truncate each
    sequence to its last T=256 events") and left-pad with `pad_value` up to
    length `T` if shorter (spec §4: "left-pad shorter sequences so the
    final position is always a real event") -- the shared truncate/pad
    windowing every per-user field (MCC/trx_category/channel_type/currency/
    amount indices) uses identically; `dt` uses `_window_gap_days` instead,
    since its pad-boundary value is NOT simply `pad_value` repeated (the
    first REAL position in the window must reset to 0, not carry the gap
    from whatever event truncation cut away)."""
    keep = list(seq[-T:])
    pad_len = T - len(keep)
    return [pad_value] * pad_len + keep


def _window_gap_days(gap_days_full: Sequence[float], T: int) -> list[float]:
    """`_truncate_and_leftpad`'s counterpart for `dt`: after keeping the
    last `T` of a user's full-history day-gaps, the FIRST kept position is
    forced back to 0.0 -- the window's first real event has no predecessor
    WITHIN the window (same no-t=0-exemption convention `probes.py`'s
    `TIMESTAMPED_TASKS` and this module's pendulum section already use),
    even though `gap_days_full` may have recorded a real (now truncated-
    away) predecessor for it. Left-padding then fills the remainder with
    0.0, same as every pad position elsewhere in the batch contract."""
    kept = list(gap_days_full[-T:])
    if kept:
        kept[0] = 0.0
    pad_len = T - len(kept)
    return [0.0] * pad_len + kept


def _encode_user_sequence(
    mcc: Sequence[int],
    trx_category: Sequence[str],
    channel_type: Sequence[str],
    currency: Sequence[int],
    amount: Sequence[float],
    timestamps: Sequence,
    vocabs: _RosbankVocabs,
    T: int,
) -> tuple[list[int], list[int], list[int], list[int], list[float], list[float], list[bool]]:
    """One user's raw, already chronologically-sorted event fields -> the
    seven length-`T` per-user rows `RosbankLoader` stacks into its split
    tensors: integer-encoded MCC/trx_category/channel_type/currency,
    `log1p(amount)`, day-gap `dt`, and the real-vs-pad `valid` mask.

    Pure function (no pandas): every argument is a plain sequence, so this
    is testable directly against hand-built lists.
    """
    n = len(mcc)
    gap_days_full = _user_gap_days(timestamps)
    mcc_idx = [vocabs.mcc.encode(v) for v in mcc]
    trxcat_idx = [vocabs.trx_category.encode(v) for v in trx_category]
    channel_idx = [vocabs.channel_type.encode(v) for v in channel_type]
    currency_idx = [vocabs.currency.encode(v) for v in currency]
    amount_log1p = [math.log1p(v) for v in amount]

    return (
        _truncate_and_leftpad(mcc_idx, T, 0),
        _truncate_and_leftpad(trxcat_idx, T, 0),
        _truncate_and_leftpad(channel_idx, T, 0),
        _truncate_and_leftpad(currency_idx, T, 0),
        _truncate_and_leftpad(amount_log1p, T, 0.0),
        _window_gap_days(gap_days_full, T),
        _valid_mask(n, T),
    )


@dataclass(frozen=True)
class _RosbankSplitData:
    """One split's (train/val/test) precomputed, integer-encoded, padded/
    truncated tensors (spec §5: "integer-encoded sequences are precomputed
    once, one-hot expansion happens per batch") -- `_rosbank_onehot_batch`
    expands a row-index slice of these into the `(x, dt, y)` batch
    contract."""

    mcc: torch.Tensor  # (N, T) int64
    trx_category: torch.Tensor  # (N, T) int64
    channel_type: torch.Tensor  # (N, T) int64
    currency: torch.Tensor  # (N, T) int64
    amount_log1p: torch.Tensor  # (N, T) float32
    dt: torch.Tensor  # (N, T) float32
    valid: torch.Tensor  # (N, T) bool
    y: torch.Tensor  # (N,) int64


def _build_split_data(
    df, user_ids: Sequence[int], vocabs: _RosbankVocabs, T: int
) -> _RosbankSplitData:
    """Group the already parsed/sorted `df` by `cl_id` and encode each of
    `user_ids`' event sequences via `_encode_user_sequence`, stacking the
    per-user rows into one `_RosbankSplitData`.

    `df` is expected to carry the `_ts` (parsed `TRDATETIME`, ascending
    within each user -- `RosbankLoader.__init__`'s stable global sort
    guarantees this) and `_channel` (NaN-filled `channel_type`) columns
    `RosbankLoader.__init__` adds before calling this.
    """
    grouped = df.groupby("cl_id", sort=False)
    mcc_rows, trxcat_rows, channel_rows, currency_rows = [], [], [], []
    amount_rows, dt_rows, valid_rows, y_rows = [], [], [], []
    for uid in user_ids:
        g = grouped.get_group(uid)
        mcc_idx, trxcat_idx, channel_idx, currency_idx, amount_log1p, dt, valid = (
            _encode_user_sequence(
                mcc=g["MCC"].tolist(),
                trx_category=g["trx_category"].tolist(),
                channel_type=g["_channel"].tolist(),
                currency=g["currency"].tolist(),
                amount=g["amount"].tolist(),
                timestamps=g["_ts"].tolist(),
                vocabs=vocabs,
                T=T,
            )
        )
        mcc_rows.append(mcc_idx)
        trxcat_rows.append(trxcat_idx)
        channel_rows.append(channel_idx)
        currency_rows.append(currency_idx)
        amount_rows.append(amount_log1p)
        dt_rows.append(dt)
        valid_rows.append(valid)
        # target_flag is constant within a user (spec's verified dataset facts).
        y_rows.append(int(g["target_flag"].iloc[0]))
    return _RosbankSplitData(
        mcc=torch.tensor(mcc_rows, dtype=torch.int64),
        trx_category=torch.tensor(trxcat_rows, dtype=torch.int64),
        channel_type=torch.tensor(channel_rows, dtype=torch.int64),
        currency=torch.tensor(currency_rows, dtype=torch.int64),
        amount_log1p=torch.tensor(amount_rows, dtype=torch.float32),
        dt=torch.tensor(dt_rows, dtype=torch.float32),
        valid=torch.tensor(valid_rows, dtype=torch.bool),
        y=torch.tensor(y_rows, dtype=torch.int64),
    )


def _rosbank_onehot_batch(
    split: _RosbankSplitData, idx: torch.Tensor, vocabs: _RosbankVocabs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand row-index slice `idx` of `split` into one `last_step_timed`
    batch (spec §6 feature layout: MCC one-hot || trx_category one-hot ||
    channel_type one-hot || currency one-hot || log1p(amount)) -- the
    per-batch one-hot expansion spec §5 calls for, rather than a
    whole-dataset materialization.

    Pad positions are zeroed AFTER concatenation via `split.valid`, so a
    pad row is an all-zero feature vector even though its underlying
    category indices are the arbitrary placeholder 0 (spec §6: "Pad
    positions are all-zero feature vectors").
    """
    mcc = F.one_hot(split.mcc[idx], vocabs.mcc.size)
    trx_category = F.one_hot(split.trx_category[idx], vocabs.trx_category.size)
    channel_type = F.one_hot(split.channel_type[idx], vocabs.channel_type.size)
    currency = F.one_hot(split.currency[idx], vocabs.currency.size)
    categorical = torch.cat([mcc, trx_category, channel_type, currency], dim=-1).float()
    amount = split.amount_log1p[idx].unsqueeze(-1)
    x = torch.cat([categorical, amount], dim=-1)
    x = x * split.valid[idx].unsqueeze(-1)
    dt = split.dt[idx]
    y = split.y[idx]
    return x, dt, y


def _rosbank_ordered_batches(
    split: _RosbankSplitData, vocabs: _RosbankVocabs, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """`_rosbank_onehot_batch` over fixed (non-shuffled) `batch_size` slices
    of `split`, in dataset order -- val/test, mirroring psMNIST's
    `_ordered_batches` reproducibility guarantee for this loader's 3-tuple
    contract."""
    n = split.y.shape[0]
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n))
        yield _rosbank_onehot_batch(split, idx, vocabs)


def _rosbank_shuffled_batches(
    split: _RosbankSplitData, vocabs: _RosbankVocabs, batch_size: int, gen: torch.Generator
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """`_rosbank_ordered_batches` over a fresh per-call shuffle of `split`'s
    row order, seeded by `gen` -- training, reshuffled every epoch (mirrors
    psMNIST's `_shuffled_batches`)."""
    order = torch.randperm(split.y.shape[0], generator=gen)
    for start in range(0, order.shape[0], batch_size):
        idx = order[start : start + batch_size]
        yield _rosbank_onehot_batch(split, idx, vocabs)


class RosbankLoader:
    """`Loader`-shaped (spec §6, `Loader` Protocol docstring above) loader
    for the churn task (spec §4): downloads (or reuses a cached)
    `train.csv.gz` at the pinned HF revision, parses `TRDATETIME`, stable-
    sorts each user's events by `(timestamp, original row order)`, splits
    users 4000/500/500 (`split_seed`), builds train-split-derived
    categorical vocabularies, and precomputes each split's truncated/
    left-padded integer-encoded tensors -- `train_epoch`/`val`/`test` then
    one-hot-expand a batch of rows at a time (`_rosbank_onehot_batch`).

    pandas is imported lazily here, not at module level (module docstring
    for this section) -- constructing this loader is the only thing that
    requires it installed.
    """

    def __init__(
        self,
        split_seed: int,
        batch_size: int,
        root: str = "./data/rosbank",
        download: bool = True,
        csv_path: str | None = None,
        T: int = ROSBANK_T,
        mcc_cap: int = ROSBANK_MCC_CAP,
        currency_cap: int = ROSBANK_CURRENCY_CAP,
        n_train_users: int = ROSBANK_TRAIN_USERS,
        n_val_users: int = ROSBANK_VAL_USERS,
        n_test_users: int = ROSBANK_TEST_USERS,
        expected_labeled_users: int = ROSBANK_EXPECTED_LABELED_USERS,
    ) -> None:
        """
        Parameters
        ----------
        split_seed : int
            Seeds the 4000/500/500 user split (spec §4/§6); recorded in
            each ledger row's `config` by the (later) driver task.
        batch_size : int
            Rows per yielded batch.
        root : str
            Cache directory for the downloaded `train.csv.gz` (spec §4:
            "cached under `./data/rosbank/`"). Ignored if `csv_path` is
            given.
        download : bool
            If the cache is missing, download when true; raise
            `FileNotFoundError` when false (mirrors `PsMNISTLoader`'s
            `download` flag). Ignored if `csv_path` is given.
        csv_path : str | None
            Load directly from this path instead of the download/cache
            path -- lets tests point at a synthetic mini-CSV fixture (spec
            §9/§10) without touching the network or the real cache
            directory.
        T : int
            Truncation/padding window length (256 in production).
        mcc_cap, currency_cap : int
            Train-split top-N vocabulary caps (spec §6 feature layout: 100
            and 5 in production) -- exposed so tests can shrink them
            against a small synthetic fixture.
        n_train_users, n_val_users, n_test_users : int
            User-split sizes (4000/500/500 in production) -- exposed for
            the same reason.
        expected_labeled_users : int
            Schema-drift check target for the total labeled-user count
            (spec §4) -- exposed so tests can point it at a small
            synthetic fixture's true user count instead of the production
            5000.
        """
        import pandas as pd  # local import: optional dependency (section docstring)

        self.split_seed = split_seed
        self.batch_size = batch_size
        self.T = T

        resolved_path = (
            Path(csv_path) if csv_path is not None else _ensure_rosbank_csv(Path(root), download)
        )
        df = pd.read_csv(resolved_path)
        _validate_rosbank_schema(df, expected_labeled_users=expected_labeled_users)

        df["_ts"] = pd.to_datetime(df["TRDATETIME"], format=ROSBANK_TRDATETIME_FORMAT)
        df["_channel"] = df["channel_type"].fillna(ROSBANK_NAN_CHANNEL_TYPE)
        # Stable global sort by timestamp: `kind="mergesort"` preserves
        # original row order among ties, so (a) each user's own event
        # subsequence stays sorted (a subsequence of a sorted sequence is
        # sorted) and (b) same-user same-timestamp ties (spec's "many
        # midnight-only timestamps") keep their original CSV row order --
        # spec §9's "(timestamp, original row order)" stable-sort criterion.
        df = df.sort_values("_ts", kind="mergesort")

        user_ids = sorted(df["cl_id"].unique().tolist())
        train_ids, val_ids, test_ids = _split_user_ids(
            user_ids, split_seed, n_train_users, n_val_users, n_test_users
        )

        train_df = df[df["cl_id"].isin(train_ids)]
        self.vocabs = _RosbankVocabs(
            mcc=_build_vocab(train_df["MCC"].tolist(), cap=mcc_cap),
            trx_category=_build_vocab(train_df["trx_category"].tolist(), cap=None),
            channel_type=_build_vocab(train_df["_channel"].tolist(), cap=None),
            currency=_build_vocab(train_df["currency"].tolist(), cap=currency_cap),
        )

        self._train = _build_split_data(df, train_ids, self.vocabs, T)
        self._val = _build_split_data(df, val_ids, self.vocabs, T)
        self._test = _build_split_data(df, test_ids, self.vocabs, T)

    @property
    def d_in(self) -> int:
        """Feature width of `x` (spec §6: 126 for the real dataset)."""
        return self.vocabs.d_in

    def train_epoch(
        self, gen: torch.Generator
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        yield from _rosbank_shuffled_batches(self._train, self.vocabs, self.batch_size, gen)

    def val(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        yield from _rosbank_ordered_batches(self._val, self.vocabs, self.batch_size)

    def test(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        yield from _rosbank_ordered_batches(self._test, self.vocabs, self.batch_size)


# ------------------------------------------------------- canonical TaskSpecs
# Concrete TaskSpec singletons for the four benchmark-round tasks (spec §4
# Global Constraints; landed here per the orchestrator's Task 4 brief, since
# Tasks 1-2 deliberately shipped the TaskSpec *contract* only -- see their
# reports: "S5's training-step budget is explicitly not yet determined...
# Concrete S5_TASK/MQAR_TASK TaskSpec singletons... are left for whichever
# task owns the pilot/lab-driver work").
#
# Fields FROZEN by the round's Global Constraints (fit_metric,
# fit_threshold, fit_direction, robustness, eval_protocol, seeds) are the
# real, binding values -- not placeholders, WITH ONE UPDATE: spec §4's
# pilot-calibration paragraph originally named three quantities still
# pending a pilot run before any seed matrix -- "a pilot ... fixes the S5
# and psMNIST training budgets and the pendulum tau" -- and
# `PENDULUM_TASK.fit_threshold` (tau) has since been calibrated and FROZEN
# at the pre-matrix technical review (2026-07-19); see `PENDULUM_TAU`'s own
# comment below for the calibration and rationale. `Budget` values are NOT
# all frozen yet: `S5_TASK.budget.steps` and `PSMNIST_TASK.budget.epochs`
# remain PILOT-PLACEHOLDER round 2 (see each TaskSpec's own comment) --
# reasonable starting points (S5's mirrors probes.py's historical
# MAX_STEPS=1600 default, now probing 8x after round-1 pilot rows sat at
# chance; psMNIST's is the design spec's ~30-epoch table value), each
# overridden from the lab driver's CLI (`--steps`/`--epochs`) -- never
# treated as frozen until a pilot task records their calibration in the
# round entries.
#
# MQAR's budget is NOT named in spec §4's pilot-calibration list (only S5,
# psMNIST, and pendulum's tau are) -- its `steps`/`eval_every` below are a
# committed working default (associative recall at T=64/8-pairs converges
# quickly at this model scale in the literature this task cites), not a
# formal pilot-gated placeholder, though still CLI-overridable like every
# other task's budget.

S5_TASK = TaskSpec(
    name="s5",
    loss_mode="dense",
    data=make_s5,
    fit_metric="val128",
    fit_threshold=0.99,
    fit_direction="ge",
    robustness=(0.98, 0.99, 0.995),
    eval_protocol=(EvalConfig(T=256), EvalConfig(T=512), EvalConfig(T=1024)),
    # FROZEN (pilot-calibrated, user decision 2026-07-19): steps=12800 (8x the
    # S3-scale 1600). Calibration record (bench-s5-01): all piloted arms
    # (log/givens/delta) sat at chance at 1600 AND at 12800 (val128 0.017->0.022
    # vs chance 0.0083) -- the matrix at this budget records an honest
    # expressivity result in the non-solvable-group regime, not an expected fit.
    budget=Budget(lr=3e-3, batch_size=128, steps=12800, eval_every=100),
    seeds=36,
)

MQAR_TASK = TaskSpec(
    name="mqar",
    loss_mode="masked_query",
    data=make_mqar,
    fit_metric="val_qacc",
    fit_threshold=0.99,
    fit_direction="ge",
    robustness=(0.98, 0.99, 0.995),
    eval_protocol=(
        EvalConfig(T=256, num_pairs=16),
        EvalConfig(T=256, num_pairs=32),
    ),
    # FROZEN (pilot-calibrated, user decision 2026-07-19): steps=12800.
    # Calibration record (bench-mqar-01): delta 0.21 val_qacc at 1600, 0.998+
    # at 12800 (fits); log at chance at both (no recall mechanism).
    budget=Budget(lr=3e-3, batch_size=128, steps=12800, eval_every=100),
    seeds=36,
)

PSMNIST_TASK = TaskSpec(
    name="psmnist",
    loss_mode="last_step",
    data=make_psmnist_loader,
    fit_metric="val_acc",
    fit_threshold=0.90,
    fit_direction="ge",
    robustness=(0.88, 0.90, 0.92),
    eval_protocol=(),  # test-set accuracy reported directly (spec §4); no length/pair-count generalization axis
    # FROZEN (pilot-calibrated, user decision 2026-07-19): epochs=30, lr=1e-3
    # (the design spec's training-config table; the earlier 3e-3 provisional
    # default was a deviation caught from pilot rows). Calibration record
    # (bench-psmnist-01): log 0.73-0.78 at 10 epochs climbing; at 30 epochs
    # log ~0.80 (plateau), delta straddles the 0.90 bar (val 0.898/0.907) --
    # the plan-frozen threshold discriminates arms at this budget.
    budget=Budget(lr=1e-3, batch_size=128, epochs=30),
    seeds=12,
)

# FROZEN (pre-matrix technical review, 2026-07-19): pure-noise-copy baseline
# MSE is PENDULUM_OBS_NOISE_STD**2 == 0.05**2 == 0.0025, NOT
# 2 * PENDULUM_OBS_NOISE_STD**2 -- the driver's regression loss
# (`benchmark_lab._forward_regression`'s `sq_err.mean()`) is a PER-ELEMENT
# mean over B*T*2 entries, not a per-position sum over the 2 (sin, cos)
# dims, so predicting the noisy observation `x` verbatim for `y` gives
# error == the observation noise at every one of those elements: the
# copy-baseline MSE is sigma^2, not 2*sigma^2 (an earlier version of this
# comment computed the wrong baseline). PENDULUM_TAU is calibrated from
# pilot rows (jobs at 368ce8b, round `bench-pendulum-01`, log arm seeds
# 0-2, `ckpt.val_mse` mean 0.0009 -- see `experiments/lab_results.jsonl`):
# 0.0009 * 1.5 slack = 0.00135, rounded up to 0.0014. 0.0014 sits strictly
# below the 0.0025 copy baseline, so the threshold still demands genuine
# denoising rather than a no-op copy. Frozen: never overridden by a later
# pilot task, and no longer a `dataclasses.replace` target.
PENDULUM_TAU = 0.0014

PENDULUM_TASK = TaskSpec(
    name="pendulum",
    loss_mode="regression",
    data=make_pendulum,
    fit_metric="val_mse",
    fit_threshold=PENDULUM_TAU,
    fit_direction="le",
    robustness=(1.25 * PENDULUM_TAU, PENDULUM_TAU, 0.8 * PENDULUM_TAU),
    eval_protocol=(),  # no post-selection generalization sweep defined by spec §4 beyond the fit metric itself
    budget=Budget(lr=3e-3, batch_size=128, steps=1600, eval_every=100),
    seeds=36,
)

TASKS: dict[str, TaskSpec] = {
    "s5": S5_TASK,
    "mqar": MQAR_TASK,
    "psmnist": PSMNIST_TASK,
    "pendulum": PENDULUM_TASK,
}

# ------------------------------------------------------- ledger round tags
# Current seed-matrix population's ledger `round` tag per task -- the single
# source of truth `scripts/gpu_benchmark_campaign.py` (writes these on every
# emitted row) and `scripts/report_benchmarks.py` (reads rows tagged with
# these) both bind to directly, rather than each hardcoding its own copy (a
# pre-matrix technical review found three independently hardcoded copies of
# this same mapping across campaign/report/`gpu_check.py` -- a sibling-
# duplication trigger). Bumped `bench-<task>-01` -> `bench-<task>-02` at
# that same review (2026-07-19): the `-01` tags are now the recorded
# pilot/calibration population (heterogeneous per-seed training budgets --
# e.g. `bench-s5-01`'s seed 0 ran at steps=1600 while seed 10 ran at
# steps=12800, see `experiments/lab_results.jsonl`), so a fresh tag per task
# keeps the real seed matrices from dedup-colliding with those pilot rows
# and keeps the report from silently pooling heterogeneous-budget rows into
# one set of statistics. `scripts/gpu_check.py`'s dedup/finish-handler
# validation is the one reader that must keep accepting BOTH generations
# (old pilot job logs/sidecars must stay parseable) -- it hardcodes its own
# frozen `-01` tuple alongside importing this mapping's values for the
# current `-02` half; see its own `_BENCHMARKS_ROUNDS` comment.
BENCH_ROUND_TAGS: dict[str, str] = {
    "s5": "bench-s5-02",
    "mqar": "bench-mqar-02",
    "psmnist": "bench-psmnist-02",
    "pendulum": "bench-pendulum-02",
}

# Probe round tags (Amendments, `.claude/output/intent/2026-07-19-benchmark-
# round-intent.md`, 2026-07-20 entry): a follow-up S5-only probe testing
# whether two suspected experiment-design artifacts -- `signed-rotation`'s
# missing K=5 snap order and `signed-delta`'s low nh=2 product count --
# rather than a genuine mechanism limit, explain S5's 0/36 rows for those
# families. The three probe arms (`experiments.benchmark_lab.PROBE_ARMS`)
# write under THIS distinct tag, never under `BENCH_ROUND_TAGS["s5"]`
# (`bench-s5-02`) -- a fresh tag keeps the clean eight-arm matrix population
# from ever being touched by probe rows (see `PROBE_ARMS`'s own comment and
# `scripts/report_benchmarks.py::_rows_for_task`'s `MATRIX_ARMS`-scoped
# planned-arm set). Only `"s5"` has an entry: the probe never runs the other
# three tasks (no decay-capability question to probe there), so a probe arm
# requested against any other task has no round tag to write under --
# `scripts/gpu_benchmark_campaign.py`'s round-tag resolution fails loud in
# that case rather than guessing one. `scripts/gpu_check.py`'s
# `_BENCHMARKS_ROUNDS` unions these values in alongside the `-01` pilot and
# `-02` matrix tags so the finish handler accepts probe rows too.
BENCH_PROBE_ROUND_TAGS: dict[str, str] = {
    "s5": "bench-s5-probe-01",
}

# Reference round tags (Amendments, `.claude/output/intent/2026-07-19-
# benchmark-round-intent.md`, 2026-07-20 "gru-large grounding reference"
# entry): a literature-scale `gru-large` arm (hidden-256, 2-layer `nn.GRU`,
# `experiments.benchmark_lab.REF_ARMS`) that validates the GRU code path and
# grounds the family's absolute numbers -- explicitly NOT a matched
# competitor, so its rows must never land under `BENCH_ROUND_TAGS` (the
# matched `-02` tag every `MATRIX_ARMS` seed uses). Same distinct-tag
# convention as `BENCH_PROBE_ROUND_TAGS` above (own source-of-truth mapping,
# `scripts/gpu_benchmark_campaign.py`'s round-tag resolution fails loud for
# any `(task, ref arm)` pair with no entry here rather than silently falling
# back to the matched tag). Only `"psmnist"` has an entry: this round only
# runs `gru-large` on psMNIST (the task whose matched `gru` control landed
# below the literature vanilla-GRU band); a ref arm requested against any
# other task has no round tag to write under. `scripts/gpu_check.py`'s
# `_BENCHMARKS_ROUNDS` unions these values in alongside the pilot, matrix,
# and probe tags so the finish handler accepts reference rows too.
BENCH_REF_ROUND_TAGS: dict[str, str] = {
    "psmnist": "bench-psmnist-ref-01",
}

# Per-arm round-tag correction overrides (design spec, `.claude/output/specs/
# 2026-07-21-round-tag-override-design.md`): the single source of truth
# consulted by write (`scripts/gpu_benchmark_campaign.py`'s per-arm round-tag
# resolver), read (`scripts/report_benchmarks.py`'s `_source_round_for_arm`),
# ingest (`scripts/gpu_check.py`'s `_BENCHMARKS_ROUNDS` allow-list), and
# disclosure (report payload's `arm_round_overrides` key) -- mirroring how
# `BENCH_ROUND_TAGS` above is already consulted by write/read/ingest. Shape
# is one dimension deeper than the population tag maps: `arm -> {task ->
# correction round tag}`. `BENCH_ARM_ROUND_OVERRIDES.get(arm, {}).get(task)`
# is the correction tag if present, else `None` -- an absent arm or absent
# task means no override, and every touchpoint falls through to its normal
# population routing (matrix tag / probe tag / ref tag) unchanged. This is
# the two-layer model: population-default tag vs per-arm correction tag.
# Populated for the `signed-rotation` composer-order fix (rotation-order
# rename+swap landed; this is the correction re-run, not a fresh `-03` full
# matrix): `signed-rotation` gets a correction tag on all four matrix tasks,
# and its K=5 probe sibling `signed-rotation-k5` gets one on `"s5"` only (the
# probe never runs the other three tasks, same as `BENCH_PROBE_ROUND_TAGS`
# above). No other arm has an entry here -- every other matrix arm, the two
# other probe arms, and the ref arm are untouched and keep writing/reading
# under their existing population tags. Adding or removing an entry here is the
# only edit that routes or un-routes an arm's correction round across all four
# touchpoints.
BENCH_ARM_ROUND_OVERRIDES: dict[str, dict[str, str]] = {
    "signed-rotation": {
        "s5": "bench-s5-rotfix-01",
        "mqar": "bench-mqar-rotfix-01",
        "psmnist": "bench-psmnist-rotfix-01",
        "pendulum": "bench-pendulum-rotfix-01",
    },
    "signed-rotation-k5": {
        "s5": "bench-s5-probe-rotfix-01",
    },
}
