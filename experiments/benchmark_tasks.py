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
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

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


class Loader(Protocol):
    """Epoch-based counterpart to the synthetic tasks' `make(batch, T, gen)`
    generators (spec §6: `data: Callable | Loader`). psMNIST's `data` is an
    instance of this protocol rather than a stateless per-call callable,
    since an epoch dataset owns a fixed split and a fixed pixel permutation
    that must not be regenerated on every batch (module docstring).

    Every method yields `last_step` batches (module docstring): `x` float
    `(batch, T, d_in)`, `y` int64 `(batch,)`.
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
# real, binding values -- not placeholders. `Budget` values are NOT all
# frozen yet: spec §4's pilot-calibration paragraph names exactly three
# quantities still pending a pilot run before any seed matrix -- "a pilot
# ... fixes the S5 and psMNIST training budgets and the pendulum tau" --
# so `S5_TASK.budget.steps`, `PSMNIST_TASK.budget.epochs`, and
# `PENDULUM_TASK.fit_threshold` (tau) below are PILOT-PLACEHOLDER values:
# reasonable starting points (S5's mirrors probes.py's historical
# MAX_STEPS=1600 default; psMNIST's is a conservative epoch count for a
# 50k-row set; tau is chosen below the "predict the noisy observation
# verbatim" baseline MSE so the threshold demands real denoising), each
# easily overridden from the lab driver's CLI (`--steps`/`--epochs`/
# `--fit-threshold` are NOT how tau is overridden -- tau is a TaskSpec
# field, not a Budget field; a pilot script overrides it by constructing
# its own `TaskSpec` via `dataclasses.replace(PENDULUM_TASK, ...)` rather
# than through the lab driver's CLI) -- never treated as frozen until the
# pilot task records its calibration in the round entries.
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
    # PILOT-PLACEHOLDER round 2: all arms (log/givens/delta) sat at chance at the
    # S3-scale steps=1600 (pilot jobs at 368ce8b, bench-s5-01 rows); probing 8x.
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
    # PILOT-PLACEHOLDER round 2: delta reached only 0.21 val_qacc at the working
    # default steps=1600 (pilot jobs at 368ce8b, bench-mqar-01 rows); probing 8x.
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
    # PILOT-PLACEHOLDER round 2: log reached 0.73-0.78 val_acc still climbing at
    # epochs=10 (pilot jobs at 368ce8b, bench-psmnist-01 rows); probing the spec
    # 4 table's ~30 epochs. lr corrected 3e-3 -> 1e-3: the design spec's 2
    # training-config table fixes psMNIST at Adam 1e-3 (the earlier 3e-3 was a
    # provisional-default deviation caught from the pilot rows' config).
    budget=Budget(lr=1e-3, batch_size=128, epochs=30),
    seeds=12,
)

# PILOT-PLACEHOLDER: pure-noise-copy baseline MSE is 2 * PENDULUM_OBS_NOISE_STD**2
# = 0.005 (predicting the noisy observation `x` verbatim for `y` gives error ==
# the observation noise on both of the 2 dims); PENDULUM_TAU sits below that so
# the fit threshold demands genuine denoising, not a no-op copy. Not yet
# pilot-calibrated (spec §4) -- a later pilot task overrides this field (e.g.
# via `dataclasses.replace(PENDULUM_TASK, fit_threshold=..., robustness=...)`)
# and freezes it in the round entries before the pendulum seed matrix runs.
PENDULUM_TAU = 0.003

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
