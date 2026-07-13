"""Expressivity probes: GRU vs minGRU vs SignedMinGRU vs RotationMinGRU.

Tasks (seq2seq tagging, dense supervision as in Merrill et al. 2024):
  parity              : running XOR over {0,1}. In TC0; the natural
                         one-scan solution needs a transition eigenvalue
                         at -1, so positive-diagonal minGRU lacks it
                         while SignedMinGRU and GRU have it.
  S3                  : running product in the symmetric group S3
                         (smallest non-abelian group). Order-sensitive;
                         commutative (diagonal) scans of ANY sign should
                         fail at one layer, while a GRU encodes the
                         6-state automaton directly.
  session-parity      : running XOR that RESETS at session boundaries
                         (an inter-event gap exceeding
                         SESSION_GAP_THRESHOLD). Demonstrates the decay
                         mechanism learning a timescale that separates
                         within-session gaps from boundary gaps. Default
                         rows follow the fairness rule: every model
                         receives f(delta_t) = log1p(delta_t) as an
                         appended input feature, and only -tdecay rows
                         additionally consume delta_t mechanically. The
                         -tdecay-mech rows are a channel ablation: they
                         consume delta_t ONLY through the mechanical
                         decay path (feature channel disabled), isolating
                         that channel's contribution from the
                         feature+mechanism -tdecay row and the
                         feature-only baseline; see TIMESTAMPED_TASKS and
                         MECHANICAL_ONLY_MODELS.
  parity-timestamped  : plain (non-resetting) parity, but with the same
                         delta_t distribution as session-parity supplied
                         to every model. Recovery check: a -tdecay row's
                         learned lambda should approach 0 (nothing to
                         forget) and its accuracy should match the
                         non-decay variant's.

Train at T=64; evaluate at T=64 (in-dist) and T=256 (length gen).

Usage: python probes.py TASK MODEL [N_LAYERS]
       TASK in {parity, S3, session-parity, parity-timestamped}; MODEL
       in {GRU, minGRU, minGRU-signed, minGRU-signed-tanh,
       minGRU-signed-tanh-tdecay, minGRU-signed-tanh-tdecay-mech,
       minGRU-rotsnap} (GRU is not valid for the two timestamped tasks:
       it has no delta_t input path); N_LAYERS defaults to 1 (applies to
       all models).
       MAX_STEPS env var overrides the training budget (default 1600).
       CKPT=1 replaces early-stop with best-checkpoint selection by
       validation accuracy at T=128 (seed 5, n_batches=2), evaluated
       every EVAL_EVERY steps over the full step budget; off by default
       so legacy early-stop rows stay reproducible.
"""

import os
import sys
import time
from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from min_gru import MinGRUStack

# probes.py model name -> (mixer, mixer_kwargs). minGRU-signed is pinned to
# coupled=True: it is the pre-promotion SignedMinGRU parameterization, kept
# reachable under its historical name so recorded rows keep their meaning
# (spec: variant-promotion §6). minGRU-signed-tanh is the new decoupled
# default; minGRU-rotsnap is RotationMinGRU with its default snap grid.
# minGRU-signed-tanh-tdecay is minGRU-signed-tanh plus learnable time decay
# (spec §6 decay-row naming: base name + "-tdecay"), log1p_delta=True so the
# mechanically-consumed delta_t and the fairness-rule feature (also
# log1p(delta_t), see TIMESTAMPED_TASKS) are the same transformed quantity
# -- avoids needing lambda to span the raw ~0.1..100 gap range itself,
# which would otherwise push softplus(rho) toward the extremes to
# separate within-session from boundary gaps.
# decay_rate=0.05 (this registry row's config only -- min_gru.py's
# class-level decay_rate=1.0 default is spec-locked and untouched):
# persist-by-default init, so the mechanism starts near "never decay"
# (gamma(0-ish lambda) ~ 1) and must be pulled toward heavier decay by
# gradient pressure where the session-boundary signal actually rewards
# it, rather than starting at lambda=1.0 and needing to unlearn heavy
# decay against weak gradient pressure at T_TRAIN=64 (the failure mode
# observed in the first evidence pass -- see task-3-report.md's
# lambda-init comparison).
MIXER_REGISTRY = {
    "minGRU": ("log", {}),
    "minGRU-signed": ("signed", {"coupled": True}),
    "minGRU-signed-tanh": ("signed", {}),
    "minGRU-signed-tanh-tdecay": (
        "signed",
        {"decay": "learnable", "log1p_delta": True, "decay_rate": 0.05},
    ),
    # Same mixer config as minGRU-signed-tanh-tdecay, but see
    # MECHANICAL_ONLY_MODELS below: this row skips the fairness-rule
    # log1p(delta_t) feature concat entirely, consuming delta_t ONLY
    # through the stack's mechanical decay path -- a channel ablation
    # isolating the mechanism channel, as opposed to the
    # feature+mechanism fairness-rule comparison the plain -tdecay row
    # runs under.
    "minGRU-signed-tanh-tdecay-mech": (
        "signed",
        {"decay": "learnable", "log1p_delta": True, "decay_rate": 0.05},
    ),
    "minGRU-rotsnap": ("rotation", {}),
}

# Model names whose TimestampedMinGRUTagger skips the fairness-rule
# log1p(delta_t) feature concat (mechanical-only delta_t consumption):
# stack input_size stays D_MODEL (no appended feature), and delta_t
# reaches the model only via the stack's decay path. Kept out of
# MIXER_REGISTRY's own (mixer, mixer_kwargs) shape -- a per-row
# behavioral flag, not a mixer config -- the same way TIMESTAMPED_TASKS
# is kept out of TASKS. Every model NOT in this set keeps the original
# fairness-rule behavior (feature always concatenated) unchanged.
MECHANICAL_ONLY_MODELS = {"minGRU-signed-tanh-tdecay-mech"}

# CKPT protocol (spec §4): best-val@128 selection over the full step budget,
# replacing early stop. Not one of the eval/test lengths (T_TRAIN=64,
# T_GEN=256), so it can't leak into either reported metric.
CKPT_T = 128

D_MODEL = 64
T_TRAIN, T_GEN = 64, 256
BATCH = 128
MAX_STEPS = int(os.environ.get("MAX_STEPS", 1600))
EVAL_EVERY = 100
LR = 3e-3

# ---------------------------------------------------------------- tasks
def make_parity(batch, T, gen):
    x = torch.randint(0, 2, (batch, T), generator=gen)
    y = x.cumsum(dim=1) % 2
    return x, y

# S3 as permutations of (0,1,2); element 0 is the identity.
S3 = torch.tensor(
    [[0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 1, 0], [1, 2, 0], [2, 0, 1]]
)

def _compose_table():
    idx = {tuple(p.tolist()): k for k, p in enumerate(S3)}
    table = torch.zeros(6, 6, dtype=torch.long)
    for i in range(6):
        for j in range(6):
            comp = S3[i][S3[j]]  # p_i o p_j
            table[i, j] = idx[tuple(comp.tolist())]
    return table

COMPOSE = _compose_table()

def make_s3(batch, T, gen):
    x = torch.randint(0, 6, (batch, T), generator=gen)
    y = torch.zeros_like(x)
    state = torch.zeros(batch, dtype=torch.long)  # identity
    for t in range(T):
        state = COMPOSE[x[:, t], state]  # g_t o r_{t-1}
        y[:, t] = state
    return x, y

# ------------------------------------------------------- session-parity
# Session-parity (spec §6): running XOR that RESETS at session boundaries,
# where a boundary is an inter-event gap exceeding SESSION_GAP_THRESHOLD.
# Within-session gaps and boundary gaps are drawn from disjoint ranges well
# below / well above the threshold, so "boundary" is unambiguous from
# delta_t alone -- the draw that decides which range a gap comes from
# doubles as the ground-truth reset signal used to build y.
SESSION_GAP_THRESHOLD = 10.0
WITHIN_SESSION_GAP_RANGE = (0.1, 1.0)
BOUNDARY_GAP_RANGE = (50.0, 100.0)
BOUNDARY_PROB = 0.05  # ~3.2 boundaries/session-resets expected at T=64
assert WITHIN_SESSION_GAP_RANGE[1] < SESSION_GAP_THRESHOLD < BOUNDARY_GAP_RANGE[0], (
    "within-session and boundary gap ranges must straddle the threshold"
)

# Tasks whose make_fn returns (x, delta_t, y) instead of (x, y): delta_t is
# consumed as a log1p(delta_t) input feature by every model (fairness
# rule, spec §6) and, for -tdecay mixer rows, additionally mechanically by
# the mixer's decay path. Kept out of TASKS itself so TASKS keeps the
# plain (make, vocab, n_cls) shape that experiments/variants.py's
# `make, vocab, n_cls = TASKS[task]` unpack depends on.
TIMESTAMPED_TASKS = {"session-parity", "parity-timestamped"}


def _make_timestamps(batch, T, gen):
    """Shared delta_t generator for the two TIMESTAMPED_TASKS.

    Draws, per position, whether the preceding gap is a session boundary
    (probability BOUNDARY_PROB) and samples the gap from the matching
    range (BOUNDARY_GAP_RANGE if boundary, else WITHIN_SESSION_GAP_RANGE).
    Position 0 is forced non-boundary with delta_t=0 (true sequence
    start, per the module's no-t=0-exemption convention: nothing precedes
    it to decay from or reset out of).

    Returns
    -------
    tuple of torch.Tensor
        ``(delta_t, is_boundary)``, each ``(batch, T)``: ``delta_t`` in
        raw gap units (float), ``is_boundary`` a bool mask -- session-
        parity resets its running XOR where ``is_boundary`` is True;
        parity-timestamped ignores it.
    """
    is_boundary = torch.rand(batch, T, generator=gen) < BOUNDARY_PROB
    is_boundary[:, 0] = False
    within = torch.empty(batch, T).uniform_(*WITHIN_SESSION_GAP_RANGE, generator=gen)
    across = torch.empty(batch, T).uniform_(*BOUNDARY_GAP_RANGE, generator=gen)
    delta_t = torch.where(is_boundary, across, within)
    delta_t[:, 0] = 0.0
    return delta_t, is_boundary


def make_session_parity(batch, T, gen):
    x = torch.randint(0, 2, (batch, T), generator=gen)
    delta_t, is_boundary = _make_timestamps(batch, T, gen)
    y = torch.zeros_like(x)
    state = torch.zeros(batch, dtype=x.dtype)
    for t in range(T):
        state = torch.where(is_boundary[:, t], x[:, t], state ^ x[:, t])
        y[:, t] = state
    return x, delta_t, y


def make_parity_timestamped(batch, T, gen):
    """Plain (non-resetting) parity with the session-parity delta_t
    distribution attached: the lambda->0 recovery check (spec §9.3)
    trains a -tdecay row on this task and expects its learned lambda to
    approach 0 and its accuracy to match the non-decay variant's, since
    decaying here (even at boundary-scale gaps) only discards state
    parity needs."""
    x, y = make_parity(batch, T, gen)
    delta_t, _ = _make_timestamps(batch, T, gen)
    return x, delta_t, y


TASKS = {
    "parity": (make_parity, 2, 2),
    "S3": (make_s3, 6, 6),
    "session-parity": (make_session_parity, 2, 2),
    "parity-timestamped": (make_parity_timestamped, 2, 2),
}

# ---------------------------------------------------------------- models
class GRUTagger(nn.Module):
    def __init__(self, vocab, n_cls, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.rnn = nn.GRU(D_MODEL, D_MODEL, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(D_MODEL, n_cls)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.head(h)


class MinGRUTagger(nn.Module):
    def __init__(self, vocab, n_cls, mixer, mixer_kwargs, n_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.stack = MinGRUStack(
            D_MODEL, D_MODEL, n_layers=n_layers, mixer=mixer, mixer_kwargs=mixer_kwargs
        )
        self.head = nn.Linear(D_MODEL, n_cls)

    def forward(self, x):
        return self.head(self.stack(self.emb(x))[0])


# The three delta_t-consumption configurations TimestampedMinGRUTagger
# supports. Collapses what used to be two independent bools
# (mechanical_decay, include_feature) into one Literal so the invalid
# "neither channel active" combination is unrepresentable: "decay-only"
# always implies mechanical consumption, so a model can never be built
# with delta_t reaching it through neither channel.
#   "feature"      : delta_t only as the log1p(delta_t) input feature
#                    (the feature-only baseline, e.g. minGRU-signed-tanh).
#   "feature+decay": both channels (the fairness-rule -tdecay rows).
#   "decay-only"   : delta_t only through the mixer's mechanical decay
#                    path (the -tdecay-mech channel-ablation rows).
DeltaTMode = Literal["feature", "feature+decay", "decay-only"]


class TimestampedMinGRUTagger(nn.Module):
    """MinGRUTagger variant for TIMESTAMPED_TASKS (session-parity,
    parity-timestamped).

    ``delta_t_mode`` (see ``DeltaTMode``) selects which of the two
    delta_t channels are active:

    - ``"feature"`` / ``"feature+decay"``: every model receives
      ``f(delta_t) = log1p(delta_t)`` concatenated onto the token
      embedding as an appended input feature, projected down to
      ``D_MODEL`` by the stack's own ``in_proj`` (input_size =
      D_MODEL + 1), so no extra linear layer is needed here.
    - ``"feature+decay"`` / ``"decay-only"``: the mixer additionally
      consumes raw ``delta_t`` through the stack's mechanical decay path
      (the mixer applies its own configured ``log1p_delta`` internally).
    - ``"feature"`` (feature only, no decay) never passes ``delta_t`` to
      the stack -- a stack with no decayed block rejects it
      (``MinGRUStack``'s mode-error rule).
    - ``"decay-only"`` (spec §6 channel ablation, ``MECHANICAL_ONLY_MODELS``
      rows) skips the feature concat entirely (stack input_size stays
      ``D_MODEL``), isolating the mechanical-decay channel from the
      feature+mechanism fairness-rule comparison.
    """

    def __init__(
        self,
        vocab,
        n_cls,
        mixer,
        mixer_kwargs,
        n_layers=1,
        delta_t_mode: DeltaTMode = "feature",
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.delta_t_mode = delta_t_mode
        self.include_feature = delta_t_mode != "decay-only"
        self.mechanical_decay = delta_t_mode != "feature"
        stack_input_size = D_MODEL + 1 if self.include_feature else D_MODEL
        self.stack = MinGRUStack(
            stack_input_size,
            D_MODEL,
            n_layers=n_layers,
            mixer=mixer,
            mixer_kwargs=mixer_kwargs,
        )
        self.head = nn.Linear(D_MODEL, n_cls)

    def forward(self, x, delta_t):
        h = self.emb(x)
        if self.include_feature:
            feat = torch.log1p(delta_t).unsqueeze(-1)
            h = torch.cat([h, feat], dim=-1)
        stack_dt = delta_t if self.mechanical_decay else None
        out, _ = self.stack(h, delta_t=stack_dt)
        return self.head(out)


def build(task, name, vocab, n_cls, n_layers=1):
    """Construct a model for ``task``/``name`` (``vocab``/``n_cls`` from
    ``TASKS[task]``, ``n_layers`` blocks).

    ``timestamped = task in TIMESTAMPED_TASKS`` is derived internally
    (rather than taken as a caller-supplied bool) since ``task`` already
    determines it and every call site already has ``task`` in hand.
    """
    timestamped = task in TIMESTAMPED_TASKS
    if timestamped:
        if name == "GRU":
            raise ValueError(
                "GRU has no delta_t input path; timestamped tasks require "
                "a MinGRU-family model from MIXER_REGISTRY."
            )
        if name not in MIXER_REGISTRY:
            raise ValueError(f"unknown model {name!r}; valid: {list(MIXER_REGISTRY)}")
        mixer, mixer_kwargs = MIXER_REGISTRY[name]
        if name in MECHANICAL_ONLY_MODELS:
            delta_t_mode: DeltaTMode = "decay-only"
        elif mixer_kwargs.get("decay") is not None:
            delta_t_mode = "feature+decay"
        else:
            delta_t_mode = "feature"
        return TimestampedMinGRUTagger(
            vocab,
            n_cls,
            mixer,
            mixer_kwargs,
            n_layers=n_layers,
            delta_t_mode=delta_t_mode,
        )
    if name == "GRU":
        return GRUTagger(vocab, n_cls, n_layers)
    if name not in MIXER_REGISTRY:
        valid = ["GRU", *MIXER_REGISTRY]
        raise ValueError(f"unknown model {name!r}; valid: {valid}")
    mixer, mixer_kwargs = MIXER_REGISTRY[name]
    return MinGRUTagger(vocab, n_cls, mixer, mixer_kwargs, n_layers=n_layers)


def _forward_batch(model, make, batch, T, gen, timestamped):
    """Draw one batch and run the model forward.

    Uniform across the plain ``(x, y)`` tasks and the ``(x, delta_t, y)``
    tasks in TIMESTAMPED_TASKS, so the training loop and ``accuracy()``
    can share this one call path instead of two drifting copies of the
    same branch.

    Returns
    -------
    tuple of torch.Tensor
        ``(logits, y)``.
    """
    if timestamped:
        x, dt, y = make(batch, T, gen)
        return model(x, dt), y
    x, y = make(batch, T, gen)
    return model(x), y


# `timestamped` stays a trailing bool kwarg here (not folded into a
# task-derived closure like build()'s mode) because experiments/variants.py
# calls `accuracy(model, make, val_T, seed=5, n_batches=val_B)` and
# `accuracy(model, make, T, seed=4)` positionally up through `seed`/
# `n_batches`; the signature up to and including `n_batches` is a pinned
# external contract, so `timestamped` must stay an appended, defaulted
# trailing parameter.
@torch.no_grad()
def accuracy(model, make, T, seed, n_batches=4, timestamped=False):
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        logits, y = _forward_batch(model, make, BATCH, T, gen, timestamped)
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total


class LambdaSummary(NamedTuple):
    """Pooled per-channel decay rate (``lambda``) summary; see
    ``lambda_summary()``."""

    max: float
    mean: float


def lambda_summary(model):
    """Learned decay-rate (``lambda = softplus(rho)``, or the fixed
    ``decay_rate``) summary for a model built by ``build()``.

    Inspects every block of ``model.stack`` (absent on ``GRUTagger``,
    which returns None) whose mixer has decay enabled, and pools their
    per-channel rates.

    Parameters
    ----------
    model : nn.Module
        A model returned by ``build()``.

    Returns
    -------
    LambdaSummary or None
        ``LambdaSummary(max, mean)`` over every decay-enabled block's
        per-channel lambda, or None if the model has no ``stack``
        attribute or no block has decay enabled.
    """
    stack = getattr(model, "stack", None)
    if stack is None:
        return None
    lambdas = []
    for block in stack.blocks:
        mixer = block.mingru
        if mixer.decay == "learnable":
            lambdas.append(F.softplus(mixer.rho).detach())
        elif mixer.decay == "fixed":
            lambdas.append(mixer.decay_rate_buf.detach().reshape(1))
    if not lambdas:
        return None
    all_lambda = torch.cat(lambdas)
    return LambdaSummary(max=all_lambda.max().item(), mean=all_lambda.mean().item())


def run_one(task, name, n_layers=1, max_steps=MAX_STEPS, ckpt=None, seed=0):
    make, vocab, n_cls = TASKS[task]
    timestamped = task in TIMESTAMPED_TASKS
    # seed=0 reproduces the legacy manual_seed(0)/manual_seed(1) pair
    # exactly (1 + 10_000 * 0 == 1); other seeds vary init and data order,
    # mirroring experiments/variants.py's run_cell convention. The
    # 10_000 multiplier keeps seed>=1's train generator (10001, 20001,
    # ...) far from the small fixed eval seeds (2/3/4/5) used below --
    # plain `1 + seed` put seed=1's train gen at 2 (colliding with the
    # early-stop eval seed) and seed=2's at 3 (colliding with the acc_in
    # eval seed), silently overlapping train and eval examples at
    # T_TRAIN for those two seeds. Eval seeds themselves are untouched
    # (changing them would break the legacy pin, recorded under eval
    # seed 4).
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1 + 10_000 * seed)
    model = build(task, name, vocab, n_cls, n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0, steps_used = time.time(), max_steps
    # Checkpoint selection: rotation-snap's exact solution is reachable but
    # not a stable attractor of training (runs wander in and out of it), so
    # best-val@128 selection replaces early stop, evaluated over the full
    # step budget. ckpt=None defers to the CKPT env var (off by default:
    # legacy rows keep the early-stop protocol they were recorded under);
    # GRID rows pin their own protocol per row. Ported from
    # experiments/variants.py run_cell's ckpt_select branch.
    if ckpt is None:
        ckpt = os.environ.get("CKPT", "") not in ("", "0")
    ckpt_select = ckpt
    best_val, best_state, best_step = -1.0, None, 0
    for step in range(1, max_steps + 1):
        logits, y = _forward_batch(model, make, BATCH, T_TRAIN, gen, timestamped)
        loss = F.cross_entropy(logits.reshape(-1, n_cls), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            if ckpt_select:
                val = accuracy(model, make, CKPT_T, seed=5, n_batches=2, timestamped=timestamped)
                if val > best_val:
                    best_val, best_step = val, step
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
            elif accuracy(model, make, T_TRAIN, seed=2, n_batches=2, timestamped=timestamped) >= 0.999:
                steps_used = step
                break
    if ckpt_select and best_state is not None:
        model.load_state_dict(best_state)
        steps_used = best_step
    acc_in = accuracy(model, make, T_TRAIN, seed=3, timestamped=timestamped)
    acc_gen = accuracy(model, make, T_GEN, seed=4, timestamped=timestamped)
    if ckpt_select:
        if best_state is None:  # budget below EVAL_EVERY: no eval ever ran
            ckpt_info = " | ckpt: none taken (MAX_STEPS < EVAL_EVERY)"
        else:
            ckpt_info = f" | ckpt@{CKPT_T}: {best_val:.3f} (step {best_step})"
    else:
        ckpt_info = ""
    lambda_info = ""
    if timestamped and getattr(model, "mechanical_decay", False):
        ls = lambda_summary(model)
        if ls is not None:
            lambda_info = f" | lambda: max={ls.max:.4f} mean={ls.mean:.4f}"
    print(
        f"{task:>7} | {name:<18} | L={n_layers} | seed={seed} | "
        f"acc@{T_TRAIN}: {acc_in:.3f} | acc@{T_GEN}: {acc_gen:.3f} | "
        f"steps: {steps_used:>4} | {time.time() - t0:5.1f}s{ckpt_info}{lambda_info}",
        flush=True,
    )
    return model


GRID = [
    # (task, model, n_layers, max_steps, ckpt) — ckpt=True rows run the
    # best-val@128 checkpoint-selection protocol their README results are
    # recorded under (required for minGRU-rotsnap; see RotationMinGRU docs).
    ("parity", "GRU", 1, 1500, False),
    ("parity", "minGRU", 1, 1500, False),
    ("parity", "minGRU-signed", 1, 1500, False),
    ("S3", "GRU", 1, 1500, False),
    ("S3", "minGRU", 1, 1500, False),
    ("S3", "minGRU-signed", 1, 1500, False),
    ("parity", "minGRU", 4, 1600, False),
    ("parity", "minGRU-signed", 4, 1600, False),
    ("S3", "minGRU", 4, 1600, False),
    ("S3", "minGRU-signed", 4, 1600, False),
    ("parity", "minGRU-signed-tanh", 1, 1500, False),
    ("S3", "minGRU-signed-tanh", 1, 1500, False),
    ("parity", "minGRU-signed-tanh", 4, 1600, False),
    ("S3", "minGRU-signed-tanh", 4, 1600, False),
    # session-parity: signed-tanh-based (not rotation), so early-stop is
    # the default protocol like the other signed rows above (ckpt=False),
    # not the rotation-only best-val@128 selection.
    ("session-parity", "minGRU-signed-tanh-tdecay", 1, 1500, False),
    ("session-parity", "minGRU-signed-tanh", 1, 1500, False),
    # mechanical-only (no fairness-rule feature): channel ablation
    # isolating the mechanism channel, same budget/protocol as the two
    # rows above.
    ("session-parity", "minGRU-signed-tanh-tdecay-mech", 1, 1500, False),
    ("parity", "minGRU-rotsnap", 1, 1600, True),
    ("S3", "minGRU-rotsnap", 1, 1600, True),
]


def run_grid():
    # MAX_STEPS / CKPT env vars, when set, override the per-entry grid
    # values (docstring contract); otherwise each entry uses its own.
    env_steps = "MAX_STEPS" in os.environ
    env_ckpt = "CKPT" in os.environ
    for task, name, n_layers, max_steps, ckpt in GRID:
        run_one(
            task,
            name,
            n_layers,
            MAX_STEPS if env_steps else max_steps,
            None if env_ckpt else ckpt,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        run_grid()
    else:
        run_one(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
