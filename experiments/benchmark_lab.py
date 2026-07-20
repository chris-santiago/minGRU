"""Benchmark lab driver: the task-agnostic single-arm train/eval loop for
the accepted-benchmark validation round (spec:
`.claude/output/specs/2026-07-19-benchmark-round-design.md`, §4/§5/§6).

Consumes the `TaskSpec` contract and data generators/loaders from
`experiments/benchmark_tasks.py` (Tasks 1-2 of this round). One invocation
trains one seed of one packaged-mixer arm on one task, selects a
checkpoint by the task's own fit metric on held-out data, evaluates the
task's generalization protocol, and emits one ledger row -- mirroring
`experiments/hetero_lab.py`'s `run_arm` shape (seeding convention,
checkpoint-selection loop, row assembly, `--dry-run` convention) but
dispatching on `TaskSpec.loss_mode` instead of hardcoding one task's batch
contract, since this round's four tasks span four different label shapes
(dense / masked_query / last_step / regression -- spec §6).

Models are built through the packaged mixer registry path
(`min_gru.MinGRUStack`, mirroring `probes.py`'s `build()`/`MIXER_REGISTRY`
pattern): `ARM_REGISTRY` fixes the registered arms' configs -- five from
the round's Global Constraints (`log`, `signed`, `rotation`, `givens` at
block_size=8/rounds=3, `delta` at its promoted native-state default
nh=2/n_heads=4/d_k=16/d_v=16), d_model=64, 2 pre-norm blocks, plus a sixth
arm, `rotation-hetero`, added by amendment at the evidence-phase gate
(`.claude/output/intent/2026-07-19-benchmark-round-intent.md`, Amendments):
`rotation`'s pure 2-block stack is `MinGRUStack`'s documented STE-
compounding broken baseline (more than one `"rotation"` block always
warns once at construction, unconditional on `snap` -- see
`MinGRUStack`'s docstring); the round runs it as-is AND adds
`rotation-hetero` -- one `"rotation"` block (default snap grid) composed
with one `"signed"` block, mirroring `probes.py`'s `"minGRU-hetero-rs"`
row -- this repo's evidenced fix for rotation-stack STE compounding
(not a snap on/off comparison against `rotation`: both arms' rotation
block uses the same default snap grid; the fix is the composition, not
the snap setting), which avoids the multi-rotation warning (a single
`"rotation"` entry in a mixed stack doesn't warn). See `ARM_REGISTRY`'s
own comment for the full rationale, including why a same-type second
reading was tried and rejected (byte-identical to `rotation`, caught by
`tests/test_report_benchmarks.py`'s distinct-param-count invariant).

Two more arms, `signed-givens` and `signed-delta`, were added by a
second, later amendment (same intent-ledger file, Amendments) that runs
them on ALL FOUR tasks, exactly like the other six arms -- these eight
arms (`MATRIX_ARMS`) are the clean seed-matrix population every task,
`scripts/report_benchmarks.py`'s `-02` completeness accounting, and
`scripts/gpu_benchmark_campaign.py`'s default `--arms` all still iterate.
They ground this round's arm list in the repo's already-evidenced
promoted hetero structures rather than inventing a new one: `signed-givens`
is `probes.py`'s `"minGRU-hetero-sg8"` row (`["signed", "givens"], None`,
the S3-hier promoted fit-rate winner, GPU-re-evidenced as
`hetero_lab.py`'s `"hetero-pg8"`); `signed-delta` mirrors
`hetero_lab.py`'s `"hetero-pd1024"` row (the delta block at its
matched-state/GPU-36-round native config, nh=2/n_heads=4/d_k=16/d_v=16 --
identical to this round's own `delta` arm's kwargs) composed with one
`signed` block. See `ARM_REGISTRY`'s own comment for the provenance
caveat on `signed-delta` (the lab lineage row used signed-TANH, not the
packaged `signed` mixer this round's registry path uses). No new mixer
code -- every arm is an existing packaged class, called through
`MinGRUStack`'s `mixer=`/`mixer_kwargs=` contract.

A third amendment (same intent-ledger file, 2026-07-20 entry) added three
more arms, `PROBE_ARMS` (`rotation-hetero-k5`, `signed-delta-nh3`,
`signed-delta-nh4`) -- an S5-only follow-up probe testing whether two
suspected experiment-design artifacts, not a mechanism limit, explain
S5's 0/36 rows for `rotation-hetero` and `signed-delta`. Unlike the
first two amendments, these do NOT join the eight-arm seed-matrix
population: `ARM_REGISTRY` (this module's `build_model`/CLI lookup) is
`MATRIX_ARMS` unioned with `PROBE_ARMS`, but `scripts
/report_benchmarks.py`'s `-02` completeness accounting and
`scripts/gpu_benchmark_campaign.py`'s default `--arms` read `MATRIX_ARMS`
only -- probe arms run solely when explicitly named via `--arms` and
write under their own distinct ledger round tag
(`experiments.benchmark_tasks.BENCH_PROBE_ROUND_TAGS`), never the matrix
`-02` tag. See `PROBE_ARMS`'s own comment for the corrected-config
rationale.

A fourth amendment (same intent-ledger file, 2026-07-20 "standard GRU
control arm" entry) added a ninth `MATRIX_ARMS` arm, `gru`: a depth-matched
(2-layer, d_model=64) `nn.GRU` classical control, run on all four tasks
like every other matrix arm -- unlike the S5-only `PROBE_ARMS`, this is
genuine matrix expansion (`scripts/report_benchmarks.py`'s `-02`
completeness accounting and `scripts/gpu_benchmark_campaign.py`'s default
`--arms` both pick it up automatically since both read `MATRIX_ARMS`).
This closes a gap the first eight arms left open: every one of them
compares WITHIN the minGRU family (`log` is the vanilla-minGRU baseline,
not a classical RNN), so the round had no external anchor for what an
absolute fit rate even means. `gru` is not a `MinGRUStack` mixer -- it has
no `mixer`/`mixer_kwargs` shape at all -- so it is not built through
`build_model`'s `ARM_REGISTRY[arm]` unpacking; see `MATRIX_ARMS`'s own
comment for the sentinel-tuple representation and `build_model`'s `arm ==
"gru"` branch for construction. Params are reported, not equalized,
exactly like the other eight arms. Excluded from `DECAY_CAPABLE_ARMS`
(see that set's own comment): plain `nn.GRU` has no decay/`delta_t`
mechanism at all, so pendulum's `dt` reaches it only via the `log1p(dt)`
feature-concat channel, the same treatment `delta`/the hetero arms get.

Usage:
  uv run python experiments/benchmark_lab.py \
      --round bench-s5-02 --task s5 --model log --seed 0
  uv run python experiments/benchmark_lab.py --selftest

`--selftest` runs a tiny (few-step, small-batch) sweep of all four tasks
under the `log` arm, dry-run only -- a wiring smoke test, not evidence.
psMNIST substitutes a synthetic stand-in `Loader` (no torchvision, no MNIST
download) per the task brief.

Pendulum's decay-channel wiring reuses `probes.py`'s existing
timestamped-input mechanism (`TimestampedMinGRUTagger`'s fairness rule),
rather than inventing a new one: every arm receives `log1p(dt)`
concatenated onto its input features, and the four mixers whose
`DecayMixin` contract supports it (`log`/`signed`/`rotation`/`givens`)
additionally consume raw `dt` mechanically via `MinGRUStack`'s
`delta_t=` argument (`decay="learnable"`, `log1p_delta=True`,
`decay_rate=0.05` -- the same config and persist-by-default-init rationale
as `probes.py`'s `minGRU-signed-tanh-tdecay` row). `DeltaMinGRU` rejects
`decay`/`delta_t` unconditionally (frozen contract, see its class
docstring in `src/mingru/min_gru.py`) -- the delta arm on the pendulum
task therefore only sees `dt` through the feature channel, never
mechanically; this asymmetry is inherent to the packaged mixer's contract,
not something this driver invents or can route around.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root: `min_gru`, `experiments.*`

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.benchmark_tasks import (
    MQAR_KEY_VOCAB,
    MQAR_VALUE_VOCAB,
    S5_ELEMENTS,
    TASKS,
    EvalConfig,
    TaskSpec,
)
from min_gru import MinGRUStack

D_MODEL = 64
N_LAYERS = 2
TRAIN_T = 64  # S5/MQAR/pendulum training sequence length (spec §4: "Train at T=64")
S5_CKPT_T = 128  # S5 checkpoint-selection length (spec §4), distinct from train (64) and gen (256/512/1024)

# Fixed eval seeds, independent of --seed (probes.py/hetero_lab.py convention:
# only the training stream and model init vary by seed; every arm's held-out
# eval draws from the same synthetic distribution). Moved out of the 0..35
# run-seed range at the pre-matrix technical review (2026-07-19, item 6):
# the previous values (CKPT_EVAL_SEED=5, FINAL_EVAL_SEED=4) collided with
# run seeds 4/5's `torch.manual_seed` model-init stream -- a distinct
# `torch.Generator` object so no actual shared RNG state, but a confusing
# coincidence that made "seed 5" ambiguous between "run seed 5" and "the
# fixed checkpoint-eval seed". FIT_EVAL_SEED is new (item 5: the honest
# fit-metric re-eval draws its own dedicated fixed seed, distinct from both
# CKPT_EVAL_SEED and FINAL_EVAL_SEED, so no two of the three re-use the
# same draw). The `-01` pilot population was evaluated under the old
# 4/5 seeds -- comparability across `-01`/`-02` tags is not claimed
# (CLAUDE.md: evidence strata are never mixed silently).
CKPT_EVAL_SEED = 1_000_005
FINAL_EVAL_SEED = 1_000_004
FIT_EVAL_SEED = 1_000_006
CKPT_EVAL_BATCHES = 4
FINAL_EVAL_BATCHES = 4

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_results.jsonl")

# Registry keys -> (MinGRUStack mixer name, mixer_kwargs) -- the registered
# arms' configs (probes.py's MIXER_REGISTRY/build() pattern, generalized to
# this round's packaged mixers instead of probes.py's historical model
# names): five fixed by the round's Global Constraints, plus the amended
# sixth arm below, plus two more `list[str]` hetero arms (`signed-givens`,
# `signed-delta`) added by a second amendment -- see the module docstring's
# "Two more arms" paragraph. mixer is a `str` (one type, every block, flat
# kwargs) for five of the eight arms; `rotation-hetero`/`signed-givens`/
# `signed-delta` are the three `list[str]` arms (per-block mixer types,
# kwargs keyed by type name or `None` -- see `MinGRUStack`'s own docstring
# for this schema). Every arm here runs on all four tasks -- there is no
# per-task arm subset; `scripts/report_benchmarks.py` and
# `scripts/gpu_benchmark_campaign.py` both iterate this same dict.
#
# rotation-hetero: a single "rotation" block (RotationMinGRU's own default
# snap grid, (2, 3, 4, 6) -- the same default the plain "rotation" arm's
# blocks already use) composed with a single "signed" block (decoupled
# tanh default -- same config as this round's own "signed" arm), mirroring
# probes.py's "minGRU-hetero-rs" MIXER_REGISTRY row exactly
# (`(["rotation", "signed"], None)`). This is NOT a snap on/off comparison
# against "rotation" -- both arms' rotation block snaps identically by
# default; the difference is the composition. It is this repo's evidenced
# fix for the rotation-stack STE-compounding problem: a homogeneous
# 2-block "rotation" stack (this round's plain "rotation" arm, unchanged,
# run "as is") is `MinGRUStack`'s documented broken baseline (more than
# one "rotation" block always warns, unconditional on `snap` -- see its
# docstring); pairing rotation-hetero with a SECOND identical rotation
# block instead of "signed" would be architecturally byte-identical to
# "rotation" (same param count, same per-seed training trajectory, zero
# new information for real GPU spend -- caught by
# tests/test_report_benchmarks.py's "every arm's param count is distinct"
# invariant when first tried). Composing with "signed" instead: (a) is a
# genuinely different, non-duplicate construction (different param count,
# different weights); (b) avoids `MinGRUStack`'s multi-rotation
# UserWarning (a single "rotation" entry in a mixed stack does not warn --
# see its docstring), unlike "rotation" itself, which retains the warning
# as the documented, unmitigated STE-compounding broken baseline. Not
# added to DECAY_CAPABLE_ARMS below: probes.py has no established
# hetero-mixer + decay row to mirror (its hetero-sr/-rs rows only ever run
# on S3/S3-hier, never the timestamped tasks), so inventing a decay split
# across two mixer types for the pendulum task would be new, unauthorized
# design -- rotation-hetero is therefore treated like "delta" on pendulum:
# dt reaches it only via the log1p(dt) feature-concat channel, never
# mechanically.
#
# signed-givens: a single "signed" block (decoupled tanh default, same
# config as this round's own "signed" arm) composed with a single
# "givens" block at its own class default kwargs (block_size=8, rounds=3
# -- confirmed against GivensMinGRU's constructor defaults in
# src/mingru/min_gru.py, the same config this round's own "givens" arm
# passes explicitly). mixer_kwargs=None: the list-mixer schema applies
# every type's default kwargs, exactly like rotation-hetero above.
# Byte-identical to probes.py's "minGRU-hetero-sg8" MIXER_REGISTRY row
# (`(["signed", "givens"], None)`) -- the S3-hier promoted fit-rate
# winner, GPU-re-evidenced as hetero_lab.py's "hetero-pg8" arm
# (signed-tanh + packaged GivensMinGRU, block_size=8/rounds=3). This
# round's registry path builds the block from the packaged "signed"
# mixer (decoupled tanh, same as every other "signed"-family arm in this
# module), not probes.py's historical "signed-tanh" name -- same
# mechanism, different registry/module, no behavioral gap.
#
# signed-delta: a single "signed" block at default kwargs composed with
# a single "delta" block at its promoted native-state config (nh=2,
# n_heads=4, d_k=16, d_v=16 -- identical kwargs to this round's own
# "delta" arm above). mirrors hetero_lab.py's "hetero-pd1024" row
# (matched-state/GPU-36-round evidence), which composed the delta block
# with a "signed-tanh" block, NOT the packaged "signed" mixer this row
# uses -- a provenance difference worth stating plainly, not glossing
# over: "signed-tanh" is hetero_lab.py's own lab-local decoupled-tanh
# block (frozen for that module's evidence lineage), while "signed" here
# is the promoted packaged SignedMinGRU class this whole round's registry
# path builds every arm from (the promoted packaged equivalent, not a
# byte-identical carryover the way signed-givens/hetero-sg8 is above).
# Unlike rotation-hetero/signed-givens, mixer_kwargs here is NOT `None`:
# the delta block needs its native config explicitly, so this is a
# type-keyed dict (`{"delta": {...}}`) with "signed" omitted (defaults to
# `None`, i.e. the class's own default kwargs -- see MinGRUStack's
# docstring for the omitted-key-defaults-to-None resolution rule).
#
# Neither signed-givens nor signed-delta is decay-capable (see
# DECAY_CAPABLE_ARMS below): same rationale as rotation-hetero -- no
# established repo precedent for splitting decay wiring across a hetero
# stack's two mixer types, so this round doesn't invent one for either
# new arm. Both receive dt only via the log1p(dt) feature-concat channel
# on the pendulum task, like delta and rotation-hetero.
#
# gru: the classical external control arm (fourth amendment, module
# docstring's "standard GRU control arm" paragraph) -- a depth-matched
# (2-layer, d_model=64) plain `nn.GRU`, the ONLY arm here that is not a
# `MinGRUStack` mixer. Its tuple value is therefore a SENTINEL, never
# destructured into `(mixer, mixer_kwargs)`: `build_model` special-cases
# `arm == "gru"` before it ever unpacks `ARM_REGISTRY[arm]` (see
# `build_model`'s own comment). The tuple exists only so `"gru"` satisfies
# `MATRIX_ARMS`/`ARM_REGISTRY` dict membership -- the same membership every
# other arm gets for free -- which is what makes `gru` show up
# automatically in the CLI's `--model`/`--arms` choices, the `-02` report's
# per-arm iteration, and `gpu_benchmark_campaign.py`'s default `--arms`,
# with no separate registry a caller would have to know to also check.
MATRIX_ARMS: dict[str, tuple[str | list[str], dict | None]] = {
    "log": ("log", {}),
    "signed": ("signed", {}),
    "rotation": ("rotation", {}),
    "rotation-hetero": (["rotation", "signed"], None),
    "givens": ("givens", {"block_size": 8, "rounds": 3}),
    "delta": ("delta", {"nh": 2, "n_heads": 4, "d_k": 16, "d_v": 16}),
    "signed-givens": (["signed", "givens"], None),
    "signed-delta": (["signed", "delta"], {"delta": {"nh": 2, "n_heads": 4, "d_k": 16, "d_v": 16}}),
    "gru": ("gru", None),
}

# Probe arms (Amendments, `.claude/output/intent/2026-07-19-benchmark-round-
# intent.md`, 2026-07-20 entry): three corrected-config S5-only follow-up
# arms testing whether two suspected experiment-design artifacts -- not a
# mechanism limit -- explain S5's 0/36 rows for `rotation-hetero` and
# `signed-delta`.
#
# rotation-hetero-k5: byte-identical to `rotation-hetero` above (same
# `["rotation", "signed"]` stack) except the rotation block's snap grid is
# widened to include K=5 -- S5's element orders are {2, 3, 4, 5, 6} (a
# 5-cycle needs an order-5 rotation to land an exact attractor), but
# `RotationMinGRU`'s own default snap grid, `(2, 3, 4, 6)`, is missing it.
# `RotationMinGRU`'s docstring: "the snap grid must contain the [element]
# order[s]" -- this arm is the grid corrected to match S5's actual group,
# not a new comparison axis. `snap` is a registered buffer, not a
# parameter (`RotationMinGRU.__init__`), so this arm's param count is
# IDENTICAL to `rotation-hetero`'s (see `PROBE_ARMS`'s param-count
# invariant note below and `tests/test_benchmark_lab.py`'s pinned test) --
# the all-arms-distinct-param-count invariant that holds across
# `MATRIX_ARMS` does NOT extend to this arm, by construction, not by bug.
#
# signed-delta-nh3 / signed-delta-nh4: `signed-delta`'s stack
# (`["signed", "delta"]`) with the delta block's Householder-product count
# `nh` raised from this round's matrix value (nh=2) to 3 and 4
# respectively -- `n_heads`/`d_k`/`d_v` unchanged, matching `DeltaMinGRU`'s
# constructor (`n_heads=4, nh=1, d_k=None, d_v=None` defaults) as ground
# truth. `nh` IS a real constructor-time shape parameter (more Householder
# reflections per delta step), so these two arms DO have distinct param
# counts from `signed-delta` and from each other.
PROBE_ARMS: dict[str, tuple[str | list[str], dict | None]] = {
    "rotation-hetero-k5": (["rotation", "signed"], {"rotation": {"snap": (2, 3, 4, 5, 6)}}),
    "signed-delta-nh3": (
        ["signed", "delta"],
        {"delta": {"nh": 3, "n_heads": 4, "d_k": 16, "d_v": 16}},
    ),
    "signed-delta-nh4": (
        ["signed", "delta"],
        {"delta": {"nh": 4, "n_heads": 4, "d_k": 16, "d_v": 16}},
    ),
}

# The build_model/CLI lookup: matrix arms (the clean nine-arm seed-matrix
# population, `scripts/report_benchmarks.py`'s `-02` completeness accounting
# reads `MATRIX_ARMS` directly, never this union) union probe arms. Keeping
# this as one dict (rather than two independent registries `build_model`
# would have to check in turn) means every existing single-lookup call site
# below (`build_model`, the CLI's `--model`/`--arms` choices) transparently
# resolves both arm families with no dispatch change.
ARM_REGISTRY: dict[str, tuple[str | list[str], dict | None]] = {**MATRIX_ARMS, **PROBE_ARMS}

# Arms whose packaged mixer class mixes in DecayMixin (accepts decay=
# "learnable"/"fixed" and a delta_t argument). DeltaMinGRU is deliberately
# excluded: its constructor rejects any decay is not None (frozen contract,
# see its class docstring) -- pendulum's dt reaches it only via the
# feature-concat channel below, never mechanically. rotation-hetero,
# signed-givens, and signed-delta are excluded for the same reason (see
# MATRIX_ARMS's comment): no established repo precedent for splitting
# decay wiring across either mixer type of a hetero stack, so this round
# doesn't invent one for any of the three hetero arms -- each reaches dt
# on pendulum only via the log1p(dt) feature-concat channel, never
# mechanically. The three PROBE_ARMS are excluded for the identical
# reason -- each is a hetero stack of the same two mixer-type families
# (rotation+signed, signed+delta) as its non-decay-capable matrix sibling.
# gru is excluded for a DIFFERENT reason than the hetero arms above: it is
# not a DecayMixin-mixing packaged mixer at all -- plain nn.GRU has no
# decay/delta_t mechanism to enable in the first place, not merely one
# this round declines to wire up. Pendulum's dt reaches it only via the
# log1p(dt) feature-concat channel, same as every other excluded arm here.
DECAY_CAPABLE_ARMS = {"log", "signed", "rotation", "givens"}


# ---------------------------------------------------------------- models
class TokenSequenceModel(nn.Module):
    """`dense`/`masked_query` tasks (S5, MQAR): int token ids -> per-
    position logits over `n_cls` classes. Loss/eval masking (dense vs
    masked-query) is entirely the caller's job -- this model has no
    notion of which positions matter."""

    def __init__(
        self,
        vocab: int,
        n_cls: int,
        mixer: str | list[str],
        mixer_kwargs: dict,
        n_layers: int = N_LAYERS,
        d_model: int = D_MODEL,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.stack = MinGRUStack(
            d_model, d_model, n_layers=n_layers, mixer=mixer, mixer_kwargs=mixer_kwargs
        )
        self.head = nn.Linear(d_model, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.stack(self.emb(x))
        return self.head(out)


class FeatureSequenceModel(nn.Module):
    """`last_step`/`regression` tasks (psMNIST, pendulum): float features
    -> per-position outputs, with an optional `delta_t` routed straight to
    the stack's decay channel unchanged. Reading only the final position
    (psMNIST) vs every position (pendulum's regression target) is the
    caller's job, not this model's."""

    def __init__(
        self,
        input_size: int,
        n_out: int,
        mixer: str | list[str],
        mixer_kwargs: dict,
        n_layers: int = N_LAYERS,
        d_model: int = D_MODEL,
    ) -> None:
        super().__init__()
        self.stack = MinGRUStack(
            input_size, d_model, n_layers=n_layers, mixer=mixer, mixer_kwargs=mixer_kwargs
        )
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.stack(x, delta_t=delta_t)
        return self.head(out)


class GRUTokenSequenceModel(nn.Module):
    """`gru` arm counterpart to `TokenSequenceModel` (dense/masked_query
    tasks: S5, MQAR): int token ids -> embedding -> a depth-matched
    (default `N_LAYERS=2`) plain `nn.GRU` -> per-position logits over
    `n_cls` classes -- the classical external control this round's other
    eight arms (all `MinGRUStack` mixers) are anchored against (see the
    module docstring's fourth-amendment paragraph). Ported from
    `probes.py`'s `GRUTagger` cell pattern (`nn.GRU(D_MODEL, D_MODEL,
    num_layers=n_layers, batch_first=True)`), generalized to this round's
    `vocab`/`n_cls` per-task shapes.

    `nn.GRU` dispatches to cuDNN internally and never reads `MINGRU_SCAN` --
    a `gru`-arm run under `MINGRU_SCAN=triton` (the production job's export
    for the other eight arms' Triton dispatch) is harmless, not a bug: the
    env var is simply irrelevant to this arm, and no special-casing is
    needed to keep it that way."""

    def __init__(
        self,
        vocab: int,
        n_cls: int,
        n_layers: int = N_LAYERS,
        d_model: int = D_MODEL,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.emb(x))
        return self.head(h)


class GRUFeatureSequenceModel(nn.Module):
    """`gru` arm counterpart to `FeatureSequenceModel` (last_step/regression
    tasks: psMNIST, pendulum): float features -> a depth-matched plain
    `nn.GRU` -> per-position outputs.

    `delta_t` is accepted (not omitted) so this model is a drop-in for the
    same `model(x)` / `model(model_in, delta_t=decay_dt)` call shapes
    `_forward_last_step`/`_forward_regression` already use uniformly across
    every arm -- but it must always be `None` here: plain `nn.GRU` has no
    decay/dt mechanism, and `gru` is excluded from `DECAY_CAPABLE_ARMS`, so
    `_forward_regression`'s `decay_dt = dt if arm in DECAY_CAPABLE_ARMS else
    None` never passes this model anything else. Pendulum's `dt` reaches it
    only through the `log1p(dt)` feature `build_model` concatenates onto
    the input, exactly like `delta`/the hetero arms."""

    def __init__(
        self,
        input_size: int,
        n_out: int,
        n_layers: int = N_LAYERS,
        d_model: int = D_MODEL,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor | None = None) -> torch.Tensor:
        assert delta_t is None, (
            "GRUFeatureSequenceModel has no delta_t/decay path (plain nn.GRU) -- "
            "gru is excluded from DECAY_CAPABLE_ARMS, so this must never be called "
            "with a non-None delta_t"
        )
        h, _ = self.rnn(x)
        return self.head(h)


def _model_shape(task_name: str) -> tuple[int, int]:
    """(vocab|d_in, n_cls|d_out) for `task_name`'s model -- derived from
    the task module's own data-shape constants (mirrors `probes.py`'s
    `TASKS` tuple convention: model shape lives beside the driver, not
    inside the frozen `TaskSpec` contract, which has no vocab/dim
    fields)."""
    if task_name == "s5":
        n = S5_ELEMENTS.shape[0]
        return n, n
    if task_name == "mqar":
        vocab = MQAR_KEY_VOCAB + MQAR_VALUE_VOCAB
        return vocab, vocab
    if task_name == "psmnist":
        return 1, 10
    if task_name == "pendulum":
        return 2, 2
    raise ValueError(f"no model shape registered for task {task_name!r}")


def build_model(task: TaskSpec, arm: str) -> nn.Module:
    """Construct `arm`'s model for `task` via `ARM_REGISTRY` (spec §6
    arm configs) and `task.loss_mode` (spec §6 batch contracts).

    `arm == "gru"` is a distinct branch, checked first: `gru` is not a
    `MinGRUStack` mixer (its `ARM_REGISTRY` value is a sentinel tuple, never
    destructured -- see `MATRIX_ARMS`'s comment), so it builds
    `GRUTokenSequenceModel`/`GRUFeatureSequenceModel` directly instead of
    going through the `mixer`/`mixer_kwargs` unpacking every other arm uses.
    It still satisfies the exact same per-loss_mode shape contract
    (`_forward_dense`/`_forward_masked_query`/`_forward_last_step`/
    `_forward_regression` never know or care which branch built the model),
    and gets the same `log1p(dt)` feature-concat treatment on pendulum as
    the other `DECAY_CAPABLE_ARMS`-excluded arms -- just never the
    mechanical-decay `mixer_kwargs` update, since it has no `mixer_kwargs`.

    `arm_kwargs` is `None` for `rotation-hetero` (the one list-mixer arm --
    `MinGRUStack`'s own convention for "no per-type overrides", see its
    docstring); every other arm's `arm_kwargs` is a flat dict, possibly
    empty. `dict(x) if x else {}` normalizes both to a real dict here
    (mirroring `MinGRUBlock.__init__`'s own `mixer_kwargs` normalization)
    so downstream code never juggles `None`."""
    if arm not in ARM_REGISTRY:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_REGISTRY)}")
    in_dim, out_dim = _model_shape(task.name)
    if arm == "gru":
        if task.loss_mode in ("dense", "masked_query"):
            return GRUTokenSequenceModel(in_dim, out_dim)
        if task.loss_mode == "last_step":
            return GRUFeatureSequenceModel(in_dim, out_dim)
        if task.loss_mode == "regression":
            return GRUFeatureSequenceModel(in_dim + 1, out_dim)  # +1: log1p(dt) feature
        raise ValueError(f"unknown loss_mode {task.loss_mode!r}")
    mixer, arm_kwargs = ARM_REGISTRY[arm]
    mixer_kwargs = dict(arm_kwargs) if arm_kwargs else {}
    if task.loss_mode in ("dense", "masked_query"):
        return TokenSequenceModel(in_dim, out_dim, mixer, mixer_kwargs)
    if task.loss_mode == "last_step":
        return FeatureSequenceModel(in_dim, out_dim, mixer, mixer_kwargs)
    if task.loss_mode == "regression":
        input_size = in_dim + 1  # +1: log1p(dt) fairness-rule feature (see module docstring)
        if arm in DECAY_CAPABLE_ARMS:
            mixer_kwargs.update(decay="learnable", log1p_delta=True, decay_rate=0.05)
        return FeatureSequenceModel(input_size, out_dim, mixer, mixer_kwargs)
    raise ValueError(f"unknown loss_mode {task.loss_mode!r}")


# ---------------------------------------------------------- loss/eval dispatch
def _forward_dense(model, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """`dense` (S5): cross-entropy + accuracy over every position."""
    logits = model(x)
    n_cls = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, n_cls), y.reshape(-1))
    correct = (logits.argmax(-1) == y).sum().item()
    return loss, correct, y.numel()


def _forward_masked_query(
    model, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, int, int]:
    """`masked_query` (MQAR): cross-entropy + accuracy ONLY where `mask` is
    true -- non-query positions never reach the loss or the accuracy
    count, regardless of `y`'s (unspecified) value there."""
    logits = model(x)
    logits_m, y_m = logits[mask], y[mask]
    loss = F.cross_entropy(logits_m, y_m)
    correct = (logits_m.argmax(-1) == y_m).sum().item()
    return loss, correct, y_m.numel()


def _forward_last_step(model, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """`last_step` (psMNIST): cross-entropy + accuracy at the FINAL
    position's logits only -- every earlier position's logits are
    computed (the model has no "read only the end" mode) but discarded
    here before the loss ever sees them."""
    logits = model(x)[:, -1, :]
    loss = F.cross_entropy(logits, y)
    correct = (logits.argmax(-1) == y).sum().item()
    return loss, correct, y.numel()


def _forward_regression(
    model, x: torch.Tensor, dt: torch.Tensor, y: torch.Tensor, arm: str
) -> tuple[torch.Tensor, float, int]:
    """`regression` (pendulum): MSE over every position. `dt` always
    reaches the model as a `log1p(dt)` concatenated feature; it
    additionally reaches the stack's mechanical decay channel for arms in
    `DECAY_CAPABLE_ARMS` (see module docstring), `None` otherwise (the
    delta arm, which rejects `delta_t` unconditionally)."""
    feat = torch.log1p(dt).unsqueeze(-1)
    model_in = torch.cat([x, feat], dim=-1)
    decay_dt = dt if arm in DECAY_CAPABLE_ARMS else None
    pred = model(model_in, delta_t=decay_dt)
    sq_err = (pred - y).pow(2)
    return sq_err.mean(), sq_err.sum().item(), sq_err.numel()


def _forward_step_based(task: TaskSpec, arm: str, model, batch: tuple[torch.Tensor, ...]):
    """Dispatch one generator-drawn batch to its loss_mode's forward
    function. Not used by psMNIST (`last_step`, loader-based -- see
    `_train_epoch_based`), whose data acquisition already differs
    entirely from the generator tasks."""
    if task.loss_mode == "dense":
        x, y = batch
        return _forward_dense(model, x, y)
    if task.loss_mode == "masked_query":
        x, y, mask = batch
        return _forward_masked_query(model, x, y, mask)
    if task.loss_mode == "regression":
        x, dt, y = batch
        return _forward_regression(model, x, dt, y, arm)
    raise ValueError(f"_forward_step_based does not support loss_mode {task.loss_mode!r}")


def _draw_batch(
    task: TaskSpec, batch_size: int, T: int, gen: torch.Generator, num_pairs: int | None = None
) -> tuple[torch.Tensor, ...]:
    """Task-agnostic batch draw for the three generator-based tasks
    (S5/MQAR/pendulum). `num_pairs` is MQAR-only; every generator's own
    default is used when omitted (training config; eval_protocol configs
    pass it explicitly)."""
    if num_pairs is not None:
        return task.data(batch_size, T, gen, num_pairs=num_pairs)
    return task.data(batch_size, T, gen)


def _to_device(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(device) for t in batch)


def _ckpt_eval_T(task_name: str, train_T: int) -> int:
    """Checkpoint-selection sequence length (spec §4): S5 checkpoints at a
    fixed T=128, distinct from both its train (64) and gen (256/512/1024)
    lengths so selection never leaks into a reported metric; MQAR and
    pendulum checkpoint "at the training configuration" -- i.e. whatever
    `train_T` this run actually used."""
    if task_name == "s5":
        return S5_CKPT_T
    return train_T


@torch.no_grad()
def _eval_generator_task(
    task: TaskSpec,
    arm: str,
    forward_model,
    T: int,
    seed: int,
    n_batches: int,
    batch_size: int,
    num_pairs: int | None = None,
    device: torch.device | None = None,
) -> float:
    """Weighted-average metric (accuracy for dense/masked_query, MSE for
    regression) over `n_batches` batches at a FIXED eval seed (independent
    of the run's training seed) -- weighted by each batch's element count,
    not per-batch-averaged, so a ragged final batch never skews the
    result."""
    gen = torch.Generator().manual_seed(seed)
    forward_model.eval()
    num, den = 0.0, 0
    for _ in range(n_batches):
        batch = _draw_batch(task, batch_size, T, gen, num_pairs=num_pairs)
        if device is not None:
            batch = _to_device(batch, device)
        _, m_num, m_den = _forward_step_based(task, arm, forward_model, batch)
        num += m_num
        den += m_den
    forward_model.train()
    return num / den


@torch.no_grad()
def _eval_last_step_loader(forward_model, batches, device: torch.device | None = None) -> float:
    """Weighted-average accuracy over a `Loader`'s `val()`/`test()`
    batches (psMNIST) -- same weighted-averaging discipline as
    `_eval_generator_task`."""
    forward_model.eval()
    num, den = 0, 0
    for x, y in batches:
        if device is not None:
            x, y = x.to(device), y.to(device)
        _, c, t = _forward_last_step(forward_model, x, y)
        num += c
        den += t
    forward_model.train()
    return num / den


def _run_eval_protocol(
    task: TaskSpec,
    arm: str,
    forward_model,
    device: torch.device,
    batch_size: int,
    n_batches: int,
    seed: int,
) -> dict[str, float]:
    """Post-selection generalization eval over `task.eval_protocol` (spec
    §4/§6): one accuracy entry per `EvalConfig`, keyed `"T{T}"` or
    `"T{T}_p{num_pairs}"`. Tasks with no eval_protocol entries (psMNIST,
    pendulum) get `{}` here -- psMNIST's test accuracy is computed by its
    own dedicated loader-based path in `run_arm`, and pendulum has no
    post-selection generalization sweep defined by spec §4 beyond the fit
    metric itself."""
    acc: dict[str, float] = {}
    for ec in task.eval_protocol:
        key = f"T{ec.T}" if ec.num_pairs is None else f"T{ec.T}_p{ec.num_pairs}"
        acc[key] = round(
            _eval_generator_task(
                task,
                arm,
                forward_model,
                T=ec.T,
                seed=seed,
                n_batches=n_batches,
                batch_size=batch_size,
                num_pairs=ec.num_pairs,
                device=device,
            ),
            4,
        )
    return acc


# ------------------------------------------------------------- training loops
def _train_step_based(
    task: TaskSpec,
    arm: str,
    state_model: nn.Module,
    forward_model,
    opt: torch.optim.Optimizer,
    gen: torch.Generator,
    steps: int,
    eval_every: int,
    ckpt_T: int,
    train_T: int,
    batch_size: int,
    direction: str,
    device: torch.device,
) -> tuple[int, float, dict | None]:
    """Checkpoint-selection training loop for the three generator-based
    tasks (S5/MQAR/pendulum). `state_model` (the raw, uncompiled module)
    owns `state_dict()`/`load_state_dict()`; `forward_model` (possibly
    `torch.compile`-wrapped) is what actually gets called -- compiling a
    second time for eval would be wasteful, and a compiled module's
    `state_dict()` keys can carry a `_orig_mod.` prefix, so checkpointing
    always goes through `state_model` (mirrors `hetero_lab.py`'s
    `run_arm`/`_DeviceCastTagger` convention)."""
    better = (lambda m, b: m > b) if direction == "ge" else (lambda m, b: m < b)
    best_metric = -1.0 if direction == "ge" else 1e9
    best_state, best_step = None, 0
    for step in range(1, steps + 1):
        state_model.train()
        batch = _to_device(_draw_batch(task, batch_size, train_T, gen), device)
        loss, _, _ = _forward_step_based(task, arm, forward_model, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if eval_every and step % eval_every == 0:
            metric = _eval_generator_task(
                task,
                arm,
                forward_model,
                T=ckpt_T,
                seed=CKPT_EVAL_SEED,
                n_batches=CKPT_EVAL_BATCHES,
                batch_size=batch_size,
                device=device,
            )
            if better(metric, best_metric):
                best_metric, best_step = metric, step
                best_state = {k: v.detach().clone() for k, v in state_model.state_dict().items()}
    return best_step, best_metric, best_state


def _train_epoch_based(
    state_model: nn.Module,
    forward_model,
    opt: torch.optim.Optimizer,
    loader,
    gen: torch.Generator,
    epochs: int,
    device: torch.device,
) -> tuple[int, float, dict | None]:
    """Epoch-based training loop (psMNIST): checkpoint once per epoch on
    val accuracy (spec §4/Budget docstring: epoch-based tasks have no
    `eval_every`, they checkpoint once per epoch instead)."""
    best_metric, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        state_model.train()
        for x, y in loader.train_epoch(gen):
            x, y = x.to(device), y.to(device)
            loss, _, _ = _forward_last_step(forward_model, x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_acc = _eval_last_step_loader(forward_model, loader.val(), device=device)
        if val_acc > best_metric:
            best_metric, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in state_model.state_dict().items()}
    return best_epoch, best_metric, best_state


# --------------------------------------------------------------- ledger
def _dedup_key(row: dict) -> tuple:
    return (row["round"], row["task"], row["variant"], row["seed"])


def _row_exists(row: dict) -> bool:
    """True iff `RESULTS` already has a row with the same (round, task,
    variant, seed) key (spec §7 dedup guard)."""
    if not os.path.exists(RESULTS):
        return False
    key = _dedup_key(row)
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if _dedup_key(json.loads(line)) == key:
                return True
    return False


def _require_checkpoint(best_state: dict | None, task_name: str, hint: str) -> None:
    """Fail loud rather than let `run_arm` assemble a row around the
    sentinel metric (-1.0 for "ge", 1e9 for "le") a training loop returns
    when no checkpoint was ever selected -- e.g. `eval_every` never
    divides any step in `[1, steps]`, or `eval_every=0`. The ledger is
    append-only, so a row must never be built from a sentinel that looks
    like a real measurement; raising here (rather than silently forcing an
    extra end-of-run eval) keeps checkpoint selection exactly what
    `eval_every` configured -- a misconfigured budget is a bug to fix and
    re-run, not a training loop to silently patch around."""
    if best_state is None:
        raise RuntimeError(
            f"{task_name}: no checkpoint was ever selected ({hint}) -- refusing to "
            "assemble a row around the sentinel metric. Check that eval_every "
            "divides at least one step in [1, steps] (or, for epoch-based tasks, "
            "that epochs >= 1)."
        )


def _resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"--device cuda requested but CUDA is not available in this "
            f"environment (torch.cuda.is_available() is False, "
            f"torch=={torch.__version__}); run on a CUDA-enabled machine, "
            "or omit --device to use cpu."
        )
    return device


def _model_layers(model: nn.Module) -> int:
    """Layer count for the ledger row's `"layers"` field, uniform across
    both model families `build_model` constructs: `MinGRUStack`-based
    models (`len(model.stack.blocks)`, the eight mixer arms) and the `gru`
    arm's plain `nn.GRU`-based models (`model.rnn.num_layers`), mirroring
    `probes.py`'s `run_one` convention for the same GRU-vs-MinGRUStack
    split (`model.rnn.num_layers if isinstance(model, GRUTagger) else
    len(model.stack.blocks)`)."""
    stack = getattr(model, "stack", None)
    if stack is not None:
        return len(stack.blocks)
    return model.rnn.num_layers


# ------------------------------------------------------------------ run_arm
def run_arm(
    *,
    round_tag: str,
    task: TaskSpec,
    arm: str,
    seed: int,
    lr: float | None = None,
    batch_size: int | None = None,
    steps: int | None = None,
    epochs: int | None = None,
    eval_every: int | None = None,
    train_T: int | None = None,
    device: str = "cpu",
    compile_model: bool = False,
    dry_run: bool = False,
) -> dict:
    """Train one seed of one arm on one task, select a checkpoint on the
    task's fit metric, evaluate its generalization protocol, and emit one
    ledger row (spec §4 "Single-arm run"). Side-effect-free on import --
    a later in-job campaign script imports this function directly (spec
    §5 "In-job campaign script") and calls it with `dry_run=True`.

    Parameters
    ----------
    round_tag : str
        Ledger `round` value (e.g. `"bench-s5-02"`).
    task : TaskSpec
        One of `experiments.benchmark_tasks.TASKS`'s values (or a
        `dataclasses.replace`d variant, e.g. for tests/selftest).
    arm : str
        One of `ARM_REGISTRY` (`"log"`/`"signed"`/`"rotation"`/
        `"givens"`/`"delta"`).
    seed : int
        Seeds `torch.manual_seed` (model init) and the training generator
        (`1 + 10_000 * seed`, spec §7 seeding convention); eval seeds are
        fixed and independent of this value.
    lr, batch_size, steps, epochs, eval_every : optional
        Override the corresponding `task.budget` field; `None` keeps the
        `TaskSpec`'s own value. `steps`/`epochs` are ignored for tasks that
        don't use that field (`task.budget.steps`/`epochs` is `None`).
    train_T : int, optional
        Training sequence length for the three generator-based tasks
        (default `TRAIN_T`); unused for psMNIST. Exposed for tests/
        selftest, not the CLI (spec's constant "Train at T=64" is fixed
        for real runs).
    device : {"cpu", "cuda"}, default "cpu"
    compile_model : bool, default False
        Wrap the built model with `torch.compile` before training/eval.
    dry_run : bool, default False
        Print the row and return it without appending to `RESULTS`.

    Returns
    -------
    dict
        The assembled ledger row (spec §6 schema).
    """
    if arm not in ARM_REGISTRY:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_REGISTRY)}")
    if task.loss_mode == "last_step":
        assert task.budget.epochs is not None, f"{task.name}: last_step task must set Budget.epochs"
    else:
        assert task.budget.steps is not None, f"{task.name}: step-based task must set Budget.steps"

    device_obj = _resolve_device(device)
    effective_lr = task.budget.lr if lr is None else lr
    effective_batch_size = task.budget.batch_size if batch_size is None else batch_size
    effective_steps = None if task.budget.steps is None else (task.budget.steps if steps is None else steps)
    effective_epochs = (
        None if task.budget.epochs is None else (task.budget.epochs if epochs is None else epochs)
    )
    effective_eval_every = (
        None
        if task.budget.eval_every is None
        else (task.budget.eval_every if eval_every is None else eval_every)
    )
    effective_train_T = TRAIN_T if train_T is None else train_T

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(1 + 10_000 * seed)

    model = build_model(task, arm)
    model = model.to(device_obj)
    forward_model = torch.compile(model) if compile_model else model

    opt = torch.optim.Adam(model.parameters(), lr=effective_lr)
    t0 = time.time()

    if task.loss_mode == "last_step":
        loader = task.data(effective_batch_size)
        assert loader.batch_size == effective_batch_size, (
            f"loader batch_size ({loader.batch_size}) does not match the "
            f"resolved Budget.batch_size ({effective_batch_size}) -- "
            "wiring bug (quality-review carry-over, Task 2 report)"
        )
        best_epoch, best_metric, best_state = _train_epoch_based(
            model, forward_model, opt, loader, gen, effective_epochs, device_obj
        )
        _require_checkpoint(best_state, task.name, f"epochs={effective_epochs}")
        model.load_state_dict(best_state)
        selection_value = round(best_metric, 4)
        # Honest fit metric (item 5): re-evaluate the SELECTED checkpoint
        # once more on the val split for `ckpt[task.fit_metric]` -- `val()`
        # iterates in a fixed, non-shuffled order (no seed to vary), so this
        # re-eval is deterministic and equals `selection_value` exactly for
        # psMNIST; still recorded via the same code path as the step-based
        # branch below so `ckpt[task.fit_metric]` always means "the selected
        # checkpoint's held-out metric", not a max-over-selection-evals
        # statistic, uniformly across all four tasks.
        fit_metric_value = round(
            _eval_last_step_loader(forward_model, loader.val(), device=device_obj), 4
        )
        acc = {"test": round(_eval_last_step_loader(forward_model, loader.test(), device=device_obj), 4)}
        selected_step, ckpt_key = best_epoch, "epoch"
        config_budget = {"lr": effective_lr, "batch_size": effective_batch_size, "epochs": effective_epochs}
        config_extra: dict = {"permutation_seed": loader.permutation_seed}
    else:
        ckpt_T = _ckpt_eval_T(task.name, effective_train_T)
        best_step, best_metric, best_state = _train_step_based(
            task,
            arm,
            model,
            forward_model,
            opt,
            gen,
            effective_steps,
            effective_eval_every,
            ckpt_T,
            effective_train_T,
            effective_batch_size,
            task.fit_direction,
            device_obj,
        )
        _require_checkpoint(
            best_state, task.name, f"steps={effective_steps}, eval_every={effective_eval_every}"
        )
        model.load_state_dict(best_state)
        selection_value = round(best_metric, 4)
        # Honest fit metric (item 5, "kill the winner's curse"): checkpoint
        # selection above picks the best of <= steps/eval_every evaluations
        # at CKPT_EVAL_SEED -- a max-over-N (min-over-N for "le" tasks)
        # selection statistic that is biased relative to the selected
        # checkpoint's TRUE held-out metric. Re-evaluate the SELECTED
        # checkpoint exactly ONCE more, at the same checkpoint-selection
        # configuration (T=ckpt_T, same batch count), on a NEW dedicated
        # fixed seed (FIT_EVAL_SEED -- distinct from both CKPT_EVAL_SEED and
        # FINAL_EVAL_SEED, so no two of the three re-use the same draw).
        # That single fresh evaluation, not the selection-time value, is
        # what `ckpt[task.fit_metric]` reports; the selection-time value is
        # kept separately as `ckpt["selection_<metric>"]` for transparency.
        fit_metric_value = round(
            _eval_generator_task(
                task,
                arm,
                forward_model,
                T=ckpt_T,
                seed=FIT_EVAL_SEED,
                n_batches=CKPT_EVAL_BATCHES,
                batch_size=effective_batch_size,
                device=device_obj,
            ),
            4,
        )
        acc = _run_eval_protocol(
            task, arm, forward_model, device_obj, effective_batch_size, FINAL_EVAL_BATCHES, FINAL_EVAL_SEED
        )
        selected_step, ckpt_key = best_step, "step"
        config_budget = {
            "lr": effective_lr,
            "batch_size": effective_batch_size,
            "steps": effective_steps,
            "eval_every": effective_eval_every,
        }
        config_extra = {}

    config = {"budget": config_budget, **config_extra}
    if task.name == "pendulum":
        # Record tau (the TaskSpec's own fit_threshold) directly in the row
        # (item 3): a later code edit to PENDULUM_TAU must never be able to
        # silently re-judge already-landed rows against a different
        # threshold than the one they were actually run/selected under.
        config["tau"] = task.fit_threshold
    if device_obj.type != "cpu":
        config["device"] = device
        config["torch"] = torch.__version__
    if compile_model:
        config["compile"] = True
    scan_mode = os.environ.get("MINGRU_SCAN")
    if scan_mode:
        config["scan"] = scan_mode

    rec = {
        "round": round_tag,
        "task": task.name,
        "variant": arm,
        "layers": _model_layers(model),
        "seed": seed,
        "steps": selected_step,  # step index for step-based tasks, epoch index for psMNIST
        "acc": acc,
        "secs": round(time.time() - t0, 1),
        "ckpt": {
            ckpt_key: selected_step,
            task.fit_metric: fit_metric_value,
            f"selection_{task.fit_metric}": selection_value,
        },
        "config": config,
    }
    print(json.dumps(rec), flush=True)
    if not dry_run:
        if _row_exists(rec):
            print(
                f"skip: duplicate row for (round={round_tag}, task={task.name}, "
                f"variant={arm}, seed={seed}) -- ledger append skipped",
                flush=True,
            )
        else:
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
    return rec


# ------------------------------------------------------------------- selftest
class _SyntheticPsMNISTLoader:
    """Synthetic stand-in for `PsMNISTLoader`, used only by `--selftest`/
    tests (brief: psMNIST "may use synthetic stand-in batches ... to avoid
    download"). Implements the same `Loader` protocol (`train_epoch`/
    `val`/`test`) with small random `last_step` batches -- no torchvision,
    no MNIST, no fixed pixel permutation (irrelevant to a wiring smoke
    test)."""

    def __init__(
        self,
        batch_size: int,
        n_train: int = 16,
        n_val: int = 8,
        n_test: int = 8,
        T: int = 8,
        seed: int = 0,
    ) -> None:
        self.batch_size = batch_size
        self.permutation_seed = None  # no fixed pixel permutation -- synthetic, not real psMNIST
        g = torch.Generator().manual_seed(seed)
        self._train = (
            torch.rand(n_train, T, 1, generator=g),
            torch.randint(0, 10, (n_train,), generator=g),
        )
        self._val = (
            torch.rand(n_val, T, 1, generator=g),
            torch.randint(0, 10, (n_val,), generator=g),
        )
        self._test = (
            torch.rand(n_test, T, 1, generator=g),
            torch.randint(0, 10, (n_test,), generator=g),
        )

    def _batches(self, x: torch.Tensor, y: torch.Tensor):
        for start in range(0, x.shape[0], self.batch_size):
            yield x[start : start + self.batch_size], y[start : start + self.batch_size]

    def train_epoch(self, gen: torch.Generator):
        yield from self._batches(*self._train)

    def val(self):
        yield from self._batches(*self._val)

    def test(self):
        yield from self._batches(*self._test)


def _tiny_task_overrides() -> dict[str, TaskSpec]:
    """Tiny per-task `TaskSpec` variants for `--selftest`: the same
    loss_mode/data wiring as the canonical `TASKS` registry (so the real
    model/loss/eval dispatch code path is exercised end-to-end), but a
    handful of steps/epochs and a small batch so the four-task sweep runs
    in well under a second. psMNIST substitutes `_SyntheticPsMNISTLoader`;
    MQAR's generator is called with a smaller `num_pairs` so its
    presentation+query span still fits inside the shrunk sequence length.
    """
    tiny_eval_protocol: dict[str, tuple[EvalConfig, ...]] = {
        "s5": (EvalConfig(T=32),),
        "mqar": (EvalConfig(T=32, num_pairs=2),),
        "psmnist": (),
        "pendulum": (),
    }
    tiny: dict[str, TaskSpec] = {}
    for name, task in TASKS.items():
        tiny_budget = replace(
            task.budget,
            batch_size=8,
            steps=(4 if task.budget.steps is not None else None),
            epochs=(1 if task.budget.epochs is not None else None),
            eval_every=(2 if task.budget.eval_every is not None else None),
        )
        data = task.data
        if name == "mqar":

            def data(batch, T, gen, num_pairs=2, _make=task.data):
                return _make(batch, T, gen, num_pairs=num_pairs)

        elif name == "psmnist":

            def data(batch_size):
                return _SyntheticPsMNISTLoader(batch_size)

        tiny[name] = replace(task, budget=tiny_budget, data=data, eval_protocol=tiny_eval_protocol[name])
    return tiny


def _selftest() -> None:
    """Tiny model, few steps, all four tasks, all four loss modes --
    a wiring smoke test (brief), not evidence. Dry-run only."""
    print("benchmark_lab selftest: tiny arm x task sweep (all four loss modes)")
    required = ("round", "task", "variant", "layers", "seed", "steps", "acc", "secs", "ckpt", "config")
    for name, task in _tiny_task_overrides().items():
        row = run_arm(round_tag="selftest", task=task, arm="log", seed=0, device="cpu", dry_run=True)
        missing = [k for k in required if k not in row]
        assert not missing, f"{name}: row missing required key(s) {missing}"
        assert task.fit_metric in row["ckpt"], f"{name}: ckpt missing fit metric {task.fit_metric!r}"
        print(f"  {name:>9} ok: ckpt={row['ckpt']}, acc={row['acc']}")
    print("benchmark_lab selftest: all four tasks ok")


# ------------------------------------------------------------------------ CLI
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true", help="tiny model, few steps, all four tasks; exits after")
    p.add_argument("--round", help="ledger round tag, e.g. bench-s5-02")
    p.add_argument("--task", choices=sorted(TASKS), help="s5 | mqar | psmnist | pendulum")
    p.add_argument(
        "--model",
        choices=sorted(ARM_REGISTRY),
        help="log | signed | rotation | rotation-hetero | givens | delta | "
        "signed-givens | signed-delta | gru (the nine MATRIX_ARMS) | "
        "rotation-hetero-k5 | signed-delta-nh3 | signed-delta-nh4 "
        "(the three S5-only PROBE_ARMS)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None, help="override TaskSpec.budget.lr")
    p.add_argument("--batch-size", type=int, default=None, help="override TaskSpec.budget.batch_size")
    p.add_argument("--steps", type=int, default=None, help="override TaskSpec.budget.steps (step-based tasks)")
    p.add_argument("--epochs", type=int, default=None, help="override TaskSpec.budget.epochs (psMNIST)")
    p.add_argument("--eval-every", type=int, default=None, help="override TaskSpec.budget.eval_every")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--compile", action="store_true", dest="compile_model")
    p.add_argument("--dry-run", action="store_true", help="print row, skip ledger append")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.round or not args.task or not args.model:
        parser.error("--round/--task/--model are required (unless --selftest)")
    task = TASKS[args.task]
    run_arm(
        round_tag=args.round,
        task=task,
        arm=args.model,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        steps=args.steps,
        epochs=args.epochs,
        eval_every=args.eval_every,
        device=args.device,
        compile_model=args.compile_model,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
