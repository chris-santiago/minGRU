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
  S3-hier             : like S3, but the running product is over
                         GENERATORS PER PAIR of sub-tokens rather than
                         per sub-token: neither sub-token in a pair
                         determines the generator alone (a fixed
                         non-additive Latin-square lookup gives uniform
                         marginals over both), so a single-layer mixer
                         must both extract the pair's generator AND
                         compose it non-commutatively. Tests whether
                         depth buys the hierarchical (pair -> generator)
                         feature a single-layer mixer of any kind lacks,
                         via a heterogeneous stack (signed feature layer
                         extracting the pair, rotation layer composing
                         it); see minGRU-hetero-sr/-rs in MIXER_REGISTRY.
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
       TASK in {parity, S3, S3-hier, session-parity, parity-timestamped};
       MODEL in {GRU, minGRU, minGRU-signed, minGRU-signed-tanh,
       minGRU-signed-tanh-tdecay, minGRU-signed-tanh-tdecay-mech,
       minGRU-rotsnap, minGRU-hetero-sr, minGRU-hetero-rs,
       minGRU-rotation2} (GRU is not valid for the two timestamped tasks:
       it has no delta_t input path); N_LAYERS defaults to 1 for
       single-mixer models. The three list-mixer models
       (minGRU-hetero-sr/-rs, minGRU-rotation2) fix N_LAYERS to their
       mixer list's length (2); omit N_LAYERS for those or pass the
       matching value -- an explicit conflicting value raises
       ValueError. minGRU-rotation2 (two rotation blocks) is the known
       STE-compounding broken baseline -- constructing it emits exactly
       one UserWarning (see MinGRUStack).
       MAX_STEPS env var overrides the training budget (default 1600).
       CKPT=1 replaces early-stop with best-checkpoint selection by
       validation accuracy at T=128 (seed 5, n_batches=2), evaluated
       every EVAL_EVERY steps over the full step budget; off by default
       so legacy early-stop rows stay reproducible. Rows containing a
       rotation block (minGRU-rotsnap, minGRU-hetero-sr, minGRU-hetero-rs,
       minGRU-rotation2) are validated only under this protocol.
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
# decay against weak gradient pressure at T_TRAIN=64 (with a 1.0 init,
# lambda barely moved and the decay row tied its feature-only baseline;
# with the 0.05 init it wins consistently -- the README's channel-
# ablation section records the numbers).
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
    # Heterogeneous stacks (spec section 6, Task 1's MinGRUStack list-mixer
    # contract): one signed (decoupled tanh, default kwargs -- same config
    # as minGRU-signed-tanh) block and one rotation block (default snap
    # grid, mirroring minGRU-rotsnap's config) in a 2-layer stack, in each
    # order. mixer_kwargs=None: the list-mixer schema applies every type's
    # default kwargs (no per-type overrides needed here). Layer count is
    # fixed to len(mixer) == 2 by the list-mixer N_LAYERS rule (see
    # _resolve_n_layers) -- omit N_LAYERS or pass 2 explicitly.
    "minGRU-hetero-sr": (["signed", "rotation"], None),
    "minGRU-hetero-rs": (["rotation", "signed"], None),
    # Broken-baseline reference for the hetero rows above: two rotation
    # blocks, no signed block. Deliberate addition beyond spec section 6's
    # two listed hetero rows (which does not name this row) -- added so
    # the leg-A rotation x2 evidence cell quoted in the README has a
    # runnable public path through the committed registry, rather than
    # only existing via a scratch script's temporary MIXER_REGISTRY entry.
    # mixer_kwargs=None mirrors the hetero rows' rotation entry (default
    # snap grid, same as minGRU-rotsnap); constructing this row emits the
    # multi-rotation UserWarning by design (Task 1's MinGRUStack contract)
    # -- expected, not suppressed.
    "minGRU-rotation2": (["rotation", "rotation"], None),
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

# ------------------------------------------------------------- S3-hier
# S3-hier (spec section 6): each token is one of 6 sub-tokens ({0..5});
# consecutive sub-token PAIRS each select a generator via a fixed 6x6
# Latin-square lookup, which is then composed onto the running S3
# product -- so, unlike S3 above (one sub-token = one generator), a
# single sub-token never determines the generator alone. Dense labels
# update only when a pair completes (odd position); even positions (mid-
# pair, no completed pair yet) carry the previous composition, identity
# before the first completed pair. Chance level ~= 1/6.
#
# LATIN is a fixed, hard-coded Latin square -- NOT COMPOSE (S3's own
# Cayley table), and not isotopic to ANY group's Cayley table.
#
# An earlier version of this constant reused COMPOSE directly. That was
# insufficient: COMPOSE is (trivially) isotopic to S3 itself, and a
# rotation layer's per-token angle assignment (linear_theta is a LEARNED
# linear map of the input embedding, i.e. a free relabeling of which
# sub-token maps to which angle) gives it exactly the relabeling freedom
# an isotopy allows -- row permutation (relabel row-sub-tokens),
# column permutation (relabel column-sub-tokens), and symbol permutation
# (relabel which S3 element each snap angle represents). So a pair
# function isotopic to ANY group of the state-tracking task's order is
# partially representable by one rotation layer via that relabeling, not
# just a literal-index additive one (the narrower condition
# `_has_additive_violation` below checks). This was confirmed
# empirically: LATIN = COMPOSE let L=1 rotsnap reach 0.377 accuracy on
# S3-hier, well above chance (1/6 ~= 0.167).
#
# Order 6 has exactly two groups up to isomorphism (Z6, the cyclic
# group, and S3 ~= D3, the symmetric/dihedral group -- the standard
# classification of small groups), and RotationMinGRU's snap grid
# (2, 3, 4, 6) realizes both cyclic (Z_K) and dihedral (D_K, including
# D3 ~= S3) subgroups of O(2) -- see RotationMinGRU's docstring. So the
# correct diagnostic requirement is that LATIN be isotopic to NEITHER
# group: no row permutation f, column permutation g, symbol permutation
# h with LATIN[i][j] == h(G[f(i)][g(j)]) for G in {Z6, S3}. This is
# exactly the (informal) "quadrangle criterion" for Latin-square/group
# isotopy, specialized to order 6 where only two group isomorphism
# classes exist to rule out.
#
# Verified OFFLINE (not at import time -- an exhaustive isotopy search
# is O(6!^2) per candidate group, too expensive to repeat on every
# import): generated via 300 random "intercalate swaps" from the Z6
# addition table (a swap of the two values in a 2x2 checkerboard
# sub-block preserves the Latin property but is NOT an isotopy -- unlike
# row/col/symbol relabeling, which by definition stays inside one
# isotopy class, intercalate swaps can and typically do land outside the
# two group-isotopy classes; order 6 has 22 Latin-square isotopy classes
# total, only 2 of which are group-based). The resulting square was
# confirmed NOT isotopic to Z6's table and NOT isotopic to COMPOSE (S3's
# table) via brute-force isotopy search over all row/column permutations
# with a consistency-checked symbol map (script:
# generate_latin.py, kept out of the repo per this task's scratch-only
# constraint). It also still violates the cheaper, narrower additive
# check below (`_has_additive_violation`), which is retained as a fast
# import-time sanity check but is not sufficient on its own (see above).
LATIN = torch.tensor(
    [
        [3, 4, 5, 0, 1, 2],
        [4, 2, 3, 1, 5, 0],
        [2, 3, 4, 5, 0, 1],
        [0, 1, 2, 3, 4, 5],
        [1, 5, 0, 4, 2, 3],
        [5, 0, 1, 2, 3, 4],
    ]
)


def _is_latin_square(table):
    """True iff every row and column of ``table`` is a permutation of
    ``range(6)`` -- neither sub-token in a pair informs the selected
    generator alone (uniform marginals)."""
    six = list(range(6))
    rows_ok = all(sorted(table[i].tolist()) == six for i in range(6))
    cols_ok = all(sorted(table[:, j].tolist()) == six for j in range(6))
    return rows_ok and cols_ok


def _has_additive_violation(table):
    """Fast, NECESSARY-BUT-INSUFFICIENT sanity check -- not the real
    non-representability guarantee. True iff some cell breaks
    ``table[i][j] == (table[i][0] + table[0][j] - table[0][0]) % 6``,
    the narrow "literal-index" additive form. Passing this check (a
    violation exists) only rules out that one specific form; it does
    NOT rule out ``LATIN`` being isotopic to a group under some
    row/column/symbol relabeling -- exactly the loophole that let
    ``LATIN = COMPOSE`` (isotopic to S3, but additive-violating by this
    check) leak into a rotation layer's representable functions in an
    earlier revision (see the module comment above ``LATIN``). The real
    invariant is non-isotopy to both order-6 groups, checked by the slow
    ``_latin_is_group_isotopic`` / ``_verify_latin_non_isotopic``
    (opt-in via ``VERIFY_LATIN=1``, below) -- this function is retained
    only as a cheap, always-on early warning, not a substitute for that
    check."""
    base = table[0][0].item()
    return any(
        table[i][j].item() != (table[i][0].item() + table[0][j].item() - base) % 6
        for i in range(6)
        for j in range(6)
    )


assert _is_latin_square(LATIN), (
    "LATIN must be a Latin square (every row/column a permutation of "
    "range(6)); S3-hier's diagnostic power depends on neither sub-token "
    "informing the generator alone"
)
assert _has_additive_violation(LATIN), (
    "LATIN must not be additively decomposable, or a single rotation "
    "layer could solve the pair function by angle addition and S3-hier "
    "would not be diagnostic of leg B"
)


def _latin_is_group_isotopic(latin, group_table):
    """Exhaustive isotopy search: True iff there exists a row permutation
    ``f``, column permutation ``g``, and symbol permutation ``h`` with
    ``latin[f(i)][g(j)] == h(group_table[i][j])`` for every ``i, j`` --
    i.e. ``latin`` is isotopic to ``group_table``. ``h`` is built
    incrementally and checked for consistency (early-exits per
    ``(f, g)`` candidate on the first inconsistency), so unrelated
    tables typically fail fast; only genuinely isotopic pairs pay
    the full O(n^2) inner check. Order n! * n! * n^2 overall -- see
    ``_verify_latin_non_isotopic`` for the opt-in, order-6-sized call
    site; not intended to be called at import time or on every run.
    """
    import itertools

    n = latin.shape[0]
    latin_rows = latin.tolist()
    group_rows = group_table.tolist()
    for f in itertools.permutations(range(n)):
        permuted = [latin_rows[f[i]] for i in range(n)]
        for g in itertools.permutations(range(n)):
            symbol_map = {}
            consistent = True
            for i in range(n):
                for j in range(n):
                    latin_val = permuted[i][g[j]]
                    group_val = group_rows[i][j]
                    mapped = symbol_map.get(group_val)
                    if mapped is None:
                        symbol_map[group_val] = latin_val
                    elif mapped != latin_val:
                        consistent = False
                        break
                if not consistent:
                    break
            if consistent:
                return True
    return False


def _verify_latin_non_isotopic():
    """The real non-representability guarantee behind S3-hier's
    diagnostic power (spec section 6; see the module comment above
    ``LATIN``): asserts ``LATIN`` is isotopic to NEITHER order-6 group
    (Z6, the cyclic group; and S3, ``COMPOSE`` -- the only two groups of
    order 6 up to isomorphism). ``_has_additive_violation`` above is
    only a fast, necessary-but-insufficient proxy for this; this
    function is the actual invariant, ported from the offline
    ``generate_latin.py`` scratch script that originally produced
    ``LATIN`` so the guarantee is re-verifiable from the repo itself,
    not only from a script that never shipped.

    Deliberately slow (~seconds; two O(6!^2) exhaustive isotopy
    searches) and NOT run at import time -- opt in with
    ``VERIFY_LATIN=1`` (see the bottom of this function's call site,
    module level, below). Fails loudly (assertion error naming which
    group) if ``LATIN`` is ever isotopic to either group -- e.g. if
    someone edits the constant without re-running this check.
    """
    z6 = torch.tensor([[(i + j) % 6 for j in range(6)] for i in range(6)])
    assert not _latin_is_group_isotopic(LATIN, z6), (
        "LATIN must not be isotopic to Z6 (the cyclic group of order 6): "
        "a rotation layer's learned per-token angle assignment could "
        "represent this pair function via row/column/symbol relabeling, "
        "the same leak that made the original LATIN = COMPOSE diagnostic "
        "insufficient (see the module comment above LATIN)"
    )
    assert not _latin_is_group_isotopic(LATIN, COMPOSE), (
        "LATIN must not be isotopic to S3 (COMPOSE, the other group of "
        "order 6): a rotation layer's learned per-token angle assignment "
        "could represent this pair function via row/column/symbol "
        "relabeling, the same leak that made the original "
        "LATIN = COMPOSE diagnostic insufficient (see the module comment "
        "above LATIN)"
    )


# Opt-in, slow (~seconds) re-verification of the real LATIN invariant
# (non-isotopy to both order-6 groups) -- off by default so import stays
# fast, matching this module's other opt-in env flags (MAX_STEPS, CKPT):
#     VERIFY_LATIN=1 python probes.py S3-hier minGRU-hetero-sr
if os.environ.get("VERIFY_LATIN", "") not in ("", "0"):
    _verify_latin_non_isotopic()
    print("VERIFY_LATIN=1: LATIN confirmed non-isotopic to Z6 and to S3/COMPOSE", flush=True)


def make_s3_hier(batch, T, gen):
    """S3-hier generator (see module comment above): sub-tokens
    ``x in {0..5}``; pair ``(x[2k], x[2k+1])`` selects generator
    ``g = LATIN[x[2k], x[2k+1]]``, composed onto the running S3 product
    once the pair completes. Dense labels: even ``t`` (mid-pair) carries
    the previous composition (identity before the first completed
    pair); odd ``t`` is the just-updated composition."""
    x = torch.randint(0, 6, (batch, T), generator=gen)
    y = torch.zeros(batch, T, dtype=torch.long)
    state = torch.zeros(batch, dtype=torch.long)  # identity
    for t in range(T):
        if t % 2 == 1:
            g = LATIN[x[:, t - 1], x[:, t]]
            state = COMPOSE[g, state]  # g o state, mirroring make_s3
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
    "S3-hier": (make_s3_hier, 6, 6),
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


def _resolve_n_layers(mixer, n_layers):
    """Resolve a registry row's ``n_layers`` against its ``mixer`` spec
    (spec section 4/section 6: list-mixer N_LAYERS conflict rule).

    A list-mixer row (``minGRU-hetero-sr``/``-rs``) fixes the layer
    count to ``len(mixer)``: omitting ``n_layers`` (``None``, the CLI
    default) defers to that length; an explicit value that matches is a
    no-op; an explicit value that conflicts raises ``ValueError``. A
    single-mixer (``str``) row is unconstrained -- ``n_layers`` defaults
    to 1 when omitted, any explicit value passes through unchanged
    (prior behavior, bit-identical).

    Parameters
    ----------
    mixer : str or list of str
        The registry row's mixer spec (``MIXER_REGISTRY[name][0]``).
    n_layers : int or None
        Caller-supplied layer count, or ``None`` if omitted.

    Returns
    -------
    int
        The resolved layer count.

    Raises
    ------
    ValueError
        If ``mixer`` is a list and ``n_layers`` is given but does not
        equal ``len(mixer)``.
    """
    if isinstance(mixer, list):
        implied = len(mixer)
        if n_layers is not None and n_layers != implied:
            raise ValueError(
                f"n_layers={n_layers} conflicts with list-mixer row "
                f"mixer={mixer!r} (layer count is fixed at {implied}); "
                f"omit n_layers or pass {implied} explicitly"
            )
        return implied
    return 1 if n_layers is None else n_layers


def build(task, name, vocab, n_cls, n_layers=None):
    """Construct a model for ``task``/``name`` (``vocab``/``n_cls`` from
    ``TASKS[task]``, ``n_layers`` blocks).

    ``timestamped = task in TIMESTAMPED_TASKS`` is derived internally
    (rather than taken as a caller-supplied bool) since ``task`` already
    determines it and every call site already has ``task`` in hand.

    ``n_layers=None`` (the CLI default) defers to ``_resolve_n_layers``:
    1 for single-mixer (``str``) rows and ``GRU``, the mixer list's
    length for list-mixer rows; an explicit value that conflicts with a
    list-mixer row's length raises ``ValueError``.
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
        n_layers = _resolve_n_layers(mixer, n_layers)
        if name in MECHANICAL_ONLY_MODELS:
            delta_t_mode: DeltaTMode = "decay-only"
        elif (mixer_kwargs or {}).get("decay") is not None:
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
        return GRUTagger(vocab, n_cls, 1 if n_layers is None else n_layers)
    if name not in MIXER_REGISTRY:
        valid = ["GRU", *MIXER_REGISTRY]
        raise ValueError(f"unknown model {name!r}; valid: {valid}")
    mixer, mixer_kwargs = MIXER_REGISTRY[name]
    n_layers = _resolve_n_layers(mixer, n_layers)
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


def run_one(task, name, n_layers=None, max_steps=MAX_STEPS, ckpt=None, seed=0):
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
    # Read back the model's actual layer count for logging: n_layers may
    # have been None (CLI omitted) or a list-mixer row's implied value,
    # either resolved inside build() -- this avoids re-deriving the same
    # registry-lookup logic here just to print an accurate L=.
    n_layers = model.rnn.num_layers if isinstance(model, GRUTagger) else len(model.stack.blocks)
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
    # Heterogeneous stacks (leg A: does one rotation block survive depth
    # inside a mixed stack, on the existing S3 task; leg B: does depth buy
    # hierarchy, on the new S3-hier task). Rows containing a rotation
    # block run CKPT (best-val@128); the pure-signed L=2 row and GRU keep
    # early-stop. n_layers=2 matches both hetero rows' fixed list length.
    ("S3", "minGRU-hetero-sr", 2, 1600, True),
    ("S3", "minGRU-hetero-rs", 2, 1600, True),
    ("S3", "minGRU-rotation2", 2, 1600, True),
    ("S3-hier", "GRU", 1, 1600, False),
    ("S3-hier", "minGRU-signed-tanh", 2, 1600, False),
    ("S3-hier", "minGRU-rotsnap", 1, 1600, True),
    ("S3-hier", "minGRU-hetero-sr", 2, 1600, True),
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
        run_one(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
