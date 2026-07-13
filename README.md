# minGRU

PyTorch implementation of the minGRU from Feng, Tung, Ahmed, Bengio &
Hajimirsadeghi, *Were RNNs All We Needed?* (arXiv:2410.01201) — a recurrent
layer whose gates depend only on the current input, so the whole sequence
trains in one parallel scan instead of step-by-step backpropagation through
time (BPTT). Single file, no dependencies beyond `torch`.

This repo ships the base minGRU plus two variants that each fix a specific
gap in it: **`SignedMinGRU`** (state can flip sign, not just decay toward
zero) and **`RotationMinGRU`** (state can track operations that don't
commute, like composing permutations). All three share one `mixer=`
interface on `MinGRUBlock`/`MinGRUStack`. What each variant actually buys
you, in measured accuracy, is below.

## What this shows

Two word-problem tasks probe exactly the gaps above (full task/eval setup
in "Reproducing," below):

- **parity** — label each prefix of a bit string with its running XOR
  (chance 0.5). The natural solution is a state that flips sign on a 1 and
  holds on a 0: a transition coefficient of −1.
- **S3** — label each prefix of a sequence of permutations of 3 objects
  (picture cups being swapped and rotated) with their net composition so
  far (chance ≈ 0.17). Composition order matters, so this needs a
  transition that doesn't commute.

Class names (used in the code) and probe/results names (used in the table
below and in `probes.py`) don't match 1:1 — here's the bridge:

| module class (`mixer=`) | name in probes / results table | notes |
|---|---|---|
| `MinGRU` (`mixer="log"`, default) | `minGRU` | log-space, positive states; chance-level on both tasks at any depth |
| `SignedMinGRU`, `coupled=False` (default) | `minGRU-signed-tanh` | decoupled eigenvalue; recommended for parity-like tasks |
| `SignedMinGRU`, `coupled=True` (legacy) | `minGRU-signed` | pinned to `coupled=True` in `probes.py`; kept under its historical name |
| `RotationMinGRU` (`mixer="rotation"`) | `minGRU-rotsnap` | L=1 only; needs the `CKPT=1` best-val@128 protocol below |
| `torch.nn.GRU` (reference baseline) | `GRU` | state-dependent gating; the ceiling both tasks are measured against |

**Recommended:** for parity-like problems (state must flip sign based on a
running property), use `SignedMinGRU` with its default `coupled=False`
(`minGRU-signed-tanh` below) — it holds 0.996 mean accuracy at 16x the
training length (n=6, worst seed 0.979), vs. 0.610 for the legacy
`coupled=True` form. For problems needing non-commutative state tracking
(composing operations where order matters), use `RotationMinGRU`
(`mixer="rotation"`, one layer only; `minGRU-rotsnap` below) with the
`CKPT=1` best-val@128 protocol described in "Rotation variant" — it reaches
0.889 mean accuracy at 16x length (n=8), though only 2 of 8 seeds land the
exact solution, so budget for retries. The base `MinGRU` (log-space,
`mixer="log"`) stays at chance on both tasks regardless of depth — a
parameterization limit, not a training one. A standard GRU remains the
ceiling: state-dependent gating solves both tasks exactly at every tested
length with a single layer.

Numbers below are multi-seed means (torch 2.5.1, CPU; seed counts stated
per row). Protocol: seq2seq tagging (dense supervision), T_train=64,
d_model=64, batch 128, Adam lr 3e-3, budget ≤1600 steps, early-stop at
99.9% train-length accuracy — **except** `minGRU-rotsnap`, which uses the
best-val@128 protocol instead of early-stop.

| task | model | layers | seeds | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|---|---|
| parity | `GRU` | 1 | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| parity | `minGRU-signed` (`coupled=True`) | 1 | 3 | 1.000 | 0.894 | 0.719 | 0.610 |
| parity | **`minGRU-signed-tanh` (default)** | 1 | 6 | 1.000 | ≥0.9999 | 0.999 | 0.996 (worst seed 0.979) |
| S3 | `GRU` | 1 | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| S3 | `minGRU-signed` (`coupled=True`) | 1 | 3 | 0.414 | 0.339 | 0.275 | 0.223 |
| S3 | `minGRU-signed` (`coupled=True`) | 4 | 3 | 0.885 | 0.544 | 0.426 | 0.342 |
| S3 | **`minGRU-rotsnap` (best-val@128 protocol)** | 1 | 8 | 0.999 | 0.987 | 0.956 | 0.889 (exact-to-16x in 2/8 seeds) |

An earlier, single-seed run of this project reported 0.655 for the
S3/`coupled=True`/L=4/@256 cell; the 3-seed mean above (0.544) supersedes
it and is within normal seed variance, not a regression (full comparison
in `experiments/EXPERIMENTS.md`).

Base `MinGRU` (log-space) isn't re-tabulated multi-seed above: it cannot
represent a −1 transition or a non-commuting one at any width, so it
stays at chance on both tasks regardless of seed or depth — a
parameterization failure, not a training one.

A few caveats apply to every number above: all runs use one learning
rate, and null/partial results are budget-relative ("didn't land the
exact solution in 1600 steps" ≠ "cannot"). The minGRU wrappers also
include a block MLP that the `GRU` baseline lacks, which favors the
minGRU variants — this strengthens their negative results (chance-level
despite the extra capacity) and mildly weakens attribution of their
positive ones.

## Model

The minGRU removes the hidden-state dependency from the GRU's gates (which
eliminates the reset gate) and drops the tanh range restriction:

```
z_t  = sigmoid(Linear_z(x_t))
h~_t = Linear_h(x_t)
h_t  = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t
```

Because `z_t` and `h~_t` depend only on `x_t`, the recurrence is a
first-order linear scan `h_t = a_t ⊙ h_{t-1} + b_t`, computable over the
full sequence in parallel (Heinsen log-space scan via `logcumsumexp`) — no
BPTT. Parameter count is `O(2·d_h·d_x)` vs. `O(3·d_h·(d_x + d_h))` for a
standard GRU.

This implements the paper's **log-space variant** (their Appendix B, the
numerically stable form). It applies `g(x) = x + 0.5 (x ≥ 0), sigmoid(x)
(x < 0)` to candidate and initial states, which means **hidden states are
strictly positive**. That is a property of this parameterization, not of
the "vanilla" minGRU in the paper's Appendix A.

## Components

| class / fn | role |
|---|---|
| `MinGRU` | one scan layer (log-space, positive states, `a_t ∈ (0,1)`); parallel `forward`, recurrent `step` |
| `SignedMinGRU` | signed diagonal transitions (linear-space scan, `a_t ∈ (−1,1)`, unconstrained states); `coupled=False` (default, decoupled eigenvalue) or `coupled=True` (legacy reproduction); same API |
| `RotationMinGRU` | 2x2 block rotation transitions (non-diagonal, matrix scan); `snap` grid manufactures exact-angle attractors; validated at `L=1` only; same API |
| `linear_scan` | Hillis–Steele associative scan for `h_t = a_t·h_{t−1} + b_t` with signed scalar coefficients |
| `matrix_scan` | Hillis–Steele associative scan for `h_t = M_t @ h_{t−1} + b_t` with 2x2 matrix coefficients (non-commutative composition) |
| `MinGRUBlock` | pre-norm residual block: LN → mixer → +x, LN → MLP → +x (`mixer` selects `"log"`, `"signed"`, or `"rotation"`; `mixer_kwargs` forwarded to its constructor) |
| `MinGRUStack` | input projection → N blocks → final LN; full state threading; same `mixer=`/`mixer_kwargs` applied to every block |

`MinGRU` is deliberately atomic. A single layer's gates cannot condition on
accumulated state (that is the parallelism trade); stacking recovers
history-dependent gating with one level of indirection — layer *l*'s gates
see layer *l−1*'s hidden states — and the block MLP supplies the
cross-channel interaction the diagonal scan cannot (channel *i* never
touches channel *j* inside the recurrence). `RotationMinGRU`'s 2x2 blocks
mix their own 2 channels internally, but not across blocks — the MLP still
supplies interaction between blocks. Set `mlp_expansion=0` for scan-only
blocks to ablate the channel mixer separately.

## Usage

Training (parallel over T), default log-space mixer:

```python
stack = MinGRUStack(input_size=32, d_model=256, n_layers=4)   # mixer="log" (default)
y = stack(x)                          # (B, T, 32) -> (B, T, 256)
```

Selecting a different mixer via `mixer=`/`mixer_kwargs`:

```python
signed_stack = MinGRUStack(32, 256, n_layers=4, mixer="signed")
# decoupled eigenvalue (default) -- the better length-generalizing form

legacy_stack = MinGRUStack(
    32, 256, n_layers=4, mixer="signed", mixer_kwargs={"coupled": True}
)
# bit-exact reproduction of the previous default parameterization

rot_stack = MinGRUStack(32, 256, n_layers=1, mixer="rotation")
# non-diagonal 2x2 rotation blocks; validated at n_layers=1 only, and only
# under the best-val@128 training protocol -- see "Rotation variant" below
```

Streaming inference (O(1) memory; state is `n_layers × d_model` per sample):

```python
state = stack.init_state()
for x_t in event_stream:              # x_t: (B, input_size)
    y_t, state = stack.step(x_t, state)
```

Chunked / TBPTT training:

```python
carry = None
for chunk in sequence_chunks:         # each (B, T_chunk, input_size)
    y, carry = stack(chunk, state=carry, return_state=True)
    loss(y).backward()
    carry = [h.detach() for h in carry]
```

Learned initial state (module-owned, positivity by construction):

```python
m = MinGRU(32, 64, learnable_h0=True)   # zero-init == fixed g(0)=0.5 default
```

`learnable_h0` routes to the block/stack's mixer only when `mixer="log"` or
`"signed"`; `RotationMinGRU` owns its `h_0` unconditionally (see "Rotation
variant" below) and does not accept the flag.

## State conventions

All exposed state is **real hidden state** — an output of `forward()` or
`step()` — under one convention across every mixer:

- `forward(h_0=...)` takes `(B, 1, d_h)`; `step(h_prev=...)` takes
  `(B, d_h)`. Crossing between streaming and chunked modes needs an
  explicit `unsqueeze` (intentional — no silent dimension coercion). Same
  shapes for `MinGRU`, `SignedMinGRU`, and `RotationMinGRU`.
- **Do not pass pre-activations.** The paper's reference code treats
  `h_0` as a pre-activation and applies `g` to it internally; this
  implementation does not. Passing a pre-activation here silently
  double-applies `g` (~0.43 max error vs. 1e-5 when fixed) — a real
  footgun if you reuse the paper's chunked-carry pattern verbatim. For a
  learned initial state, use `learnable_h0=True` rather than passing a
  pre-activation through the call signature.
- **Validation (log-space `MinGRU` only).** Strictly negative entries in
  `h_0` raise, via `torch._assert_async` (device-side on CUDA, so chunked
  loops incur no per-chunk host sync). Exact zeros are accepted and
  clamped to the dtype's smallest normal before `log()`, since
  legitimately small states underflow to 0.0 in fp16/bf16. `SignedMinGRU`
  and `RotationMinGRU` accept any real `h_0` — no clamp, no check; the
  positivity constraint is a property of the log-space parameterization
  only.
- `RotationMinGRU.h_0` is an intrinsic learned parameter, not an optional
  flag (see "Rotation variant," below): a zero state has no orbit under
  the group action, so it can't demonstrate tracking, and a state on a
  reflection axis collapses reflections onto rotations.

The full derivation and code path for each rule above is in the
corresponding docstrings in `min_gru.py` (`forward`, `step`, `log_g`).

## Implementation notes

- **`log_g` gradient at 0.** This implementation uses a nested `where`
  instead of the paper's relu-guarded branch: value-identical everywhere,
  but it gives the correct gradient (`1/g(0) = 2`) at `x = 0` instead of
  the relu guard's incorrect 0, which silently deadens any zero-initialized
  parameter routed through it (e.g. `learnable_h0` at its default init —
  no crash, no wrong output, the parameter just never trains). Full
  derivation is in the `log_g` docstring.
- `step()` runs under `@torch.no_grad()` for every mixer. Training is
  intended through the parallel `forward`; if you need BPTT through the
  sequential path, remove the decorator.
- Numerical agreement between the parallel and sequential paths, fp32 at
  T=128: ~1e-5 for `MinGRU` (`logcumsumexp` accumulation), ~1e-7 for
  `SignedMinGRU` (no exp/log round-trip), ~1e-6 for `RotationMinGRU` —
  none exact by construction.

## The ladder: fading, flipping, turning, reading

Every mixer in this module is a machine that carries a small memory
along a sequence and updates it at each step. The differences between
them come down to one question: **what is the update allowed to do to
the memory?** Answer it four increasingly generous ways and you get a
ladder — each rung does everything below it, plus exactly one new
thing.

**Rung 1 — memory that fades (`MinGRU`).** The base minGRU's update
is: keep some fraction of what you had, blend in some of what you just
saw. The fraction is always between 0 and 1, so old memory can only
shrink toward new input. That's a moving average with an
input-controlled blend — genuinely useful (it is most of what
"context" means in practice), but it can *only* fade. Ask it whether
it has seen an even or odd number of 1s and it's stuck: an average of
what you saw carries no trace of even-versus-odd. Measured: chance on
parity at any depth.

**Rung 2 — memory that flips (`SignedMinGRU`).** Let the keep-fraction
go negative: multiply the memory by −1 and it flips sign.
Even-versus-odd is now trivial — flip on every 1, hold on every 0,
read the sign at the end. The new capability is alternation: state
that oscillates and cancels instead of only decaying. The subtlety the
experiments surfaced is that *reachable in principle* isn't enough —
the −1 has to be a place training naturally settles. That is why the
default parameterization matters: `tanh` saturates at exactly −1,
while the legacy coupled form can approach but never reach it, and the
shortfall compounds with sequence length.

**Rung 3 — memory that turns (`RotationMinGRU`).** A flip is still
just multiplication by a number, and numbers commute: ×(−1) then ×(+1)
equals ×(+1) then ×(−1). So no rung-2 machine can track anything where
*order* matters. The fix: make the memory a little arrow in a plane
and each update a rotation (or reflection) of that arrow. Rotations
don't commute — turn-then-flip lands somewhere different from
flip-then-turn — so the arrow can carry the running composition of
operations, like following three cups through a shuffle. The rung-2
subtlety returns, sharper: the useful angles (a third of a turn, for
three cups) are not places training naturally settles, so this variant
snaps its angles to an exact grid — which is also why the grid must
contain the angles your problem needs.

**Rung 4 — memory that reads itself (a standard GRU).** Everything
below shares one discipline: the update at step *t* is chosen by the
input at step *t* alone, never by looking at the current memory. That
discipline is the entire reason rungs 1–3 train in one parallel pass —
updates fixed in advance can be composed in any grouping, so a scan
works. A GRU breaks the discipline: it reads its memory before
deciding how to change it. That buys genuinely sequential computation
(one layer solves both probe tasks at every length tested, and harder
tasks this repo doesn't attempt), and it costs exactly the thing this
repo exists to keep — each step must wait for the one before it.

The ladder is not "worse to better." It's a menu of what you can
afford: rungs 1–3 keep parallel training and each buys one specific
new kind of memory; rung 4 buys the rest and pays with sequentiality.
The probes in "What this shows" measure precisely these boundaries,
and "Expressivity limits" below is the formal version of this picture.

## Expressivity limits

- **Positive states (log-space variant).** Each `MinGRU` block's scan
  output lives in `(0, ∞)` (the residual stream itself is signed). If
  hidden-state magnitude carries task signal, the halved per-dimension
  range is worth checking in a proxy run rather than assuming parity
  with a standard GRU. `SignedMinGRU` removes this constraint.
- **Expressivity class.** Input-dependent, state-independent transitions
  put all three mixers in the same broad class as Mamba/S6 and GLA:
  fixed-depth stacks are TC⁰ (Merrill et al., *The Illusion of State in
  State-Space Models*) and cannot do unbounded state tracking that a
  single nonlinear GRU layer can. (Circuit-complexity shorthand used
  throughout: **TC⁰** is the class of problems solvable by
  constant-depth parallel circuits — depth that does *not* grow with
  sequence length, which is exactly what a stack of parallel scans is.
  **NC¹** is the strictly-harder-believed class needing depth that
  grows logarithmically with input length; its complete problems, like
  composing arbitrary permutations of 5 elements (S5), are the
  canonical "inherently sequential" computations — the ones a true
  recurrence handles step by step and a constant-depth parallel model
  is believed unable to.) Depth recovers bounded-depth nonlinear
  interaction with history, not sequential computation. Depth-matched
  comparisons against a standard GRU confound layer count with
  mechanism. Non-diagonal transitions (`RotationMinGRU`) lift the
  *commutativity* restriction and provably solve bounded non-abelian
  tracking (S3, a solvable group) that no diagonal variant can at any
  width — but that is not the same as the unbounded, non-solvable-group
  tracking (e.g. S5, NC¹-complete) that requires state-dependent gates;
  see the Signed and Rotation variant sections below. The probes make
  all of this measurable.

## Signed variant (`SignedMinGRU`)

The log-space scan requires positive transition coefficients (log of
1−z), which hard-codes two limitations at once: hidden states confined
to (0, ∞) via `g`, and transition eigenvalues confined to (0, 1) —
monotone EWMA-style memory only. `SignedMinGRU` lifts both by switching
to a linear-space associative scan (`linear_scan`). Two parameterizations
of the eigenvalue are available:

```
a_t = tanh(Linear_s(x_t))                    # coupled=False (default)
h_t = a_t ⊙ h_{t−1} + z_t ⊙ Linear_h(x_t)    # eigenvalues in (−1, 1), no g

a_t = (1 − z_t) ⊙ tanh(Linear_s(x_t))        # coupled=True (legacy)
h_t = a_t ⊙ h_{t−1} + z_t ⊙ Linear_h(x_t)
```

`coupled=True` is a bit-exact reproduction of the parameterization this
module shipped before this update (identical parameter shapes and
construction order, so identical seeds give identical weights) — kept as
the one-flag legacy path.

**Why the default changed.** The coupled form imposes a ceiling
`|a_t| ≤ 1 − z_t`: reaching the eigenvalue −1 that a task like parity
needs asks the gate to *also* saturate (`z_t → 0`) — one target value,
two simultaneous saturations to pay for. That shows up as the
length-generalization decay in the parity rows of "What this shows,"
above. The decoupled form removes the ceiling entirely: `tanh`'s own
asymptote is the eigenvalue's attractor (the value the gate settles
toward under training), so it needs only one saturation and reaches the
target "for free," holding much closer to exact out to 4x-16x the
training length. That is why `coupled=False` is now the default — the
previous parameterization remains exactly reproducible behind
`coupled=True`.

This decoupled mechanism is Grazzi et al.'s (ICLR 2025, *Unlocking
State-Tracking in Linear RNNs Through Negative Eigenvalues*)
negative-eigenvalue range, instantiated in minGRU's gate structure — a
repo improvement, not an independent novelty claim (no incumbent
comparison against Grazzi et al.'s own parameterization has been run
here; see `experiments/SUMMARY.md`, Open work). The general motivation
for negative eigenvalues also traces to Merrill, Petty & Sabharwal
(2024, ICML): negative eigenvalues restore per-layer parity /
sign-alternation dynamics that a positive-diagonal scan cannot
represent at any width.

What decoupling does **not** restore: diagonal transitions still
commute (scalar multiplication), so non-abelian state tracking
(permutation composition and everything NC¹-complete) remains out of
reach at any width, coupled or not — that requires non-diagonal
transitions (`RotationMinGRU`, below) or state-dependent gates (i.e., a
standard GRU). All diagonal variants, signed or not, remain in TC⁰ per
Merrill et al.'s iterated-scalar-product argument.

Practical differences from `MinGRU`: any real `h_0` is legal (no
positivity check, no underflow clamp), and there are 3 linear heads
instead of 2 (mind parameter-matched comparisons). The scan is
O(T log T) work / O(log T) depth in pure torch ops. Numerical agreement
vs. the sequential path is covered in "Implementation notes," above.

The expressivity ladder, per layer:

| variant | scan | per-layer capability |
|---|---|---|
| `MinGRU` (a ∈ (0,1)) | log-space | monotone EWMA memory |
| `SignedMinGRU` (a ∈ (−1,1)) | linear-space | + parity, abelian tracking |
| `RotationMinGRU` (2x2 blocks) | matrix-combine | + non-commutative: S3/D3 exactly representable |
| state-dependent gates | none (sequential) | = a standard GRU |

The ladder's non-diagonal rung is implemented — see below.

## Rotation variant (`RotationMinGRU`)

Completes the expressivity ladder. State is `n = hidden_size / 2`
independent 2D (planar) blocks; per block, the transition is a full
2x2 affine map instead of a scalar one:

```
M_t = R(theta_t) @ diag(1, tanh(u_t))
h_t = M_t @ h_{t−1} + b_t,     b_t = z_t ⊙ Linear_h(x_t)
```

(`h_{t−1}`, `h_t`, `b_t` viewed as 2-vectors per block.) 2x2
rotation/reflection matrices form a non-abelian group under
composition (unlike scalar multiplication), so `RotationMinGRU` — built
on a non-commutative parallel scan, `matrix_scan` (Hillis–Steele over
the 2x2 affine monoid) — can represent state-tracking automata over
non-abelian groups that no diagonal scan can, at any width. D3
(isomorphic to S3, the smallest non-abelian group) embeds in O(2), so
one layer exactly represents the S3 running-product task; see the
mechanism verification in `experiments/SUMMARY.md` (per-block matrices
extracted from a trained model satisfy the D3 composition table to
~1e-4 — a certifiable automaton, not an inferred one).

The underlying idea — non-diagonal transitions from the orthogonal
family unlocking non-abelian state tracking in a parallelizable linear
RNN — is DeltaProduct's (Siems et al., NeurIPS 2025): they build
transitions as products of generalized Householder reflections (two
reflections compose to a rotation) and demonstrate group state
tracking including dihedral tasks. What is specific to this variant is
the fixed 2x2 planar-block parameterization with an explicit angle
head, the STE snap grid that manufactures attractors at exact group
angles, and the weights-level homomorphism certificate. No head-to-head
comparison against DeltaProduct/DeltaNet has been run here (see
`experiments/SUMMARY.md`, Open work) — treat this as a minGRU-native
take on their mechanism, not a claimed improvement over it.

**Angle snapping (`snap`).** With `snap` set (default `(2, 3, 4, 6)`,
cycled across blocks), `theta_t` is quantized per block to an exact
multiple of `2π/K` via a straight-through estimator (STE): forward uses
the snapped angle, gradient passes through the pre-snap "soft" angle
unchanged. This manufactures an attractor at exact group elements, the
same role `tanh`'s asymptote plays for `SignedMinGRU`'s eigenvalue:
without it (`snap=None`, a legitimate, documented ladder rung),
continuous rotation angles have no attractor and drift under length
generalization, since error in the angle compounds with sequence
length. **The snap grid must contain the group being tracked**: choose
`K` values whose rotations (`2π/K`) generate, or coincide with, the
target group's rotation subgroup, or the exact automaton is not
representable on the grid at all (e.g. tracking Z/5 needs a multiple
of 5 in `snap`). The default `snap=(2, 3, 4, 6)` was chosen for the
D3/S3 task; other state-tracking targets need their own grid.

**Validated at L=1 only.** Stacking `RotationMinGRU` mixers is not
supported: the straight-through discontinuity compounds across layers
and breaks snap training. Treat depth as an open question for this
mixer, not a configuration to rely on.

**Training protocol: best-val@128 selection + retry-on-flag.** The
exact automaton is reachable but is **not** a stable attractor of
standard training — runs wander in and out of the exact solution over
the course of optimization, and once train accuracy saturates the loss
is blind to the difference between an exact solution and a decaying
shortcut. The validated protocol replaces early-stop with
best-checkpoint selection by validation accuracy at a length *longer*
than the training length (T=128 when training at T=64 — not one of the
eventual eval lengths, so it cannot leak into reported metrics),
evaluated over the full step budget. A best-val@128 score below 1.0
flags the run as failed; in the recorded evidence (n=8 seeds) this
perfectly separated good from bad seeds — **retry** a flagged run
rather than trust it. `probes.py`'s `CKPT=1` env var implements this
protocol:

```
CKPT=1 python probes.py S3 minGRU-rotsnap
```

**Seed success rate.** Across 8 fresh seeds under this protocol, only
2 of 8 land the exact solution (accuracy 1.0 to the checked precision
at every length out to 1024); the rest are detectably flagged (best
val@128 < 1.0) and decay measurably at 4x-16x the training length. The
mean length-generalization numbers in "What this shows," above (0.987
@256, 0.956 @512, 0.889 @1024) are the honest average over all 8
seeds, including the flagged ones — not a best-seed headline. Per the
mechanism verification in `experiments/SUMMARY.md`, every seed —
including the flagged ones — contains a D3 representation readable off
its weights; failed seeds are simply 5-15x less exact, not missing the
mechanism. Budget for retries when reproducing this variant.

**Alternatives tried and dropped.** Three other fixes were tested and
abandoned: a full orthogonality constraint on the transition matrices,
a regularizer that pulls angles toward the snap grid, and post-hoc
projection/ablation of near-exact blocks at inference. Each either
hurt length generalization or was redundant with the best-val@128
selection above (full comparison in `experiments/SUMMARY.md`, rounds 5
and 8).

Practical differences from the other mixers: 4 linear heads (z, h,
theta, u) vs. `SignedMinGRU`'s 3 / `MinGRU`'s 2 (mind parameter-matched
comparisons); `h_0` is an intrinsic learned parameter with no
`learnable_h0` flag (see "State conventions"); `hidden_size` must be
even (`ValueError` otherwise). The scan is O(T log T) work / O(log T)
depth in pure torch ops, same tradeoff as `linear_scan`. Numerical
agreement vs. the sequential path is covered in "Implementation notes,"
above.

## Reproducing

`probes.py` tests the ladder empirically on the two word problems
defined in "What this shows" (seq2seq tagging with dense supervision,
following Merrill et al.'s setup). Models train at T=64 and are
evaluated at T=64 (in-distribution) and longer lengths (256/512/1024,
length generalization) — the length-gen columns are what separate
"expresses the recurrent solution" from "learned a depth-bounded
shortcut for the training length."

```
python probes.py TASK MODEL [N_LAYERS]
# TASK      in {parity, S3}
# MODEL     in {GRU, minGRU, minGRU-signed, minGRU-signed-tanh, minGRU-rotsnap}
#           (minGRU-signed is pinned to coupled=True: the legacy parameterization,
#            kept under its historical name so recorded rows keep their meaning)
# N_LAYERS  defaults to 1; RotationMinGRU (minGRU-rotsnap) is validated at L=1 only
# MAX_STEPS env var overrides the training budget (default 1600)
# CKPT=1    env var replaces early-stop with best-val@128 checkpoint selection --
#           required for minGRU-rotsnap's validated protocol; off by default so
#           legacy early-stop rows stay reproducible
```

Single-cell examples:

```
MAX_STEPS=200 python probes.py parity minGRU-signed     # legacy coupled reproduction
CKPT=1 python probes.py S3 minGRU-rotsnap                # protocol-correct rotation-snap run
```

A smoke-test grid covering the ladder (single seed per cell — the
multi-seed numbers in "What this shows," above, come from
`experiments/variants.py`, same protocol plus extra seeds and eval
lengths 512/1024):

```
for t in parity S3; do
  python probes.py $t GRU
  for m in minGRU minGRU-signed minGRU-signed-tanh; do
    python probes.py $t $m 1
    python probes.py $t $m 4
  done
done
CKPT=1 python probes.py parity minGRU-rotsnap 1
CKPT=1 python probes.py S3 minGRU-rotsnap 1
```

CPU-only is sufficient; most cells run in seconds to ~15 minutes
(1-layer cells fastest), rotation-snap cells run ~1-2 minutes each.
Expect small numeric differences across torch versions but the same
qualitative pattern.

The @512 column in "What this shows" is not separately tabulated in
`experiments/SUMMARY.md`; it is computed directly from
`experiments/lab_results.jsonl` using the identical seed sets as the
corresponding @256/@1024 cells. See `experiments/SUMMARY.md` for the
fully reproduced evidence trail (per-seed results, the mechanism
verification, and the record of what was tried and dropped).

## Tests

`python min_gru.py` runs the built-in suite: parallel-vs-sequential
equivalence for every mixer (`MinGRU`, `SignedMinGRU` decoupled and
`coupled=True`, `RotationMinGRU`, and their stacks), chunked-vs-full
equivalence for the same set (including carries with negative/unbounded
states), `SignedMinGRU(coupled=True)` construction-order determinism
(bit-exact legacy reproduction), `matrix_scan` vs. brute-force
sequential recurrence plus a gradcheck, `RotationMinGRU`'s snapped-angle
grid exactness and gradient flow into all four heads and `h_0`,
`RotationMinGRU`'s `ValueError` on odd `hidden_size`, `MinGRUBlock`'s
`ValueError` on an unknown `mixer` name, gradient flow through both
diagonal scans and into `h0_pre` **from zero-init** (guards the
`log_g` fix), `log_g` gradient at 0 equal to 2, and `h_0` validation
for the log-space variant (underflowed zeros accepted, negatives
raise).

## Further reading

Full per-seed evidence, the mechanism-verification detail, and the
record of what was tried and dropped behind the numbers in this README
live in `experiments/`:

- `experiments/SUMMARY.md` — curated synthesis: full multi-seed tables,
  the D3 mechanism-verification detail, what was tried and dropped, and
  open work.
- `experiments/EXPERIMENTS.md` — round-by-round experiment log with
  per-round detail.
- `experiments/lab_results.jsonl` — raw per-seed result rows.

## References

Feng, L., Tung, F., Ahmed, M. O., Bengio, Y., & Hajimirsadeghi, H.
(2024). *Were RNNs All We Needed?* arXiv:2410.01201.

Merrill, W., Petty, J., & Sabharwal, A. (2024). *The Illusion of State
in State-Space Models.* ICML 2024. arXiv:2404.08819.

Grazzi, R., et al. (2025). *Unlocking State-Tracking in Linear RNNs
Through Negative Eigenvalues.* ICLR 2025.

Siems, J., Carstensen, T., Zela, A., Hutter, F., Pontil, M., &
Grazzi, R. (2025). *DeltaProduct: Improving State-Tracking in Linear
RNNs via Householder Products.* NeurIPS 2025. arXiv:2502.10297.
