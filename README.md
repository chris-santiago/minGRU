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
interface on `MinGRUBlock`/`MinGRUStack`, and all three optionally accept
a time-aware decay term for irregularly-spaced sequences — a real-world
gap between events, not just a token count (see "Time-aware decay,"
below). What each variant actually buys you, in measured accuracy, is
below.

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
| `RotationMinGRU` (`mixer="rotation"`) | `minGRU-rotsnap` | runs at L=1; depth via the list-mixer rows (`minGRU-hetero-*`, `minGRU-rotation2`); needs the `CKPT=1` best-val@128 protocol below |
| `torch.nn.GRU` (reference baseline) | `GRU` | state-dependent gating; the ceiling both tasks are measured against |

**Recommended:** for parity-like problems (state must flip sign based on a
running property), use `SignedMinGRU` with its default `coupled=False`
(`minGRU-signed-tanh` below) — it holds 0.994 mean accuracy at 16x the
training length (n=6, worst seed 0.984), vs. 0.592 for the legacy
`coupled=True` form. For problems needing non-commutative state tracking
(composing operations where order matters), use `RotationMinGRU`
(`mixer="rotation"`, one layer only; `minGRU-rotsnap` below) with the
`CKPT=1` best-val@128 protocol described in "Rotation variant" — it reaches
0.958 mean accuracy at 16x length (n=8), though only 1 of 8 seeds lands the
exact solution, so budget for retries. If you only need behavior near the
training length, `SignedMinGRU` fits S3-like composition in-distribution
too (0.999 @64) — its shortcut just decays with length; rotation is what
holds at 4x-16x. The base `MinGRU` (log-space,
`mixer="log"`) stays at chance on both tasks regardless of depth — a
parameterization limit, not a training one. A standard GRU remains the
ceiling: state-dependent gating holds both tasks at 1.000 (to three
decimals) at every tested length with a single layer.

Numbers below are multi-seed means (torch 2.5.1, CPU; seed counts stated
per row). Protocol: seq2seq tagging (dense supervision), T_train=64,
d_model=64, batch 128, Adam lr 3e-3, budget ≤1600 steps; the selection
protocol is labeled per row. parity rows use early-stop at 99.9%
train-length accuracy; every S3 minGRU-variant row uses the best-val@128
checkpoint protocol ("Rotation variant," below), so the diagonal and
rotation rows compare *mechanisms* under one selection procedure rather
than mechanism-plus-protocol against mechanism. Base `minGRU` rows never
reach the early-stop criterion and consume the full budget.

| task | model | layers | seeds | protocol | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|---|---|---|
| parity | `GRU` | 1 | 3 | early-stop | 1.000 | 1.000 | 1.000 | 1.000 |
| parity | `minGRU` (base, log-space) | 1 | 3 | early-stop | 0.542 | 0.511 | 0.506 | 0.502 |
| parity | `minGRU-signed` (`coupled=True`) | 1 | 3 | early-stop | 1.000 | 0.866 | 0.687 | 0.592 |
| parity | **`minGRU-signed-tanh` (default)** | 1 | 6 | early-stop | 1.000 | 1.000 | 0.999 | 0.994 (worst seed 0.984) |
| S3 | `GRU` | 1 | 3 | early-stop | 1.000 | 1.000 | 1.000 | 1.000 |
| S3 | `minGRU` (base, log-space) | 1 | 3 | early-stop | 0.229 | 0.182 | 0.175 | 0.171 |
| S3 | `minGRU-signed` (`coupled=True`) | 1 | 3 | best-val@128 | 0.470 | 0.344 | 0.277 | 0.226 |
| S3 | `minGRU-signed` (`coupled=True`) | 4 | 3 | best-val@128 | 0.994 | 0.713 | 0.506 | 0.368 |
| S3 | `minGRU-signed-tanh` (default) | 1 | 3 | best-val@128 | 0.999 | 0.943 | 0.864 | 0.732 |
| S3 | `minGRU-signed-tanh` (default) | 4 | 3 | best-val@128 | 0.997 | 0.908 | 0.769 | 0.574 |
| S3 | **`minGRU-rotsnap`** | 1 | 8 | best-val@128 | 1.000 | 1.000 | 0.996 | 0.958 (exact-to-16x in 1/8 seeds) |

The S3 diagonal rows tell a sharper story than "can't fit": the
decoupled `signed-tanh` fits S3 in-distribution (0.999 @64) through a
bounded-depth shortcut and decays steadily with length — 0.732 @1024 at
L=1, and depth doesn't rescue it (0.574 @1024 at L=4) — while the
rotation row holds near the exact automaton (0.958 @1024). The coupled
legacy form fails to fit S3 at L=1 even under checkpoint selection —
the one knob separating it from the `signed-tanh` row is the
`|a_t| ≤ 1 − z_t` ceiling ("Signed variant," below), which makes every
strong eigenvalue cost a second saturation. So the diagonal/rotation
separation lives at length generalization, not training fit — exactly
where the representability argument ("Expressivity limits," below)
predicts it.

The deeper S3 diagonal cells carry high seed-to-seed variance (per-seed
acc@256 spans 0.66-0.76 for `coupled=True` L=4 and 0.83-0.95 for
`signed-tanh` L=4), so treat those means as indicative rather than
tight. Earlier early-stop records for the coupled S3 rows, and detail
on a train/eval generator-seeding fix applied across this table (seed 0
unaffected), live in `experiments/EXPERIMENTS.md`.

Base `MinGRU` (log-space) is measured at chance on both tasks, and
depth doesn't move it: L=4 rows (3 seeds, round `base-mingru` in
`experiments/lab_results.jsonl`) match the L=1 rows above — parity
0.516 @64 falling to 0.501 @1024, S3 0.222 @64 falling to 0.170 @1024.
It cannot represent a −1 transition or a non-commuting one at any
width, so this is a parameterization failure, not a training one.

A few caveats apply to every number above: all runs use one learning
rate, and null/partial results are budget-relative ("didn't land the
exact solution in 1600 steps" ≠ "cannot"). The minGRU wrappers also
include a block MLP that the `GRU` baseline lacks, which favors the
minGRU variants — this strengthens their negative results (chance-level
despite the extra capacity) and mildly weakens attribution of their
positive ones. Finally, every comparison in the table above is
internal — against the repo's own floor (base `MinGRU`), its prior
parameterization (`coupled=True`), or its ceiling (`torch.nn.GRU`). A
Grazzi-style negative-eigenvalue scan is not run as a baseline anywhere
in this repo. DeltaNet/DeltaProduct's transition rule is compared, but
as a mechanism-level reimplementation under this repo's protocol, not
their released code — see "Incumbent comparison (mechanism-level)"
under "Rotation variant," below. So the table above establishes
"reproduces the predicted behavior, minGRU-natively," not "competitive
with the incumbents"; the incumbent-comparison subsection is the one
place in this README that compares a published mechanism directly, and
even there it is transition-rule-vs-transition-rule under one protocol,
not a claim about the incumbents' released systems.

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
| `MinGRU` | one scan layer (log-space, positive states, `a_t ∈ (0,1)`); parallel `forward`, recurrent `step`; optional time-decay (`decay=`, `decay_rate=`, `log1p_delta=`) scales `a_t` only — see "Time-aware decay" |
| `SignedMinGRU` | signed diagonal transitions (linear-space scan, `a_t ∈ (−1,1)`, unconstrained states); `coupled=False` (default, decoupled eigenvalue) or `coupled=True` (legacy reproduction); same API; same time-decay kwargs as `MinGRU` |
| `RotationMinGRU` | 2x2 block rotation transitions (non-diagonal, matrix scan); `snap` grid manufactures exact-angle attractors; depth is a measured tradeoff, not a fixed limit (see "Rotation variant"); same API; same time-decay kwargs, scaling the whole 2x2 block (direction unaffected) |
| `linear_scan` | Hillis–Steele associative scan for `h_t = a_t·h_{t−1} + b_t` with signed scalar coefficients |
| `matrix_scan` | Hillis–Steele associative scan for `h_t = M_t @ h_{t−1} + b_t` with 2x2 matrix coefficients (non-commutative composition) |
| `MinGRUBlock` | pre-norm residual block: LN → mixer → +x, LN → MLP → +x (`mixer` selects `"log"`, `"signed"`, or `"rotation"`; `mixer_kwargs` forwarded to its constructor); `forward`/`step` take an optional `delta_t`, forwarded to the block's mixer unchanged |
| `MinGRUStack` | input projection → N blocks → final LN; full state threading; `mixer=` is a `str` (applied to every block, unchanged) or a `list[str]` of length `n_layers` mixing types per block (e.g. one rotation block inside a signed/log stack); `mixer_kwargs` is the current flat dict for a `str` mixer, or `None`/a dict keyed by mixer type for a `list` mixer, applied to every block of that type (two blocks of the same type share one config); more than one `"rotation"` entry warns once per construction (STE-compounding, doesn't block construction) and proceeds; `decay_layers="all"` (default) or `"last"` selects which blocks receive the resolved kwargs' decay keys, positionally, whatever each block's type; `forward`/`step` take an optional `delta_t`, routed only to blocks whose mixer has decay enabled |

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
y, state = stack(x)                   # (B, T, 32) -> (B, T, 256), per-block states
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
# non-diagonal 2x2 rotation blocks; single-mixer use, and only under the
# best-val@128 training protocol -- see "Rotation variant" below
```

Mixing mixer types across layers (`mixer=` as a `list[str]`, one name per
block; `mixer_kwargs` keyed by type instead of flat):

```python
import torch
from min_gru import MinGRUStack

hetero_stack = MinGRUStack(
    32, 256, n_layers=2, mixer=["signed", "rotation"],
    mixer_kwargs={"signed": {"coupled": True}, "rotation": {"snap": (2, 3, 6)}},
)
x = torch.randn(4, 10, 32)
y, state = hetero_stack(x)            # (B, T, 32) -> (B, T, 256), per-block states
# one rotation block inside a deeper signed stack (extract-then-compose);
# mixer_kwargs is keyed by mixer type, not applied flat, when mixer is a
# list -- see "Rotation variant" for the measured depth/hierarchy tradeoffs.
# A mixer list with more than one "rotation" entry (e.g. ["rotation",
# "rotation"]) warns once per construction (STE-compounding) and proceeds.
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
    y, carry = stack(chunk, state=carry)
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
parity at any depth. ("Time-aware decay," below, is a different axis
entirely: it makes this fading time-continuous — gated by the
real-world gap between events rather than by token count — and it
composes with every rung above, scaling how fast existing memory fades
between events without adding a new capability of its own.)

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
contain the angles your problem needs. Whether depth can supply the
feature extraction a single rung-3 layer lacks turns out to be a real
tradeoff, not a yes/no — see "Rotation variant," below.

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
angles, and the weights-level homomorphism certificate. Their
transition rule is compared here at the mechanism level — see
"Incumbent comparison (mechanism-level)," below — but no comparison
against their released code or published numbers has been run; treat
this variant as a minGRU-native take on their mechanism, not a claimed
improvement over it.

**Angle snapping (`snap`).** With `snap` set (default `(2, 3, 4, 6)`,
cycled across blocks), `theta_t` is quantized per block to an exact
multiple of `2π/K` via a straight-through estimator (STE): forward uses
the snapped angle, gradient passes through the pre-snap "soft" angle
unchanged. This manufactures an attractor at exact group elements, the
same role `tanh`'s asymptote plays for `SignedMinGRU`'s eigenvalue:
without it (`snap=None`, a legitimate, documented ladder rung),
continuous rotation angles have no attractor and drift under length
generalization, since error in the angle compounds with sequence
length. **The snap grid must contain the rotations being tracked.** A
block with `K` in its grid can land exactly only on angles that are
whole multiples of `2π/K` — picture a clock face with `K` positions.
If the pattern your task cycles through has a period that isn't on
that clock, the exact solution doesn't exist on the grid and the
model can only approximate it, no matter how long it trains.
Concretely: a task cycling through 5 states needs steps of `2π/5`,
which a `K=5` (or 10, 15, ...) clock has and a `K=4` or `K=6` clock
does not — so tracking Z/5 needs a multiple of 5 somewhere in `snap`.
Rule of thumb: for every cyclic pattern of period `n` the task must
track exactly, include a `K` that is a multiple of `n`. The default
`snap=(2, 3, 4, 6)` was chosen for the D3/S3 task — its rotation part
is a 3-cycle, covered by the `3` and `6` in the grid; other
state-tracking targets need their own grid.

How do you find your task's periods? Ask, for each repeatable
operation in your data: *if this operation happens over and over,
after how many repeats is the tracked state back where it started?*
That count is a period the grid must cover. A toggle (on/off, sign
flip) has period 2. A three-way cyclic swap has period 3. A running
counter mod 7 — day of week, for example — has period 7. If several
operations coexist, collect each one's period and cover them all (one
`K` per period, or a single `K` equal to their least common multiple).
And if nothing in your task ever cycles back exactly — states just
grow, drift, or fade — then there's no period to snap to, and the
rotation grid isn't the right tool for that part of the problem:
that's what the fading (decay) and flipping (signed) rungs are for.

**Depth: L=2 is validated under the best-val@128 protocol; L=4 is
untested.** Measured through the `MinGRUStack` list-mixer path (task
`S3`, best-val@128 protocol, 3 seeds each (0, 1, 2), `MAX_STEPS=1600`),
multi-seed means:

| L=2 stack (task `S3`) | seeds | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|
| rotation → signed (`minGRU-hetero-rs`) | 3 | 1.000 | 1.000 | 0.999 | 0.966 |
| rotation × 2 (`minGRU-rotation2`) | 3 | 1.000 | 0.999 | 0.983 | 0.941 |
| signed → rotation (`minGRU-hetero-sr`) | 3 | 0.999 | 0.976 | 0.940 | 0.863 |
| *L=1 reference (`minGRU-rotsnap`, above)* | *8* | *1.000* | *1.000* | *0.996* | *0.958* |

All three depth-2 configurations land close to the L=1 reference;
rotation×2 (two rotation blocks) still emits the multi-rotation
warning below. Seed caveat: these are n=3 cells on a variant whose own
n=8 run spans 0.859–1.000 @1024, so three seeds sit inside the
demonstrated seed-noise band — read the table as "depth-2 stacks train
and stay in the L=1 reference's range," not as a ranking of the three
configurations. Deeper homogeneous rotation stacks (L=4) are untested
under this protocol and remain open. (Run history:
`experiments/EXPERIMENTS.md`.)

**Depth vs. hierarchy: a tradeoff, not a ranking.** A second probe
task, `S3-hier` (chance ≈ 1/6), pairs consecutive sub-tokens through a
fixed Latin-square lookup built to be non-representable by either group
of order six. The generator for each pair must be *extracted* before it
can be *composed*: a lone rotation layer can't absorb the lookup by
relabeling angles the way it can for a plain group operation. Measured
at `MAX_STEPS=1600` unless noted (seeds per row as listed; every number
budget-relative), multi-seed means. Rotation- and delta-bearing rows
use best-val@128 checkpoint selection (val@384 where noted);
signed-tanh and `GRU` rows use the early-stop protocol they are
recorded under. The n=6 `minGRU-hetero-sr` row pools rounds
`hetero-legB-v2` (seeds 0–2) and `hetero-loop-03-base-ext` (seeds
3–5), same protocol:

| config (task `S3-hier`, chance ≈ 0.167) | seeds | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|
| signed → givens8 (`hetero-sg8`, `experiments/hetero_lab.py`) | 6 | 0.998 | 0.944 | 0.840 | 0.656 |
| signed → deltaproduct2 (`hetero-sd2`, `experiments/hetero_lab.py`) | 6 | 1.000 | 0.987 | 0.814 | 0.530 |
| signed → 64-state delta (`hetero-sdm`) | 6 | 0.607 | 0.501 | 0.450 | 0.357 |
| signed-tanh, L=2 (homogeneous) | 3 | 0.985 | 0.866 | 0.754 | 0.620 |
| signed → rotation (`minGRU-hetero-sr`) | 6 | 0.572 | 0.442 | 0.375 | 0.309 |
| rotation → signed (`minGRU-hetero-rs`) | 3 | 0.448 | 0.325 | 0.247 | 0.206 |
| rotation × 2 (`minGRU-rotation2`) | 3 | 0.365 | 0.256 | 0.212 | 0.189 |
| `deltaproduct2`, L=1 | 6 | 0.238 | 0.184 | 0.176 | 0.171 |
| `GRU`, L=1 | 3 | 0.232 | 0.184 | 0.175 | 0.171 |
| `minGRU-rotsnap`, L=1 | 3 | 0.225 | 0.183 | 0.174 | 0.170 |
| signed → rotation at 4x budget (`MAX_STEPS=6400`) | 3 | 0.890 | 0.658 | 0.570 | 0.502 |

Homogeneous 2-layer signed-tanh wins in-budget fit among the
rotation-family rows but shows the same length decay documented for
`SignedMinGRU` above: a diagonal-scan approximation to a genuinely
non-commutative composition, not the exact automaton. The
extract-then-compose rotation stack (signed → rotation) fits on one
seed in six at the standard budget (best-val@128 0.988 on that seed)
and sits near chance on the rest — its n=6 mean hides a 0.186–0.535
per-seed spread @1024. Rotation-bearing configurations show
a 0.86–1.0 per-seed spread at n=8 on plain `S3` — the large gaps here
(fit vs. chance) are the trustworthy signal, not small differences
between adjacent rows.

The 4x-budget row's mean hides a split: one of three seeds finds the
exact solution and generalizes to 0.9834 @1024 (the best
length-generalization figure measured on `S3-hier`), while the other
two don't. Best-val@128 correctly flags those two seeds (0.573, 0.763)
as runs to retry, not trust. This is the same fast-but-unstable
training dynamic already documented for plain `minGRU-rotsnap`,
extending into the hetero stack's harder joint extract-and-compose
optimization problem.

Why the results split this way: composing first is hard to undo — a
rotation layer fed raw sub-tokens builds a mixture nothing downstream
is observed to take apart — a design rationale consistent with
rotation→signed winning on plain `S3` (where tokens *are* the
operations) and sitting at chance here, though the order effect is not
itself isolated from other differences between the two stacks. Why the
working order still trains poorly is measured rather than
hypothesized, as far as linear probes can see: the extractor is not
the bottleneck on any recorded seed. On instrumented reruns, a linear
probe decodes the pair's generator from layer-1 states at ≥0.95 within
400–800 steps on every seed — including seeds that plateau
permanently — while the composer's snapped angles keep oscillating
between grid cells (7–24% of snapped indices flip between checkpoints
on plateau seeds, collapsing to ~0.1–1% only on a run that locks). The
failure is the composer's search for an exact discrete solution, and
that solution's training basin is measurably tiny at the standard
budget: retraining for 1600 steps from the winning weights perturbed
by 1% per-tensor noise recovers near-exact accuracy on 2 of 3 seeds
but returns to the exact automaton on none, at any perturbation scale
tested (one winner checkpoint, Gaussian perturbations, n=3 per scale;
longer retrain budgets untested — round `hetero-loop-14-basin`).
Auxiliary supervision or extractor-first schedules would target a
coupling bottleneck that does not exist; best-val@128 remains the
working tool — it hits 1.0 on runs that land the exact automaton and
flags the ones that don't.

Swapping the composer mechanism makes the attribution causal. Keeping
the signed extractor and replacing the snapped-rotation composer with
a DeltaProduct-style nh=2 delta-rule layer (`hetero-sd2`, top table
row) takes `S3-hier` fit from 1/6 seeds to 6/6 at the same budget
(best-val@128 = 1.0 by step 500–1600, no retries, no seed lottery),
while that same delta layer alone at L=1 sits at chance on all six
seeds — the extract-then-compose depth split is still doing the work.
The cost is exactness: the delta composer reaches 0.530 @1024 (n=6),
rising to ≈0.60 at 4x budget (selected at val@384, a non-test length,
since val@128 saturates once fit) and approaching an asymptote well
below the 0.983 @1024 that the rotation composer's rare exact run
holds. Diagnostics rule out amplitude decay, extractor feature drift,
input non-stationarity, and per-class map noise as fixable causes of
that drift — quantizing the composer's input preserves in-distribution
fit and *worsens* generalization, so the composer's continuous code is
functional, not noisy (per-seed evidence:
`experiments/EXPERIMENTS.md`, hetero-loop rounds).

Within the rotation family, per-token map richness separates reliable
composers from unreliable ones — measured by holding the per-token
state at 64 elements (every promoted mixer's size) and parameters
nearly constant (+2.2% full-stack). 2-dimensional rotation blocks fit
1/6 seeds whether snapped or continuous; 8-dimensional rotation
blocks (`givens8`: per-token transitions built from three rounds of
Givens rotations, special-orthogonal by construction, continuous,
trained through the same parallel associative scan) fit 4/6 with the
misses near-fit (best-val@128 0.94/0.98) rather than at chance. The
delta rule shrunk to the same 64-element state fits 2/6 with chance
plateaus (`hetero-sdm` above) — so the full-size delta composer's 6/6
owes much of its reliability to its 16x-larger state (1,024 elements
per token; capacity disclosure above) — though the Givens-vs-small-
delta comparison itself is a two-seed difference at n=6 with unmatched
parameter counts (14,624 vs 3,306 composer parameters), suggestive
rather than established. At matched state the Givens-8 composer also
posts the best length generalization among the checkpoint-selected
configurations that fit reliably (0.840 @512, 0.656 @1024 mean; best
seed 0.956/0.812), above the full-size delta composer's; the
homogeneous signed-tanh L=2 row (0.620 @1024) sits close behind under
its different (early-stop) protocol — per-seed rows and parameter
counts in `experiments/EXPERIMENTS.md`.

A lone rotation layer, rotation×2, rotation→signed, a lone delta-rule
layer (`deltaproduct2`, L=1), and an unconstrained `GRU` all sit at or
near chance on `S3-hier`: no single capability (depth alone, or a
composition mechanism alone) is enough — extracting the pair and
composing it non-commutatively both have to happen. Read together:
homogeneous signed depth trains reliably and decays with length;
composer reliability rises with per-token map richness within the
rotation family and with state size within the delta family; and
exactness at length remains unique to the snapped composer's rare
winner (0.983 @1024) — no continuous composer reached it. No configuration wins outright and no claim is
budget-independent: every number above is relative to the stated step
budget, not a claim that any configuration solves `S3-hier`.

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
flags the run as failed and should be retried rather than trusted.
`probes.py`'s `CKPT=1` env var implements the selection half of this
protocol:

```
CKPT=1 python probes.py S3 minGRU-rotsnap
```

**Seed success rate.** Across 8 fresh seeds under this protocol, only
1 of 8 lands the exact solution (accuracy 1.0 to the checked precision
at every length out to 1024); the other 7 reach best-val@128 = 1.0 (no
flag) yet still decay measurably at 4x-16x the training length (worst
seed 0.859 @1024). The mean length-generalization numbers in "What this
shows," above (1.000 @256, 0.996 @512, 0.958 @1024) are the honest
average over all 8 seeds, including the non-exact ones — not a
best-seed headline. Per the mechanism verification in
`experiments/SUMMARY.md`, every seed — including the non-exact ones —
contains a D3 representation readable off its weights; failed seeds
are simply less exact, not missing the mechanism.

The flag is therefore one-directional: best-val@128 < 1.0 reliably
means retry, but the flag passing does **not** certify exactness at
length — confirm length generalization directly rather than relying on
the flag alone, and budget for retries when reproducing this variant.

**Incumbent comparison (mechanism-level).** The DeltaNet (Yang et al.)
/ DeltaProduct (Siems et al., NeurIPS 2025) transition rule is
reimplemented as a lab mixer, `DeltaNetMixer` in
`experiments/variants.py`: `nh=1` (`"deltanet"`, a single Householder
reflection per token) and `nh=2` (`"deltaproduct2"`, two reflections
per token, composing to a rotation). This is a reimplementation of the
transition rule, not their released code (Triton/CUDA, a different
training stack that does not run in this repo's CPU-only environment),
measured under the same protocol as the S3 table above (3 seeds,
best-val@128 checkpoint selection, `MAX_STEPS=1600`):

| task | model | seeds | protocol | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|---|---|
| S3 | `deltanet` (`nh=1`), L=1 | 3 | best-val@128 | 0.419 | 0.356 | 0.345 | 0.339 |
| S3 | `deltaproduct2` (`nh=2`), L=1 | 3 | best-val@128 | 1.000 | 1.000 | 1.000 | 0.989 |
| S3 | *`minGRU-rotsnap`, L=1 (reference, above)* | *8* | *best-val@128* | *1.000* | *1.000* | *0.996* | *0.958* |

`nh=1` cannot fit S3 (mean@64 = 0.419, in the diagonal/reflection chance
band). A single reflection has determinant -1 on every application and
cannot compose to the even-permutation elements S3 needs, matching
DeltaProduct's own representability theory. `nh=2` fits S3 on every
seed and trains reliably with no retry protocol: all three seeds reach
best-val@128 = 1.000 by step 200-300 (against the 1600-step budget) and
no seed raises the sub-1.0 flag that `minGRU-rotsnap` relies on to mark
a run for retry. That is a materially better training profile than
`minGRU-rotsnap` on this task at this budget: the snap grid's rotsnap
reference lands the exact solution in only 1 of 8 seeds and still needs
the retry-on-flag protocol to separate that seed from the other 7,
while `deltaproduct2`'s 3 of 3 seeds all land near-exactly without any
retries (an unequal-sample comparison — n=3 vs. n=8 — though the
val@128-by-step-300 unanimity leaves little room for a seed-luck
reading). On parity, both incumbent configurations show the same
length-generalization inconsistency across seeds: `deltanet` mean@1024
= 0.851 (per-seed range 0.730-1.000) and `deltaproduct2` mean@1024 =
0.810 (per-seed range 0.692-1.000), both below this repo's recorded
`signed-tanh` parity result (n=6, mean@1024 = 0.994, worst seed 0.984).
Signed-tanh's tanh asymptote gives it a length-generalization
attractor the delta rule's beta/k parameterization does not share
under this protocol.

What the snap mechanism still uniquely provides: an exact solution when
a seed lands it (`minGRU-rotsnap`'s 1 exact-to-16x seed of 8, vs.
`deltaproduct2`'s near-exact-but-not-exact 0.974-0.999 @1024 across all
3 seeds) and the weights-level D3 homomorphism certificate (per-block
matrices extracted from a trained model satisfy the D3 composition
table to ~1e-4 — see above and `experiments/SUMMARY.md`).

**Capacity disclosure.** At the shared `d_model=64`, parameter counts
are `deltanet` 16,900 / `deltaproduct2` 25,480 vs. `RotationMinGRU`
12,544 / `SignedMinGRU` 12,480. Both incumbent configurations carry
more parameters than the minGRU variants they are compared against,
and `deltaproduct2` carries roughly 2x `deltanet`'s own count. More
materially, both incumbent configurations carry a per-token state of
`n_heads * d_k * d_v` = 1,024 elements (4 heads x 16 x 16), against 64
state elements per token for the minGRU variants. The incumbents
carry roughly 16x more state per token at this `d_model`, which favors
their fit and length-generalization numbers above. These are
same-`d_model`, not same-capacity, comparisons.

All of the above is budget-relative (1600-step budget, 3 seeds, this
repo's harness and protocol only): it is not a claim about the
incumbents' released systems, published training regimes, or published
numbers, and it does not run in the other direction either: nothing
here shows the minGRU variants would still lead at a larger budget or a
different protocol. (Run history: `experiments/EXPERIMENTS.md`.)

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

## Time-aware decay

Every mixer above assumes evenly-spaced steps: one update per token,
no notion of how much real time elapsed between events. `decay=` adds
an optional per-event forgetting term so a gap between events shrinks
whatever the recurrence was already going to keep, without touching
how it injects new information. Available on `MinGRU`, `SignedMinGRU`,
and `RotationMinGRU` alike, and threaded through `MinGRUBlock` /
`MinGRUStack`.

What follows works mechanism-first: the transition math and the
`delta_t` contract below, then two experiments — a channel ablation
asking what the mechanism adds beyond just handing a model the raw
time gaps as an input feature ("Does the decay mechanism add anything
beyond a `delta_t` feature?" — short answer: the feature does most of
the work; the mechanism adds a smaller, separately-measurable margin
on top), and a recovery check showing what an unhelpful decay rate
costs ("Recovery check"). Skip ahead to either if you want the
evidence before the mechanism.

**Mechanism.** Each step's transition coefficient is scaled by
`gamma = exp(-lambda * f(delta_t))`, where `lambda >= 0` is a per-
channel (per-block, for `RotationMinGRU`) decay rate and `f` is
identity or `log1p`. `gamma` multiplies the transition only —
injection (`b_t`, the new information entering at this step) is never
decayed, in every mixer:

- `MinGRU` (log-space): additive in log-space —
  `log(gamma * (1 - z_t)) = log(1 - z_t) - lambda * f(delta_t)` — no
  new `exp`/`log` round-trip.
- `SignedMinGRU`: `gamma * a_t` (the signed diagonal transition).
- `RotationMinGRU`: `gamma * M_t` — decay scales the whole 2x2 block
  by a positive scalar, so the snapped rotation angle is recovered
  unchanged from the decayed matrix; only amplitude is affected
  (direction/amplitude separation, unaffected by decay).

Because `lambda >= 0` and `delta_t >= 0`, `gamma` is always in `(0,
1]` — decay can only shrink the transition, never amplify it.

**The `delta_t` contract.** `delta_t` is `(B, T)` or `(B, T, 1)` to
`forward` (squeezed internally), `(B,)` or `(B, 1)` to `step`.
`delta_t[:, t]` is the gap *preceding* event `t`. `delta_t = 0` gives
`gamma = 1` exactly — no discount — and this holds at every position,
**including `t = 0`**: there is no implicit "first event is exempt
from decay" special case (see design notes, below); callers
who want no decay at the start of a sequence pass `delta_t[:, 0] = 0`
themselves. Negative entries are clamped to `0` with a
once-per-module-instance warning (a data-quality event, not a hard
error; the check runs on CPU tensors only — on CUDA it is skipped
rather than forcing a host sync, though the clamp always applies).
`log1p_delta=True` applies `log1p` before scaling by `lambda`
— useful when raw gaps span orders of magnitude, compressing the
range `lambda` needs to operate over. Passing `delta_t` with decay
disabled, or enabling decay without passing `delta_t`, both raise
`ValueError` at call time.

**Two modes**, set via `decay=`:

- `"fixed"`: `lambda = decay_rate`, a scalar buffer, uniform across
  channels/blocks — not learned.
- `"learnable"`: `lambda = softplus(rho)`, one `rho` per hidden
  channel (`MinGRU`/`SignedMinGRU`) or per block (`RotationMinGRU`);
  `rho` is initialized so `lambda == decay_rate` exactly at
  construction (`decay_rate` is then an *init* value, not a fixed
  constant).

**`decay_layers`** (stack-level, `MinGRUStack(..., decay_layers=...)`):
`"all"` (default) applies the decay keys in `mixer_kwargs` to every
block uniformly; `"last"` strips them from every block but the final
one, for last-layer-only decay (a common convention in stacked
recurrent encoders). Any other string raises `ValueError` at
construction.
`delta_t` passed to a stack's `forward`/`step` is routed only to
blocks whose mixer has decay enabled; if no block in the stack decays,
passing `delta_t` raises `ValueError`.

**Design notes** — two choices worth knowing about, since other
time-decayed RNNs commonly make them differently:

1. **`softplus` on the rate parameter itself, not a learned projection
   of `delta_t`.** A common alternative computes the rate as
   `relu(proj(delta_t))`. Both halves of that form were rejected:
   `relu` has the module's own documented dead-gradient problem (see
   `log_g` in "Implementation notes"), and swapping in `softplus` over
   a projection would make `softplus(proj(0))` strictly positive —
   decaying even at zero gap and violating the `delta_t = 0 =>
   gamma = 1` contract. Resolution: `softplus` on a plain learnable
   rate (`lambda = softplus(rho)`), not on any function of `delta_t`.
2. **No `t = 0` exemption.** Some implementations exempt the first
   event from decay as a special case; this module doesn't need one —
   the `delta_t[:, 0] = 0` convention (above) already gives `gamma = 1`
   there for a true sequence start. Not special-casing position 0 is
   also what keeps chunked-vs-full equivalence exact: an
   absolute-position exemption would have to know which chunk is the
   first one.

```python
from min_gru import MinGRUStack
import torch

stack = MinGRUStack(
    input_size=32, d_model=64, n_layers=2, mixer="signed",
    mixer_kwargs={"decay": "learnable", "decay_rate": 0.05, "log1p_delta": True},
)
x = torch.randn(8, 100, 32)                    # (B, T, input_size)
delta_t = torch.rand(8, 100).clamp(min=0.01)   # gap preceding each event
delta_t[:, 0] = 0                              # true sequence start
y, state = stack(x, delta_t=delta_t)           # (B, T, 64), per-block states
```

### Does the decay mechanism add anything beyond a `delta_t` feature?

`delta_t` can reach a model two ways: as an appended input **feature**
(`log1p(delta_t)` concatenated onto the token embedding, available to
every model) or **mechanically** (consumed by a mixer's decay path,
scaling its transition). `session-parity` (a running-XOR task that
resets at session boundaries — gaps well above a threshold, vs. gaps
well below it within a session) isolates the two channels under a
fairness rule: every model gets the feature; only decay-enabled rows
additionally get the mechanism. Three probe rows form a **channel
ablation**, not a head-to-head model comparison: feature channel only
(`minGRU-signed-tanh`), mechanism channel only
(`minGRU-signed-tanh-tdecay-mech`), and both channels together
(`minGRU-signed-tanh-tdecay`).

Protocol: seq2seq tagging, `T_train=64`, budget <=1500 steps,
early-stop at 99.9% train-length accuracy (the same protocol as the
other signed-tanh rows), 3 seeds (0, 1, 2), `decay="learnable"` at
`decay_rate=0.05` init, `log1p_delta=True`.

| task | model (channel) | seeds | acc@64 | acc@256 | steps to early-stop |
|---|---|---|---|---|---|
| session-parity | both channels (`minGRU-signed-tanh-tdecay`) | 3 | 1.000 | 0.998 | 233 (avg) |
| session-parity | feature channel only (`minGRU-signed-tanh`) | 3 | 0.998 | 0.993 | 600 (avg) |
| session-parity | mechanism channel only (`minGRU-signed-tanh-tdecay-mech`) | 3 | 0.971 | 0.947 | never (full 1500-step budget, every seed) |

**The feature channel does most of the work.** Isolating the mechanism
channel alone shows this directly: it loses to feature-only on both
accuracy axes (acc@64 0.971 vs. 0.998, acc@256 0.947 vs. 0.993) and
never reaches the early-stop threshold within the standard 1500-step
budget. A 4x-budget check (6000 steps, same 3 seeds) narrows the gap —
acc@256 rises to a 0.975 mean — but doesn't close it and still never
early-stops; the deficit looks structural, not simply an artifact of
under-training. Removing the feature channel doesn't change the
qualitative recovery-check picture either (below) — the mechanism-only
row's behavior there tracks the both-channels row closely.

**Both channels together beats feature-only, modestly and with a
disclosed confound**: it wins acc@256 on every matched seed (0.997 vs.
0.990, 0.999 vs. 0.997, 0.998 vs. 0.992 — though the ranges touch:
both-channels' worst seed equals feature-only's best at the ledger's
precision), converges to the early-stop threshold ~2.6x faster on
average (233 vs. 600 steps), and reaches a ~3x lower error rate (0.2%
vs. 0.7%). The learned rate also lands in a consistent, narrow band
across seeds (mean 0.052-0.053) — the mechanism settles on a timescale
that separates the within-session gap range (`[0.1, 1.0]`) from the
boundary range (`[50, 100]`) rather than drifting seed-to-seed. The
both-channels row also carries extra per-channel rate parameters the
feature-only row lacks, so its win is not the mechanism in isolation.

Read the two comparisons together: **the feature and mechanism
channels are complementary, not substitutable** — but asymmetrically
so. The feature channel carries the both-channels win; the mechanism
channel underperforms on its own and adds a separately-measurable,
smaller improvement on top of the feature, rather than standing in
for it.

### Recovery check: does an unhelpful decay rate anneal away?

On plain parity — a task that never needs a reset, but with the exact
`delta_t` distribution from session-parity supplied anyway — a
well-behaved decay mechanism should settle near "don't decay." This
checks that half, plus whether accuracy actually recovers to the
non-decay level (same protocol/seeds as above; `steps` omitted below
since both rows converge in <=100 steps):

| task | model | seeds | acc@64 | acc@256 | lambda mean |
|---|---|---|---|---|---|
| parity-timestamped | both channels (`minGRU-signed-tanh-tdecay`) | 3 | 1.000 | 0.786 | 0.0499 |
| parity | non-decay comparison (`minGRU-signed-tanh`) | 3 | 1.000 | 1.000 | n/a |

The learned rate does stay near its low `0.05` init (mean 0.0499, not
drifting toward heavier decay) — the qualitative half of the recovery
check holds. Accuracy does **not** fully recover: a ~21-point acc@256
gap remains (0.786 vs. 1.000). Cause: `log1p` of a boundary-scale gap
(`delta_t` in `[50, 100]`) is already ~4-4.6, so even at
`lambda ≈ 0.05`, `gamma ≈ exp(-0.05 * 4.3) ≈ 0.81` per boundary-scale
event — and session-parity's distribution puts roughly a dozen such
gaps into a 256-step sequence, compounding to a large state loss by
T=256 even though `lambda` looks small in isolation. Practical
consequence: `decay_rate` (or its learnable init) is a real
hyperparameter, not a knob training reliably anneals to zero on its
own when a task doesn't need decay — pick it with the task's actual
gap distribution in mind.

A finer init sweep (same protocol and seeds, 3 per cell; the `0.05`
rows are the tables above) makes the tradeoff explicit:

| `decay_rate` init | session-parity acc@256 | steps to early-stop (avg) | recovery acc@256 |
|---|---|---|---|
| 0.005 | 0.997 | 633 | 0.997 |
| 0.01 | 0.995 | 533 | 0.986 |
| 0.02 | 0.997 | 467 | 0.895 |
| 0.05 | 0.998 | 233 | 0.786 |
| *reference* | *feature-only: 0.993* | *600* | *non-decay: 1.000* |

Three things move monotonically and in lockstep with the per-boundary
`gamma = exp(-R * ~4.3)` (0.98 at `R=0.005` down to 0.81 at
`R=0.05`). The recovery gap closes as the init drops — essentially
gone (~0.3pp) at `0.005`. The session-parity advantage thins in the
other direction: every init beats the feature-only baseline at the
mean, but the accuracy differences between inits sit within seed
noise; what cleanly separates `0.05` is convergence speed (233 steps,
vs. 467-633 at the smaller inits — the smallest init is slower than
the baseline's 600). No single init gets both ends.

The sweep also sharpens *why* training won't fix this for you: the
learned rate moves **up but never down**. On session-parity, `lambda`
climbs above its init at every `R` (e.g. 0.0050 → 0.0057, 0.0500 →
0.0530) — when decay helps fit the training sequences, the gradient
signal exists and training uses it. On the recovery task it sits
exactly at init at every `R`: over-decay's cost only materializes at
lengths beyond `T_train`, so the training objective contains no
signal against it. The init is a prior that training refines in one
direction only. Since the trend is smooth and monotone, the reliable
way to set `decay_rate` is validation at your *target* sequence
length (the same longer-length selection lesson as the checkpoint
protocol above): around `0.05` when gaps are known-informative,
`0.005–0.01` as the conservative default when unsure, and
`decay=None` when they're known-irrelevant.

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
```

- **`TASK`** — one of `parity`, `S3`, `session-parity`,
  `parity-timestamped`. The last two supply `delta_t` (see "Time-aware
  decay"); `GRU` has no `delta_t` input path and raises `ValueError` on
  them.
- **`MODEL`** — one of `GRU`, `minGRU`, `minGRU-signed`,
  `minGRU-signed-tanh`, `minGRU-signed-tanh-tdecay`,
  `minGRU-signed-tanh-tdecay-mech`, `minGRU-rotsnap`, `minGRU-hetero-sr`,
  `minGRU-hetero-rs`, `minGRU-rotation2`. `minGRU-signed` is
  pinned to `coupled=True`: the legacy parameterization, kept under its
  historical name so recorded rows keep their meaning. The two
  `-tdecay` rows are `decay="learnable"` at `decay_rate=0.05`,
  `log1p_delta=True`; `-mech` skips the `delta_t` feature concat (see
  "Time-aware decay"). `minGRU-hetero-sr`/`-rs` are 2-layer
  `MinGRUStack(mixer=["signed", "rotation"])` stacks, one order each
  (signed→rotation, rotation→signed); `minGRU-rotation2` is
  `mixer=["rotation", "rotation"]`, the broken-baseline reference (emits
  one `UserWarning` at construction). All three fix `N_LAYERS` to the
  mixer list's length (2) — omit it or pass 2 explicitly; a conflicting
  value raises `ValueError`. See "Rotation variant" for their measured
  accuracy.
- **`N_LAYERS`** — defaults to `1`. The single-mixer `minGRU-rotsnap`
  row runs at `L=1` (its recorded protocol); rotation at depth runs
  through the list-mixer rows above — see "Rotation variant" for the
  measured depth tradeoff.
- **`MAX_STEPS`** (env var) — overrides the training budget (default
  1600).
- **`CKPT=1`** (env var) — replaces early-stop with best-val@128
  checkpoint selection; required for `minGRU-rotsnap`'s validated
  protocol, off by default so legacy early-stop rows stay reproducible.

Single-cell examples:

```
MAX_STEPS=200 python probes.py parity minGRU-signed     # legacy coupled reproduction
CKPT=1 python probes.py S3 minGRU-rotsnap                # protocol-correct rotation-snap run
python probes.py session-parity minGRU-signed-tanh-tdecay   # time-aware decay channel ablation
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

Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024).
*Parallelizing Linear Transformers with the Delta Rule over Sequence
Length.* arXiv:2406.06484.

Siems, J., Carstensen, T., Zela, A., Hutter, F., Pontil, M., &
Grazzi, R. (2025). *DeltaProduct: Improving State-Tracking in Linear
RNNs via Householder Products.* NeurIPS 2025. arXiv:2502.10297.
