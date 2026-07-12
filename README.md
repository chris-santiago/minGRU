# minGRU

PyTorch implementation of the minGRU from Feng, Tung, Ahmed, Bengio &
Hajimirsadeghi, *Were RNNs All We Needed?* (arXiv:2410.01201), with a
parallel-scan training path, an O(1)-memory streaming path, chunked/TBPTT
state carry, and stacked residual blocks. The module ships three sequence
mixers — log-space (`MinGRU`), signed diagonal (`SignedMinGRU`), and
non-diagonal 2x2 rotation (`RotationMinGRU`) — selected through a common
`mixer=` interface on `MinGRUBlock`/`MinGRUStack`.

Single file, no dependencies beyond `torch`.

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
# bit-exact reproduction of the pre-promotion SignedMinGRU

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

Full per-seed evidence, the mechanism-verification homomorphism test, and
the closed repair agenda behind the numbers in this README live in
`experiments/` — `experiments/SUMMARY.md` (curated synthesis),
`experiments/EXPERIMENTS.md` (round-by-round log), and
`experiments/lab_results.jsonl` (raw cells).

## State conventions

All exposed state is **real hidden state** — an output of `forward()` or
`step()` — under one convention across every entry point and every mixer:

- `MinGRU.forward(h_0=...)` takes `(B, 1, d_h)`; `step(h_prev=...)` takes
  `(B, d_h)`. Crossing between streaming and chunked modes needs an
  explicit unsqueeze — intentional, no silent dim coercion. Same shapes
  for `SignedMinGRU` and `RotationMinGRU`.
- **Do not pass pre-activations.** This deviates from the paper's
  reference code, which applies `g` to `h_0` (treating it as a
  pre-activation). That convention makes the natural chunked-carry
  pattern `forward(h_0=prev[:, -1:])` silently wrong (double-applies
  `g`, ~0.43 max error vs. 1e-5 when fixed). Here `log(h_0)` is used
  directly; the learned-init use case, where the pre-activation
  convention is genuinely useful, is encapsulated behind
  `learnable_h0=True` instead of the call signature.
- Validation: strictly negative entries in `h_0` (the signature of
  pre-activation misuse) raise, via `torch._assert_async` — device-side
  on CUDA, so chunked loops incur no per-chunk host sync. Exact zeros
  are **accepted and clamped** to the dtype's smallest normal before
  `log()`: legitimately small states underflow to 0.0 in fp16/bf16
  (fp16 floor is ~6.1e-5; decayed states routinely sit below it), and
  rejecting them would fail valid carries. This positivity check is a
  property of the log-space `MinGRU` only — `SignedMinGRU` and
  `RotationMinGRU` accept any real `h_0`, no clamp, no check.
- `RotationMinGRU.h_0` is an intrinsic learned parameter, not an optional
  flag: `h_0 = 0` has no orbit under the group action (a fixed point
  cannot demonstrate state tracking), and a state vector on a reflection
  axis collapses reflections onto rotations. A random nonzero learned
  vector avoids both failure modes.

## Implementation notes

- **`log_g` gradient at 0.** The paper's reference `log_g` guards the
  `x ≥ 0` branch with `relu` (necessary: `torch.where` evaluates both
  branches, and an unguarded `(x + 0.5).log()` NaNs for `x < −0.5`,
  poisoning gradients even when unselected). But `relu'(0) = 0` gives
  `log_g` gradient exactly 0 at `x = 0` — outside the true
  subdifferential `[0.5, 2]` — which deadens any zero-initialized
  parameter fed through it (e.g. `learnable_h0` at its default init:
  no crash, no wrong output, the parameter just never trains). This
  implementation uses a nested `where` instead: value-identical to
  the relu guard, but the selected branch at 0 is plain `x`, giving
  the correct gradient `1/g(0) = 2` while keeping the unselected
  branch finite.
- `step()` runs under `@torch.no_grad()` for every mixer. Training is
  intended through the parallel `forward`; if you need BPTT through the
  sequential path, remove the decorator.
- Parallel/sequential agreement is ~1e-5 in fp32 at T=128 for `MinGRU`
  (`logcumsumexp` accumulation), ~1e-7 for `SignedMinGRU` (no exp/log
  round-trip), and ~1e-6 for `RotationMinGRU` — none exact.

## Caveats for sequence modeling

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

`coupled=True` is a bit-exact reproduction of the pre-promotion class
(identical parameter shapes and construction order, so identical seeds
give identical weights) — kept as the one-flag legacy path.

**Why the default changed.** The coupled form imposes a ceiling
`|a_t| ≤ 1 − z_t`: reaching the eigenvalue −1 that a task like parity
needs asks the gate to *also* saturate (`z_t → 0`) — one target value,
two simultaneous saturations to pay for. In practice this shows up as
length-generalization decay: the coupled form solves parity at the
training length but its accuracy erodes with distance as the
imperfect eigenvalue compounds over more steps (current-env, 3-seed
mean: 0.894 @256 → 0.610 @1024; see Results). The decoupled form
removes the ceiling entirely: `tanh`'s own asymptote is the eigenvalue's
attractor, so it needs only one saturation and reaches the target
"for free," holding much closer to exact out to 4x-16x the training
length (0.996 mean, worst seed 0.979, n=6; see Results). That is why
`coupled=False` is now the default — the previous parameterization
remains exactly reproducible behind `coupled=True`.

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
positivity check, no underflow clamp), parallel/sequential agreement is
~1e−7 rather than ~1e−5 (no exp/log round-trip), and there are 3 linear
heads instead of 2 (mind parameter-matched comparisons). The scan is
O(T log T) work / O(log T) depth in pure torch ops.

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
multiple of `2π/K` via a straight-through estimator: forward uses the
snapped angle, gradient passes through the pre-snap "soft" angle
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

**Seed-rate, stated plainly.** Across 8 fresh current-env seeds under
this protocol, only 2 of 8 land the exact solution (accuracy 1.0 to
the checked precision at every length out to 1024); the rest are
detectably-flagged (best val@128 < 1.0) and decay measurably at
4x-16x the training length. The mean length-generalization numbers in
Results (0.987 @256, 0.956 @512, 0.889 @1024) are the honest average
over all 8 seeds, including the flagged ones — not a best-seed
headline. Per `experiments/SUMMARY.md`'s mechanism verification, every
seed (including the flagged ones) contains a D3 representation
readable off its weights; failed seeds are simply 5-15x less exact,
not missing the mechanism. Budget for retries when reproducing this
variant.

Excludes refuted experiment-loop mechanisms: no full orthogonality
constraint, no grid-attraction regularizer, no post-hoc
projection/ablation masks. All were tried and either hurt length
generalization or were redundant with best-val selection above (see
`experiments/SUMMARY.md`, rounds 5 and 8, and the closed repair agenda).

Practical differences from the other mixers: 4 linear heads (z, h,
theta, u) vs. `SignedMinGRU`'s 3 / `MinGRU`'s 2 (mind parameter-matched
comparisons); `h_0` is an intrinsic learned parameter with no
`learnable_h0` flag (see State conventions); `hidden_size` must be
even (`ValueError` otherwise); parallel/sequential agreement is
~1e-6. The scan is O(T log T) work / O(log T) depth in pure torch ops,
same tradeoff as `linear_scan`.

## Expressivity probes

`probes.py` tests the ladder empirically on two word problems (seq2seq
tagging with dense supervision, following Merrill et al.'s setup):

**parity**: label each prefix with the running XOR of the bits so far.
The natural one-channel recurrent solution is a sign flip: hold
`h_t = −h_{t−1}` on input 1, `h_t = +h_{t−1}` on input 0, and read the
answer off the sign. That is a transition coefficient of −1, exactly
the eigenvalue `SignedMinGRU` adds and the positive-diagonal `MinGRU`
(a ∈ (0,1)) cannot represent at any width. Parity is in TC⁰, so a
failure here is a parameterization limit, not a complexity-class one.
Chance is 0.5.

**S3**: each token is one of the 6 permutations of three objects
(picture three cups being swapped and rotated); the label at each step
is the net permutation so far. Composition order matters:
swap-then-rotate ≠ rotate-then-swap. A diagonal scan's per-channel
state is a running product of scalars, and scalar multiplication
commutes, so the mechanism is order-blind: diagonal variants of any
sign fail per layer. `RotationMinGRU`'s non-commutative 2x2 blocks
close this gap (see Results). Chance is 1/6 ≈ 0.17.

Models train at T=64 and are evaluated at T=64 (in-distribution) and
longer lengths (256/512/1024, length generalization) — the length-gen
columns are what separate "expresses the recurrent solution" from
"learned a depth-bounded shortcut for the training length."

### Results

All numbers below are **current-environment** (torch 2.5.1, CPU)
**multi-seed means**, drawn from `experiments/lab_results.jsonl` as
tabulated in `experiments/SUMMARY.md`; seed counts are stated per row.
Protocol: seq2seq tagging (dense supervision), T_train=64, d_model=64,
batch 128, Adam lr 3e-3, budget ≤1600 steps, early-stop at 99.9%
train-length accuracy — **except** `minGRU-rotsnap`, which uses the
best-val@128 protocol described above instead of early-stop. The @512
column is not separately tabulated in `experiments/SUMMARY.md`; it is
computed directly from `lab_results.jsonl` using the identical seed
sets as the corresponding @256/@1024 cells.

| task | model | layers | seeds | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|---|---|
| parity | `GRU` | 1 | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| parity | `minGRU-signed` (`coupled=True`) | 1 | 3 | 1.000 | 0.894 | 0.719 | 0.610 |
| parity | `minGRU-signed-tanh` (default) | 1 | 6 | 1.000 | ≥0.9999 | 0.999 | 0.996 (worst seed 0.979) |
| S3 | `GRU` | 1 | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| S3 | `minGRU-signed` (`coupled=True`) | 1 | 3 | 0.414 | 0.339 | 0.275 | 0.223 |
| S3 | `minGRU-signed` (`coupled=True`) | 4 | 3 | 0.885 | 0.544 | 0.426 | 0.342 |
| S3 | `minGRU-rotsnap` (best-val@128 protocol) | 1 | 8 | 0.999 | 0.987 | 0.956 | 0.889 (exact-to-16x in 2/8 seeds) |

Positive-diagonal `minGRU` (log-space) remains at chance on both tasks
regardless of depth — it cannot represent the sign-flip (parity) or
non-commutative (S3) mechanism at any width, a parameterization
failure rather than a training one. It is not re-tabulated
multi-seed here (the experiment loop scoped its runs to the signed and
rotation variants, not the log-space baseline) since a chance-level
result doesn't shift with seed or environment.

**Footnote on prior numbers.** Earlier (pre-promotion) versions of this
README quoted single-seed numbers that carried seed and environment
variance and are **not comparable** to the table above: the old
single-seed `minGRU-signed` L=4 S3 @256 figure was 0.655; the current
3-seed current-env mean is 0.544 — within seed variance, not an
environment regression (`experiments/EXPERIMENTS.md`, "Baselines
re-grounded"). Any comparative claim in this README is backed only by
the current-env multi-seed table above.

Interpretation:

- **Parity separates the sign; decoupling removes the residual decay.**
  Coupled `minGRU-signed` solves parity at the training length but
  decays with distance (0.894 → 0.610 mean, @256→@1024, n=3) because
  reaching `a = −1` costs two simultaneous saturations. Decoupling
  removes that ceiling: `minGRU-signed-tanh` holds ≥0.996 mean out to
  1024 (n=6, worst seed 0.979) at the same L=1, no depth needed.
- **S3 separates commutativity; only a non-diagonal scan closes it.**
  Both coupled parameterizations plateau near a Z2 (even/odd) quotient
  at L=1 (0.414 → 0.223 mean across lengths) and depth substitutes for
  recurrence at L=4 (0.885 @64 falling to 0.342 @1024) — a
  length-bounded shortcut, not the automaton, the same pattern Merrill
  et al. report for Mamba/S4/transformers. `minGRU-rotsnap`'s
  non-commutative 2x2 blocks close most of this gap in a single layer
  (0.987 @256, 0.889 @1024, n=8), and per the mechanism verification in
  `experiments/SUMMARY.md`, the winning seeds *do* contain an
  inspectable, near-exact D3 representation (composition error ~1e-4)
  — not a shortcut.
- **Rotation-snap's residual gap is seed-rate, not capability.** Only
  2 of 8 fresh seeds land the exact solution to 16x length; the rest
  find the same representation 5-15x less exact (readable off the
  weights) and decay measurably. Best-val@128 selection + retry-on-flag
  is the shipped mitigation, not a cure — budget for retries when
  reproducing this variant.
- **GRU remains the ceiling.** State-dependent gating solves both tasks
  exactly at every tested length (1.000, n=3) with a single layer — the
  automaton construction the diagonal and rotation mixers approximate.

Caveats that still apply: one learning rate, and null/partial results
are budget-relative ("didn't land the exact solution in 1600 steps" ≠
"cannot"). The minGRU wrappers include a block MLP the GRU baseline
lacks, which favors the minGRU variants — strengthening their negative
results, mildly weakening attribution of their positive ones.

### Reproducing

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
current-env *multi-seed* numbers in Results above come from
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
qualitative pattern; see `experiments/SUMMARY.md` for the fully
reproduced current-env evidence trail (loop rounds, mechanism
verification, closed repair agenda).

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
