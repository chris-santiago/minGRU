# minGRU

PyTorch implementation of the minGRU from Feng, Tung, Ahmed, Bengio &
Hajimirsadeghi, *Were RNNs All We Needed?* (arXiv:2410.01201) — a recurrent
layer whose gates depend only on the current input, so the whole sequence
trains in one parallel scan instead of step-by-step backpropagation through
time (BPTT). The library is the `mingru` package (pip name `mingru-scans`); pure PyTorch, no dependencies beyond `torch`.

This repo ships the base minGRU plus four variants that each fix a specific
gap in it: **`SignedMinGRU`** (state can flip sign, not just decay toward
zero), **`RotationMinGRU`** (state can track operations that don't
commute, like composing permutations), **`GivensMinGRU`** (richer
non-commuting state via rotations across 8-dimensional blocks instead of
2D planes), and **`DeltaMinGRU`** (a per-head matrix associative memory
updated by the delta rule — the most reliably trainable composer measured,
and the recommended default when per-token state is free to grow). All
five share one `mixer=` interface on `MinGRUBlock`/`MinGRUStack`, and all
five optionally accept a time-aware decay term for
irregularly-spaced sequences — a real-world gap between events, not just
a token count (see "Time-aware decay," below). What each variant actually buys you, in measured
accuracy, is below.

## Install

```bash
pip install mingru-scans
```

Requires `torch>=2.8` (CPU or CUDA); the import name is `mingru`:

```python
import torch
from mingru import MinGRUStack

stack = MinGRUStack(input_size=32, d_model=256, n_layers=4)  # mixer="log" (default)
x = torch.randn(4, 10, 32)
y, state = stack(x)   # (B, T, 32) -> (B, T, 256), per-block states
```

The optional Triton scan backend is imported lazily on first use of a Triton-backed symbol, so `import mingru` works on CPU-only installs; the `MINGRU_SCAN` control and GPU path are documented on the docs site. On Linux the default PyPI torch bundles the matching Triton automatically; if your CUDA-capable torch came without it (`mingru.available()` reports this), `pip install "mingru-scans[triton]"` is an unpinned fallback. Full documentation, including the Givens & Delta composer deep dive and the Triton-scans deep dive, lives at <https://chris-santiago.github.io/minGRU/>.

## GPU acceleration (fused Triton kernels)

An optional Triton backend supplies fused forward+backward GPU kernels for all four scan primitives (`linear_scan`, `matrix_scan`, `matrix_affine_scan`, `parallel_scan_log`) plus an angle-fused rotation path for `GivensMinGRU`. It sits behind a zero-config dispatch seam set by `MINGRU_SCAN`: `auto` (default: use Triton when a CUDA GPU and a working Triton install are present, else the pure-PyTorch eager path), `eager`, or `triton` (force it). CPU-only installs never import Triton.

The win is concentrated in the matrix/block ops. Measured on an NVIDIA L4 (torch 2.8.0+cu128, Triton 3.4.0), forward+backward the Triton kernels run 39x–168x faster than eager on `matrix_scan`/`matrix_affine_scan`, and the angle-fused `GivensMinGRU` backward cuts peak memory from 395 MB to 38 MB. The log-space `parallel_scan_log` is the exception. Its recompute-in-backward kernel is 0.70x–0.80x *slower* than eager forward+backward, so eager stays the right default for the baseline `MinGRU`/`SignedMinGRU` mixers. All 590 CPU-vs-GPU parity checks pass. `DeltaMinGRU`'s chunked-WY `forward` is not one of these scan ops: it has its own Triton kernel (forward trio + torch-composed backward, gated separately), but `torch.compile` — not this kernel — is the recommended CUDA path for `mixer="delta"`. A GPU probe found `torch.compile` recovers 68–89% of the available fusion headroom, while the kernel runs 1.85x–3.79x slower than compile on every probed shape; under `MINGRU_SCAN=auto` the kernel therefore engages only in a narrow measured win region (long sequences, narrow head dims), where it beats eager and saves ~3x memory, and stays eager everywhere else (see "Delta variant," below). Kernel design and the full benchmark tables are in the [Triton scan kernels explanation](https://chris-santiago.github.io/minGRU/explanation/triton-scans/) and `experiments/bench/` (`scan_bench.md`, `scan_memory.md`, `scan_parity.md`).

## Benchmark validation

The mixer family is validated on four accepted public benchmarks — S5 symmetric-group word problems, MQAR (multi-query associative recall), psMNIST (permuted-pixel MNIST), and an irregular-timestep pendulum-regression control — with a depth-matched classical `nn.GRU` control anchoring every comparison. This is a validation round, not a leaderboard entry. Cells are the fit count at each task's fixed bar (S5/MQAR/pendulum $n=36$, psMNIST $n=12$); on the generator tasks a "fit" is trainability at the bar, not length generalization. All numbers are the **L4 stratum** (NVIDIA L4, torch 2.8.0+cu128, Triton 3.4.0, `MINGRU_SCAN=triton`), never mixed with the CPU (torch 2.5.1) rows elsewhere in this README.

| arm | S5 (`val128` $\ge 0.99$) | MQAR (`val_qacc` $\ge 0.99$) | psMNIST (`val_acc` $\ge 0.90$) | pendulum (`val_mse` $\le 0.0014$) |
|---|---|---|---|---|
| log | 0/36 | 0/36 | 0/12 | 36/36 |
| signed | 0/36 | 0/36 | 0/12 | 36/36 |
| rotation | 0/36 | 0/36 | 0/12 | 36/36 |
| rotation-hetero | 0/36 | 0/36 | 0/12 | 36/36 |
| givens | 0/36 | 0/36 | 0/12 | 36/36 |
| delta | 0/36 | 36/36 | 10/12 | 36/36 |
| signed-givens | 1/36 | 0/36 | 0/12 | 36/36 |
| signed-delta | 0/36 | 36/36 | 12/12 | 36/36 |
| gru | 0/36 | 0/36 | 3/12 | 36/36 |

The read is a two-dial mechanism story: delta is the broadly dominant workhorse (associative recall, accumulation ordering, and — at Householder count $nh=4$ in the S5 probe — group composition), while givens is a narrow group-composition specialist (its only matched S5 fit is `signed-givens`, $1/36$). Pendulum is a positive control (all nine arms fit, so the decay channel is not discriminating there). Full per-task tables (raw and fit-only generalization, Fisher contrasts, the S5 $nh$ probe, and the hidden-256 gru-large grounding reference) are on the docs site's [Benchmark validation](https://chris-santiago.github.io/minGRU/explanation/benchmark-validation/) page, with the round-by-round ledger in `experiments/EXPERIMENTS.md`.

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
| `GivensMinGRU` (`mixer="givens"`) | `minGRU-hetero-sg8` (the `mixer=["signed", "givens"]` composer stack; no standalone single-mixer entry in `probes.py`'s `MIXER_REGISTRY`) | measured on `S3-hier`, not the plain `S3`/parity columns below — see "Givens variant" |
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
decimals) at every tested length with a single layer. For deeper stacks
that must both extract a non-commutative operation from raw input and
then compose it (a harder joint problem than tracking a single operation
directly), swapping the composer from a 2D rotation block to
`GivensMinGRU` (`mixer="givens"`) raises fit reliability on the `S3-hier`
task from 1 of 12 seeds to 8 of 12 (Fisher exact p ≈ 0.009), at a matched
64-element per-token state and within +2.2% of full-stack parameters —
though, like the other continuous composers here, it buys no attractor at
length and still decays by T=1024. See "The hierarchical task: S3-hier,"
below, for the task's construction and the full cross-mechanism evidence
table, and "Givens variant" for the mechanism and the measured decay,
and "Delta variant" for the fuller two-axis picture against the
delta-rule composer (`DeltaMinGRU`, `mixer="delta"`). That picture has
since resolved into a recommendation conditioned on two axes rather
than a single winner: **state free to grow → `mixer="delta"` at
native state is the promoted default** (the most reliable trained
composer measured, on both CPU and GPU, with cost and memory that
stay flat as state grows — see "Measured CPU cost" under "Delta
variant"), while **state that must stay small (a matched 64-element
budget) → `GivensMinGRU` remains decisively more reliable**, and its
small-state fit cohort also extrapolates further at extreme lengths
than the full-state delta cohort does. Full evidence for both
branches is in "Givens variant" and "Delta variant," below, and at
the docs site's
[Choose a mixer](https://chris-santiago.github.io/minGRU/how-to/choose-a-mixer/#the-two-axis-guidance)
guide.

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
| `GivensMinGRU` | k-dim block-rotation transitions (non-diagonal, `matrix_affine_scan`); each transition is a product of `rounds` brick-wall Givens layers per `block_size`-dim block, special-orthogonal and continuous (no snap) — richer per-token maps than `RotationMinGRU`'s 2x2 at the same per-token state (see "Givens variant"); same API; same time-decay kwargs, scaling the whole block by a scalar (rotation unaffected) |
| `DeltaMinGRU` | DeltaNet/DeltaProduct delta-rule mixer (`mixer="delta"`): per-head `d_k×d_v` associative-memory matrix state updated by `nh` generalized-Householder rank-1 corrections per token (kwargs `n_heads`, `nh`, `d_k`, `d_v`); parallel chunked-WY `forward` (`chunk_size` is a performance-only knob — results invariant), sequential `step` returns `(y_t, h_t)` since readout ≠ state (`carries_matrix_state=True`); state flattened `(B, n_heads·d_k·d_v)`, zero-init (no learned `h_0`); time-decay supported (Gated-DeltaNet-style per-token per-head scalar gate; the chunked-WY parallel `forward` is preserved via a decay-ratio change of variables, so `decay=None` stays bit-identical; decay-active forward is eager-only, `torch.compile` for CUDA) — see "Measured CPU cost" under "Delta variant" for the measured step cost |
| `linear_scan` | Hillis–Steele associative scan for `h_t = a_t·h_{t−1} + b_t` with signed scalar coefficients |
| `matrix_scan` | Hillis–Steele associative scan for `h_t = M_t @ h_{t−1} + b_t` with 2x2 matrix coefficients (non-commutative composition) |
| `matrix_affine_scan` | Hillis–Steele associative scan for `h_t = M_t @ h_{t−1} + b_t` with k×k matrix coefficients (the `block_size` generalization of `matrix_scan`, used by `GivensMinGRU`) |
| `MinGRUBlock` | pre-norm residual block: LN → mixer → +x, LN → MLP → +x (`mixer` selects `"log"`, `"signed"`, `"rotation"`, `"givens"`, or `"delta"`; `mixer_kwargs` forwarded to its constructor); `forward`/`step` take an optional `delta_t`, forwarded to the block's mixer unchanged |
| `MinGRUStack` | input projection → N blocks → final LN; full state threading; `mixer=` is a `str` (applied to every block, unchanged) or a `list[str]` of length `n_layers` mixing types per block (e.g. one rotation block inside a signed/log stack); `mixer_kwargs` is the current flat dict for a `str` mixer, or `None`/a dict keyed by mixer type for a `list` mixer, applied to every block of that type (two blocks of the same type share one config); more than one `"rotation"` entry warns once per construction (STE-compounding, doesn't block construction) and proceeds, while multiple `"givens"` entries do **not** warn (the continuous Givens transition has no straight-through snap to compound); `decay_layers="all"` (default) or `"last"` selects which blocks receive the resolved kwargs' decay keys, positionally, whatever each block's type; `forward`/`step` take an optional `delta_t`, routed only to blocks whose mixer has decay enabled |

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

givens_stack = MinGRUStack(32, 256, n_layers=2, mixer="givens")
# k-dim block-rotation transitions (default block_size=8, rounds=3);
# richer non-commutative per-token maps than "rotation", continuous (no
# snap); d_model must be a multiple of block_size -- see "Givens variant"
```

Mixing mixer types across layers (`mixer=` as a `list[str]`, one name per
block; `mixer_kwargs` keyed by type instead of flat):

```python
import torch
from mingru import MinGRUStack

hetero_stack = MinGRUStack(
    32, 256, n_layers=2, mixer=["signed", "rotation"],
    mixer_kwargs={"signed": {"coupled": True}, "rotation": {"snap": (2, 3, 6)}},
)
x = torch.randn(4, 10, 32)
y, state = hetero_stack(x)            # (B, T, 32) -> (B, T, 256), per-block states
# one rotation block inside a deeper signed stack (extract-then-compose);
# mixer_kwargs is keyed by mixer type, not applied flat, when mixer is a
# list -- see "The hierarchical task: S3-hier" for the measured depth/hierarchy tradeoffs.
# A mixer list with more than one "rotation" entry (e.g. ["rotation",
# "rotation"]) warns once per construction (STE-compounding) and proceeds.
```

```python
hetero_stack_givens = MinGRUStack(
    32, 256, n_layers=2, mixer=["signed", "givens"],
)
x = torch.randn(4, 10, 32)
y, state = hetero_stack_givens(x)     # (B, T, 32) -> (B, T, 256), per-block states
# signed extractor -> Givens composer (block_size=8, rounds=3 defaults);
# no multi-rotation-style warning on construction -- see "Givens variant"
# for the measured fit-reliability and length-generalization tradeoffs
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
`"signed"`; `RotationMinGRU` and `GivensMinGRU` each own their `h_0`
unconditionally (see "Rotation variant" and "Givens variant" below) and do
not accept the flag.

## State conventions

All exposed state is **real hidden state** — an output of `forward()` or
`step()` — under one convention across every mixer:

- `forward(h_0=...)` takes `(B, 1, d_h)`; `step(h_prev=...)` takes
  `(B, d_h)`. Crossing between streaming and chunked modes needs an
  explicit `unsqueeze` (intentional — no silent dimension coercion). Same
  shapes for `MinGRU`, `SignedMinGRU`, `RotationMinGRU`, and `GivensMinGRU`.
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
  legitimately small states underflow to 0.0 in fp16/bf16. `SignedMinGRU`,
  `RotationMinGRU`, and `GivensMinGRU` accept any real `h_0` — no clamp,
  no check; the positivity constraint is a property of the log-space
  parameterization only.
- `RotationMinGRU.h_0` and `GivensMinGRU.h_0` are each an intrinsic
  learned parameter, not an optional flag (see "Rotation variant" and
  "Givens variant," below): a zero state has no orbit under the group
  action, so it can't demonstrate tracking, and a state on a reflection
  axis collapses reflections onto rotations.

The full derivation and code path for each rule above is in the
corresponding docstrings in the `mingru` package (`forward`, `step`, `log_g`).

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
  `SignedMinGRU` (no exp/log round-trip), ~1e-6 for `RotationMinGRU`, and
  ~1e-5 for `GivensMinGRU` (a product of `rounds` k×k blocks per step, so
  more matrix accumulation than the single 2x2) — none exact by
  construction.

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
(`GivensMinGRU` occupies this same rung with a richer per-token map —
k-dimensional block rotations instead of a single 2D plane — not a fifth
rung; see "Givens variant," below.)

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
  put all five mixers in the same broad class as Mamba/S6 and GLA:
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
  mechanism. Non-diagonal transitions (`RotationMinGRU`, `GivensMinGRU`)
  lift the *commutativity* restriction and provably solve bounded
  non-abelian tracking (S3, a solvable group) that no diagonal variant
  can at any width — but that is not the same as the unbounded, non-solvable-group
  tracking (e.g. S5, NC¹-complete) that requires state-dependent gates;
  see the Signed and Rotation variant sections below. The probes make
  all of this measurable.
- **A natural objection: "rotations are abelian."** Rotations confined
  to fixed disjoint 2×2 planes with pure rotation angles are equivalent
  to a complex-diagonal transition (the LRU parameterization); they
  commute — angles just add — so that family is order-blind exactly like
  the real diagonal case, and the objection is correct *for that
  family*. Neither promoted rotation mixer is in it, and the recorded
  measurements separate them. `RotationMinGRU`'s per-block map is
  `R(θ_t)·diag(1, tanh u_t)` — rotation **and** flip, an O(2)-valued
  family generating the dihedral group, which is non-abelian — and it
  fits S3 at one layer (1.000 / 1.000 / 0.996 / 0.958, n=8, "Rotation
  variant" below), a result unavailable to any commuting transition
  family at any width: commuting maps yield the same final state under
  any reordering of the inputs, while S3 labels change under
  reordering. `GivensMinGRU` composes rotations in *staggered* planes —
  disjoint planes within a round commute; the stagger across rounds is
  what breaks commutativity — at fixed block size, so per-token cost
  stays linear in width rather than the O(d²) of a full SO(d)
  transition. Two adjacent corrections the objection often carries:
  parity is Z₂ — abelian — and is held at length by the signed
  *diagonal* mixer (0.994 @1024, n=6, worst seed 0.984), which needs an
  eigenvalue at −1, not non-commutativity (the delta-rule
  reimplementations record 0.851 and 0.810 @1024 on the same probe, n=3
  each — "Incumbent comparison" below); and S5 is not what non-diagonal
  transitions unlock — it is NC¹-complete, a ceiling on this entire
  fixed-depth class (previous bullet), while non-diagonal transitions
  unlock the *solvable* groups inside TC⁰.

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
transitions (`RotationMinGRU` or `GivensMinGRU`, below) or state-dependent
gates (i.e., a standard GRU). All diagonal variants, signed or not, remain
in TC⁰ per Merrill et al.'s iterated-scalar-product argument.

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

Depth-2 rotation stacks on the harder extract-then-compose task — and
the cross-mechanism comparison they sit in — are covered in "The
hierarchical task: S3-hier," below.

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
theta, u) vs. `SignedMinGRU`'s 3 / `GivensMinGRU`'s 3 / `MinGRU`'s 2
(mind parameter-matched comparisons); `h_0` is an intrinsic learned
parameter with no `learnable_h0` flag (see "State conventions");
`hidden_size` must be
even (`ValueError` otherwise). The scan is O(T log T) work / O(log T)
depth in pure torch ops, same tradeoff as `linear_scan`. Numerical
agreement vs. the sequential path is covered in "Implementation notes,"
above.

## The hierarchical task: S3-hier (extract, then compose)

*(This section is the evidence base for the promoted `GivensMinGRU`
mixer among others; "Givens variant," below, gives that mixer's own
summary, mechanism, and costs without repeating this table. Protocol
terms — best-val@128 selection, the retry flag — are defined under
"Rotation variant," above.)*

**The task.** `S3-hier` (chance ≈ 1/6) is the harder of the repo's two
S3 probes: the group operation is hidden inside a *pair* of sub-tokens.
Sub-tokens are drawn uniformly from `{0..5}`; each consecutive pair
`(x[2k], x[2k+1])` selects a generator `g = LATIN[a, b]`, composed onto
the running S3 product when the pair completes. Labels are dense: odd
positions carry the just-updated composition, even (mid-pair) positions
carry the previous one.

`LATIN` is a fixed 6x6 Latin square (every symbol exactly once per row
and column), so a single sub-token carries no information about the
generator: the distribution of `g` given `a` alone — or `b` alone — is
uniform, and no per-token shortcut exists. It is additionally verified
non-isotopic to both groups of order six (Z6 and S3): no relabeling of
rows, columns, and symbols turns it into either group's Cayley table,
so a rotation layer cannot absorb the lookup by relabeling its angle
assignment the way it can for a plain group operation. (`VERIFY_LATIN=1`
re-runs the import-time sanity check; `probes.py` documents the
exhaustive offline isotopy verification.) Extraction is therefore
genuine work that must happen *before* composition — the property the
heterogeneous stacks below are built around.

A worked sample, verbatim from `probes.make_s3_hier` (seed-0 training
stream, batch row 0, first six steps):

```
x      1  5   0  2   1  1  ...
pairs  (1,5)  (0,2)  (1,1)
g        0      5      2       g = LATIN[a, b]
y      0  0   0  5   5  1  ...
```

The label updates only when a pair completes and carries mid-pair; the
first drawn pair happens to map to the identity, so the label holds at
0 through step 4.

**Depth vs. hierarchy: a tradeoff, not a ranking.** Measured at
`MAX_STEPS=1600` unless noted (seeds per row as listed; every number
budget-relative), multi-seed means. Rotation- and delta-bearing rows
use best-val@128 checkpoint selection (val@384 where noted);
signed-tanh and `GRU` rows use the early-stop protocol they are
recorded under. The n=6 `minGRU-hetero-sr` row pools rounds
`hetero-legB-v2` (seeds 0-2) and `hetero-loop-03-base-ext` (seeds
3-5), same protocol:

| config (task `S3-hier`, chance ≈ 0.167) | seeds | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|
| signed → givens8 (`hetero-sg8`, `experiments/hetero_lab.py`) | 12 | 0.949 | 0.885 | 0.787 | 0.613 |
| signed → deltaproduct2 (`hetero-sd2`, `experiments/hetero_lab.py`) | 6 | 1.000 | 0.987 | 0.814 | 0.530 |
| signed → 64-state delta (`hetero-sdm`) | 12 | 0.575 | 0.495 | 0.457 | 0.376 |
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
1/12 seeds continuous (1/6 snapped); 8-dimensional rotation blocks
(`givens8`: per-token transitions built from three rounds of Givens
rotations, special-orthogonal by construction, continuous, trained
through the same parallel associative scan) fit 8/12 (Fisher exact
p ≈ 0.009 against the 2D rate), with three of the four misses
near-fit (best-val@128 0.83–0.98) and one low seed.
A rounds ablation at fixed block size attributes that gap to the coupling that breaks within-block commutativity, not to block size: with `rounds=1` (four disjoint, commuting 2D planes inside the 8D block) the composer fits 0/12 — no better than the 2D composer — while the single staggered coupling layer of `rounds=2` recovers 6/12 and `rounds=3` reaches the 8/12 above (0/12 vs 8/12 p ≈ 0.0013; rounds 2 vs 3 inseparable, p ≈ 0.68; fit-only generalization unchanged across rounds — per-seed rows under `hetero-loop-19-rounds`). The delta rule
shrunk to the same 64-element state fits 4/12 with chance plateaus on
the misses (`hetero-sdm` above) — so the full-size delta composer's
6/6 owes much of its reliability to its 16x-larger state (1,024
elements per token, p ≈ 0.013 for the state-size effect; capacity
disclosure above) — though the Givens-vs-small-delta comparison
itself (8/12 vs 4/12, p ≈ 0.22, unmatched composer parameters: 14,624
vs 3,306) is suggestive rather than established. Fit quality is
mechanism-independent: seeds that fit generalize equally under either
composer (0.733 vs 0.739 @1024 fit-only means) — the mechanisms
differ in how often training finds a solution, not in how well the
found solutions generalize. Among configurations that fit reliably,
the Givens-8 composer's fit-only profile (0.927 @512, 0.733 @1024;
best seed 0.956/0.812) leads the full-size delta composer's
(0.815/0.530 across its 6/6 fits); the homogeneous signed-tanh L=2
row (0.620 @1024) sits close behind under its different (early-stop)
protocol — per-seed rows, parameter counts, and Fisher tests in
`experiments/EXPERIMENTS.md`.

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

## Givens variant (`GivensMinGRU`)

Enriches the non-commutative rung of the ladder without leaving it.
`RotationMinGRU` gives each transition a single 2x2 planar rotation;
`GivensMinGRU` builds each per-token transition from a product of Givens
rotations acting on a `block_size`-dimensional block of state, a strictly
richer family of orthogonal maps at the same per-token state.

**What a Givens rotation is.** A Givens rotation is the smallest possible
rotation: it acts on one coordinate plane of a `k`-dimensional space and
leaves the other `k−2` directions fixed — the identity matrix except for
`cos`/`sin` entries in two rows. It is the atom of orthogonal structure:
exactly orthogonal at every parameter value (no normalization step),
determinant `+1`, one angle per plane. `RotationMinGRU` is already a Givens
machine at `k = 2` — one plane, one input-dependent angle per token.
`GivensMinGRU` composes many of them: at the default `block_size = 8`,
`rounds = 3`, each per-token transition is a product of three brick-wall
layers of Givens rotations (round 0 pairs planes (0,1),(2,3),(4,5),(6,7);
round 1 the staggered planes (1,2),(3,4),(5,6),(7,0); round 2 repeats
round 0's pattern), all angles emitted by one linear head. Disjoint planes
within a round commute; the stagger between rounds is what couples all
eight dimensions. Products of enough Givens rotations reach all of SO(k),
so three rounds are a deliberate budget, not a representational limit.
The brick-wall mesh itself is the standard construction from the
orthogonal/unitary-RNN literature — EUNN (Jing et al., 2017), built on the
rectangular interferometer mesh of Clements et al. (2016), with the Givens
rotation as described in Golub & Van Loan — borrowed deliberately: what is
new here is the input-dependent angles per token and the measured
trainability result, not the factorization. The
result stays continuous, stays exactly special-orthogonal, stays parallel
(the same associative scan, generalized to k×k blocks by
`matrix_affine_scan`), and keeps the standard 64-element per-token state at
the repo's `d_model`.

**Where the richer map earns its place.** The extra structure is a
*trainability* lever, not a jump in expressivity class — an 8D Givens
product tracks the same solvable-group automata as a 2x2 rotation, and
both remain below the state-dependent-gate line. What it changes is how
often training finds a working composer. Within the rotation family, at
matched 64-element per-token state, 8-dimensional Givens blocks fit
`S3-hier` on 8 of 12 seeds against 1 of 12 for continuous 2D rotation
blocks (Fisher exact p ≈ 0.009). The full measured evidence — per-length
means, the delta-family comparison, and the map-richness argument — is
tabulated under "The hierarchical task: S3-hier," above; this section covers the
mechanism and its costs rather than restating that table.

**The trade it does not escape.** Orthogonality buys norm preservation: no
amplitude decay across hundreds of compositions. But continuity still
means no attractor — nothing pins a learned angle to the exact group
element it approximates, so small angle errors compound with length
exactly as an unsnapped `RotationMinGRU`'s do, and accuracy still decays by
`T = 1024`. Among seeds that fit, the givens8 composer holds 0.927 @512
and 0.733 @1024 (best seed 0.956 / 0.812); pooled across all 12 seeds the
`signed → givens8` profile is 0.949 / 0.885 / 0.787 / 0.613 at
@64 / @256 / @512 / @1024 (the top row of "The hierarchical task: S3-hier"
table). Exactness at length remains unique to the snapped 2D composer's
rare exact seed (0.983 @1024) — no continuous composer, Givens included,
reaches it. Snapping the Givens angles is the obvious hybrid and an open
question, not a promise.

**Capacity disclosure.** At matched 64-element per-token state, the
givens8 composer carries 14,624 parameters against the 2D-rotation
composer's 12,544 — the map-richness gradient above costs +2.2% at the
full stack, not a step change in capacity, so the fit-rate separation is
not attributable to parameter count. Against the delta-rule composer
shrunk to the same 64-element state (which fits 4 of 12 seeds), the
comparison is parameter-unmatched (14,624 vs 3,306) and reads as
suggestive only (p ≈ 0.22).

Cost, packaged-mixer trainability, and the two-axis recommendation
that follows from them now live under "Delta variant," below,
alongside `DeltaMinGRU`'s own cost profile and GPU story.

Practical differences from the other mixers: 3 linear heads (theta, z, h)
vs. `RotationMinGRU`'s 4 / `SignedMinGRU`'s 3 (mind parameter-matched
comparisons); `hidden_size` must be a multiple of `block_size` and
`block_size` must be even (`ValueError` otherwise); `h_0` is an intrinsic
learned parameter with no `learnable_h0` flag (see "State conventions");
stacks holding more than one Givens block construct without the
multi-rotation warning, since the continuous transition has no
straight-through snap to compound. The scan is O(T log T) work / O(log T)
depth in pure torch ops. Numerical agreement vs. the sequential path is
covered in "Implementation notes," above.

## Delta variant (`DeltaMinGRU`)

The promoted default when per-token state is free to grow. Where the
rotation family carries a per-token vector through a non-commutative
scan, `DeltaMinGRU` carries a per-head `d_k×d_v` matrix as its state:
a real associative memory, not a rotated vector, updated at each
token by `nh` generalized-Householder rank-1 corrections. That is the
DeltaNet (Yang et al., 2024) / DeltaProduct (Siems et al., NeurIPS
2025) delta rule, implemented here as a packaged mixer rather than a
lab reimplementation (contrast the mechanism-level `DeltaNetMixer`
reimplementation under "Rotation variant," above, used only for the
S3/parity incumbent comparison). The parallel `forward` computes the
update through the chunked-WY transform, the DeltaNet literature's
parallel form of the delta rule (`chunk_size` is a performance-only
knob; results are invariant to it); the sequential `step` returns
`(y_t, h_t)` rather than `h_t` alone, since readout and carried state
differ (`carries_matrix_state=True`). Capacity is set through four
kwargs, `n_heads`, `nh`, `d_k`, `d_v`, flattened to `(B,
n_heads·d_k·d_v)` and zero-initialized (no learned `h_0`); as the
"Measured CPU cost" note below covers, this per-token state is a
capacity knob independent of `d_model`, and cost/memory stay nearly
flat as it grows.

**Measured CPU cost.** The parallel k×k scan (`GivensMinGRU`'s) is not
the cheap path on CPU, and neither is any sequential loop: the cheap
path is `DeltaMinGRU`'s chunked-WY `forward` (`mixer="delta"`), the
DeltaNet literature's parallel form of the delta rule, implemented in
this package. Under the pinned bench (`experiments/bench/delta_paths.md`;
torch 2.5.1, the same machine as every number in this README), one
uncontended forward+backward step at `B = 128, T = 64` (min of three
runs) costs 0.0577s for the chunked-WY delta against 0.1617s for the
sequential delta step-loop and 0.9493s for the parallel-scan givens8
transition; the givens8 arm reproduces the recorded 0.961s within
~1.2%, so these rows sit on the recorded evidence scale. At `T = 1024`
the figures are 1.4541s / 19.9742s / 24.9996s. Sequentiality is not what
makes a delta path cheap (the chunked parallel form beats the sequential
loop 2.80x at T=64 and 13.74x at T=1024) — the chunked-WY transform is
what does. Delta's per-token state is also a capacity knob independent
of `d_model`, and cost/memory stay nearly flat as that state grows: a
mechanism × state scaling probe (`experiments/bench/scaling_frontier.md`,
same pin) measures delta at 0.050s/419 MiB at state 64 rising only to
0.077s/676 MiB at state 4096, while `GivensMinGRU` climbs from
0.984s/887 MiB at state 64 to 70.6s/20.0 GiB at state 4096.

**Packaged-mixer trainability: two rounds close the earlier open
caveat.** An older note here read "no trainability run of the packaged
chunked path is recorded"; two later rounds close that gap for the
shipped `DeltaMinGRU`. A matched-state CPU round (`hetero-loop-20-pd64`,
`hetero-loop-21-pd1024`, torch 2.5.1, 12 seeds/arm, `S3-hier`,
best-val@128) trained the packaged mixer directly (bridge-verified
`state_dict`-compatible with the lab's oracle, forward parity
≤2.7e-6) and found: givens@64 8/12 fits, delta@64-matched 4/12, delta@1024
12/12 — neither matched-state contrast reaches significance at n=12
(Fisher p=0.22 and p=0.09), but delta@1024 fits every seed at 1/16 the
per-step cost and 0.57x the memory of givens@64. A 36-seed GPU campaign
(`hetero-gpu36-sg8`/`pd64`/`pd1024`; torch 2.8.0+cu128, NVIDIA L4, triton
3.4.0 — a separate stratum, never comparable to the CPU rows above)
resolves the matched-state tie at larger sample size: givens@64 25/36 vs
delta@64 10/36 (Fisher p=0.00084), while delta@1024 reaches 35/36 (vs
givens@64, p=0.0030). Per-seed GPU wall time is parity across all three
arms (~15-18s) — Givens' Triton scan path erases the CPU-side 16x cost
gap, so the GPU story is about reliability and state economics, not
speed. Fit-only length generalization at 16x train length stays small-
state-favoring on GPU too: givens 0.679, delta@64-matched 0.699, both
ahead of delta@1024's 0.570.

**The resulting recommendation is conditioned on two axes, not one
winner.** State free to grow → `mixer="delta"` at native state: the most
reliable composer measured on either stratum (12/12 CPU, 35/36 GPU), flat
cost in state size. State that must stay small (a matched 64-element
budget) → `GivensMinGRU`: decisively more reliable at that budget on the
stratum with the power to show it, and its small-state fit cohort
extrapolates further at extreme length than the full-state delta cohort.
Do not read delta@64's low pooled accuracy as "delta is less accurate" —
its trained models generalize like Givens' when they fit (matched-state
fit-only 0.721 delta vs 0.733 givens at T=1024); the low pooled mean is a
fit-rate artifact. Delta@1024 is in fact the most accurate arm at or near
training length on both strata (CPU: 1.000@64/0.988@256; GPU:
0.984@64/0.971@256) and only trails on the long-extrapolation tail. On
GPU, delta's recommended path is `torch.compile`: a fusion-headroom
probe (`experiments/bench/gpu_delta_probe.md`, NVIDIA L4) found compile
recovers 68-89% of available headroom. A hand-written Triton chunked-WY
kernel now exists but does not reach compile-class speed (1.85x-3.79x
slower than compile on every probed shape; a fully fused Triton backward
was 8-12x slower and was reverted; the same relative gap holds on an
A100, a separate stratum), so `MINGRU_SCAN=auto` engages it only in a
narrow measured win region — long sequences, narrow head dims — where it
beats eager (0.61x) and saves ~3x peak memory, and stays eager
otherwise. Remaining caveat: no comparison against the incumbents'
released tuned kernels has been measured. Full mechanism, tables, and Fisher statistics for both
composers are at the docs site's
[Givens & Delta deep dive](https://chris-santiago.github.io/minGRU/explanation/givens-delta/)
and in `experiments/EXPERIMENTS.md` (rounds `hetero-loop-20-pd64`,
`hetero-loop-21-pd1024`, `hetero-gpu36-*`).

`DeltaMinGRU`'s time-decay support — a per-token gate on the carried
matrix state, distinct from the other mixers' per-step scalar decay —
is covered in "Time-aware decay," below.

## Time-aware decay

Every mixer above assumes evenly-spaced steps: one update per token,
no notion of how much real time elapsed between events. `decay=` adds
an optional per-event forgetting term so a gap between events shrinks
whatever the recurrence was already going to keep, without touching
how it injects new information. Available on all five mixers, and threaded through
`MinGRUBlock` / `MinGRUStack`. `DeltaMinGRU` supports it with a
distinct mechanism: it gates its carried matrix state once per token
rather than scaling a per-step transition (see the `DeltaMinGRU`
bullet under "Mechanism" below, and the float32 note further down).

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
channel (per-block, for `RotationMinGRU`/`GivensMinGRU`) decay rate and
`f` is identity or `log1p`. `gamma` multiplies the transition only —
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
- `GivensMinGRU`: `gamma * M_t` per block, the identical semantics to
  `RotationMinGRU` — a scalar `gamma` commutes with the orthogonal block
  action, so the composed rotation is recovered unchanged from the
  decayed matrix; only amplitude fades.
- `DeltaMinGRU` (matrix state, distinct form): rather than scaling a
  per-step transition coefficient, `gamma` gates the carried per-head
  matrix state once at each token boundary — `H <- gamma * H` before
  that token's `nh` undecayed Householder writes — so older memory
  fades while the token's own writes are undamped. The chunked-WY
  parallel `forward` reproduces this exactly via a log-space
  decay-ratio change of variables, so `decay=None` is bit-identical and
  the decay-active path is eager-only. See the float32 note below for
  behavior under very large or `+inf` gaps.

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
range `lambda` needs to operate over. For `DeltaMinGRU` under
float32, very large or `+inf` gaps (sentinel or corrupted
timestamps) are handled correctly automatically: its chunked forward
carries a per-token cumulative log-decay, and an internal float32
clamp on that log-decay keeps output finite, chunk-size invariant,
and matching the sequential `step` oracle even when a gap saturates
(the affected token's memory is fully wiped, exactly as the oracle
wipes it at `gamma = 0`). When a gap is large enough to saturate, the
module emits a once-per-instance warning naming `log1p_delta` (on CPU
inputs only; the correctness clamp always applies, but the warning is
skipped on CUDA to avoid a host sync).
`log1p_delta=True` is then *recommended* — not required for
correctness — if you want very large gaps to stay *distinguishable*
from one another rather than all saturating to a full wipe: it maps
the gap through `log1p` first (`1e10 -> ~23`), keeping distinct large
gaps distinct. The other four mixers apply decay per step with no
chunk-cumulative log and are unaffected. Passing `delta_t` with
decay disabled, or enabling decay without passing `delta_t`, both
raise `ValueError` at call time.

**Two modes**, set via `decay=`:

- `"fixed"`: `lambda = decay_rate`, a scalar buffer, uniform across
  channels/blocks — not learned.
- `"learnable"`: `lambda = softplus(rho)`, one `rho` per hidden
  channel (`MinGRU`/`SignedMinGRU`) or per block
  (`RotationMinGRU`/`GivensMinGRU`); `rho` is initialized so
  `lambda == decay_rate` exactly at
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
from mingru import MinGRUStack
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

**Two torch pins.** The installed package advertises a `torch>=2.8` floor so the optional Triton scan backend is available to library users. The evidence commands here run from a repo checkout, not an install: the root `min_gru.py`/`triton_scans.py` drivers re-export the packaged library from `src/`, so every recorded command reproduces verbatim under the frozen `torch==2.5.1` evidence pin with nothing installed and no Triton import attempted. Pin it explicitly with, for example, `uv run --python 3.12 --with 'torch==2.5.1' python min_gru.py` (or the `probes.py` commands below).

```
python probes.py TASK MODEL [N_LAYERS]
```

- **`TASK`** — one of `parity`, `S3`, `S3-hier`, `session-parity`,
  `parity-timestamped`. `S3-hier` is the harder extract-then-compose
  probe (chance ≈ 0.167; see "The hierarchical task:
  S3-hier"). The last two supply `delta_t` (see "Time-aware decay");
  `GRU` has no `delta_t` input path and raises `ValueError` on them.
- **`MODEL`** — one of `GRU`, `minGRU`, `minGRU-signed`,
  `minGRU-signed-tanh`, `minGRU-signed-tanh-tdecay`,
  `minGRU-signed-tanh-tdecay-mech`, `minGRU-rotsnap`, `minGRU-hetero-sr`,
  `minGRU-hetero-rs`, `minGRU-rotation2`, `minGRU-hetero-sg8`.
  `minGRU-signed` is
  pinned to `coupled=True`: the legacy parameterization, kept under its
  historical name so recorded rows keep their meaning. The two
  `-tdecay` rows are `decay="learnable"` at `decay_rate=0.05`,
  `log1p_delta=True`; `-mech` skips the `delta_t` feature concat (see
  "Time-aware decay"). `minGRU-hetero-sr`/`-rs` are 2-layer
  `MinGRUStack(mixer=["signed", "rotation"])` stacks, one order each
  (signed→rotation, rotation→signed); `minGRU-rotation2` is
  `mixer=["rotation", "rotation"]`, the broken-baseline reference (emits
  one `UserWarning` at construction). `minGRU-hetero-sg8` is the
  `mixer=["signed", "givens"]` stack (signed extractor → Givens composer)
  measured on `S3-hier`; it constructs without a warning. All four fix
  `N_LAYERS` to the mixer list's length (2) — omit it or pass 2
  explicitly; a conflicting value raises `ValueError`. See "Rotation
  variant" and "Givens variant" for their measured accuracy.
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
`coupled=True`, `RotationMinGRU`, `GivensMinGRU`, and their stacks),
chunked-vs-full equivalence for the same set (including carries with
negative/unbounded states), `SignedMinGRU(coupled=True)`
construction-order determinism (bit-exact legacy reproduction),
`matrix_scan`/`matrix_affine_scan` vs. brute-force sequential recurrence
plus a gradcheck, `RotationMinGRU`'s snapped-angle grid exactness and
gradient flow into all four heads and `h_0`, `GivensMinGRU`'s exact
orthogonality/determinant and gradient flow into all three heads and
`h_0`, `RotationMinGRU`'s `ValueError` on odd `hidden_size`,
`GivensMinGRU`'s `ValueError` on indivisible `hidden_size` and odd
`block_size`, `MinGRUBlock`'s `ValueError` on an unknown `mixer` name,
gradient flow through both diagonal scans and into `h0_pre` **from
zero-init** (guards the `log_g` fix), `log_g` gradient at 0 equal to 2,
and `h_0` validation for the log-space variant (underflowed zeros
accepted, negatives raise).

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

Jing, L., Shen, Y., Dubcek, T., Peurifoy, J., Skirlo, S., LeCun, Y.,
Tegmark, M., & Soljačić, M. (2017). *Tunable Efficient Unitary Neural
Networks (EUNN) and their application to RNNs.* ICML 2017.
arXiv:1612.05231.

Clements, W. R., Humphreys, P. C., Metcalf, B. J., Kolthammer, W. S.,
& Walmsley, I. A. (2016). *Optimal design for universal multiport
interferometers.* Optica 3(12).

Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*
(4th ed.). Johns Hopkins University Press. (Givens rotations.)
