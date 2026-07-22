# Length-generalization loop

**Goal:** a new minGRU variant that improves length generalization
(train @64, eval @256/512/1024) while keeping parallel (associative-scan)
training.

**Baselines to beat** (from README, same protocol):

| task | model | L | acc@64 | acc@256 |
|---|---|---|---|---|
| parity | minGRU-signed | 1 | 1.000 | 0.859 |
| parity | minGRU-signed | 4 | 1.000 | 1.000 |
| S3 | minGRU-signed | 4 | 0.993 | 0.655 |
| both | GRU | 1 | 1.000 | 1.000 |

**Diagnosis driving the backlog:** parity decay at length comes from
`|a| = (1−z)·|tanh(s)| < 1` strictly (reaching −1 needs two saturations);
S3@L=4 passes only at training length because diagonal scans commute —
depth substitutes for recurrence.

## Hypothesis backlog

- **H1 decouple** — `a = tanh(s)` (drop the `(1−z)` factor from `a`;
  one saturation to approach −1 instead of two).
- **H2 closed interval** — `a = hardtanh(s)`; −1 exactly reachable.
- **H3 STE sign** — straight-through `sign(tanh(s))` if H2's zero
  gradient at saturation stalls learning.
- **H4 non-abelian rung** — 2×2 block transitions, convex mix of
  rotation and reflection (spectral norm ≤ 1; exact O(2) elements at
  saturation). S3 ≅ D₃ ⊂ O(2): representable in one layer; scan stays
  associative via matrix combine.
- **H5 LRU-style** — input-independent learned eigenvalues, if
  input-dependence itself is the noise source.
- **H6 combine + validate** — merge winners; multi-seed; longer lengths.

## Protocol

Identical to `probes.py` (same seeds, batch 128, d64, Adam 3e-3,
early stop at 99.9% train-length acc, budget 1600) with extra eval
lengths 512 and 1024. Lab code in `variants.py`; results appended to
`lab_results.jsonl`. `min_gru.py`/`probes.py` untouched. Max 10 rounds.

## Round log

### Round 1 — H1 + H2 (parity screen) — DONE

| variant | steps | @64 | @256 | @512 | @1024 |
|---|---|---|---|---|---|
| signed-coupled (calib) | 200 | 1.000 | 0.857 | 0.678 | 0.590 |
| signed-tanh (H1) | 100 | 1.000 | **1.000** | **1.000** | **1.000** |
| signed-hardtanh (H2) | 100 | 1.000 | 0.9996 | 0.797 | 0.649 |

Findings: calibration reproduces README (harness valid). **H1
confirmed** — decoupling `a = tanh(s)` from the update gate gives
perfect parity length gen to 16x training length and halves steps to
solve. H2 is worse than H1: hardtanh's zero-gradient dead zone
(|s| > 1) plausibly freezes eigenvalue calibration; boundary
exactness is not the binding constraint. H3 (STE) moot — skipped.

### Round 2 — H1 on S3 + H4 rotation variant

New variant `rotation`: 2x2 block transitions
`M_t = R(theta_t) @ diag(1, tanh(u_t))` per block pair — spectral
norm 1, covers SO(2) rotations (u -> +1) and O(2) reflections
(u -> -1); at theta = 0 the second channel reduces to signed-tanh.
Scan is Hillis-Steele over 2x2 matrix products (still associative,
O(log T) depth); learnable nonzero h_0 per block (the group acts on
a generic vector; h_0 = 0 has no orbit). S3 ≅ D3 ⊂ O(2), so one
layer can represent the S3 automaton exactly.

Cells: parity/rotation L=1 (sanity), S3 L=1 for {tanh, rotation},
S3 L=4 for {coupled (extended-length calib), tanh, rotation}.

Results — DONE:

| cell | steps | @64 | @256 | @512 | @1024 |
|---|---|---|---|---|---|
| parity rotation L=1 | 100 | 1.000 | 0.873 | 0.737 | 0.630 |
| S3 signed-tanh L=1 | 1100 | 0.9994 | 0.945 | 0.857 | 0.709 |
| S3 rotation L=1 | 100 | 1.000 | 0.9996 | 0.974 | 0.863 |
| S3 signed-coupled L=4 (calib) | 1600 | 0.822 | 0.471 | 0.362 | 0.276 |
| S3 signed-tanh L=4 | 800 | 0.9993 | 0.930 | 0.808 | 0.619 |
| S3 rotation L=4 | 200 | 1.000 | 0.983 | 0.868 | 0.631 |

Findings:

1. **H4 confirmed**: rotation L=1 solves S3 in 100 steps (diagonal
   plateau was 0.372) and holds 0.9996@256 vs baseline L=4 0.655@256.
2. **Attractor insight**: tanh length-generalizes perfectly because
   its asymptote sits AT the needed eigenvalue (-1) — saturation
   self-calibrates. Rotations have no attractor at group angles
   (theta must hit 2*pi/3 exactly, measure-zero), so angle error
   compounds linearly with length: parity/rotation decays (0.873@256)
   where parity/tanh didn't; S3/rotation decays at 1024 (0.863).
3. Signed-tanh nearly solves S3@64 even at L=1 (0.9994) — a strong
   bounded-horizon shortcut (decays with length), echoing the
   "optimization, not expressivity" observation for the coupled
   variants. Depth adds nothing for tanh on S3 (L=4 no better than
   L=1); rotation L=4 is worse at 1024 than rotation L=1 (deeper =
   more shortcut, less clean automaton).
4. **Calibration flag**: today's coupled S3 L=4 (0.822/0.471) does
   not match README (0.993/0.655) though parity L=1 matched. That
   cell trains at the edge; Round 3 re-runs probes.py directly to
   attribute (env variance vs harness bug).

### Round 3 — H10 snap-rotation + calibration re-run

`rotation-snap`: theta snapped to per-block exact group angles
2*pi/K, K cycled over {2, 3, 4, 6}, via straight-through estimator —
gives rotations attractors at group elements (the linear-scan
analogue of the GRU's error-correcting state attractors). Forward
always uses exact group elements => zero angle drift at any length;
gradient flows through the soft angle. Scan unchanged.

Cells: parity/snap L=1, S3/snap L=1 (headline: expect ~1.0 at all
lengths), S3/snap L=4; plus `python probes.py S3 minGRU-signed 4`
for the calibration question.

Results — DONE:

| cell | steps | @64 | @256 | @512 | @1024 |
|---|---|---|---|---|---|
| parity rotation-snap L=1 | 200 | 1.000 | 0.9999 | 0.964 | 0.843 |
| S3 rotation-snap L=1 | 300 | 1.000 | **1.000** | **1.000** | **1.000** |
| S3 rotation-snap L=4 | 1600 | 0.398 | 0.311 | 0.249 | 0.212 |
| probes.py S3 coupled L=4 rerun | 1600 | 0.822 | 0.471 | — | — |

Findings:

1. **H10 confirmed — S3 solved exactly.** Snap-rotation L=1 is
   perfect at every length to 16x training length: the automaton,
   not a shortcut. GRU-equivalent state tracking from a parallel
   scan.
2. **Depth breaks snap**: L=4 fails to train (STE noise through
   stacked quantized layers). Snap is an L=1 mechanism — limitation
   to document.
3. Parity under snap imperfect (0.843@1024) while plain signed-tanh
   was perfect; motivates the K=1-block unification (Round 4).
4. **Calibration resolved**: probes.py today reproduces the harness
   exactly (0.822/0.471) — harness bit-consistent; README's S3-L4
   row carries torch-version variance (edge-of-trainability cell).
   Add a README footnote when promoting results.

### Round 4 — unification + multi-seed validation

`rotation-snap1` = snap grid (1, 2, 3, 4, 6): K=1 blocks are theta=0
always, i.e. pure signed-tanh channel pairs — one variant holding
both winning mechanisms (tanh asymptote for sign/abelian, snapped
rotations for non-abelian). Cells: both tasks with snap1 (seed 0),
then seeds {1, 2} for parity/signed-tanh, S3/rotation-snap, and both
tasks with snap1. Single-seed results near trainability edges are
not trustworthy (see calibration saga).

Results — DONE (seeds 0/1/2):

| cell (L=1) | @256 | @1024 |
|---|---|---|
| parity signed-tanh | 1.0 / 1.0 / 0.9999 | 1.0 / 1.0 / 0.979 |
| S3 rotation-snap | 1.0 / 0.958 / 0.962 | 1.0 / 0.636 / 0.632 |
| parity rotation-snap1 | 1.0 / 0.9998 / 1.0 | 0.999 / 0.676 / 0.938 |
| S3 rotation-snap1 | 0.976 / 0.997 / 1.0 | 0.483 / 0.791 / 0.650 |

Findings:

1. **signed-tanh parity claim survives seeds** (worst 0.979@1024).
2. **Rotation-snap's perfect S3@1024 was a seed-0 draw.** Robust:
   ~0.97@256 (vs 0.372/0.655 baselines); high variance @1024.
3. Unified snap1 dominated by specialists — dropped.
4. **Variance diagnosed**: S3 needs det = -1 reflections =
   R(theta_snapped) @ diag(1, tanh(u)) — theta is exact but tanh(u)
   only approaches -1; tanh(u) = -0.995 decays to ~0.006 by T=1024.
   Half the drift was left unsnapped.

### Round 5 — H13 ortho-snap (exact O(2) transitions)

`ortho-snap` = rotation-snap1 + STE sign on the u channel: every
block transition an exact O(2) element (det +/-1), zero decay at any
length for any learned solution. K=1 blocks become exact +/-1
diagonals (should erase parity's seed-2 wobble too). Trade-off: no
contraction => cannot forget; right for state tracking, wrong
default for general sequence modeling (free-tanh remains the
general-purpose variant). Cells: 3 seeds x {parity, S3}, L=1.

Results — DONE (seeds 0/1/2): **H13 REFUTED.**

| cell (L=1) | @256 | @1024 |
|---|---|---|
| parity ortho-snap | 0.895 / 0.876 / 0.9996 | 0.601 / 0.595 / 0.882 |
| S3 ortho-snap | 0.680 / 0.982 / 1.0 | 0.301 / 0.502 / 0.930 |

Worse than rotation-snap everywhere, higher variance. Missed decay
mechanism: **orthogonal transitions can't forget the injection
stream**. h_t = A_{t:1} h_0 + sum_k A_{t:k+1} b_k — with exact O(2)
transitions every past injection keeps constant norm, so noise grows
~sqrt(t) while signal stays O(1): SNR decays with length even though
group arithmetic is exact. Free tanh(u)'s slight contraction was
doing double duty: (bad) reflection signal decay, (good) injection
noise cleanup. Revised picture: length gen needs exact-enough
transitions AND a forgetting path. Backlog H14 (only if needed):
mixed grid — snap u on K>=2 blocks only, keep K=1 blocks free tanh.

### Round 6 — undertraining test (no early stop)

Early stop fires at 99.9% train acc (steps 100-600), plausibly
before z closes and tanh(u) saturates deep; length gen is known to
keep improving after train acc saturates. Re-run winners with
EXP_NO_EARLYSTOP=1 (full 1600 steps): S3/rotation-snap seeds 0-2,
parity/signed-tanh seed 2 (the 0.979). If @1024 climbs, the
remaining variance is calibration time, not architecture.

Results — DONE: **undertraining refuted; instability diagnosed.**

| cell | @64 | @1024 | vs early-stopped |
|---|---|---|---|
| S3 snap s0 (1600) | 1.000 | 0.753 | was 1.000 at step 300 — worse |
| S3 snap s1 (1600) | 1.000 | 1.000 | was 0.636 at step 100 — better |
| S3 snap s2 (1600) | 0.433 | 0.235 | was 0.9998@64 at 600 — collapsed |
| parity tanh s2 (1600) | 1.000 | 0.989 | 0.979 — mild gain |

The exact-automaton solution is reachable but NOT a stable attractor
of training: runs wander in (s1) and out (s0) of it, and s2's late
collapse to 0.433@64 looks like STE gradient mismatch compounding.
Once train acc = 1.0, dense CE at T=64 has no signal distinguishing
the exact group solution from a decaying shortcut.

### Round 7 — H15 checkpoint selection (val@128)

Track val acc at T=128 during training (not a test length — no
peeking at 256/512/1024), evaluate the best checkpoint instead of
the final state. Standard length-gen protocol; directly counters
wander-out. Cells: both winners x 3 seeds, full budget, EXP_CKPT=1.
Fallback if insufficient: H16 — replace STE with a differentiable
grid-attraction penalty on theta.

Results — DONE (seeds 0/1/2, best ckpt by val@128):

| cell (L=1) | ckpt | @256 | @1024 |
|---|---|---|---|
| S3 rotation-snap | 300/200/1300 | 1.0 / 1.0 / 0.981 | 1.0 / 1.0 / 0.726 |
| parity signed-tanh | 100/100/100 | 1.0 / 1.0 / 0.9999 | 1.0 / 1.0 / 0.979 |

**H15 confirmed.** Selection recovers the exact solution wherever
the run contained one (S3 s0 rescued from Round 6's wander-out; s0
and s1 now exact through 16x training length). S3 s2's best val@128
was 0.9992 — that run never contained the exact solution, so no
selection rule can recover it; residual variance is a training-
dynamics issue, out of scope. Converged; loop closed at 7/10 rounds.

## Final synthesis

**Deliverable: two variants + one protocol, all parallel-scan
compatible.**

1. `signed-tanh` — decouple the eigenvalue from the update gate:
   `a = tanh(Linear_s(x))` instead of `(1-z)*tanh(...)`. One-line
   change vs SignedMinGRU. Parity: 0.857@256 / 0.590@1024 ->
   ~1.0 / ~0.99 (worst seed 0.979). General-purpose (keeps
   contraction => can forget).
2. `rotation-snap` — 2x2 block transitions
   `M = R(theta_snapped) @ diag(1, tanh(u))`, theta snapped to
   exact group angles (2*pi/K, K in {2,3,4,6}) via STE; Hillis-
   Steele scan over 2x2 matrix products. First non-commutative rung:
   S3 goes from 0.372@64 (any diagonal variant, L=1) to exact
   automaton — 1.0 at ALL lengths to 16x in 2/3 seeds; ~0.98@256
   worst seed. L=1 mechanism (depth breaks STE training).
3. Protocol: full budget + best-checkpoint by val@T=128 (held-out
   length, not in the test set). Necessary because exact solutions
   are reachable but not stable attractors of training (Round 6).

**Design laws found (the transferable part):**

- Decouple the transition eigenvalue from the update gate: needing
  two saturations to reach a target eigenvalue is one too many.
- Length generalization requires attractors at the exact transition
  values the task needs — tanh's asymptote at -1 is one; STE
  snapping manufactures them for rotation angles. Parameterizations
  whose target values are measure-zero interior points (plain
  rotation angles) drift, and the error compounds with length.
- But do NOT make everything exact: fully orthogonal transitions
  cannot forget, and the injection stream's noise then accumulates
  (~sqrt(T)) against an O(1) signal. A forgetting path must remain.
- Once train accuracy saturates, the loss cannot see the difference
  between the exact solution and a decaying shortcut — select
  checkpoints on a moderately longer held-out length.

**Honest caveats:** 3 seeds; one task pair; d=64; the S3 result
needs the selection protocol (architecture alone is not stable);
snap variant untested beyond L=1; README S3-L4 baseline row carries
torch-version variance (today's env: 0.822/0.471, not 0.993/0.655).

Refuted en route: H2 hardtanh (dead zone), H3 STE-sign eigenvalues
(moot), unified snap grid (dominated), H13 full orthogonality
(injection-noise accumulation), undertraining (Round 6: training
wanders in AND out of solutions).

*(Loop extended by user request: rounds 8-10 chase the failing
seed.)*

### Round 8 — H16 grid-attraction penalty + sharper selection

Seed 2's run never contained the exact solution (best val@128 =
0.9992) — selection can't fix what training never finds. H16 makes
exact solutions attractors of the LOSS: keep STE snap in the
forward, add `reg * mean((theta_soft - theta_snapped)^2)` — shrinks
the STE forward/backward mismatch suspected in the Round 6 collapse
and anchors found solutions. u stays free (Round 5 lesson).
Selection sharpened to val@192 x 4 batches (still not a test
length; resolves ~0.05% cracks that val@128 x 2 saturated over).

Cells: S3 s2 {no-reg control, reg=0.1, reg=0.01}, S3 s0 reg=0.1
(regression guard), parity s2 (sharper selection alone).

Results — DONE: **H16 refuted; sharper selection null.**

| cell | @256 | @1024 | best val |
|---|---|---|---|
| S3 s2 control | 0.981 | 0.726 | 0.994 (same ckpt as R7) |
| S3 s2 reg=0.1 | 0.914 | 0.481 | 0.971 — worse |
| S3 s2 reg=0.01 | 0.938 | 0.709 | 0.963 — no better |
| S3 s0 reg=0.1 guard | 1.0 | 0.921 | regression vs plain 1.0 |
| parity s2 sharper sel | 0.9999 | 0.979 | unchanged |

The penalty over-constrains the search before the right grid
assignment is found (hurts even the good seed); sharper validation
picks the same checkpoint. Seed 2's failure is an init/trajectory
pathology — not fixable by selection or loss shaping as tried.
(Cosmetic: the jsonl "val128" key holds val at whatever EXP_CKPT_T
was — 192 in this round.)

### Round 9 — success-rate quantification

No more machinery. The honest final claim is a per-seed success
RATE. Plain rotation-snap + standard val@128 selection, S3 seeds
3-7; parity signed-tanh seeds 3-5. Pool with Round 7 (seeds 0-2)
for n=8 / n=6 totals.

Results — DONE. Pooled (Rounds 7 + 9):

**S3 rotation-snap L=1, n=8** (@256 / @1024 per seed):
1.0/1.0, 1.0/1.0, 0.98/0.73, 1.0/0.97, 1.0/0.998, 0.998/0.89,
0.92/0.62, 0.999/0.91. Mean 0.987@256 (median ~1.0), mean
0.889@1024 (median 0.94). Exact to 16x: 2/8; >=0.97@1024: 4/8.

**Parity signed-tanh L=1, n=6**: all >=0.9999@256; @1024 mean
0.996, worst 0.979.

**Failed runs are detectable**: the two weak S3 seeds (s2, s6) are
exactly the ones whose best val@128 never hit 1.0 (0.994, 0.974).
A restart-on-imperfect-validation protocol raises the effective
rate without ever touching test lengths (6/8 runs validate perfect;
among those, @1024 mean 0.961, min 0.886).

### Round 10 — synthesis (final)

Loop closed at 9 experimental rounds. Deliverable and design laws
in the Final synthesis section above; the success-rate numbers in
this round supersede the 3-seed numbers quoted there.

## Post-loop verification (external review recs 2-3)

### Mechanism probes (`mechanism_probes.py`, `mechanism_results.json`)

**Homomorphism test** — extract per-token 2x2 transitions M(g) from
trained rotation-snap models, check M(g)M(h) vs M(g o h) over all 36
pairs against the S3 composition table:

| seed | acc@1024 | best-block hom err | faithful | \|det\| |
|---|---|---|---|---|
| s0 | 1.0 | 0.00014 (3 blocks < 1e-3) | yes | 0.99992 |
| s1 | 1.0 | 0.0004 | yes | 0.99976 |
| s2 | 0.726 | 0.0020 | yes | 0.99870 |
| s6 | 0.616 | 0.0007 | yes | 0.99967 |

Winning seeds learned genuine faithful D3 representations (compose
to ~1e-4): the headline is demonstrated, not inferred. Refinement:
failed seeds have representations too, just 5-15x less exact —
"insufficiently exact solution," not "no solution." Caveats: (a)
automaton-block injection gates are NOT ~0 (s1: z=0.70 with perfect
acc) — the injection-noise account is incomplete; probe measured z,
not ||b||; (b) s6's best block is more exact than s2's despite worse
accuracy — downstream readout structure also matters.

**Eigenvalue test** (parity, signed-tanh): both probed seeds have 3
channels at (a(hold), a(flip)) = (+0.99999, -0.99999) — the automaton
channels, saturated deep enough that a^1024 ~ 0.99. Mechanism
confirmed. Does not explain s2's small @1024 gap (its flip channels
are equally deep; leak is elsewhere).

### Baselines re-grounded (current env, 3 seeds, early-stop protocol)

| row | mean (seeds 0/1/2) | README | verdict |
|---|---|---|---|
| parity coupled L=1 @256 | 0.894 | 0.859 | consistent |
| parity coupled L=1 @1024 | 0.610 | — | new |
| S3 coupled L=1 @64 | 0.41 | 0.372 | consistent |
| S3 coupled L=4 @256 | 0.54 (0.47/0.50/0.66) | 0.655 | **within seed variance** — a lucky seed, not env drift; fair quote is 0.54 |
| GRU L=1, both tasks, @256-1024 | 1.0 all seeds | only @64/256 | ceiling confirmed at claimed lengths |

**Superseded by the `reseed-fix` round below**: this table's seeds
1/2 were drawn under the train/eval generator-seeding collision fixed
in that round (seed 0 is unaffected and unchanged). Read the
`reseed-fix` section for the current numbers; this table is kept as
the historical record of what `verify-baselines` observed under the
pre-fix seeding.

Comparative claims in the synthesis should be read against these
means; README-sourced numbers are struck from comparisons. Corrected
headline comparisons (current-env, re-validated under the `reseed-fix`
seeding fix — see that section below; pre-fix values in parens):

- parity @1024: coupled 0.592 (was 0.610) -> signed-tanh 0.994, n=6
  (was 0.996); GRU 1.0.
- S3 @256: coupled L=4 0.649 (was 0.544) -> rotation-snap 1.000, n=8
  (was 0.987); GRU 1.0.
- S3 @1024: coupled L=4 0.347 (was 0.342) -> rotation-snap 0.958, n=8
  (was 0.889); GRU 1.0.

Not run (scoped out, review rec 1): Grazzi-parameterized incumbent,
DeltaNet. Signed-tanh should be presented as the Grazzi negative-
eigenvalue mechanism instantiated in minGRU until that comparison
runs — a repo improvement, not a novelty claim.

## Repair rounds 1-2 (`repair_probes.py`, `repair_round2.py`)

Tested whether any cheap intervention converts the failed
rotation-snap seeds. All four arms closed; none ships.

**R1a — inference-time O(2) projection** (snap tanh(u) -> +/-1 on
identified near-hom blocks; zero training): s2 0.726 -> 0.807@1024,
s6 0.616 -> 0.645. Partial — most of the residual failure is
readout-side, not transition calibration. Control clean (s0 stays
1.0). Projecting ALL blocks hurts even s0 (1.0 -> 0.673@1024):
Round 5's exactness-must-be-local law reproduced at inference with
zero training.

**R1b — injection probes**: the two perfect seeds implement
DIFFERENT exact solutions. s0 = orbit automaton (ablating its
automaton-block injections costs nothing). s1 = injection-driven:
its automaton block's ||b|| is 2.35 (10x median) and ablating it
collapses the model to 0.64 even at train length. With exact group
transitions the injection sum is a group-convolution feature —
signal, not noise. Round-5 law restated: injections through exact
transitions are signal; injections trapped in non-contracting
NON-automaton channels are noise.

**R2a — projection + 400-step fine-tune**: trades in-distribution
accuracy for length robustness instead of repairing (s6 mask-off:
0.807@1024, +19, but @64 drops 0.99 -> 0.93; val@128 falls during
fine-tune). Failed seeds are globally mis-oriented, not
under-polished. Retired.

**R2b — hom-error vs val@128 checkpoint selection**: exact null.
Both criteria select the IDENTICAL checkpoint on both failed seeds.
Standard best-validation practice is sufficient; the homomorphism
certificate remains a diagnostic (mechanism verification + horizon
prediction via eps*T), not a selector. Retired.

**Closed shipping story**: rotation-snap + best-val@128 selection +
retry-on-flag (best val < 1.0 flags every bad run). No repair
machinery earned a place.

## Round: RNG hygiene fix + re-validation (`reseed-fix`)

**Bug found:** `variants.py`'s `run_cell` seeded its training-data
generator `torch.Generator().manual_seed(1 + seed)`, while eval calls
inside the same cell use fixed literal seeds — 2 (early-stop check),
3 (in-distribution accuracy), 4 (generalization accuracy), 5 (val@128
checkpoint selection). For seed >= 1 this put the training generator's
seed on top of one of those eval seeds: seed=1's train generator (2)
collides with the early-stop probe, seed=2's (3) with the
in-distribution accuracy probe, seed=3's (4) with the generalization
probe, seed=4's (5) with the val@128 probe. Train and eval draw
different-shaped batches from the shared seed, so the two streams
overlap only at the very start of training before diverging — a
partial, seed-dependent train/eval contamination, not a full leak, but
a real one. `probes.py`'s `run_one` had the identical flaw and was
fixed first, to `manual_seed(1 + 10_000 * seed)` (see its docstring
for the derivation); the same fix is applied here. `seed=0` maps to
`manual_seed(1)` under both the old and new formula, so every seed-0
row is bit-exact across the fix.

**Cells re-run** (`EXP_ROUND=reseed-fix`, same protocol/budgets as the
rows they replace; appended to `lab_results.jsonl`), with the seed-0
exact-reproduction check:

| cell | seeds | seed-0 exact match to its pre-fix row? |
|---|---|---|
| parity GRU L=1 | 0,1,2 | yes |
| parity signed-coupled L=1 | 0,1,2 | yes |
| parity signed-tanh L=1 | 0-5 | yes |
| S3 GRU L=1 | 0,1,2 | yes |
| S3 signed-coupled L=1 | 0,1,2 | yes |
| S3 signed-coupled L=4 | 0,1,2 | yes |
| S3 rotation-snap L=1 (`EXP_CKPT=1`) | 0-7 | yes |

**Before -> after (README table cells; means over the seed counts
recorded in the table):**

| task/model/L (n) | before @64/@256/@512/@1024 | after @64/@256/@512/@1024 |
|---|---|---|
| parity GRU/1 (3) | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 |
| parity signed-coupled/1 (3) | 1.000/0.894/0.719/0.610 | 1.000/0.866/0.687/0.592 |
| parity signed-tanh/1 (6) | 1.000/≥0.9999/0.999/0.996 (worst 0.979) | 1.000/1.000/0.999/0.994 (worst 0.984) |
| S3 GRU/1 (3) | 1.000/1.000/1.000/1.000 | 1.000/1.000/1.000/1.000 |
| S3 signed-coupled/1 (3) | 0.414/0.339/0.275/0.223 | 0.419/0.337/0.270/0.220 |
| S3 signed-coupled/4 (3) | 0.885/0.544/0.426/0.342 | 0.938/0.649/0.471/0.347 |
| S3 rotation-snap/1 (8) | 0.999/0.987/0.956/0.889 (exact 2/8) | 1.000/1.000/0.996/0.958 (exact 1/8) |

Every non-GRU row moved beyond its displayed 3-decimal rounding.

**Full history of the S3 signed-coupled L=4 @256 cell** (the largest
shift in the table, and the one README now only summarizes as a
high-variance caveat): a single-seed run early in this project
reported 0.655 for this cell. A later 3-seed mean of 0.544 was
reported in README as the fairer, multi-seed number, read at the time
as "0.655 was a lucky seed, not env drift." That 3-seed mean's seeds 1
and 2 were drawn under the train/eval generator-seeding bug fixed in
this round: seed 1's training generator (old formula `1 + seed` = 2)
collided with the early-stop eval seed (also 2), and seed 2's (3)
collided with the in-distribution-accuracy eval seed (also 3) — see
"Bug found," above. Re-run under the fix, seed-1/seed-2 @256 rose from
0.503/0.659 to 0.711/0.765 (seed 0, uncontaminated either way, stayed
at 0.471, an exact match to its pre-fix row). The corrected 3-seed
mean is 0.649 — close to the original single-seed report of 0.655,
not well below it as the pre-fix mean had suggested. Both the size and
direction of this shift are consistent with the seeding fix removing
train/eval overlap that had been suppressing this cell's
higher-variance seeds, not with a regression or an error in either
prior number; the cell remains genuinely high-variance seed-to-seed
(current per-seed acc@256 spans 0.471-0.765), which is why README
flags it as indicative rather than tight.

rotation-snap's exact-seed count dropped from 2/8 to 1/8, but — more
importantly — none of its 8 clean seeds were flagged by best-val@128
< 1.0, including the 7 that are not exact at length; pre-fix, every
non-exact seed had been flagged. The "best-val@128 perfectly separates
good from bad seeds" claim does not replicate under clean seeding and
has been corrected in README's "Rotation variant" section: the flag is
still worth retrying when it fires, but is not a sufficient pass/fail
check on its own.

README.md, this file's "Corrected headline comparisons" (above), and
SUMMARY.md's TL;DR table and Setup note have been updated to these
`after` values. The historical "Baselines re-grounded" table above and
the round-by-round hypothesis/outcome log earlier in this file are
left as the record of what rounds 1-9 and `verify-baselines` actually
observed under the pre-fix seeding — read them as history, not as
current claims.

## Round: heterogeneous stacks (hetero-legA / legB-v2 / ceiling)

Task 1 added per-layer mixer specification to `MinGRUStack` (a `list[str]`
`mixer`, one entry per block, e.g. `["signed", "rotation"]`). Task 2 built
on that contract: a new probe task, `S3-hier`, and multi-seed evidence for
two questions -- does one rotation block survive being stacked with
signed blocks (leg A, task `S3`), and does depth buy hierarchical
composition (leg B, task `S3-hier`). Per-seed rows for every cell below
are appended to `lab_results.jsonl` under the round tags named in each
subsection; full narrative and provenance discussion lives in the task's
own report (not part of this repo -- superseded by this section as the
durable record).

### Leg A: does one rotation block survive depth? (task `S3`, best-val@128 protocol, `MAX_STEPS=1600`, 3 seeds; round `hetero-legA`)

Unaffected by the later `LATIN` fix below -- task `S3` composes
per-token and does not reference `LATIN` at all.

| model | seeds | ckpt@128 (per seed) | mean acc@64/256/512/1024 |
|---|---|---|---|
| `minGRU-rotation2` (rotation x2, broken baseline) | 0,1,2 | 1.000/1.000/1.000 | 1.000/0.999/0.9828/0.9413 |
| `minGRU-hetero-sr` (signed -> rotation) | 0,1,2 | 0.985/0.999/1.000 | 0.999/0.976/0.9403/0.8626 |
| `minGRU-hetero-rs` (rotation -> signed) | 0,1,2 | 1.000/1.000/1.000 | 1.000/1.000/0.9986/0.9663 |

Recorded L=1 rotsnap reference (README, n=8, CKPT protocol): acc@64/256/512/1024
= 1.000/1.000/0.996/0.958. All three leg-A configurations land close to
that row; every construction of `minGRU-rotation2` (two rotation blocks)
emitted exactly one `UserWarning` (the multi-rotation STE-compounding
notice), as designed.

### Leg B v2: does depth buy hierarchy? (task `S3-hier`, `MAX_STEPS=1600` unless noted, 3 seeds; chance ~= 1/6 ~= 0.167; round `hetero-legB-v2`)

Current record, run under the fixed `LATIN` constant (non-isotopic to
either group of order 6 -- see `probes.py`'s `LATIN` comment). None of
these six 1x-budget configurations solves `S3-hier` in budget; every
number is budget-relative.

| model | protocol | mean acc@64/256/512/1024 |
|---|---|---|
| `minGRU-rotsnap` L=1 (no feature layer) | CKPT | 0.2247/0.1827/0.1743/0.1703 (near chance) |
| `minGRU-signed-tanh` L=2 (no composition) | early-stop | 0.985/0.866/0.754/0.6205 |
| `minGRU-hetero-sr` (signed -> rotation) | CKPT | 0.4863/0.3307/0.2951/0.2637 |
| `minGRU-hetero-rs` (rotation -> signed) | CKPT | 0.4477/0.3253/0.2468/0.2063 (near chance) |
| `minGRU-rotation2` (rotation x2) | CKPT | 0.3647/0.2563/0.2115/0.1889 (near chance) |
| `GRU` (intended ceiling) | early-stop | 0.2317/0.184/0.1749/0.1707 (near chance -- ceiling did not hold in budget) |

### Ceiling check: `minGRU-hetero-sr` at 4x budget (`MAX_STEPS=6400`, CKPT, 3 seeds; round `hetero-legB-ceiling`)

| seed | ckpt@128 | acc@64/256/512/1024 |
|---|---|---|
| 0 | 0.573 | 0.754/0.390/0.2825/0.2237 |
| 1 | **1.000** | **1.000/1.000/0.9999/0.9834** |
| 2 | 0.763 | 0.917/0.583/0.4268/0.2979 |
| **mean** | **0.779** | **0.890/0.658/0.570/0.502** |

Seed 1 finds the exact solution (best-val@128 = 1.000) and generalizes to
0.9834@1024 -- the best single length-generalization figure measured on
`S3-hier`. Seeds 0 and 2 do not, and best-val@128 correctly flags both
(0.573, 0.763) as runs to retry -- the same fast-but-unstable training
dynamic already documented for plain `minGRU-rotsnap`, extending into the
hetero stack's harder joint extract-and-compose optimization problem.
`minGRU-signed-tanh` L=2 was not re-run at 4x: its 1x fit is already
saturated (0.973-0.997@64 across seeds), and this file's own Round 6
finding ("once train accuracy saturates, the loss cannot see the
difference between the exact solution and a decaying shortcut") is why
extra budget is not expected to close its length-decay gap.

### Superseded: leg B under `LATIN = COMPOSE` (round `hetero-legB-v1-superseded`)

The original `S3-hier` evidence used `LATIN = COMPOSE` (S3's own Cayley
table). `COMPOSE` is trivially isotopic to S3, and `RotationMinGRU`'s
per-token angle assignment (a learned linear map) gives a rotation layer
the same relabeling freedom an isotopy allows -- so this pair function
was partially representable by one rotation layer, not just the narrower
"literal-index additive" form the cheap `_has_additive_violation` check
screens for. This leaked signal into every rotation-containing row
(`minGRU-rotsnap` L=1 reached 0.377@64 under the leak, vs. 0.225@64 after
the fix; `GRU`, with no rotation mechanism at all, moved from 0.401@64 to
0.232@64, confirming the leak was in the task, not one architecture), and
inflated the original headline finding ("heterogeneous stack wins") well
beyond what the fixed task supports (`minGRU-hetero-sr` dropped from
0.799@64, v1's "clear winner," to 0.486@64, v2, well below
`minGRU-signed-tanh` L=2's 0.985@64). All 18 v1 per-seed rows are
preserved in `lab_results.jsonl` under round `hetero-legB-v1-superseded`
for audit purposes, each tagged with a `note` field naming this leak --
**do not cite those numbers as current evidence; see the "Leg B v2"
table above for the corrected record.**

### Round tags in `lab_results.jsonl`

`hetero-legA` (9 rows), `hetero-legB-v2` (18 rows), `hetero-legB-ceiling`
(3 rows), `hetero-legB-v1-superseded` (18 rows, each with a `note`
explaining the leak). Full per-seed detail (steps, secs, ckpt step,
rotation-warning counts) beyond the means tabulated above is in those
rows.

## Round: time-decay evidence rescue (tdecay-probe / tdecay-sweep)

Per-seed rows behind the README's "Time-aware decay" section (channel
ablation, lambda-recovery check, and the `decay_rate` init sweep) were
gathered across several scratch sessions whose own reports were later
deleted; rescued here from the raw run logs before the branch that
produced them closed, tagged into `lab_results.jsonl` under two round
names:

- `tdecay-probe` (47 rows): the original 4-condition evidence sweep
  (session-parity / `minGRU-signed-tanh-tdecay`, session-parity /
  `minGRU-signed-tanh` feature-only baseline, parity-timestamped /
  `minGRU-signed-tanh-tdecay` lambda-recovery check, plain parity /
  `minGRU-signed-tanh` non-decay comparison point), its mechanical-only
  channel-ablation counterpart (`minGRU-signed-tanh-tdecay-mech`) and
  that mechanism's `MAX_STEPS=6000` ceiling check, a `decay_rate=1.0`
  init variant run for comparison against the recorded `decay_rate=0.05`
  default, and the post-fix reruns of seeds 1-2 for every condition
  above once the train/eval RNG-collision bug (documented earlier in
  this file, `reseed-fix`) was found and corrected. Seed 0 is unaffected
  by that fix in every case (reproduces `manual_seed(1)` regardless);
  each row's `note` field states whether it predates or postdates the
  fix, and which sibling row it should be combined with for the
  corrected 3-seed record.
- `tdecay-sweep` (18 rows): a finer `decay_rate` init sweep (R in
  `{0.005, 0.01, 0.02}`, both timestamped tasks, 3 seeds each, run after
  the RNG fix) motivating the choice of `0.05` as the shipped default --
  R=0.05 itself is the `tdecay-probe` rows above, not rerun in this
  sweep.

Only `acc@64`/`acc@256` were ever measured for these conditions (no
512/1024 eval was run); each row's `acc` dict has just those two keys,
reconstructed as faithfully as the source logs allow. See README's
"Time-aware decay" section for the recorded means and protocol
narrative; the per-seed rows here are the durable evidence trail behind
those numbers.

## Round: research-review remediation (s3-diag-ckpt / base-mingru)

An adversarial results/methods audit of the README flagged three
evidence gaps in the "What this shows" S3 table: (1) the only diagonal
comparator shown was the weak legacy `coupled=True` form — the repo's
recommended decoupled `signed-tanh` had just a single-seed round-2 S3
record, which the >=3-seed README policy kept out of the table, leaving
the misleading impression that diagonals cannot fit S3 at all; (2) the
`minGRU-rotsnap` row was recorded under best-val@128 checkpoint
selection while the diagonal rows used early-stop, so rotation's win
was confounded with the selection procedure; (3) the "base `MinGRU`
stays at chance regardless of depth" sentence had zero logged rows
behind it (entailed for parity, asserted for S3).

Two rounds close all three (torch 2.5.1, CPU, standard probes protocol:
T_train=64, d_model=64, batch 128, Adam lr 3e-3, budget 1600 steps,
seeds 0/1/2, eval lengths 64/256/512/1024 via `experiments/variants.py`):

- `s3-diag-ckpt` (12 rows): S3 x {`signed-tanh` L1/L4,
  `signed-coupled` L1/L4} under `EXP_CKPT=1` — best-val@128 selection
  over the full budget, the same protocol as the recorded
  `minGRU-rotsnap` rows.
- `base-mingru` (12 rows): {parity, S3} x base minGRU x {L1, L4}. The
  `log` entry was added to `variants.py`'s `VARIANTS` (importing
  `min_gru.MinGRU`) so these rows have a runnable public path.
  Early-stop is armed but never triggers on at-chance runs; every cell
  consumes the full 1600 steps.

Per-seed results (acc@64 / acc@256 / acc@512 / acc@1024):

| round | task | variant | L | seed | @64 | @256 | @512 | @1024 |
|---|---|---|---|---|---|---|---|---|
| s3-diag-ckpt | S3 | signed-tanh | 1 | 0 | 0.9996 | 0.9541 | 0.8778 | 0.7523 |
| s3-diag-ckpt | S3 | signed-tanh | 1 | 1 | 0.9991 | 0.9440 | 0.8648 | 0.7443 |
| s3-diag-ckpt | S3 | signed-tanh | 1 | 2 | 0.9986 | 0.9324 | 0.8503 | 0.7002 |
| s3-diag-ckpt | S3 | signed-tanh | 4 | 0 | 0.9998 | 0.9531 | 0.8403 | 0.6504 |
| s3-diag-ckpt | S3 | signed-tanh | 4 | 1 | 0.9913 | 0.8326 | 0.6382 | 0.4437 |
| s3-diag-ckpt | S3 | signed-tanh | 4 | 2 | 0.9993 | 0.9375 | 0.8290 | 0.6281 |
| s3-diag-ckpt | S3 | signed-coupled | 1 | 0 | 0.4813 | 0.3437 | 0.2721 | 0.2231 |
| s3-diag-ckpt | S3 | signed-coupled | 1 | 1 | 0.4593 | 0.3505 | 0.2910 | 0.2334 |
| s3-diag-ckpt | S3 | signed-coupled | 1 | 2 | 0.4680 | 0.3374 | 0.2691 | 0.2222 |
| s3-diag-ckpt | S3 | signed-coupled | 4 | 0 | 0.9892 | 0.6622 | 0.4637 | 0.3311 |
| s3-diag-ckpt | S3 | signed-coupled | 4 | 1 | 0.9978 | 0.7109 | 0.5100 | 0.3990 |
| s3-diag-ckpt | S3 | signed-coupled | 4 | 2 | 0.9958 | 0.7645 | 0.5439 | 0.3740 |
| base-mingru | parity | log | 1 | 0 | 0.5163 | 0.5041 | 0.5032 | 0.5007 |
| base-mingru | parity | log | 1 | 1 | 0.5484 | 0.5122 | 0.5070 | 0.5026 |
| base-mingru | parity | log | 1 | 2 | 0.5622 | 0.5172 | 0.5065 | 0.5040 |
| base-mingru | parity | log | 4 | 0 | 0.5163 | 0.5041 | 0.5032 | 0.5007 |
| base-mingru | parity | log | 4 | 1 | 0.5204 | 0.5051 | 0.5037 | 0.5010 |
| base-mingru | parity | log | 4 | 2 | 0.5125 | 0.5036 | 0.5004 | 0.5010 |
| base-mingru | S3 | log | 1 | 0 | 0.2243 | 0.1822 | 0.1744 | 0.1711 |
| base-mingru | S3 | log | 1 | 1 | 0.2296 | 0.1819 | 0.1742 | 0.1706 |
| base-mingru | S3 | log | 1 | 2 | 0.2334 | 0.1832 | 0.1758 | 0.1704 |
| base-mingru | S3 | log | 4 | 0 | 0.2093 | 0.1806 | 0.1721 | 0.1697 |
| base-mingru | S3 | log | 4 | 1 | 0.2003 | 0.1748 | 0.1712 | 0.1686 |
| base-mingru | S3 | log | 4 | 2 | 0.2549 | 0.1886 | 0.1783 | 0.1717 |

Conclusions:

1. The decoupled diagonal **fits S3 in-distribution and decays with
   length**: signed-tanh L1 means 0.999 @64 -> 0.943/0.864/0.732
   @256/512/1024. Depth does not rescue the shortcut (L4 mean 0.574
   @1024, with high seed variance: 0.44-0.65). The old table's
   "diagonals can't even fit S3" impression was an artifact of showing
   only the coupled baseline.
2. **Checkpoint selection does not rescue the coupled form at L=1**
   (0.470 @64 mean): its fit failure is the parameterization, not the
   protocol. Coupled L4 improves modestly over its early-stop record
   (0.368 vs 0.347 @1024) without changing any conclusion.
3. **Same-protocol control (the review's kill-shot check): rotation's
   S3 win survives.** With every S3 minGRU row under best-val@128, the
   best diagonal reaches 0.732 @1024 vs rotation's 0.958 — the win
   attributes to the mechanism, not to checkpoint selection.
4. **Base minGRU is measured at chance at L1 and L4 on both tasks**
   (parity 0.516-0.542 @64 falling to ~0.50; S3 0.222-0.229 @64 falling
   to ~0.17), closing the previously-unlogged "regardless of depth"
   claim.

The README's "What this shows" S3 table now quotes the `s3-diag-ckpt`
rows with a per-row protocol column; the earlier early-stop coupled S3
records (rounds 2 / `reseed-fix`) stay in `lab_results.jsonl` unchanged
and are superseded in the README table by these same-protocol rows.

## Round: incumbent comparison (incumbent-delta)

**Motivation:** closes the research review's MAJOR-3 scope gap at the
mechanism level. `DeltaNetMixer` (`experiments/variants.py`) reimplements
the DeltaNet (Yang et al.) / DeltaProduct (Siems et al., NeurIPS 2025)
generalized-Householder delta rule as a plain-torch, sequential
recurrence — nh=1 registered as `"deltanet"`, nh=2 (the rotation-capable
configuration) as `"deltaproduct2"`. The official DeltaNet/DeltaProduct
code is Triton/CUDA and trained under a different regime that cannot run
in this repo's CPU-only environment, so this round compares **transition
rules under this repo's protocol**, not the incumbents' released systems
or published numbers.

**Protocol:** `T_train=64`, `d_model=64`, batch 128, Adam lr 3e-3, budget
1600 steps (`MAX_STEPS` default), seeds 0/1/2, `L=1`; parity uses
early-stop (as the diagonal/rotation rows do), S3 uses best-val@128
checkpoint selection (`EXP_CKPT=1`, matching `minGRU-rotsnap`'s recorded
protocol); eval lengths 64/256/512/1024; `uv run --python 3.12 --with
'torch==2.5.1' python experiments/variants.py`, torch 2.5.1, CPU.

**Per-seed results** (all 12 `round: "incumbent-delta"` rows in
`lab_results.jsonl`; `val128` is the S3 checkpoint-selection val-acc@128,
blank where not applicable):

| task | variant | seed | steps | @64 | @256 | @512 | @1024 | val128 |
|---|---|---|---|---|---|---|---|---|
| parity | deltanet | 0 | 100 | 1.0000 | 0.9994 | 0.9038 | 0.7301 | — |
| parity | deltanet | 1 | 100 | 1.0000 | 1.0000 | 0.8989 | 0.8242 | — |
| parity | deltanet | 2 | 100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — |
| parity | deltaproduct2 | 0 | 100 | 1.0000 | 1.0000 | 0.9734 | 0.7389 | — |
| parity | deltaproduct2 | 1 | 100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | — |
| parity | deltaproduct2 | 2 | 100 | 1.0000 | 0.9999 | 0.8828 | 0.6924 | — |
| S3 | deltanet | 0 | 1600 | 0.4145 | 0.3551 | 0.3447 | 0.3377 | 0.3846 |
| S3 | deltanet | 1 | 1300 | 0.4197 | 0.3536 | 0.3449 | 0.3396 | 0.3830 |
| S3 | deltanet | 2 | 700 | 0.4231 | 0.3581 | 0.3439 | 0.3393 | 0.3858 |
| S3 | deltaproduct2 | 0 | 200 | 1.0000 | 1.0000 | 0.9999 | 0.9938 | 1.0000 |
| S3 | deltaproduct2 | 1 | 300 | 1.0000 | 1.0000 | 0.9995 | 0.9741 | 1.0000 |
| S3 | deltaproduct2 | 2 | 300 | 1.0000 | 1.0000 | 1.0000 | 0.9991 | 1.0000 |

**Means (n=3 each):**

| task | variant | mean @64 | mean @256 | mean @512 | mean @1024 |
|---|---|---|---|---|---|
| parity | deltanet | 1.0000 | 0.9998 | 0.9342 | 0.8514 |
| parity | deltaproduct2 | 1.0000 | 1.0000 | 0.9521 | 0.8104 |
| S3 | deltanet | 0.4191 | 0.3556 | 0.3445 | 0.3389 |
| S3 | deltaproduct2 | 1.0000 | 1.0000 | 0.9998 | 0.9890 |

**Conclusions:**

1. **nh=1 (`deltanet`) fits parity instantly but cannot fit S3** — parity
   mean@64 = 1.000 in 100 steps (same budget as every other parity row in
   this file), while S3 mean@64 = 0.419 (near the ~0.37-0.42 diagonal
   chance band recorded for other commutative/reflection-only mixers in
   this file's "research-review remediation" round). A single Householder
   reflection has determinant -1 on every application: it cannot compose
   to the even-permutation elements S3/D3 needs, matching DeltaProduct's
   own representability theory (single reflections are the paper's own
   ablation floor, not a state-tracking-complete mechanism).
2. **nh=2 (`deltaproduct2`) solves S3 near-exactly on every seed, and
   trains reliably with no retry protocol.** All three seeds hit
   val128 = 1.000 by step 200-300 (vs. a full 1600-step budget), and hold
   0.999-1.000 through @512 and 0.974-0.999 @1024 — composing two
   reflections per token realizes a rotation, which is state-tracking-
   complete for S3. This is a materially different training profile from
   this repo's `minGRU-rotsnap` (`RotationMinGRU`, L=1), whose recorded
   n=8 result (`reseed-fix` round, above) is 1.000/1.000/0.996/0.958
   @64/256/512/1024 with only 1/8 seeds landing the exact solution. The
   documented retry-on-flag rule is one-directional there — a sub-1.0
   val@128 reliably marks a bad run, but in that n=8 record every seed
   passed the flag and 7/8 still decayed, so retries cannot select for
   exactness on S3. `deltaproduct2` raised no flag and needed no retry
   across its 3 seeds.
3. **Parity length-generalization under early-stop is seed-inconsistent
   for both incumbents.** `deltanet` @1024 ranges 0.730-1.000 (mean
   0.851); `deltaproduct2` @1024 ranges 0.692-1.000 (mean 0.810). Both
   are well below this repo's recorded `signed-tanh` parity result
   (`reseed-fix` round, above), n=6: 1.000/1.000/0.999/0.994 (worst
   0.984) — signed-tanh's tanh asymptote sits at the exact eigenvalue
   parity needs, giving it an attractor the delta-rule's beta/k
   parameterization does not share under this protocol.
4. **All of the above is budget-relative** (1600-step budget, 3 seeds,
   this repo's harness and protocol only) — it is not a claim about the
   incumbents' released systems, published training regimes, or
   published numbers.

**Capacity disclosure (d_model=64):** parameter and per-token state-size
counts, computed by constructing each module and summing
`p.numel() for p in module.parameters()` (state size per the formulas
noted); script run via `uv run --python 3.12 --with 'torch==2.5.1'
python capacity_counts.py`, output quoted verbatim:

```
module                                      params   state/token
DeltaNetMixer nh=1 (n_heads=4)               16900          1024
DeltaNetMixer nh=2 (n_heads=4)               25480          1024
RotationMinGRU (minGRU-rotsnap config)       12544            64
SignedMinGRU (default, coupled=False)        12480            64
```

State size is `n_heads * d_k * d_v` per token for `DeltaNetMixer`
(4 heads x 16 x 16 = 1024; nh only changes how many micro-steps update
that same state per token, not its size) and `hidden_size` per token for
`RotationMinGRU`/`SignedMinGRU` (64). At equal `d_model`, both delta-rule
configurations carry a materially larger per-token state (1024 vs. 64)
and, for nh=2, roughly 2x `deltanet`'s own parameter count (25480 vs.
16900) — the comparisons above are same-`d_model`, not same-capacity.

## Round: hetero training-fix loop, part 1 (hetero-loop-01..06)

**Context.** The research synthesis
(`.claude/output/research/2026-07-13-hetero-stack-hypotheses.md`) ranked
training-side fixes for `minGRU-hetero-sr`'s S3-hier trainability
problem (recorded: near chance at 1600; 1/3 seeds exact at 6400, round
`hetero-legB-ceiling`). This round runs an autonomous hypothesize→test→
refine loop over that ranking: every arm pre-registers a hit/kill rule
before running, and every arm runs through
`experiments/hetero_lab.py` — whose flags-off path was verified to
reproduce the recorded `hetero-legB-v2` seed-0 row to 4 decimals
(steps 1300, val128 0.3243, acc 0.457/0.2447/0.2043/0.1864) before any
arm counted. Protocol per row: CKPT best-val@128 over the full budget,
acc@64 seed 3, gen lengths seed 4, T_TRAIN=64, batch 128, Adam 3e-3.

**Loop 1 — instrumented diagnostics (no new accuracy rows).**
Deterministic reruns of the three `hetero-legB-ceiling` seeds (best
val/step reproduced exactly: 0.5726@5100 / 1.0@5100 / 0.7626@5200) with
per-eval instrumentation on a fixed diagnostic batch:

1. *Coupling refuted.* A linear probe decodes the pair's Latin-square
   generator from layer-1 states at ≥0.95 by step 400–800 on every
   seed (plateau seeds reach 0.999 while task accuracy is near
   chance). Extraction is never the bottleneck. The winner's probe
   accuracy *drops* to ~0.92 as it locks in.
2. *The exact solution is not all-angles-on-grid.* The winner's mean
   angle residual stays ~0.27 snap-steps at loss 0.0 — unused blocks
   live off-grid. A global commitment penalty contradicts the observed
   solution geometry.
3. *Snap-flip oscillation is the lock-in signature.* Winner: flip rate
   0.10–0.27 while wandering, 0.001–0.01 once locked (~step 4900),
   with val128 bouncing 0.94→0.75→0.99→0.76 over steps 4400–4800
   before sticking. Plateau seeds flip at 0.07–0.24 indefinitely.
4. *Plateau seeds keep climbing at 6400* (seed 0 val128 0.35→0.53 over
   the back half), motivating the budget diagnostic below.

**Loop 2 — budget 2x (`hetero-loop-02-budget`).** Baseline flags-off at
12800 steps on the two plateau seeds:

| seed | best step | val128 | @64 | @256 | @512 | @1024 |
|---|---|---|---|---|---|---|
| 0 | 11500 | 0.657 | 0.885 | 0.421 | 0.294 | 0.232 |
| 2 | 7100 | 0.782 | 0.968 | 0.564 | 0.392 | 0.279 |

Neither locks at double budget (6400-budget bests: 0.573/0.763). The
failure is basin search, not pure speed; acceleration-family arms
demoted without spend.

**Loop 3 — soft-warmup composer, KILLED
(`hetero-loop-03-softwarmup`).** Tier-1 #1 (Guo soft-then-hard): 400
soft steps + 100-step STE blend, seeds 0–5 at 1600. Worse than
baseline on every matched seed — acc@64
0.246/0.262/0.312/0.287/0.425/0.271 vs baseline
0.457/0.533/0.469/0.442/0.999/0.534; best val128 0.373. Snapping at
step 400–500 destroys the soft-phase progress and the run ends worse
than training snapped from scratch: the soft solution's used angles
are not near grid points (consistent with diagnostic finding 2).
STE-from-scratch beats soft-then-hard here.
*Protocol incident:* the first launch scored CKPT selection in
training mode, letting a soft-phase val (0.367@400) win selection and
collapse at the final hard eval; fixed to deployment-mode selection
(`_hard_mode_eval`, commit e7c1cff), the one flawed row purged
(backup `.git/lab_results.jsonl.bak`), all seeds relaunched clean.

**Loop 3 bonus — baseline seed extension
(`hetero-loop-03-base-ext`).** Flags-off seeds 3–5 at 1600:

| seed | best step | val128 | @64 | @256 | @512 | @1024 |
|---|---|---|---|---|---|---|
| 3 | 1600 | 0.387 | 0.442 | 0.360 | 0.346 | 0.300 |
| 4 | 1200 | 0.988 | 0.999 | 0.929 | 0.730 | 0.535 |
| 5 | 1600 | 0.433 | 0.534 | 0.368 | 0.291 | 0.230 |

Seed 4 near-fits at the 1600 budget: the baseline hit rate at 1600 is
~1/6 (n=6 pooling `hetero-legB-v2` seeds 0–2), not 0/3 — screen
verdicts in this round compare against that rate, and the seed
lottery is wide (fit arrives by step 1200 when it arrives).

**Loop 5 — annealed gradient noise, KILLED
(`hetero-loop-05-gradnoise`).** Tier-1 #4 (Neelakantan):
`sigma^2 = 0.01/(1+t)^0.55` added to `p.grad` pre-Adam, seeds 0–5 at
1600. Unanimous catastrophic kill: every seed at chance (acc@64
0.185–0.192, val128 0.177–0.183) — below every baseline seed. Through
Adam the noise (sigma 0.1 at t=1, 0.013 at t=1600) swamps the raw CE
gradients and training never gets a clean phase; the parameterization
does not transplant from SGD-era setups onto Adam at this scale. Not
re-parameterized: the family's escape-bad-minima mechanism mismatches
the weak-attractor/long-wander diagnosis.

**Rank state after part 1.** Killed: soft-warmup (this schedule),
gradient noise (as parameterized). Demoted without spend:
identity-composer warmup (its extractor-first rationale died with
diagnostic finding 1), acceleration family (Loop 2). Live: composer
swap `signed→deltaproduct2` (Loop 4, `hetero-loop-04-sd2`), length
curriculum under the composer-credit rationale (Loop 6,
`hetero-loop-06-curriculum`), near-solution init epsilon-sweep,
wd x init-norm grid. Loops 4/6 report in part 2.

## Round: hetero training-fix loop, part 2 (hetero-loop-04..13)

**Loop 4 — composer swap, HIT 6/6 (`hetero-loop-04-sd2`).** Tier-2 #7:
`hetero-sd2` = signed-tanh extractor -> DeltaProduct nh=2 composer
(VariantBlock stack, `experiments/hetero_lab.py`), S3-hier, 1600 steps:

| seed | fit step | val128 | @64 | @256 | @512 | @1024 |
|---|---|---|---|---|---|---|
| 0 | 500 | 1.0 | 1.0 | 0.9838 | 0.8021 | 0.5260 |
| 1 | 1000 | 1.0 | 1.0 | 0.9885 | 0.8371 | 0.5562 |
| 2 | 1200 | 1.0 | 1.0 | 0.9894 | 0.8379 | 0.5570 |
| 3 | 900 | 1.0 | 1.0 | 0.9901 | 0.8010 | 0.5029 |
| 4 | 1600 | 0.9998 | 1.0 | 0.9918 | 0.8384 | 0.5484 |
| 5 | 800 | 1.0 | 1.0 | 0.9759 | 0.7703 | 0.4903 |

Every seed fits exactly (baseline `minGRU-hetero-sr`: 1/6 near-fit at
this budget); the seed lottery disappears; length gen decays smoothly
and consistently. Trainability is a property of the composer mechanism.

**Loop 6 — length curriculum, KILLED as a fixer
(`hetero-loop-06-curriculum`).** 25% of batches at even T in [2,16],
seeds 0-5: no seed reaches val128 0.9 (max 0.657). 4/6 matched seeds
improve modestly (acc@64 mean 0.638 vs 0.572) but the baseline's one
near-fit seed regresses (0.773 vs 0.999) — floor-raiser,
ceiling-lowerer.

**Loop 7 — attribution control, PASSES (`hetero-loop-07-d2L1`).**
`deltaproduct2` L=1 on S3-hier, seeds 0-5: all near chance (acc@64
0.225-0.251, val128 0.198-0.213). A single delta layer cannot both
extract the pair function and compose — the hetero win in Loop 4 is
depth buying hierarchy, not DeltaProduct solving S3-hier outright.

**Loop 8 — budget consolidates gen; selection-saturation artifact
fixed (`hetero-loop-08-sd2-6400`, `-v384`).** First attempt at 4x
budget selected by val@128, which saturates at 1.0 by step 500 —
the 6400 run reproduced the 1600 row exactly (row retained as the
saturation record). Fix: `--ckpt-t` selection-length flag with a
guard rejecting the reported test lengths; selection at T=384. Seeds
0-2: val384 0.965/0.949/0.960 (best steps 6400/4000/6100), acc@256
0.996-0.997, @512 0.877-0.889, @1024 0.590-0.597 (from ~0.53 at 1600).
Drift shrinks with budget but approaches an asymptote well below
exactness.

**Loops 9-13 — drift-mechanism refutation cascade (diagnostics; only
Loop 12 produced accuracy rows).** All diagnostics ran on
deterministic reruns that reproduced their ledger anchors exactly.

1. *Amplitude decay — refuted.* Learned beta sits far from 2 (medians
   0.53/0.89 per micro-step) yet composer state norm grows 22->29
   over 1024 steps; no collapse.
2. *Extraction drift — refuted.* Layer-1 generator probe at positions
   >=768 matches the early-position control (0.990 vs 0.988).
3. *Beta-exactness as the S3-vs-S3-hier discriminator — refuted.*
   Plain-S3 deltaproduct2 (which holds 0.974+ through 1024) learns
   the same non-orthogonal beta regime (medians 0.91/1.28); a
   "snap the delta gate" arm was killed before build.
4. *Composer-input non-stationarity — refuted.* Post-norm bucket
   statistics match an analytically stationary control (cos >=0.9996).
5. *Within-class map variance — measured real, causally refuted as
   the drift driver.* Per-token composed maps vary MORE within a
   generator class than across classes (spread/separation 1.4-2.0 per
   head). But collapsing the variance does not help: a from-scratch
   VQ8 interface bottleneck breaks fit entirely
   (`hetero-loop-12-vq8`: 6/6 chance), and post-hoc k-means
   quantization of the trained composer's input is chance at k<=16,
   and at k=64 preserves @64 (0.992) while making gen WORSE
   (0.671/0.444 vs 0.801/0.503 at 512/1024). The within-class clouds
   are functional — the composer's solution is a continuous
   distributed code, not one map per class.

**Where this leaves the mechanism picture.** The drift is intrinsic to
the continuous solution family the delta composer learns at this
scale/budget: it shrinks with training but asymptotes below exactness,
and every cheap structural handle (orthogonality, input hygiene,
discretization) has been ruled out by measurement or causal test. The
program-level trade, both directions mechanistically grounded:
discrete-map composers (rotation-snap) are exact when training finds
them but training rarely finds them; continuous-map composers
(DeltaProduct nh=2) are found by training every time but are never
exact. Loop 14 (basin radius of the exact solution under near-solution
init) reports in the close.

## Round: hetero training-fix loop, part 3 + program conclusions (hetero-loop-14)

**Loop 14 — basin radius of the exact rotation-composer solution
(`hetero-loop-14-basin`).** The `hetero-legB-ceiling` seed-1 winner was
regenerated deterministically (val128 1.0 @ 5100, matching the recorded
row), then retrained for 1600 CKPT steps from perturbed inits
`theta* + eps * std(tensor) * N(0,1)` with fresh data streams, 3 seeds
per eps:

| eps | val128 (3 seeds) | @1024 (3 seeds) | near-exact rate |
|---|---|---|---|
| 0.01 | 0.999 / 0.630 / 0.997 | 0.890 / 0.359 / 0.835 | 2/3 |
| 0.03 | 0.915 / 0.999 / 0.618 | 0.483 / 0.822 / 0.256 | 1/3 |
| 0.10 | 0.973 / 0.563 / 0.621 | 0.479 / 0.230 / 0.371 | 0/3 |
| 0.30 | 0.554 / 0.458 / 0.472 | 0.228 / 0.202 / 0.249 | 0/3 |
| 1.00 | 0.364 / 0.440 / 0.207 | 0.210 / 0.338 / 0.172 | 0/3 |

No run at any eps — including 1% noise — returned to exact
val128 = 1.0: the basin of exact re-attainment at this budget is
effectively zero. Near-solution init rescues approximate accuracy
(0.84-0.89 @1024 at eps=0.01) but not exactness; training orbits the
exact point without re-entering it. This upgrades the Round-6
"reachable but not a stable attractor" observation to a measurement.
Budget-relative: 1600-step retrains; longer budgets untested.

**Program conclusions (14 loops, rounds hetero-loop-01..14).**

1. **Trainability is a composer-mechanism property.** Swapping the
   snapped-rotation composer for DeltaProduct nh=2 inside the same
   2-layer hetero stack takes S3-hier fit from 1/6 seeds to 6/6 with
   no seed lottery; the L=1 control confirms the depth split
   (extraction below, composition above) is still doing the work.
2. **The coupling/supervision hypothesis family is refuted.** Layer-1
   generator information is linearly decodable (>=0.95) by step
   400-800 on every seed, including permanent plateaus; auxiliary
   supervision and extractor-first schedules target a bottleneck that
   does not exist.
3. **The reliability<->exactness trade is mechanism-level, measured
   from both sides.** Continuous-map composers train every time and
   asymptote below exactness (drift mechanisms exhaustively ruled
   out: amplitude, extraction drift, beta-orthogonality, input
   stationarity, within-class map variance — the last causally, via
   post-hoc quantization that preserves fit while worsening gen).
   Discrete-map composers are exact when found but the exact solution
   has a near-zero training basin even under informed initialization.
4. **Killed training-side arms** (pre-registered rules): soft-warmup
   schedules (soft progress does not survive snap-on), annealed
   gradient noise (Neelakantan parameterization is Adam-incompatible
   at this scale — all seeds to chance), length curriculum as a fixer
   (floor-raiser, ceiling-lowerer), budget doubling for the rotation
   composer (creep, no lock), VQ interface bottlenecks (from-scratch
   breaks fit; post-hoc worsens gen).
5. **Protocol lessons now encoded in the harness:** checkpoint
   selection must score the deployment-mode model
   (`_hard_mode_eval`), and must use a selection length that still
   discriminates once val@128 saturates (`--ckpt-t`, guarded against
   test-length leakage).

## Round: parallel-only extension (hetero-loop-15..16)

User constraint: mixers must remain parallel (associative scan); no
sequential code promoted. Two arms:

**Loop 15 — continuity is not the trainability driver, KILLED
(`hetero-loop-15-nosnap`).** `signed -> rotation(snap=None)` (existing
promoted mixers, continuous angles through `matrix_scan`), seeds 0-5 at
1600: 1/6 fits (seed 4, val128 = 1.0, 0.669 @1024) — the same rate and
the same lottery seed as the snapped baseline. Removing the STE snap
does not confer the delta composer's 6/6 reliability, so the snap
discontinuity was never the trainability blocker; the delta rule's
higher-dimensional matrix state / rank-1 update structure is implicated
instead.

**Loop 16 — the delta composer as an exact associative scan
(`hetero-loop-16-sd2par`).** `DeltaScanMixer`
(`experiments/hetero_lab.py --selftest`) shares `DeltaNetMixer`'s
parameters exactly and computes the same recurrence via a doubling scan
over affine pairs (A_t = composed per-token Householders, B_t =
accumulated injection). Outputs and gradients match the sequential
forward to 5-7e-7 (nh = 1 and 2, non-power-of-two T). Training
confirms: seed 0 fits at the same step as the sequential run (val128 =
1.0 @ 500) with accs within float-reordering noise
(1.0/0.981/0.798/0.517 vs 1.0/0.984/0.802/0.526). Wall-clock, however,
is worse on CPU — 3637s vs 1581s for the run; micro-benchmark 5.5x
slower at the training shape and 16x at T=1024 — because the scan
materializes dense 16x16 transitions (O(dk^3) per compose, log T
rounds) where the sequential path exploits the update's rank-1
structure (O(dk*dv) per token). Seeds 1-2 were stopped as redundant
(equivalence is proven; further seeds only re-confirm arithmetic).

**Conclusion.** The winning composer mechanism is expressible as a
parallel associative scan — the repo's design identity is satisfiable
in kind — but an *efficient* parallel form requires the chunked WY
representation from the DeltaNet literature, which is the actual
engineering a promotion would need. Naive matrix-scan parallelism is a
net loss on CPU at lab scale.

## Round: matched-capacity composers (hetero-loop-17..18)

The parallel-only extension's discriminating pair: is the delta
composer's 6/6 reliability a property of its mechanism (rank-1
write-erase updates) or of its 16x-larger state? Two composers at the
promoted mixers' 64-element per-token state, both trained through the
parallel `matrix_affine_scan`, `signed ->` hetero stacks on S3-hier,
seeds 0-5 at 1600, CKPT best-val@128:

**Loop 17 — `hetero-sg8` (GivensMinGRU: 8 blocks x 8 dims, per-token
transitions = 3 rounds of brick-wall Givens rotations,
special-orthogonal by construction, continuous). HIT, 4/6.**

| seed | best step | val128 | @64 | @256 | @512 | @1024 |
|---|---|---|---|---|---|---|
| 0 | 1100 | 1.000 | 1.000 | 0.9941 | 0.9105 | 0.6619 |
| 1 | 1500 | 0.942 | 0.9901 | 0.7853 | 0.5839 | 0.4293 |
| 2 | 1400 | 1.000 | 1.000 | 0.9953 | 0.9307 | 0.7447 |
| 3 | 1600 | 0.997 | 1.000 | 0.9941 | 0.9436 | 0.7751 |
| 4 | 1600 | 0.981 | 0.9989 | 0.8960 | 0.7137 | 0.5148 |
| 5 | 1400 | 1.000 | 1.000 | 0.9972 | 0.9558 | 0.8120 |

4/6 seeds at val128 >= 0.99; the two misses are near-fits (0.94/0.98,
@64 0.990/0.999) — no chance plateaus. n=6 means
0.998/0.944/0.840/0.656: the best @512 and @1024 of any reliably
trainable configuration recorded on S3-hier, above `hetero-sd2`
(0.987/0.815/0.530) at 16x less state.

**Loop 18 — `hetero-sdm` (DeltaScanMixer, n_heads=1, d_k=d_v=8: the
delta rule at 64 state elements, parallel scan). KILL, 2/6.**

| seed | best step | val128 | @64 | @256 | @512 | @1024 |
|---|---|---|---|---|---|---|
| 0 | 1600 | 0.210 | 0.2515 | 0.1877 | 0.1783 | 0.1723 |
| 1 | 1600 | 0.632 | 0.7979 | 0.4320 | 0.2999 | 0.2324 |
| 2 | 1400 | 1.000 | 1.000 | 0.9954 | 0.8982 | 0.6200 |
| 3 | 1100 | 1.000 | 1.000 | 0.9968 | 0.9576 | 0.7672 |
| 4 | 1100 | 0.227 | 0.2769 | 0.1924 | 0.1820 | 0.1748 |
| 5 | 1600 | 0.238 | 0.3132 | 0.2032 | 0.1845 | 0.1745 |

Two exact fits (with good gen when landed), one partial, three chance
plateaus — shrinking the delta state to 64 loses most of its
reliability (6/6 -> 2/6) and reintroduces the plateau failure mode.

**Conclusion (mechanism x state-size factorization; n=6 rates, being
extended to n=12 per external review).** With Loops 4/15: fit
reliability rises with per-token map richness within the rotation
family — 2D rotations 1/6 whether snapped or continuous, 8D Givens
4/6 with near-fit misses — at nearly matched parameters (full-stack
94,694 vs 92,614, +2.2%), so that within-family gradient is not a
parameter effect. Across mechanism families the evidence is weaker
and stated as such: 8D-state delta at 2/6 vs 8D Givens at 4/6 is a
two-seed difference at n=6 (far from significant on its own), and the
64-state delta composer also carries 4.4x fewer composer parameters
(3,306 vs 14,624), so mechanism and parameter count are confounded in
that cell. What IS well-powered: depth (6/6 vs 0/6) and within-delta
state size (6/6 at 1,024 state vs 2/6 at 64 state). Composer
parameter counts (d_model=64): deltaproduct2 25,480 / givens8 14,624
/ rotation-2d 12,544 / deltamini 3,306; full stacks hetero-sd2
105,550 / hetero-sg8 94,694 / hetero-sr 92,614 / hetero-sdm 83,376.

Efficiency, measured (single-process microbenchmark, min of 3,
uncontended; per-op FLOP arithmetic overstated these gaps and is not
quoted): fwd+bwd at the training shape (B=128, T=64) — sequential
delta16 0.179s, parallel-scan delta16 1.955s, parallel-scan givens8
0.961s, parallel-scan deltamini 0.268s. The Givens scan is ~2x
cheaper than the delta16 scan but still slower than the sequential
delta path on CPU, and even the small-state deltamini scan trails it
(~1.5x) — no parallel-scan config beats the sequential rank-1
implementation on CPU. A promotion case for GivensMinGRU rests on
the parallel-only design constraint plus reliability and length
generalization, not on CPU cost. Exactness at length remains unique to the snapped composer's
rare winner (0.983 @1024, 1/6 with a near-zero re-attainment basin);
no continuous composer reached it.

**n=12 extension (external-review remediation, same rounds).** Seeds
6-11 added to `hetero-loop-17-sg8`, `-18-sdm`, and `-15-nosnap` under
the identical protocol. Pooled fit rates (val@128 >= 0.99) and n=12
means:

| config | fits | @64 | @256 | @512 | @1024 | fit-only @512 / @1024 |
|---|---|---|---|---|---|---|
| `hetero-sg8` | 8/12 | 0.949 | 0.885 | 0.787 | 0.613 | 0.927 / 0.733 |
| `hetero-sdm` | 4/12 | 0.575 | 0.495 | 0.457 | 0.376 | 0.942 / 0.739 |
| `hetero-sr-nosnap` | 1/12 | 0.515 | 0.404 | 0.352 | 0.293 | 0.902 / 0.669 |

Fisher exact (two-sided) on fit rates: within the rotation family
(8/12 vs 1/12) p = 0.0094 — the map-richness gradient is established;
within the delta family (6/6 at 1,024 state vs 4/12 at 64 state)
p = 0.0128 — the state-size effect is established; across mechanisms
at matched state (8/12 vs 4/12) p = 0.22 — suggestive only, and still
parameter-confounded (14,624 vs 3,306 composer params). sg8's four
misses: three near-fits (val@128 0.83-0.98) and one low seed (0.384)
— the extension surfaced one plateau-like miss, so "no chance
plateaus" holds only as a tendency, not a rule. Notable: fit-only
generalization is indistinguishable between sg8 and sdm fits (0.733
vs 0.739 @1024) — the mechanisms differ in how OFTEN training finds a
solution, not in how well the found solutions generalize.

Efficiency, re-measured uncontended (min of 3, fwd+bwd at B=128,
T=64): sequential delta16 0.179s; parallel-scan delta16 1.955s;
parallel-scan givens8 0.961s; parallel-scan deltamini 0.268s. The
earlier same-window numbers were CPU-contended; conclusions
unchanged except one correction: the deltamini scan is ~1.5x SLOWER
than the sequential delta16 path, not faster — no parallel-scan
config beats the sequential rank-1 implementation on CPU.

## Round: GivensMinGRU promotion evidence replication (`givens-promotion-replication-01`)

Provenance-transfer gate for the promoted `min_gru.GivensMinGRU` /
`matrix_affine_scan`: the bit-identity self-tests
(`hetero_lab.py --selftest`) prove the promoted `GivensMinGRU` class and
scan are byte-identical to the frozen lab class at default flags, but
the pooled n=12 `hetero-loop-17-sg8` evidence was trained through
`experiments/hetero_lab.py`'s `_HeteroVariantTagger`
(`("signed-tanh", "givens8")` factories), not through `probes.py`'s
`MinGRUStack`. This round closes that gap: one `probes.py` run of the
promoted registry row `minGRU-hetero-sg8` (`S3-hier`, seed 0, CKPT
best-val@128, `MAX_STEPS=1600`) reproduces the recorded
`hetero-loop-17-sg8` seed-0 row exactly, showing the promoted path IS
the evidence path.

Command:

```
uv run --python 3.12 --with 'torch==2.5.1' python - <<'PY'
import probes
model = probes.run_one("S3-hier", "minGRU-hetero-sg8", ckpt=True, max_steps=1600, seed=0)
make, _, _ = probes.TASKS["S3-hier"]
for T, seed in [(64, 3), (256, 4), (512, 4), (1024, 4)]:
    print(T, round(probes.accuracy(model, make, T, seed=seed), 4))
PY
```

Recorded (`hetero-loop-17-sg8`, seed 0) vs replicated
(`givens-promotion-replication-01`, seed 0):

| field | recorded | replicated | match? |
|---|---|---|---|
| steps (best checkpoint step) | 1100 | 1100 | yes |
| ckpt val128 | 1.0 | 1.0 | yes |
| acc@64 (seed 3) | 1.0 | 1.0 | yes |
| acc@256 (seed 4) | 0.9941 | 0.9941 | yes |
| acc@512 (seed 4) | 0.9105 | 0.9105 | yes |
| acc@1024 (seed 4) | 0.6619 | 0.6619 | yes |

Exact match on every metric — no tolerance widening was needed. The
replication row is appended to `lab_results.jsonl` under round
`givens-promotion-replication-01`, `config.replicates` pointing back at
`hetero-loop-17-sg8` and `config.commit` recording the promoted-code
commit the run was executed against. This transfers the pooled n=12
`hetero-loop-17-sg8`/`-18-sdm`/`-15-nosnap` evidence (README's Givens
mechanism numbers) onto the promoted `min_gru.py` code path without
rerunning the multi-seed campaign.

## Round: Givens rounds ablation (`hetero-loop-19-rounds`)

Motivation: an external adversarial review of `TECHNICAL_REPORT.md` flagged the headline `hetero-loop-17-sg8` finding (Givens-8 fits 8/12 vs 1/12 for the continuous 2D composer at matched 64-element state) as causally under-controlled: the comparison moves per-token map expressivity ($SO(8)$ vs $SO(2)$) and within-scan block connectivity (8 coupled channels vs 2) together, so "map richness raises fit rate" was supported only correlationally. The isolating ablation holds `block_size=8` fixed and varies `rounds` $\in \{1, 2, 3\}$: at `rounds=1` the brick-wall mesh is 4 disjoint, commuting 2D planes inside the 8D block (32 planes per token, plane-count-matched to the 2D composer's 32 blocks, no cross-plane coupling); `rounds=2` adds the one staggered layer that couples all 8 dims and breaks commutativity; `rounds=3` is the recorded promoted configuration.

Design: `hetero-sg8r1` / `hetero-sg8r2` arms added to `experiments/hetero_lab.py` (`givens8r1`/`givens8r2` factories, same `_HeteroVariantTagger` signed-tanh extractor stack, same CKPT best-val@128 protocol), n=12 seeds each on `S3-hier` at the 1600-step budget; `rounds=3` is the existing `hetero-loop-17-sg8` n=12 arm, not rerun. Runs executed 4-way parallel with `OMP_NUM_THREADS=2`.

Command (per run):

```
OMP_NUM_THREADS=2 uv run --python 3.12 --with 'torch==2.5.1' python experiments/hetero_lab.py \
    --round hetero-loop-19-rounds --model hetero-sg8r1 --seed 0 --steps 1600
```

Results (fit = ckpt val@128 >= 0.99; means over all 12 seeds):

| composer | rounds | fits | acc@64 | acc@256 | acc@512 | acc@1024 | fit-only @512/@1024 |
|---|---|---|---|---|---|---|---|
| givens8r1 | 1 | 0/12 | 0.376 | 0.235 | 0.196 | 0.182 | — |
| givens8r2 | 2 | 6/12 | 0.805 | 0.761 | 0.675 | 0.515 | 0.916 / 0.704 |
| givens8 (recorded, `hetero-loop-17-sg8`) | 3 | 8/12 | 0.949 | 0.885 | 0.787 | 0.613 | 0.927 / 0.733 |

Fisher exact (two-sided): rounds 1 vs 3 $p \approx 0.0013$; rounds 1 vs 2 $p \approx 0.014$; rounds 2 vs 3 $p \approx 0.68$ (inseparable); rounds 1 vs the 2D composer's 1/12 $p = 1.0$ (indistinguishable). Threshold-robust: at fit threshold 0.98 the counts read 0/12, 8/12 (two near-fits at 0.983), 9/12 — same ordering.

Conclusion: **the block-size/connectivity hypothesis is refuted and the non-commutativity attribution is earned.** An 8D block of disjoint commuting planes fits 0/12 — no better than the 2D composer despite the larger coupled state block — while the single staggered layer that breaks within-block commutativity recovers most of the fit-rate effect (6/12), and the third layer adds a statistically inseparable increment (8/12). Fit-only generalization is unchanged across rounds (0.704 vs 0.733 at T=1024), consistent with the reliability-vs-quality separation from `hetero-loop-17/18`. Residual caveat on the cross-family anchor only: `givens8r1` is a pure rotation (no $\tanh u$ scale channel, cannot contract), while the 2D composer carries the scale, so the r1-vs-2D equivalence is suggestive rather than exact; the within-family monotonicity (0/12 → 6/12 → 8/12 under one factory, one protocol) is the controlled result. Rows appended to `lab_results.jsonl` under `hetero-loop-19-rounds`.

## Round: reversal-emulation-01 (division-reversal error, committed reproduction)

The design-phase emulation that rejected the division-based reversible backward for the rotation-family mixers (chunked reversal error growing as $\sigma_{\min}^{-\text{chunklen}}$; the $9.3 \times 10^{19}$ figure at default decay) predated this ledger and was never committed. This round commits a reproducible protocol: `experiments/reversal_emulation.py` (fp32, $T=4096$, $D=256$, $C=64$, seed 0, per-step decays uniform in $[\gamma_{\min}, 1)$) isolating the division channel of the reversal. Measured decay-gate gradient error (global metric), with the analytic worst-case bound $\varepsilon\,\gamma_{\min}^{-(C-1)}$ alongside: $\gamma_{\min}=0.86$: $1.8 \times 10^{-5}$ (bound $1.6 \times 10^{-3}$); $0.48$: $1.5 \times 10^{3}$ (bound $1.4 \times 10^{13}$); $0.23$: $6.1 \times 10^{12}$ (bound $1.9 \times 10^{33}$). Strict per-element relative error reaches $4.1 \times 10^{14}$ at $\gamma_{\min}=0.23$. Same mechanism and conclusion as the design-phase run (protocol-dependent mantissa); artifacts at `experiments/bench/reversal_emulation.{json,md}`. The shipped backward remains the exact $C=1$ stored-state recompute for both rotation-family mixers.

## Round: delta-mixer chunked-WY vs naive-affine-scan bench (`delta-mixer-bench-01`)

The design's efficiency claim for the newly-promoted `DeltaMinGRU` (spec section 9.9: "the efficiency claim is validated by measurement ... never asserted") required a CPU bench row showing its chunked-WY forward beating the naive per-token affine-scan reduction at $B=128, T=64$ and at $B=128, T=1024$ before the docs (Task 5) can honestly describe a parallel, efficient delta path. `scripts/bench_delta.py` times four arms (uncontended, single process, 1 discarded warmup + min of 3 timed forward+backward iterations) from one shared seeded `DeltaMinGRU` instance and input tensor per shape (`input_size=hidden_size=64, n_heads=4, nh=2, d_k=d_v=16, chunk_size=64`, seed 0), so the three delta arms see identical weights and data and differ only in computation path: the sequential per-token/per-micro-step recurrence (the lab-certified oracle), the naive affine-scan reduction (per-token maps through the frozen `matrix_affine_scan` — the measured-slow path), and the shipped chunked-WY UT-transform `forward`. A fourth arm, the packaged `GivensMinGRU` (`block_size=8, rounds=3`, the `hetero-loop-17/18` config) rides along as a cross-mixer reference — a different mixer/function, not part of the three-way delta comparison.

**Evidence-pin environment.** This bench is deliberately recorded under `torch==2.5.1` — the same torch version (and machine, Apple M2 Pro) as the previously-recorded `hetero-loop-17/18` row (`GivensMinGRU` 0.961s, sequential `delta16` 0.179s at the lab shape) — not the packaged distribution's declared `torch>=2.8` install floor (packaging metadata for the separately-gated Triton GPU surface only; every arm here runs the eager CPU path). `scripts/bench_delta.py` asserts `torch.__version__ == "2.5.1"` at runtime and refuses to write an artifact under any other version, and bootstraps `src/` onto `sys.path` (mirroring the root `min_gru.py` evidence driver) so `from mingru import ...` resolves even when the evidence-pin invocation (`uv run --no-project --with torch==2.5.1 python scripts/bench_delta.py`) doesn't sync this project. **Pre-timing agreement gate:** before any timing, the three delta arms are forward-compared pairwise at `atol=1e-5` at the full bench config's $T=64$ and at a small ragged shape ($B=4, T=13$, `chunk_size=5`, exercising a chunk-boundary the $T=64$ config's `chunk_size=64` ($\geq T$) never reaches) — all six comparisons passed (max abs diff $\leq 3.1\times10^{-6}$), so the timing that follows is guaranteed to be timing three implementations of the *same* function, not an artifact of one arm silently computing something cheaper.

**Measured** (current recorded artifact, `experiments/bench/delta_paths.md`): at $T=64$, sequential 0.1617s, naive affine-scan 2.0514s, chunked-WY 0.0577s, `GivensMinGRU` 0.9493s; at $T=1024$, sequential 19.9742s, naive affine-scan 71.7562s, chunked-WY 1.4541s, `GivensMinGRU` 24.9996s. Chunked-WY beats naive affine-scan by **35.55x at $T=64$ and 49.35x at $T=1024$** (also beating the sequential oracle, 2.80x and 13.74x respectively) — the spec section 9.9 acceptance criterion holds cleanly at both shapes, PASS, with no reduced-batch fallback needed (the naive arm's ~70s/iteration at the full $B=128$ was directly tractable). The `GivensMinGRU` reference arm is 16.45x/17.19x slower than chunked-WY `DeltaMinGRU` on this run (not a like-for-like comparison — different mixer, different math — included only for cross-mixer context); more importantly, its own absolute number (0.9493s at the lab shape) reproduces the `hetero-loop-17/18` recorded 0.961s within ~1.2%, on the same torch version and machine — direct evidence that the evidence-pin environment is comparable to that earlier round, not just an assertion. This is consistent with this ledger's earlier `hetero-loop-17/18` finding that the *lab's* naive parallel-scan delta path (`DeltaScanMixer`) never beats its own sequential path on CPU — that finding is about a different (non-chunked) parallel reduction and is not contradicted here; the chunked-WY form is a different algorithm (Yang et al., arXiv:2406.06484) specifically designed to close that gap, and this round is the first measurement of it. Artifacts at `experiments/bench/delta_paths.{json,md}`.

## Round: matched-state fit reliability, packaged delta vs recorded Givens (`hetero-loop-20-pd64`, `hetero-loop-21-pd1024`)

The first trainability evidence for the packaged chunked-WY `DeltaMinGRU` (`mixer="delta"`, promoted on `feat/delta-mixer`), answering the round's two decision questions — which mixer trains more reliably on $S3\text{-hier}$ at what cost, and whether the Triton chunked-WY kernel question opens. Two new 12-seed arms ran the exact `hetero-loop-17/18` protocol (CKPT=1 best-val@128 selection over the full 1600-step budget — never early stopping, per the §5.2/round-6 finding that S3-hier training wanders out of exact solutions after saturation — $T_{train}=64$, $d_{model}=64$, batch 128, Adam 3e-3, eval $T \in \{64,256,512,1024\}$, seeds 0–11) through the untouched `run_arm` driver: `hetero-loop-20-pd64` (`signed-tanh` $\to$ packaged `DeltaMinGRU(nh=2, n_heads=1, d_k=d_v=8)`, 64-element composer state, 3,306 composer params, mirroring the recorded `deltamini`) and `hetero-loop-21-pd1024` (`signed-tanh` $\to$ packaged `DeltaMinGRU(nh=2, n_heads=4, d_k=d_v=16)`, 1024-element state, 25,480 params, mirroring the recorded `hetero-sd2` deltaproduct2 arm). Comparison arm: the recorded 12-seed `hetero-loop-17-sg8` Givens row (14,624 composer params) — balanced 12v12, same environment class, no pooling across rounds.

**Provenance and environment.** All 24 runs under the evidence pin (`torch==2.5.1`, CPU, Apple M2 Pro), enforced in-process by the new `--require-torch 2.5.1` lab guard before any ledger write. Before any seed launched, the delta bridge selftest proved the packaged mixer `state_dict`-compatible with the lab's `DeltaScanMixer` in both load directions at both arm configs, with forward parity $\leq 2.7\times10^{-6}$ (tolerance $10^{-5}$) at $T=64$ and ragged $T=13$ — the new arms train the same function the recorded delta arms trained, differing only in float ordering (chunked-WY vs sequential). The 24 runs executed via `scripts/run_matched_state.py` (pool of 6 workers, children capped to 1 torch thread — recorded in `experiments/bench/matched_state_cost.json`; contended pool wall times are logged there as non-evidence; per-child full-training peak RSS 1.42–1.96 GiB both arms). Fit counts, means, threshold-robustness, and Fisher contrasts below are computed from the ledger rows by `scripts/run_matched_state.py report` (exact two-sided Fisher, verified bit-identical to scipy on the recorded contrasts), not hand-transcribed; cost/memory columns come exclusively from the uncontended subprocess-isolated scaling probe (`experiments/bench/scaling_frontier.{json,md}`, same pin, $B=128$, $T=64$, mixer-layer forward+backward, median of 5 timed steps).

| config | seeds | fits | acc@64 | acc@256 | acc@512 | acc@1024 | fit-only @512/@1024 | per-step cost | peak mem |
|---|---|---|---|---|---|---|---|---|---|
| givens@64 (recorded `hetero-loop-17-sg8`) | 12 | 8/12 | 0.949 | 0.885 | 0.787 | 0.613 | 0.927 / 0.733 | 0.984s | 887 MiB |
| delta@64-matched (`hetero-loop-20-pd64`) | 12 | 4/12 | 0.548 | 0.484 | 0.448 | 0.364 | 0.939 / 0.721 | 0.021s | 324 MiB |
| delta@1024 (`hetero-loop-21-pd1024`) | 12 | 12/12 | 1.000 | 0.988 | 0.833 | 0.556 | 0.833 / 0.556 | 0.060s | 508 MiB |

Threshold-robustness (fits at $\{0.98, 0.99, 0.995\}$): givens@64 $9/12, 8/12, 8/12$; delta@64-matched $4/12, 4/12, 4/12$; delta@1024 $12/12, 12/12, 12/12$ — every ordering below is threshold-stable. Two-sided Fisher exact vs the recorded givens@64: delta@64-matched $4/12$ vs $8/12$, $p = 0.2203$; delta@1024 $12/12$ vs $8/12$, $p = 0.0932$. Neither contrast clears $0.05$: per the round's pre-stated statistics honesty rule, 12v12 resolves only large gaps, so the matched-state fit-rate read is **no large difference at matched state** (and the delta@64 arm carries $4.4\times$ fewer composer parameters, so even the direction of the gap is confounded by capacity), and cost/memory decide. Supersession notes: the packaged delta@64 arm reproduces the recorded sequential `hetero-sdm` arm's fit count exactly ($4/12$, same bimodal profile — fits at val@128 $= 1.0$, non-fits at chance plateaus $0.21$–$0.36$; fit seeds $\{2,3,8,11\}$) and supersedes it as the packaged-mixer evidence row; delta@1024's $12/12$ extends the recorded 6-seed `hetero-sd2` $6/6$ to $n=12$ under the packaged forward. Fit-only length generalization stays mechanism-independent at matched state ($0.721$ delta vs $0.733$ givens at $T=1024$), while the delta@1024 cohort — every seed fitting — generalizes worse at length ($0.556$), reproducing the recorded state-size pattern (recorded `hetero-sd2`: $0.530$).

**Mechanism×state scaling frontier** (`experiments/bench/scaling_frontier.{json,md}`; all 10 configs completed, no timeouts/OOM — the Givens wall is practical, not hard): delta per-step cost and memory are nearly flat in state size — $0.050$s/419 MiB at state 64 to $0.077$s/676 MiB at state 4096 ($n_{heads}=4$ sweep) — while Givens scales steeply: $0.984$s/887 MiB at 64, $3.73$s/2.1 GiB at 256, $14.6$s/6.8 GiB at 1024, $70.6$s/20.0 GiB at 4096. At matched state 1024 the delta path is $\approx 244\times$ faster per step and $\approx 14\times$ lighter; at 4096 it is $\approx 916\times$ faster and $\approx 30\times$ lighter, with givens@4096's 20.0 GiB nearly saturating the 32 GB machine. Consistency cross-checks: the probe's givens@64 ($0.984$s) reproduces the recorded $0.961$s / bench $0.9493$s within $\approx 4\%$, and the probe's delta@1024 training-arm config ($0.0596$s) matches the pinned bench's chunked-WY $0.0577$s within $\approx 3\%$. In budget terms: **at a $\approx 900$ MiB per-step memory budget, delta reaches state 4096 where Givens caps at 64; the delta@1024 arm fits $12/12$ at $0.060$s/step and 508 MiB where givens@64 fits $8/12$ at $0.984$s/step and 887 MiB.**

**Verdict.** At matched 64-element state, neither mechanism outperforms the other on fit reliability ($p = 0.2203$, param-unmatched); on cost and memory, delta outperforms Givens at every measured budget — concretely, **delta at its native 1024-element state outperforms givens@64 at every measured cost and memory budget $M \geq 508$ MiB (fit rate $12/12$ vs $8/12$ at $\frac{1}{16}$ the per-step cost and $0.57\times$ the memory)** — while the recorded Givens fit cohort retains the better fit-only length generalization at $16\times$ length ($0.733$ vs the delta@1024 cohort's $0.556$). Decision outputs: (1) mixer recommendation — the delta mechanism is the training-reliability and cost winner at practical budgets; a promotion of `mixer="delta"` to the recommended composer is supported by this round *for fit reliability and cost*, with the fit-only length-generalization caveat disclosed (promotion and the consequent docs/slides updates are a deliberate follow-on, not part of this round). (2) Triton disposition — the user's pre-stated rule ("if delta ties or beats at matched state") is met by the tie: **the chunked-WY Triton kernel question opens**, as a speedup-worth-it judgment (profile eager chunked-WY on CUDA at real workload shapes against the matmul-FLOP floor before building; Triton never serves MPS/Mac regardless). Artifacts: per-seed rows in `experiments/lab_results.jsonl` (rounds `hetero-loop-20-pd64`, `hetero-loop-21-pd1024`), cost sidecar `experiments/bench/matched_state_cost.json`, frontier `experiments/bench/scaling_frontier.{json,md}`.

## Round: CUDA fusion-headroom probe for the Triton chunked-WY kernel decision (`gpu-delta-probe-01`)

The matched-state round opened the Triton kernel question as a speedup-worth-it judgment; this round answers it. `scripts/gpu_delta_probe.py` (submitted via `scripts/gpu_check.py --job delta-probe`, Lightning studio-mode L4 job at commit `bcca33d`) measured, per shape, three arms: eager chunked-WY `DeltaMinGRU` forward+backward (CUDA-event timed, 3 warmup + median of 10), an **approximate matmul-FLOP floor** (the seven dominant GEMM/solve contractions of `_forward_chunked` timed standalone at their actual shapes, forward-only ×3 fwd→fwd+bwd convention — disclosed in the artifact as approximate, and rougher for the `solve_triangular` component, whose backward is a second triangular solve plus an outer-product matmul, not two GEMMs), and the same layer under `torch.compile`. **Environment (new GPU evidence stratum — never comparable to the pinned-CPU rows):** NVIDIA L4, torch 2.8.0+cu128, triton 3.4.0, TF32 matmul off / `float32_matmul_precision=highest`, full env block in `experiments/bench/gpu_delta_probe.json`.

| config | T | eager | floor (approx) | compile | headroom (eager/floor) | compile-recovered | eager peak mem |
|---|---|---|---|---|---|---|---|
| d=64, 4h (pd1024) | 64 | 7.38 ms | 2.83 ms | 4.12 ms | 2.60x | 0.72 | 178 MiB |
| d=64, 4h (pd1024) | 256 | 27.63 ms | 12.69 ms | 15.42 ms | 2.18x | 0.82 | 646 MiB |
| d=64, 4h (pd1024) | 1024 | 209.78 ms | 53.87 ms | 67.97 ms | 3.89x | 0.91 | 2540 MiB |
| d=256, 4h | 256 | 84.61 ms | 25.89 ms | 43.56 ms | 3.27x | 0.70 | 1399 MiB |
| d=256, 4h | 1024 | 699.24 ms | 105.05 ms | 189.41 ms | 6.66x | 0.86 | 5543 MiB |

All five shapes completed with `status` clean: no compile failures, no `floor_suspect` rows — this run doubled as the probe's first functional GPU test and passed. Fusion headroom over eager is real (2.2x–6.7x, growing with $T$ and width), but `torch.compile` recovers 70–91% of it with zero engineering: compile beats eager by 1.79x/1.79x/3.09x/1.94x/3.69x across the rows, leaving a **maximum residual kernel win of 1.22x–1.80x versus a floor that is deliberately optimistic** (matmul-only, no launch/reduction overhead, rough solve-backward convention). **Verdict: the Triton chunked-WY kernel is not worth building now — adopt `torch.compile` for CUDA delta training.** The kernel question can reopen if a workload emerges where the compile-to-floor residual is material at scale (the residual grows with width: 1.80x at d=256/T=1024) or where compile is unusable; Triton never serves MPS/Mac regardless.

Provenance note: the job's main command completed but the run's keepalive heartbeat (added on the mistaken assumption that the Lightning tier's 10-minute idle shutdown applies to jobs — it applies to interactive studio sessions only) kept the studio-mode job alive, so the submitter's `job.wait()` never returned; the job was stopped via the SDK and the artifact extracted from the recovered logs' `MINGRU_GPU_PROBE_RESULT` line — the same guarded-extraction path the submitter uses. The heartbeat has been removed from the job command entirely (`scripts/gpu_check.py`; jobs keep their command chain foreground-only), so subsequent runs terminate normally. The probe's measured numbers are unaffected (the heartbeat printed a timestamp every 5 minutes from a sleeping shell loop; all arms are CUDA-event timed). Artifacts: `experiments/bench/gpu_delta_probe.{json,md}`.

## Round: GPU 36-seed campaign, givens-triton vs delta-compiled (`hetero-gpu36-sg8`, `hetero-gpu36-pd64`, `hetero-gpu36-pd1024`)

**Purpose (user-framed): package next-steps decision input, not cross-stratum science.** Three arms, 36 seeds each (0–35), single Lightning L4 job at commit `1d6c1dc` (`mingru-gpu-hetero36-1d6c1dc`), each mixer on its best-practical GPU path: the packaged `GivensMinGRU` composer (`hetero-pg8`, block_size=8, rounds=3) under the forced Triton scan path (`MINGRU_SCAN=triton`, fail-loud), and the two packaged `DeltaMinGRU` configs under `torch.compile`. Protocol identical to TECHNICAL_REPORT §4.4 (CKPT=1 best-val@128 over the full 1600-step budget — never early stopping — $T_{train}=64$, $d_{model}=64$, batch 128, Adam 3e-3, eval $T \in \{64,256,512,1024\}$), run by the untouched `run_arm` through the lab's new additive `--device cuda`/`--compile` flags. **Stratum disclosure (mandatory): every number here is torch 2.8.0+cu128 / NVIDIA L4 / triton 3.4.0 / compiled-or-Triton-kernel execution — a different numerics stratum from the pinned torch-2.5.1 CPU rounds, deliberately NOT comparable to them (no CPU-GPU replication was run, by explicit user decision); no cross-stratum statistical contrast is drawn.** Every row self-describes the stratum (`config: {device, torch, compile|scan}`); env sidecar `experiments/bench/gpu36_env.json`; extraction accounting reconciles exactly (108 extracted = 108 appended + 0 duplicate + 0 invalid + 0 intra-batch).

| config | seeds | fits | acc@64 | acc@256 | acc@512 | acc@1024 | fit-only @512/@1024 | per-seed wall |
|---|---|---|---|---|---|---|---|---|
| givens@64, triton scan (`hetero-gpu36-sg8`) | 36 | 25/36 | 0.929 | 0.857 | 0.750 | 0.571 | 0.889 / 0.679 | 15.3 s |
| delta@64-matched, compiled (`hetero-gpu36-pd64`) | 36 | 10/36 | 0.556 | 0.465 | 0.421 | 0.340 | 0.923 / 0.699 | 17.9 s |
| delta@1024, compiled (`hetero-gpu36-pd1024`) | 36 | 35/36 | 0.984 | 0.971 | 0.828 | 0.561 | 0.843 / 0.570 | 17.2 s |

Threshold-robustness at $\{0.98, 0.99, 0.995\}$: givens $25/25/24$, delta@64 $10/10/10$, delta@1024 $35/35/35$ — every ordering below is threshold-stable. Composer parameter counts unchanged from the matched-state round (14,624 / 3,306 / 25,480; delta@64 remains param-unmatched low). Per-seed wall costs are job-observed means (sidecar); the arms' max values (pd64 107.9 s, pd1024 48.0 s) are the one-time `torch.compile` warmups on each arm's first seeds; steady-state is ~15–18 s/seed for all three arms.

Within-stratum contrasts (two-sided Fisher exact, 36v36, computed from rows by `scripts/_evidence_stats.py`):

1. **Matched 64-element state: givens is more reliable, now decisively.** $25/36$ vs $10/36$, $p = 0.00084$. The pinned-CPU round's 12v12 tie ($8/12$ vs $4/12$, $p = 0.22$) was a power limitation, not an absence of effect — at $n=36$ (in this stratum) the gap is large and significant. The mechanisms also fail differently: delta@64's non-fits sit at chance plateaus (val@128 $0.22$–$0.43$, one $0.59$, one $0.93$), while givens' non-fits include many near-misses ($0.82$–$0.95$ for 6 of 11) — the delta failure mode is not-finding-the-solution-at-all; the givens failure mode is often almost-finding-it.
2. **Native-state delta beats givens: $35/36$ vs $25/36$, $p = 0.0030$.** The delta mechanism's reliability at its natural $16\times$ state extends from the CPU stratum's $12/12$ to $35/36$ at triple the seed count.
3. **State size dominates within delta: $10/36$ vs $35/36$, $p \approx 3.9\times10^{-10}$.**
4. Fit-only length generalization repeats the recorded pattern: mechanism-independent at matched state ($0.699$ delta vs $0.679$ givens at $T=1024$), and the delta@1024 cohort — nearly every seed fitting — generalizes worse at length ($0.570$).

**Package-decision reading.** On GPU, per-seed cost is neutral across all three arms (~15–18 s) — the CPU stratum's 16× cost argument against Givens disappears when Givens runs its Triton scan path, and delta's compile path is no cheaper than the Givens kernel path at these shapes. What differentiates the mixers on GPU is reliability and state economics: `mixer="delta"` at its native state is the most reliable composer measured ($35/36$, threshold-stable) at cost parity; `GivensMinGRU` is the decisively better composer at small matched state ($25/36$ vs $10/36$) and its fit cohort (like small-delta's) keeps the better long-extrapolation profile than the large-state delta cohort ($0.68$–$0.70$ vs $0.57$ at $16\times$ length). For the package this reads: keep BOTH mixers first-class — delta as the default recommendation when state size is free to grow (most reliable per training run, flat cost in state), Givens as the recommendation when per-token state must stay small or extreme-length fit-only generalization matters — and the docs' guidance should be conditioned on those two axes rather than a single winner. (Promotion wording and any Givens-deep-dive decision remain the user's call per the recorded memories.) Job hygiene note: the campaign command was foreground-only (no keepalive) and the job terminated normally — `job.wait()` returned without intervention, closing the loop on the first GPU round's orphaned-heartbeat incident. Artifacts: 108 per-seed rows in `experiments/lab_results.jsonl`, env sidecar `experiments/bench/gpu36_env.json`.

## Round: DeltaMinGRU chunked-WY Triton kernel — parity-proven, narrow win region, two negative results (`gpu-delta-kernel-01`)

This round deliberately overrode the `gpu-delta-probe-01` "don't build" verdict on an explicit user decision ("I want to next tackle the Triton kernel for the delta mixer. Compiling isn't always an option for users.", 2026-07-18): the kernel's value proposition was never to beat `torch.compile`, but to give eager-only users a compile-class option. The frozen intent ledger for the round is `.claude/output/intent/2026-07-18-delta-triton-kernel-intent.md` (with 2026-07-19 user-decision addenda). The result is nuanced and is recorded in full here so no future session re-litigates or rebuilds blind: a parity-proven kernel that ships, a speed bar judged **FAIL on every probed shape under both backend configurations tried**, one real (memory-plus-latency) win region narrow enough that `auto` gates the kernel to it, and two decisive negative results (a fused-Triton backward, and an A100 re-measurement) that locate the failure in the *design against modern baselines*, not in one GPU.

**What was built.** All kernel code lives in `src/mingru/triton_scans.py` (no new module, per user). The forward is a trio of `@triton.jit` kernels driving the two-stage WY decomposition of `DeltaMinGRU._forward_chunked`: a pre-pass (`_delta_prepass_kernel`, the within-chunk UT-transform solve), a sequential state pass over chunks (`_delta_state_kernel`, carrying the inter-chunk `H`), and a readout (`_delta_readout_kernel`). The backward is a hand-derived `torch.autograd.Function` (`_DeltaScanFn`) whose target is the eager-exact reverse-chunk loop `_delta_backward_torch` (torch-composed, called directly — this is the *shipping* backward). The path joins the `MINGRU_SCAN` contract exactly like the four scan ops and the angle-fused path: `auto` engages the kernel on CUDA only inside a measured win region (silent eager fallback otherwise), explicit `MINGRU_SCAN=triton` drives the full kernel envelope fail-loud, `MINGRU_SCAN=eager` is unchanged, and the eager `_forward_chunked` / `step` / `_coeffs` math is byte-untouched (verified by `scripts/check_frozen_ast.py`; `DeltaMinGRU` is not on the frozen surface, but its numeric methods were held constant). The kernel's own envelope is $d_k = d_v \in \{4,8,16,32,64\}$, $nh \cdot \text{chunk\_size} \le 128$, $\text{chunk\_size} \le 64$, fp32, and covers ragged tail chunks and nonzero `h_0`. Probe/parity machinery: `scripts/gpu_delta_probe.py` (submitted via `scripts/gpu_check.py --job delta-probe`) times eager / matmul-FLOP floor / `torch.compile` / triton per shape; `scripts/gpu_check.py --job check` runs the forward+grad parity harness; `scripts/delta_smem_probe.py` is the measured shared-memory ground-truth loop used in the fused-backward campaign.

**Parity (PASS).** The final conformance run was Lightning job `mingru-gpu-check-5955b15` (commit `5955b15`, the shipping configuration), which completed **exit 0** with **746 parity rows** all passing in-job (forward and gradient conformance of the Triton path against the eager chunked oracle, across the envelope including ragged chunks and nonzero `h_0`). Tolerance rule: forward `atol` = $\max(10\times$ the fp32-vs-fp64 reference deviation, $10^{-5})$; gradient `atol` = $\max(10\times$ the fp64-ref deviation, $10^{-3})$. Caveat: the parity artifact was **not retrieved to a local file** — the evidence is the job's exit-0 completion and its in-job row count, cited here, not a committed conformance table. All existing delta oracles (`step()` equivalence, `_ref_delta_forward`, chunk-invariance, eager fp64 gradcheck, bridge selftest) and the 252-test CPU suite remain green on `feat/delta-mixer @ 5955b15`.

**Speed bar: FAIL on every shape (shipping torch-composed backward).** Authoritative artifact `experiments/bench/gpu_delta_probe.{json,md}`, regenerated on the shipping configuration at `5955b15` (L4 stratum: torch 2.8.0+cu128, NVIDIA L4 capability [8,9], triton 3.4.0, TF32 matmul off / `float32_matmul_precision=highest`, $B=128$, warmup 3 + median of 10, generated 2026-07-19T11:31:55Z). `pd1024` = `n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64` (1024-element state); `stepup` = same but `d_k=d_v=64` (16384-element state). The spec §9.1 bar (triton fwd+bwd median $\le 1.2\times$ compile **and** $\le$ eager, both judged on this run's own medians) is judged per shape:

| config | T | eager (s) | floor (s, approx) | compile (s) | triton (s) | headroom (eager/floor) | compile-recovered | triton/compile | triton/eager | bar |
|---|---|---|---|---|---|---|---|---|---|---|
| pd1024 | 64 | 0.0078 | 0.0029 | 0.0044 | 0.0084 | 2.75x | 68.57% | 1.90 | 1.07 | FAIL |
| pd1024 | 256 | 0.0276 | 0.0126 | 0.0154 | 0.0311 | 2.20x | 81.16% | 2.02 | 1.12 | FAIL |
| pd1024 | 1024 | 0.2100 | 0.0521 | 0.0690 | 0.1273 | 4.03x | 89.31% | 1.85 | 0.61 | FAIL |
| stepup | 256 | 0.0850 | 0.0256 | 0.0442 | 0.1677 | 3.31x | 68.68% | 3.79 | 1.97 | FAIL |
| stepup | 1024 | 0.6990 | 0.1053 | 0.1961 | 0.7119 | 6.64x | 84.69% | 3.63 | 1.02 | FAIL |

The bar fails on all five shapes: the kernel is $1.85$–$3.79\times$ slower than `torch.compile` everywhere, and beats eager (triton/eager $< 1$) only at `pd1024`/$T{=}1024$ (0.61), with near-parity at `stepup`/$T{=}1024$ (1.02). The floor is deliberately optimistic (matmul/solve-only, no launch or reduction overhead, and a rougher $3\times$ backward convention for the `solve_triangular` component — see each row's `floor_method` in the JSON); `torch.compile` already recovers 68–89% of the headroom over that floor with zero engineering. There is one genuine bright spot the win-region gate is built on: at `pd1024`/$T{=}1024$ the kernel is both faster than eager (0.61x) and much lighter on memory (0.30x, below).

**Memory bar: PASS 4/5 (shipping).** Spec §7 invariant (triton peak training memory $\le$ eager peak), same run:

| config | T | eager peak (MB) | triton peak (MB) | triton/eager | bar |
|---|---|---|---|---|---|
| pd1024 | 64 | 186.9 | 232.8 | 1.25 | FAIL |
| pd1024 | 256 | 677.7 | 326.9 | 0.48 | PASS |
| pd1024 | 1024 | 2663.8 | 789.9 | 0.30 | PASS |
| stepup | 256 | 1467.2 | 813.4 | 0.55 | PASS |
| stepup | 1024 | 5812.6 | 3147.6 | 0.54 | PASS |

Memory passes on four of five shapes ($0.30$–$0.55\times$ eager, i.e. roughly $2$–$3\times$ savings), failing only at `pd1024`/$T{=}64$ (1.25x) — the short-sequence shape the `auto` gate excludes anyway (below), so the shipped default never hits the one memory-losing case.

**Negative result 1 — fused Triton backward (built, measured, reverted).** After the shipping backward missed the bar, a fully fused Triton backward trio (mirroring the forward's chunk-parallel structure) was built and tuned across **six shared-memory iterations** (`num_stages`, feature-blocking, a grad-kernel B3 split into `_delta_bwd_grad_a_kernel`/`_delta_bwd_grad_b_kernel`, and G-dot blocking), with a **measured ground-truth SMEM loop** (`scripts/delta_smem_probe.py`) rather than estimated footprints. On L4 it reached **15/15 envelope-class engagement with zero fallbacks** under the $101{,}376$-byte (~101 KB) opt-in shared-memory limit and full parity green (artifact `experiments/bench/gpu_delta_smem_l4.jsonl`, terminal `MINGRU_SMEM_DONE {"classes": 15, "engaged": 15, "fallbacks": []}`). But engagement was bought at the cost of speed: fitting the $\approx 101$ KB grad part into SMEM forced tiling that destroyed `tl.dot` efficiency, and the fused backward measured **8–12$\times$ slower than compile** (triton/compile $8.02$–$11.98$; triton/eager $2.61$–$6.17$; still FAIL every shape). That measurement is preserved as a **provenance-marked, transcript-recovered snapshot** in `experiments/bench/gpu_delta_probe_fused_l4.md` (verbatim copy of the probe `.md` as generated 2026-07-19T05:36:38Z at the fused configuration, commit lineage `7de1be3`; the corresponding JSON was not preserved — treat these as recovered-from-read, not re-measured):

| config | T | eager (s) | compile (s) | triton (s) | triton/compile | triton/eager | bar |
|---|---|---|---|---|---|---|---|
| pd1024 | 64 | 0.0063 | 0.0037 | 0.0358 | 9.77 | 5.69 | FAIL |
| pd1024 | 256 | 0.0270 | 0.0152 | 0.1376 | 9.06 | 5.10 | FAIL |
| pd1024 | 1024 | 0.2100 | 0.0685 | 0.5490 | 8.02 | 2.61 | FAIL |
| stepup | 256 | 0.0842 | 0.0434 | 0.5198 | 11.98 | 6.17 | FAIL |
| stepup | 1024 | 0.6950 | 0.1920 | 2.0904 | 10.89 | 3.01 | FAIL |

On the user's second stop-rule decision (2026-07-19) the fused backward was **reverted** at `5955b15` back to the torch-composed reverse-chunk loop, which is what ships. (The fused snapshot's memory was actually *better* than the shipping backward — $0.38$–$0.74\times$ eager — but latency, not memory, was decisive.)

**Negative result 2 — A100 re-measurement (refutes "L4-only failure").** To test whether the slowness was L4-specific, the user requested an A100 run of the fast (unblocked) fused design on a throwaway branch (`probe/a100-fused-fast @ 5c7bd71`). Artifact `experiments/bench/gpu_delta_probe_a100.{json,md}` (A100-SXM4-80GB capability [8,0], torch 2.8.0+cu128, triton 3.4.0, generated 2026-07-19T11:18:08Z) — a **distinct A100 stratum, never mixed with the L4 or pinned-CPU rows**:

| config | T | eager (s) | compile (s) | triton (s) | triton/compile | triton/eager | bar |
|---|---|---|---|---|---|---|---|
| pd1024 | 64 | 0.0067 | 0.0037 | 0.0457 | 12.46 | 6.77 | FAIL |
| pd1024 | 256 | 0.0176 | 0.0098 | 0.1659 | 16.92 | 9.44 | FAIL |
| pd1024 | 1024 | 0.0612 | 0.0313 | 0.6462 | 20.62 | 10.56 | FAIL |
| stepup | 256 | 0.0272 | 0.0184 | 0.2865 | 15.55 | 10.53 | FAIL |
| stepup | 1024 | 0.1469 | 0.0692 | 1.1295 | 16.32 | 7.69 | FAIL |

The A100 result is **relatively worse**, not better: triton/compile rises to $12.5$–$20.6$ and triton/eager to $6.8$–$10.6$. The A100 speeds the eager and compile baselines up by roughly $3$–$5\times$ (e.g. `stepup`/$T{=}1024$ eager $0.6990 \to 0.1469$s, compile $0.1961 \to 0.0692$s) while the SMEM-bound kernel barely moves — exactly the signature of a design that trades compute-parallelism for shared-memory residency. The A100 SMEM campaign (`experiments/bench/gpu_delta_smem_a100.jsonl`) engaged **13/15** at the A100's $166{,}912$-byte (~163 KB) opt-in limit, with two fallbacks: the `d_k=64` / `M=128` classes require $233{,}472$ and $221{,}184$ bytes (~221–233 KB), both over the A100's cap; by datasheet inference (no H100 was measured) the larger also exceeds an H100's ~$232{,}448$-byte opt-in cap while the smaller would just fit one. **Conclusion: the failure is design-vs-modern-baseline, not one GPU** — the kernel does not become competitive on bigger hardware; the baselines it must beat get faster there too.

**Shipping semantics (`5955b15`).** `auto` engages the delta kernel only in the L4-measured win region: sequence spanning at least `_DELTA_AUTO_MIN_T_CHUNKS = 16` chunks **and** head dim `_DELTA_AUTO_MAX_DK = 16`, i.e. $T \ge 16 \cdot \text{chunk\_size}$ (default $\ge 1024$) and $d_k \le 16$. The basis is this round's probe: that region is precisely the `pd1024`/$T{=}1024$ cell — the only shape where the kernel beat eager (0.61x) and where it also saves $\approx 3.3\times$ memory (0.30x). Every other in-envelope shape stays eager under `auto`, silently, by design (not a degradation); the `pd1024`/$T{=}64$ memory-losing shape is excluded by the same $T$ gate. Explicit `MINGRU_SCAN=triton` ignores the region and drives the full envelope fail-loud; `MINGRU_SCAN=eager` is untouched. The gate's basis is documented inline at `src/mingru/min_gru.py` (~L119–138) and the reversal/rebuild-blind warning at the `triton_scans.py` module docstring.

**Speed-bar verdict, stated plainly.** Intent statement 2 ("compile-class speed without compile") is judged **FAIL / unreachable on this hardware class** after six measured iterations. The claim "compile-class without compile" was **not** achieved and is not made anywhere in the shipped docs. `torch.compile` remains the recommended CUDA path for `mixer="delta"` when it is available (it recovers 68–89% of the fusion headroom over an optimistic floor). What the kernel does deliver, and why it ships gated rather than being deleted: for **eager-only users who cannot use `torch.compile`**, at long sequences with narrow head dims it is both faster than eager (0.61x at the win-region shape) and $2$–$3\times$ lighter on peak training memory — a real, if narrow, benefit that `auto` now routes to automatically. What remains open: no comparison against the incumbents' released tuned CUDA kernels was measured; the kernel question could reopen if a fused backward that keeps `tl.dot` efficiency without SMEM-thrashing is found, but this round's evidence is that the straightforward fusion does not clear the bar on L4 or A100.

Artifacts: `experiments/bench/gpu_delta_probe.{json,md}` (authoritative shipping config, L4), `experiments/bench/gpu_delta_probe_fused_l4.md` (provenance-marked fused-backward snapshot, transcript-recovered), `experiments/bench/gpu_delta_probe_a100.{json,md}` (A100 stratum), `experiments/bench/gpu_delta_smem_l4.jsonl` + `experiments/bench/gpu_delta_smem_a100.jsonl` (SMEM engagement ground-truth). Commits: forward trio `867d300..4894226`, fused-backward experiment `5e209fc..7de1be3`, revert + shipping semantics `5955b15`; A100 exploration on throwaway branch `probe/a100-fused-fast @ 5c7bd71`. Parity job `mingru-gpu-check-5955b15` (exit 0, 746 rows). Intent ledger `.claude/output/intent/2026-07-18-delta-triton-kernel-intent.md`.

## Round: accepted-benchmark validation on public tasks (`bench-s5-02` / `bench-mqar-02` / `bench-psmnist-02` / `bench-pendulum-02`)

**Purpose (user-framed): validation, not competition.** Intent statement 1 — the round produces a validation claim ("the mixer family is validated on accepted public benchmarks") backed by fit-rate evidence; published numbers are context, not a leaderboard entry. The packaged mixers are put on four accepted public benchmarks — S5 symmetric-group word problems, MQAR (multi-query associative recall), psMNIST (permuted-pixel MNIST), and an irregular-timestep pendulum-regression decay arm — to test whether they train reliably on tasks the literature already uses, with a classical `nn.GRU` control anchoring every within-family comparison in absolute terms.

**Stratum disclosure (mandatory).** Every number in this round is the **L4 stratum**: NVIDIA L4, torch 2.8.0+cu128, triton 3.4.0, `MINGRU_SCAN=triton` (each artifact's own line: `device=cuda, torch=2.8.0+cu128, scan=triton, compile=None`). This stratum is deliberately never compared to the pinned-CPU rounds (torch 2.5.1) or the A100 kernel-probe rows; no cross-stratum contrast is drawn. Nine arms run on each task: the six packaged single-stack mixers (`log`/`signed`/`rotation`/`givens`/`delta`) plus the working `rotation-hetero` stack, extended mid-matrix by the two promoted hetero stacks (`signed-givens`, `signed-delta`) and a depth-matched classical `gru` control (2-layer `nn.GRU`, $d_{model}=64$). Seed budget is tiered: 36 seeds for S5/MQAR/pendulum, 12 for psMNIST. Per-task fit bars were fixed before the matrices ran: S5 `val128` $\ge 0.99$, MQAR `val_qacc` $\ge 0.99$, psMNIST `val_acc` $\ge 0.90$, pendulum `val_mse` $\le 0.0014$. Fisher reference arm is `log` (vanilla minGRU) throughout. On the generator tasks (S5, MQAR, pendulum) a "fit" is **trainability** — the seed reached its own validation-metric bar — not length generalization; the raw/fit-only accuracy columns carry the generalization read separately, and the fit-only column conditions on the fitting seeds.

**Table A — master fit matrix** (cells are fit count at each task's fixed bar; S5/MQAR/pendulum $n=36$, psMNIST $n=12$):

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

**Cross-task synthesis.** The four tasks dissociate the mechanisms cleanly. Pendulum is a **positive control**: all nine arms including gru fit at the $0.0014$ MSE bar, so the harness trains and the decay channel is not the discriminating axis here (norm-preserving arms fit equally). MQAR **isolates the delta mechanism**: only the delta family fits ($36/36$ for delta and signed-delta), every non-delta arm including gru at $0/36$ — consistent with the Zoology recall-capacity tradeoff (arXiv:2312.04927). psMNIST is an **accumulation-ordering** task where the delta family again leads on fit rate (signed-delta $12/12$, delta $10/12$; gru $3/12$). S5 group word problems are solved by exactly one matched arm (signed-givens $1/36$); the probe round below shows a second config-corrected arm (signed-delta-nh4) also reaches it. The headline is a **two-dial mechanism story on these public tasks**: delta is the broadly dominant mechanism (recall + accumulation, and — via the probe — S5 groups at nh=4), while givens is a narrow group-composition specialist (its only matched S5 fit is the signed-givens arm). Delta's two dials are state size and nh (Householder product count). This does **not** contradict the earlier S3-hier result where givens won on richness — that finding is task-specific and stands; the read here is "two dials for two task regimes," not "delta beats givens everywhere."

### S5 symmetric-group word problems (`bench-s5-02`)

Fit bar `val128` $\ge 0.99$, 36 seeds/arm. **Table B** (transcribed from `experiments/bench/bench_s5.md`):

| arm | seeds | fits | acc@T1024 (raw/fit-only) | acc@T256 (raw/fit-only) | acc@T512 (raw/fit-only) | params |
|---|---|---|---|---|---|---|
| log | 36/36 | 0/36 | 0.010 / n/a | 0.016 / n/a | 0.012 / n/a | 98,936 |
| signed | 36/36 | 0/36 | 0.015 / n/a | 0.025 / n/a | 0.019 / n/a | 107,256 |
| rotation | 36/36 | 0/36 | 0.012 / n/a | 0.020 / n/a | 0.015 / n/a | 107,384 |
| rotation-hetero | 36/36 | 0/36 | 0.013 / n/a | 0.023 / n/a | 0.017 / n/a | 107,320 |
| givens | 36/36 | 0/36 | 0.010 / n/a | 0.015 / n/a | 0.012 / n/a | 111,544 |
| delta | 36/36 | 0/36 | 0.011 / n/a | 0.016 / n/a | 0.012 / n/a | 133,256 |
| signed-givens | 36/36 | 1/36 | 0.038 / 0.817 | 0.058 / 1.000 | 0.049 / 0.976 | 109,400 |
| signed-delta | 36/36 | 0/36 | 0.011 / n/a | 0.018 / n/a | 0.014 / n/a | 120,256 |
| gru | 36/36 | 0/36 | 0.017 / n/a | 0.024 / n/a | 0.019 / n/a | 65,400 |

Threshold-robustness ($\{0.98, 0.99, 0.995\}$): only signed-givens registers a fit ($1/1/1$); all other arms $0/0/0$ — the single ordering is threshold-stable. Two-sided Fisher exact vs `log` ($0/36$): every arm $p = 1$, including signed-givens ($1/36$ vs $0/36$, $p = 1$) — at $n=36$ a single-seed margin does not separate from the reference. All arms 36/36 present; complete.

Reading: S5 is solved at the correct config by exactly one matched arm, signed-givens ($1/36$, the continuous coupled-8D rotation+sign stack), whose one fitting seed generalizes cleanly (fit-only $1.000$ at $T256$, $0.976$ at $T512$, $0.817$ at $T1024$). The matched matrix ran signed-delta at nh=2 (deltaproduct-2), under-powered for S5 by design; the S5 design-correction probe below lifts nh to 4 and recovers $7/36$, so signed-delta's matched $0/36$ is a config artifact, not a mechanism verdict. The classical `gru` control is $0/36$ here. **This is not evidence that "GRU cannot state-track"**: the Illusion of State result (arXiv:2404.08819) establishes that nonlinear RNNs like GRU can track state; the $0/36$ is a same-budget/same-config outcome under a delta-calibrated training budget (seed-matched, $d_{model}=64$), not a fundamental capability limit. Fit here is trainability at this budget, not a generalization claim.

### MQAR multi-query associative recall (`bench-mqar-02`)

Fit bar `val_qacc` $\ge 0.99$, 36 seeds/arm; generalization columns are query accuracy at $T=256$ with 16 and 32 key-value pairs. **Table B** (transcribed from `experiments/bench/bench_mqar.md`):

| arm | seeds | fits | acc@T256_p16 (raw/fit-only) | acc@T256_p32 (raw/fit-only) | params |
|---|---|---|---|---|---|
| log | 36/36 | 0/36 | 0.112 / n/a | 0.081 / n/a | 91,712 |
| signed | 36/36 | 0/36 | 0.044 / n/a | 0.036 / n/a | 100,032 |
| rotation | 36/36 | 0/36 | 0.083 / n/a | 0.064 / n/a | 100,160 |
| rotation-hetero | 36/36 | 0/36 | 0.047 / n/a | 0.038 / n/a | 100,096 |
| givens | 36/36 | 0/36 | 0.030 / n/a | 0.030 / n/a | 104,320 |
| delta | 36/36 | 36/36 | 0.931 / 0.931 | 0.493 / 0.493 | 126,032 |
| signed-givens | 36/36 | 0/36 | 0.036 / n/a | 0.034 / n/a | 102,176 |
| signed-delta | 36/36 | 36/36 | 0.928 / 0.928 | 0.690 / 0.690 | 113,032 |
| gru | 36/36 | 0/36 | 0.111 / n/a | 0.076 / n/a | 58,176 |

Threshold-robustness ($\{0.98, 0.99, 0.995\}$): delta $36/36/36$, signed-delta $36/36/36$; all seven other arms $0/0/0$ — threshold-stable. Two-sided Fisher exact vs `log` ($0/36$): delta ($36/36$) $p = 2.322\times10^{-13}$, signed-delta ($36/36$) $p = 2.322\times10^{-13}$; the other six arms $p = 1$. Complete. (For delta and signed-delta the fit-only column equals raw because all 36 seeds fit.)

Reading: MQAR is a pure delta dissociation — only the delta family fits ($36/36$ delta and signed-delta), and every non-delta arm including the classical gru sits at $0/36$. The gru control is recall-limited: its raw query accuracy is $0.111$ at 16 pairs and $0.076$ at 32 pairs, near the log reference ($0.112$ / $0.081$) and far below the fit bar. This matches the Zoology recall-capacity tradeoff (arXiv:2312.04927): fixed-state recurrent models trade recall capacity against state size, and associative recall is exactly where the delta update's key-value binding pays off. Framing note: this is a mildly novel benchmark framing — no paper we found benchmarks a vanilla GRU on MQAR — so the gru row is reported as the classical anchor, not a reproduction. The delta family's raw accuracy degrades from 16 to 32 pairs (delta $0.931 \to 0.493$; signed-delta $0.928 \to 0.690$) while still clearing the validation-metric fit bar at every seed.

### psMNIST permuted-pixel MNIST (`bench-psmnist-02`)

Fit bar `val_acc` $\ge 0.90$, 12 seeds/arm. **Table B** (transcribed from `experiments/bench/bench_psmnist.md`):

| arm | seeds | fits | acc@test (raw/fit-only) | params |
|---|---|---|---|---|
| log | 12/12 | 0/12 | 0.784 / n/a | 84,234 |
| signed | 12/12 | 0/12 | 0.857 / n/a | 92,554 |
| rotation | 12/12 | 0/12 | 0.571 / n/a | 92,682 |
| rotation-hetero | 12/12 | 0/12 | 0.868 / n/a | 92,618 |
| givens | 12/12 | 0/12 | 0.290 / n/a | 96,842 |
| delta | 12/12 | 10/12 | 0.905 / 0.908 | 118,554 |
| signed-givens | 12/12 | 0/12 | 0.651 / n/a | 94,698 |
| signed-delta | 12/12 | 12/12 | 0.924 / 0.924 | 105,554 |
| gru | 12/12 | 3/12 | 0.885 / 0.897 | 38,474 |

Threshold-robustness ($\{0.88, 0.90, 0.92\}$): signed-delta $12/12/12$, delta $12/10/2$, gru $11/3/0$, rotation-hetero $3/0/0$; all other arms $0/0/0$. The delta, gru, and rotation-hetero counts move with threshold (delta $10/12$ at the $0.90$ bar falls to $2/12$ at $0.92$; gru $3/12$ at $0.90$), so those orderings are threshold-sensitive; signed-delta's $12/12$ is threshold-stable across the triple. Two-sided Fisher exact vs `log` ($0/12$): signed-delta ($12/12$) $p = 7.396\times10^{-7}$, delta ($10/12$) $p = 6.73\times10^{-5}$, gru ($3/12$) $p = 0.2174$ (not separated from log at $n=12$); all other arms $p = 1$. Complete.

Reading: psMNIST is an accumulation-ordering task and the delta family leads on fit rate — signed-delta $12/12$ (best, and the only threshold-stable fitting arm), delta $10/12$, gru $3/12$. On raw test accuracy the ordering is signed-delta $0.924$ > delta $0.905$ > gru $0.885$ > rotation-hetero $0.868$ > signed $0.857$ > log $0.784$ > signed-givens $0.651$ > rotation $0.571$ > givens $0.290$ (worst). Two structure reads follow: the stacked pure-rotation configuration is the weakest region ($0.290$ givens, $0.571$ rotation), and adding a sign channel to givens *hurts* accumulation rather than helping — signed-givens ($0.651$) sits well below plain signed ($0.857$), the mirror of the S5 result where the sign channel is what makes givens work. The `gru` control lands at $0.885$ raw (fit-only $0.897$ on its 3 fitting seeds), just under the $0.90$ bar; the reference round below grounds whether that is the code path or the budget.

### Pendulum irregular-timestep regression (`bench-pendulum-02`)

Fit bar `val_mse` $\le 0.0014$, 36 seeds/arm (regression task; no length-generalization accuracy columns). **Table B** (transcribed from `experiments/bench/bench_pendulum.md`):

| arm | seeds | fits | params |
|---|---|---|---|
| log | 36/36 | 36/36 | 83,970 |
| signed | 36/36 | 36/36 | 92,290 |
| rotation | 36/36 | 36/36 | 92,354 |
| rotation-hetero | 36/36 | 36/36 | 92,226 |
| givens | 36/36 | 36/36 | 96,466 |
| delta | 36/36 | 36/36 | 118,162 |
| signed-givens | 36/36 | 36/36 | 94,306 |
| signed-delta | 36/36 | 36/36 | 105,162 |
| gru | 36/36 | 36/36 | 38,338 |

Threshold-robustness ($\{0.00175, 0.0014, 0.00112\}$): every arm $36/36$ at all three thresholds — fully threshold-stable. Two-sided Fisher exact vs `log` ($36/36$): every arm $p = 1$. Complete.

Reading: pendulum is a **positive control**, not a decay-benchmark result. All nine arms including the norm-preserving mixers and the classical gru fit at the $0.0014$ MSE bar at every seed and every robustness threshold. Because norm-preserving arms fit equally, the decay channel is not the discriminating axis on this task; the arm proves the harness trains end-to-end on an irregular-timestep regression, nothing more. The gru arm takes a $\log(1 + \Delta t)$ input feature (it has no native `dt` decay path), like the non-decay mixer arms. A decay-isolating variant that actually separates the mechanisms is a deferred follow-up.

Artifacts: per-seed rows in `experiments/lab_results.jsonl` (round tags `bench-s5-02`, `bench-mqar-02`, `bench-psmnist-02`, `bench-pendulum-02`); report tables `experiments/bench/bench_{s5,mqar,psmnist,pendulum}.{json,md}` (regenerated whole by `scripts/report_benchmarks.py`, never hand-edited); env sidecars `experiments/bench/bench_{s5,mqar,psmnist,pendulum}_env.json`. Campaign commit lineage (matrix assembled across shards): `2f845a5` (initial six-arm shards) → `4e0e75e` (signed-givens/signed-delta) → `7cfda8b` (gru shards + probe rotation-hetero-k5) → `9796a8e` (gru-large + probe signed-delta-nh3/nh4); env sidecars record the most-recent per-task commit (s5/psmnist `9796a8e`; mqar/pendulum `7cfda8b`). Report tables generated at `e919442` (the report host self-reports this commit in each artifact's `Env:` line) and committed at `7eeb7ff`. Intent ledger `.claude/output/intent/2026-07-19-benchmark-round-intent.md`.

## Round: S5 design-correction probe (`bench-s5-probe-01`)

**Purpose (amendment 2026-07-20, S5 design-correction probe).** The matched S5 comparison handicapped two arms by config: `rotation-hetero` snapped to element orders $(2,3,4,6)$, missing S5's order-5, and `signed-delta` ran at nh=2 (a low deltaproduct count). This probe re-runs three config-corrected arms on S5 only, to separate genuine mechanism limits from experiment-design artifacts before the round's S5 conclusions are trusted. It is a descriptive design-correction population (`experiments.benchmark_lab.PROBE_ARMS`), **not** part of the matched `bench-s5-02` nine-arm accounting, and it carries **no Fisher-exact contrast** (no competing-arm-vs-`log` judgment is made).

**Stratum.** L4 (torch 2.8.0+cu128, triton 3.4.0, `MINGRU_SCAN=triton`; artifact line `device=cuda, torch=2.8.0+cu128, scan=triton, compile=None`); same stratum as the matched round, never mixed with pinned-CPU or A100 rows.

**Table C** (transcribed from `experiments/bench/bench_s5_probe.md`):

| arm | seeds | fits | acc@T1024 (raw/fit-only) | acc@T256 (raw/fit-only) | acc@T512 (raw/fit-only) | params |
|---|---|---|---|---|---|---|
| rotation-hetero-k5 | 36/36 | 0/36 | 0.013 / n/a | 0.023 / n/a | 0.016 / n/a | 107,320 |
| signed-delta-nh3 | 36/36 | 0/36 | 0.014 / n/a | 0.030 / n/a | 0.020 / n/a | 128,836 |
| signed-delta-nh4 | 36/36 | 7/36 | 0.144 / 0.618 | 0.247 / 0.992 | 0.211 / 0.892 | 137,416 |

Threshold-robustness ($\{0.98, 0.99, 0.995\}$): rotation-hetero-k5 $0/0/0$, signed-delta-nh3 $0/0/0$, signed-delta-nh4 $8/7/7$. All three arms 36/36 present; complete.

Reading: adding S5's order-5 to the rotation snap grid (`rotation-hetero-k5`, `snap=(2,3,4,5,6)`) does not rescue rotation — still $0/36$, so that family's matched $0/36$ is a genuine mechanism limit, not the missing snap order. Raising the delta product count does: signed-delta at nh=3 is still $0/36$, but nh=4 reaches $7/36$ (threshold-stable at $7/36$ for the $0.99$ and $0.995$ bars, $8/36$ at $0.98$), with clean fit-only generalization on the fitting seeds ($0.992$ at $T256$, $0.892$ at $T512$, $0.618$ at $T1024$). So S5 is solvable by the delta mechanism once nh is large enough, and the matched matrix's nh=2 signed-delta $0/36$ was under-powered by design. Net across matched + probe evidence: S5 has two solving arms — the continuous signed-givens ($1/36$ matched) and the Householder signed-delta-nh4 ($7/36$ probe) — confirming both dials reach the S5 group structure.

Artifacts: per-seed rows in `experiments/lab_results.jsonl` (round tag `bench-s5-probe-01`, `experiments.benchmark_lab.PROBE_ARMS`); report `experiments/bench/bench_s5_probe.{json,md}` (`scripts/report_benchmarks.py`, no Fisher). Commit lineage: probe rotation-hetero-k5 at `7cfda8b`, signed-delta-nh3/nh4 at `9796a8e`; report generated at `e919442`, committed at `f09aeab`. Intent ledger `.claude/output/intent/2026-07-19-benchmark-round-intent.md` (amendment 2026-07-20, S5 design-correction probe).

## Round: gru-large grounding reference (`bench-psmnist-ref-01`)

**Purpose (amendment 2026-07-20, gru-large grounding reference).** The matched `gru` control (hidden 64) landed at $0.885$ raw on psMNIST, below the literature vanilla-GRU band (~92–94% at hidden 256; the $>0.98$ figures are specialized architectures / unpermuted sMNIST). This reference arm runs a hidden-256, 2-layer `nn.GRU` at a literature-scale budget (psMNIST 60 epochs) to (a) validate the GRU code path is correct and (b) ground the family results against a literature-scale GRU. It is an explicitly **NON-matched REFERENCE** arm (`experiments.benchmark_lab.REF_ARMS`): not capacity-matched, not part of the matched `bench-psmnist-02` accounting, and it carries **no Fisher-exact contrast** — its hidden-256 / 60-epoch / $596{,}234$-param budget is a different budget stratum from the matched 30-epoch `log` reference (CLAUDE.md: strata are never mixed silently).

**Stratum.** L4 (torch 2.8.0+cu128, triton 3.4.0, `MINGRU_SCAN=triton`; artifact line `device=cuda, torch=2.8.0+cu128, scan=triton, compile=None`).

**Table D** (transcribed from `experiments/bench/bench_psmnist_ref.md`):

| arm | seeds | fits | acc@test (raw/fit-only) | params |
|---|---|---|---|---|
| gru-large | 12/12 | 12/12 | 0.922 / 0.922 | 596,234 |

Threshold-robustness ($\{0.88, 0.90, 0.92\}$): $12/12$, $12/12$, $11/12$. 12/12 present; complete.

Reading: the hidden-256 GRU reaches $0.922$ raw test accuracy and fits $12/12$ at the $0.90$ bar ($11/12$ even at $0.92$), squarely inside the literature vanilla-GRU band. This confirms the GRU code path is correct — the matched control's $0.885$ is a capacity/budget effect, not a bug — and grounds the family numbers: the matched signed-delta ($0.924$, $12/12$) and delta ($0.905$, $10/12$) arms match or exceed a literature-scale GRU at under a fifth of its parameter count ($105{,}554$ / $118{,}554$ vs $596{,}234$). Reported as a reference row only, never a matched competitor.

Artifacts: per-seed rows in `experiments/lab_results.jsonl` (round tag `bench-psmnist-ref-01`, `experiments.benchmark_lab.REF_ARMS`); report `experiments/bench/bench_psmnist_ref.{json,md}` (`scripts/report_benchmarks.py`, no Fisher). gru-large job `mingru-gpu-benchmarks-9796a8e` (commit `9796a8e`); report generated at `e919442`, committed at `7eeb7ff`. Intent ledger `.claude/output/intent/2026-07-19-benchmark-round-intent.md` (amendment 2026-07-20, gru-large grounding reference).

## Round: signed-rotation composer order correction (`bench-*-rotfix-01`)

**Purpose (correction round, 2026-07-22).** The accepted-benchmark round's rotation composer arm was registered `["rotation", "signed"]` (rotation-first), while its two sibling composer arms compose extract-then-compose, signed-first: `signed-givens` (`["signed", "givens"]`) and `signed-delta` (`["signed", "delta"]`). Testing the rotation composer in a different block architecture than its siblings confounded the composer comparison. This round re-runs the arm in the corrected signed-first order `["signed", "rotation"]` (renamed `rotation-hetero` to `signed-rotation`, and its S5 K=5 probe `rotation-hetero-k5` to `signed-rotation-k5`), so all three composer arms share one architecture and mirror `probes.py`'s `minGRU-hetero-sr` row. The swap is parameter-identical (both orders build the same shapes), so any difference is the block order alone. The original `bench-*-02` / `bench-s5-probe-01` `rotation-hetero(-k5)` rows are retained above as superseded history; the corrected arm writes under fresh `bench-*-rotfix-01` round tags, resolved per arm by `experiments.benchmark_tasks.BENCH_ARM_ROUND_OVERRIDES`.

**Stratum.** L4 (torch 2.8.0+cu128, triton 3.4.0, `MINGRU_SCAN=triton`; artifact line `device=cuda, torch=2.8.0+cu128, scan=triton, compile=None`), the same stratum as the matched round. Jobs `mingru-gpu-benchmarks-6765332` (matrix arm, four tasks) and `mingru-gpu-benchmarks-6765332-e2ook` (S5 probe), commit `6765332`.

**Corrected `signed-rotation` results** (transcribed from the regenerated `experiments/bench/bench_{s5,mqar,psmnist,pendulum}.md`, sourced from `bench-*-rotfix-01`; the composite matrix reports disclose the mixed provenance via an `arm_round_overrides` record and a rendered provenance line):

| task | seeds | fits | key generalization (raw) | params |
|---|---|---|---|---|
| S5 | 36/36 | 0/36 | acc@T256/T512/T1024 = 0.016 / 0.013 / 0.011 | 107,320 |
| MQAR | 36/36 | 0/36 | acc@T256_p16/p32 = 0.064 / 0.051 | 100,096 |
| psMNIST | 12/12 | 0/12 | acc@test = 0.736 | 92,618 |
| pendulum | 36/36 | 36/36 | val_mse fit metric (no post-selection sweep) | 92,226 |

Probe `signed-rotation-k5` (S5 only, `bench-s5-probe-rotfix-01`): 36/36 present, 0/36 fit, acc@T256/T512/T1024 = 0.015 / 0.012 / 0.011, 107,320 params.

**What the correction changed.** Three of four tasks are unchanged within noise: S5 and MQAR stay $0/36$ (near-chance generalization under both orders), pendulum stays $36/36$, and the S5 K=5 probe stays $0/36$. Only psMNIST moves, and it moves substantially.

**psMNIST block-order ablation** (same arm, same task, same protocol; only the composer block order differs):

| block order | round | present | fits (val_acc $\ge 0.90$) | acc@test (raw) | params |
|---|---|---|---|---|---|
| rotation-first `["rotation","signed"]` | `bench-psmnist-02` (superseded) | 12/12 | 0/12 | 0.868 | 92,618 |
| signed-first `["signed","rotation"]` | `bench-psmnist-rotfix-01` | 12/12 | 0/12 | 0.736 | 92,618 |

Reading: the corrected extract-then-compose order costs the rotation composer $0.868 \to 0.736$ raw test accuracy ($-0.132$) on psMNIST, with neither order clearing the $0.90$ fit bar. psMNIST is an accumulation-ordering task, not group composition: the block that touches the raw pixel stream first shapes the accumulation, and the richer non-diagonal rotation block on the input (rotation-first) accumulated better than routing the stream through the diagonal sign block first (signed-first). The original $0.868$ was therefore partly an artifact of the accumulation-favorable rotation-first order; the apples-to-apples number, with the arm in the same architecture as its givens/delta siblings, is $0.736$. In the corrected psMNIST raw-accuracy ordering `signed-rotation` falls from 4th to 6th: signed-delta $0.924$ > delta $0.905$ > gru $0.885$ > signed $0.857$ > log $0.784$ > signed-rotation $0.736$ > signed-givens $0.651$ > rotation $0.571$ > givens $0.290$. Net: block order is task-dependent, helping group composition (the evidenced extract-then-compose rule) and hurting accumulation.

**S5 conclusion, re-evaluated.** The matched round read S5's rotation-family $0/36$ as "a genuine mechanism limit, not the missing snap order" (the K=5 probe did not rescue it). The corrected order confirms and extends this: `signed-rotation` and `signed-rotation-k5` are still $0/36$ near-chance on S5, so the $0/36$ is not the missing snap order and not the rotation-first block-order confound either. S5's two solving arms remain the continuous `signed-givens` ($1/36$ matched) and the Householder `signed-delta-nh4` ($7/36$ probe).

Artifacts: per-seed rows in `experiments/lab_results.jsonl` (round tags `bench-{s5,mqar,psmnist,pendulum}-rotfix-01`, `bench-s5-probe-rotfix-01`; the superseded `rotation-hetero(-k5)` rows under `bench-*-02` / `bench-s5-probe-01` are retained, never rewritten). Regenerated reports `experiments/bench/bench_{s5,mqar,psmnist,pendulum,s5_probe}.{json,md}` (`scripts/report_benchmarks.py`; the composite matrix reports source `signed-rotation` from its correction round and carry an `arm_round_overrides` provenance disclosure). Per-arm round-tag override mechanism: `experiments.benchmark_tasks.BENCH_ARM_ROUND_OVERRIDES` (design spec `.claude/output/specs/2026-07-21-round-tag-override-design.md`); commit lineage `c1bc57c` (rename + composer-order swap) through `6765332` (override mechanism + integrity tests).
