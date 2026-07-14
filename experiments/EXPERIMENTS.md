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

Efficiency, measured (single-process microbenchmark, min of 3, all
four configs timed in the same window; per-op FLOP arithmetic
overstated these gaps and is not quoted): fwd+bwd at the training
shape (B=128, T=64) — sequential delta16 0.54s, parallel-scan delta16
2.53s, parallel-scan givens8 1.42s, parallel-scan deltamini 0.35s.
The Givens scan is cheaper than the delta16 scan but still slower
than the sequential delta on CPU; the small-state deltamini scan is
the one parallel config that beats the sequential path outright. A
promotion case for GivensMinGRU rests on the parallel-only design
constraint plus reliability and length generalization, not on CPU
cost. Exactness at length remains unique to the snapped composer's
rare winner (0.983 @1024, 1/6 with a near-zero re-attainment basin);
no continuous composer reached it.
