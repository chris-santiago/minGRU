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

Comparative claims in the synthesis should be read against these
means; README-sourced numbers are struck from comparisons. Corrected
headline comparisons (all current-env):

- parity @1024: coupled 0.61 -> signed-tanh 0.996 (n=6); GRU 1.0.
- S3 @256: coupled L=4 0.54 -> rotation-snap 0.987 (n=8); GRU 1.0.
- S3 @1024: coupled L=4 0.33 -> rotation-snap 0.889 (n=8); GRU 1.0.

Not run (scoped out, review rec 1): Grazzi-parameterized incumbent,
DeltaNet. Signed-tanh should be presented as the Grazzi negative-
eigenvalue mechanism instantiated in minGRU until that comparison
runs — a repo improvement, not a novelty claim.
