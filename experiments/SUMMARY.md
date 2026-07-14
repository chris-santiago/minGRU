# Length-generalization investigation — summary & handoff

One-document synthesis of the experiment loop (9 rounds + post-loop
verification, 2026-07-12; incumbent-comparison and promotion status
updated 2026-07-13; hetero training-fix loop findings added
2026-07-14). The chronological lab log with per-round
detail is `EXPERIMENTS.md`; raw per-cell results are
`lab_results.jsonl`; mechanism probe outputs are
`mechanism_results.json`. This file is the curated view: what was
built, what is known, what to do next.

## TL;DR

**Goal:** a minGRU variant with better length generalization (train
@T=64, eval @256/512/1024) while keeping parallel associative-scan
training.

**Delivered:** two variants + a selection protocol + a mechanism
certificate.

Numbers below are re-validated under the `reseed-fix` round (a
train/eval generator-seeding collision fix in `experiments/variants.py`
— seed 0 unchanged, seeds >= 1 corrected; see `EXPERIMENTS.md`).

| task (train @64) | baseline (current env, 3 seeds) | best variant | result |
|---|---|---|---|
| parity @256 | 0.866 (coupled L=1) | signed-tanh L=1 | 1.000 all 6 seeds |
| parity @1024 | 0.592 | signed-tanh L=1 | mean 0.994, worst 0.984 (n=6) |
| S3 @256 | 0.649 (coupled L=4) | rotation-snap L=1 | mean 1.000 (n=8) |
| S3 @1024 | 0.347 (coupled L=4) | rotation-snap L=1 | mean 0.958; exact 1.0 in 1/8 seeds |
| reference | GRU L=1: 1.0 everywhere (verified @1024, 3 seeds) | | |

Mechanism verified: winning S3 models contain extractable 2×2
transition matrices satisfying the D3 composition table to ~1e-4 —
the automaton itself, not a shortcut. Failed seeds contain the same
representation 5–15× less exact; measured hom error predicts the
decay horizon (~ε·T).

## Setup

- Repo: single-file minGRU (`min_gru.py`), probes in `probes.py`
  (parity = running XOR; S3 = running product in the smallest
  non-abelian group; seq2seq tagging, dense supervision, d=64,
  batch 128, Adam 3e-3, budget 1600 steps).
- Environment: torch 2.5.1, CPU (run via `uv run --with torch`).
  All numbers in this directory are from this torch version. README-sourced
  numbers are NOT comparable (S3 coupled L=4 @256: README 0.655 is
  a single-seed report; current-env, `reseed-fix`-round 3-seed mean is
  0.649 — close to the single-seed number, not a lucky-seed outlier as
  the pre-fix 3-seed mean of 0.54 had suggested; see `EXPERIMENTS.md`).
- Harness: `variants.py`. Same seeds/protocol as `probes.py`
  (seed-0 calibration cell reproduces probes.py bit-consistently);
  extra eval lengths 512/1024; results append to
  `lab_results.jsonl` (schema: round, task, variant, layers, seed,
  steps, acc{64,256,512,1024}, secs, max_steps, optional ckpt).

## The variants

**`signed-tanh`** — decouple the transition eigenvalue from the
update gate:

    a_t = tanh(Linear_s(x_t))          # was (1 - z_t) * tanh(...)
    h_t = a_t * h_{t-1} + z_t * Linear_h(x_t)

One saturation instead of two to reach a = -1; tanh's asymptote sits
exactly at the needed eigenvalue, so saturation self-calibrates.
Keeps contraction (can forget) — the general-purpose choice. NOTE:
this is the Grazzi et al. (ICLR 2025) negative-eigenvalue mechanism
instantiated in minGRU — a repo improvement, not a novelty claim.
The delta-rule incumbent comparison has since run (round
`incumbent-delta`, `EXPERIMENTS.md`): at matched d_model, signed-tanh's
parity length profile (mean 0.994 @1024) beats both DeltaNet nh=1
(0.851) and DeltaProduct nh=2 (0.810), while DeltaProduct nh=2 solves
S3 reliably (val@128 = 1.0 by step 300, all seeds) where rotation-snap
needs retries. The Grazzi-parameterized signed scan itself remains
un-run (see Open work).

**`rotation-snap`** — 2×2 block transitions on n = d/2 planar
blocks:

    M_t = R(theta_t) @ diag(1, tanh(u_t))
    theta_t snapped to grid {2*pi/K}, K cycled over (2,3,4,6),
    via straight-through estimator (forward exact, backward identity)

Non-commutative matrix scan (Hillis-Steele over 2×2 products,
`matrix_scan`), O(log T) depth. S3 ≅ D3 ⊂ O(2) is exactly
representable in one layer. Learnable nonzero h0 per block (zero
has no orbit; reflection-axis vectors collapse reflections onto
rotations). L=1 mechanism only — depth breaks STE training.

**Protocol** — full 1600-step budget + best-checkpoint selection by
val accuracy at T=128 (2 batches, seed 5; not a test length).
Required because exact solutions are reachable but NOT stable
attractors of training: runs wander in and out (Round 6). Best val@128
< 1.0 flags a run as failed and it should be retried; under the
`reseed-fix` round's clean 8-seed re-run, though, no run was flagged
this way even though 7/8 were not exact at length — the "perfectly
separated good from bad seeds" property reported for the pre-fix n=8
evidence does not replicate under clean seeding (see EXPERIMENTS.md,
`reseed-fix` round). Treat the flag as necessary, not sufficient.

## Round-by-round (hypothesis → outcome)

| round | hypothesis | outcome |
|---|---|---|
| 1 | H1 decouple a from z; H2 hardtanh boundary | **H1 confirmed** (parity perfect to 16×); H2 worse than tanh (dead-zone gradients); H3 STE-sign moot |
| 2 | H4 2×2 rotation blocks break commutativity | **Confirmed** — S3 L=1 solved in 100 steps (diagonal plateau 0.37); angles drift at length (no attractor at 2π/3) |
| 3 | H10 STE angle-snap = manufactured attractors | **Confirmed** — S3 perfect to 16× (seed 0); depth breaks snap (L=4 fails to train); harness validated vs probes.py |
| 4 | unify (K=1 blocks = tanh channels); multi-seed | Unified variant dominated by specialists; seed variance real (S3 @1024: 1.0/0.64/0.63); diagnosis: unsnapped reflection magnitudes tanh(u) |
| 5 | H13 snap u too (exact O(2) everywhere) | **Refuted** — worse everywhere. Orthogonal transitions can't forget; injection noise accumulates ~sqrt(T) vs O(1) signal |
| 6 | undertraining (train past early stop) | **Refuted** — training wanders in AND out of exact solutions; one solved seed collapsed to 0.43@64. Instability, not undertraining |
| 7 | H15 checkpoint selection val@128 | **Confirmed** — recovers exact solution wherever the run contained one (S3 2/3 seeds exact to 16×) |
| 8 | H16 grid-attraction penalty; sharper selection | **Both refuted** — penalty hurts at both strengths (over-constrains search); sharper val selects same ckpt. Failing seed never contains exact solution |
| 9 | success-rate quantification (fresh seeds) | S3 n=8: 0.987@256, 0.889@1024, exact 2/8; parity n=6: 0.996@1024. val@128<1.0 flags bad runs. **Superseded** — these seeds were drawn under the train/eval generator-seeding bug fixed in `reseed-fix`; see that round below for the corrected numbers |
| post | mechanism probes + baseline re-grounding | See below |
| reseed-fix | train/eval generator-seeding collision fix + re-validation (seed 0 unchanged) | S3 n=8: 1.000@256, 0.958@1024, exact 1/8, no runs flagged by val@128; parity n=6: 0.994@1024, worst 0.984. Full comparison in `EXPERIMENTS.md` |

## Mechanism verification (the strongest result)

Per-token transitions are input-only (embedding → LayerNorm → mixer,
no state dependence), so the learned automaton can be READ OFF THE
WEIGHTS and checked exhaustively — 36 matrix products vs the S3
composition table:

| seed | acc@1024 | best-block hom err | |det| |
|---|---|---|---|
| s0 | 1.0 | 0.00014 (3 blocks <1e-3) | 0.99992 |
| s1 | 1.0 | 0.0004 | 0.99976 |
| s2 | 0.726 | 0.0020 | 0.99870 |
| s6 | 0.616 | 0.0007 | 0.99967 |

All faithful (6 distinct matrices). Parity: 3 channels at
(a_hold, a_flip) = (+0.99999, -0.99999) in both probed seeds.

Implications:

1. **Claim upgrade**: behavioral → mechanistic. The model contains
   the automaton as an inspectable, certifiable object. ε·T roughly
   predicts the decay horizon (s2: ε=2e-3 → breaks ~T=500-1000,
   as observed). This model class is unusually auditable — the
   state-tracking core can be certified from weights, no
   long-sequence eval needed.
2. **Failure is calibration, not search**: every seed finds the
   representation; seeds differ in exactness. The repair agenda is
   polish-in-place, not better exploration.
3. **Injection surprise**: automaton blocks run with z up to 0.70
   (not ~0). Either ||b|| is small anyway (unmeasured — probe gap)
   or injections are group-structured features, not noise. The
   Round-5 "forgetting path" law likely needs restating as: only
   unstructured channels need contraction.
4. **Readout matters**: s6's best block is MORE exact than s2's yet
   its accuracy is worse; parity s2's flip channels are as deep as
   the perfect seed's yet it leaks 2% @1024. Mechanism present ≠
   mechanism exclusively used — heads also read decaying
   non-automaton channels.

## Design laws (transferable findings)

1. Don't make the eigenvalue pay two saturations to reach its
   target (drop the (1-z) coupling).
2. Length generalization needs attractors at the exact transition
   values the task requires — tanh's asymptote is one; STE snapping
   manufactures them for rotation angles. Interior target values
   with no attractor (plain rotation angles) drift, and error
   compounds with length.
3. Keep a forgetting path — but per finding 3 above, this may apply
   only to non-automaton channels; exactness should be local, not
   global.
4. Once train accuracy saturates, the loss is blind to the
   difference between exact solutions and decaying shortcuts —
   select checkpoints on a held-out moderate length (or better:
   on hom error, see next work).

## Repair agenda — CLOSED (rounds 1-2, negative)

All cheap repairs tested and retired (`repair_probes.py`,
`repair_round2.py`, results in `repair_results.json` /
`repair2_results.json`; detail in EXPERIMENTS.md):

- Inference-time projection: partial (+3..8 pts @1024); residual
  failure is readout-side. Projecting all blocks hurts even perfect
  seeds — exactness-must-be-local confirmed at inference.
- Projection + fine-tune: trades in-dist accuracy for length
  robustness; no clean recovery. Failed seeds are globally
  mis-oriented, not under-polished.
- Hom-error checkpoint selection: exact null vs best-val@128 (same
  checkpoint selected on both failed seeds). Standard best-val
  practice suffices; the certificate is a diagnostic, not a selector.
- New finding: the two perfect seeds implement DIFFERENT exact
  solutions — s0 orbit-based (injections ablatable), s1
  injection-driven (automaton-block ||b|| 10x median; ablation
  collapses it). Injections through exact transitions are signal.

**Final shipping story: variant + best-val@128 selection +
retry-on-flag (best val < 1.0 flags a run as failed and it should be
retried — though the `reseed-fix` round found this does not catch
every non-exact run under clean seeding; see "Protocol," above).**

## Hetero training-fix loop (2026-07-14, rounds `hetero-loop-01..14`)

A 14-loop hypothesize→test→refine program on `minGRU-hetero-sr`'s
S3-hier trainability (full record: `EXPERIMENTS.md` parts 1–3; driver:
`experiments/hetero_lab.py`). What it settled:

- **Coupling refuted (as far as linear probes can see).** Layer-1
  generator extraction is linearly decodable (≥0.95) by step 400–800
  on every recorded seed, including permanent plateaus —
  supervision/curriculum/identity-warmup arms target an extraction
  bottleneck the probes rule out at this budget.
- **Trainability is a composer-mechanism property.** signed →
  DeltaProduct-nh=2 (`hetero-sd2`) fits S3-hier 6/6 seeds at 1600
  steps (baseline 1/6); `deltaproduct2` L=1 alone is chance (6/6), so
  depth-buys-hierarchy survives its attribution control.
- **The reliability↔exactness trade is mechanism-level, measured from
  both sides.** Delta-composer drift (0.53→0.60 @1024 with budget,
  asymptotic) survives a five-story refutation cascade (amplitude,
  extraction drift, beta-orthogonality, input stationarity,
  within-class map variance — the last causally); the rotation
  composer's exact solution has a near-zero training basin (0/3 exact
  re-attainment from 1% weight perturbation at any tested scale).
- **Reliability rises with per-token map richness (mechanism x
  state-size factorization, loops 15-18, n=12).** At matched
  64-element state: 2D continuous rotations 1/12, 64-state delta 4/12
  with chance-plateau misses, 8D Givens rotations (`GivensMinGRU`,
  `hetero_lab.py`) 8/12 with mostly-near-fit misses. The
  within-rotation-family gradient (1/12 -> 8/12, Fisher p = 0.0094)
  comes at +2.2% total params — established, not a parameter effect.
  Within-delta state size (6/6 at 1,024 vs 4/12 at 64, p = 0.0128) —
  established. Cross-mechanism at matched state (8/12 vs 4/12,
  p = 0.22, composer params unmatched 14,624 vs 3,306) — suggestive
  only. Fit quality is mechanism-independent (fit-only @1024: 0.733
  Givens vs 0.739 small-delta); mechanisms differ in fit RATE.
  GivensMinGRU is promoted to `min_gru.py` as the selectable `givens`
  mixer under the parallel-only constraint on reliability + gen + design
  fit; measured CPU cost still favors the sequential delta path (no
  parallel-scan config beats it on CPU; microbenchmark in EXPERIMENTS.md).
- **Killed arms** (pre-registered rules): soft-warmup, Neelakantan
  gradient noise under Adam, curriculum-as-fixer, budget-2x for the
  rotation composer, VQ interface bottlenecks, distill-then-snap
  (sequential teacher in the training path), matched-capacity delta
  as a reliability route.
- **Protocol lessons in the harness:** checkpoint selection scores the
  deployment-mode model; `--ckpt-t` keeps selection discriminating
  after val@128 saturates (guarded against test-length leakage).

## Open work, prioritized

1. **Head sparsification** (only remaining repair candidate,
   readout-side): prune the head to certified channels; targets the
   anomalies (s6 paradox, parity s2 leak). Task-side machinery —
   does not block promotion.
2. **Probe gap:** readout-attribution weighting for block ranking.
3. **Incumbent grid (external review rec 1, gates novelty claims):**
   DeltaNet n_h∈{1,2} DONE (round `incumbent-delta`, 2026-07-13:
   mechanism-level `DeltaNetMixer` in `experiments/variants.py` with
   delta-rule self-tests; nh=2 solves S3 near-exactly at 1x budget,
   nh=1 cannot fit it, both lose to signed-tanh on parity length
   profile; capacity asymmetry disclosed in the README). GRU already
   done. Remaining: Grazzi-parameterized signed scan (near-free — a
   `_coeffs` variant).
4. **Promotion** DONE (2026-07-12): SignedMinGRU/RotationMinGRU and
   the mixer-selector stack in `min_gru.py` (step() methods, docs,
   self-tests), `probes.py` registry + GRID wiring, README results
   at current-env numbers. GivensMinGRU and its k×k `matrix_affine_scan`
   are likewise promoted (2026-07-14): selectable `mixer="givens"`
   (defaults `block_size=8, rounds=3`), `probes.py` row
   `minGRU-hetero-sg8` on `S3-hier` and GRID-wired, with self-tests
   covering orthogonality, scan-vs-sequential, and forward-vs-step. The
   pooled n=12 `hetero-loop-17-sg8` evidence carries onto the promoted
   path by construction bit-identity plus one exact end-to-end
   replication (round `givens-promotion-replication-01`, every fit/gen/
   ckpt metric matching the recorded seed-0 row exactly). README ships
   the Givens mechanism section with the parameter (14,624 vs 12,544 at
   matched 64-element state) and measured-CPU-cost (parallel-scan givens8
   0.961s vs sequential delta16 0.179s, uncontended fwd+bwd, B=128, T=64)
   disclosures.

## How to run

```bash
uv run --with torch python experiments/variants.py selftest   # scan checks
uv run --with torch python experiments/variants.py TASK VARIANT [L]
EXP_ROUND=x EXP_CKPT=1 uv run --with torch python experiments/variants.py screen
uv run --with torch python experiments/mechanism_probes.py
```

Provenance: the recorded numbers in `lab_results.jsonl` were produced
under torch 2.5.1. Environment matters here (the S3 coupled-L=4 cell
is seed- and env-sensitive); re-run baselines before comparing
against results from a different torch version.

- `SCREEN` list in `variants.py` holds the current batch of cells;
  edit per round. Registry: signed-{coupled,tanh,hardtanh,ste},
  rotation, rotation-snap, rotation-snap1, ortho-snap,
  rotation-snap-reg{01,001}, gru.
- Env knobs: `MAX_STEPS`, `EXP_ROUND` (jsonl tag), `EXP_CKPT`
  (+`EXP_CKPT_T`, `EXP_CKPT_B`), `EXP_NO_EARLYSTOP`.
- Gotchas: seed 0 = probes.py-identical construction order (don't
  reorder module creation in mixers or the calibration property
  breaks); `rotation*` variants have no `step()`; models are NOT
  checkpointed to disk — `mechanism_probes.py` retrains its cells
  (fast); the jsonl `ckpt.val128` key holds val at whatever
  `EXP_CKPT_T` was (192 in round 8, else 128).
