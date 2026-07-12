# Length-generalization investigation — summary & handoff

One-document synthesis of the experiment loop (9 rounds + post-loop
verification, 2026-07-12). The chronological lab log with per-round
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

| task (train @64) | baseline (current env, 3 seeds) | best variant | result |
|---|---|---|---|
| parity @256 | 0.894 (coupled L=1) | signed-tanh L=1 | ≥0.9999 all 6 seeds |
| parity @1024 | 0.610 | signed-tanh L=1 | mean 0.996, worst 0.979 (n=6) |
| S3 @256 | 0.54 (coupled L=4) | rotation-snap L=1 | mean 0.987 (n=8) |
| S3 @1024 | 0.34 (coupled L=4) | rotation-snap L=1 | mean 0.889; exact 1.0 in 2/8 seeds |
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
  a lucky seed; current-env 3-seed mean is 0.54).
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
instantiated in minGRU — a repo improvement, not a novelty claim,
until the incumbent comparison runs (see Open work).

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
attractors of training: runs wander in and out (Round 6). Failed
runs are detectable at train time: best val@128 < 1.0 flags them
(perfectly separated good from bad seeds, n=8).

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
| 9 | success-rate quantification (fresh seeds) | S3 n=8: 0.987@256, 0.889@1024, exact 2/8; parity n=6: 0.996@1024. val@128<1.0 flags bad runs |
| post | mechanism probes + baseline re-grounding | See below |

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
retry-on-flag (best val < 1.0 catches every bad run).**

## Open work, prioritized

1. **Head sparsification** (only remaining repair candidate,
   readout-side): prune the head to certified channels; targets the
   anomalies (s6 paradox, parity s2 leak). Task-side machinery —
   does not block promotion.
2. **Probe gap:** readout-attribution weighting for block ranking.
3. **Incumbent grid (external review rec 1, gates novelty claims):**
   Grazzi-parameterized signed scan (near-free — a `_coeffs`
   variant), GRU already done, DeltaNet n_h∈{1,2} (expensive:
   faithful implementation is a day-plus; a sloppy one strawmans
   the incumbent — either implement carefully or cite published
   numbers as cross-paper).
4. **Promotion** (separate decision): winners into `min_gru.py`
   (need `step()` methods, docs, self-tests), `probes.py` wiring,
   README ladder/results update with current-env numbers.

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
