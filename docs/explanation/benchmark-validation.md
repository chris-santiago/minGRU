# Benchmark validation on public tasks

## What this round validates

This page reports the mixer family's results on four accepted public benchmarks: S5 symmetric-group word problems, MQAR (multi-query associative recall), psMNIST (permuted-pixel MNIST), and an irregular-timestep pendulum-regression control. The claim is **validation, not competition**: the round tests whether the packaged mixers train reliably on tasks the literature already uses, with a depth-matched classical `nn.GRU` control anchoring every within-family comparison in absolute terms. The numbers are context for choosing a mixer, not a leaderboard entry.

For *how* to run these benchmarks yourself, see [Run the benchmarks](../how-to/run-the-benchmarks.md); for the mechanism reasoning behind the two dominant mixers, see the [Givens & Delta deep dive](givens-delta.md).

### Stratum disclosure

Every number on this page is the **L4 stratum**: NVIDIA L4, torch 2.8.0+cu128, triton 3.4.0, `MINGRU_SCAN=triton` (each artifact's own env line: `device=cuda, torch=2.8.0+cu128, scan=triton, compile=None`). This stratum is deliberately never compared to the pinned-CPU rounds (torch 2.5.1) behind the README's word-problem tables, nor to the A100 kernel-probe rows. No cross-stratum contrast is drawn here.

### How to read a "fit"

Nine arms run on each task: the six packaged single-stack mixers (`log`, `signed`, `rotation`, `rotation-hetero`, `givens`, `delta`), the two promoted heterogeneous stacks (`signed-givens`, `signed-delta`), and a depth-matched classical `gru` control (2-layer `nn.GRU`, $d_{model}=64$). Seed budget is tiered: 36 seeds for S5/MQAR/pendulum, 12 for psMNIST. Each task's fit bar was fixed before the matrices ran: S5 `val128` $\ge 0.99$, MQAR `val_qacc` $\ge 0.99$, psMNIST `val_acc` $\ge 0.90$, pendulum `val_mse` $\le 0.0014$. The Fisher reference arm is `log` throughout.

On the generator tasks (S5, MQAR, pendulum) a "fit" is **trainability** — the seed reached its own validation-metric bar — not length generalization. The raw and fit-only accuracy columns carry the generalization read separately: raw pools all seeds, fit-only conditions on the fitting seeds. Both are always reported together.

## Master fit matrix

Cells are the fit count at each task's fixed bar (S5/MQAR/pendulum $n=36$, psMNIST $n=12$):

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

The four tasks dissociate the mechanisms cleanly. Pendulum is a positive control (every arm fits). MQAR isolates the delta mechanism (only the delta family fits). psMNIST is an accumulation-ordering task where the delta family leads on fit rate. S5 group word problems are solved at the matched config by exactly one arm, `signed-givens`; the design-correction probe below recovers a second, config-corrected delta arm. Read the per-task detail before the [cross-task synthesis](#cross-task-reading).

## Per-task detail

### S5 symmetric-group word problems

Fit bar `val128` $\ge 0.99$, 36 seeds/arm. Generalization columns are word-problem accuracy at $T \in \{256, 512, 1024\}$ (raw / fit-only), transcribed from `experiments/bench/bench_s5.md`:

| arm | seeds | fits | acc@T256 (raw/fit-only) | acc@T512 (raw/fit-only) | acc@T1024 (raw/fit-only) | params |
|---|---|---|---|---|---|---|
| log | 36/36 | 0/36 | 0.016 / n/a | 0.012 / n/a | 0.010 / n/a | 98,936 |
| signed | 36/36 | 0/36 | 0.025 / n/a | 0.019 / n/a | 0.015 / n/a | 107,256 |
| rotation | 36/36 | 0/36 | 0.020 / n/a | 0.015 / n/a | 0.012 / n/a | 107,384 |
| rotation-hetero | 36/36 | 0/36 | 0.023 / n/a | 0.017 / n/a | 0.013 / n/a | 107,320 |
| givens | 36/36 | 0/36 | 0.015 / n/a | 0.012 / n/a | 0.010 / n/a | 111,544 |
| delta | 36/36 | 0/36 | 0.016 / n/a | 0.012 / n/a | 0.011 / n/a | 133,256 |
| signed-givens | 36/36 | 1/36 | 0.058 / 1.000 | 0.049 / 0.976 | 0.038 / 0.817 | 109,400 |
| signed-delta | 36/36 | 0/36 | 0.018 / n/a | 0.014 / n/a | 0.011 / n/a | 120,256 |
| gru | 36/36 | 0/36 | 0.024 / n/a | 0.019 / n/a | 0.017 / n/a | 65,400 |

S5 is solved at the correct config by exactly one matched arm, `signed-givens` ($1/36$, the continuous coupled-8D rotation+sign stack), whose one fitting seed generalizes cleanly (fit-only $1.000$ at $T256$, $0.976$ at $T512$, $0.817$ at $T1024$). Threshold-robustness across $\{0.98, 0.99, 0.995\}$: only `signed-givens` registers a fit ($1/1/1$); every other arm is $0/0/0$, so the single ordering is threshold-stable. Two-sided Fisher exact vs `log` ($0/36$): every arm $p = 1$, including `signed-givens` ($1/36$ vs $0/36$). At $n=36$ a single-seed margin does not separate from the reference. The matched matrix ran `signed-delta` at $nh=2$, under-powered for S5 by design; the [design-correction probe](#s5-design-correction-probe) below lifts $nh$ to 4 and recovers $7/36$.

The classical `gru` control is $0/36$ here. **This is not evidence that a GRU cannot state-track.** The Illusion of State result (arXiv:2404.08819) establishes that nonlinear RNNs like the GRU *can* track state; the $0/36$ is a same-budget, same-config outcome under a delta-calibrated training budget (seed-matched, $d_{model}=64$), not a fundamental capability limit. Fit here is trainability at this budget, not a generalization claim.

### MQAR multi-query associative recall

Fit bar `val_qacc` $\ge 0.99$, 36 seeds/arm. Generalization columns are query accuracy at $T=256$ with 16 and 32 key-value pairs (raw / fit-only), transcribed from `experiments/bench/bench_mqar.md`:

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

MQAR is a pure delta dissociation: only the delta family fits ($36/36$ for `delta` and `signed-delta`), and every non-delta arm including the classical `gru` sits at $0/36$. Threshold-robustness across $\{0.98, 0.99, 0.995\}$: `delta` and `signed-delta` are $36/36$ at all three; the other seven arms $0/0/0$. Two-sided Fisher exact vs `log` ($0/36$): `delta` and `signed-delta` each $p = 2.322\times10^{-13}$; the other six arms $p = 1$. (For the two delta arms the fit-only column equals raw because all 36 seeds fit.)

This matches the Zoology recall-capacity tradeoff (arXiv:2312.04927): fixed-state recurrent models trade recall capacity against state size, and associative recall is exactly where the delta update's key-value binding pays off. The `gru` control is recall-limited here: its raw query accuracy ($0.111$ at 16 pairs, $0.076$ at 32) sits near the `log` reference ($0.112$ / $0.081$) and far below the fit bar. No paper we found benchmarks a vanilla GRU on MQAR, so the `gru` row is the classical anchor, not a reproduction. The delta family's raw accuracy degrades from 16 to 32 pairs ($0.931 \to 0.493$ for `delta`; $0.928 \to 0.690$ for `signed-delta`) while still clearing the validation-metric fit bar at every seed.

### psMNIST permuted-pixel MNIST

Fit bar `val_acc` $\ge 0.90$, 12 seeds/arm. Generalization column is test accuracy (raw / fit-only), transcribed from `experiments/bench/bench_psmnist.md`:

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

psMNIST is an accumulation-ordering task and the delta family leads on fit rate: `signed-delta` $12/12$ (best, and the only threshold-stable fitting arm), `delta` $10/12$, `gru` $3/12$. On raw test accuracy the ordering is `signed-delta` $0.924$ > `delta` $0.905$ > `gru` $0.885$ > `rotation-hetero` $0.868$ > `signed` $0.857$ > `log` $0.784$ > `signed-givens` $0.651$ > `rotation` $0.571$ > `givens` $0.290$. Threshold-robustness across $\{0.88, 0.90, 0.92\}$: `signed-delta` $12/12/12$ (stable), `delta` $12/10/2$, `gru` $11/3/0$, `rotation-hetero` $3/0/0$; all others $0/0/0$. So the delta, gru, and rotation-hetero orderings are threshold-sensitive while `signed-delta`'s is stable. Two-sided Fisher exact vs `log` ($0/12$): `signed-delta` ($12/12$) $p = 7.396\times10^{-7}$, `delta` ($10/12$) $p = 6.73\times10^{-5}$, `gru` ($3/12$) $p = 0.2174$ (not separated from `log` at $n=12$); all other arms $p = 1$.

Two structure reads follow. The stacked pure-rotation configuration is the weakest region ($0.290$ givens, $0.571$ rotation), and adding a sign channel to givens *hurts* accumulation rather than helping: `signed-givens` ($0.651$) sits well below plain `signed` ($0.857$), the mirror of the S5 result where the sign channel is what makes givens work. The `gru` control lands at $0.885$ raw (fit-only $0.897$ on its 3 fitting seeds), just under the $0.90$ bar; the [gru-large reference](#gru-large-grounding-reference) below grounds whether that is the code path or the budget.

### Pendulum irregular-timestep regression

Fit bar `val_mse` $\le 0.0014$, 36 seeds/arm (regression task; no length-generalization accuracy columns), transcribed from `experiments/bench/bench_pendulum.md`:

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

Pendulum is a **positive control**, not a decay-benchmark result. All nine arms including the norm-preserving mixers and the classical `gru` fit at the $0.0014$ MSE bar at every seed and every robustness threshold ($\{0.00175, 0.0014, 0.00112\}$, all $36/36$); Fisher exact vs `log` is $p = 1$ for every arm. Because norm-preserving arms fit equally, the decay channel is not the discriminating axis on this task: the arm proves the harness trains end-to-end on an irregular-timestep regression, nothing more. The `gru` arm takes a $\log(1 + \Delta t)$ input feature (it has no native `dt` decay path), like the non-decay mixer arms. A decay-isolating variant that actually separates the mechanisms is a deferred follow-up.

## S5 design-correction probe

The matched S5 comparison handicapped two arms by config: `rotation-hetero` snapped to element orders $(2,3,4,6)$, missing S5's order-5, and `signed-delta` ran at $nh=2$ (a low deltaproduct count). This probe re-runs three config-corrected arms on S5 only, to separate genuine mechanism limits from experiment-design artifacts. It is a **descriptive design-correction population, not part of the matched nine-arm accounting**, and it carries **no Fisher-exact contrast** (no competing-arm-vs-`log` judgment). Same L4 stratum. Transcribed from `experiments/bench/bench_s5_probe.md`:

| arm | seeds | fits | acc@T256 (raw/fit-only) | acc@T512 (raw/fit-only) | acc@T1024 (raw/fit-only) | params |
|---|---|---|---|---|---|---|
| rotation-hetero-k5 | 36/36 | 0/36 | 0.023 / n/a | 0.016 / n/a | 0.013 / n/a | 107,320 |
| signed-delta-nh3 | 36/36 | 0/36 | 0.030 / n/a | 0.020 / n/a | 0.014 / n/a | 128,836 |
| signed-delta-nh4 | 36/36 | 7/36 | 0.247 / 0.992 | 0.211 / 0.892 | 0.144 / 0.618 | 137,416 |

Adding S5's order-5 to the rotation snap grid (`rotation-hetero-k5`, `snap=(2,3,4,5,6)`) does not rescue rotation: still $0/36$, so that family's matched $0/36$ is a genuine mechanism limit, not the missing snap order. Raising the delta product count does: `signed-delta` at $nh=3$ is still $0/36$, but $nh=4$ reaches $7/36$ (threshold-stable at $7/36$ for the $0.99$ and $0.995$ bars, $8/36$ at $0.98$), with clean fit-only generalization on the fitting seeds ($0.992$ at $T256$, $0.892$ at $T512$, $0.618$ at $T1024$). A $k$-cycle needs $k-1$ reflections, and S5's 5-cycle needs four, so $nh=4$ is the Householder threshold, and the matched matrix's $nh=2$ `signed-delta` $0/36$ was under-powered by design. Net across matched and probe evidence, S5 has two solving arms: the continuous `signed-givens` ($1/36$ matched) and the Householder `signed-delta-nh4` ($7/36$ probe).

## gru-large grounding reference

The matched `gru` control (hidden 64) landed at $0.885$ raw on psMNIST, below the literature vanilla-GRU band (~92–94% at hidden 256). This reference arm runs a hidden-256, 2-layer `nn.GRU` at a literature-scale budget (psMNIST 60 epochs) to (a) validate the GRU code path and (b) ground the family results against a literature-scale GRU. It is an explicitly **non-matched reference arm**: not capacity-matched, not part of the matched accounting, and it carries **no Fisher-exact contrast**: its hidden-256 / 60-epoch / $596{,}234$-param budget is a different budget stratum from the matched 30-epoch `log` reference, and the two are never mixed silently. Same L4 stratum. Transcribed from `experiments/bench/bench_psmnist_ref.md`:

| arm | seeds | fits | acc@test (raw/fit-only) | params |
|---|---|---|---|---|
| gru-large | 12/12 | 12/12 | 0.922 / 0.922 | 596,234 |

The hidden-256 GRU reaches $0.922$ raw test accuracy and fits $12/12$ at the $0.90$ bar ($11/12$ even at $0.92$), squarely inside the literature vanilla-GRU band. This confirms the GRU code path is correct (the matched control's $0.885$ is a capacity/budget effect, not a bug) and grounds the family numbers: the matched `signed-delta` ($0.924$, $12/12$) and `delta` ($0.905$, $10/12$) arms match or exceed a literature-scale GRU at under a fifth of its parameter count ($105{,}554$ / $118{,}554$ vs $596{,}234$). Reported as a reference row only, never a matched competitor.

## Cross-task reading

The headline across the four tasks is a **two-dial mechanism story on these public tasks**, not a single winner:

- **Delta is the broadly dominant workhorse.** It fits associative recall (MQAR, only the delta family), accumulation ordering (psMNIST, `signed-delta` $12/12$ and `delta` $10/12$ leading fit rate), and, once $nh$ is large enough, group composition (S5 via `signed-delta-nh4`, $7/36$ in the probe). Its two dials are state size (recall capacity) and $nh$ (the Householder product count that sets group-composition reach).
- **Givens is a narrow group-composition specialist.** Its only matched S5 fit is the `signed-givens` arm ($1/36$), the continuous coupled-8D rotation+sign stack.
- **Pendulum is a positive control.** All nine arms fit, so the harness trains end-to-end and the decay channel is not the discriminating axis on this task.
- **The GRU path is validated by gru-large.** The hidden-256 reference reaches $0.922$ on psMNIST, confirming the matched control's sub-bar $0.885$ is capacity, not a code bug.

This two-dial reading does **not** contradict the earlier `S3-hier` result where `GivensMinGRU` won on richness at a matched small state (see the [Givens & Delta deep dive](givens-delta.md) and [Choose a mixer](../how-to/choose-a-mixer.md)). That finding is task-specific and stands. The read here is "two dials for two task regimes," not "delta beats givens everywhere."

## Full evidence

- Raw per-seed rows: `experiments/lab_results.jsonl` (round tags `bench-s5-02`, `bench-mqar-02`, `bench-psmnist-02`, `bench-pendulum-02`, `bench-s5-probe-01`, `bench-psmnist-ref-01`).
- Machine tables: `experiments/bench/bench_{s5,mqar,psmnist,pendulum}.md`, `bench_s5_probe.md`, `bench_psmnist_ref.md` (regenerated whole by `scripts/report_benchmarks.py`, never hand-edited).
- Round-by-round ledger: [`experiments/EXPERIMENTS.md`](https://github.com/chris-santiago/minGRU/blob/main/experiments/EXPERIMENTS.md).
