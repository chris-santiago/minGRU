# Benchmark round: s5 PROBE arm(s) (`bench-s5-probe-01`)

**S5 design-correction probe arm(s) -- NOT part of the matched `bench-s5-02` seed-matrix accounting.** Tests whether two suspected experiment-design artifacts -- `signed-rotation`'s missing K=5 snap order and `signed-delta`'s low nh=2 product count -- rather than a genuine mechanism limit, explain S5's 0/36 rows for those families (see `experiments.benchmark_lab.PROBE_ARMS`'s own comment). No Fisher-exact comparison is computed here: this is a descriptive design-correction population, not a competing arm judged against `log`.

Fit metric: `ckpt.val128` >= 0.99 (robustness triple: 0.98, 0.99, 0.995). Computed from `experiments/lab_results.jsonl` (rows matching this probe round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 67653323f22a5db6a2e32113ec6e18e1eb975cf5, generated 2026-07-22T14:27:14.560001+00:00.

Stratum(s) observed: device=cuda, torch=2.8.0+cu128, scan=triton, compile=None

Provenance: signed-rotation-k5 <- `bench-s5-probe-rotfix-01` (other arms: `bench-s5-probe-01`) -- this report mixes rounds; see `arm_round_overrides` in the JSON payload.

## Fits and generalization accuracy (raw / fit-only)

| arm | seeds (present/planned) | fits | acc@T1024 (raw/fit-only) | acc@T256 (raw/fit-only) | acc@T512 (raw/fit-only) | params |
| --- | --- | --- | --- | --- | --- | --- |
| signed-rotation-k5 | 36/36 | 0/36 | 0.011 / n/a | 0.015 / n/a | 0.012 / n/a | 107,320 |
| signed-delta-nh3 | 36/36 | 0/36 | 0.014 / n/a | 0.030 / n/a | 0.020 / n/a | 128,836 |
| signed-delta-nh4 | 36/36 | 7/36 | 0.144 / 0.618 | 0.247 / 0.992 | 0.211 / 0.892 | 137,416 |

## Threshold-robustness

| arm | 0.98 | 0.99 | 0.995 |
| --- | --- | --- | --- |
| signed-rotation-k5 | 0/36 | 0/36 | 0/36 |
| signed-delta-nh3 | 0/36 | 0/36 | 0/36 |
| signed-delta-nh4 | 8/36 | 7/36 | 7/36 |

## Completeness (present vs planned seed matrix)

- signed-rotation-k5: 36/36 present; complete
- signed-delta-nh3: 36/36 present; complete
- signed-delta-nh4: 36/36 present; complete

