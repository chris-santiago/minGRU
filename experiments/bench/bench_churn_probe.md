# Benchmark round: churn PROBE arm(s) (`bench-churn-probe-01`)

**S5 design-correction probe arm(s) -- NOT part of the matched `bench-churn-02` seed-matrix accounting.** Tests whether two suspected experiment-design artifacts -- `signed-rotation`'s missing K=5 snap order and `signed-delta`'s low nh=2 product count -- rather than a genuine mechanism limit, explain S5's 0/36 rows for those families (see `experiments.benchmark_lab.PROBE_ARMS`'s own comment). No Fisher-exact comparison is computed here: this is a descriptive design-correction population, not a competing arm judged against `log`.

Fit metric: `ckpt.val_auc` >= 0.8 (robustness triple: 0.78, 0.8, 0.82). Computed from `experiments/lab_results.jsonl` (rows matching this probe round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 4b5cdaa2b6fde7fd09937a3fa70f5b0c15ec6697, generated 2026-07-25T19:36:25.451518+00:00.

Stratum(s) observed: device=cuda, torch=2.8.0+cu128, scan=triton, compile=None

## Fits and generalization AUROC (raw / fit-only)

| arm | seeds (present/planned) | fits | AUROC@test (raw/fit-only) | params |
| --- | --- | --- | --- | --- |
| signed-rotation-k5 | 0/36 | 0/0 | n/a / n/a | 100,162 |
  (0 rows found for arm `signed-rotation-k5`)
| signed-delta-nh3 | 0/36 | 0/0 | n/a / n/a | 121,678 |
  (0 rows found for arm `signed-delta-nh3`)
| signed-delta-nh4 | 36/36 | 36/36 | 0.823 / 0.823 | 130,258 |

## Threshold-robustness

| arm | 0.78 | 0.8 | 0.82 |
| --- | --- | --- | --- |
| signed-rotation-k5 | n/a | n/a | n/a |
| signed-delta-nh3 | n/a | n/a | n/a |
| signed-delta-nh4 | 36/36 | 36/36 | 20/36 |

## Completeness (present vs planned seed matrix)

- signed-rotation-k5: 0/36 present; missing seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
- signed-delta-nh3: 0/36 present; missing seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
- signed-delta-nh4: 36/36 present; complete

