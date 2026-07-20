# Benchmark round: pendulum (`bench-pendulum-02`)

Fit metric: `ckpt.val_mse` <= 0.0014 (robustness triple: 0.00175, 0.0014, 0.00112). Fisher reference arm: `log`. Computed from `experiments/lab_results.jsonl` (rows matching this round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 4e0e75e6c39866bd06194023912d161d7fad66e7, generated 2026-07-20T11:38:12.498229+00:00.

Stratum(s) observed: device=cuda, torch=2.8.0+cu128, scan=triton, compile=None

## Fits and generalization accuracy (raw / fit-only)

| arm | seeds (present/planned) | fits | params |
| --- | --- | --- | --- |
| log | 36/36 | 36/36 | 83,970 |
| signed | 36/36 | 36/36 | 92,290 |
| rotation | 36/36 | 36/36 | 92,354 |
| rotation-hetero | 36/36 | 36/36 | 92,226 |
| givens | 36/36 | 36/36 | 96,466 |
| delta | 36/36 | 36/36 | 118,162 |
| signed-givens | 0/36 | 0/0 | 94,306 |
  (0 rows found for arm `signed-givens`)
| signed-delta | 0/36 | 0/0 | 105,162 |
  (0 rows found for arm `signed-delta`)

## Threshold-robustness

| arm | 0.00175 | 0.0014 | 0.00112 |
| --- | --- | --- | --- |
| log | 36/36 | 36/36 | 36/36 |
| signed | 36/36 | 36/36 | 36/36 |
| rotation | 36/36 | 36/36 | 36/36 |
| rotation-hetero | 36/36 | 36/36 | 36/36 |
| givens | 36/36 | 36/36 | 36/36 |
| delta | 36/36 | 36/36 | 36/36 |
| signed-givens | n/a | n/a | n/a |
| signed-delta | n/a | n/a | n/a |

## Two-sided Fisher exact vs `log`

- signed (36/36) vs log (36/36): p = 1
- rotation (36/36) vs log (36/36): p = 1
- rotation-hetero (36/36) vs log (36/36): p = 1
- givens (36/36) vs log (36/36): p = 1
- delta (36/36) vs log (36/36): p = 1
- signed-givens vs log: n/a (one arm has 0 rows)
- signed-delta vs log: n/a (one arm has 0 rows)

## Completeness (present vs planned seed matrix)

- log: 36/36 present; complete
- signed: 36/36 present; complete
- rotation: 36/36 present; complete
- rotation-hetero: 36/36 present; complete
- givens: 36/36 present; complete
- delta: 36/36 present; complete
- signed-givens: 0/36 present; missing seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
- signed-delta: 0/36 present; missing seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

