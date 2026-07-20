# Benchmark round: mqar (`bench-mqar-02`)

Fit metric: `ckpt.val_qacc` >= 0.99 (robustness triple: 0.98, 0.99, 0.995). Fisher reference arm: `log`. Computed from `experiments/lab_results.jsonl` (rows matching this round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 4e0e75e6c39866bd06194023912d161d7fad66e7, generated 2026-07-20T11:38:12.455137+00:00.

Stratum(s) observed: device=cuda, torch=2.8.0+cu128, scan=triton, compile=None

## Fits and generalization accuracy (raw / fit-only)

| arm | seeds (present/planned) | fits | acc@T256_p16 (raw/fit-only) | acc@T256_p32 (raw/fit-only) | params |
| --- | --- | --- | --- | --- | --- |
| log | 36/36 | 0/36 | 0.112 / n/a | 0.081 / n/a | 91,712 |
| signed | 36/36 | 0/36 | 0.044 / n/a | 0.036 / n/a | 100,032 |
| rotation | 36/36 | 0/36 | 0.083 / n/a | 0.064 / n/a | 100,160 |
| rotation-hetero | 0/36 | 0/0 | n/a / n/a | n/a / n/a | 100,096 |
  (0 rows found for arm `rotation-hetero`)
| givens | 0/36 | 0/0 | n/a / n/a | n/a / n/a | 104,320 |
  (0 rows found for arm `givens`)
| delta | 0/36 | 0/0 | n/a / n/a | n/a / n/a | 126,032 |
  (0 rows found for arm `delta`)
| signed-givens | 36/36 | 0/36 | 0.036 / n/a | 0.034 / n/a | 102,176 |
| signed-delta | 36/36 | 36/36 | 0.928 / 0.928 | 0.690 / 0.690 | 113,032 |

## Threshold-robustness

| arm | 0.98 | 0.99 | 0.995 |
| --- | --- | --- | --- |
| log | 0/36 | 0/36 | 0/36 |
| signed | 0/36 | 0/36 | 0/36 |
| rotation | 0/36 | 0/36 | 0/36 |
| rotation-hetero | n/a | n/a | n/a |
| givens | n/a | n/a | n/a |
| delta | n/a | n/a | n/a |
| signed-givens | 0/36 | 0/36 | 0/36 |
| signed-delta | 36/36 | 36/36 | 36/36 |

## Two-sided Fisher exact vs `log`

- signed (0/36) vs log (0/36): p = 1
- rotation (0/36) vs log (0/36): p = 1
- rotation-hetero vs log: n/a (one arm has 0 rows)
- givens vs log: n/a (one arm has 0 rows)
- delta vs log: n/a (one arm has 0 rows)
- signed-givens (0/36) vs log (0/36): p = 1
- signed-delta (36/36) vs log (0/36): p = 2.322e-13

## Completeness (present vs planned seed matrix)

- log: 36/36 present; complete
- signed: 36/36 present; complete
- rotation: 36/36 present; complete
- rotation-hetero: 0/36 present; missing seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
- givens: 0/36 present; missing seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
- delta: 0/36 present; missing seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
- signed-givens: 36/36 present; complete
- signed-delta: 36/36 present; complete

