# Benchmark round: psmnist (`bench-psmnist-02`)

Fit metric: `ckpt.val_acc` >= 0.9 (robustness triple: 0.88, 0.9, 0.92). Fisher reference arm: `log`. Computed from `experiments/lab_results.jsonl` (rows matching this round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 4e0e75e6c39866bd06194023912d161d7fad66e7, generated 2026-07-20T11:38:12.474704+00:00.

Stratum(s) observed: device=cuda, torch=2.8.0+cu128, scan=triton, compile=None

## Fits and generalization accuracy (raw / fit-only)

| arm | seeds (present/planned) | fits | acc@test (raw/fit-only) | params |
| --- | --- | --- | --- | --- |
| log | 12/12 | 0/12 | 0.784 / n/a | 84,234 |
| signed | 12/12 | 0/12 | 0.857 / n/a | 92,554 |
| rotation | 12/12 | 0/12 | 0.571 / n/a | 92,682 |
| rotation-hetero | 12/12 | 0/12 | 0.868 / n/a | 92,618 |
| givens | 12/12 | 0/12 | 0.290 / n/a | 96,842 |
| delta | 12/12 | 10/12 | 0.905 / 0.908 | 118,554 |
| signed-givens | 0/12 | 0/0 | n/a / n/a | 94,698 |
  (0 rows found for arm `signed-givens`)
| signed-delta | 0/12 | 0/0 | n/a / n/a | 105,554 |
  (0 rows found for arm `signed-delta`)

## Threshold-robustness

| arm | 0.88 | 0.9 | 0.92 |
| --- | --- | --- | --- |
| log | 0/12 | 0/12 | 0/12 |
| signed | 0/12 | 0/12 | 0/12 |
| rotation | 0/12 | 0/12 | 0/12 |
| rotation-hetero | 3/12 | 0/12 | 0/12 |
| givens | 0/12 | 0/12 | 0/12 |
| delta | 12/12 | 10/12 | 2/12 |
| signed-givens | n/a | n/a | n/a |
| signed-delta | n/a | n/a | n/a |

## Two-sided Fisher exact vs `log`

- signed (0/12) vs log (0/12): p = 1
- rotation (0/12) vs log (0/12): p = 1
- rotation-hetero (0/12) vs log (0/12): p = 1
- givens (0/12) vs log (0/12): p = 1
- delta (10/12) vs log (0/12): p = 6.73e-05
- signed-givens vs log: n/a (one arm has 0 rows)
- signed-delta vs log: n/a (one arm has 0 rows)

## Completeness (present vs planned seed matrix)

- log: 12/12 present; complete
- signed: 12/12 present; complete
- rotation: 12/12 present; complete
- rotation-hetero: 12/12 present; complete
- givens: 12/12 present; complete
- delta: 12/12 present; complete
- signed-givens: 0/12 present; missing seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
- signed-delta: 0/12 present; missing seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

