# Benchmark round: psmnist (`bench-psmnist-02`)

Fit metric: `ckpt.val_acc` >= 0.9 (robustness triple: 0.88, 0.9, 0.92). Fisher reference arm: `log`. Computed from `experiments/lab_results.jsonl` (rows matching this round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit e91944201560019e4bb24485f004bde56665af73, generated 2026-07-20T20:52:33.287492+00:00.

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
| signed-givens | 12/12 | 0/12 | 0.651 / n/a | 94,698 |
| signed-delta | 12/12 | 12/12 | 0.924 / 0.924 | 105,554 |
| gru | 12/12 | 3/12 | 0.885 / 0.897 | 38,474 |

## Threshold-robustness

| arm | 0.88 | 0.9 | 0.92 |
| --- | --- | --- | --- |
| log | 0/12 | 0/12 | 0/12 |
| signed | 0/12 | 0/12 | 0/12 |
| rotation | 0/12 | 0/12 | 0/12 |
| rotation-hetero | 3/12 | 0/12 | 0/12 |
| givens | 0/12 | 0/12 | 0/12 |
| delta | 12/12 | 10/12 | 2/12 |
| signed-givens | 0/12 | 0/12 | 0/12 |
| signed-delta | 12/12 | 12/12 | 12/12 |
| gru | 11/12 | 3/12 | 0/12 |

## Two-sided Fisher exact vs `log`

- signed (0/12) vs log (0/12): p = 1
- rotation (0/12) vs log (0/12): p = 1
- rotation-hetero (0/12) vs log (0/12): p = 1
- givens (0/12) vs log (0/12): p = 1
- delta (10/12) vs log (0/12): p = 6.73e-05
- signed-givens (0/12) vs log (0/12): p = 1
- signed-delta (12/12) vs log (0/12): p = 7.396e-07
- gru (3/12) vs log (0/12): p = 0.2174

## Completeness (present vs planned seed matrix)

- log: 12/12 present; complete
- signed: 12/12 present; complete
- rotation: 12/12 present; complete
- rotation-hetero: 12/12 present; complete
- givens: 12/12 present; complete
- delta: 12/12 present; complete
- signed-givens: 12/12 present; complete
- signed-delta: 12/12 present; complete
- gru: 12/12 present; complete

