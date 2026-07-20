# Benchmark round: pendulum (`bench-pendulum-02`)

Fit metric: `ckpt.val_mse` <= 0.0014 (robustness triple: 0.00175, 0.0014, 0.00112). Fisher reference arm: `log`. Computed from `experiments/lab_results.jsonl` (rows matching this round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit e91944201560019e4bb24485f004bde56665af73, generated 2026-07-20T20:52:33.309618+00:00.

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
| signed-givens | 36/36 | 36/36 | 94,306 |
| signed-delta | 36/36 | 36/36 | 105,162 |
| gru | 36/36 | 36/36 | 38,338 |

## Threshold-robustness

| arm | 0.00175 | 0.0014 | 0.00112 |
| --- | --- | --- | --- |
| log | 36/36 | 36/36 | 36/36 |
| signed | 36/36 | 36/36 | 36/36 |
| rotation | 36/36 | 36/36 | 36/36 |
| rotation-hetero | 36/36 | 36/36 | 36/36 |
| givens | 36/36 | 36/36 | 36/36 |
| delta | 36/36 | 36/36 | 36/36 |
| signed-givens | 36/36 | 36/36 | 36/36 |
| signed-delta | 36/36 | 36/36 | 36/36 |
| gru | 36/36 | 36/36 | 36/36 |

## Two-sided Fisher exact vs `log`

- signed (36/36) vs log (36/36): p = 1
- rotation (36/36) vs log (36/36): p = 1
- rotation-hetero (36/36) vs log (36/36): p = 1
- givens (36/36) vs log (36/36): p = 1
- delta (36/36) vs log (36/36): p = 1
- signed-givens (36/36) vs log (36/36): p = 1
- signed-delta (36/36) vs log (36/36): p = 1
- gru (36/36) vs log (36/36): p = 1

## Completeness (present vs planned seed matrix)

- log: 36/36 present; complete
- signed: 36/36 present; complete
- rotation: 36/36 present; complete
- rotation-hetero: 36/36 present; complete
- givens: 36/36 present; complete
- delta: 36/36 present; complete
- signed-givens: 36/36 present; complete
- signed-delta: 36/36 present; complete
- gru: 36/36 present; complete

