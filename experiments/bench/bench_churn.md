# Benchmark round: churn (`bench-churn-02`)

Fit metric: `ckpt.val_auc` >= 0.8 (robustness triple: 0.78, 0.8, 0.82). Fisher reference arm: `log`. Computed from `experiments/lab_results.jsonl` (rows matching this round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 7836bddce36774c24da74f0cd1f7120e8a616c95, generated 2026-07-25T16:51:20.789131+00:00.

Stratum(s) observed: device=cuda, torch=2.8.0+cu128, scan=triton, compile=None

## Fits and generalization AUROC (raw / fit-only)

| arm | seeds (present/planned) | fits | AUROC@test (raw/fit-only) | params |
| --- | --- | --- | --- | --- |
| log | 36/36 | 36/36 | 0.822 / 0.822 | 91,906 |
| signed | 36/36 | 36/36 | 0.819 / 0.819 | 100,226 |
| rotation | 36/36 | 36/36 | 0.808 / 0.808 | 100,290 |
| signed-rotation | 36/36 | 34/36 | 0.813 / 0.815 | 100,162 |
| givens | 36/36 | 1/36 | 0.767 / 0.763 | 104,402 |
| delta | 36/36 | 35/36 | 0.818 / 0.818 | 126,098 |
| signed-givens | 36/36 | 21/36 | 0.795 / 0.805 | 102,242 |
| signed-delta | 36/36 | 35/36 | 0.821 / 0.821 | 113,098 |
| gru | 36/36 | 36/36 | 0.834 / 0.834 | 62,146 |

## Threshold-robustness

| arm | 0.78 | 0.8 | 0.82 |
| --- | --- | --- | --- |
| log | 36/36 | 36/36 | 25/36 |
| signed | 36/36 | 36/36 | 23/36 |
| rotation | 36/36 | 36/36 | 6/36 |
| signed-rotation | 36/36 | 34/36 | 10/36 |
| givens | 25/36 | 1/36 | 0/36 |
| delta | 36/36 | 35/36 | 6/36 |
| signed-givens | 33/36 | 21/36 | 4/36 |
| signed-delta | 36/36 | 35/36 | 13/36 |
| gru | 36/36 | 36/36 | 36/36 |

## Two-sided Fisher exact vs `log`

- signed (36/36) vs log (36/36): p = 1
- rotation (36/36) vs log (36/36): p = 1
- signed-rotation (34/36) vs log (36/36): p = 0.493
- givens (1/36) vs log (36/36): p = 1.975e-12
- delta (35/36) vs log (36/36): p = 1
- signed-givens (21/36) vs log (36/36): p = 9.638e-06
- signed-delta (35/36) vs log (36/36): p = 1
- gru (36/36) vs log (36/36): p = 1

## Completeness (present vs planned seed matrix)

- log: 36/36 present; complete
- signed: 36/36 present; complete
- rotation: 36/36 present; complete
- signed-rotation: 36/36 present; complete
- givens: 36/36 present; complete
- delta: 36/36 present; complete
- signed-givens: 36/36 present; complete
- signed-delta: 36/36 present; complete
- gru: 36/36 present; complete

