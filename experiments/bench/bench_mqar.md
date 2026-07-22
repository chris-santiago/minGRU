# Benchmark round: mqar (`bench-mqar-02`)

Fit metric: `ckpt.val_qacc` >= 0.99 (robustness triple: 0.98, 0.99, 0.995). Fisher reference arm: `log`. Computed from `experiments/lab_results.jsonl` (rows matching this round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 67653323f22a5db6a2e32113ec6e18e1eb975cf5, generated 2026-07-22T14:27:14.492362+00:00.

Stratum(s) observed: device=cuda, torch=2.8.0+cu128, scan=triton, compile=None

Provenance: signed-rotation <- `bench-mqar-rotfix-01` (other arms: `bench-mqar-02`) -- this report mixes rounds; see `arm_round_overrides` in the JSON payload.

## Fits and generalization accuracy (raw / fit-only)

| arm | seeds (present/planned) | fits | acc@T256_p16 (raw/fit-only) | acc@T256_p32 (raw/fit-only) | params |
| --- | --- | --- | --- | --- | --- |
| log | 36/36 | 0/36 | 0.112 / n/a | 0.081 / n/a | 91,712 |
| signed | 36/36 | 0/36 | 0.044 / n/a | 0.036 / n/a | 100,032 |
| rotation | 36/36 | 0/36 | 0.083 / n/a | 0.064 / n/a | 100,160 |
| signed-rotation | 36/36 | 0/36 | 0.064 / n/a | 0.051 / n/a | 100,096 |
| givens | 36/36 | 0/36 | 0.030 / n/a | 0.030 / n/a | 104,320 |
| delta | 36/36 | 36/36 | 0.931 / 0.931 | 0.493 / 0.493 | 126,032 |
| signed-givens | 36/36 | 0/36 | 0.036 / n/a | 0.034 / n/a | 102,176 |
| signed-delta | 36/36 | 36/36 | 0.928 / 0.928 | 0.690 / 0.690 | 113,032 |
| gru | 36/36 | 0/36 | 0.111 / n/a | 0.076 / n/a | 58,176 |

## Threshold-robustness

| arm | 0.98 | 0.99 | 0.995 |
| --- | --- | --- | --- |
| log | 0/36 | 0/36 | 0/36 |
| signed | 0/36 | 0/36 | 0/36 |
| rotation | 0/36 | 0/36 | 0/36 |
| signed-rotation | 0/36 | 0/36 | 0/36 |
| givens | 0/36 | 0/36 | 0/36 |
| delta | 36/36 | 36/36 | 36/36 |
| signed-givens | 0/36 | 0/36 | 0/36 |
| signed-delta | 36/36 | 36/36 | 36/36 |
| gru | 0/36 | 0/36 | 0/36 |

## Two-sided Fisher exact vs `log`

- signed (0/36) vs log (0/36): p = 1
- rotation (0/36) vs log (0/36): p = 1
- signed-rotation (0/36) vs log (0/36): p = 1
- givens (0/36) vs log (0/36): p = 1
- delta (36/36) vs log (0/36): p = 2.322e-13
- signed-givens (0/36) vs log (0/36): p = 1
- signed-delta (36/36) vs log (0/36): p = 2.322e-13
- gru (0/36) vs log (0/36): p = 1

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

