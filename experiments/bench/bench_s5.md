# Benchmark round: s5 (`bench-s5-02`)

Fit metric: `ckpt.val128` >= 0.99 (robustness triple: 0.98, 0.99, 0.995). Fisher reference arm: `log`. Computed from `experiments/lab_results.jsonl` (rows matching this round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 67653323f22a5db6a2e32113ec6e18e1eb975cf5, generated 2026-07-22T14:27:14.472065+00:00.

Stratum(s) observed: device=cuda, torch=2.8.0+cu128, scan=triton, compile=None

Provenance: signed-rotation <- `bench-s5-rotfix-01` (other arms: `bench-s5-02`) -- this report mixes rounds; see `arm_round_overrides` in the JSON payload.

## Fits and generalization accuracy (raw / fit-only)

| arm | seeds (present/planned) | fits | acc@T1024 (raw/fit-only) | acc@T256 (raw/fit-only) | acc@T512 (raw/fit-only) | params |
| --- | --- | --- | --- | --- | --- | --- |
| log | 36/36 | 0/36 | 0.010 / n/a | 0.016 / n/a | 0.012 / n/a | 98,936 |
| signed | 36/36 | 0/36 | 0.015 / n/a | 0.025 / n/a | 0.019 / n/a | 107,256 |
| rotation | 36/36 | 0/36 | 0.012 / n/a | 0.020 / n/a | 0.015 / n/a | 107,384 |
| signed-rotation | 36/36 | 0/36 | 0.011 / n/a | 0.016 / n/a | 0.013 / n/a | 107,320 |
| givens | 36/36 | 0/36 | 0.010 / n/a | 0.015 / n/a | 0.012 / n/a | 111,544 |
| delta | 36/36 | 0/36 | 0.011 / n/a | 0.016 / n/a | 0.012 / n/a | 133,256 |
| signed-givens | 36/36 | 1/36 | 0.038 / 0.817 | 0.058 / 1.000 | 0.049 / 0.976 | 109,400 |
| signed-delta | 36/36 | 0/36 | 0.011 / n/a | 0.018 / n/a | 0.014 / n/a | 120,256 |
| gru | 36/36 | 0/36 | 0.017 / n/a | 0.024 / n/a | 0.019 / n/a | 65,400 |

## Threshold-robustness

| arm | 0.98 | 0.99 | 0.995 |
| --- | --- | --- | --- |
| log | 0/36 | 0/36 | 0/36 |
| signed | 0/36 | 0/36 | 0/36 |
| rotation | 0/36 | 0/36 | 0/36 |
| signed-rotation | 0/36 | 0/36 | 0/36 |
| givens | 0/36 | 0/36 | 0/36 |
| delta | 0/36 | 0/36 | 0/36 |
| signed-givens | 1/36 | 1/36 | 1/36 |
| signed-delta | 0/36 | 0/36 | 0/36 |
| gru | 0/36 | 0/36 | 0/36 |

## Two-sided Fisher exact vs `log`

- signed (0/36) vs log (0/36): p = 1
- rotation (0/36) vs log (0/36): p = 1
- signed-rotation (0/36) vs log (0/36): p = 1
- givens (0/36) vs log (0/36): p = 1
- delta (0/36) vs log (0/36): p = 1
- signed-givens (1/36) vs log (0/36): p = 1
- signed-delta (0/36) vs log (0/36): p = 1
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

