# Benchmark round: psmnist REFERENCE arm(s) (`bench-psmnist-ref-01`)

**Non-matched reference/grounding arm(s) -- NOT part of the matched `bench-psmnist-02` seed-matrix accounting.** No Fisher-exact comparison is computed here: this population runs under its own training budget and round tag, distinct from the matched arms (CLAUDE.md: evidence strata are never mixed silently).

Fit metric: `ckpt.val_acc` >= 0.9 (robustness triple: 0.88, 0.9, 0.92). Computed from `experiments/lab_results.jsonl` (rows matching this reference round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 67653323f22a5db6a2e32113ec6e18e1eb975cf5, generated 2026-07-22T14:27:14.544686+00:00.

Stratum(s) observed: device=cuda, torch=2.8.0+cu128, scan=triton, compile=None

## Fits and generalization accuracy (raw / fit-only)

| arm | seeds (present/planned) | fits | acc@test (raw/fit-only) | params |
| --- | --- | --- | --- | --- |
| gru-large | 12/12 | 12/12 | 0.922 / 0.922 | 596,234 |

## Threshold-robustness

| arm | 0.88 | 0.9 | 0.92 |
| --- | --- | --- | --- |
| gru-large | 12/12 | 12/12 | 11/12 |

## Completeness (present vs planned seed matrix)

- gru-large: 12/12 present; complete

