# Benchmark round: psmnist REFERENCE arm(s) (`bench-psmnist-ref-01`)

**Non-matched reference/grounding arm(s) -- NOT part of the matched `bench-psmnist-02` seed-matrix accounting.** No Fisher-exact comparison is computed here: this population runs under its own training budget and round tag, distinct from the matched arms (CLAUDE.md: evidence strata are never mixed silently).

Fit metric: `ckpt.val_acc` >= 0.9 (robustness triple: 0.88, 0.9, 0.92). Computed from `experiments/lab_results.jsonl` (rows matching this reference round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 7cfda8b9f4f1011caf0ce0457b097e5a1dfa76b9, generated 2026-07-20T14:09:21.318062+00:00.

Stratum(s) observed: none (0 rows).

## Fits and generalization accuracy (raw / fit-only)

| arm | seeds (present/planned) | fits | params |
| --- | --- | --- | --- |
| gru-large | 0/12 | 0/0 | 596,234 |
  (0 rows found for arm `gru-large`)

## Threshold-robustness

| arm | 0.88 | 0.9 | 0.92 |
| --- | --- | --- | --- |
| gru-large | n/a | n/a | n/a |

## Completeness (present vs planned seed matrix)

- gru-large: 0/12 present; missing seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

