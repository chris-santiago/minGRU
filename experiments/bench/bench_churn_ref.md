# Benchmark round: churn REFERENCE arm(s) (`bench-churn-ref-01`)

**Non-matched reference/grounding arm(s) -- NOT part of the matched `bench-churn-02` seed-matrix accounting.** No Fisher-exact comparison is computed here: this population runs under its own training budget and round tag, distinct from the matched arms (CLAUDE.md: evidence strata are never mixed silently).

Fit metric: `ckpt.val_auc` >= 0.8 (robustness triple: 0.78, 0.8, 0.82). Computed from `experiments/lab_results.jsonl` (rows matching this reference round's tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.

Env: torch 2.13.0, commit 4b5cdaa2b6fde7fd09937a3fa70f5b0c15ec6697, generated 2026-07-25T19:36:25.423008+00:00.

Stratum(s) observed: device=cuda, torch=2.8.0+cu128, scan=triton, compile=None

## Fits and generalization AUROC (raw / fit-only)

| arm | seeds (present/planned) | fits | AUROC@test (raw/fit-only) | params |
| --- | --- | --- | --- | --- |
| gru-large | 36/36 | 36/36 | 0.827 / 0.827 | 690,946 |

## Threshold-robustness

| arm | 0.78 | 0.8 | 0.82 |
| --- | --- | --- | --- |
| gru-large | 36/36 | 36/36 | 36/36 |

## Completeness (present vs planned seed matrix)

- gru-large: 36/36 present; complete

