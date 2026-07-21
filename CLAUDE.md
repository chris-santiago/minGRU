# CLAUDE.md — minGRU

Project rules for Claude sessions. These extend `~/.claude/CLAUDE.md`; project-specific rules here win on conflict.

## Commands

- Python runs use `uv run` (never bare python/conda): `uv run pytest tests/ -q`, `uv run ruff check .`, `uv run ruff format --check .`
- Evidence-pin runs use `uv run --no-project --with torch==2.5.1 python <script>` — the pyproject floor is install-only; all measured CPU evidence is pinned to torch 2.5.1
- The frozen-AST gate must pass before any commit touching `src/mingru/` or root `min_gru.py`/`triton_scans.py`: `uv run python scripts/check_frozen_ast.py`

## Evidence discipline (the load-bearing rules)

- `experiments/EXPERIMENTS.md` is the round-by-round evidence ledger; `experiments/index.md` maps every file in that folder. Rounds append at the end; entries are never rewritten.
- **Numbers are transcribed from artifacts, never from memory or conversation.** Bench artifacts (`experiments/bench/*.{json,md}`) are regenerated whole by their named scripts, never hand-edited. A hand-transcribed p-value shipped wrong once; reviewers now check cell-by-cell.
- **Evidence strata are never mixed silently**: pinned-CPU (torch 2.5.1) vs L4 (torch 2.8/cu128, triton 3.4) vs A100 rows are labeled everywhere they appear.
- `experiments/lab_results.jsonl` is append-only; per-seed rows carry their stratum config.
- Docs framing rules: never say "delta is less accurate" unqualified (acc@T vs fit-only distinction — see `docs/how-to/choose-a-mixer.md`); never claim the delta Triton kernel reaches compile-class speed (measured FAIL, round `gpu-delta-kernel-01`); `torch.compile` is the documented CUDA recommendation for `mixer="delta"`.

## Frozen surface

- `scripts/check_frozen_ast.py` holds the four original mixer classes, the four scan ops, `DecayMixin`, dispatch internals, and 5 constants AST-identical to the pinned evidence baseline. The freeze is owner-controlled: never modify a frozen definition; call frozen helpers from new code instead. Per-definition unfreezing is the user's decision, not yours.
- `DeltaMinGRU` is NOT on the frozen surface, but its eager `_forward_chunked`/`step`/`_coeffs` math is evidence-bearing — semantic changes invalidate recorded rounds. The CPU fp64 guard (`tests/test_scan_ops.py::TestDeltaBackwardTorchVsEagerAutograd`) pins the backward against the eager oracle; keep it green.

## Kernel/dispatch conventions

- All Triton kernel code lives in `src/mingru/triton_scans.py` — no new modules (owner decision; recorded debt to split if that ever lifts).
- `MINGRU_SCAN` contract: `auto` may fall back silently (warn-once) but never hard-fails; `triton` fails loud with a reason, never silently downgrades; `eager` never imports triton. New kernel paths join this exact pattern (envelope validation + `ScanFallback` funnel inside the `triton_scans` entry point, `hasattr` discovery, dispatch helper in `min_gru.py`).
- Only permitted division in kernel/scan math: the implicit unit diagonal of unitriangular substitution.

## GPU jobs (Lightning AI)

- Credentials live in `.claude/settings.local.json` `env` block and load into the session environment at startup. If `LIGHTNING_API_KEY` is missing (e.g. creds were added mid-session), export the env block explicitly before `scripts/gpu_check.py`.
- Job modes: `--job check` (parity suite), `--job delta-probe`, `--job hetero36`; `--machine` overrides L4.
- Job command chains are **foreground-only — never background a keepalive** (a backgrounded loop outlives the command, hangs `job.wait()`, and bills). The 10-minute idle shutdown applies to interactive studios only.
- Recovery for a stuck job: `lightning_sdk.Job(name, teamspace=..., org=...)` → `.stop()` → `.logs`.

## Docs and slides

- Site config is `zensical.toml` (not mkdocs). Published decks live in `docs/slides/`; any edit to a deck in `slides/` must be re-rendered (marp) and re-copied to its `docs/slides/` counterpart. `slides/` deck files are gitignored; only `docs/slides/` copies are tracked.

## Known accepted debt (don't "fix" in passing)

- `scripts/gpu_check.py` has pre-existing ruff-format drift; `scripts/bench_scans.py` has project-excluded lint findings. Both recorded; leave unless the task is specifically their cleanup.
- `_scan_env` is deliberately triplicated (triton_scans.py, gpu_hetero_campaign.py, gpu_delta_probe.py) — hoist only if a fourth site appears.
- The guarded-JSONL-read parse is deliberately duplicated across 4 sites with 3 distinct contracts: `gpu_check.py` `_existing_keys_by_key` and `report_benchmarks.py` `_load_all_rows` (full guard: skip blanks + try/except json.loads); `benchmark_lab.py` `_row_exists` and `run_matched_state.py` `_load_rows_by_round` (blank-line-skip only, malformed raises). Hoist deferred by owner decision.
- 3 `tests/test_packaging.py` failures under the 2.5.1 evidence pin are expected (triton-lazy-import tests assume torch ≥ 2.8); the suite is green under the default env.
- Benchmark-round code cohesion (design review 2026-07-20, deferred by owner decision; no shipped-behavior bug, suite green, evidence verified). Four findings, address only in a dedicated cleanup pass: (1) fit/robustness/direction logic is duplicated between `report_benchmarks.py` and the parameterized `_evidence_stats.arm_stats`, which the round bypasses, with no cross-check test binding the two copies; (2) arm identity is spread across ~7 parallel structures (`MATRIX_ARMS`, `PROBE_ARMS`, `REF_ARMS`, `ARM_REGISTRY`, `DECAY_CAPABLE_ARMS`, `REF_ARM_BUDGETS`, the `BENCH_*_ROUND_TAGS` maps) instead of one `Arm` descriptor; each amendment added another. (3) `gru`/`gru-large` are sentinel tuples `(name, None)` under a `(mixer, kwargs)` annotation, safe only because `build_model` special-cases them before unpacking. (4) `_train_epoch_based` hardcodes higher-is-better and ignores `task.fit_direction`: latent only, since psMNIST is `ge` (no live bug), but a future epoch-based `le` task would silently select the worst checkpoint.

## Release

- Merging to `main` publishes nothing. Release = tag a GitHub Release (OIDC publish to PyPI) — always the user's explicit decision.
