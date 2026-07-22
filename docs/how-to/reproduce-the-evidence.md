# Reproduce the evidence

This guide reproduces a recorded accuracy row from a repo checkout, exactly as it was measured. It is the counterpart to the tutorials: those *build* models, this one *replays the science*. It assumes a clone of the repository (the evidence lives in `probes.py` and `experiments/`, which the installed wheel does not ship).

## The two-pin story

The library and its evidence run under two different torch pins, on purpose:

- **Installed users** get `torch >= 2.8`: the floor that guarantees the full advertised API, including the Triton dispatch path.
- **Evidence replication** runs under the frozen **`torch == 2.5.1`** pin from a repo checkout, the exact version the recorded numbers were produced on.

The two never conflict because the evidence path does not depend on the package metadata: the root `min_gru.py` / `triton_scans.py` files re-export the packaged modules for repo consumers, so `probes.py` imports resolve to `src/mingru/` with **no installation** and **no Triton import attempted**. `uv run --with 'torch==2.5.1'` overrides the `torch >= 2.8` floor for the run, and torch 2.5.1 is what loads.

## Step 1: Replay the S3-hier / `signed → givens` row

The recommended hierarchical stack from the [two-layer stacks tutorial](../tutorials/two-layer-stacks.md) (`mixer=["signed", "givens"]`, registered as `minGRU-hetero-sg8`) has a recorded seed-0 row. Run this verbatim from the repo root (round `givens-promotion-replication-01` in `experiments/EXPERIMENTS.md`):

```bash
uv run --python 3.12 --with 'torch==2.5.1' python - <<'PY'
import probes
model = probes.run_one("S3-hier", "minGRU-hetero-sg8", ckpt=True, max_steps=1600, seed=0)
make, _, _ = probes.TASKS["S3-hier"]
for T, seed in [(64, 3), (256, 4), (512, 4), (1024, 4)]:
    print(T, round(probes.accuracy(model, make, T, seed=seed), 4))
PY
```

It trains the stack under best-val@128 checkpoint selection (`ckpt=True`) at the 1600-step budget, then evaluates at four lengths. The run takes a few minutes on CPU.

## Step 2: Match the recorded output

The seed-0 row is deterministic; the replication matched every metric exactly (no tolerance widening). Your output must read:

```
64 1.0
256 0.9941
512 0.9105
1024 0.6619
```

The best checkpoint lands at step 1100 with val@128 = 1.0. An exact match confirms the promoted `min_gru.py` code path *is* the evidence path: the same class and scan that produced the pooled multi-seed `S3-hier` numbers in the README.

## Run any other recorded cell

`probes.py` is the general driver behind every row in the README's tables:

```bash
uv run --python 3.12 --with 'torch==2.5.1' python probes.py TASK MODEL [N_LAYERS]
```

- `TASK`: `parity`, `S3`, `S3-hier`, `session-parity`, or `parity-timestamped`.
- `MODEL`: e.g. `minGRU-signed-tanh`, `minGRU-rotsnap`, `minGRU-hetero-sr`, `minGRU-hetero-sg8`, `GRU`.
- `CKPT=1` (env var): best-val@128 checkpoint selection, required for the rotation-family and hetero rows.
- `MAX_STEPS` (env var): override the 1600-step budget.

For example, the protocol-correct rotation-snap run on plain `S3`:

```bash
CKPT=1 uv run --python 3.12 --with 'torch==2.5.1' python probes.py S3 minGRU-rotsnap
```

Expect small numeric differences across torch versions but the same qualitative pattern; the frozen `torch == 2.5.1` pin reproduces the recorded numbers to the decimal. The full per-seed evidence trail is in `experiments/EXPERIMENTS.md` and `experiments/lab_results.jsonl`.

## Step 3: Replay a cell from the accepted-benchmark validation round

The tables above cover `probes.py`'s older word-problem tasks. A separate, newer round validates the mixer family on four accepted public benchmarks (S5, MQAR, psMNIST, pendulum); see [Benchmark validation](../explanation/benchmark-validation.md) for the results and [Choose a mixer](choose-a-mixer.md) for how they inform mixer selection. That round has its own driver, `experiments/benchmark_lab.py`, and its own stratum: the recorded numbers are the **L4 stratum** (`torch==2.8.0+cu128`, `triton==3.4.0`, `MINGRU_SCAN=triton`), not the pinned-CPU `torch==2.5.1` stratum used above. The two strata are never mixed.

`benchmark_lab.py` trains one seed of one arm on one task and emits a ledger row:

```bash
uv run python experiments/benchmark_lab.py --round bench-s5-02 --task s5 --model delta --seed 0
```

- `--task`: `s5`, `mqar`, `psmnist`, or `pendulum`.
- `--model`: any `ARM_REGISTRY` arm, e.g. `log`, `signed`, `rotation`, `signed-rotation`, `givens`, `delta`, `signed-givens`, `signed-delta`, or the classical `gru` control.
- `--device`: `cpu` (default) or `cuda`. `--round`/`--task`/`--model` are required unless `--selftest` is passed.

The committed evidence trains every arm on `cuda` under the L4 stratum via a Lightning AI batch job (foreground-only, no keepalive): `uv run python scripts/gpu_check.py --job benchmarks`, which runs `scripts/gpu_benchmark_campaign.py` inside the job. That job needs Lightning credentials and a GPU-billed machine, so it is not a local, drop-in command the way Step 1's replay is; if you have Lightning access, `--tasks`/`--arms`/`--seeds` subset which cells the job trains. Reproducing the full nine-arm x four-task matrix at the recorded seed counts is a GPU-stratum undertaking, not a CPU one.

What you *can* verify locally, on CPU, without any GPU or Lightning access:

```bash
uv run python experiments/benchmark_lab.py --selftest
```

This runs a tiny few-step, small-batch sweep of all four tasks under the `log` arm, dry-run only. It confirms the driver's wiring (data generation, model construction, checkpoint selection, all four loss modes) but is **not evidence** — it does not reproduce a recorded number, only that the code path runs.

The report artifacts (`experiments/bench/bench_{s5,mqar,psmnist,pendulum}.{json,md}`) are regenerated whole from `experiments/lab_results.jsonl` by:

```bash
uv run python scripts/report_benchmarks.py
```

This script imports the packaged `mingru` distribution to compute param counts but touches no GPU, data, or randomness-sensitive state, so it runs under the project's normal (unpinned) environment, not the `torch==2.5.1` evidence pin. The per-seed rows it reads are tagged `bench-s5-02` / `bench-mqar-02` / `bench-psmnist-02` / `bench-pendulum-02` in `experiments/lab_results.jsonl`; the round-by-round narrative is in `experiments/EXPERIMENTS.md`.

Note on `probes.py`'s delta coverage: `probes.py`'s own `MIXER_REGISTRY` (the `MODEL` list in Step 1's "run any other recorded cell" section above) has no `delta` entry. Its rows run `log`/`signed`/`rotation`/`givens` stacks and their hetero combinations only, so there is no valid delta `MODEL` name to run through `probes.py` directly. `DeltaMinGRU` is exercised through `probes.py`'s sibling driver, `experiments/hetero_lab.py`, and through `benchmark_lab.py`'s `delta`/`signed-delta` arms above.

## You have now

reproduced a recorded accuracy row bit-for-bit from a repo checkout under the frozen evidence pin, know the general `probes.py` command for replaying any other cell in the README, and know the accepted-benchmark round's driver, its L4-vs-pinned-CPU stratum split, and which parts of it (the CPU selftest, the report regeneration) are locally runnable versus which part (the full seed matrix) needs the GPU stratum.
