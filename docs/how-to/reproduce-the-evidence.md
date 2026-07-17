# Reproduce the evidence

This guide reproduces a recorded accuracy row from a repo checkout, exactly as it was measured. It is the counterpart to the tutorials: those *build* models, this one *replays the science*. It assumes a clone of the repository (the evidence lives in `probes.py` and `experiments/`, which the installed wheel does not ship).

## The two-pin story

The library and its evidence run under two different torch pins, on purpose:

- **Installed users** get `torch >= 2.8` — the floor that guarantees the full advertised API, including the Triton dispatch path.
- **Evidence replication** runs under the frozen **`torch == 2.5.1`** pin from a repo checkout, the exact version the recorded numbers were produced on.

The two never conflict because the evidence path does not depend on the package metadata: the root `min_gru.py` / `triton_scans.py` files re-export the packaged modules for repo consumers, so `probes.py` imports resolve to `src/mingru/` with **no installation** and **no Triton import attempted**. `uv run --with 'torch==2.5.1'` overrides the `torch >= 2.8` floor for the run, and torch 2.5.1 is what loads.

## Step 1 — Replay the S3-hier / `signed → givens` row

The recommended hierarchical stack from the [two-layer stacks tutorial](../tutorials/two-layer-stacks.md) — `mixer=["signed", "givens"]`, registered as `minGRU-hetero-sg8` — has a recorded seed-0 row. Run this verbatim from the repo root (round `givens-promotion-replication-01` in `experiments/EXPERIMENTS.md`):

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

## Step 2 — Match the recorded output

The seed-0 row is deterministic; the replication matched every metric exactly (no tolerance widening). Your output must read:

```
64 1.0
256 0.9941
512 0.9105
1024 0.6619
```

The best checkpoint lands at step 1100 with val@128 = 1.0. An exact match confirms the promoted `min_gru.py` code path *is* the evidence path — the same class and scan that produced the pooled multi-seed `S3-hier` numbers in the README.

## Run any other recorded cell

`probes.py` is the general driver behind every row in the README's tables:

```bash
uv run --python 3.12 --with 'torch==2.5.1' python probes.py TASK MODEL [N_LAYERS]
```

- `TASK` — `parity`, `S3`, `S3-hier`, `session-parity`, or `parity-timestamped`.
- `MODEL` — e.g. `minGRU-signed-tanh`, `minGRU-rotsnap`, `minGRU-hetero-sr`, `minGRU-hetero-sg8`, `GRU`.
- `CKPT=1` (env var) — best-val@128 checkpoint selection, required for the rotation-family and hetero rows.
- `MAX_STEPS` (env var) — override the 1600-step budget.

For example, the protocol-correct rotation-snap run on plain `S3`:

```bash
CKPT=1 uv run --python 3.12 --with 'torch==2.5.1' python probes.py S3 minGRU-rotsnap
```

Expect small numeric differences across torch versions but the same qualitative pattern; the frozen `torch == 2.5.1` pin reproduces the recorded numbers to the decimal. The full per-seed evidence trail is in `experiments/EXPERIMENTS.md` and `experiments/lab_results.jsonl`.

## You have now

reproduced a recorded accuracy row bit-for-bit from a repo checkout under the frozen evidence pin, and know the general `probes.py` command for replaying any other cell in the README.
