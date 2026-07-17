# Run the benchmarks

This guide runs the Triton scan benchmarks from a repo checkout and reproduces the committed speedup and memory artifacts. It assumes a CUDA GPU (the timing and memory runs need one) and a checkout of the repository, not a `pip install`.

## The driver

`scripts/bench_scans.py` has three independent modes; pass at least one:

| Flag | What it does | Writes to `experiments/bench/` |
|---|---|---|
| `--check` | Full Triton-vs-eager equivalence suite (correctness, per-case error) | `scan_parity.json` / `.md` |
| `--bench` | Timing matrix: Triton vs eager vs `torch.compile`, per op and shape | `scan_bench.json` / `.md` |
| `--memory` | `GivensMinGRU` angle-fused vs generic backward peak memory | `scan_memory.json` / `.md` |

The script inserts the repo root on `sys.path` itself, so run it from anywhere in the checkout. On a machine with no CUDA device, `--check` loud-skips (it cannot compare against Triton); the timing and memory modes need a GPU to produce numbers.

## Step 1 — Verify correctness first

Establish that the Triton kernels match eager before trusting any timing:

```bash
uv run --python 3.12 --with 'torch==2.8.*' python scripts/bench_scans.py --check
```

This writes the per-case parity table to `experiments/bench/scan_parity.md`. Every registered op is checked forward and backward, in-envelope and (where applicable) out-of-envelope.

## Step 2 — Run the timing matrix

```bash
uv run --python 3.12 --with 'torch==2.8.*' python scripts/bench_scans.py --bench
```

This benchmarks each scan op at two shapes — a "lab" shape (`B=128, T=64`) and a long-sequence shape (`T=1024`) — across Triton, eager, and `torch.compile`, forward and forward+backward, and writes `experiments/bench/scan_bench.md`. The committed run (NVIDIA L4, torch 2.8) records:

| op | direction | Triton speedup vs eager |
|---|---|---|
| `matrix_scan` (long-T) | fwd+bwd | 168.53× |
| `matrix_scan` (lab) | fwd+bwd | 94.20× |
| `matrix_affine_scan` (long-T) | fwd+bwd | 53.56× |
| `matrix_affine_scan` (lab) | fwd+bwd | 38.89× |
| `linear_scan` (long-T) | fwd+bwd | 4.64× |
| `parallel_scan_log` (lab) | fwd+bwd | 0.70× (slower) |
| `parallel_scan_log` (long-T) | fwd+bwd | 0.80× (slower) |

The block-structured scans (behind `RotationMinGRU`/`GivensMinGRU`) are where Triton wins big. The elementwise `parallel_scan_log` (behind `MinGRU`/`SignedMinGRU`) is memory-bound and already cheap in eager form, so its Triton forward+backward is *slower* — run those mixers with `MINGRU_SCAN=eager` (see [Control scan dispatch](control-scan-dispatch.md)).

## Step 3 — Measure the memory win

```bash
uv run --python 3.12 --with 'torch==2.8.*' python scripts/bench_scans.py --memory
```

This compares `GivensMinGRU`'s angle-fused Triton backward against the generic eager backward and writes `experiments/bench/scan_memory.md`. The committed run records peak backward memory dropping from **395.09 MB to 38.25 MB** at `B=128, T=64, d=64, k=8`.

## Step 4 — Compare against the committed numbers

Your GPU will produce different absolute microseconds, but the *pattern* should hold: large fwd+bwd speedups on the block scans, near-parity-or-slower on the elementwise log scan, and roughly an order-of-magnitude memory reduction on the fused Givens backward. Diff your regenerated `experiments/bench/*.md` against the committed versions to confirm.

## You have now

verified Triton correctness, regenerated the timing and memory artifacts on your own GPU, and confirmed the block-scan speedups and the fused-backward memory reduction against the committed numbers.
