# Triton on GPU

The eager PyTorch scans run everywhere. On a CUDA GPU, an optional Triton backend accelerates the scan primitives behind a zero-config dispatch seam: you do not change any model code — the same `MinGRUStack` runs faster because the scan functions route to Triton kernels when it is safe to do so. This tutorial confirms Triton is present, checks that it engages, and shows you the one status call that tells you which backend is live.

By the end you will know whether your environment has the Triton path, how to verify it from Python, and what to expect on machines that don't.

## Step 1 — Understand what "having Triton" means

There is no separate Triton install step. `triton` ships as a dependency of the CUDA builds of PyTorch, so whether you have it is entirely determined by which `torch` wheel you installed:

| Platform / install | Triton present? | Behavior |
|---|---|---|
| Linux x86_64, default PyPI `torch` (CUDA build) | Yes | `auto` runs Triton kernels on CUDA tensors |
| CPU-only index (`--index-url .../cpu`), Windows, macOS | No | eager fallback everywhere, one warning on a CUDA-input fallback |
| Any of the above, but running on CPU tensors | Unused | eager path, no warning |

So on a standard Linux CUDA box you already have everything. On macOS, Windows, or a CPU-only PyTorch, the library still works — it just always runs the eager scans.

## Step 2 — Check availability from Python

`mingru.available()` reports the status. It returns `True` only when a CUDA device is present **and** `triton` is importable; otherwise it returns a human-readable reason string.

```python
import mingru

print(mingru.available())
```

**Output on a CUDA machine with Triton:**

```
True
```

**Output on CPU-only / macOS / Windows:**

```
CUDA not available
```

Touching `mingru.available()` is also the first thing that imports the Triton module — `import mingru` on its own never does, so environments below the `torch >= 2.8` floor can still import the eager library.

## Step 3 — Confirm the kernels engage

With `MINGRU_SCAN` left at its default (`auto`), CUDA-resident inputs automatically use Triton. Move a model and its inputs to the GPU and run as usual — no other change:

```python
import torch
from mingru import MinGRUStack

assert torch.cuda.is_available(), "this step needs a CUDA GPU"

model = MinGRUStack(input_size=32, d_model=64, n_layers=2, mixer="signed").cuda()
x = torch.randn(8, 512, 32, device="cuda")

out, _ = model(x)          # scans dispatch to Triton on CUDA tensors
print(out.shape, out.device)
```

**Expected output:**

```
torch.Size([8, 512, 64]) cuda:0
```

To prove the dispatch seam is doing what you think, force the backend explicitly with the `MINGRU_SCAN` environment variable and compare. `MINGRU_SCAN=triton` *requires* a kernel — it raises rather than silently falling back — so a clean run under it is positive proof the Triton path is live:

```bash
MINGRU_SCAN=triton python your_script.py    # raises if Triton can't serve the op
MINGRU_SCAN=eager  python your_script.py    # forces the pure-PyTorch path, for A/B timing
```

The full `auto` / `eager` / `triton` semantics live in [Control scan dispatch](../how-to/control-scan-dispatch.md).

## What you gain (and where you don't)

The block-structured scans — the ones behind `RotationMinGRU` and `GivensMinGRU` — are where Triton pays off, because their eager form materializes large intermediate tensors. On an NVIDIA L4 (torch 2.8, from `experiments/bench/scan_bench.md`), forward-plus-backward speedups over eager reach **39× for `matrix_affine_scan` and up to 169× for `matrix_scan`** at long sequences, and `GivensMinGRU`'s angle-fused backward cuts peak memory from 395 MB to 38 MB.

The elementwise log-space scan (`parallel_scan_log`, behind `MinGRU`/`SignedMinGRU`) is the honest exception: it is already memory-bound and cheap in eager form, so its Triton forward+backward measures **0.70–0.80× — slower than eager**. `auto` still runs it on GPU for consistency; if you only use the diagonal mixers, `MINGRU_SCAN=eager` is the faster choice. The full table is in [Run the benchmarks](../how-to/run-the-benchmarks.md) and [the Triton scans explanation](../explanation/triton-scans.md).

## What you did

You learned that the Triton path is determined by your `torch` wheel, verified it with `mingru.available()`, confirmed the kernels engage on CUDA tensors under the default `auto` dispatch, and saw where the acceleration helps (the block scans) and where it doesn't (the elementwise log scan).

## Next steps

- [Control scan dispatch](../how-to/control-scan-dispatch.md) — the `MINGRU_SCAN` modes in full.
- [Run the benchmarks](../how-to/run-the-benchmarks.md) — reproduce the speedup table on your own GPU.
- [Triton scan kernels](../explanation/triton-scans.md) — how the kernels are built and why the numbers land where they do.
