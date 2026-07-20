# Control scan dispatch

The four scan functions (`parallel_scan_log`, `linear_scan`, `matrix_scan`, `matrix_affine_scan`) pick their backend at call time from the `MINGRU_SCAN` environment variable. This guide shows how to force a backend and what each mode guarantees. It assumes you have completed the [getting-started tutorial](../tutorials/getting-started.md).

## The three modes

`MINGRU_SCAN` is read from `os.environ` on every scan call (not cached at import), so you can set it before launching a process or from inside one via `os.environ`.

| `MINGRU_SCAN` | On CUDA tensors | On CPU tensors | Imports `triton_scans`? |
|---|---|---|---|
| `auto` (default) | Triton kernel if available, else eager (warns once) | eager | only when a CUDA input reaches the seam |
| `eager` | eager | eager | never |
| `triton` | Triton kernel, or **raises** | **raises** | yes |

## Force the eager path

Set `MINGRU_SCAN=eager` to guarantee the pure-PyTorch implementation everywhere. This never imports the Triton module, so it is the mode to use below the `torch >= 2.8` floor, for deterministic A/B timing against Triton, or when you only use the diagonal mixers (whose elementwise log scan is *faster* in eager form, see [Run the benchmarks](run-the-benchmarks.md)).

```bash
MINGRU_SCAN=eager python your_script.py
```

```python
import os
os.environ["MINGRU_SCAN"] = "eager"   # set before the first scan call
import torch
from mingru import parallel_scan_log

log_coeffs = -torch.rand(2, 5, 3)
log_values = -torch.rand(2, 6, 3)
print(parallel_scan_log(log_coeffs, log_values).shape)   # torch.Size([2, 5, 3])
```

## Require the Triton path

Set `MINGRU_SCAN=triton` to make the Triton kernel mandatory: if a kernel is unavailable (no CUDA device, `triton` not importable, inputs on CPU, or no kernel registered for that op), the call raises `RuntimeError` instead of silently downgrading to eager. Use it in CI or a benchmark harness to prove the accelerated path is actually live.

```python
import os
os.environ["MINGRU_SCAN"] = "triton"
import torch
from mingru import parallel_scan_log

# CPU tensors under triton mode -> hard error, never a silent fallback:
parallel_scan_log(-torch.rand(2, 5, 3), -torch.rand(2, 6, 3))
```

```
RuntimeError: MINGRU_SCAN=triton requested for 'parallel_scan_log' but Triton is unavailable: CUDA not available
```

## Understand `auto` (the default)

`auto` is what you want in production: CUDA-resident inputs with a usable Triton kernel run on Triton; everything else (CPU tensors, a missing Triton module, an out-of-envelope shape, or an op with no kernel yet) falls through to the unchanged eager implementation. A fallback that happens **despite CUDA inputs** warns exactly once per process, naming the reason, so a silent performance cliff can't hide:

```
UserWarning: MINGRU_SCAN=auto fell back to the eager scan implementation for 'parallel_scan_log' despite CUDA inputs: <reason>
```

CPU inputs under `auto` fall back silently (that is the expected path, not a regression) and never import the Triton module.

## Check the current status

`mingru.available()` tells you whether the Triton path *can* run at all: `True`, or a reason string like `"CUDA not available"`:

```python
import mingru
print(mingru.available())
```

An invalid value (anything other than `auto`/`eager`/`triton`) raises `ValueError` from the scan call, so a typo fails loudly rather than being treated as `auto`.

## You have now

forced the eager or Triton backend per process, made the Triton path mandatory where you need proof it is live, and know how `auto` decides and warns. For the GPU-side walkthrough, see [Triton on GPU](../tutorials/triton-on-gpu.md).
