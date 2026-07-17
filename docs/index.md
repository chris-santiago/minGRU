# mingru-scans

`mingru-scans` is a research library of parallel-scan minGRU variants: the log-space baseline `MinGRU`, the signed-gate `SignedMinGRU`, and the rotation-family mixers `RotationMinGRU` and `GivensMinGRU`, whose recurrences run as associative scans instead of sequential loops. An optional Triton backend accelerates the scan primitives on CUDA GPUs behind a zero-config dispatch seam; on CPU, or wherever Triton is absent, the pure-PyTorch eager path runs unchanged.

Use it when you want GRU-style sequence mixing that trains with parallel-scan speed, when you need state-tracking capacity beyond what diagonal-transition RNNs offer, or when you want to study the rotation-family parameterizations behind those capabilities.

## Install

```bash
pip install mingru-scans
```

Requires Python 3.10+ and `torch>=2.8`. The import name is `mingru`.

## Where to go

- **New here?** [Start with the tutorial](tutorials/getting-started.md) — install, build a model, train it on real data.
- **Have a task?** [Browse the how-to guides](how-to/index.md) — choose a mixer, control dispatch, reproduce the evidence.
- **Need signatures?** [Browse the API reference](reference/index.md) — generated from the NumPy docstrings.
- **Want the why?** [Read the explanations](explanation/index.md) — the GivensMinGRU deep dive and the Triton kernel design.
