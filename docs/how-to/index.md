# How-to guides

Task-oriented recipes. Each assumes you have completed the [getting-started tutorial](../tutorials/getting-started.md).

- [Choose a mixer](choose-a-mixer.md) — pick `mixer=` from your state shape and per-token budget; CPU-only, no GPU needed.
- [Control scan dispatch](control-scan-dispatch.md) — force the eager or Triton backend via `MINGRU_SCAN`; a CUDA GPU is needed only to exercise the Triton path itself.
- [Enable time decay](enable-time-decay.md) — turn on `decay=`/`delta_t` for a mixer or stack, including the delta-mixer's eager-only path; CPU-only.
- [Run the benchmarks](run-the-benchmarks.md) — reproduce the Triton scan speedup/memory artifacts; needs a CUDA GPU.
- [Reproduce the evidence](reproduce-the-evidence.md) — replay a recorded accuracy row under the frozen `torch==2.5.1` pin from a repo checkout; CPU-only.
