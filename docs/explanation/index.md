# Explanation

Explanation answers *why*, not *how*. These articles cover the design decisions, the trade-offs, and the rejected alternatives behind the two parts of this library that carry the most non-obvious reasoning: the `GivensMinGRU` mixer and the Triton scan-kernel backend. Where a page needs a function signature or a keyword-argument list, it links to [Reference](../reference/index.md) rather than repeating it; where it needs a runnable command, it links to [How-to](../how-to/index.md).

Both articles are grounded in the committed measured evidence. Every quantitative claim traces to an artifact in the repository: the multi-seed accuracy ledger behind `README.md` and `experiments/EXPERIMENTS.md`, the review-hardened `experiments/TECHNICAL_REPORT.md`, and the benchmark and conformance tables under `experiments/bench/`. That includes the results that do not flatter the design, which are stated as plainly as the ones that do.

## Articles

- [**GivensMinGRU deep dive**](givens-mingru.md) — why the non-commutative rung of the ladder gets a richer per-token map, how the brick-wall Givens parameterization is built and why that specific mesh, what the `S3-hier` evidence does and does not demonstrate against diagonal-transition RNNs, the rounds ablation that separates map richness from block size, and the backward-pass design decision that rejected division-based reversal in favor of an exact $C=1$ stored-state recompute.
- [**Triton scan kernels**](triton-scans.md) — why the four scan operations are one associative-scan family, how the kernels are laid out (one program per lane, a sequential-in-$T$ register prefix), the fp32-accumulation and TF32/IEEE precision story, the dispatch seam that keeps the kernels optional, and the measured speedups — including the honest rows where the Triton path is slower than eager.

## Where to go next

- To pick a mixer for a task, start with [How-to: choose a mixer](../how-to/choose-a-mixer.md).
- To build the two-layer extract-then-compose stack these articles reference, see [Tutorials: two-layer stacks](../tutorials/two-layer-stacks.md).
- To enable the Triton backend on a GPU, see [Tutorials: Triton on GPU](../tutorials/triton-on-gpu.md) and [How-to: control scan dispatch](../how-to/control-scan-dispatch.md).
- For the class and function signatures, see [Reference](../reference/index.md).
