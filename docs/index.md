# mingru-scans

minGRU is the minimal GRU of [Feng et al., "Were RNNs All We Needed?", arXiv:2410.01201](https://arxiv.org/abs/2410.01201), a GRU stripped to a scannable linear recurrence. A parallel (associative) scan solves that recurrence by combining elements pairwise instead of one at a time, so the whole sequence resolves in logarithmic depth rather than a sequential loop.

`mingru-scans` is a research library of parallel-scan minGRU variants: the log-space baseline `MinGRU`, the signed-gate `SignedMinGRU`, the rotation-family mixers `RotationMinGRU` and `GivensMinGRU`, whose recurrences run as associative scans instead of sequential loops, and the delta-rule `DeltaMinGRU`, whose chunked-WY forward parallelizes the DeltaNet recurrence over the sequence. An **optional Triton backend** supplies fused forward+backward GPU kernels for the scan primitives behind a zero-config dispatch seam; on CPU, or wherever Triton is absent, the pure-PyTorch eager path runs unchanged. Measured forward+backward against plain PyTorch (eager) on an NVIDIA L4:

| scan op | lab shape ($B{=}128$, $T{=}64$) | long-$T$ ($B{=}16$, $T{=}1024$) |
|---|---|---|
| `matrix_scan` | **94.20x** | **168.53x** |
| `matrix_affine_scan` | **38.89x** | **53.56x** |
| `linear_scan` | 3.01x | 4.64x |
| `parallel_scan_log` | 0.70x (slower) | 0.80x (slower) |

The log-space op's backward recomputes its forward, so eager remains the right default for the baseline `MinGRU`/`SignedMinGRU` mixers; the win concentrates in the matrix/block ops the rotation-family mixers use. Full tables, memory results (395 → 38 MB), and the kernel design are in the [Triton scan kernels explanation](explanation/triton-scans.md) and `experiments/bench/`.

Use it when you want GRU-style sequence mixing that trains with parallel-scan speed, when you need state-tracking capacity beyond what diagonal-transition RNNs offer, or when you want to study the rotation-family parameterizations behind those capabilities.

Each mixer has its own lineage. The log-space `MinGRU` training path uses [Heinsen's (2023, arXiv:2311.06281)](https://arxiv.org/abs/2311.06281) parallel scan. `SignedMinGRU`'s negative-eigenvalue gates follow the state-tracking analyses of [Merrill, Petty & Sabharwal (2024, arXiv:2404.08819)](https://arxiv.org/abs/2404.08819) and [Grazzi et al. (2025, arXiv:2411.12537)](https://arxiv.org/abs/2411.12537). The `GivensMinGRU` brick-wall rotation mesh comes from the orthogonal/unitary-RNN literature: EUNN ([Jing et al. 2017, arXiv:1612.05231](https://arxiv.org/abs/1612.05231)) and the [Clements et al. (2016, arXiv:1603.08788)](https://arxiv.org/abs/1603.08788) rectangular mesh, built from the Givens plane rotations of Golub & Van Loan's *Matrix Computations*. `DeltaMinGRU` reimplements DeltaNet ([Yang et al. 2024, arXiv:2406.06484](https://arxiv.org/abs/2406.06484)) and its DeltaProduct generalization ([Siems et al. 2025, arXiv:2502.10297](https://arxiv.org/abs/2502.10297)), using the former's chunked WY representation as its parallel forward. Full citations live in each class's [API reference](reference/mixers.md) entry and the [Givens & Delta deep dive](explanation/givens-delta.md).

For the two composers built for extract-then-compose stacks, `DeltaMinGRU` at its native state is the promoted default when per-token state is free to grow (the most reliable trained composer measured on both CPU and GPU), while `GivensMinGRU` remains the recommendation when state must stay small or extreme-length extrapolation matters. See [How-to: choose a mixer](how-to/choose-a-mixer.md#the-two-axis-guidance) for the evidence behind that split. For how the family performs on accepted public benchmarks (MQAR associative recall, psMNIST, S5 group composition, and a pendulum irregular-time control) against a depth-matched GRU baseline, see [Benchmark validation](explanation/benchmark-validation.md).

## Install

```bash
pip install mingru-scans
```

Until the first PyPI release lands, install from the repository: `pip install git+https://github.com/chris-santiago/minGRU`.

Requires Python 3.10+ and `torch>=2.8`. The import name is `mingru`.

On Linux with the default PyPI torch, the Triton backend arrives automatically (torch's CUDA wheels pin and install the matching `triton`). If your torch is CUDA-capable but came from a channel that omitted triton (`mingru.available()` will say so), install the extra as a fallback: `pip install "mingru-scans[triton]"`. The extra is deliberately unpinned; if the resolved triton mismatches your torch build, install the exact version your torch was compiled against instead (see [Triton on GPU](tutorials/triton-on-gpu.md)).

## Where to go

- **New here?** [Start with the tutorial](tutorials/getting-started.md): install, build a model, train it on real data.
- **Have a task?** [Browse the how-to guides](how-to/index.md): choose a mixer, control dispatch, reproduce the evidence.
- **Need signatures?** [Browse the API reference](reference/index.md): generated from the NumPy docstrings.
- **Want the why?** [Read the explanations](explanation/index.md): the Givens & Delta deep dive, the benchmark-validation results, and the Triton kernel design.
