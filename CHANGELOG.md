# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Float32 decay accuracy in `DeltaMinGRU`.** The decay-active forward formed each decay ratio as a difference of two running log-decay sums, which lost float32 precision once several large gaps shared a chunk and made accuracy depend on `chunk_size`. Each ratio is now a direct sum over only the tokens it spans, so the forward matches the sequential `step` oracle at every `chunk_size`. This replaces the internal saturation clamp and its floor constant, and also fixes the same loss in float64 after a sanitized `+inf` gap. The saturation warning remains, now worded as a modeling signal.
- **Mismatched `delta_t` dtype.** A float64 `delta_t` (the numpy and pandas default) passed to a float32 stack crashed with a dtype error that never named `delta_t`. `delta_t` is now converted to the input's dtype, never narrower than float32, at `MinGRUBlock` and `DeltaMinGRU` `forward` and `step`.
- **Zero-length sequences in `DeltaMinGRU`.** `forward` raised `ValueError` when $T = 0$. It now returns an empty `(B, 0, hidden)` output with the state unchanged, as the other four mixers do.

### Changed

- **Decay-active float32 outputs of `DeltaMinGRU` change.** Outputs differ from 0.2.0 at rounding level, and by up to the old error on bursty inputs with several large gaps in one chunk. `decay=None` is bit-identical.
- **Documentation.** The delta kernel's `MINGRU_SCAN=auto` region is described as extrapolated from one measured L4 cell, not a measured region. The chunked-vs-sequential tolerance figures are qualified as measured on a randomly initialized layer and larger as $\beta \to 2$. The README and time-decay how-to now state that the float32 `mixer="log"` forward loses accuracy as the running total of $\lambda f(\Delta t)$ grows (a sanitized `+inf` gap in raw mode corrupts the rest of the sequence), that `log1p_delta=True` or gaps in units comparable to $1/\lambda$ keep it accurate, and that `step` and the signed, rotation, and Givens mixers are unaffected. `DeltaMinGRU`'s departures from DeltaNet are disclosed in its docstring.

## [0.2.0] - 2026-07-21

Adds the `DeltaMinGRU` delta-rule composer and its ecosystem. Backward
compatible: the four existing mixers are unchanged.

### Added

- **`DeltaMinGRU` mixer (`mixer="delta"`).** A DeltaNet/DeltaProduct delta-rule
  composer on the shared `mixer=` interface: a per-head `d_k × d_v` matrix
  associative-memory state updated by `nh` generalized-Householder rank-one
  corrections per token, a chunked-WY parallel `forward` (`chunk_size` is a
  performance-only knob; results are invariant), and a recurrent `step` that
  returns `(y_t, h_t)` since the readout is not the carried state. State
  capacity (`n_heads`, `nh`, `d_k`, `d_v`) is independent of `d_model`.
- **Time decay on `DeltaMinGRU`.** The shared decay contract (`decay="fixed"` /
  `"learnable"`, `decay_rate`, `log1p_delta`): a Gated-DeltaNet-style per-token
  per-head gate folded into the chunked-WY parallel form via a log-space
  decay-ratio reparameterization, so `decay=None` stays bit-identical to the
  decay-free path. Very large or `+inf` gaps under float32 are handled by an
  internal saturation clamp plus a once-per-instance warning. The decay-active
  forward is eager-only; `torch.compile` is the recommended CUDA path.
- **Delta Triton kernel.** A hand-written chunked-WY forward kernel, gated under
  `MINGRU_SCAN=auto` to a measured win region (long sequences, narrow head
  dims); `torch.compile` is recommended outside it, and explicit
  `MINGRU_SCAN=triton` fails loud where unsupported.
- **Benchmark validation.** An accepted-benchmark round on four public tasks
  (MQAR associative recall, psMNIST, S5 group composition, and a pendulum
  irregular-time control) with a depth-matched GRU control, published as a
  results page and choose-a-mixer tables, with CPU and L4-GPU strata disclosed.
- **Documentation.** A dedicated Delta variant section, a "time-aware decay on
  the delta-rule memory" explanation, an "enable time decay" how-to, a "your
  first delta model" tutorial, the benchmark-validation results page, and a
  benchmark-reproduction guide.

## [0.1.0] - 2026-07-17

Initial release of `mingru-scans`.

### Added

- **Library (`mingru`).** Parallel-scan minGRU variants with a shared
  `mixer=` interface on `MinGRUBlock`/`MinGRUStack`: `MinGRU` (log-space),
  `SignedMinGRU` (sign-flipping state), `RotationMinGRU` (non-commutative 2D
  rotation transitions), and `GivensMinGRU` (block-rotation transitions). The
  four scan primitives (`parallel_scan_log`, `linear_scan`, `matrix_scan`,
  `matrix_affine_scan`) and an optional time-aware decay term are included.
  Ships a `py.typed` marker.
- **Triton scan backend (`mingru.triton_scans`).** Optional GPU scan kernels,
  imported lazily via PEP 562 so `import mingru` stays Triton-free on CPU-only
  installs. `MINGRU_SCAN` selects the implementation (`auto` | `eager` |
  `triton`) with a warn-once eager fallback.
- **Documentation.** Diátaxis docs site (tutorials, how-to guides, API
  reference generated from NumPy docstrings, and explanation deep dives on
  GivensGRU and the Triton scan kernels).
- **Tests.** CPU-only pytest suite covering scan ops, mixers, dispatch
  semantics, the packaging surface, and docstring coverage, alongside the
  preserved `__main__` selftests in the root evidence drivers.
- **Release tooling.** Docs deploy on push to main (`docs.yml`) and PyPI
  trusted-publishing on GitHub Release (`publish.yaml`); a `check_wheel.sh`
  wheel-and-install proof against a fresh venv; nox sessions (`lint`,
  `freeze`, `test`, `doctests`, `evidence`, `wheel`, `gpu`, `build`, `docs`) as the
  local gate suite run before every release.
- **Optional `[triton]` extra.** Unpinned convenience extra for CUDA-capable
  torch builds from channels that omit triton; torch's own Linux CUDA wheels
  already bundle the matching triton.
- **Evidence artifacts.** Committed division-reversal error emulation
  (`experiments/reversal_emulation.py` + `experiments/bench/` artifacts)
  backing the exact stored-state backward decision, alongside the GPU
  conformance (590/590), benchmark, and memory artifacts.
- **Slides.** The parallel-GRUs deck and appendix published on the docs site.

[0.2.0]: https://github.com/chris-santiago/minGRU/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/chris-santiago/minGRU/releases/tag/v0.1.0
