# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

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
- **Release tooling.** CPU test CI (`tests.yaml`), docs deploy (`docs.yml`),
  and PyPI trusted-publishing on release (`publish.yaml`); a `check_wheel.sh`
  wheel-and-install proof against a fresh venv.

[0.1.0]: https://github.com/chris-santiago/minGRU/releases/tag/v0.1.0
