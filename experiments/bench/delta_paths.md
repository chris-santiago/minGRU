# DeltaMinGRU forward-path bench (chunked-WY vs naive affine-scan vs sequential)

Uncontended CPU forward+backward timing (1 discarded warmup, min of 3 timed iterations), per spec section 9.9 / intent ledger statement 4 -- validates the chunked-WY efficiency claim by measurement.

Config: input_size=64, hidden_size=64, n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64, seed=0. GivensMinGRU reference arm: block_size=8, rounds=3, same input_size/hidden_size.

torch 2.5.1 (evidence pin, deliberately chosen -- see below), CPU: Apple M2 Pro (6 torch threads), macOS-15.7.5-arm64-arm-64bit, commit 5e8d8147d09897deb1d81d12a350ab4e3af2bf10, generated 2026-07-18T03:42:59.879631+00:00.

Environment note: this run is pinned to torch==2.5.1 (asserted at runtime; the script refuses to write this artifact under any other torch version) so every number below sits on the same torch version and machine as the previously-recorded evidence in `experiments/EXPERIMENTS.md`'s `hetero-loop-17/18` round (GivensMinGRU 0.961s, sequential delta16 0.179s at the lab shape) -- directly comparable, not merely contextual. The packaged `mingru` distribution's declared `torch>=2.8` floor (`pyproject.toml`) is install/packaging metadata for the separately-gated Triton GPU kernel surface; the eager CPU path every arm below exercises runs under the pin by design.

Agreement gate (forward-only, before any timing): all pairwise comparisons among the three delta arms passed at atol=1e-5, at T=64 (full bench config) and at a ragged shape (B=4, T=13, chunk_size=5) -- see console output for the per-pair max abs diffs. GivensMinGRU is excluded from this gate (a different mixer computing a different function).

| arm | T=64 fwd+bwd (s) | T=1024 fwd+bwd (s) |
| --- | --- | --- |
| sequential step-loop (oracle) | 0.1617 | 19.9742 |
| naive affine-scan reduction (oracle) | 2.0514 | 71.7562 |
| chunked-WY (shipped forward) | 0.0577 | 1.4541 |
| GivensMinGRU (packaged, cross-mixer reference) | 0.9493 | 24.9996 |

Ratios (chunked-WY speedup, delta arms only):
- vs naive affine-scan: 35.55x at T=64, 49.35x at T=1024
- vs sequential step-loop: 2.80x at T=64, 13.74x at T=1024

Cross-mixer reference (not part of the delta comparison above): GivensMinGRU / chunked-WY time ratio 16.45x at T=64, 17.19x at T=1024 (>1 means GivensMinGRU is slower than chunked-WY DeltaMinGRU on this run; not a like-for-like comparison -- different mixer, different math -- included only for same-environment context against the torch-2.5.1-era 0.961s figure).

Acceptance (spec section 9.9, delta arms only): chunked-WY beats naive affine-scan at both shapes -- PASS.
