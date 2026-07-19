# CUDA probe snapshot: fused-Triton-backward experiment (L4, reverted)

Provenance: verbatim copy of `experiments/bench/gpu_delta_probe.md` as generated 2026-07-19T05:36:38.414221+00:00 by the delta-probe job at the fused-backward configuration (commit 7de1be3 lineage: fused backward + SMEM tiling), preserved by the orchestrator before the artifact was regenerated on the shipping configuration (revert commit 5955b15). Recovered from the orchestrator's read of the artifact, not re-measured. This is the negative-result evidence for the fused-backward reversion; the corresponding JSON was not preserved.

Env: torch 2.8.0+cu128 (CUDA 12.8), device NVIDIA L4 (capability [8, 9]), triton 3.4.0, platform Linux-6.8.0-1058-gcp-x86_64-with-glibc2.39, B=128, warmup=3, timed steps=10.

| shape | config | B | T | eager median (s) | floor (s, approx) | compile median (s) | compile status | triton median (s) | headroom (eager/floor) | compile recovered | triton/compile | triton/eager | bar met |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pd1024_T64 | pd1024: n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64, state_elements=1024 | 128 | 64 | 0.0063 | 0.0029 | 0.0037 | ok | 0.0358 | 2.16 | 77.77% | 9.77 | 5.69 | FAIL |
| pd1024_T256 | pd1024: n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64, state_elements=1024 | 128 | 256 | 0.0270 | 0.0120 | 0.0152 | ok | 0.1376 | 2.25 | 78.75% | 9.06 | 5.10 | FAIL |
| pd1024_T1024 | pd1024: n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64, state_elements=1024 | 128 | 1024 | 0.2100 | 0.0535 | 0.0685 | ok | 0.5490 | 3.92 | 90.45% | 8.02 | 2.61 | FAIL |
| stepup_T256 | stepup: n_heads=4, nh=2, d_k=64, d_v=64, chunk_size=64, state_elements=16384 | 128 | 256 | 0.0842 | 0.0262 | 0.0434 | ok | 0.5198 | 3.21 | 70.46% | 11.98 | 6.17 | FAIL |
| stepup_T1024 | stepup: n_heads=4, nh=2, d_k=64, d_v=64, chunk_size=64, state_elements=16384 | 128 | 1024 | 0.6950 | 0.1048 | 0.1920 | ok | 2.0904 | 6.63 | 85.24% | 10.89 | 3.01 | FAIL |

| shape | eager peak mem (MB) | triton peak mem (MB) | triton/eager peak mem | memory bar met |
| --- | --- | --- | --- | --- |
| pd1024_T64 | 186.9 | 81.1 | 0.43 | PASS |
| pd1024_T256 | 677.7 | 268.3 | 0.40 | PASS |
| pd1024_T1024 | 2663.8 | 1017.0 | 0.38 | PASS |
| stepup_T256 | 1467.2 | 1092.6 | 0.74 | PASS |
| stepup_T1024 | 5812.6 | 4238.4 | 0.73 | PASS |
