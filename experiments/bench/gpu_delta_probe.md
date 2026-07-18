# CUDA fusion-headroom probe: DeltaMinGRU chunked-WY (Task 6)

Eager chunked-WY forward+backward vs. its matmul-FLOP floor (approximate, see each row's `floor_method`) vs. `torch.compile`, per `.git/sdd/task-6-brief.md` / `scripts/gpu_delta_probe.py`. This is a GPU evidence stratum: nothing below is comparable to the pinned-CPU rows in `experiments/lab_results.jsonl` / `EXPERIMENTS.md`.

Env: torch 2.8.0+cu128 (CUDA 12.8), device NVIDIA L4 (capability [8, 9]), triton 3.4.0, platform Linux-6.8.0-1058-gcp-x86_64-with-glibc2.39, B=128, warmup=3, timed steps=10, generated 2026-07-18T09:57:13.187222+00:00.

| shape | config | B | T | eager median (s) | floor (s, approx) | compile median (s) | compile status | headroom (eager/floor) | compile recovered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pd1024_T64 | pd1024: n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64, state_elements=1024 | 128 | 64 | 0.0074 | 0.0028 | 0.0041 | ok | 2.60 | 71.63% |
| pd1024_T256 | pd1024: n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64, state_elements=1024 | 128 | 256 | 0.0276 | 0.0127 | 0.0154 | ok | 2.18 | 81.73% |
| pd1024_T1024 | pd1024: n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64, state_elements=1024 | 128 | 1024 | 0.2098 | 0.0539 | 0.0680 | ok | 3.89 | 90.96% |
| stepup_T256 | stepup: n_heads=4, nh=2, d_k=64, d_v=64, chunk_size=64, state_elements=16384 | 128 | 256 | 0.0846 | 0.0259 | 0.0436 | ok | 3.27 | 69.91% |
| stepup_T1024 | stepup: n_heads=4, nh=2, d_k=64, d_v=64, chunk_size=64, state_elements=16384 | 128 | 1024 | 0.6992 | 0.1050 | 0.1894 | ok | 6.66 | 85.80% |

`floor` rows are an explicit approximation of the dominant GEMM/triangular-solve contractions only, scaled by the standard 3x fwd-GEMM convention -- see each shape's `floor_method` in the JSON artifact for the full disclosure. A `compile status` other than `ok` means Inductor failed on that shape; see the JSON artifact's `compile_error` for that row.

