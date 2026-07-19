# CUDA fusion-headroom probe: DeltaMinGRU chunked-WY (Task 6)

Eager chunked-WY forward+backward vs. its matmul-FLOP floor (approximate, see each row's `floor_method`) vs. `torch.compile` vs. the Triton chunked-WY kernel, per `.git/sdd/task-6-brief.md` / `scripts/gpu_delta_probe.py`. This is a GPU evidence stratum: nothing below is comparable to the pinned-CPU rows in `experiments/lab_results.jsonl` / `EXPERIMENTS.md`.

Env: torch 2.8.0+cu128 (CUDA 12.8), device NVIDIA A100-SXM4-80GB (capability [8, 0]), triton 3.4.0, platform Linux-6.8.0-1058-gcp-x86_64-with-glibc2.39, B=128, warmup=3, timed steps=10, generated 2026-07-19T11:18:08.031085+00:00.

| shape | config | B | T | eager median (s) | floor (s, approx) | compile median (s) | compile status | triton median (s) | headroom (eager/floor) | compile recovered | triton/compile | triton/eager | bar met |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pd1024_T64 | pd1024: n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64, state_elements=1024 | 128 | 64 | 0.0067 | 0.0015 | 0.0037 | ok | 0.0457 | 4.62 | 58.30% | 12.46 | 6.77 | FAIL |
| pd1024_T256 | pd1024: n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64, state_elements=1024 | 128 | 256 | 0.0176 | 0.0056 | 0.0098 | ok | 0.1659 | 3.16 | 64.71% | 16.92 | 9.44 | FAIL |
| pd1024_T1024 | pd1024: n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64, state_elements=1024 | 128 | 1024 | 0.0612 | 0.0214 | 0.0313 | ok | 0.6462 | 2.87 | 74.95% | 20.62 | 10.56 | FAIL |
| stepup_T256 | stepup: n_heads=4, nh=2, d_k=64, d_v=64, chunk_size=64, state_elements=16384 | 128 | 256 | 0.0272 | 0.0112 | 0.0184 | ok | 0.2865 | 2.43 | 54.85% | 15.55 | 10.53 | FAIL |
| stepup_T1024 | stepup: n_heads=4, nh=2, d_k=64, d_v=64, chunk_size=64, state_elements=16384 | 128 | 1024 | 0.1469 | 0.0444 | 0.0692 | ok | 1.1295 | 3.31 | 75.78% | 16.32 | 7.69 | FAIL |

`floor` rows are an explicit approximation of the dominant GEMM/triangular-solve contractions only, scaled by the standard 3x fwd-GEMM convention -- see each shape's `floor_method` in the JSON artifact for the full disclosure. A `compile status` other than `ok` means Inductor failed on that shape; see the JSON artifact's `compile_error` for that row. `bar met` is the spec section 9.1 speed bar (triton fwd+bwd median <= 1.2x the recorded compile median AND <= the recorded eager median, both judged on this same run's own medians) -- `n/a` means the vs-compile leg couldn't be judged because that shape's compile arm failed, not that the bar failed.

| shape | eager peak mem (MB) | triton peak mem (MB) | triton/eager peak mem | memory bar met |
| --- | --- | --- | --- | --- |
| pd1024_T64 | 186.9 | 68.0 | 0.36 | PASS |
| pd1024_T256 | 677.7 | 215.9 | 0.32 | PASS |
| pd1024_T1024 | 2663.8 | 807.3 | 0.30 | PASS |
| stepup_T256 | 1467.2 | 1301.7 | 0.89 | PASS |
| stepup_T1024 | 5812.6 | 4343.6 | 0.75 | PASS |

`memory bar met` is the spec section 7 memory invariant (triton-path peak training memory <= eager-path peak training memory), judged per shape from `eager_peak_mem_bytes`/`triton_peak_mem_bytes` in the JSON artifact.

