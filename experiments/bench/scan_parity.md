# Scan kernel parity conformance

torch 2.8.0+cu128, triton 3.4.0, device NVIDIA L4, driver 580.159.03, commit 72ffe5b1df9a1b7fa90189ae544dc61a96909058, generated 2026-07-16T22:40:07.649760+00:00.

| op | shape | dtype | direction | gate atol | gate rtol | max abs err | max rel err | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parallel_scan_log | B=2 T=1 D=64 | float32 | fwd | 1e-04 | 1e-04 | 2.38e-07 | 1.89e-07 | PASS |
| parallel_scan_log | B=2 T=13 D=64 | float32 | fwd | 1e-04 | 1e-04 | 1.31e-06 | 9.18e-07 | PASS |
| parallel_scan_log | B=2 T=64 D=64 | float32 | fwd | 1e-04 | 1e-04 | 6.20e-06 | 4.89e-06 | PASS |
| parallel_scan_log | B=2 T=128 D=64 | float32 | fwd | 1e-04 | 1e-04 | 1.55e-05 | 9.28e-06 | PASS |
| parallel_scan_log | B=2 T=1024 D=64 | float32 | fwd | 1e-04 | 1e-04 | 1.31e-04 | 1.06e-04 | PASS |
| parallel_scan_log | B=128 T=1 D=64 | float32 | fwd | 1e-04 | 1e-04 | 7.15e-07 | 2.96e-07 | PASS |
| parallel_scan_log | B=128 T=13 D=64 | float32 | fwd | 1e-04 | 1e-04 | 2.15e-06 | 1.40e-06 | PASS |
| parallel_scan_log | B=128 T=64 D=64 | float32 | fwd | 1e-04 | 1e-04 | 8.58e-06 | 6.22e-06 | PASS |
| parallel_scan_log | B=128 T=128 D=64 | float32 | fwd | 1e-04 | 1e-04 | 2.10e-05 | 1.28e-05 | PASS |
| parallel_scan_log | B=128 T=1024 D=64 | float32 | fwd | 1e-04 | 1e-04 | 1.78e-04 | 1.09e-04 | PASS |
| linear_scan | B=2 T=1 D=64 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| linear_scan | B=2 T=13 D=64 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 5.42e-06 | PASS |
| linear_scan | B=2 T=64 D=64 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.52e-03 | PASS |
| linear_scan | B=2 T=128 D=64 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.51e-04 | PASS |
| linear_scan | B=2 T=1024 D=64 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.84e-03 | PASS |
| linear_scan | B=128 T=1 D=64 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| linear_scan | B=128 T=13 D=64 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.38e-03 | PASS |
| linear_scan | B=128 T=64 D=64 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.84e-03 | PASS |
| linear_scan | B=128 T=128 D=64 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 4.60e-03 | PASS |
| linear_scan | B=128 T=1024 D=64 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 4.17e+00 | PASS |
| matrix_scan | B=2 T=1 n=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_scan | B=2 T=13 n=4 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 1.71e-05 | PASS |
| matrix_scan | B=2 T=64 n=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.06e-05 | PASS |
| matrix_scan | B=2 T=128 n=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.38e-05 | PASS |
| matrix_scan | B=2 T=1024 n=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 9.02e-04 | PASS |
| matrix_scan | B=128 T=1 n=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_scan | B=128 T=13 n=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 4.22e-01 | PASS |
| matrix_scan | B=128 T=64 n=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.87e-03 | PASS |
| matrix_scan | B=128 T=128 n=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 1.14e-02 | PASS |
| matrix_scan | B=128 T=1024 n=4 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 6.10e-03 | PASS |
| matrix_affine_scan | B=2 T=1 k=1 v=1 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=1 v=2 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=1 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=1 v=8 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=1 v=16 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=2 v=1 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=2 v=2 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=2 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=2 v=8 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=2 v=16 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=4 v=1 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=4 v=2 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=4 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=4 v=8 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=4 v=16 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=8 v=1 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=8 v=2 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=8 v=8 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=8 v=16 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=16 v=1 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=16 v=2 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=16 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=16 v=8 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=16 v=16 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=13 k=1 v=1 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 2.35e-07 | PASS |
| matrix_affine_scan | B=2 T=13 k=1 v=2 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 1.99e-06 | PASS |
| matrix_affine_scan | B=2 T=13 k=1 v=4 | float32 | fwd | 1e-05 | 0e+00 | 1.49e-08 | 1.32e-06 | PASS |
| matrix_affine_scan | B=2 T=13 k=1 v=8 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 2.61e-06 | PASS |
| matrix_affine_scan | B=2 T=13 k=1 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 9.84e-07 | PASS |
| matrix_affine_scan | B=2 T=13 k=2 v=1 | float32 | fwd | 1e-05 | 0e+00 | 1.49e-08 | 9.36e-07 | PASS |
| matrix_affine_scan | B=2 T=13 k=2 v=2 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 7.00e-06 | PASS |
| matrix_affine_scan | B=2 T=13 k=2 v=4 | float32 | fwd | 1e-05 | 0e+00 | 4.47e-08 | 1.59e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=2 v=8 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 5.47e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=2 v=16 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 2.53e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=4 v=1 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 8.02e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=4 v=2 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 1.04e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=4 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.55e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=4 v=8 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 3.41e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=4 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.04e-03 | PASS |
| matrix_affine_scan | B=2 T=13 k=8 v=1 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 7.71e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=8 v=2 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 5.99e-03 | PASS |
| matrix_affine_scan | B=2 T=13 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 4.47e-08 | 6.94e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=8 v=8 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 7.50e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=8 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.86e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=16 v=1 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 8.94e-03 | PASS |
| matrix_affine_scan | B=2 T=13 k=16 v=2 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 1.09e-03 | PASS |
| matrix_affine_scan | B=2 T=13 k=16 v=4 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 1.55e-03 | PASS |
| matrix_affine_scan | B=2 T=13 k=16 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.28e-03 | PASS |
| matrix_affine_scan | B=2 T=13 k=16 v=16 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 1.57e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=1 v=1 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 6.15e-07 | PASS |
| matrix_affine_scan | B=2 T=64 k=1 v=2 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 3.89e-06 | PASS |
| matrix_affine_scan | B=2 T=64 k=1 v=4 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 2.76e-06 | PASS |
| matrix_affine_scan | B=2 T=64 k=1 v=8 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 2.36e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=1 v=16 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 7.40e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=2 v=1 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 4.23e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=2 v=2 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 9.37e-06 | PASS |
| matrix_affine_scan | B=2 T=64 k=2 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.37e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=2 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.08e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=2 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 6.23e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=4 v=1 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 9.31e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=4 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.83e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=4 v=4 | float32 | fwd | 1e-05 | 0e+00 | 4.47e-08 | 4.01e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=4 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.46e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=4 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.30e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=8 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.34e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=8 v=2 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 6.75e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 6.45e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=8 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.41e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=8 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.32e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=16 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.21e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=16 v=2 | float32 | fwd | 1e-05 | 0e+00 | 3.73e-08 | 1.39e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=16 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.29e-02 | PASS |
| matrix_affine_scan | B=2 T=64 k=16 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.71e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=16 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.23e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=1 v=1 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 1.10e-06 | PASS |
| matrix_affine_scan | B=2 T=128 k=1 v=2 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 2.36e-06 | PASS |
| matrix_affine_scan | B=2 T=128 k=1 v=4 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 8.17e-06 | PASS |
| matrix_affine_scan | B=2 T=128 k=1 v=8 | float32 | fwd | 1e-05 | 0e+00 | 4.47e-08 | 1.13e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=1 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.02e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=2 v=1 | float32 | fwd | 1e-05 | 0e+00 | 4.47e-08 | 1.30e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=2 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.96e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=2 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.68e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=2 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 3.34e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=2 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 7.10e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=4 v=1 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 4.74e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=4 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 8.72e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=4 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.48e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=4 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.67e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=4 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 8.89e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=8 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.66e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=8 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.33e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 8.71e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=8 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.26e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=8 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.61e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=16 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 7.71e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=16 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.37e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=16 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 4.14e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=16 v=8 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 8.83e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=16 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.49e-07 | 1.45e-01 | PASS |
| matrix_affine_scan | B=2 T=1024 k=1 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 8.76e-06 | PASS |
| matrix_affine_scan | B=2 T=1024 k=1 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.64e-05 | PASS |
| matrix_affine_scan | B=2 T=1024 k=1 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.32e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=1 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.78e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=1 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 7.09e-05 | PASS |
| matrix_affine_scan | B=2 T=1024 k=2 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.04e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=2 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.82e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=2 v=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 2.37e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=2 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.40e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=2 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 2.76e-02 | PASS |
| matrix_affine_scan | B=2 T=1024 k=4 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 8.78e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=4 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 4.99e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=4 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.89e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=4 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 6.70e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=4 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 9.62e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=8 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 6.08e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=8 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.98e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 5.71e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=8 v=8 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 8.91e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=8 v=16 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 3.62e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=16 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 6.14e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=16 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.77e-02 | PASS |
| matrix_affine_scan | B=2 T=1024 k=16 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 8.03e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=16 v=8 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 4.60e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=16 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.49e-07 | 3.10e-01 | PASS |
| matrix_affine_scan | B=128 T=1 k=1 v=1 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=1 v=2 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=1 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=1 v=8 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=1 v=16 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=2 v=1 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=2 v=2 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=2 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=2 v=8 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=2 v=16 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=4 v=1 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=4 v=2 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=4 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=4 v=8 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=4 v=16 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=8 v=1 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=8 v=2 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=8 v=8 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=8 v=16 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=16 v=1 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=16 v=2 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=16 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=16 v=8 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=16 v=16 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=13 k=1 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.02e-06 | PASS |
| matrix_affine_scan | B=128 T=13 k=1 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 8.59e-06 | PASS |
| matrix_affine_scan | B=128 T=13 k=1 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 4.45e-05 | PASS |
| matrix_affine_scan | B=128 T=13 k=1 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.20e-04 | PASS |
| matrix_affine_scan | B=128 T=13 k=1 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.02e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=2 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.07e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=2 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 8.57e-05 | PASS |
| matrix_affine_scan | B=128 T=13 k=2 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.55e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=2 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.50e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=2 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 8.46e-04 | PASS |
| matrix_affine_scan | B=128 T=13 k=4 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 4.41e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=4 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 6.92e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=4 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.82e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=4 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.11e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=4 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.49e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=8 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 7.28e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=8 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.74e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.46e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=8 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 4.72e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=8 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.79e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=16 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.20e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=16 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.63e-01 | PASS |
| matrix_affine_scan | B=128 T=13 k=16 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.83e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=16 v=8 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 2.76e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=16 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 4.25e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=1 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 6.88e-05 | PASS |
| matrix_affine_scan | B=128 T=64 k=1 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.37e-04 | PASS |
| matrix_affine_scan | B=128 T=64 k=1 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.48e-04 | PASS |
| matrix_affine_scan | B=128 T=64 k=1 v=8 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.86e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=1 v=16 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 4.86e-04 | PASS |
| matrix_affine_scan | B=128 T=64 k=2 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.15e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=2 v=2 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.25e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=2 v=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 1.92e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=2 v=8 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 2.22e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=2 v=16 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 4.63e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=4 v=1 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 2.22e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=4 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 4.83e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=4 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.24e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=4 v=8 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 1.25e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=4 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 9.28e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=8 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.56e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=8 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.06e-01 | PASS |
| matrix_affine_scan | B=128 T=64 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 1.84e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=8 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 4.40e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=8 v=16 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 6.82e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=16 v=1 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 2.31e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=16 v=2 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 5.76e-01 | PASS |
| matrix_affine_scan | B=128 T=64 k=16 v=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 7.17e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=16 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 4.96e-01 | PASS |
| matrix_affine_scan | B=128 T=64 k=16 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.79e-07 | 7.89e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=1 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.63e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=1 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 3.22e-04 | PASS |
| matrix_affine_scan | B=128 T=128 k=1 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 2.37e-04 | PASS |
| matrix_affine_scan | B=128 T=128 k=1 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.01e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=1 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 3.14e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=2 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 7.16e-04 | PASS |
| matrix_affine_scan | B=128 T=128 k=2 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 3.31e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=2 v=4 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 2.96e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=2 v=8 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 1.35e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=2 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 5.96e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=4 v=1 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 5.38e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=4 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.00e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=4 v=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 4.29e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=4 v=8 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 1.00e+00 | PASS |
| matrix_affine_scan | B=128 T=128 k=4 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 3.89e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=8 v=1 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 1.87e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=8 v=2 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 1.31e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 3.91e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=8 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 2.28e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=8 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.83e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=16 v=1 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 1.43e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=16 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 2.41e+00 | PASS |
| matrix_affine_scan | B=128 T=128 k=16 v=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 5.23e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=16 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 2.20e+00 | PASS |
| matrix_affine_scan | B=128 T=128 k=16 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.79e-07 | 2.12e+00 | PASS |
| matrix_affine_scan | B=128 T=1024 k=1 v=1 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 4.37e-04 | PASS |
| matrix_affine_scan | B=128 T=1024 k=1 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 5.47e-03 | PASS |
| matrix_affine_scan | B=128 T=1024 k=1 v=4 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 4.98e-03 | PASS |
| matrix_affine_scan | B=128 T=1024 k=1 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 4.90e-03 | PASS |
| matrix_affine_scan | B=128 T=1024 k=1 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.20e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=2 v=1 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 9.06e-04 | PASS |
| matrix_affine_scan | B=128 T=1024 k=2 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.27e-07 | 1.45e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=2 v=4 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 7.27e-03 | PASS |
| matrix_affine_scan | B=128 T=1024 k=2 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 5.34e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=2 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.88e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=4 v=1 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 5.48e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=4 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 5.97e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=4 v=4 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 6.89e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=4 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 2.78e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=4 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.16e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=8 v=1 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 5.33e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=8 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 7.67e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 2.14e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=8 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 4.06e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=8 v=16 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.95e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=16 v=1 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 3.45e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=16 v=2 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.32e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=16 v=4 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.30e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=16 v=8 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.46e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=16 v=16 | float32 | fwd | 1e-05 | 0e+00 | 2.38e-07 | 4.16e-01 | PASS |
| matrix_affine_scan | B=2 T=1 n=3 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=13 n=3 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 2.98e-08 | 1.15e-03 | PASS |
| matrix_affine_scan | B=2 T=64 n=3 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 1.17e-03 | PASS |
| matrix_affine_scan | B=2 T=128 n=3 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 3.12e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 n=3 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 3.26e-02 | PASS |
| matrix_affine_scan | B=128 T=1 n=3 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=13 n=3 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 5.96e-08 | 9.91e-02 | PASS |
| matrix_affine_scan | B=128 T=64 n=3 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 8.94e-08 | 4.13e-02 | PASS |
| matrix_affine_scan | B=128 T=128 n=3 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 7.04e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 n=3 k=8 v=4 | float32 | fwd | 1e-05 | 0e+00 | 1.19e-07 | 1.07e-01 | PASS |
| parallel_scan_log | B=2 T=1 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| parallel_scan_log | B=2 T=13 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| parallel_scan_log | B=2 T=64 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| parallel_scan_log | B=2 T=128 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| parallel_scan_log | B=2 T=1024 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| parallel_scan_log | B=128 T=1 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| parallel_scan_log | B=128 T=13 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| parallel_scan_log | B=128 T=64 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| parallel_scan_log | B=128 T=128 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| parallel_scan_log | B=128 T=1024 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| linear_scan | B=2 T=1 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| linear_scan | B=2 T=13 D=64 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.44e-04 | PASS |
| linear_scan | B=2 T=64 D=64 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 4.98e-04 | PASS |
| linear_scan | B=2 T=128 D=64 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 4.98e-04 | PASS |
| linear_scan | B=2 T=1024 D=64 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 9.82e-03 | PASS |
| linear_scan | B=128 T=1 D=64 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| linear_scan | B=128 T=13 D=64 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.35e-03 | PASS |
| linear_scan | B=128 T=64 D=64 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 4.18e-02 | PASS |
| linear_scan | B=128 T=128 D=64 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 4.68e-01 | PASS |
| linear_scan | B=128 T=1024 D=64 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 4.44e-01 | PASS |
| matrix_scan | B=2 T=1 n=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_scan | B=2 T=13 n=4 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 5.12e-05 | PASS |
| matrix_scan | B=2 T=64 n=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 4.11e-04 | PASS |
| matrix_scan | B=2 T=128 n=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 6.61e-04 | PASS |
| matrix_scan | B=2 T=1024 n=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 6.61e-04 | PASS |
| matrix_scan | B=128 T=1 n=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_scan | B=128 T=13 n=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.20e-03 | PASS |
| matrix_scan | B=128 T=64 n=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 8.43e-03 | PASS |
| matrix_scan | B=128 T=128 n=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.87e-02 | PASS |
| matrix_scan | B=128 T=1024 n=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 9.80e-01 | PASS |
| matrix_affine_scan | B=2 T=1 k=1 v=1 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=1 v=2 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=1 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=1 v=8 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=1 v=16 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=2 v=1 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=2 v=2 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=2 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=2 v=8 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=2 v=16 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=4 v=1 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=4 v=2 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=4 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=4 v=8 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=4 v=16 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=8 v=1 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=8 v=2 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=8 v=8 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=8 v=16 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=16 v=1 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=16 v=2 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=16 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=16 v=8 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=1 k=16 v=16 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=13 k=1 v=1 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 2.24e-06 | PASS |
| matrix_affine_scan | B=2 T=13 k=1 v=2 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 6.79e-07 | PASS |
| matrix_affine_scan | B=2 T=13 k=1 v=4 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 5.42e-06 | PASS |
| matrix_affine_scan | B=2 T=13 k=1 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 6.36e-06 | PASS |
| matrix_affine_scan | B=2 T=13 k=1 v=16 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 7.49e-06 | PASS |
| matrix_affine_scan | B=2 T=13 k=2 v=1 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 1.87e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=2 v=2 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 1.69e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=2 v=4 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 2.69e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=2 v=8 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 7.33e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=2 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.46e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=4 v=1 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 4.85e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=4 v=2 | float32 | grad | 1e-03 | 0e+00 | 3.58e-07 | 1.67e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=4 v=4 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 1.13e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=4 v=8 | float32 | grad | 1e-03 | 0e+00 | 2.98e-07 | 7.83e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=4 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.65e-03 | PASS |
| matrix_affine_scan | B=2 T=13 k=8 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.07e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=8 v=2 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 4.73e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.59e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=8 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.97e-05 | PASS |
| matrix_affine_scan | B=2 T=13 k=8 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.69e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=16 v=1 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 1.01e-03 | PASS |
| matrix_affine_scan | B=2 T=13 k=16 v=2 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 6.42e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=16 v=4 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 3.95e-03 | PASS |
| matrix_affine_scan | B=2 T=13 k=16 v=8 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 4.40e-04 | PASS |
| matrix_affine_scan | B=2 T=13 k=16 v=16 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 2.29e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=1 v=1 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 3.94e-06 | PASS |
| matrix_affine_scan | B=2 T=64 k=1 v=2 | float32 | grad | 1e-03 | 0e+00 | 2.98e-07 | 5.85e-06 | PASS |
| matrix_affine_scan | B=2 T=64 k=1 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.49e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=1 v=8 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 7.47e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=1 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.43e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=2 v=1 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 1.52e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=2 v=2 | float32 | grad | 1e-03 | 0e+00 | 3.58e-07 | 2.72e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=2 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.92e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=2 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.18e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=2 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.72e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=4 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 4.99e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=4 v=2 | float32 | grad | 1e-03 | 0e+00 | 3.58e-07 | 9.89e-05 | PASS |
| matrix_affine_scan | B=2 T=64 k=4 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.95e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=4 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.86e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=4 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.98e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=8 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.53e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=8 v=2 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.84e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 5.42e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=8 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.09e-04 | PASS |
| matrix_affine_scan | B=2 T=64 k=8 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.21e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=16 v=1 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 2.59e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=16 v=2 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 1.00e-01 | PASS |
| matrix_affine_scan | B=2 T=64 k=16 v=4 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 9.82e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=16 v=8 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 1.11e-03 | PASS |
| matrix_affine_scan | B=2 T=64 k=16 v=16 | float32 | grad | 1e-03 | 0e+00 | 1.43e-06 | 3.11e-02 | PASS |
| matrix_affine_scan | B=2 T=128 k=1 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.71e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=1 v=2 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 8.54e-06 | PASS |
| matrix_affine_scan | B=2 T=128 k=1 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.49e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=1 v=8 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 7.47e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=1 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.03e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=2 v=1 | float32 | grad | 1e-03 | 0e+00 | 2.38e-07 | 5.23e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=2 v=2 | float32 | grad | 1e-03 | 0e+00 | 3.58e-07 | 8.43e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=2 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.92e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=2 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 9.57e-05 | PASS |
| matrix_affine_scan | B=2 T=128 k=2 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.72e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=4 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.52e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=4 v=2 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 4.58e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=4 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.95e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=4 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.53e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=4 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 5.92e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=8 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 7.92e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=8 v=2 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 4.64e-04 | PASS |
| matrix_affine_scan | B=2 T=128 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 4.24e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=8 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.47e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=8 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 6.21e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=16 v=1 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 1.05e-02 | PASS |
| matrix_affine_scan | B=2 T=128 k=16 v=2 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 1.00e-01 | PASS |
| matrix_affine_scan | B=2 T=128 k=16 v=4 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 2.22e-02 | PASS |
| matrix_affine_scan | B=2 T=128 k=16 v=8 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 6.80e-03 | PASS |
| matrix_affine_scan | B=2 T=128 k=16 v=16 | float32 | grad | 1e-03 | 0e+00 | 1.43e-06 | 3.11e-02 | PASS |
| matrix_affine_scan | B=2 T=1024 k=1 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.58e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=1 v=2 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.33e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=1 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 8.51e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=1 v=8 | float32 | grad | 1e-03 | 0e+00 | 5.36e-07 | 3.56e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=1 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 5.59e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=2 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.04e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=2 v=2 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.57e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=2 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 5.61e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=2 v=8 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 8.00e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=2 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 9.85e-04 | PASS |
| matrix_affine_scan | B=2 T=1024 k=4 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 4.62e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=4 v=2 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.15e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=4 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.95e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=4 v=8 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 2.66e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=4 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.18e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=8 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.47e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=8 v=2 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 9.09e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 3.41e-02 | PASS |
| matrix_affine_scan | B=2 T=1024 k=8 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.24e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=8 v=16 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 8.55e-03 | PASS |
| matrix_affine_scan | B=2 T=1024 k=16 v=1 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 2.49e-02 | PASS |
| matrix_affine_scan | B=2 T=1024 k=16 v=2 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 7.68e-02 | PASS |
| matrix_affine_scan | B=2 T=1024 k=16 v=4 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 6.38e-02 | PASS |
| matrix_affine_scan | B=2 T=1024 k=16 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.11e-01 | PASS |
| matrix_affine_scan | B=2 T=1024 k=16 v=16 | float32 | grad | 1e-03 | 0e+00 | 1.91e-06 | 2.71e-02 | PASS |
| matrix_affine_scan | B=128 T=1 k=1 v=1 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=1 v=2 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=1 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=1 v=8 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=1 v=16 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=2 v=1 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=2 v=2 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=2 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=2 v=8 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=2 v=16 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=4 v=1 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=4 v=2 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=4 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=4 v=8 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=4 v=16 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=8 v=1 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=8 v=2 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=8 v=8 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=8 v=16 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=16 v=1 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=16 v=2 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=16 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=16 v=8 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1 k=16 v=16 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=13 k=1 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 5.84e-05 | PASS |
| matrix_affine_scan | B=128 T=13 k=1 v=2 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 6.84e-05 | PASS |
| matrix_affine_scan | B=128 T=13 k=1 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 6.59e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=1 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.68e-04 | PASS |
| matrix_affine_scan | B=128 T=13 k=1 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.62e-04 | PASS |
| matrix_affine_scan | B=128 T=13 k=2 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.57e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=2 v=2 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 1.57e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=2 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.57e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=2 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.69e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=2 v=16 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.57e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=4 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 4.12e-04 | PASS |
| matrix_affine_scan | B=128 T=13 k=4 v=2 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 3.55e-04 | PASS |
| matrix_affine_scan | B=128 T=13 k=4 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.06e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=4 v=8 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 2.00e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=4 v=16 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 2.34e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=8 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.37e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=8 v=2 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.06e-03 | PASS |
| matrix_affine_scan | B=128 T=13 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.30e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=8 v=8 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.42e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=8 v=16 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 2.80e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=16 v=1 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.13e-01 | PASS |
| matrix_affine_scan | B=128 T=13 k=16 v=2 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.89e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=16 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.93e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=16 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 8.65e-02 | PASS |
| matrix_affine_scan | B=128 T=13 k=16 v=16 | float32 | grad | 1e-03 | 0e+00 | 1.19e-06 | 1.26e+00 | PASS |
| matrix_affine_scan | B=128 T=64 k=1 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 6.48e-04 | PASS |
| matrix_affine_scan | B=128 T=64 k=1 v=2 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 1.33e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=1 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.13e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=1 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.56e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=1 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 7.61e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=2 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 1.39e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=2 v=2 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 7.00e-04 | PASS |
| matrix_affine_scan | B=128 T=64 k=2 v=4 | float32 | grad | 1e-03 | 0e+00 | 5.96e-07 | 2.92e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=2 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 8.00e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=2 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.56e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=4 v=1 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 5.03e+00 | PASS |
| matrix_affine_scan | B=128 T=64 k=4 v=2 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.70e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=4 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.95e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=4 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.23e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=4 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 8.47e-03 | PASS |
| matrix_affine_scan | B=128 T=64 k=8 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.80e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=8 v=2 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.01e-01 | PASS |
| matrix_affine_scan | B=128 T=64 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 1.32e-01 | PASS |
| matrix_affine_scan | B=128 T=64 k=8 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 7.69e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=8 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.68e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=16 v=1 | float32 | grad | 1e-03 | 0e+00 | 1.43e-06 | 1.13e-01 | PASS |
| matrix_affine_scan | B=128 T=64 k=16 v=2 | float32 | grad | 1e-03 | 0e+00 | 1.19e-06 | 5.72e-02 | PASS |
| matrix_affine_scan | B=128 T=64 k=16 v=4 | float32 | grad | 1e-03 | 0e+00 | 1.19e-06 | 1.11e-01 | PASS |
| matrix_affine_scan | B=128 T=64 k=16 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.96e-01 | PASS |
| matrix_affine_scan | B=128 T=64 k=16 v=16 | float32 | grad | 1e-03 | 0e+00 | 1.91e-06 | 5.96e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=1 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 2.54e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=1 v=2 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 2.01e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=1 v=4 | float32 | grad | 1e-03 | 0e+00 | 5.96e-07 | 1.15e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=1 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.56e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=1 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 7.61e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=2 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 7.43e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=2 v=2 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.36e-03 | PASS |
| matrix_affine_scan | B=128 T=128 k=2 v=4 | float32 | grad | 1e-03 | 0e+00 | 5.96e-07 | 2.50e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=2 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.20e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=2 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.56e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=4 v=1 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.25e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=4 v=2 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 2.70e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=4 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.69e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=4 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.23e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=4 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.27e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=8 v=1 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.44e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=8 v=2 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.67e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.38e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=8 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 9.67e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=8 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 4.00e-02 | PASS |
| matrix_affine_scan | B=128 T=128 k=16 v=1 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.26e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=16 v=2 | float32 | grad | 1e-03 | 0e+00 | 1.43e-06 | 4.36e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=16 v=4 | float32 | grad | 1e-03 | 0e+00 | 1.43e-06 | 1.85e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=16 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 6.65e-01 | PASS |
| matrix_affine_scan | B=128 T=128 k=16 v=16 | float32 | grad | 1e-03 | 0e+00 | 1.91e-06 | 2.10e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=1 v=1 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 7.87e-03 | PASS |
| matrix_affine_scan | B=128 T=1024 k=1 v=2 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.88e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=1 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.28e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=1 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 6.13e-03 | PASS |
| matrix_affine_scan | B=128 T=1024 k=1 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.00e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=2 v=1 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.71e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=2 v=2 | float32 | grad | 1e-03 | 0e+00 | 1.43e-06 | 3.57e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=2 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.67e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=2 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.00e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=2 v=16 | float32 | grad | 1e-03 | 0e+00 | 1.43e-06 | 6.54e-02 | PASS |
| matrix_affine_scan | B=128 T=1024 k=4 v=1 | float32 | grad | 1e-03 | 0e+00 | 7.15e-07 | 1.71e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=4 v=2 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.56e+03 | PASS |
| matrix_affine_scan | B=128 T=1024 k=4 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 2.00e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=4 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.79e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=4 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 1.19e+05 | PASS |
| matrix_affine_scan | B=128 T=1024 k=8 v=1 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 6.71e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=8 v=2 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 7.06e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 4.29e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=8 v=8 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 4.47e+04 | PASS |
| matrix_affine_scan | B=128 T=1024 k=8 v=16 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 7.00e+00 | PASS |
| matrix_affine_scan | B=128 T=1024 k=16 v=1 | float32 | grad | 1e-03 | 0e+00 | 1.43e-06 | 6.59e+03 | PASS |
| matrix_affine_scan | B=128 T=1024 k=16 v=2 | float32 | grad | 1e-03 | 0e+00 | 1.43e-06 | 1.17e+00 | PASS |
| matrix_affine_scan | B=128 T=1024 k=16 v=4 | float32 | grad | 1e-03 | 0e+00 | 1.19e-06 | 1.05e+01 | PASS |
| matrix_affine_scan | B=128 T=1024 k=16 v=8 | float32 | grad | 1e-03 | 0e+00 | 1.07e-06 | 1.25e+00 | PASS |
| matrix_affine_scan | B=128 T=1024 k=16 v=16 | float32 | grad | 1e-03 | 0e+00 | 2.38e-06 | 1.28e+05 | PASS |
| matrix_affine_scan | B=2 T=1 n=3 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=2 T=13 n=3 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 3.58e-07 | 3.30e-04 | PASS |
| matrix_affine_scan | B=2 T=64 n=3 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.23e-02 | PASS |
| matrix_affine_scan | B=2 T=128 n=3 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 4.77e-07 | 3.23e-02 | PASS |
| matrix_affine_scan | B=2 T=1024 n=3 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 4.31e-02 | PASS |
| matrix_affine_scan | B=128 T=1 n=3 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 0.00e+00 | 0.00e+00 | PASS |
| matrix_affine_scan | B=128 T=13 n=3 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 5.96e-07 | 5.17e-01 | PASS |
| matrix_affine_scan | B=128 T=64 n=3 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 6.69e-02 | PASS |
| matrix_affine_scan | B=128 T=128 n=3 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 3.33e-01 | PASS |
| matrix_affine_scan | B=128 T=1024 n=3 k=8 v=4 | float32 | grad | 1e-03 | 0e+00 | 9.54e-07 | 4.33e+00 | PASS |
| givens decay=None | B=4 T=96 input_size=32 hidden_size=64 | float32 | fwd | 1e-05 | 0e+00 | 6.68e-06 | 1.61e-03 | PASS |
| givens decay=None | B=4 T=96 input_size=32 hidden_size=64 | float32 | grad | 1e-03 | 1e-03 | 1.98e-03 | 1.68e-03 | PASS |
| givens decay=learnable (class-default decay_rate=1.0) | B=4 T=96 input_size=32 hidden_size=64 | float32 | fwd | 1e-05 | 0e+00 | 4.77e-07 | 7.59e-03 | PASS |
| givens decay=learnable (class-default decay_rate=1.0) | B=4 T=96 input_size=32 hidden_size=64 | float32 | grad | 1e-03 | 1e-03 | 1.14e-05 | 7.67e-03 | PASS |
| rotation snap=None decay=None | B=4 T=96 input_size=32 hidden_size=64 | float32 | fwd | 1e-05 | 0e+00 | 7.15e-07 | 7.71e-03 | PASS |
| rotation snap=None decay=None | B=4 T=96 input_size=32 hidden_size=64 | float32 | grad | 1e-03 | 1e-03 | 2.29e-05 | 9.08e-04 | PASS |
| rotation snap=(2,3,4,6) decay=None | B=4 T=96 input_size=32 hidden_size=64 | float32 | fwd | 1e-05 | 0e+00 | 4.77e-06 | 4.90e-04 | PASS |
| rotation snap=(2,3,4,6) decay=None | B=4 T=96 input_size=32 hidden_size=64 | float32 | grad | 1e-03 | 1e-03 | 1.83e-04 | 3.35e-04 | PASS |
| rotation snap=(2,3,4,6) decay=learnable | B=4 T=96 input_size=32 hidden_size=64 | float32 | fwd | 1e-05 | 0e+00 | 2.38e-07 | 7.65e-04 | PASS |
| rotation snap=(2,3,4,6) decay=learnable | B=4 T=96 input_size=32 hidden_size=64 | float32 | grad | 1e-03 | 1e-03 | 1.14e-05 | 6.63e-04 | PASS |

Counts: {"total": 590, "passed": 590, "failed": 0, "by_direction": {"fwd": {"total": 295, "passed": 295, "failed": 0}, "grad": {"total": 295, "passed": 295, "failed": 0}}}
