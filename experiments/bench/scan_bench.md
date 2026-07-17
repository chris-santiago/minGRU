# Scan kernel benchmark

torch 2.8.0+cu128, device NVIDIA L4. Warmup/reps per shape: {'lab': {'warmup': 10, 'reps': 30}, 'long-T': {'warmup': 5, 'reps': 10}}.

| op | shape | direction | triton (us) | eager (us) | torch_compile (us) | triton speedup vs eager |
| --- | --- | --- | --- | --- | --- | --- |
| linear_scan | {"B": 128, "D": 64, "T": 64} | fwd | 114.38 | 599.89 | 128.68 | 5.24x |
| linear_scan | {"B": 128, "D": 64, "T": 64} | fwdbwd | 1188.12 | 3571.44 | 1301.74 | 3.01x |
| matrix_scan | {"B": 128, "T": 64, "k": 2, "n": 32, "v": 1} | fwd | 331.33 | 5337.02 | 5275.75 | 16.11x |
| matrix_scan | {"B": 128, "T": 64, "k": 2, "n": 32, "v": 1} | fwdbwd | 1000.24 | 94222.61 | 97409.43 | 94.20x |
| matrix_affine_scan | {"B": 128, "T": 64, "k": 8, "n": 8, "v": 1} | fwd | 286.14 | 5462.94 | 4954.49 | 19.09x |
| matrix_affine_scan | {"B": 128, "T": 64, "k": 8, "n": 8, "v": 1} | fwdbwd | 905.25 | 35200.82 | 28935.82 | 38.89x |
| parallel_scan_log | {"B": 128, "D": 64, "T": 64} | fwd | 96.94 | 99.16 | 111.55 | 1.02x |
| parallel_scan_log | {"B": 128, "D": 64, "T": 64} | fwdbwd | 1399.53 | 979.87 | 1013.56 | 0.70x |
| linear_scan | {"B": 16, "D": 64, "T": 1024} | fwd | 184.83 | 1074.59 | 197.12 | 5.81x |
| linear_scan | {"B": 16, "D": 64, "T": 1024} | fwdbwd | 894.36 | 4151.50 | 1631.95 | 4.64x |
| matrix_scan | {"B": 16, "T": 1024, "k": 2, "n": 32, "v": 1} | fwd | 747.62 | 18737.66 | 18530.20 | 25.06x |
| matrix_scan | {"B": 16, "T": 1024, "k": 2, "n": 32, "v": 1} | fwdbwd | 1932.39 | 325657.69 | 338814.36 | 168.53x |
| matrix_affine_scan | {"B": 16, "T": 1024, "k": 8, "n": 8, "v": 1} | fwd | 1055.95 | 24373.86 | 27216.79 | 23.08x |
| matrix_affine_scan | {"B": 16, "T": 1024, "k": 8, "n": 8, "v": 1} | fwdbwd | 2668.24 | 142901.25 | 119721.37 | 53.56x |
| parallel_scan_log | {"B": 16, "D": 64, "T": 1024} | fwd | 350.41 | 551.53 | 200.50 | 1.57x |
| parallel_scan_log | {"B": 16, "D": 64, "T": 1024} | fwdbwd | 1722.57 | 1375.54 | 1038.75 | 0.80x |
