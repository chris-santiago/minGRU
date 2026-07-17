# GivensMinGRU backward peak memory (angle-fused vs Phase-1 generic)

Shape: {'B': 128, 'T': 64, 'd': 64, 'k': 8}

| impl | peak MB |
| --- | --- |
| eager_phase1_generic | 395.09 |
| angle_fused_triton | 38.25 |
