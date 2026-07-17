# Division-reversal error emulation (chunked reversible backward, C=64)

torch 2.8.0, CPU, fp32, T=4096, D=256, chunk C=64, seed 0. Decays sampled uniform in [gamma_min, 1). See `experiments/reversal_emulation.py` for the protocol.

| gamma_min | state err (global) | decay-grad err (global) | state max rel err | grad max rel err | worst-case bound eps*gamma_min^-(C-1) |
| --- | --- | --- | --- | --- | --- |
| 0.86 | 2.242e-05 | 1.779e-05 | 9.209e+00 | 9.209e+00 | 1.596e-03 |
| 0.48 | 2.562e+03 | 1.499e+03 | 3.585e+06 | 3.585e+06 | 1.439e+13 |
| 0.23 | 1.047e+13 | 6.099e+12 | 4.133e+14 | 4.133e+14 | 1.938e+33 |
