# Mechanism x state scaling probe (time + peak RSS)

Subprocess-isolated, uncontended forward+backward timing and peak RSS per (mechanism, state) config, per spec section 6 / intent ledger statement 4. This artifact is the sole source for the round's verdict-table cost/memory columns.

Env: torch 2.5.1 (evidence pin), CPU: Apple M2 Pro (6 torch threads), platform macOS-15.7.5-arm64-arm-64bit, B=128, T=64, warmup=1, timed steps=5, per-config timeout=600.0s, commit 4499af8e0b3bf63aa855017ae8a2fb0b55361551, generated 2026-07-18T06:56:23.847061+00:00.

RSS units depend on platform: bytes on macOS (Darwin), kilobytes on Linux -- this run's platform is `macOS-15.7.5-arm64-arm-64bit` (see the env block above); `peak_rss_bytes` below is `ru_maxrss` reported verbatim by the child, not normalized.

An `oom` row's 'oom detail' column shows whether the underlying exception message matched a known memory-allocator pattern (`recognized`) or not (`UNRECOGNIZED`, worth auditing -- this probe conservatively files any crash after a successful timed-loop start as `oom`, so an unrecognized message could be a real resource wall with an unfamiliar message, or a genuine bug at that extreme state size; see the module docstring's `oom` outcome), followed by the first line of the captured message.

| mechanism | config | state elements | params | training arm | step secs (median) | peak RSS | status | oom detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| delta | hidden_size=64, n_heads=4, nh=2, d_k=4, d_v=4, chunk_size=64 | 64 | 6808 |  | 0.0496 | 439779328 | ok |  |
| delta | hidden_size=64, n_heads=4, nh=2, d_k=8, d_v=8, chunk_size=64 | 256 | 13032 |  | 0.0533 | 466485248 | ok |  |
| delta | hidden_size=64, n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64 | 1024 | 25480 |  | 0.0588 | 548175872 | ok |  |
| delta | hidden_size=64, n_heads=4, nh=2, d_k=32, d_v=32, chunk_size=64 | 4096 | 50376 |  | 0.0771 | 708902912 | ok |  |
| delta | hidden_size=64, n_heads=1, nh=2, d_k=8, d_v=8, chunk_size=64 | 64 | 3306 | yes | 0.0215 | 340049920 | ok |  |
| delta | hidden_size=64, n_heads=4, nh=2, d_k=16, d_v=16, chunk_size=64 | 1024 | 25480 | yes | 0.0596 | 532332544 | ok |  |
| givens | hidden_size=64, block_size=8, rounds=3 | 64 | 14624 |  | 0.9836 | 930430976 | ok |  |
| givens | hidden_size=256, block_size=8, rounds=3 | 256 | 58496 |  | 3.7341 | 2274836480 | ok |  |
| givens | hidden_size=1024, block_size=8, rounds=3 | 1024 | 233984 |  | 14.5641 | 7322730496 | ok |  |
| givens | hidden_size=4096, block_size=8, rounds=3 | 4096 | 935936 |  | 70.5973 | 21467217920 | ok |  |

Timeout/OOM rows are the frontier finding, not a failure -- see the module docstring of `scripts/scaling_probe.py`.

