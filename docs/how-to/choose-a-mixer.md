# Choose a mixer

This guide picks the right `mixer=` for your problem. It assumes you have completed the [getting-started tutorial](../tutorials/getting-started.md). Accuracy figures come from two hardware strata, disclosed inline wherever they appear: the pinned CPU evidence (torch 2.5.1, the frozen evidence pin, see [Reproduce the evidence](reproduce-the-evidence.md)) behind the README's "What this shows" and `S3-hier` tables, and a newer GPU stratum (torch 2.8.0+cu128, NVIDIA L4, triton 3.4.0) for the Givens-versus-delta comparison. The two strata are never comparable to each other and are never mixed without saying so. The [explanations](../explanation/index.md) cover *why* each mixer behaves as it does.

## Decide by the shape of your state

Ask what the running state has to *do*, then read across:

| Your state must… | Use | `mixer=` | Evidence |
|---|---|---|---|
| Decay / accumulate positively (default) | `MinGRU` | `"log"` | log-space baseline |
| Flip sign on a running property (e.g. parity) | `SignedMinGRU` | `"signed"` | parity 0.994 @1024 (n=6, CPU) |
| Track non-commutative ops where tokens *are* the ops | `RotationMinGRU` | `"rotation"` | S3 0.958 @1024 (n=8, CPU), 1 layer |
| Compose a non-commutative op that must first be extracted, state free to grow | `SignedMinGRU` → `DeltaMinGRU` (native state) | `["signed", "delta"]` | S3-hier fit 12/12 (CPU) / 35/36 (GPU) |
| Compose a non-commutative op that must first be extracted, state must stay small | `SignedMinGRU` → `GivensMinGRU` | `["signed", "givens"]` | S3-hier fit 8/12 (CPU) / 25/36 (GPU) |

### Mechanism-match by task structure

A broader accepted-benchmark round (MQAR, psMNIST, S5, and a Pendulum control) ran on the same L4 stratum as the Givens-vs-delta comparison below (torch 2.8.0+cu128, triton 3.4.0). It maps task structure onto the mechanism that actually fits it:

| Task structure | Best mechanism | L4 evidence (this round) |
|---|---|---|
| Associative recall (key→value) | delta family | MQAR: only `delta`/`signed-delta` fit ($36/36$); every non-delta arm $0/36$ |
| Order-sensitive accumulation | delta family | psMNIST: `signed-delta` $12/12$ (best), `delta` $10/12$; `givens` $0/12$ (worst, raw $\approx 0.29$) |
| Group composition (permutations) | givens OR delta at $nh \ge k-1$ | S5: `signed-givens` $1/36$ (continuous coupled-8D rotation+sign); `signed-delta-nh4` $7/36$ (Householder threshold: a 5-cycle needs 4 reflections; matched `nh=2` was $0/36$) |
| (Control) irregular-time regression | not discriminating | Pendulum: all nine arms fit, including `gru` |

On these public tasks, delta is the broadly dominant mechanism (recall, accumulation, and groups at `nh=4`); givens remains the narrow group-composition specialist. This is the same two-dial story as [the two-axis guidance](#the-two-axis-guidance) below, not "delta beats givens everywhere". The earlier `S3-hier` round, where givens won at matched state (table above), still stands for that task. The full per-task fit matrices, raw and fit-only generalization, and the S5 `nh` probe are on the [Benchmark validation](../explanation/benchmark-validation.md) page.

## The two-axis guidance

`GivensMinGRU` and `DeltaMinGRU` both compose a non-commutative operation above a `SignedMinGRU` extractor, and the choice between them is no longer a single winner. It is conditioned on two axes: **how much per-token state your deployment can afford**, and **how far past training length it needs to extrapolate**. This is the section other pages point to when they need the full picture; the per-mixer sections below stay focused on usage.

- **State free to grow → `mixer="delta"` at native/full state is the promoted default.** It is the most reliable trained composer measured, on both strata: $12/12$ fits on the pinned CPU stratum (state $1024$) and $35/36$ on the GPU stratum (`torch.compile`), against `GivensMinGRU`'s $8/12$ / $25/36$ at a matched, small state. Its per-step cost and memory stay nearly flat as state grows (`experiments/bench/scaling_frontier.md`).
- **State must stay small (a matched 64-element budget) → `GivensMinGRU` is decisively more reliable.** On the GPU stratum's larger sample, givens fits $25/36$ against delta@64's $10/36$ (Fisher $p = 0.00084$); the CPU stratum's $8/12$ vs $4/12$ tie ($p = 0.22$) was a power limitation, not an absence of effect.
- **Extreme-length, fit-only generalization ($16\times$ train length) favors the small-state composers.** Fit-only cohorts: givens $0.679$ / delta@64 $0.699$, both ahead of delta@1024's $0.570$ (GPU stratum). If your deployment's binding constraint is extrapolation length rather than fit reliability, the small-state composers extrapolate further than full-state delta.
- **Never read delta's low full-seed accuracy means as "delta is less accurate."** Delta@64's pooled means are dragged down by a fit-rate artifact (its trained models generalize like Givens' when they do fit: matched-state fit-only $0.721$ delta vs $0.733$ givens at $T{=}1024$, CPU), not by the mechanism generalizing worse. Symmetrically, delta@1024 is the *most* accurate arm at or near training length on both strata ($1.000$@64/$0.988$@256 CPU; $0.984$@64/$0.971$@256 GPU) and only trails on the long-extrapolation tail.
- **On GPU, cost is a parity story, not a speed argument for either composer.** Per-seed training wall time is $\approx 15$–$18$s across all three composer arms: `GivensMinGRU`'s Triton scan path erases the CPU stratum's $\approx 16\times$ cost gap. What differentiates the composers on GPU is reliability and state economics, not speed. See the [GPU evidence](#the-gpu-evidence-n36-seeds-both-composers) table below.

The full mechanism and evidence trail for both composers is in the [Givens & Delta deep dive](../explanation/givens-delta.md).

## SignedMinGRU: sign-flipping state

Reach for `"signed"` when the answer depends on a running property that can invert, like parity or any accumulator that must swing negative. Its transition coefficient can reach $-1$, which the default log-space `MinGRU` (positive states only) cannot represent. Base `"log"` stays at chance on parity regardless of depth.

```python
from mingru import MinGRUStack

model = MinGRUStack(input_size=1, d_model=64, n_layers=2, mixer="signed")
```

Keep the default `coupled=False` (the decoupled eigenvalue form): it holds 0.994 mean accuracy at 16× the training length, versus 0.592 for the legacy `coupled=True` parameterization. Only pass `mixer_kwargs={"coupled": True}` to reproduce old runs.

`SignedMinGRU` also fits $S_3$-style composition *in-distribution* (0.999 @64). Its diagonal-scan shortcut just decays with length. If you only need behavior near the training length, it may be enough; if you need to hold at 4×–16×, move to a rotation-family mixer.

## RotationMinGRU: non-commutative tracking, one layer

Reach for `"rotation"` when the state must track operations whose order matters (composing permutations), **and the tokens themselves are those operations** (no extraction stage). Its $2\times 2$ rotation blocks are genuinely non-diagonal, so they hold $S_3$ to 0.958 @1024.

```python
model = MinGRUStack(input_size=6, d_model=64, n_layers=1, mixer="rotation")
```

Two caveats, both measured:

- **One layer.** `RotationMinGRU` is validated at $L{=}1$. Deeper single-type rotation stacks are untested; a stack with more than one rotation block warns once at construction.
- **Use the best-val@128 protocol and budget for retries.** Only 1 of 8 seeds lands the exact solution, so select the best checkpoint on a held-out length and expect to rerun weak seeds. See [Reproduce the evidence](reproduce-the-evidence.md).

## GivensMinGRU: the small-state composer

Reach for `"givens"` when your per-token state must stay small (a matched, fixed budget), or when your deployment needs to fit-and-extrapolate to extreme lengths. It composes rotations across 8-dimensional blocks (default `block_size=8`, `rounds=3`), continuous (no angle snap), and `d_model` must be a multiple of `block_size`. (`rounds` = how many staggered layers of paired-plane rotations compose per token; see the [brick-wall Givens parameterization](../explanation/givens-delta.md#the-brick-wall-givens-parameterization).)

```python
model = MinGRUStack(
    input_size=6, d_model=64, n_layers=2, mixer=["signed", "givens"],
)
```

On the hierarchical `S3-hier` probe, swapping the composer from a 2D rotation to `GivensMinGRU` raises the fit rate from **1 of 12 seeds to 8 of 12** (Fisher exact $p \approx 0.009$, CPU) at a matched 64-element per-token state; at 36 seeds on the GPU stratum under its Triton scan path it fits $25/36$, decisively ahead of a matched-state `DeltaMinGRU` composer's $10/36$ ($p = 0.00084$). Its fit-only cohort also holds up better at extreme extrapolation length ($0.679$ at $16\times$ train length) than a full-state delta composer's ($0.570$). Like the other continuous composers here, it still decays by $T{=}1024$ rather than forming an exact length-invariant attractor. See the [two-layer stacks tutorial](../tutorials/two-layer-stacks.md) for the full ordering evidence.

## DeltaMinGRU: the promoted default when state can grow

Reach for `"delta"` when your per-token state is free to grow. It is now the promoted default composer for extract-then-compose stacks. Unlike the other mixers, its per-token state is a per-head $d_k \times d_v$ *matrix* (capacity set by `n_heads`, `nh`, `d_k`, `d_v`, not by `d_model` alone), and its `forward` runs the chunked-WY parallel form of the DeltaNet/DeltaProduct recurrence rather than an associative scan.

```python
model = MinGRUStack(
    input_size=6, d_model=64, n_layers=2, mixer=["signed", "delta"],
    mixer_kwargs={"delta": {"n_heads": 4, "nh": 2}},   # 1024-element native state
)
```

At its native ($16\times$-larger) state, `DeltaMinGRU` fits `S3-hier` on $12/12$ CPU seeds and $35/36$ GPU seeds: the most reliable composer measured on either stratum. On CPU it is also the cheap path: under the pinned bench (`experiments/bench/delta_paths.md`; torch 2.5.1), one uncontended forward+backward step at $B{=}128$, $T{=}64$ costs $0.0577$s against $0.9493$s for the eager `GivensMinGRU` scan (roughly $16\times$ cheaper) and stays roughly $16\times$ cheaper at $T{=}1024$ ($1.4541$s against $24.9996$s); its cost and memory stay nearly flat as state grows (`experiments/bench/scaling_frontier.md`). **On GPU this cost gap disappears**: per-seed training wall time is parity with `GivensMinGRU`'s Triton scan path ($\approx 15$–$18$s for all arms), because `GivensMinGRU`'s Triton scan removes the CPU-side gap, not because delta got slower. Choose delta for reliability and state economics on GPU, not for speed.

### The three dials

`DeltaMinGRU`'s capacity knobs are the state size, the Householder count, and the kernel tile; only the first two change what it can represent:

| Dial | Controls | Default | Turn it up when | Evidence |
| --- | --- | --- | --- | --- |
| State size ($n_{\text{heads}} \cdot d_k \cdot d_v$) | recall / associative capacity | promoted native | recall saturates (high-load recall) | MQAR: only delta-family fits |
| `nh` (Householder count / DeltaProduct order) | group-composition reach; a $k$-cycle needs $k-1$ reflections | `nh=2` | the task needs high-order permutations | S5: nh2 $0/36$ → nh4 $7/36$ |
| `chunk_size` | nothing capability-wise: kernel WY tile; output/gradient-invariant | `64` (`32` when $nh \cdot \text{chunk\_size} > 128$) | never for accuracy; only to fit the triton micro-step envelope | probe nh3/nh4 ran at `chunk_size=32`, math-identical |

The third row is load-bearing: `chunk_size` is not an expressivity knob, only the chunked-WY tile size (output and gradients are invariant to it). That is why the S5 `nh` probe above could run its `nh3`/`nh4` arms at `chunk_size=32`: to fit the $nh \cdot \text{chunk\_size} \le 128$ envelope, not because those arms needed a smaller state. The nh3/nh4 result is not confounded by that choice.

### GPU execution path for `mixer="delta"`

`torch.compile` is the recommended CUDA path for `DeltaMinGRU` when it is available: a fusion-headroom probe (`experiments/bench/gpu_delta_probe.md`, NVIDIA L4, torch 2.8.0+cu128, triton 3.4.0) measured it recovering $68$–$89\%$ of the available fusion headroom over an optimistic matmul floor. A hand-written Triton chunked-WY kernel *does* now exist for the delta forward (with a torch-composed backward), but it does **not** reach compile-class speed: it runs $1.85$–$3.79\times$ slower than `torch.compile` on every probed shape (round `gpu-delta-kernel-01`), and a fully fused Triton backward was $8$–$12\times$ slower still and was reverted; the same relative gap holds on an A100 (a distinct stratum), so the shortfall is in the design against modern baselines, not one GPU. The kernel ships gated for the one case where it earns its place: under `MINGRU_SCAN=auto` it engages **only** in its measured L4 win region: long sequences ($T \ge 16 \cdot \text{chunk\_size}$, i.e. $\ge 1024$ at the default `chunk_size=64`) with narrow head dims ($d_k \le 16$), where it is both faster than eager ($0.61\times$) and $\approx 3\times$ lighter on peak training memory. Everywhere else `auto` silently stays eager. Setting `MINGRU_SCAN=triton` forces the full kernel envelope (fail-loud). So: use `torch.compile` if you can; the kernel is the fallback that gives eager-only users a latency-and-memory win at long-sequence, narrow-head-dim shapes without any compile step.

Three caveats before you commit to it:

- **Reliability trades against extreme-length extrapolation.** Delta's native-state fit cohort holds up worse at $16\times$ train length ($0.570$ fit-only) than either small-state composer ($0.679$ givens, $0.699$ delta@64). If your deployment needs extreme-length fit-only generalization more than near-length reliability, prefer a small-state composer instead. See [The two-axis guidance](#the-two-axis-guidance) above.
- **No time decay.** `DeltaMinGRU` rejects any `decay=` other than `None` with a `ValueError`.
- **Different state contract.** `step` returns `(y_t, h_t)` (the readout is not the carried state), and the state is the flattened memory matrix; see the [API reference](../reference/mixers.md).

## The GPU evidence: n=36 seeds, both composers

The GPU stratum (torch 2.8.0+cu128, NVIDIA L4, triton 3.4.0, never comparable to the CPU rows above) trained both composers at 36 seeds each, `GivensMinGRU` under the forced Triton scan path and both `DeltaMinGRU` configurations under `torch.compile`, on `S3-hier`:

| config | seeds | fits | acc@64 | acc@256 | acc@512 | acc@1024 | fit-only @512/@1024 | per-seed wall |
|---|---|---|---|---|---|---|---|---|
| givens@64, Triton scan | 36 | 25/36 | 0.929 | 0.857 | 0.750 | 0.571 | 0.889 / 0.679 | 15.3 s |
| delta@64-matched, compiled | 36 | 10/36 | 0.556 | 0.465 | 0.421 | 0.340 | 0.923 / 0.699 | 17.9 s |
| delta@1024, compiled | 36 | 35/36 | 0.984 | 0.971 | 0.828 | 0.561 | 0.843 / 0.570 | 17.2 s |

Read the four columns together, not any one in isolation: fit rate says native-state delta is the most reliable arm and matched-state delta is the least; per-seed wall time says all three arms cost about the same on this hardware; fit-only length generalization says the two small-state cohorts extrapolate further than the large-state one. Composer parameter counts (unchanged from the CPU matched-state round): givens $14{,}624$, delta@64 $3{,}306$, delta@1024 $25{,}480$. The delta@64 comparison remains parameter-unmatched-low. Full statistics and Fisher contrasts are in `experiments/EXPERIMENTS.md` (round `hetero-gpu36-*`) and the [Givens & Delta deep dive](../explanation/givens-delta.md#composer-reliability-givens-vs-delta-matched-state-and-native-state).

## When order matters, choose it from the task

For heterogeneous stacks the rule is *extract-then-compose*: if the operation is buried in the input, put the extractor (`"signed"`) below the composer (`"givens"`/`"delta"`/`"rotation"`); if raw tokens already are the operation, compose directly. The [two-layer stacks tutorial](../tutorials/two-layer-stacks.md) tabulates the order effect on both `S3-hier` and `S3`.

## You have now

chosen a mixer from the structure of your problem's running state and, if it's an extract-then-compose problem, from your per-token state budget and extrapolation-length needs, and know the length-generalization tradeoff and training protocol each one carries.
