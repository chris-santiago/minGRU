# Choose a mixer

This guide picks the right `mixer=` for your problem. It assumes you have completed the [getting-started tutorial](../tutorials/getting-started.md). All accuracy figures are multi-seed means from the README's "What this shows" and `S3-hier` tables (torch 2.5.1, CPU — the frozen evidence pin, see [Reproduce the evidence](reproduce-the-evidence.md)); the [explanations](../explanation/index.md) cover *why* each mixer behaves as it does.

## Decide by the shape of your state

Ask what the running state has to *do*, then read across:

| Your state must… | Use | `mixer=` | Evidence |
|---|---|---|---|
| Decay / accumulate positively (default) | `MinGRU` | `"log"` | log-space baseline |
| Flip sign on a running property (e.g. parity) | `SignedMinGRU` | `"signed"` | parity 0.994 @1024 (n=6) |
| Track non-commutative ops where tokens *are* the ops | `RotationMinGRU` | `"rotation"` | S3 0.958 @1024 (n=8), 1 layer |
| Compose a non-commutative op that must first be extracted | `SignedMinGRU` → `GivensMinGRU` | `["signed", "givens"]` | S3-hier fit 8/12 seeds |

## SignedMinGRU — sign-flipping state

Reach for `"signed"` when the answer depends on a running property that can invert, like parity or any accumulator that must swing negative. Its transition coefficient can reach $-1$, which the default log-space `MinGRU` (positive states only) cannot represent — base `"log"` stays at chance on parity regardless of depth.

```python
from mingru import MinGRUStack

model = MinGRUStack(input_size=1, d_model=64, n_layers=2, mixer="signed")
```

Keep the default `coupled=False` (the decoupled eigenvalue form): it holds 0.994 mean accuracy at 16× the training length, versus 0.592 for the legacy `coupled=True` parameterization. Only pass `mixer_kwargs={"coupled": True}` to reproduce old runs.

`SignedMinGRU` also fits $S_3$-style composition *in-distribution* (0.999 @64) — its diagonal-scan shortcut just decays with length. If you only need behavior near the training length, it may be enough; if you need to hold at 4×–16×, move to a rotation-family mixer.

## RotationMinGRU — non-commutative tracking, one layer

Reach for `"rotation"` when the state must track operations whose order matters (composing permutations), **and the tokens themselves are those operations** (no extraction stage). Its $2\times 2$ rotation blocks are genuinely non-diagonal, so they hold $S_3$ to 0.958 @1024.

```python
model = MinGRUStack(input_size=6, d_model=64, n_layers=1, mixer="rotation")
```

Two caveats, both measured:

- **One layer.** `RotationMinGRU` is validated at $L{=}1$. Deeper single-type rotation stacks are untested; a stack with more than one rotation block warns once at construction.
- **Use the best-val@128 protocol and budget for retries.** Only 1 of 8 seeds lands the exact solution, so select the best checkpoint on a held-out length and expect to rerun weak seeds. See [Reproduce the evidence](reproduce-the-evidence.md).

## GivensMinGRU — richer composer for hierarchical tasks

Reach for `"givens"` when you need a *richer* non-commutative composer than a 2D rotation — typically as the upper layer of a stack whose lower layer extracts the operation from raw input. Its transitions are rotations across 8-dimensional blocks (default `block_size=8`, `rounds=3`), continuous (no angle snap), and `d_model` must be a multiple of `block_size`. (`rounds` = how many staggered layers of paired-plane rotations compose per token; see the [brick-wall Givens parameterization](../explanation/givens-mingru.md#the-brick-wall-givens-parameterization).)

```python
model = MinGRUStack(
    input_size=6, d_model=64, n_layers=2, mixer=["signed", "givens"],
)
```

On the hierarchical `S3-hier` probe, swapping the composer from a 2D rotation to `GivensMinGRU` raises the fit rate from **1 of 12 seeds to 8 of 12** (Fisher exact $p \approx 0.009$) at a matched 64-element per-token state. It is the recommended composer for extract-then-compose problems — see the [two-layer stacks tutorial](../tutorials/two-layer-stacks.md) for the full ordering evidence. Like the other continuous composers here, it still decays by $T{=}1024$ rather than forming an exact length-invariant attractor.

## When order matters, choose it from the task

For heterogeneous stacks the rule is *extract-then-compose*: if the operation is buried in the input, put the extractor (`"signed"`) below the composer (`"givens"`/`"rotation"`); if raw tokens already are the operation, compose directly. The [two-layer stacks tutorial](../tutorials/two-layer-stacks.md) tabulates the order effect on both `S3-hier` and `S3`.

## You have now

chosen a mixer from the structure of your problem's running state, and know the length-generalization tradeoff and training protocol each one carries.
