# Enable time decay

This guide turns on `decay=` for a mixer or a stack, so a gap between events shrinks the carried state instead of treating every step as evenly spaced. It assumes you have completed the [getting-started tutorial](../tutorials/getting-started.md). For *why* the mechanism is worth the extra keyword, including the channel-ablation and recovery-check evidence, see the [Givens & Delta deep dive](../explanation/givens-delta.md#time-aware-decay-on-an-orthogonal-block) and its [delta-rule decay section](../explanation/givens-delta.md#time-aware-decay-on-the-delta-rule-memory).

## Minimal runnable example

Every mixer accepts `decay="fixed"` (a scalar rate buffer) or `decay="learnable"` (`softplus(rho)`, one rate per channel/block); `decay=None` (the default) is off and bit-identical to a mixer built without the feature. Enabling decay pairs it with a `delta_t` argument at call time: an irregular gap preceding each event.

```python
import torch
from mingru import MinGRUStack

torch.manual_seed(0)
model = MinGRUStack(
    input_size=1, d_model=64, n_layers=2, mixer="signed",
    mixer_kwargs={"decay": "learnable", "decay_rate": 1.0},
)

x = torch.randn(4, 10, 1)                  # irregular-time toy input
delta_t = torch.rand(4, 10) * 5.0
delta_t[:, 0] = 0.0                        # no gap before the first event

out, state = model(x, delta_t=delta_t)     # parallel forward
print(tuple(out.shape))                    # (4, 10, 64)

step_state = model.init_state()
y_t, step_state = model.step(x[:, 0], step_state, delta_t=delta_t[:, 0])  # streaming
print(tuple(y_t.shape))                    # (4, 64)
```

Both calls above run as written (`out` is `(4, 10, 64)`, `y_t` is `(4, 64)`). `mixer_kwargs` here is the flat dict form because `mixer` is a single `str`; a heterogeneous stack keys it by mixer type instead (see [Choose a mixer](choose-a-mixer.md#the-two-axis-guidance)).

## The `delta_t` contract

- **Shape.** `(B, T)` (or `(B, T, 1)`, squeezed internally) to `forward`; `(B,)` (or `(B, 1)`) to `step`.
- **A gap precedes its event.** `delta_t[:, t]` is the time elapsed *before* event `t`, not after it.
- **No first-event exemption.** `delta_t = 0` gives `gamma = 1` (no decay) at every position, including `t = 0`. There is no implicit "first event is free" special case; pass `delta_t[:, 0] = 0` yourself if that is what you want, as the example above does.
- **The pairing rule is enforced both directions.** Enabling decay without passing `delta_t` raises `ValueError`; passing `delta_t` to a mixer built with `decay=None` also raises `ValueError`. This fails at call time, not silently:

```python
model_off = MinGRUStack(input_size=1, d_model=32, n_layers=1, mixer="signed")
model_off(x, delta_t=delta_t)
# ValueError: delta_t was provided but decay is disabled (decay=None); construct
# the mixer with decay='fixed' or decay='learnable' to use delta_t.
```

- **Bad entries are sanitized, not rejected.** Negative, `NaN`, or infinite `delta_t` entries are clamped to finite, non-negative values on every device; on CPU this also fires a once-per-instance warning (CUDA skips the warning to avoid a host sync, but the clamp still applies).
- **`log1p_delta=True`** passes `delta_t` through `log1p` before scaling by the decay rate, compressing gaps that span orders of magnitude so the rate does not have to.

## Which layers decay: `decay_layers`

`MinGRUStack` accepts `decay_layers="all"` (default) or `"last"`, controlling which blocks receive the decay keywords out of `mixer_kwargs`:

```python
model = MinGRUStack(
    input_size=1, d_model=32, n_layers=3, mixer="signed",
    mixer_kwargs={"decay": "learnable"}, decay_layers="last",
)
# only the final block decays; delta_t is routed to it alone
```

`"last"` is positional: it strips the decay keys from every block except the one at index `n_layers - 1`, whatever mixer type that block is. In a mixed stack (`mixer=["signed", "rotation"]`), prefer placing decay keys under the specific type in a type-keyed `mixer_kwargs` instead of relying on `decay_layers="last"`, so decay lands on the type you intend regardless of position.

## Delta decay (eager-only)

`mixer="delta"` (`DeltaMinGRU`) accepts the same `decay=`, `decay_rate=`, `log1p_delta=` keywords, but the mechanism differs: rather than scaling a per-step transition, `gamma` gates the whole carried matrix state once per token, before that token's own writes (a Gated-DeltaNet-style gate). `decay=None` keeps the unchanged chunked-WY forward; enabling decay switches to a separate eager gated form.

```python
import torch
from mingru import MinGRUStack

torch.manual_seed(0)
model = MinGRUStack(
    input_size=1, d_model=64, n_layers=1, mixer="delta",
    mixer_kwargs={"n_heads": 4, "nh": 2, "decay": "learnable", "log1p_delta": True},
)
x = torch.randn(3, 12, 1)
delta_t = torch.rand(3, 12) * 3.0
delta_t[:, 0] = 0.0

out, state = model(x, delta_t=delta_t)
print(tuple(out.shape))   # (3, 12, 64)
```

Two things to know before you rely on this in production:

- **Eager-only dispatch.** There is no Triton kernel for decay yet. `MINGRU_SCAN=triton` fails loud rather than silently downgrading; `torch.compile` is the documented CUDA path when decay is active. See [Control scan dispatch: Handle decay-active `DeltaMinGRU`](control-scan-dispatch.md#handle-decay-active-deltamingru) for the full contract and the exact error text.
- **Float32 large-gap behavior.** Under float32, a token whose `lambda * delta_t` is very large (a raw gap of roughly `1e4` or more, or a sanitized `+inf`) saturates that token's decay. This is handled correctly automatically: an internal clamp keeps the chunked forward finite, chunk-size invariant, and matching the sequential `step` oracle, with the affected token's memory fully wiped, exactly as the oracle wipes it at `gamma = 0`, and a one-time `UserWarning` fires on CPU. `log1p_delta=True` (used above) is then recommended, though not required for correctness, to keep very large gaps distinguishable rather than all saturating to a full wipe. See [Choose a mixer: DeltaMinGRU](choose-a-mixer.md#deltamingru-the-promoted-default-when-state-can-grow) for the mixer-selection framing of this trade-off.

## You have now

turned on `decay="fixed"`/`"learnable"` for a mixer or a stack, supplied `delta_t` correctly on both the parallel and streaming paths, chosen which layers decay in a stack, and enabled decay on `mixer="delta"` knowing its eager-only dispatch and float32 large-gap behavior.
