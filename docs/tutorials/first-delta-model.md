# Your first delta model

`DeltaMinGRU` departs from the other four mixers entirely: instead of a diagonal or rotation scan over a vector state, it carries a per-head **matrix** built by the DeltaNet/DeltaProduct delta rule, and it is the promoted default composer once your deployment can afford a state that is free to grow (see [Choose a mixer](../how-to/choose-a-mixer.md#the-two-axis-guidance)). This tutorial builds one, trains it, and teaches the two places it behaves differently from every mixer you have used so far: its `step` return shape, and how it reacts to irregular time gaps. Every step prints output so you can see it working.

This tutorial assumes you have completed [Getting started](getting-started.md) and [Two-layer stacks](two-layer-stacks.md): it reuses the list-`mixer` and type-keyed `mixer_kwargs` schema from the latter without re-explaining it. CPU is enough; no GPU or Triton install is required.

## Step 1: Build a real delta model

Pass `mixer=["signed", "delta"]` to put a `SignedMinGRU` extractor under a `DeltaMinGRU` composer, the same extract-then-compose shape as [Two-layer stacks](two-layer-stacks.md), swapped to the composer that is free to grow. `mixer_kwargs={"delta": {"n_heads": 4, "nh": 2}}` configures the composer: 4 associative-memory heads, 2 generalized-Householder micro-steps per token (DeltaProduct order 2).

```python
import torch
from mingru import MinGRUStack

torch.manual_seed(0)

model = MinGRUStack(
    input_size=6,
    d_model=64,
    n_layers=2,
    mixer=["signed", "delta"],
    mixer_kwargs={"delta": {"n_heads": 4, "nh": 2}},
)
x = torch.randn(4, 128, 6)
out, state = model(x)

print("forward:", tuple(out.shape), "| states:", len(state))
print("block 0 mixer:", type(model.blocks[0].mingru).__name__)
print("block 1 mixer:", type(model.blocks[1].mingru).__name__)
print("block 1 state shape:", tuple(state[1].shape))
```

**Output:**

```
forward: (4, 128, 64) | states: 2
block 0 mixer: SignedMinGRU
block 1 mixer: DeltaMinGRU
block 1 state shape: (4, 1, 1024)
```

The stack forwards exactly like a homogeneous one, `(B, T, input_size) → (B, T, d_model)`, but look at the delta block's state: `(4, 1, 1024)`, not `(4, 1, 64)`. Every other mixer's per-block state is `(B, 1, d_model)`. `DeltaMinGRU`'s is `(B, 1, n_heads * d_k * d_v)`: with `n_heads=4` and `d_k`/`d_v` defaulting to `d_model // n_heads = 16`, that is $4 \times 16 \times 16 = 1024$, sixteen times `d_model`. The composer's per-token state is genuinely a matrix, and it is decoupled from `d_model` by design.

## Step 2: Train it on running parity

Train the same running-parity task from [Getting started](getting-started.md) so you can watch the new mixer combination reach the same exact solution:

```python
import torch
import torch.nn as nn
from mingru import MinGRUStack

torch.manual_seed(0)


def make_batch(batch, length):
    bits = torch.randint(0, 2, (batch, length, 1)).float()
    parity = (bits.squeeze(-1).cumsum(dim=1) % 2).long()   # running XOR
    return bits, parity


model = MinGRUStack(
    input_size=1,
    d_model=64,
    n_layers=2,
    mixer=["signed", "delta"],
    mixer_kwargs={"delta": {"n_heads": 4, "nh": 2}},
)
head = nn.Linear(64, 2)
opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

for step in range(301):
    x, y = make_batch(128, 64)
    out, _ = model(x)
    logits = head(out)
    loss = loss_fn(logits.reshape(-1, 2), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        acc = (logits.argmax(-1) == y).float().mean().item()
        print(f"step {step:3d}  loss {loss.item():.4f}  acc {acc:.3f}")
```

**Output:**

```
step   0  loss 0.7705  acc 0.504
step  50  loss 0.6144  acc 0.587
step 100  loss 0.0008  acc 1.000
step 150  loss 0.0004  acc 1.000
step 200  loss 0.0003  acc 1.000
step 250  loss 0.0002  acc 1.000
step 300  loss 0.0002  acc 1.000
```

The delta composer solves the task exactly, the same signature as the `signed`-only stack in Getting started, just with a richer composer underneath.

## Step 3: The trap — `step` does not always return one tensor

Getting started's Step 1 showed `layer.step(x_t)` returning a single tensor, the new hidden state. That is true for `MinGRU`, `SignedMinGRU`, `RotationMinGRU`, and `GivensMinGRU`. It is **not** true for `DeltaMinGRU`: its `step` returns a 2-tuple, `(y_t, h_t)`.

```python
import torch
from mingru import MinGRU, MinGRUStack

torch.manual_seed(0)

# The trap: getting-started's base MinGRU.step returns ONE tensor.
layer = MinGRU(input_size=32, hidden_size=64)
h_t = layer.step(torch.randn(4, 32))
print("MinGRU.step return type:", type(h_t).__name__, "| shape:", tuple(h_t.shape))

# DeltaMinGRU.step returns a 2-tuple (y_t, h_t): readout and carried state differ.
model = MinGRUStack(
    input_size=1,
    d_model=64,
    n_layers=2,
    mixer=["signed", "delta"],
    mixer_kwargs={"delta": {"n_heads": 4, "nh": 2}},
)
delta_block = model.blocks[1].mingru
print("delta block class:", type(delta_block).__name__)
print("carries_matrix_state:", delta_block.carries_matrix_state)

x_t = torch.randn(4, 64)  # (B, d_model) -- this block's input_size is d_model=64
result = delta_block.step(x_t)
print("DeltaMinGRU.step return type:", type(result).__name__, "| length:", len(result))
y_t, h_t = result
print("y_t (readout) shape:", tuple(y_t.shape))
print("h_t (carried state) shape:", tuple(h_t.shape))
```

**Output:**

```
MinGRU.step return type: Tensor | shape: (4, 64)
delta block class: DeltaMinGRU
carries_matrix_state: True
DeltaMinGRU.step return type: tuple | length: 2
y_t (readout) shape: (4, 64)
h_t (carried state) shape: (4, 1024)
```

`carries_matrix_state = True` is the class attribute that names this: for every other mixer, the scanned state *is* the output, so `step` hands back one tensor doing both jobs. `DeltaMinGRU` breaks that coincidence honestly instead of hiding it. Its readout is $y_t = H_t^\top q_t$, a projection of the memory through the current query, while the carried state $h_t$ is the flattened memory matrix $H_t$ itself, shape `(B, n_heads * d_k * d_v)`. Readout and state differ in both value and shape (`(4, 64)` vs `(4, 1024)` above), so both have to come back.

You will not normally call `delta_block.step` directly like this: `MinGRUBlock.step` and `MinGRUStack.step` (the ones you used in Getting started's Step 4) already normalize every mixer, including `"delta"`, to the same `(output, state)` convention, dispatching on `carries_matrix_state` internally. This step is only to show you what is underneath, so the tuple never surprises you if you do reach for the raw mixer.

## Step 4: The capacity knobs

`DeltaMinGRU`'s state size is `n_heads * d_k * d_v`, and it is set independently of `d_model` by four constructor keywords:

| Knob | Controls | Default |
|---|---|---|
| `n_heads` | number of parallel associative-memory heads | `4` |
| `nh` | Householder micro-steps per token (DeltaProduct order; `nh=1` is plain DeltaNet) | `1` |
| `d_k` | per-head key/query dimension | `hidden_size // n_heads` |
| `d_v` | per-head value dimension | `d_k` |

Turn `n_heads`/`d_k`/`d_v` up when you need more associative-recall capacity; turn `nh` up when the task needs to compose higher-order permutations (a $k$-cycle needs $k-1$ reflections, so `nh` bounds how rich a group element the composer can represent per token). See [Choose a mixer: the three dials](../how-to/choose-a-mixer.md#the-three-dials) for the evidence behind each knob.

```python
import torch
from mingru import DeltaMinGRU

torch.manual_seed(0)

configs = [
    dict(n_heads=4, nh=1),                     # DeltaNet default
    dict(n_heads=4, nh=2),                     # DeltaProduct, 2 Householders/token
    dict(n_heads=8, nh=1, d_k=8, d_v=8),        # more, narrower heads
]

for cfg in configs:
    layer = DeltaMinGRU(input_size=32, hidden_size=64, **cfg)
    n_params = sum(p.numel() for p in layer.parameters())
    state_size = layer.n_heads * layer.d_k * layer.d_v
    print(
        f"{cfg} -> d_k={layer.d_k} d_v={layer.d_v} "
        f"state_per_sample={state_size} params={n_params}"
    )
```

**Output:**

```
{'n_heads': 4, 'nh': 1} -> d_k=16 d_v=16 state_per_sample=1024 params=10628
{'n_heads': 4, 'nh': 2} -> d_k=16 d_v=16 state_per_sample=1024 params=14984
{'n_heads': 8, 'nh': 1, 'd_k': 8, 'd_v': 8} -> d_k=8 d_v=8 state_per_sample=512 params=10760
```

Raising `nh` from 1 to 2 leaves the state size unchanged (still `1024`) but adds a second set of `k`/`v`/`beta` projections, so the parameter count grows. Explicit `d_k=8, d_v=8` with more heads halves the state (`512`) at roughly the same parameter budget as the default. Neither `d_model` nor `chunk_size` (the kernel's tiling knob, performance-only) appears in this table: `chunk_size` never changes what the model can represent, only how the parallel forward tiles the computation.

## Step 5: Adding time decay

`DeltaMinGRU` accepts the same `decay="fixed"`/`"learnable"` contract as every other mixer (see [Enable time decay](../how-to/enable-time-decay.md)), but it decays a different kind of state: the carried memory matrix $H$, gated once per token at the token boundary ($H \leftarrow \gamma_t H$) before that token's own writes land undecayed. Pass `delta_t`, the gap preceding each event, alongside `decay=`.

One thing to expect before you run this: with a large enough gap, you will likely see a `UserWarning` about `delta_t` saturating the decay in float32. That is expected, not a bug: it means a gap was large enough (roughly $\lambda \cdot \Delta t > 10^4$) that the affected token's memory is fully wiped, correctly, but very large gaps become indistinguishable from each other in the internal computation. Pass `log1p_delta=True` at construction to compress large gaps and avoid it if you need gaps of very different sizes to stay distinguishable.

The example below feeds the *same* recurring token through the composer under two gap schedules, so any change in state norm is attributable to decay, not to the input changing: `gapped` inserts one huge gap before event 3; `control` uses `delta_t=0` throughout, which gives $\gamma = \exp(0) = 1$, i.e. no decay at all even though `decay="learnable"` is on.

```python
import warnings

import torch
from mingru import DeltaMinGRU

torch.manual_seed(0)

layer = DeltaMinGRU(input_size=8, hidden_size=8, n_heads=1, decay="learnable")

x = torch.randn(1, 1, 8).expand(1, 6, 8)
gapped = torch.tensor([[0.0, 1.0, 1.0, 20000.0, 1.0, 1.0]])
control = torch.zeros_like(gapped)


def trajectory(delta_t):
    norms = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for t in range(1, x.size(1) + 1):
            _, h = layer(x[:, :t], delta_t=delta_t[:, :t], return_state=True)
            norms.append(h.norm().item())
    return norms, list(caught)


gapped_norms, gapped_warnings = trajectory(gapped)
control_norms, _ = trajectory(control)

print("gap before event:  ", gapped[0].tolist())
print("state norm, gapped: ", [f"{n:.3f}" for n in gapped_norms])
print("state norm, control:", [f"{n:.3f}" for n in control_norms])
print("warnings raised:", len(gapped_warnings))
print("warning:", str(gapped_warnings[0].message))
```

**Output:**

```
gap before event:   [0.0, 1.0, 1.0, 20000.0, 1.0, 1.0]
state norm, gapped:  ['1.383', '1.458', '1.462', '1.383', '1.458', '1.462']
state norm, control: ['1.383', '1.586', '1.616', '1.621', '1.622', '1.622']
warnings raised: 1
warning: delta_t gaps are large enough that lambda*delta_t exceeds 10000, saturating the gated float32 forward: the affected tokens fully wipe memory (handled correctly via an internal clamp) but very large gaps become indistinguishable from one another. Pass log1p_delta=True or rescale delta_t to keep large gaps distinguishable.
```

Read the two trajectories side by side. `control` accumulates monotonically and settles as the same token keeps writing into memory. `gapped` tracks `control` for the first three events, then the huge gap resets the norm from `1.462` back down to `1.383`, exactly the norm after the *first* event, and the following two entries (`1.458`, `1.462`) repeat the first two events' values exactly. The gap did not just shrink the state, it wiped it back to a fresh start, and the warning you were told to expect fired right on cue.

If your data has gaps of very different orders of magnitude and you need the decay to distinguish between a large gap and a huge one rather than saturating both to a full wipe, pass `log1p_delta=True`:

```python
import warnings

import torch
from mingru import DeltaMinGRU

torch.manual_seed(0)
layer = DeltaMinGRU(
    input_size=8, hidden_size=8, n_heads=1, decay="learnable", log1p_delta=True
)
x = torch.randn(1, 6, 8)
gapped = torch.tensor([[0.0, 1.0, 1.0, 20000.0, 1.0, 1.0]])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _, h = layer(x, delta_t=gapped, return_state=True)
print("warnings raised with log1p_delta=True:", len(caught))
```

**Output:**

```
warnings raised with log1p_delta=True: 0
```

## What you built

You constructed a `["signed", "delta"]` stack with an explicit `n_heads`/`nh` budget, trained it to an exact solution on running parity, saw `DeltaMinGRU.step`'s honest `(y_t, h_t)` return where readout and carried state differ, sized the composer with its four capacity knobs, and turned on learnable time decay, including the float32 saturation warning it was always going to raise on a large gap.

| Step | Result |
|---|---|
| Delta block state shape | `(4, 1, 1024)` vs `d_model=64` |
| Trained accuracy on parity | 1.000 |
| `DeltaMinGRU.step` return | `(y_t, h_t)`, shapes `(4, 64)` / `(4, 1024)` |
| Capacity sweep | state `1024` → `512` at `n_heads=8, d_k=d_v=8` |
| Decay wipe after a $\Delta t = 20000$ gap | state norm resets to the pre-accumulation value |

## Next steps

- [Choose a mixer](../how-to/choose-a-mixer.md#the-two-axis-guidance): the full state-vs-extrapolation trade-off between `DeltaMinGRU` and `GivensMinGRU`.
- [Enable time decay](../how-to/enable-time-decay.md#delta-decay-eager-only): the how-to recipe for `decay=`/`delta_t`, including the `decay_layers` knob for mixed stacks.
- [Givens & Delta deep dive](../explanation/givens-delta.md#time-aware-decay-on-the-delta-rule-memory): why the chunked-WY parallel form survives per-token decay, and the float32 clamp's derivation.
- [Benchmark validation](../explanation/benchmark-validation.md): where delta is the dominant mechanism across public tasks (associative recall, order-sensitive accumulation) and where it is not.
