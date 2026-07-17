# Getting started

This tutorial takes you from an empty environment to a trained sequence model. You will install the package, check a single layer's shapes, then train a two-layer stack to solve **running parity** — labelling each prefix of a bit string with its cumulative XOR — and watch its accuracy climb to 1.000 and hold at four times the training length. Every step prints output so you can see it working.

## Prerequisites

- Python 3.10 or newer
- `torch >= 2.8`

```bash
pip install mingru-scans
```

The distribution is `mingru-scans`; the import name is `mingru`. CPU is enough for this tutorial — no GPU or Triton install is required (see [Triton on GPU](triton-on-gpu.md) for the accelerated path).

## Step 1 — Build a layer and check its shapes

A single `MinGRU` layer maps an input sequence `(B, T, input_size)` to a hidden sequence `(B, T, hidden_size)` in one parallel scan. The same layer also exposes an $O(1)$-memory streaming `step`.

```python
import torch
from mingru import MinGRU

layer = MinGRU(input_size=32, hidden_size=64)
x = torch.randn(4, 128, 32)

h = layer(x)                  # parallel training-mode forward
print("forward:", tuple(h.shape))

h_t = layer.step(x[:, 0])     # one streaming step
print("step:", tuple(h_t.shape))
```

**Output:**

```
forward: (4, 128, 64)
step: (4, 64)
```

## Step 2 — Train your first model on running parity

Parity is the smallest task that separates these mixers from an ordinary decaying RNN: the label is the running XOR of a bit stream, so the state has to *flip sign* on a `1` and *hold* on a `0`. The default mixer, `"log"`, keeps its state positive and cannot flip, so it stays near chance. We therefore pick `mixer="signed"`, whose transition coefficient can reach $-1$. (The [Choose a mixer](../how-to/choose-a-mixer.md) guide covers when to reach for each mixer.)

Build a two-layer `MinGRUStack`, add a linear classification head over its `d_model=64` output, and train with Adam. The data generator produces fresh random batches each step, so there is no separate dataset to download.

```python
import torch
import torch.nn as nn
from mingru import MinGRUStack

torch.manual_seed(0)


def make_batch(batch, length):
    bits = torch.randint(0, 2, (batch, length, 1)).float()
    parity = (bits.squeeze(-1).cumsum(dim=1) % 2).long()   # running XOR
    return bits, parity


model = MinGRUStack(input_size=1, d_model=64, n_layers=2, mixer="signed")
head = nn.Linear(64, 2)
opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

for step in range(301):
    x, y = make_batch(128, 64)
    out, _ = model(x)                       # (B, T, 64)
    logits = head(out)                      # (B, T, 2)
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
step   0  loss 0.6950  acc 0.496
step  50  loss 0.6852  acc 0.506
step 100  loss 0.0044  acc 0.999
step 150  loss 0.0005  acc 1.000
step 200  loss 0.0003  acc 1.000
step 250  loss 0.0002  acc 1.000
step 300  loss 0.0001  acc 1.000
```

The model sits at chance (accuracy near 0.5) for the first fifty-odd steps, then snaps to an exact solution once the sign-flip transition is found.

## Step 3 — Check length generalization

The real test of a state-tracking solution is whether it holds *past the training length*. Train at `T=64`, evaluate at `T=256`:

```python
model.eval()
with torch.no_grad():
    x, y = make_batch(256, 256)
    out, _ = model(x)
    acc = (head(out).argmax(-1) == y).float().mean().item()
    print(f"acc @ T=256 (trained at T=64): {acc:.3f}")
```

**Output:**

```
acc @ T=256 (trained at T=64): 1.000
```

A model that had merely memorized a depth-64 shortcut would decay here. Holding at 1.000 four times out is the signature of the exact recurrent solution — the property these mixers are built for.

## Step 4 — Stream one step at a time

The trained model runs as a constant-memory recurrence for inference. `init_state()` gives one slot per block; `step` consumes a single timestep `(B, input_size)` and returns the output plus the updated state.

```python
state = model.init_state()
bit = torch.tensor([[1.0]])               # one sample, one feature
y_t, state = model.step(bit, state)
print("stream out:", tuple(y_t.shape), "| state entries:", len(state))
```

**Output:**

```
stream out: (1, 64) | state entries: 2
```

## What you built

You installed the package, verified a layer's shapes, and trained a two-layer `SignedMinGRU` stack that solves running parity exactly and generalizes to four times its training length, then ran it as a streaming recurrence.

| Step | Result |
|---|---|
| Layer forward / step | `(4, 128, 64)` / `(4, 64)` |
| Trained accuracy @ T=64 | 1.000 |
| Length generalization @ T=256 | 1.000 |
| Streaming state | 2 blocks × `d_model` per sample |

## Next steps

- [Two-layer stacks](two-layer-stacks.md) — compose *different* mixers in the right order for hierarchical tasks.
- [Triton on GPU](triton-on-gpu.md) — enable the Triton scan kernels and confirm they engage.
- [Choose a mixer](../how-to/choose-a-mixer.md) — which of the four mixers fits your problem.
- [Why the mixers differ](../explanation/index.md) — the state-tracking ideas behind the ladder.
