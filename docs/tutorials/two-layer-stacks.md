# Two-layer stacks

A single mixer tracks one kind of state. Hard sequence problems are often *hierarchical*: something has to be **extracted** from the raw input before the running quantity can be **composed**. `MinGRUStack` lets you assign a different mixer to each layer, so you can put an extractor underneath a composer. This tutorial builds a `SignedMinGRU` extractor feeding a `GivensMinGRU` composer, one of the two recommended composer choices for the repo's hierarchical probe (the small-state option; see [How-to: choose a mixer](../how-to/choose-a-mixer.md#the-two-axis-guidance) for when `DeltaMinGRU` is the better composer instead), and shows why extract-then-compose *order* is the one to reach for regardless of which composer you pick.

By the end you will have constructed the ordered stack with the public API, configured each layer independently, seen the warning that flags a fragile ordering, and read the measured evidence that fixes the order for this class of task.

## The problem shape: extract, then compose

The motivating task is `S3-hier` (the repo's harder probe; chance $\approx 1/6$). The group operation is hidden inside a *pair* of sub-tokens: a running product over the permutation group $S_3$ is updated only when a pair completes, and a fixed Latin square maps each pair to its generator. Because the square is non-isotopic to any group of order six, no single-token shortcut exists: the generator must genuinely be **extracted** from the pair before it can be **composed** onto the running product. That is a two-stage computation, and it maps cleanly onto two layers: a lower layer that reads pairs into generators, an upper layer that accumulates them.

## Step 1: Build the ordered stack

Pass `mixer` as a list, one entry per block, bottom layer first. `["signed", "givens"]` puts the `SignedMinGRU` extractor at layer 0 and the `GivensMinGRU` composer at layer 1.

```python
import torch
from mingru import MinGRUStack

model = MinGRUStack(input_size=6, d_model=64, n_layers=2, mixer=["signed", "givens"])
x = torch.randn(4, 128, 6)
out, state = model(x)

print("forward:", tuple(out.shape), "| states:", len(state))
print("block 0 mixer:", type(model.blocks[0].mingru).__name__)
print("block 1 mixer:", type(model.blocks[1].mingru).__name__)
```

**Output:**

```
forward: (4, 128, 64) | states: 2
block 0 mixer: SignedMinGRU
block 1 mixer: GivensMinGRU
```

The stack forwards exactly like a homogeneous one: `(B, T, input_size) → (B, T, d_model)` plus one state per block. Only the per-layer mixer changed. `GivensMinGRU` requires `d_model` to be a multiple of its `block_size` (default 8); `64` satisfies that.

## Step 2: Configure each layer independently

With a list `mixer`, `mixer_kwargs` is keyed **by mixer type**, not applied flat. Each type's dict is that mixer's constructor kwargs, shared by every block of that type. Here we pin the extractor to the decoupled `SignedMinGRU` parameterization and set the composer's block geometry explicitly:

```python
model = MinGRUStack(
    input_size=6,
    d_model=64,
    n_layers=2,
    mixer=["signed", "givens"],
    mixer_kwargs={
        "signed": {"coupled": False},          # decoupled: the better length-generalizing form
        "givens": {"block_size": 8, "rounds": 3},   # 8-dim rotation blocks, 3 brick-wall rounds
    },
)
print("configured forward:", tuple(model(x)[0].shape))
```

**Output:**

```
configured forward: (4, 128, 64)
```

`rounds` = how many staggered layers of paired-plane rotations compose per token; see the [brick-wall Givens parameterization](../explanation/givens-delta.md#the-brick-wall-givens-parameterization) for why that specific mesh.

A flat `mixer_kwargs` dict alongside a list `mixer` (or a type-keyed dict alongside a single-string `mixer`) raises `ValueError` naming both schemas, so the two forms cannot be mixed up silently.

## Step 3: Read the ordering warning

`signed → givens` constructs silently. A stack with more than one `rotation` block does not: the straight-through angle snap in `RotationMinGRU` can compound across rotation layers, and only the $L{=}2$ case is validated, so construction emits exactly one `UserWarning`. `GivensMinGRU` is continuous (no snap), so it never triggers this warning.

```python
import warnings

for spec in (["signed", "givens"], ["rotation", "rotation"]):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MinGRUStack(input_size=6, d_model=64, n_layers=2, mixer=spec)
        print(f"{spec} -> {len(w)} warning(s)")
```

**Output:**

```
['signed', 'givens'] -> 0 warning(s)
['rotation', 'rotation'] -> 1 warning(s)
```

Treat that warning as a signal to prefer a `GivensMinGRU` composer over stacked rotation blocks when you need a richer non-commutative composer.

## Why this order? The measured evidence

Layer order on `S3-hier` is not a matter of taste. It is measured: extract-then-compose beats the reverse, and a `GivensMinGRU` composer beats a stacked-rotation one at matched state. Putting the extractor first (`signed → givens`) holds well above chance out past $4\times$ training length, while reversing the order (`rotation → signed`) collapses toward chance, and stacking two rotation blocks trails a single Givens composer decisively at the same 64-element per-token state. That said, the rule is *extract-then-compose*, not *signed-always-first*: on the simpler `S3` probe, where each token already **is** the operation and there is nothing to extract, the ordering flips and `rotation → signed` matches or beats `signed → rotation`. Choose the order from the task's structure, not a fixed preference for either mixer.

The full multi-seed tables, the Fisher-exact significance tests, and the round tags behind these numbers live in [Choose a mixer](../how-to/choose-a-mixer.md#the-two-axis-guidance) and the [Givens & Delta deep dive](../explanation/givens-delta.md); both link back to the same underlying `experiments/EXPERIMENTS.md` rounds.

## Reproduce the full fit

The `signed → givens` numbers above are reproducible from a repo checkout. The [Reproduce the evidence](../how-to/reproduce-the-evidence.md) how-to gives the exact seed-0 command and the recorded output it must match (best checkpoint at step 1100, val@128 = 1.0, acc@64/256/512/1024 = 1.0 / 0.9941 / 0.9105 / 0.6619).

## What you built

You constructed a hierarchical stack (a `SignedMinGRU` extractor feeding a `GivensMinGRU` composer, the small-state option for extract-then-compose problems), configured each layer through the type-keyed `mixer_kwargs` schema, saw the warning that flags a fragile stacked-rotation ordering, and grounded the ordering choice in the measured `S3-hier` and `S3` evidence. Swap the composer for `mixer=["signed", "delta"]` with `mixer_kwargs={"delta": {"n_heads": 4, "nh": 2}}` instead if your per-token state is free to grow: [Your first delta model](first-delta-model.md) builds and trains exactly that stack.

## Next steps

- [Your first delta model](first-delta-model.md): swap the composer for `DeltaMinGRU` when your per-token state is free to grow, and learn its `(y_t, h_t)` step return.
- [Choose a mixer](../how-to/choose-a-mixer.md): the per-mixer decision table.
- [Reproduce the evidence](../how-to/reproduce-the-evidence.md): run the ordering experiment yourself.
- [Givens & Delta deep dive](../explanation/givens-delta.md): the brick-wall Givens parameterization, the chunked-WY delta-rule composer, and the evidence that separates them.
