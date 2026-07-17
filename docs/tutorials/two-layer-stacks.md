# Two-layer stacks

A single mixer tracks one kind of state. Hard sequence problems are often *hierarchical*: something has to be **extracted** from the raw input before the running quantity can be **composed**. `MinGRUStack` lets you assign a different mixer to each layer, so you can put an extractor underneath a composer. This tutorial builds the recommended stack for the repo's hierarchical probe — a `SignedMinGRU` extractor feeding a `GivensMinGRU` composer — and shows why that order, and that pairing, is the one to reach for.

By the end you will have constructed the ordered stack with the public API, configured each layer independently, seen the warning that flags a fragile ordering, and read the measured evidence that fixes the order for this class of task.

## The problem shape: extract, then compose

The motivating task is `S3-hier` (the repo's harder probe; chance $\approx 1/6$). The group operation is hidden inside a *pair* of sub-tokens: a running product over the permutation group $S_3$ is updated only when a pair completes, and a fixed Latin square maps each pair to its generator. Because the square is non-isotopic to any group of order six, no single-token shortcut exists — the generator must genuinely be **extracted** from the pair before it can be **composed** onto the running product. That is a two-stage computation, and it maps cleanly onto two layers: a lower layer that reads pairs into generators, an upper layer that accumulates them.

## Step 1 — Build the ordered stack

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

The stack forwards exactly like a homogeneous one — `(B, T, input_size) → (B, T, d_model)` plus one state per block. Only the per-layer mixer changed. `GivensMinGRU` requires `d_model` to be a multiple of its `block_size` (default 8); `64` satisfies that.

## Step 2 — Configure each layer independently

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

A flat `mixer_kwargs` dict alongside a list `mixer` (or a type-keyed dict alongside a single-string `mixer`) raises `ValueError` naming both schemas, so the two forms cannot be mixed up silently.

## Step 3 — Read the ordering warning

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

Layer order on `S3-hier` is not a matter of taste — it is measured. All figures below are multi-seed means at the standard 1600-step budget with best-val@128 checkpoint selection, from `experiments/EXPERIMENTS.md` (rounds `hetero-legB-v2`, `hetero-loop-17-sg8`) and the README's `S3-hier` table. Chance is $\approx 0.167$.

| stack (`mixer=`) | seeds | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|
| `["signed", "givens"]` (extract → compose) | 12 | 0.949 | 0.885 | 0.787 | 0.613 |
| `["signed", "rotation"]` (extract → 2D compose) | 6 | 0.572 | 0.442 | 0.375 | 0.309 |
| `["rotation", "signed"]` (compose → extract) | 3 | 0.448 | 0.325 | 0.247 | 0.206 |
| `["rotation", "rotation"]` | 3 | 0.365 | 0.256 | 0.212 | 0.189 |

Two facts drive the recommendation:

- **Extract before you compose.** Putting the extractor first (`signed → …`) beats the reverse (`rotation → signed`, near chance). Composing raw sub-tokens first builds a mixture that nothing downstream is observed to take apart.
- **Use a Givens composer, not a stacked-rotation one.** At a matched 64-element per-token state, swapping the 2D-rotation composer for `GivensMinGRU` raises the fit rate from **1 of 12 seeds to 8 of 12** (Fisher exact $p \approx 0.009$), for under +2.2% of full-stack parameters. The rounds ablation (`hetero-loop-19-rounds`) confirms this is the coupling: `rounds=1` (disjoint commuting planes) fits 0 of 12, `rounds=2` fits 6 of 12, `rounds=3` fits 8 of 12.

Like every continuous composer here, `signed → givens` still decays with length (0.613 @1024) rather than forming an exact length-invariant attractor. The large gaps — fit versus chance — are the trustworthy signal, not small differences between adjacent rows.

## Order is task-dependent

The rule is *extract-then-compose*, not *signed-always-first*. On the simpler `S3` probe, where each token **is** an operation (no extraction stage), there is nothing to extract first, and the ordering flips: `rotation → signed` matches or slightly beats `signed → rotation` (round `hetero-legA`, task `S3`, best-val@128, 3 seeds):

| stack (`mixer=`, task `S3`) | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|
| `["rotation", "signed"]` | 1.000 | 1.000 | 0.9986 | 0.9663 |
| `["signed", "rotation"]` | 0.999 | 0.976 | 0.9403 | 0.8626 |

So: choose the order from the *task's* structure. If raw tokens carry the operation, compose directly; if the operation is buried and must be recovered first, extract in the lower layer and compose above it.

## Reproduce the full fit

The `signed → givens` numbers above are reproducible from a repo checkout. The [Reproduce the evidence](../how-to/reproduce-the-evidence.md) how-to gives the exact seed-0 command and the recorded output it must match (best checkpoint at step 1100, val@128 = 1.0, acc@64/256/512/1024 = 1.0 / 0.9941 / 0.9105 / 0.6619).

## What you built

You constructed the recommended hierarchical stack — a `SignedMinGRU` extractor feeding a `GivensMinGRU` composer — configured each layer through the type-keyed `mixer_kwargs` schema, saw the warning that flags a fragile stacked-rotation ordering, and grounded the ordering choice in the measured `S3-hier` and `S3` evidence.

## Next steps

- [Choose a mixer](../how-to/choose-a-mixer.md) — the per-mixer decision table.
- [Reproduce the evidence](../how-to/reproduce-the-evidence.md) — run the ordering experiment yourself.
- [GivensMinGRU deep dive](../explanation/givens-mingru.md) — the brick-wall Givens parameterization and why it composes.
