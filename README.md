# minGRU

PyTorch implementation of the minGRU from Feng, Tung, Ahmed, Bengio &
Hajimirsadeghi, *Were RNNs All We Needed?* (arXiv:2410.01201), with a
parallel-scan training path, an O(1)-memory streaming path, chunked/TBPTT
state carry, and stacked residual blocks.

Single file, no dependencies beyond `torch`.

## Model

The minGRU removes the hidden-state dependency from the GRU's gates (which
eliminates the reset gate) and drops the tanh range restriction:

```
z_t  = sigmoid(Linear_z(x_t))
h~_t = Linear_h(x_t)
h_t  = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t
```

Because `z_t` and `h~_t` depend only on `x_t`, the recurrence is a
first-order linear scan `h_t = a_t ⊙ h_{t-1} + b_t`, computable over the
full sequence in parallel (Heinsen log-space scan via `logcumsumexp`) — no
BPTT. Parameter count is `O(2·d_h·d_x)` vs. `O(3·d_h·(d_x + d_h))` for a
standard GRU.

This implements the paper's **log-space variant** (their Appendix B, the
numerically stable form). It applies `g(x) = x + 0.5 (x ≥ 0), sigmoid(x)
(x < 0)` to candidate and initial states, which means **hidden states are
strictly positive**. That is a property of this parameterization, not of
the "vanilla" minGRU in the paper's Appendix A.

## Components

| class | role |
|---|---|
| `MinGRU` | one scan layer; parallel `forward`, recurrent `step` |
| `MinGRUBlock` | pre-norm residual block: LN → minGRU → +x, LN → MLP → +x |
| `MinGRUStack` | input projection → N blocks → final LN; full state threading |

`MinGRU` is deliberately atomic. A single layer's gates cannot condition on
accumulated state (that is the parallelism trade); stacking recovers
history-dependent gating with one level of indirection — layer *l*'s gates
see layer *l−1*'s hidden states — and the block MLP supplies the
cross-channel interaction the diagonal scan cannot (channel *i* never
touches channel *j* inside the recurrence). Set `mlp_expansion=0` for
scan-only blocks to ablate the channel mixer separately.

## Usage

Training (parallel over T):

```python
stack = MinGRUStack(input_size=32, d_model=256, n_layers=4)
y = stack(x)                          # (B, T, 32) -> (B, T, 256)
```

Streaming inference (O(1) memory; state is `n_layers × d_model` per sample):

```python
state = stack.init_state()
for x_t in transaction_stream:        # x_t: (B, input_size)
    y_t, state = stack.step(x_t, state)
```

Chunked / TBPTT training:

```python
carry = None
for chunk in sequence_chunks:         # each (B, T_chunk, input_size)
    y, carry = stack(chunk, state=carry, return_state=True)
    loss(y).backward()
    carry = [h.detach() for h in carry]
```

Learned initial state (module-owned, positivity by construction):

```python
m = MinGRU(32, 64, learnable_h0=True)   # zero-init == fixed g(0)=0.5 default
```

## State conventions

All exposed state is **real hidden state** — an output of `forward()` or
`step()` — under one convention across every entry point:

- `MinGRU.forward(h_0=...)` takes `(B, 1, d_h)`; `step(h_prev=...)` takes
  `(B, d_h)`. Crossing between streaming and chunked modes needs an
  explicit unsqueeze — intentional, no silent dim coercion.
- **Do not pass pre-activations.** This deviates from the paper's
  reference code, which applies `g` to `h_0` (treating it as a
  pre-activation). That convention makes the natural chunked-carry
  pattern `forward(h_0=prev[:, -1:])` silently wrong (double-applies
  `g`, ~0.43 max error vs. 1e-5 when fixed). Here `log(h_0)` is used
  directly; the learned-init use case, where the pre-activation
  convention is genuinely useful, is encapsulated behind
  `learnable_h0=True` instead of the call signature.
- Validation: strictly negative entries in `h_0` (the signature of
  pre-activation misuse) raise, via `torch._assert_async` — device-side
  on CUDA, so chunked loops incur no per-chunk host sync. Exact zeros
  are **accepted and clamped** to the dtype's smallest normal before
  `log()`: legitimately small states underflow to 0.0 in fp16/bf16
  (fp16 floor is ~6.1e-5; decayed states routinely sit below it), and
  rejecting them would fail valid carries. On CUDA the async assert
  surfaces at the next sync point, so a misuse traceback may point
  downstream of the offending call.

## Implementation notes

- **`log_g` gradient at 0.** The paper's reference `log_g` guards the
  `x ≥ 0` branch with `relu` (necessary: `torch.where` evaluates both
  branches, and an unguarded `(x + 0.5).log()` NaNs for `x < −0.5`,
  poisoning gradients even when unselected). But `relu'(0) = 0` gives
  `log_g` gradient exactly 0 at `x = 0` — outside the true
  subdifferential `[0.5, 2]` — which deadens any zero-initialized
  parameter fed through it (e.g. `learnable_h0` at its default init:
  no crash, no wrong output, the parameter just never trains). This
  implementation uses a nested `where` instead: value-identical to
  the relu guard, but the selected branch at 0 is plain `x`, giving
  the correct gradient `1/g(0) = 2` while keeping the unselected
  branch finite.
- `step()` runs under `@torch.no_grad()`. Training is intended through
  the parallel `forward`; if you need BPTT through the sequential path,
  remove the decorator.
- Parallel/sequential agreement is ~1e-5 in fp32 at T=128
  (`logcumsumexp` accumulation), not exact.

## Caveats for sequence modeling

- **Positive states.** Each block's scan output lives in `(0, ∞)` (the
  residual stream itself is signed). If hidden-state magnitude carries
  task signal, the halved per-dimension range is worth checking in a
  proxy run rather than assuming parity with a standard GRU.
- **Expressivity class.** Input-dependent, state-independent transitions
  put minGRU in the same class as Mamba/S6 and GLA: fixed-depth stacks
  are TC⁰ (Merrill et al., *The Illusion of State in State-Space
  Models*) and cannot do unbounded state tracking that a single
  nonlinear GRU layer can. Depth recovers bounded-depth nonlinear
  interaction with history, not sequential computation. Depth-matched
  comparisons against a standard GRU confound layer count with
  mechanism; include a deeper minGRU variant in any comparison grid.

## Tests

`python min_gru.py` runs the built-in suite: parallel-vs-sequential
equivalence (single layer, stack, and with `learnable_h0` off its
default), chunked-vs-full equivalence for both `MinGRU` and
`MinGRUStack`, gradient flow through the scan and into `h0_pre` **from
zero-init** (guards the `log_g` fix), `log_g` gradient at 0 equal to 2,
and `h_0` validation (underflowed zeros accepted, negatives raise).

## Reference

Feng, L., Tung, F., Ahmed, M. O., Bengio, Y., & Hajimirsadeghi, H.
(2024). *Were RNNs All We Needed?* arXiv:2410.01201.
