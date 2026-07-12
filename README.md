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

| class / fn | role |
|---|---|
| `MinGRU` | one scan layer (log-space, positive states); parallel `forward`, recurrent `step` |
| `SignedMinGRU` | signed-transition variant (linear-space scan, unconstrained states); same API |
| `linear_scan` | Hillis–Steele associative scan for `h_t = a_t·h_{t−1} + b_t` with signed coefficients |
| `MinGRUBlock` | pre-norm residual block: LN → mixer → +x, LN → MLP → +x (`signed=True` selects the mixer) |
| `MinGRUStack` | input projection → N blocks → final LN; full state threading; `signed=True` throughout |

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
for x_t in event_stream:              # x_t: (B, input_size)
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

- **Positive states (log-space variant).** Each `MinGRU` block's scan
  output lives in `(0, ∞)` (the residual stream itself is signed). If
  hidden-state magnitude carries task signal, the halved per-dimension
  range is worth checking in a proxy run rather than assuming parity
  with a standard GRU. `SignedMinGRU` removes this constraint.
- **Expressivity class.** Input-dependent, state-independent transitions
  put minGRU in the same class as Mamba/S6 and GLA: fixed-depth stacks
  are TC⁰ (Merrill et al., *The Illusion of State in State-Space
  Models*) and cannot do unbounded state tracking that a single
  nonlinear GRU layer can. Depth recovers bounded-depth nonlinear
  interaction with history, not sequential computation. Depth-matched
  comparisons against a standard GRU confound layer count with
  mechanism; include a deeper minGRU variant in any comparison grid.
  The probes below make both points measurable.

## Signed variant (`SignedMinGRU`)

The log-space scan requires positive transition coefficients (log of
1−z), which hard-codes two limitations at once: hidden states confined
to (0, ∞) via `g`, and transition eigenvalues confined to (0, 1) —
monotone EWMA-style memory only. `SignedMinGRU` lifts both by switching
to a linear-space associative scan (`linear_scan`) and computing

```
a_t = (1 − z_t) ⊙ tanh(Linear_s(x_t))      # eigenvalues in (−1, 1)
h_t = a_t ⊙ h_{t−1} + z_t ⊙ Linear_h(x_t)  # no g, states unconstrained
```

When the sign head saturates positive this reduces to the vanilla
(Appendix A) minGRU. Motivation comes from Merrill, Petty & Sabharwal,
*The Illusion of State in State-Space Models* (ICML 2024,
arXiv:2404.08819) and Grazzi et al., *Unlocking State-Tracking in
Linear RNNs Through Negative Eigenvalues* (ICLR 2025): negative
eigenvalues restore per-layer parity / sign-alternation dynamics.
What it does **not** restore: diagonal transitions commute, so
non-abelian state tracking (permutation composition and everything
NC¹-complete) remains out of reach at any width — that requires
non-diagonal transitions or state-dependent gates (i.e., a GRU). All
diagonal variants, signed or not, remain in TC⁰ per Merrill et al.'s
iterated-scalar-product argument.

Practical differences from `MinGRU`: any real `h_0` is legal (no
positivity check, no underflow clamp), parallel/sequential agreement is
~1e−7 rather than ~1e−5 (no exp/log round-trip), and there are 3 linear
heads instead of 2 (mind parameter-matched comparisons). The scan is
O(T log T) work / O(log T) depth in pure torch ops.

The expressivity ladder, per layer:

| variant | scan | per-layer capability |
|---|---|---|
| `MinGRU` (a ∈ (0,1)) | log-space | monotone EWMA memory |
| `SignedMinGRU` (a ∈ (−1,1)) | linear-space | + parity, abelian tracking |
| non-diagonal (not implemented) | matrix-combine | + non-commutative, toward S₅ |
| state-dependent gates | none (sequential) | = a standard GRU |

## Expressivity probes

`probes.py` tests the ladder empirically on two word problems (seq2seq
tagging with dense supervision, following Merrill et al.'s setup):
**parity** (running XOR over {0,1}; in TC⁰, but the one-scan solution
needs an eigenvalue at −1) and **S3** (running product in the smallest
non-abelian group; order-sensitive, so commutative scans of any sign
should fail per layer). Models train at T=64 and are evaluated at T=64
(in-distribution) and T=256 (length generalization) — the length-gen
column is what separates "expresses the recurrent solution" from
"learned a depth-bounded shortcut for the training length."

Results (d_model=64, batch 128, Adam lr 3e−3, budget 1600 steps with
early stop at 99.9% train-length accuracy; single seed):

| task | model | layers | acc@64 | acc@256 | outcome |
|---|---|---|---|---|---|
| parity | GRU | 1 | 1.000 | 1.000 | solved, step 100 |
| parity | minGRU | 1 | 0.525 | 0.507 | chance |
| parity | minGRU | 4 | 0.516 | 0.504 | chance |
| parity | minGRU-signed | 1 | 1.000 | 0.859 | solved; decays at 4× length |
| parity | minGRU-signed | 4 | 1.000 | 1.000 | solved, step 250 |
| S3 | GRU | 1 | 1.000 | 1.000 | solved, step 300 |
| S3 | minGRU | 1 | 0.243 | 0.188 | near chance |
| S3 | minGRU | 4 | 0.266 | 0.193 | near chance; loss ~1.5 |
| S3 | minGRU-signed | 1 | 0.372 | 0.334 | abelian quotient only |
| S3 | minGRU-signed | 4 | 0.993 | 0.655 | passes @64, fails @256 |

Interpretation:

- **Parity separates the sign.** Positive-diagonal minGRU sits at
  chance even at depth 4; the signed variant solves it in a few hundred
  steps. The signed L=1 decay at 4× length reflects the strict
  |a| < 1 of the `(1−z)·tanh` parameterization (the parity eigenvalue
  can approach −1 but not reach it); depth 4 compensates fully.
- **S3 separates commutativity.** The signed variant's L=1 plateau
  (~0.37) is consistent with capturing the sign-of-permutation quotient
  (Z₂) while non-abelian structure stays out of reach. At L=4 it grinds
  to 0.993 at training length but only 0.655 at 4× — depth substituting
  for recurrence, i.e., a length-bounded shortcut, exactly the pattern
  Merrill et al. report for Mamba/S4/transformers. The one-layer GRU is
  1.000/1.000 on both tasks: the automaton construction.
- **The positive variant's failure is optimization, not just
  expressivity.** Four layers of MLPs should make parity@64
  expressible, and depth demonstrably suffices for the otherwise
  identical signed sibling — yet the positive variant stays at chance
  on both tasks with loss barely moving. Within this budget, the
  positive parameterization is not just one rung down the ladder but
  hostile to gradient descent on order-sensitive structure.

Caveats: single seed per cell, one learning rate, and null results are
budget-relative ("didn't learn in 1600 steps" ≠ "cannot learn"). The
minGRU wrappers include a block MLP the GRU baseline lacks, which
favors the minGRU variants — strengthening their negative results,
mildly weakening attribution of their positive ones.

### Reproducing

```
python probes.py TASK MODEL [N_LAYERS]
# TASK   ∈ {parity, S3}
# MODEL  ∈ {GRU, minGRU, minGRU-signed}
# N_LAYERS defaults to 1; MAX_STEPS env overrides the budget (1600)
```

The full grid above is ten invocations:

```
for t in parity S3; do
  python probes.py $t GRU
  for m in minGRU minGRU-signed; do
    python probes.py $t $m 1
    python probes.py $t $m 4
  done
done
```

CPU-only is sufficient (each cell runs seconds to ~15 minutes,
1-layer cells fastest). The original runs used a fixed seed; expect
small numeric differences across torch versions but the same
qualitative pattern.

## Tests

`python min_gru.py` runs the built-in suite: parallel-vs-sequential
equivalence (single layer, stack, `learnable_h0` off its default, and
`SignedMinGRU` plus a signed stack), chunked-vs-full equivalence for
`MinGRU`, `MinGRUStack`, and `SignedMinGRU` (including a carry with
negative states), gradient flow through both scans and into `h0_pre`
**from zero-init** (guards the `log_g` fix), `log_g` gradient at 0
equal to 2, and `h_0` validation for the log-space variant
(underflowed zeros accepted, negatives raise).

## References

Feng, L., Tung, F., Ahmed, M. O., Bengio, Y., & Hajimirsadeghi, H.
(2024). *Were RNNs All We Needed?* arXiv:2410.01201.

Merrill, W., Petty, J., & Sabharwal, A. (2024). *The Illusion of State
in State-Space Models.* ICML 2024. arXiv:2404.08819.

Grazzi, R., et al. (2025). *Unlocking State-Tracking in Linear RNNs
Through Negative Eigenvalues.* ICLR 2025.
