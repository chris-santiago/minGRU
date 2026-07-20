# Triton scan kernels

## Orientation

The four scan operations that back the minGRU mixers ([`linear_scan`, `matrix_scan`, `matrix_affine_scan`, and `parallel_scan_log`](../reference/scan-ops.md)) are written in pure PyTorch as Hillis-Steele associative scans. They are correct and portable, but on a GPU the pure-torch matrix scans launch a great many small kernels and materialize large intermediate tensors. This library ships an optional Triton backend that fuses each scan into a handful of custom kernels. This article explains why the four operations are really one family, how the kernels are laid out, the precision decisions that keep them bit-parity with the eager reference, and the measured results, including the rows where the Triton path is *slower* than eager, which are reported as plainly as the wins.

The backend is entirely optional and off by default on CPU. For how to turn it on and control which path runs, see [How-to: control scan dispatch](../how-to/control-scan-dispatch.md) and [Tutorials: Triton on GPU](../tutorials/triton-on-gpu.md). For the registered symbols, see [Reference: Triton kernels](../reference/triton-scans.md). To reproduce the numbers below, see [How-to: run the benchmarks](../how-to/run-the-benchmarks.md).

`DeltaMinGRU`'s chunked-WY `forward` is not one of the four scan ops and gains no hand-written Triton kernel here: a CUDA fusion-headroom probe found `torch.compile` already recovers 70–91% of the available fusion headroom on that path, so a bespoke kernel was judged not worth building. See the [Givens & Delta deep dive](givens-delta.md#costs-and-the-two-axis-recommendation) for that measurement and the GPU cost comparison between the two composers.

## One associative-scan family

Every mixer's recurrence is a first-order affine map $h_t = A_t\, h_{t-1} + B_t$. What differs between mixers is only the shape of $A_t$: a positive scalar in $(0,1)$ for the base log-space `MinGRU`, a signed scalar for `SignedMinGRU`, a $2 \times 2$ matrix for `RotationMinGRU`, a $k \times k$ matrix for `GivensMinGRU`. An affine map like this composes associatively. Composing "apply $(A_1, B_1)$ then $(A_2, B_2)$" gives

$$(A_1, B_1) \circ (A_2, B_2) = (A_2 A_1,\; A_2 B_1 + B_2),$$

which is a monoid: it has an identity $(I, 0)$ and it is associative, so the running prefix over a sequence can be computed by any grouping of the steps rather than strictly left to right. That associativity is exactly what licenses a parallel scan in place of step-by-step recurrence, and it is the single mathematical fact the whole backend rests on. The matrix product $A_2 A_1$ does *not* commute for the $2 \times 2$ and $k \times k$ cases (which is the whole point of the rotation family), but non-commutativity does not break associativity, so the scan is still valid.

Three of the four ops are this affine scan at different block sizes: `linear_scan` is $k = 1$ (a channel-tiled scalar scan), `matrix_scan` is $k = 2$ with injection width $v = 1$, and `matrix_affine_scan` is generic $(k, v)$. The fourth, `parallel_scan_log`, is the same recurrence carried in log-space for numerical stability: a running log-coefficient cumulative sum plus an online max-shifted log-sum-exp accumulator, writing $h = \exp(\cdot)$ directly. So all four have a single conceptual core with two numerical realizations (linear-space and log-space), and all four appear in the `SCAN_IMPLS` registry the dispatcher consults.

## Kernel design

The generic affine-scan forward kernel assigns **one Triton program to a single $(b, n)$ lane** (one batch element, one state block) and walks the whole sequence in $T$ sequentially inside that program, carrying the running prefix $(\bar{A}, \bar{B})$ in registers across timesteps. This is a deliberate inversion of the pure-torch strategy: rather than a $O(\log T)$-depth doubling that materializes a full $(B, T, \dots)$ intermediate at every level, each program keeps its prefix live in registers and never writes it out, so the only global memory traffic is reading the inputs and writing the outputs. The lanes are wide (one per $(b, n)$), so the launch saturates the device across the batch-and-block dimension while the sequential-in-$T$ inner loop stays in registers.

The block size $k$ and injection width $v$ are `tl.constexpr` template parameters. Every member of the kernel envelope is a power of two, so the kernels tile $k$ and $v$ exactly with no masking, and specializing on them as compile-time constants lets the compiler unroll the inner matrix arithmetic and emit a single branchless kernel per shape. The two elementwise-in-$D$ operations (`linear_scan` and `parallel_scan_log`) use a different, channel-tiled layout: each program walks the full $T$ sequence for a `BLOCK_D`-wide tile of channels at once, carrying a scalar prefix per channel, which keeps the lanes wide when there is no matrix structure to exploit.

**Accumulation is fp32 regardless of input dtype.** Every kernel loads its inputs, casts to fp32, accumulates the prefix in fp32, and casts back to the output dtype only at the store. This matches the eager reference's numerical structure closely enough that the fp32 outputs stay parity-tight, and it means a bf16 or fp16 model does not accumulate a low-precision prefix over hundreds of steps.

## The TF32 / IEEE precision guard

There is one place where the obvious kernel is silently wrong, and the fix is worth explaining because it is the kind of bug that passes every casual test. The generic core's prefix update multiplies two $k \times k$ matrices, which is naturally expressed as a Triton `tl.dot`. On recent NVIDIA architectures (validated on an L4, sm_89), a `tl.dot` at $k \geq 16$ is large enough that Triton routes it through the tensor-core MMA path, and that path **defaults to TF32**: roughly $2^{-10}$ relative error, about $2 \times 10^{-4}$ absolute at the scaled inputs here. That is far coarser than the flat $10^{-5}$ fp32 conformance gate the outputs are held to, so a naive kernel would produce results that look plausible, pass a loose eyeball check, and fail the parity gate at the largest envelope block size.

The fix is an explicit `tl.dot(..., input_precision="ieee")`, which forces bit-exact fp32 on this architecture, applied precisely where the $(k, k, k)$ matrix product and the corresponding injection contraction occur. It bites only at the envelope's $k = 16$ (and, for the injection contraction, the larger $v$), so the precision override is placed behind the same `tl.constexpr` $k$/$v$ guards that specialize the rest of the kernel, meaning it compiles to a single branch per kernel with no runtime cost on the smaller blocks that never hit the MMA path. The kernel envelope itself is $k, v \in \{1, 2, 4, 8, 16\}$ for any $T \geq 1$ including non-power-of-two lengths; a shape outside the envelope raises `ScanFallback` rather than silently masking or miscomputing.

## Autograd integration and the reversed-scan backward

The kernels are registered with `torch.library.triton_op`, so `torch.compile` sees them as opaque custom ops without graph breaks, and their gradients are wired with `torch.library.register_autograd` on the forward `triton_op`. This is the torch 2.8-idiomatic route: the backward composes with `torch.compile`'s tracing and dispatcher, which a hand-rolled `autograd.Function` wrapper around the op would not.

The backward design mirrors the forward's zero-extra-memory discipline. For the generic affine core, the adjoint is a *single reverse-direction scan* that reads only the forward's inputs and outputs: it reverse-scans the incoming output gradients with $A_{t+1}^\top$, then forms $\partial L / \partial A_t$ from the forward outputs $\bar{A}_{t-1}$, $\bar{B}_{t-1}$ (seeded with $I$/$0$ at $t = 1$) and $\partial L / \partial B_t$ directly. No intermediate activations are saved beyond what the forward already returned. `parallel_scan_log`'s backward takes a different but equally frugal route: autograd-through-recomputation, saving only its two forward inputs (`log_coeffs`, `log_values`) and re-deriving the gradient through the eager log-space math, exact-to-eager, with no hand-derived log-space adjoint kernel to get subtly wrong on a device the developer cannot see locally.

The angle-fused Kernel 4 that serves `GivensMinGRU` and `RotationMinGRU` is a separate module-level fast path, not one of the four scan ops. Its backward is an **exact $C = 1$ stored-state recompute** rather than a reversal by division, a decision with its own measured justification (reversal error grows as $\sigma_{\min}^{-\text{chunklen}}$, reaching $9.3 \times 10^{19}$ at strong decay), covered in full in the [Givens & Delta deep dive](givens-delta.md#the-backward-pass-why-division-based-reversal-was-rejected). The consequence for this article is its memory footprint, below.

## The dispatch seam

Which implementation runs is controlled by the `MINGRU_SCAN` environment variable, resolved to one of `auto`, `eager`, or `triton`:

- **`eager`** (and `auto` on CPU, or when the Triton module cannot import below the torch $\geq 2.8$ floor) runs the pure-torch reference: the recorded evidence path, byte-identical to what the lab campaign trained on.
- **`auto`** on CUDA attempts the Triton kernel and, if the inputs fall outside the envelope or a launch fails (a `ScanFallback`), **warns once** and runs the eager reference. Never a wrong result, never a crash, and never more than one warning per process.
- **`triton`** demands the kernel: a `ScanFallback` or an unavailable Triton module re-raises as a `RuntimeError` rather than silently downgrading, so a benchmarking or correctness run cannot accidentally measure the eager path.

The seam is lazy: `import mingru` never imports the Triton module, and touching a Triton-backed symbol is what triggers its import. See [How-to: control scan dispatch](../how-to/control-scan-dispatch.md) for the operational details.

## Measured results

All numbers below are from `experiments/bench/scan_bench.md` (torch 2.8.0+cu128, NVIDIA L4), reporting the Triton speedup against the eager reference for a forward pass and a forward-plus-backward pass, at two shapes: a "lab" shape ($B=128$, $T=64$) and a "long-T" shape ($B=16$, $T=1024$).

| op | direction | lab speedup | long-T speedup |
|---|---|---|---|
| `matrix_scan` ($k=2$) | fwd | 16.11x | 25.06x |
| `matrix_scan` ($k=2$) | fwd+bwd | 94.20x | 168.53x |
| `matrix_affine_scan` ($k=8$) | fwd | 19.09x | 23.08x |
| `matrix_affine_scan` ($k=8$) | fwd+bwd | 38.89x | 53.56x |
| `linear_scan` ($k=1$) | fwd | 5.24x | 5.81x |
| `linear_scan` ($k=1$) | fwd+bwd | 3.01x | 4.64x |
| `parallel_scan_log` | fwd | 1.02x | 1.57x |
| `parallel_scan_log` | fwd+bwd | **0.70x** | **0.80x** |

The matrix scans are where the fusion pays off enormously: the pure-torch matrix scan's backward launches many small kernels and materializes large intermediates, so the Triton fwd+bwd reaches $94\times$ at the lab shape and $168\times$ at long-$T$ for `matrix_scan`, and $39$–$54\times$ for `matrix_affine_scan`. `linear_scan` gets a solid $3$–$6\times$.

The `parallel_scan_log` rows are the honest exception and must be read as they are. The forward is roughly break-even to modestly ahead ($1.02$–$1.57\times$), but the **fwd+bwd is 0.70x at the lab shape and 0.80x at long-T**: the Triton path is *slower* than eager for the base log-space mixer's full training step. The cause is the recompute-backward tradeoff: `parallel_scan_log`'s gradient is autograd-through-recomputation, which re-runs the eager log-space math, so on this op the Triton path pays for recomputation without a large enough forward win to offset it. The practical consequence is stated plainly: under `MINGRU_SCAN=auto`, enabling the Triton backend currently *slows* training of the baseline log-space mixer, and **`eager` is the right choice for that mixer today.** The backend earns its place on the matrix scans, not on the log-space scan.

### Memory

For the angle-fused `GivensMinGRU` backward at $B=128$, $T=64$, $d=64$, $k=8$, peak memory is **38.25 MB** for the angle-fused $C=1$ path against **395.09 MB** for the generic Phase-1 backward that materializes the $k \times k$ transitions, a $10.3\times$ reduction (`experiments/bench/scan_memory.md`). This is the payoff of never materializing the transition matrices and of reusing the forward output as the backward's state anchor at zero extra allocation.

### Conformance

Correctness is gated by a parity harness that compares every Triton output against the eager reference across the envelope: **590 of 590 rows pass** (`experiments/bench/scan_parity.md`, torch 2.8.0+cu128, triton 3.4.0, L4). The tolerance gates are adjudicated per op and direction: outputs are held to $(1\mathrm{e}{-5}, 0)$ for the affine scans and $(1\mathrm{e}{-4}, 1\mathrm{e}{-4})$ for the log-space scan (the looser gate reflects the `logcumsumexp` round-trip), and gradients to $(1\mathrm{e}{-3}, 0)$ for the affine scans and $(1\mathrm{e}{-3}, 1\mathrm{e}{-3})$ for the angle-fused path. The absolute output errors sit well inside these gates (typically $10^{-8}$ to $10^{-7}$), which is what the fp32 accumulation and the IEEE `tl.dot` guard are there to guarantee.

## See also

- [Reference: Triton kernels](../reference/triton-scans.md) and [Reference: scan operations](../reference/scan-ops.md): the registered symbols and the four scan functions.
- [How-to: control scan dispatch](../how-to/control-scan-dispatch.md): setting `MINGRU_SCAN` and reading the fallback warning.
- [How-to: run the benchmarks](../how-to/run-the-benchmarks.md): reproducing the speedup, memory, and parity tables.
- [Tutorials: Triton on GPU](../tutorials/triton-on-gpu.md): enabling the backend end to end.
- [Givens & Delta deep dive](givens-delta.md): the angle-fused kernel's backward decision and the rotation-family mixers it serves.
