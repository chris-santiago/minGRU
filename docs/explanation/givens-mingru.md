# GivensMinGRU: a deep dive

## Orientation

`GivensMinGRU` is the richest mixer in this library. Like `RotationMinGRU`, it sits on the non-commutative rung of the [minGRU ladder](../reference/mixers.md): its per-token transition is a non-diagonal orthogonal map, so a single layer can track state-tracking automata over non-abelian groups that no diagonal scan can represent at any width. What distinguishes it from `RotationMinGRU` is the *shape* of that map. `RotationMinGRU` gives each transition one $2 \times 2$ planar rotation; `GivensMinGRU` builds each transition from a product of Givens rotations acting on a `block_size`-dimensional block of state, a strictly richer family of orthogonal maps carried at the same per-token state.

This article explains why that richer map exists, how it is parameterized and why that particular parameterization, what the measured evidence establishes about it, and the two design decisions that are easy to get wrong: the choice of the brick-wall mesh, and the backward pass. For the constructor signature and keyword arguments, see [Reference: mixers](../reference/mixers.md). For the choice between `GivensMinGRU` and the alternatives on a real task, see [How-to: choose a mixer](../how-to/choose-a-mixer.md).

A single framing runs through everything below: **the extra structure is a trainability lever, not a jump in expressivity class.** An $8$-dimensional Givens product tracks the same solvable-group automata a $2 \times 2$ rotation does, and both stay below the state-dependent-gate line that separates this whole family from a true recurrence. What the richer map changes is how *often* training finds a working composer. Keeping that distinction sharp is what lets the evidence be read honestly.

## The rung it enriches

The base minGRU recurrence is a first-order linear scan $h_t = a_t \odot h_{t-1} + b_t$ whose transition $a_t$ is diagonal and, in the log-space parameterization, confined to $(0, 1)$. Two structural limits follow, and both are expressivity limits rather than optimization difficulties: memory can only decay monotonically, and the transition is diagonal, hence commutative, so the running product is invariant to input reordering. `SignedMinGRU` lifts the first limit by admitting $a_t \in (-1, 1)$. Neither the base nor the signed variant lifts the second: diagonal transitions commute under scalar multiplication, so permutation composition — where order changes the answer — stays out of reach at any width.

Lifting commutativity requires a non-diagonal transition. `RotationMinGRU` makes each per-block transition a full $2 \times 2$ affine map $M_t = R(\theta_t)\,\mathrm{diag}(1, \tanh u_t)$, an $O(2)$-valued family that generates the dihedral group. Since $D_3 \cong S_3$ (the smallest non-abelian group) embeds in $O(2)$, one layer can exactly represent the $S_3$ running-product automaton. `GivensMinGRU` occupies this same rung with a more expressive per-token map: instead of one plane it composes rotations across many planes of a larger block, reaching all of $SO(k)$ in the limit while keeping the per-token cost linear in width. It is emphatically not a fifth rung above rotation. The formal boundary — that all four mixers are input-dependent, state-independent linear transitions and therefore live in $\mathrm{TC}^0$, unable to do the unbounded state tracking a single nonlinear GRU layer can — is unchanged by the block size. See `experiments/TECHNICAL_REPORT.md` §1–2 for the full expressivity argument.

## The brick-wall Givens parameterization

### What a Givens rotation is

A Givens rotation is the smallest possible rotation. It acts on one coordinate plane of a $k$-dimensional space and leaves the other $k-2$ directions fixed: the identity matrix everywhere except for $\cos$/$\sin$ entries in the two rows of the chosen plane. It is the atom of orthogonal structure, exactly orthogonal at every parameter value with no normalization step, determinant $+1$, one angle per plane. In this sense `RotationMinGRU` is already a Givens machine at $k = 2$: one plane, one input-dependent angle per token.

### Composing planes in a brick-wall mesh

`GivensMinGRU` composes many Givens rotations per token. At the default `block_size` $= 8$, `rounds` $= 3$, each per-token transition is a product of three brick-wall layers of Givens rotations across an $8$-dimensional block:

- round 0 pairs the planes $(0,1),(2,3),(4,5),(6,7)$;
- round 1 pairs the staggered planes $(1,2),(3,4),(5,6),(7,0)$;
- round 2 repeats round 0's pattern.

All angles are emitted by one linear head from the current input. The crucial structural fact is what commutes and what does not. Disjoint planes *within* a round share no coordinate, so they commute and can be applied in any order; the mesh is embarrassingly parallel inside a round. The *stagger between rounds* is what couples all eight dimensions: round 1's plane $(1,2)$ shares a coordinate with round 0's $(0,1)$ and $(2,3)$, so the composition of round 0 then round 1 does not factor into independent planes. This is the entire point of the mesh, and, as the [rounds ablation](#the-rounds-ablation-map-richness-versus-block-size) below shows, it is measurably the source of the variant's advantage.

Products of enough Givens rotations reach all of $SO(k)$, so three rounds are a deliberate compute budget, not a representational ceiling. The result stays continuous, stays exactly special-orthogonal (no normalization, no snap grid), stays parallel through the same associative scan generalized to $k \times k$ blocks — [`matrix_affine_scan`](../reference/scan-ops.md) — and keeps the standard $64$-element per-token state at the repository's `d_model`.

### Why brick-wall, specifically

The brick-wall mesh is not an ad-hoc choice. It is the standard construction from the orthogonal/unitary-RNN literature: the EUNN factorization of Jing et al. (ICML 2017), built on the rectangular interferometer mesh of Clements et al. (2016), with the Givens rotation as described in Golub & Van Loan, *Matrix Computations*. The mesh's appeal is that it reaches a wide swath of $SO(k)$ using only $O(k)$ angles per layer — one per plane — rather than the $O(k^2)$ parameters a dense orthogonal transition would need, and it does so with a fixed, hardware-friendly connectivity pattern. What is specific to this variant is not the factorization but the two things layered on top of it: the angles are *input-dependent per token* (a fresh transition each step, emitted from $x_t$), and the contribution is the *measured trainability result* below, not a claim about the mesh itself.

### Practical shape

`GivensMinGRU` carries three linear heads ($\theta$, $z$, $h$) against `RotationMinGRU`'s four and `SignedMinGRU`'s three — worth remembering in parameter-matched comparisons. It owns its initial state $h_0$ as an intrinsic learned parameter with no `learnable_h0` flag: a zero state has no orbit under the group action, so it cannot demonstrate tracking. It requires `hidden_size` divisible by an even `block_size`. Numerical agreement between the parallel scan and the sequential `step` path is $\approx 10^{-5}$ at $T=128$ in fp32 — slightly looser than the other mixers because a product of `rounds` $k \times k$ blocks per step accumulates more matrix arithmetic than a single $2 \times 2$. Full signatures and validation rules are in [Reference: mixers](../reference/mixers.md).

## What the richer map demonstrates against diagonal RNNs

The reason a non-commutative transition matters is best seen on the task that isolates it. Diagonal transitions commute, so a stack of them yields the same final state under any reordering of the inputs. A task whose labels *change* under reordering — composing permutations, where doing $g$ then $h$ differs from $h$ then $g$ — therefore cannot be represented by any diagonal variant at any width. This is not a training gap that more depth or more steps would close; it is a representability wall. The rotation family is what climbs over it.

The evidence base is the `S3-hier` task, an extract-then-compose problem harder than the plain `S3` probe. The group operation is hidden inside a *pair* of sub-tokens drawn uniformly from $\{0,\dots,5\}$: each consecutive pair $(x_{2k}, x_{2k+1})$ selects a generator $g = \mathrm{LATIN}[a, b]$, composed onto the running $S_3$ product when the pair completes (chance $\approx 1/6 \approx 0.167$). `LATIN` is a fixed $6 \times 6$ Latin square, verified non-isotopic to both groups of order six ($\mathbb{Z}_6$ and $S_3$), so a single sub-token carries no information about the generator and no rotation layer can absorb the lookup by relabeling its angle assignment. Extraction is genuine work that must precede composition, which is why the promoted configuration is a two-layer stack: a signed extractor below, a Givens composer above. The task construction and the isotopy verification are documented in `probes.py` and `experiments/TECHNICAL_REPORT.md` §3.1.

Two properties make this evidence unusually trustworthy. First, `S3-hier` is a reliability finding, not an exactness one: because even an unconstrained GRU sits at chance here in budget ($0.171$ at $T=1024$), the task measures *which inductive bias trains reliably under the $1600$-step budget*, not what is representable. Composers differ in how often training finds a working solution, not in how well found solutions generalize. Second, the promoted code path is provably the evidence path.

### The seed-0 provenance transfer

The multi-seed campaign that produced the `GivensMinGRU` numbers was trained through the lab harness (`experiments/hetero_lab.py`), not through the shipped `MinGRUStack`. To close that gap, one `probes.py` run of the promoted registry row `minGRU-hetero-sg8` on `S3-hier`, seed 0, under the `CKPT=1` best-val@128 protocol at `MAX_STEPS=1600`, was compared field-by-field against the recorded lab row, under the `torch==2.5.1` evidence pin (round `givens-promotion-replication-01` in `experiments/EXPERIMENTS.md`):

| field | recorded | replicated | match |
|---|---|---|---|
| best-checkpoint step | 1100 | 1100 | yes |
| checkpoint val@128 | 1.000 | 1.000 | yes |
| acc@64 | 1.000 | 1.000 | yes |
| acc@256 | 0.9941 | 0.9941 | yes |
| acc@512 | 0.9105 | 0.9105 | yes |
| acc@1024 | 0.6619 | 0.6619 | yes |

Exact match on every metric, no tolerance widening. This transfers the pooled twelve-seed evidence onto the promoted `min_gru.py` code path without rerunning the campaign, and it is the single row a skeptic should check first: it demonstrates that the number the README reports for the shipped class is the number that was actually measured. The `1.000 / 0.9941 / 0.9105 / 0.6619` shape across $T = 64/256/512/1024$ is also the microcosm of the whole story — a near-perfect in-distribution fit that decays with length, which the trade-offs section explains.

### The map-richness result

At matched $64$-element per-token state, $8$-dimensional Givens blocks fit `S3-hier` on **8 of 12 seeds** against **1 of 12** for continuous $2$D rotation blocks (Fisher exact $p \approx 0.0094$). The separation is threshold-robust: at fit thresholds $\{0.98, 0.99, 0.995\}$ the counts read $9/12$, $8/12$, $8/12$ against $1/12$ at all three ($p = 0.0028, 0.0094, 0.0094$). Pooled across all twelve seeds, the `signed → givens8` profile is $0.949 / 0.885 / 0.787 / 0.613$ at $T = 64/256/512/1024$; among the seeds that fit, it holds $0.927$ at $T=512$ and $0.733$ at $T=1024$ (best seed $0.956 / 0.812$). The full cross-mechanism table, including the delta-rule comparison, is in `experiments/TECHNICAL_REPORT.md` §4.4 and the README's `S3-hier` section.

Reliability is separable from generalization quality. Seeds that fit generalize equally under either mechanism — fit-only means of $0.733$ (Givens) against $0.739$ (a small delta-rule composer) at $T=1024$ — so the mechanisms differ in fit rate, not in the quality of the solutions they find. That is the honest shape of the win: not "Givens generalizes better," but "Givens trains to a working composer far more often at matched state."

## The rounds ablation: map richness versus block size

The map-richness comparison on its own moves two things at once: per-token map expressivity ($SO(8)$ versus $SO(2)$) *and* within-scan block connectivity (eight coupled channels per block versus two). An external adversarial review of the technical report flagged this exact confound — "map richness raises fit rate" was supported only correlationally. The rounds ablation (round `hetero-loop-19-rounds`, n=12 per arm, same stack and protocol) resolves it by holding `block_size` $= 8$ fixed and varying `rounds`:

| rounds | per-token map | fits | acc@64 | acc@1024 | fit-only @512 / @1024 |
|---|---|---|---|---|---|
| 1 | 4 disjoint commuting planes | 0/12 | 0.376 | 0.182 | — |
| 2 | + one staggered coupling layer | 6/12 | 0.805 | 0.515 | 0.916 / 0.704 |
| 3 | + repeat (recorded arm) | 8/12 | 0.949 | 0.613 | 0.927 / 0.733 |

At `rounds` $= 1$ the brick-wall mesh is four disjoint, commuting $2$D planes inside the $8$D block: thirty-two planes per token, plane-count-matched to the $2$D composer's thirty-two blocks, but with the larger coupled state block. It fits **0/12** — no better than the $2$D composer's $1/12$ (Fisher $p = 1.0$) despite the bigger block. This refutes the block-size hypothesis outright: a larger block of *commuting* planes buys nothing. The single staggered coupling layer of `rounds` $= 2$ — the one layer that breaks within-block commutativity — recovers most of the effect ($6/12$; $p \approx 0.014$ versus `rounds` $= 1$), and the third round adds a statistically inseparable increment ($8/12$; $p \approx 0.68$ versus `rounds` $= 2$). The endpoint separation $0/12$ versus $8/12$ is $p \approx 0.0013$, and the ordering is threshold-stable. Fit-only generalization is unchanged across rounds ($0.704$ versus $0.733$ at $T=1024$), reproducing the reliability-versus-quality separation.

The reading is clean: **the lever is the commutativity-breaking coupling, not the block size.** One residual caveat applies to the cross-family anchor only — `rounds` $= 1$ is a pure rotation with no $\tanh u$ scale channel while the $2$D composer carries one, so their equivalence is suggestive; the within-family monotonicity $0/12 \to 6/12 \to 8/12$ under one factory and one protocol is the controlled result.

## The counterfactual: why not the delta rule?

A fair reader asks why bother with a Givens composer at all when the DeltaNet/DeltaProduct transition rule — reimplemented in this repository as a lab mixer — is available and, on `S3-hier`, trains even more reliably. Keeping the same signed extractor and swapping the Givens composer for a DeltaProduct-style two-reflection layer takes `S3-hier` fit from `GivensMinGRU`'s $8/12$ to $6/6$, with no seed lottery. The comparison is worth making precisely because it clarifies what the Givens choice is *for*.

Two facts settle it. First, the delta rule's reliability is largely a state-size effect, not a mechanism advantage. That $6/6$ configuration carries a per-token state of $n_{\text{heads}} \times d_k \times d_v = 1024$ elements — sixteen times the minGRU variants' $64$ — and shrinking the delta composer to the same $64$-element state drops its fit rate from $6/6$ to $4/12$ (Fisher $p \approx 0.013$) and reintroduces chance plateaus on the misses. At matched $64$-element state the Givens-versus-small-delta comparison is $8/12$ against $4/12$, which is only suggestive ($p \approx 0.22$) and parameter-unmatched ($14{,}624$ against $3{,}306$ composer parameters), so neither mechanism is established as better at matched state. What is established is that the full-size delta rule's headline reliability is bought with state, and among reliably-trainable configurations the Givens composer's fit-only length generalization actually leads the full-size delta composer's ($0.927 / 0.733$ against $0.815 / 0.530$ at $T=512/1024$).

Second, and decisively for the design: the delta-rule path this repository implements is a *sequential* rank-1 recurrence, and it is the cheaper option on CPU precisely because it is sequential. `GivensMinGRU` exists to keep a non-commutative composer inside the *parallel* associative scan. An efficient parallel delta form would require the chunked WY representation from the DeltaNet literature, which is not implemented here. So the honest statement is narrow and holds: where parallel associative-scan training is a hard requirement, `GivensMinGRU` is the reliably-trainable non-commutative composer at matched state; where it is not, the sequential delta rule is a strong and cheaper alternative. The full cross-mechanism table and capacity disclosure are in `experiments/TECHNICAL_REPORT.md` §4.4–§4.5.

## The trade it does not escape

Orthogonality buys norm preservation: a product of Givens rotations has determinant $+1$ and never shrinks or grows the state, so there is no amplitude decay across hundreds of compositions, unlike a diagonal scan whose eigenvalues in $(0,1)$ fade. But continuity means **no attractor.** Nothing pins a learned angle to the exact group element it approximates, so a small angle error compounds with sequence length exactly as an unsnapped `RotationMinGRU`'s does, and accuracy still decays by $T=1024$. The pooled decay from $0.949$ at $T=64$ to $0.613$ at $T=1024$ is this effect.

This is where the honest ceiling sits. Exactness at length remains unique to the *snapped* $2$D composer's rare exact seed, which reaches $0.983$ at $T=1024$ — the best length-generalization figure measured on `S3-hier` — and no continuous composer, Givens included, reaches it. `RotationMinGRU`'s straight-through angle snapping manufactures an attractor at exact group angles that `GivensMinGRU`'s continuous, exactly-orthogonal transition deliberately has none of. Snapping the Givens angles is the obvious hybrid and an open question, not a promise. The trade is therefore: `GivensMinGRU` fits far more reliably (8/12 versus a rare snapped winner) but tops out short of exact; the snapped $2$D composer almost never fits but is exact when it does.

## Time-aware decay on an orthogonal block

Every mixer optionally scales its transition by a per-event decay term $\gamma = \exp(-\lambda\, f(\Delta t))$, with $\lambda \geq 0$ per block and $f$ the identity or $\log(1+\cdot)$; see [Reference: mixers](../reference/mixers.md) for the keyword arguments and [How-to: choose a mixer](../how-to/choose-a-mixer.md) for when to enable it. For `GivensMinGRU` the semantics are identical to `RotationMinGRU`'s and worth stating because the interaction with orthogonality is exactly what makes it clean: the scalar $\gamma$ multiplies the whole block, $\gamma M_t$, and because a scalar commutes with the orthogonal block action, the composed rotation is recovered unchanged from the decayed matrix — only amplitude fades, direction is untouched. The injection $b_t$ is never decayed. Since $\lambda \geq 0$ and $\Delta t \geq 0$, $\gamma \in (0, 1]$, so decay can only shrink. The $\Delta t = 0 \Rightarrow \gamma = 1$ contract holds at every position including $t=0$, which is what keeps chunked-versus-full equivalence exact. Two modes exist: `"fixed"` (a scalar buffer, not learned) and `"learnable"` ($\lambda = \mathrm{softplus}(\rho)$, one $\rho$ per block, initialized so $\lambda$ equals `decay_rate` at construction). Note that `GivensMinGRU`'s class-default `decay_rate` is $1.0$ — a strong default that is directly relevant to the backward-pass decision below.

## The backward pass: why division-based reversal was rejected

The most consequential design decision in the whole Triton backend concerns how `GivensMinGRU`'s (and `RotationMinGRU`'s) angle-fused kernel computes gradients. The forward fast path — Kernel 4 in the [scan-kernel design](triton-scans.md) — carries the state vector in registers and applies the factored plane rotations directly from the angles, never materializing or scanning the $k \times k$ transition matrices. That is the source of its memory win. The question is the backward: to accumulate gradients of the angles, the scale channel, the injection, the decay, and $h_0$, the adjoint recurrence needs each intermediate state $h_{t-1}$, and the forward did not save them.

There are two ways to get $h_{t-1}$ back. The tempting one is *reversal*: since each forward step is $h_t = \gamma_t\, R(\theta_t)\, S_t\, h_{t-1} + b_t$ with $S_t$ the per-block scale, one can algebraically invert it — $h_{t-1} = S_t^{-1} R(\theta_t)^\top (h_t - b_t) / \gamma_t$ — and walk the states backward from a checkpoint stored every $C$ steps, reconstructing the interior of each chunk by division. This is attractive because it stores only every $C$-th state.

It is also unsound here, and the repository has the measurement to prove it. The per-block scale $\mathrm{diag}(1, \tanh u_t)$ is near-singular whenever $\tanh u_t \approx 0$, which is typical at initialization for the rotation variant, and dividing out a near-singular scale (or a strong decay $\gamma$) amplifies floating-point roundoff. A blind CPU emulation quantified the growth: **reversal error grows as $\sigma_{\min}^{-\text{chunklen}}$**, where $\sigma_{\min}$ is the smallest singular value of the per-step map. At a checkpoint interval $C = 64$, the measured gradient error against the minimum decay strength $\gamma_{\min}$ was:

| $\gamma_{\min}$ | $C=64$ gradient error |
|---|---|
| 0.86 | $1.8 \times 10^{-3}$ |
| 0.48 | $6.4 \times 10^{6}$ |
| 0.23 | $9.3 \times 10^{19}$ |

The pass/fail boundary sits near $\gamma \approx 0.87$ at $C = 64$; a decay of $\gamma = 0.8$ already fails by roughly $100\times$. Because `GivensMinGRU`'s class-default `decay_rate` is $1.0$, a user enabling decay would routinely land in the divergent regime, and under `MINGRU_SCAN=auto` the corruption would be silent — a wrong gradient, not a crash or a warning. That was deemed unacceptable.

The resolution (the user's ruling, recorded in the Task-5 report "Fix round 2" and the `min_gru.py` / `triton_scans.py` docstrings) rejected division-based reversal *entirely* for both rotation-family mixers, and removed the interval parameter so it cannot be re-enabled. The backward now uses an **exact $C = 1$ stored-state recompute**: every $h_{t-1}$ is read directly from the stored forward output, never reconstructed by inverting the forward step, so gradients are exact at any decay strength and any initialization. The elegance is that this costs *zero extra memory*: the forward output tensor is the module's return value, alive regardless, so it doubles as the anchor source with no extra allocation. The angle-fused path's memory win over the generic Phase-1 backward — never materializing the $k \times k$ transitions — is untouched by the change. A chunk-buffered forward recompute (numerically stable, no inversion) remains a benchmark-justified future option if the per-step state re-reads ever prove to cost material wall-clock, but it is not needed today.

The memory win is measured. At the lab shape $B=128$, $T=64$, $d=64$, $k=8$, backward peak memory for the angle-fused $C=1$ path is **38.25 MB** against **395.09 MB** for the generic Phase-1 backward that materializes the transitions — a $10.3\times$ reduction (`experiments/bench/scan_memory.md`). The full kernel-design context, the fp32-accumulation story, and the measured speedup table live in the [Triton scan kernels](triton-scans.md) article.

## Costs, and when to choose it

`GivensMinGRU`'s parallel $k \times k$ scan is not the cheap path on CPU. One uncontended forward-plus-backward step at $B=128$, $T=64$ (min of three runs) costs $0.961$s for the parallel-scan `givens8` transition against $0.179$s for a sequential delta-rule path, and no parallel-scan configuration measured here beats the sequential rank-1 delta implementation on CPU. Its promotion rests on three properties together, and the CPU step time is explicitly not one of them: it preserves parallel associative-scan training (a hard design constraint for some regimes), it fits `S3-hier` far more reliably than the $2$D composer at matched state, and within the reliably-trainable field its fit-only length generalization leads. On matched-state parameters it costs $14{,}624$ against the $2$D composer's $12{,}544$ (+2.2% at the full stack), so the fit-rate separation is not a capacity effect.

Choose `GivensMinGRU` when the problem is an extract-then-compose stack that must derive a non-commutative operation from raw input and then compose it — the `S3-hier` shape — *and* parallel training is a hard requirement. Use it as the composer above a signed extractor, `mixer=["signed", "givens"]`, `block_size=8`, `rounds=3`. Budget for the length decay: like every continuous composer here, it buys no attractor and still decays by $T=1024$. If tokens *are* the operations directly (the plain `S3` shape), a single `RotationMinGRU` layer under the best-val@128 protocol is the more direct tool. If the task only needs sign or parity dynamics, a diagonal `SignedMinGRU` is sufficient and cheaper. The full decision tree, with evidentiary basis and main risks per branch, is in `experiments/TECHNICAL_REPORT.md` §10 and [How-to: choose a mixer](../how-to/choose-a-mixer.md).

## See also

- [Reference: mixers](../reference/mixers.md) — the `GivensMinGRU` constructor, keyword arguments, and validation rules.
- [Reference: scan operations](../reference/scan-ops.md) — `matrix_affine_scan`, the $k \times k$ scan `GivensMinGRU` runs on.
- [Triton scan kernels](triton-scans.md) — the angle-fused kernel, the fp32/TF32 story, and the measured speedups and memory result.
- [Tutorials: two-layer stacks](../tutorials/two-layer-stacks.md) — building the signed-then-Givens extract-then-compose stack.
- [How-to: choose a mixer](../how-to/choose-a-mixer.md) and [How-to: reproduce the evidence](../how-to/reproduce-the-evidence.md) — picking a mixer and rerunning the `S3-hier` row.
- `experiments/TECHNICAL_REPORT.md` — the review-hardened report with the full multi-seed tables, the rounds ablation, and the delta-rule comparison.
