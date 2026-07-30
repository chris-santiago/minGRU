# minGRU State-Tracking Variants: A Technical Report

## Abstract

Four minimal-GRU (minGRU) variants are evaluated on length-generalization of state-tracking tasks, training at sequence length $T=64$ and measuring accuracy at $T \in \{256, 512, 1024\}$ (chance $0.5$ for parity, $\approx 0.167$ for the permutation tasks). The primary metric is length-generalized tagging accuracy. On parity, decoupling the transition eigenvalue from the update gate (`SignedMinGRU`, `coupled=False`) reaches $0.994$ mean accuracy at $T=1024$ (n=6, worst seed $0.984$), against $0.592$ for the coupled legacy parameterization and $0.502$ (chance) for the base log-space minGRU. On the $S3$ permutation-composition task, a non-diagonal $2\times2$ block-rotation transition with straight-through angle snapping (`RotationMinGRU`) lands the exact automaton on $1$ of $8$ seeds and averages $0.958$ at $T=1024$ (the other seven seeds decay, span $0.859$–$1.000$) where every diagonal variant is capped at $0.732$; in winning seeds, the most homomorphic of the model's $32$ transition blocks realizes a faithful $D_3$ representation to homomorphism error $\approx 1.4\times10^{-4}$ (one faithful block suffices to carry the automaton), certifying the learned automaton from weights. The snap attractor governs only the rotation angle; the reflection component relies on the soft $\tanh$ asymptote, so even winning seeds sit at $|\det| \approx 0.9999$ rather than exactly $1$. On a harder extract-then-compose task ($S3\text{-hier}$), an $8$-dimensional block-rotation composer (`GivensMinGRU`) built from Givens-rotation products fits $8$ of $12$ seeds against $1$ of $12$ for the $2$D rotation composer at matched $64$-element per-token state (Fisher exact $p \approx 0.009$); a rounds ablation at fixed block size attributes the gap to the coupling that breaks within-block commutativity, not to block size (an $8$D block of disjoint commuting planes fits $0$ of $12$). A standard GRU holds $1.000$ on every task and length and remains the ceiling, and a mechanism-level DeltaProduct reimplementation (nh=2) has a materially more reliable $S3$ training profile ($3/3$ seeds, no retry) than `RotationMinGRU`'s ($1/8$ exact plus retry), carrying $16\times$ the per-token state. The recommendation is therefore conditioned on a hard parallel-training constraint at matched per-token state: `SignedMinGRU` (`coupled=False`) for sign/parity dynamics and `RotationMinGRU` (with best-validation-at-$T{=}128$ checkpoint selection and retry-on-flag) or `GivensMinGRU` for non-commutative composition, all preserving parallel associative-scan training; no parallel variant matches the GRU's exactness at length, and every reported number is relative to a $1600$-step budget. A subsequent nine-arm benchmark round on four accepted public tasks (S5 word problems, MQAR, psMNIST, and an irregular-time pendulum control; a separate L4 GPU evidence stratum) validates the mechanism story externally and resolves it into a two-dial recommendation — delta as the broad workhorse, Givens as the small-state group-composition specialist (Section 8).

## 1. Background: the minGRU and its expressivity ceiling

The minGRU (Feng et al., *Were RNNs All We Needed?*, arXiv:2410.01201) removes the hidden-state dependency from the GRU's gates, which also eliminates the reset gate, and drops the $\tanh$ range restriction on the candidate state:

$$z_t = \sigma(W_z x_t), \qquad \tilde{h}_t = W_h x_t, \qquad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t.$$

Because $z_t$ and $\tilde{h}_t$ depend only on $x_t$, the recurrence is a first-order linear scan $h_t = a_t \odot h_{t-1} + b_t$ with $a_t = 1 - z_t$ and $b_t = z_t \odot \tilde{h}_t$, computable over the full sequence in parallel rather than by backpropagation through time. This repository implements the paper's log-space variant (Appendix B) via a Heinsen `logcumsumexp` scan, which applies $g(x) = x + 0.5$ for $x \geq 0$ and $\sigma(x)$ for $x < 0$ to the candidate and initial states, constraining hidden states to $(0, \infty)$.

Two structural constraints follow from the log-space parameterization and bound what a stack of these layers can compute. First, the transition coefficient $a_t = 1 - z_t$ is confined to $(0, 1)$: memory can only decay monotonically toward new input, an input-gated exponential moving average. Second, the transition is diagonal, hence commutative, so the running product is invariant to input reordering. Both constraints are expressivity limits, not optimization difficulties: no width or depth removes them within the log-space parameterization. Fixed-depth stacks of input-dependent, state-independent linear transitions sit in the complexity class $\mathrm{TC}^0$ (Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models*, ICML 2024), the same broad class as Mamba/S6 and GLA, and cannot perform the unbounded state tracking a single nonlinear GRU layer can. The variants studied here each lift exactly one constraint while preserving the parallel scan.

## 2. The variant ladder

Each variant answers one question: what is the per-step update permitted to do to the carried memory? The answers form a ladder in which each rung subsumes the one below and adds one capability.

**`MinGRU` (base, log-space).** Transition $a_t \in (0, 1)$; monotone decay only. Measured at chance on both tasks at every depth (Section 4.1). The parallel scan is `parallel_scan_log` over $\log a_t$, $\log b_t$.

**`SignedMinGRU`.** Switches to a linear-space associative scan (`linear_scan`, a Hillis-Steele doubling over the affine monoid $(A_1, B_1) \circ (A_2, B_2) = (A_2 A_1,\, A_2 B_1 + B_2)$ that admits negative coefficients), lifting the transition to $a_t \in (-1, 1)$ and hidden states to unconstrained reals. Two parameterizations:

$$a_t = \tanh(W_s x_t) \quad (\texttt{coupled=False, default}), \qquad a_t = (1 - z_t)\,\tanh(W_s x_t) \quad (\texttt{coupled=True, legacy}).$$

Both give $h_t = a_t \odot h_{t-1} + z_t \odot W_h x_t$. A negative eigenvalue supports alternation (state that flips sign and cancels rather than only fading), which is what parity requires. This instantiates the negative-eigenvalue mechanism of Grazzi et al. (*Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*, ICLR 2025) in the minGRU gate structure and is motivated by Merrill, Petty & Sabharwal (2024). Decoupling matters because the coupled form imposes a ceiling $|a_t| \leq 1 - z_t$: reaching the eigenvalue $-1$ that parity needs requires the gate to saturate ($z_t \to 0$) simultaneously, so one target value costs two saturations. The decoupled form's attractor is $\tanh$'s own asymptote at $-1$, reached with a single saturation. Decoupling does not restore non-commutativity: diagonal transitions still commute, so permutation composition remains out of reach at any width. `SignedMinGRU` carries three linear heads ($z$, $h$, $s$) against the base's two.

**`RotationMinGRU`.** State is $n = \texttt{hidden\_size}/2$ independent $2$D blocks; per block the transition is a full $2\times2$ affine map:

$$M_t = R(\theta_t)\, \mathrm{diag}(1, \tanh u_t), \qquad h_t = M_t\, h_{t-1} + b_t, \quad b_t = z_t \odot W_h x_t.$$

The scan is `matrix_scan`, a Hillis-Steele doubling over the $2\times2$ affine monoid with matrix composition $A_{\text{current}} @ A_{\text{earlier}}$, which does not commute. The per-block map $R(\theta_t)\,\mathrm{diag}(1, \tanh u_t)$ ranges over $O(2)$ (rotation and reflection) and generates the dihedral group, which is non-abelian. Since $D_3 \cong S_3$, the smallest non-abelian group, embeds in $O(2)$, one layer exactly represents the $S3$ running-product automaton, which no diagonal variant can at any width. Continuous rotation angles have no attractor at the group angles (a third of a turn for a $3$-cycle is a measure-zero interior point), so angle error compounds with length. Angle snapping (`snap`) quantizes $\theta_t$ per block to an exact multiple of $2\pi/K$ via a straight-through estimator: the forward pass uses the snapped angle, the gradient flows through the pre-snap soft angle. This manufactures an attractor at exact group elements, the rotation analogue of $\tanh$'s asymptote. The snap grid must contain the tracked group's rotation subgroup: the default $\texttt{snap} = (2, 3, 4, 6)$ was chosen for $D_3/S_3$ (its $3$-cycle is covered by the $3$ and $6$), and tracking a period-$n$ cycle needs a $K$ that is a multiple of $n$. `RotationMinGRU` carries four heads ($z$, $h$, $\theta$, $u$), owns its initial state $h_0$ as an intrinsic parameter (a zero state has no orbit under the group action), and requires even `hidden_size`.

**`GivensMinGRU`.** Occupies the same non-commutative rung as `RotationMinGRU` with a richer per-token map: each transition is a product of Givens rotations acting on a `block_size`-dimensional block, generalizing the single $2$D plane to $k$-dimensional block rotations at the same per-token state. At the default $\texttt{block\_size}=8$, $\texttt{rounds}=3$, each transition is three brick-wall layers of Givens rotations (round 0 pairs planes $(0,1),(2,3),(4,5),(6,7)$; round 1 the staggered planes $(1,2),(3,4),(5,6),(7,0)$; round 2 repeats round 0), all angles emitted by one linear head. Disjoint planes within a round commute; the stagger across rounds couples all eight dimensions and breaks commutativity. The result is exactly special-orthogonal at every parameter value (no normalization step, no snap), continuous, and runs on `matrix_affine_scan`, the $k\times k$ generalization of `matrix_scan`. The brick-wall mesh is the standard EUNN construction (Jing et al., ICML 2017) built on the interferometer mesh of Clements et al. (2016), with the Givens rotation as in Golub & Van Loan; what is specific here is the input-dependent per-token angles and the measured trainability result, not the factorization. `GivensMinGRU` carries three heads ($\theta$, $z$, $h$), owns its $h_0$, and requires `hidden_size` divisible by an even `block_size`.

The per-layer capability ladder:

| variant | scan | transition | per-layer capability |
|---|---|---|---|
| `MinGRU` | log-space | $a \in (0,1)$ diagonal | monotone EWMA memory |
| `SignedMinGRU` | linear-space | $a \in (-1,1)$ diagonal | + parity, abelian tracking |
| `RotationMinGRU` | matrix ($2\times2$) | $O(2)$ block | + non-commutative: $S3/D_3$ exactly representable |
| `GivensMinGRU` | matrix ($k\times k$) | $SO(k)$ block | non-commutative, richer per-token map |
| standard GRU | none (sequential) | state-dependent gate | unbounded state tracking |

All four minGRU variants optionally accept a time-aware exponential decay term $\gamma = \exp(-\lambda\, f(\Delta t))$, evaluated separately in Section 7. Numerical agreement between the parallel and sequential paths at $T=128$, fp32, is $\approx 10^{-5}$ (`MinGRU`), $\approx 10^{-7}$ (`SignedMinGRU`), $\approx 10^{-6}$ (`RotationMinGRU`), and $\approx 10^{-5}$ (`GivensMinGRU`).

## 3. Methods

### 3.1 Tasks

Four probe tasks isolate the ladder's boundaries, all posed as seq2seq tagging with dense per-position supervision.

**parity.** Label each prefix of a bit string with its running XOR (chance $0.5$). The natural solution is a state that flips sign on a $1$ and holds on a $0$: transition eigenvalue $-1$. This is $\mathbb{Z}_2$, abelian; it needs a negative eigenvalue, not non-commutativity.

**$S3$.** Label each prefix of a sequence of permutations of three objects with their net composition (chance $\approx 0.167$). Composition order matters, so the task requires a non-commuting transition. Tokens are the group operations directly.

**$S3\text{-hier}$.** The harder extract-then-compose task (chance $\approx 0.167$): the group operation is hidden inside a pair of sub-tokens drawn uniformly from $\{0,\dots,5\}$; each consecutive pair $(x_{2k}, x_{2k+1})$ selects a generator $g = \mathrm{LATIN}[a, b]$ composed onto the running $S3$ product when the pair completes. Labels are dense (odd positions carry the just-updated composition, even positions the previous one). `LATIN` is a fixed $6\times6$ Latin square verified non-isotopic to both groups of order six ($\mathbb{Z}_6$ and $S_3$): no relabeling turns it into either group's Cayley table, so a single sub-token carries no information about the generator and no rotation layer can absorb the lookup by relabeling its angle assignment. Extraction is therefore genuine work that must precede composition.

**session-parity / parity-timestamped.** Running-XOR variants that supply per-event time gaps $\Delta t$, used for the time-decay evaluation (Section 7).

### 3.2 Protocol

Seq2seq tagging with dense supervision, $T_{\text{train}}=64$, $d_{\text{model}}=64$, batch $128$, Adam at learning rate $3\times10^{-3}$, budget $\leq 1600$ steps, torch 2.5.1, CPU. Evaluation lengths are $64$ (in-distribution) and $256/512/1024$ (length generalization). Two selection procedures are used and stated per result:

- **early-stop:** halt at $99.9\%$ train-length accuracy; used for parity and for GRU rows.
- **best-val@128:** replace early-stop with best-checkpoint selection by validation accuracy at $T=128$ over the full budget. $T=128$ is longer than the training length but is not one of the eval lengths, so it cannot leak into reported metrics. This procedure is required for the rotation family because the exact automaton is reachable but is not a stable attractor of training (Section 5.2). A best-val@128 score below $1.0$ flags a run for retry.

The primary metric is length-generalized tagging accuracy at $256/512/1024$. In-distribution accuracy at $64$ separates "expresses the recurrent solution" from "learned a depth-bounded shortcut for the training length" only in conjunction with the length columns; length generalization is what distinguishes the exact automaton from a decaying approximation. Every $S3$ minGRU-variant row in Section 4 uses best-val@128 so that the diagonal and rotation rows compare mechanisms under one selection procedure rather than confounding mechanism with protocol.

### 3.3 Comparison structure

Every comparison is internal: against the repository's floor (base `MinGRU`), its prior parameterization (`coupled=True`), its ceiling (`torch.nn.GRU`), and mechanism-level reimplementations of the DeltaNet (Yang et al., 2024) / DeltaProduct (Siems et al., NeurIPS 2025) transition rules (`DeltaNetMixer`, $\texttt{nh} \in \{1, 2\}$). The delta-rule comparison is a transition-rule reimplementation in plain torch under this repository's protocol, not the incumbents' released Triton/CUDA systems or published numbers; it establishes "reproduces the predicted behavior, minGRU-natively," not "competitive with the incumbents." The minGRU wrappers include a block MLP that the bare GRU baseline lacks, which favors the minGRU variants and therefore strengthens their negative results (chance despite extra capacity) while mildly weakening attribution of positive ones.

## 4. Results

### 4.1 Base minGRU is at chance, independent of depth

The log-space `MinGRU` is measured at chance on both tasks at $L=1$ and $L=4$ (n=3, full $1600$-step budget consumed, round `base-mingru`):

| task | $L$ | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|
| parity | 1 | 0.542 | 0.511 | 0.506 | 0.502 |
| parity | 4 | 0.516 | 0.504 | 0.503 | 0.501 |
| $S3$ | 1 | 0.229 | 0.182 | 0.175 | 0.171 |
| $S3$ | 4 | 0.222 | 0.181 | 0.174 | 0.170 |

Depth does not move the result. The base minGRU cannot represent a $-1$ transition or a non-commuting one at any width, so this is a parameterization failure, not a training one. This bounds the ladder from below: it is the floor every variant is measured against.

### 4.2 Research question 1 — does a negative eigenvalue solve parity at length?

Decoupling the eigenvalue from the update gate resolves parity length-generalization. Primary-metric result (acc@1024), single layer, early-stop:

| model | seeds | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|
| `GRU` (ceiling) | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| `minGRU` (base) | 3 | 0.542 | 0.511 | 0.506 | 0.502 |
| `SignedMinGRU`, `coupled=True` (legacy) | 3 | 1.000 | 0.866 | 0.687 | 0.592 |
| `SignedMinGRU`, `coupled=False` (default) | 6 | 1.000 | 1.000 | 0.999 | 0.994 |

The decoupled form holds $0.994$ at $16\times$ the training length (worst of six seeds $0.984$), against $0.592$ for the coupled legacy form. The separation is length generalization, not training fit: both parameterizations reach $1.000$ at $T=64$. The mechanism is confirmed from weights (Section 5.1): probed decoupled models carry three channels at $(a_{\text{hold}}, a_{\text{flip}}) = (+0.99999, -0.99999)$, saturated deeply enough that $a^{1024} \approx 0.99$. The single knob separating the coupled and decoupled forms is the $|a_t| \leq 1 - z_t$ ceiling, which makes every strong eigenvalue cost a second saturation.

### 4.3 Research question 2 — does a non-commutative transition solve permutation composition at length?

On $S3$, all diagonal variants decay with length while the snapped rotation transition holds near the exact automaton. Every minGRU-variant row uses best-val@128 selection:

| model | $L$ | seeds | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|---|
| `GRU` (ceiling) | 1 | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| `minGRU` (base) | 1 | 3 | 0.229 | 0.182 | 0.175 | 0.171 |
| `SignedMinGRU`, `coupled=True` | 1 | 3 | 0.470 | 0.344 | 0.277 | 0.226 |
| `SignedMinGRU`, `coupled=True` | 4 | 3 | 0.994 | 0.713 | 0.506 | 0.368 |
| `SignedMinGRU`, `coupled=False` | 1 | 3 | 0.999 | 0.943 | 0.864 | 0.732 |
| `SignedMinGRU`, `coupled=False` | 4 | 3 | 0.997 | 0.908 | 0.769 | 0.574 |
| `RotationMinGRU` (`rotsnap`) | 1 | 8 | 1.000 | 1.000 | 0.996 | 0.958 |

The decoupled diagonal fits $S3$ in-distribution ($0.999$ at $T=64$) through a bounded-depth shortcut and decays steadily ($0.732$ at $T=1024$ at $L=1$; depth does not rescue it, $0.574$ at $L=4$). The coupled legacy form fails to fit $S3$ even at $L=1$ under checkpoint selection ($0.470$ at $T=64$), and the checkpoint protocol does not rescue it, so its failure is the parameterization, not the protocol. The rotation row's mean is a lottery, not a per-seed capability: exactly $1$ of $8$ seeds lands the exact automaton ($1.000$ at every length) while the other seven decay (span $0.859$–$1.000$ at $T=1024$), so the $0.958$ mean is dominated by near-miss seeds and must be read together with the $1/8$ exact rate (Section 5.2). Even so, the mean sits above the best diagonal's $0.732$ under the identical selection procedure: the win attributes to the mechanism, not to checkpoint selection. The separation lives at length generalization, exactly where the representability argument predicts it: diagonal transitions commute and yield the same final state under any input reordering, while $S3$ labels change under reordering.

The single rotation block survives being stacked with signed blocks. At depth $2$ on $S3$ (best-val@128, n=3):

| depth-2 stack | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|
| rotation $\to$ signed | 1.000 | 1.000 | 0.999 | 0.966 |
| rotation $\times 2$ | 1.000 | 0.999 | 0.983 | 0.941 |
| signed $\to$ rotation | 0.999 | 0.976 | 0.940 | 0.863 |
| $L=1$ reference (`rotsnap`, n=8) | 1.000 | 1.000 | 0.996 | 0.958 |

All three depth-2 configurations land inside the $L=1$ reference's demonstrated seed-noise band ($0.859$–$1.000$ at $T=1024$ across its eight seeds), so the table reads as "depth-2 stacks train and stay in the $L=1$ range," not as a ranking. Deeper homogeneous rotation stacks ($L=4$) are untested and remain open; angle snapping is validated as an $L=1$ mechanism because straight-through estimator noise through stacked quantized layers breaks training ($L=4$ homogeneous snap fails to train).

### 4.4 Research question 3 — does map richness improve composer trainability on the hierarchical task?

$S3\text{-hier}$ requires a stack that extracts the generator from a sub-token pair and then composes it non-commutatively. No single capability suffices in budget: a lone rotation layer, rotation $\times 2$, rotation $\to$ signed, a lone delta-rule layer, and an unconstrained GRU all sit at or near chance (even the GRU is at $0.171$ at $T=1024$, so the section measures which inductive bias trains reliably under the $1600$-step budget, not what is representable). The result is a reliability finding, not an exactness finding: composers differ in how often training finds a working solution, not in how well found solutions generalize. A seed counts as a **fit** when its selected checkpoint's validation accuracy at $T=128$ is $\geq 0.99$. Multi-seed means (chance $\approx 0.167$):

| config | seeds | fits | acc@64 | acc@256 | acc@512 | acc@1024 | fit-only @512/@1024 |
|---|---|---|---|---|---|---|---|
| signed $\to$ givens8 (`hetero-sg8`) | 12 | 8/12 | 0.949 | 0.885 | 0.787 | 0.613 | 0.927 / 0.733 |
| signed $\to$ deltaproduct2 (`hetero-sd2`) | 6 | 6/6 | 1.000 | 0.987 | 0.814 | 0.530 | — |
| signed $\to$ 64-state delta (`hetero-sdm`) | 12 | 4/12 | 0.575 | 0.495 | 0.457 | 0.376 | 0.942 / 0.739 |
| signed-tanh, $L=2$ (homogeneous) | 3 | — | 0.985 | 0.866 | 0.754 | 0.620 | — |
| signed $\to$ rotation (`hetero-sr`) | 6 | 1/6 | 0.572 | 0.442 | 0.375 | 0.309 | — |
| signed $\to$ rotation, continuous (no snap) | 12 | 1/12 | 0.515 | 0.404 | 0.352 | 0.293 | 0.902 / 0.669 |
| rotation $\to$ signed | 3 | — | 0.448 | 0.325 | 0.247 | 0.206 | — |
| rotation $\times 2$ | 3 | — | 0.365 | 0.256 | 0.212 | 0.189 | — |
| `deltaproduct2`, $L=1$ | 6 | 0/6 | 0.238 | 0.184 | 0.176 | 0.171 | — |
| `GRU`, $L=1$ | 3 | — | 0.232 | 0.184 | 0.175 | 0.171 | — |
| signed $\to$ rotation, $4\times$ budget | 3 | 1/3 | 0.890 | 0.658 | 0.570 | 0.502 | — |

Three findings, each with a controlled attribution:

1. **Depth buys hierarchy, and it is causal.** `deltaproduct2` alone at $L=1$ is at chance on all six seeds, while the same delta layer above a signed extractor fits $6/6$; the depth split (extraction below, composition above) is doing the work, not the composer solving $S3\text{-hier}$ outright.
2. **Within the rotation family, the richer $8$D Givens map fits far more reliably.** At matched $64$-element per-token state, $8$D Givens blocks fit $8/12$ against $1/12$ for continuous $2$D rotation (Fisher exact $p \approx 0.0094$), at $+2.2\%$ full-stack parameters, so the gradient is not a parameter effect. The ordering is threshold-robust: at fit thresholds $\{0.98, 0.99, 0.995\}$ the counts read $9/12$, $8/12$, $8/12$ against $1/12$ at all three (Fisher $p = 0.0028$, $0.0094$, $0.0094$). Snapping is not the trainability lever: continuous $2$D rotation fits $1/12$, the same rate as the snapped $2$D composer, whose nominal $1/6$ rests on one borderline seed at val@128 $= 0.9879$ (a near-fit that reads $0/6$ under the strict threshold; either way it is indistinguishable from the continuous row). On its own this comparison moves two things at once, per-token map expressivity ($SO(8)$ vs $SO(2)$) and within-scan state connectivity ($8$ coupled channels per block vs $2$); the rounds ablation below separates them and attributes the effect to the coupling that breaks within-block commutativity, not to block size.
3. **Reliability is separable from generalization quality.** Seeds that fit generalize equally under either mechanism (fit-only means $0.733$ Givens vs $0.739$ small-delta at $T=1024$); the mechanisms differ in fit rate. Among reliably trainable configurations, the Givens-$8$ composer's fit-only profile ($0.927$ at $T=512$, $0.733$ at $T=1024$; best seed $0.956/0.812$) leads the full-size delta composer's ($0.815/0.530$ across its $6/6$ fits).

**Rounds ablation at fixed block size.** Varying $\texttt{rounds} \in \{1, 2, 3\}$ at $\texttt{block\_size}=8$ under the identical stack and protocol (round `hetero-loop-19-rounds`, n=12 per arm; $\texttt{rounds}=3$ is the recorded arm above) isolates the two candidate causes. At $\texttt{rounds}=1$ the brick-wall mesh is $4$ disjoint, commuting $2$D planes inside the $8$D block, $32$ planes per token, plane-count-matched to the $2$D composer's $32$ blocks but with the larger coupled state block; $\texttt{rounds}=2$ adds the single staggered layer that couples all eight dimensions and breaks commutativity.

| rounds | per-token map | fits | acc@64 | acc@1024 | fit-only @512/@1024 |
|---|---|---|---|---|---|
| 1 | $4$ disjoint commuting planes | $0/12$ | $0.376$ | $0.182$ | — |
| 2 | + one staggered coupling layer | $6/12$ | $0.805$ | $0.515$ | $0.916$ / $0.704$ |
| 3 | + repeat (recorded arm) | $8/12$ | $0.949$ | $0.613$ | $0.927$ / $0.733$ |

The block-size hypothesis is refuted: the $8$D block of commuting planes fits $0/12$, no better than the $2$D composer's $1/12$ (Fisher $p = 1.0$) despite the bigger coupled block. The single commutativity-breaking layer recovers most of the effect ($6/12$; $p \approx 0.014$ vs $\texttt{rounds}=1$), the third layer adds a statistically inseparable increment ($8/12$; $p \approx 0.68$ vs $\texttt{rounds}=2$), and the endpoint separation is $p \approx 0.0013$ ($0/12$ vs $8/12$). The ordering is threshold-stable ($0/12$, $8/12$, $9/12$ at fit threshold $0.98$). Fit-only generalization is unchanged across rounds ($0.704$ vs $0.733$ at $T=1024$), reproducing the reliability-versus-quality separation. One residual caveat applies to the cross-family anchor only: $\texttt{rounds}=1$ is a pure rotation with no $\tanh u$ scale channel while the $2$D composer carries one, so their equivalence is suggestive; the within-family monotonicity ($0/12 \to 6/12 \to 8/12$ under one factory and one protocol) is the controlled result.

One head-to-head deserves stating rather than leaving in the table: on the full-mean length-generalization column the promoted Givens composer ($0.613$ at $T=1024$) is within seed noise of the simpler homogeneous signed-tanh $L=2$ stack ($0.620$, measured under a different selection protocol), so the Givens promotion rests on the fit rate and the non-commutativity argument, not on the full-mean length-generalization number. The delta rule's own reliability is largely a state-size effect: shrinking it from $1024$ to $64$ state elements per token drops its fit rate from $6/6$ to $4/12$ (Fisher $p \approx 0.0128$) and reintroduces chance plateaus. The cross-mechanism comparison at matched $64$-element state (Givens $8/12$ vs small-delta $4/12$) is $p \approx 0.22$ and parameter-unmatched ($14{,}624$ vs $3{,}306$ composer parameters), so it is suggestive only. Exactness at length remains unique to the snapped $2$D composer's rare winner: at $4\times$ budget, one of three seeds finds the exact solution and reaches $0.983$ at $T=1024$, the best length-generalization figure measured on $S3\text{-hier}$, while best-val@128 correctly flags the other two ($0.573$, $0.763$) as runs to retry.

### 4.5 Mechanism-level incumbent comparison

The DeltaNet/DeltaProduct transition rule under this repository's protocol (n=3; parity early-stop, $S3$ best-val@128):

| task | model | acc@64 | acc@256 | acc@512 | acc@1024 |
|---|---|---|---|---|---|
| parity | `deltanet` (nh=1) | 1.000 | 0.9998 | 0.9342 | 0.8514 |
| parity | `deltaproduct2` (nh=2) | 1.000 | 1.000 | 0.9521 | 0.8104 |
| $S3$ | `deltanet` (nh=1) | 0.419 | 0.356 | 0.345 | 0.339 |
| $S3$ | `deltaproduct2` (nh=2) | 1.000 | 1.000 | 0.9998 | 0.9890 |

A single Householder reflection (nh=1) has determinant $-1$ on every application and cannot compose to the even-permutation elements $S3$ needs, matching DeltaProduct's own representability theory; its $S3$ mean of $0.419$ sits in the reflection-only chance band. Two reflections (nh=2) compose to a rotation and fit $S3$ near-exactly on every seed, reaching best-val@128 $= 1.000$ by step $200$–$300$ against the $1600$-step budget with no retry protocol. This is a materially better $S3$ training profile than `RotationMinGRU`'s, whose $1/8$ exact-seed rate needs the retry-on-flag protocol. On parity, both incumbents are length-inconsistent across seeds (`deltanet` mean $0.851$ at $T=1024$, per-seed range $0.730$–$1.000$; `deltaproduct2` mean $0.810$, range $0.692$–$1.000$), below `SignedMinGRU`'s $0.994$: the $\tanh$ asymptote sits at the exact eigenvalue parity needs, an attractor the delta rule's $\beta/k$ parameterization does not share under this protocol.

The comparison is same-$d_{\text{model}}$, not same-capacity. At $d_{\text{model}}=64$:

| module | parameters | state/token |
|---|---|---|
| `DeltaNetMixer` nh=1 (4 heads) | 16,900 | 1024 |
| `DeltaNetMixer` nh=2 (4 heads) | 25,480 | 1024 |
| `RotationMinGRU` (`rotsnap`) | 12,544 | 64 |
| `SignedMinGRU` (default) | 12,480 | 64 |
| `GivensMinGRU` (givens8 composer) | 14,624 | 64 |

Both delta configurations carry $16\times$ more per-token state ($n_{\text{heads}} \times d_k \times d_v = 1024$ against $64$) than the minGRU variants, which favors their fit and length-generalization numbers. All results in this section are budget-relative and make no claim about the incumbents' released systems.

## 5. Mechanism analysis

### 5.1 The learned automaton is certifiable from weights

Per-token transitions are input-only (embedding $\to$ LayerNorm $\to$ mixer, no state dependence), so the learned automaton can be read off the weights and checked exhaustively: the homomorphism test extracts each block's per-token $2\times2$ transitions $M(g)$ from a trained rotation-snap model and verifies $M(g)M(h) \approx M(g \circ h)$ over all $36$ pairs against the $S3$ composition table. The reported error is per-block, the minimum over the model's $32$ blocks of the worst-pair Frobenius error: the certificate states that the best block realizes a faithful $D_3$ representation, not that all transitions do, and one faithful block suffices for the readout to carry the automaton.

| seed | acc@1024 | best-block hom error | faithful | $|\det|$ |
|---|---|---|---|---|
| s0 | 1.000 | 0.00014 (3 blocks $< 10^{-3}$) | yes | 0.99992 |
| s1 | 1.000 | 0.00041 | yes | 0.99976 |
| s2 | 0.726 | 0.00205 | yes | 0.99870 |
| s6 | 0.616 | 0.00072 | yes | 0.99967 |

Winning seeds learned genuine faithful $D_3$ representations composing to $\approx 10^{-4}$: the headline is demonstrated, not inferred. This model class is unusually auditable, since the state-tracking core is certified from weights with no long-sequence evaluation. Two refinements follow. Failed seeds contain the same representation $5$–$15\times$ less exact, so failure is insufficient calibration, not a missing mechanism; and the homomorphism error roughly predicts the decay horizon via $\varepsilon \cdot T$ (s2 at $\varepsilon = 2\times10^{-3}$ breaks around $T=500$–$1000$, as observed). The certificate is a diagnostic and horizon predictor, not a checkpoint selector (Section 6). Two anomalies bound the account: automaton-block injection gates are not near zero (s1 runs its automaton block at $z = 0.70$ with perfect accuracy), and s6's best block is more exact than s2's despite worse accuracy, implicating downstream readout structure. The homomorphism probe measured $z$, not $\lVert b \rVert$, which the injection probes below address.

### 5.2 The exact solution is reachable but not a training attractor

Standard training wanders in and out of the exact solution: once train accuracy saturates at $T=64$, the dense cross-entropy loss carries no signal distinguishing the exact group solution from a decaying shortcut. Under a full-budget re-run without early stopping (`experiments/SUMMARY.md` round 6), one $S3$ seed that was exact at step $300$ decayed to $0.753$ at $T=1024$, another improved from $0.636$ to $1.000$, and a third collapsed to $0.433$ at $T=64$. Best-val@128 checkpoint selection recovers the exact solution wherever a run contained one (round 7). The basin of the exact solution is measurably small: retraining for $1600$ steps from a winning $S3\text{-hier}$ checkpoint perturbed by per-tensor Gaussian noise recovers near-exact accuracy on $2$ of $3$ seeds at $1\%$ noise but returns to the exact automaton on none, at any perturbation scale tested (ledger round `hetero-loop-14-basin`, $15$ rows). This upgrades "reachable but not a stable attractor" from an observation to a measurement and explains why exactness on the rotation family is a per-seed lottery rather than a reliable outcome.

### 5.3 The two exact seeds implement different solutions

Injection probes on the two perfect $S3$ seeds reveal distinct mechanisms. Seed s0 is an orbit automaton: ablating its automaton-block injections costs nothing ($0.9998$ at $T=1024$), and its automaton-block $\lVert b \rVert$ ($0.199$) sits at the median. Seed s1 is injection-driven: its automaton block's $\lVert b \rVert$ is $2.35$ ($10\times$ the median), and ablating it collapses the model to $0.64$ even at the training length. With exact group transitions, the injection sum is a group-convolution feature, so injections through exact transitions are signal, not noise. This refines the earlier "keep a forgetting path" law: only unstructured, non-automaton channels require contraction; exactness should be local, not global. The refinement was confirmed at inference by projecting all blocks to exact $O(2)$, which degraded even the perfect seed s0 from $1.000$ to $0.673$ at $T=1024$.

## 6. Repair rounds

Four cheap interventions to convert the failed rotation-snap seeds were pre-registered and tested; none earned a place in the shipping recipe.

- **Inference-time $O(2)$ projection** (snap $\tanh u \to \pm1$ on identified near-homomorphic blocks, zero training; `experiments/repair_results.json`): partial. s2 improved $0.726 \to 0.807$ at $T=1024$, s6 $0.616 \to 0.645$; the control seed s0 stayed at $1.000$. Projecting all blocks degraded s0 to $0.673$, reproducing the exactness-must-be-local law at inference with zero training. Most of the residual failure is readout-side, not transition calibration.
- **Projection plus 400-step fine-tune** (`experiments/repair2_results.json`, `s2_ft`/`s6_ft`): trades in-distribution accuracy for length robustness rather than repairing (s6 reaches $0.807$ at $T=1024$, $+19$ points, but $T=64$ drops from $0.99$ to $0.93$). Failed seeds are globally mis-oriented, not under-polished.
- **Homomorphism-error checkpoint selection** (`experiments/repair2_results.json`, `s2_select`/`s6_select`): exact null. Both criteria select the identical checkpoint on both failed seeds. Standard best-validation practice is sufficient; the homomorphism certificate remains a diagnostic and horizon predictor, not a selector.
- **Grid-attraction penalty and sharper validation** (on $S3$ seed 2, which never contains the exact solution; `experiments/SUMMARY.md` round 8): both refuted. A soft penalty $\text{reg} \cdot \lVert \theta_{\text{soft}} - \theta_{\text{snapped}} \rVert^2$ over-constrains the search and hurts even the good seed; sharper validation selects the same checkpoint. The failing seed's pathology is init/trajectory, not fixable by selection or loss shaping as tried.

The closed shipping recipe for the rotation family is: the variant, best-val@128 selection, and retry when best val $< 1.0$.

## 7. Time-aware decay

Each variant optionally scales its transition coefficient by $\gamma = \exp(-\lambda\, f(\Delta t))$ with $\lambda \geq 0$ per channel (per block for the rotation variants) and $f$ the identity or $\log(1+\cdot)$. The scalar $\gamma$ multiplies the transition only; the injection $b_t$ is never decayed, so for the rotation variants $\gamma$ commutes with the orthogonal block action and the composed rotation is recovered unchanged from the decayed matrix, only amplitude fading. Since $\lambda \geq 0$ and $\Delta t \geq 0$, $\gamma \in (0, 1]$, so decay can only shrink, never amplify. The $\Delta t = 0 \Rightarrow \gamma = 1$ contract holds at every position including $t=0$, which is what keeps chunked-versus-full equivalence exact.

A channel ablation on `session-parity` (running XOR resetting at session boundaries, gaps well above threshold at boundaries and well below within a session) isolates the decay mechanism from the raw time gap supplied as an input feature, under a fairness rule where every model receives the feature and only decay-enabled rows additionally receive the mechanism (n=3, $\texttt{decay="learnable"}$ at init $0.05$, $\log(1+\Delta t)$):

| channel | acc@64 | acc@256 | steps to early-stop |
|---|---|---|---|
| both (feature + mechanism) | 1.000 | 0.998 | 233 |
| feature only | 0.998 | 0.993 | 600 |
| mechanism only | 0.971 | 0.947 | never (full budget) |

The feature channel carries most of the signal: mechanism-only loses on both accuracy axes and never early-stops even at $4\times$ budget (a structural deficit, not undertraining). Both channels together beat feature-only modestly ($0.997$ vs $0.990$ acc@256 per matched seed, $2.6\times$ faster convergence), with the disclosed confound that the both-channels row carries extra per-channel rate parameters. The channels are complementary but asymmetric: the feature carries the win, the mechanism adds a smaller separately-measurable margin.

A recovery check on plain parity supplied with the same $\Delta t$ distribution (a task that never needs a reset) shows the learned rate stays near its low init ($\lambda$ mean $0.0499$, not drifting toward heavier decay) but accuracy does not fully recover ($0.786$ vs $1.000$ at $T=256$): $\log(1+\Delta t)$ of a boundary-scale gap is already $\approx 4.3$, so even at $\lambda \approx 0.05$ the per-boundary $\gamma \approx 0.81$ compounds over the dozen boundary-scale gaps in a $256$-step sequence. An init sweep makes the tradeoff explicit (recovery acc@256 rises to $0.997$ at init $0.005$ but convergence slows to $633$ steps, versus $0.786$ and $233$ steps at $0.05$), and the learned rate moves up but never down: over-decay's cost materializes only beyond $T_{\text{train}}$, so the training objective contains no signal against it. `decay_rate` (or its learnable init) is therefore a genuine hyperparameter set by validation at the target sequence length, not a knob training anneals to zero on its own.

## 8. Benchmark validation on public tasks

The mechanism story of Sections 4–5 was built on this repository's own probe tasks. A subsequent benchmark round re-tests it externally: nine mixer arms on four accepted public tasks, each judged against a fit bar frozen before any seed matrix ran. The framing is validation, not a leaderboard — the question is whether each task's structure rewards the mechanism the ladder predicts, and the result is a two-dial mechanism story, not a single winner.

### 8.1 Setup and stratum

The nine arms are the six packaged single-stack mixers (`log`, `signed`, `rotation`, `signed-rotation`, `givens`, `delta`), the two promoted heterogeneous stacks (`signed-givens`, `signed-delta`), and a depth-matched classical `gru` control (2-layer `torch.nn.GRU`, $d_{\text{model}}=64$) — the control anchors every comparison but is never read as a matched competitor. The four tasks and their pre-frozen fit bars: **S5** (symmetric-group word problems, `val128` $\geq 0.99$, n=36), **MQAR** (multi-query associative recall, `val_qacc` $\geq 0.99$, n=36), **psMNIST** (permuted-pixel MNIST, `val_acc` $\geq 0.90$, n=12), and an irregular-time **pendulum** regression as positive control (`val_mse` $\leq 0.0014$, n=36). A *fit* counts seeds whose selected checkpoint clears the task's own validation bar; generalization is always reported separately, raw (pooled over all seeds) alongside fit-only (conditioned on fitting seeds), so a low pooled mean is never silently read as "less capable."

One stratum disclosure governs everything in this section: these numbers are the L4 GPU stratum (`device=cuda`, torch 2.8.0+cu128, triton 3.4.0, `MINGRU_SCAN=triton`) — a different evidence stratum from the pinned-CPU (torch 2.5.1) rounds every other section of this report is built on. The two strata are never mixed or compared, per the repository's evidence discipline. Fisher-exact contrasts are two-sided against the `log` reference arm.

### 8.2 The master fit matrix

The four tasks dissociate the mechanisms cleanly: pendulum fits for everyone (the harness trains end-to-end), MQAR fits only for the delta family, psMNIST is led by the delta family, and S5 is solved at the matched configuration by exactly one arm.

| arm | S5 (n=36) | MQAR (n=36) | psMNIST (n=12) | pendulum (n=36) |
|---|---|---|---|---|
| `log` | 0/36 | 0/36 | 0/12 | 36/36 |
| `signed` | 0/36 | 0/36 | 0/12 | 36/36 |
| `rotation` | 0/36 | 0/36 | 0/12 | 36/36 |
| `signed-rotation` | 0/36 | 0/36 | 0/12 | 36/36 |
| `givens` | 0/36 | 0/36 | 0/12 | 36/36 |
| `delta` | 0/36 | 36/36 | 10/12 | 36/36 |
| `signed-givens` | 1/36 | 0/36 | 0/12 | 36/36 |
| `signed-delta` | 0/36 | 36/36 | 12/12 | 36/36 |
| `gru` (control) | 0/36 | 0/36 | 3/12 | 36/36 |

### 8.3 Per-task readings

**MQAR — a pure delta dissociation.** Only the delta family fits: `delta` and `signed-delta` at 36/36 each, every other arm — including the `gru` control — at 0/36 (Fisher vs `log`: $p = 2.32\times10^{-13}$). Query accuracy degrades with recall load as the delta family's fixed state saturates: `delta` $0.931 \to 0.493$ raw q-acc from 16 to 32 key-value pairs, `signed-delta` $0.928 \to 0.690$. This matches the recall-capacity tradeoff of Arora et al. (*Zoology*, arXiv:2312.04927): associative recall is the delta rule's lane, and state size is the binding constraint.

**psMNIST — accumulation ordering.** The delta family leads on fit rate: `signed-delta` 12/12 (raw test $0.924$, the only threshold-stable fitting arm; Fisher $p = 7.40\times10^{-7}$), `delta` 10/12 ($0.905$, $p = 6.73\times10^{-5}$), the `gru` control 3/12 ($0.885$, just under the bar), every rotation-family arm 0/12 (`givens` weakest at $0.290$ — stacked pure rotation is the wrong prior for accumulation). A hidden-256 `gru-large` reference (60-epoch budget, non-matched, no Fisher) reaches $0.922$ at 12/12 — inside the literature vanilla-GRU band of $\approx 92$–$94\%$ — confirming the matched control's near-miss is capacity, not a code-path bug, and that `signed-delta` matches the literature-scale GRU at roughly a fifth of the parameters ($105{,}554$ vs $596{,}234$). A block-order ablation on the rotation composer (same arm, task, and protocol; only the composer block order differs) shows the corrected extract-then-compose order costs $0.868 \to 0.736$ raw test accuracy: block order is task-dependent, helping group composition and hurting accumulation.

**S5 — group composition.** Exactly one matched arm fits: `signed-givens`, 1/36, whose single fitting seed generalizes cleanly (fit-only acc $1.000 / 0.976 / 0.817$ at $T = 256/512/1024$). A single-seed margin does not separate statistically at n=36 (Fisher $p = 1$), so a design-correction probe lifted the two configurations known to be handicapped: raising the Householder product count recovers S5 — `signed-delta` at $\texttt{nh}=3$ is still 0/36, at $\texttt{nh}=4$ it reaches 7/36 with clean fit-only generalization ($0.992/0.892/0.618$) — because a $k$-cycle needs $k-1$ reflections and S5's 5-cycle needs four, so $\texttt{nh}=4$ is the representability threshold. Adding S5's order-5 to the rotation snap grid (`signed-rotation-k5`) leaves it at 0/36 under the corrected block order as well, confirming the rotation family's matched zero as a genuine mechanism limit rather than a missing snap order or the block-order confound. Net across matched and probe evidence, S5 has two solving arms: continuous `signed-givens` (1/36) and Householder `signed-delta-nh4` (7/36).

**Pendulum — positive control.** All nine arms fit at every robustness threshold ($p = 1$ throughout). This arm proves the pipeline trains end-to-end, nothing more; the decay channel is not the discriminating axis here.

### 8.4 Cross-task reading and guardrails

Two dials for two task regimes. **Delta is the broad workhorse**: it fits across associative recall (only family to fit MQAR), accumulation ordering (12/12 and 10/12 on psMNIST), and — once its Householder dial is set to the group's cycle structure — group composition (7/36 on S5 at $\texttt{nh}=4$); its dials are state size and $\texttt{nh}$. **Givens is a narrow specialist**: group composition at a small fixed state is its lane (the only matched S5 fit), it leads neither recall nor accumulation, and adding a sign channel to it even hurts accumulation on psMNIST. The earlier matched-small-state Givens win on $S3\text{-hier}$ (Section 4.4) is task-specific and stands.

Three guardrails held deliberately. This is not "delta beats givens everywhere" — the read is which mechanism each task structure rewards. The `gru` control's 0/36 on S5 is a same-budget outcome, not evidence a GRU cannot state-track (Merrill, Petty & Sabharwal, 2024). And no cross-stratum comparison is made: nothing in this section is read against the pinned-CPU rounds elsewhere in this report or the A100 kernel probes.

Full per-arm tables, robustness triples, and completeness accounting: `experiments/bench/bench_{s5,mqar,psmnist,pendulum}.md`, with the probe and reference populations in `bench_s5_probe.md` and `bench_psmnist_ref.md` (regenerated whole by `scripts/report_benchmarks.py`, never hand-edited; round tags `bench-*-02`, `bench-s5-probe-01`, `bench-psmnist-ref-01` in `experiments/lab_results.jsonl`).

## 9. Why the investigation required multiple cycles

Two confounds discovered mid-investigation invalidated first-cycle numbers and required re-runs, and both are now closed.

**Train/eval generator-seeding collision.** The lab harness seeded its training-data generator at $1 + \text{seed}$ while eval calls used fixed literal seeds $2$–$5$; for seed $\geq 1$ this overlapped the training and evaluation streams at the start of training, a partial seed-dependent contamination. The fix (`manual_seed(1 + 10^4 \cdot \text{seed})`) leaves seed $0$ bit-identical. Re-running under the fix moved the $S3$ signed-coupled $L=4$ acc@256 cell from a contaminated $3$-seed mean of $0.544$ to $0.649$ (close to the original single-seed report of $0.655$, confirming the pre-fix mean had suppressed the cell's higher-variance seeds), and corrected the rotation-snap exact-seed count from $2/8$ to $1/8$. The re-run also refuted a claimed property: best-val@128 no longer perfectly separates good from bad seeds under clean seeding (all eight clean seeds passed the flag though seven decay at length), so the flag is necessary but not sufficient.

**$S3\text{-hier}$ task leak.** The first construction used $\mathrm{LATIN} = \mathrm{COMPOSE}$ ($S3$'s own Cayley table), which is trivially isotopic to $S3$; a rotation layer's learned per-token angle assignment gives it the same relabeling freedom an isotopy allows, so a single rotation layer partially represented the pair function directly. The leak inflated every rotation-containing row (`rotsnap` reached $0.377$ at $T=64$ under the leak against $0.225$ after the fix; the GRU, with no rotation mechanism, moved from $0.401$ to $0.232$, confirming the leak was in the task). Replacing `COMPOSE` with a Latin square verified non-isotopic to both groups of order six restored the extract-then-compose structure and cut the original "heterogeneous stack wins" headline to what the fixed task supports. The superseded rows are retained in the ledger, tagged, and excluded from current claims.

The final Givens promotion evidence transfers to the shipped code path by construction bit-identity plus one exact end-to-end replication through `probes.py` (every fit, generalization, and checkpoint metric matching the recorded seed-$0$ row exactly).

## 10. Limitations

The evaluation uses two primary probe tasks (parity, $S3$) plus one hierarchical task ($S3\text{-hier}$) and two timed variants, at $d_{\text{model}}=64$ and a single learning rate; it establishes the ladder's expressivity boundaries under this protocol, not performance on natural-language or long-context modeling. Seed counts bound statistical precision to the reported ranges: parity signed-tanh is n=6, $S3$ rotation-snap is n=8, the deeper $S3$ diagonal cells and $S3\text{-hier}$ hetero rows are n=3 to n=12, and the deeper diagonal cells carry per-seed acc@256 spans of $0.66$–$0.76$ (coupled $L=4$) and $0.83$–$0.95$ (signed-tanh $L=4$), so those means are indicative rather than tight.

All null and partial results are budget-relative: a chance plateau at $1600$ steps is "did not land the exact solution in budget," not "cannot." No configuration solves $S3\text{-hier}$ in budget, and no claim is budget-independent.

Three qualifications of the headline findings deserve explicit statement. The Givens-vs-$2$D comparison changes per-token map expressivity and within-scan block connectivity together; the rounds ablation (Section 4.4) resolves the confound in favor of the commutativity-breaking coupling, though its cross-family anchor ($\texttt{rounds}=1$ vs the $2$D composer) leaves a residual scale-channel difference. The $D_3$ homomorphism certificate is a best-block statement (the most homomorphic of $32$ blocks), not a property of the trained transitions in general. And the promoted Givens composer's full-mean length generalization is within seed noise of a plain homogeneous signed-tanh stack ($0.613$ vs $0.620$ at $T=1024$); its promotion rests on fit reliability, not on that column.

The minGRU wrappers carry a block MLP the bare GRU baseline lacks, which strengthens the minGRU negative results (chance despite the extra capacity) and mildly weakens attribution of the positive ones. The rotation family requires the best-val@128 selection protocol plus retry-on-flag; the architecture alone is not a stable attractor of training (Section 5.2), and the flag is necessary but not sufficient under clean seeding, so length generalization must be confirmed directly rather than trusted from the flag.

Every comparison is internal to this repository: against its own floor, its prior parameterization, its ceiling, and mechanism-level reimplementations of the delta rule. No Grazzi-style negative-eigenvalue scan and no comparison against the incumbents' released code or published numbers is run; the delta-rule comparison is transition-rule-versus-transition-rule under one protocol at same-$d_{\text{model}}$ but not same-capacity (the delta configurations carry $16\times$ the per-token state). The report therefore establishes "reproduces the predicted behavior, minGRU-natively," not "competitive with the incumbents."

`GivensMinGRU`'s parallel $k\times k$ scan is not the cheap path on CPU: in the campaign's measurement, one uncontended forward+backward step at $B=128$, $T=64$ costs $0.961$s against $0.179$s for the lab's sequential delta-rule path. That contrast is not sequential-versus-parallel, because the parallel delta form is implemented here too: the packaged `DeltaMinGRU` (`mixer="delta"`) implements the DeltaNet literature's chunked WY representation as its `forward`, and the pinned bench (`experiments/bench/delta_paths.md`; torch 2.5.1, the same machine as the recorded evidence, whose same-run `GivensMinGRU` arm at $0.9493$s reproduces the $0.961$s above within $\approx 1.2\%$) measures that parallel form at $0.0577$s against $0.1617$s for the sequential step-loop and $2.0514$s for a naive affine-scan reduction at $B=128$, $T=64$ ($1.4541$s / $19.9742$s / $71.7562$s at $T=1024$). The delta path is thus both parallel over the sequence and roughly $16\times$ cheaper per CPU step than the Givens scan, so `GivensMinGRU`'s promotion rests on fit reliability and length generalization within the rotation family, not on training-step speed and not on parallelism. Two limits stand: no trainability run of the packaged chunked path has been recorded — every fit rate in this report comes from the lab's sequential implementation of the same function — and the benched chunked-WY is this repository's implementation, so no comparison against the incumbents' released tuned kernels has been measured.

> [Addendum 2026-07-18: superseded — the "no trainability run of the packaged chunked path has been recorded" limit above no longer holds. A matched-state CPU round (`EXPERIMENTS.md`, `hetero-loop-20-pd64`/`hetero-loop-21-pd1024`) and a 36-seed GPU campaign (`hetero-gpu36-*`) both train the packaged `DeltaMinGRU` directly and resolve the Givens-vs-delta fit-rate question this report left open: at native state, packaged delta is the most reliable composer measured on either stratum (12/12 CPU, 35/36 GPU), while `GivensMinGRU` remains decisively more reliable at a matched small state on the GPU stratum's larger sample ($p=0.00084$). The recommendation is now conditioned on per-token state budget and extrapolation length rather than a single winner; see the docs site's Givens & Delta deep dive and `EXPERIMENTS.md` for the full evidence. The CPU cost/speed numbers stated above are unaffected and remain accurate.]

## 11. Conclusions and recommendation

The evidence establishes a four-rung expressivity ladder over the minGRU, each rung lifting one constraint while preserving parallel associative-scan training, with a standard GRU as the ceiling ($1.000$ on every task and length). The base log-space minGRU is at chance on both state-tracking tasks at any depth, a parameterization limit. A negative diagonal eigenvalue, reached by decoupling the transition from the update gate, solves parity to $0.994$ at $16\times$ the training length. A non-commutative $O(2)$ block transition with straight-through angle snapping solves $S3$ exactly on $1$ of $8$ seeds (seed mean $0.958$ at $16\times$ length) and yields a weights-level best-block $D_3$ homomorphism certificate accurate to $\approx 10^{-4}$, making the state-tracking core auditable without long-sequence evaluation. On the harder extract-then-compose task, composer trainability rises sharply with the richer $8$D Givens map within the rotation family ($8/12$ against $2$D rotation's $1/12$ at matched state, $p \approx 0.009$), the rounds ablation attributes that rise to the coupling that breaks within-block commutativity rather than to block size ($0/12 \to 6/12 \to 8/12$ across $\texttt{rounds} \in \{1,2,3\}$ at fixed block size), and reliability rises with state size within the delta family, while fit-only generalization is mechanism-independent.

The recommendation, by task shape:

- **Sign or parity dynamics** (state that alternates on a running property): `SignedMinGRU` with its default `coupled=False`. Evidentiary basis: $0.994$ at $T=1024$ (n=6, worst seed $0.984$) against $0.592$ coupled and $0.502$ chance, with the eigenvalue mechanism confirmed from weights. Main risk: as a diagonal transition it cannot track non-commutative composition, and it fits such tasks only in-distribution before decaying with length.
- **Non-commutative composition where tokens are the operations** ($S3$-like): `RotationMinGRU` (`mixer="rotation"`, single layer) under best-val@128 selection with retry when best val $< 1.0$. Evidentiary basis: $0.958$ at $T=1024$ (n=8) with a certifiable automaton. Main risk: only $1/8$ seeds land the exact solution and the exact basin is near-zero, so retries must be budgeted and length generalization confirmed directly.
- **Extract-then-compose stacks that must derive an operation from raw input and then compose it** ($S3\text{-hier}$-like): `GivensMinGRU` as the composer (`mixer="givens"`, `block_size=8`, `rounds=3`) above a signed extractor. Evidentiary basis: $8/12$ fit rate at matched $64$-element state, $+2.2\%$ parameters, leading the reliably-trainable field at $0.927/0.733$ fit-only at $T=512/1024$. Main risks: continuity gives no attractor, so accuracy still decays by $T=1024$ and no continuous composer reaches the snapped composer's rare exact $0.983$; and parallelism does not favor this branch — the packaged chunked-WY delta path (`DeltaMinGRU`, `mixer="delta"`) trains parallel over the sequence at roughly $1/16$ the CPU step cost (`experiments/bench/delta_paths.md`), so the case for Givens is matched-state fit reliability ($8/12$ against the small delta composer's $4/12$, suggestive $p \approx 0.22$) and fit-only length generalization, with the caveat that the packaged chunked path has no recorded trainability run (fit rates come from the lab's sequential implementation of the same function).

> [Addendum 2026-07-18: superseded — see `EXPERIMENTS.md` rounds `hetero-loop-20-pd64`/`hetero-loop-21-pd1024` (matched-state, CPU) and `hetero-gpu36-*` (36-seed GPU campaign). The packaged chunked-WY `DeltaMinGRU` now has recorded trainability evidence, and it promotes delta to the recommended default composer *when per-token state is free to grow* (12/12 CPU fits, 35/36 GPU fits at native state, cost and memory flat in state size). The bullet above — `GivensMinGRU` as the extract-then-compose recommendation — still holds specifically for the small, matched-state regime, where it remains decisively more reliable than delta ($p=0.00084$ at $n=36$) and where its fit cohort extrapolates further at extreme length. Read this bullet as one branch of a two-axis recommendation (state budget × extrapolation length), not the sole recommendation for this task shape.]

For irregularly-spaced event streams, the optional time-aware decay term composes with every variant, but its rate is a genuine hyperparameter set by validation at the target length, not one training anneals to zero on its own. No parallel variant reaches the GRU's exactness at length; the ladder is a menu of what parallel training can afford, and the standard GRU remains the reference for tasks that need genuinely sequential computation.

## Artifacts

- `min_gru.py` — single-file implementation of `MinGRU`, `SignedMinGRU`, `RotationMinGRU`, `GivensMinGRU`, the scans (`parallel_scan_log`, `linear_scan`, `matrix_scan`, `matrix_affine_scan`), the shared `DecayMixin` time-decay machinery, and the `MinGRUBlock`/`MinGRUStack` mixer-selector, with a built-in equivalence and determinism test suite.
- `probes.py` — the four probe tasks and the training/evaluation harness with the `CKPT=1` best-val@128 protocol and the offline Latin-square isotopy verification.
- `experiments/variants.py` — the lab harness (extra seeds, eval lengths $512/1024$), the variant registry including `DeltaNetMixer`, and the RNG-hygiene fix.
- `experiments/hetero_lab.py` — driver for the heterogeneous-stack and matched-capacity composer experiments (`hetero-loop` rounds, `DeltaScanMixer`), including the `hetero-sg8r1`/`hetero-sg8r2` rounds-ablation arms.
- `experiments/lab_results.jsonl` — per-seed result ledger ($384$ rows across parity/$S3$/$S3\text{-hier}$/session-parity/parity-timestamped), tagged by round, including superseded rows retained for audit.
- `experiments/mechanism_probes.py`, `experiments/mechanism_results.json` — the $D_3$ homomorphism certificate and parity eigenvalue extraction.
- `experiments/repair_probes.py`, `experiments/repair_round2.py`, `experiments/repair_results.json`, `experiments/repair2_results.json` — the four closed repair interventions.
- `experiments/benchmark_tasks.py`, `experiments/benchmark_lab.py`, `scripts/gpu_benchmark_campaign.py`, `scripts/report_benchmarks.py` — the benchmark-round task specs, task-agnostic driver, GPU campaign runner, and report generator (Section 8).
- `experiments/bench/bench_{s5,mqar,psmnist,pendulum}.{json,md}`, `bench_s5_probe.{json,md}`, `bench_psmnist_ref.{json,md}` — the benchmark round's per-task fit/generalization/Fisher tables, regenerated whole from the ledger.
- `experiments/EXPERIMENTS.md` — the round-by-round experiment log and claim archaeology.
- `experiments/SUMMARY.md` — the curated synthesis, design laws, and open work.
- `README.md` — the orientation and current-state-of-knowledge document.

## References

Feng, L., Tung, F., Ahmed, M. O., Bengio, Y., & Hajimirsadeghi, H. (2024). *Were RNNs All We Needed?* arXiv:2410.01201.

Merrill, W., Petty, J., & Sabharwal, A. (2024). *The Illusion of State in State-Space Models.* ICML 2024. arXiv:2404.08819.

Grazzi, R., et al. (2025). *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025.

Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* arXiv:2406.06484.

Siems, J., Carstensen, T., Zela, A., Hutter, F., Pontil, M., & Grazzi, R. (2025). *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products.* NeurIPS 2025. arXiv:2502.10297.

Arora, S., Eyuboglu, S., Timalsina, A., Johnson, I., Poli, M., Zou, J., Rudra, A., & Ré, C. (2023). *Zoology: Measuring and Improving Recall in Efficient Language Models.* arXiv:2312.04927.

Jing, L., Shen, Y., Dubcek, T., Peurifoy, J., Skirlo, S., LeCun, Y., Tegmark, M., & Soljačić, M. (2017). *Tunable Efficient Unitary Neural Networks (EUNN) and their application to RNNs.* ICML 2017. arXiv:1612.05231.

Clements, W. R., Humphreys, P. C., Metcalf, B. J., Kolthammer, W. S., & Walmsley, I. A. (2016). *Optimal design for universal multiport interferometers.* Optica 3(12).

Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
