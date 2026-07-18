# Related work: GivensMinGRU prior-art survey

Literature sweep for the `GivensMinGRU` construction — input-dependent brick-wall Givens-mesh transitions on the minGRU gate structure, trained through a parallel associative scan. Conducted 2026-07-17 via web search; not exhaustive, and arXiv full-text coverage of the most recent months is imperfect. Rerun the "Givens + linear RNN" check before any submission.

**Bottom line.** No publication found of this exact construction. The architecture-level idea — input-dependent, non-diagonal orthogonal transitions to unlock non-abelian state tracking in parallel-scan linear RNNs — is established published territory (DeltaProduct, AUSSM, PD-SSM, Grazzi et al.). Two things remain unclaimed: (1) the specific parameterization (per-token angles driving a brick-wall Givens mesh as the transition of a scan-parallel linear RNN; full-text search for "Givens" + "minGRU" returns nothing relevant); (2) the trainability framing — reliability at matched per-token state as the measured quantity, separated from expressivity class and from generalization quality, with the rounds ablation isolating the commutativity-breaking stagger as the lever. The architecture alone would likely be judged incremental next to DeltaProduct; the matched-state trainability result plus the rounds ablation is the distinct contribution.

## Closest published work

### DeltaProduct (Siems et al., NeurIPS 2025) — arXiv:2502.10297

The nearest neighbor; already cited and benchmarked against in `TECHNICAL_REPORT.md` §4.4–4.5. Builds per-token transitions as products of generalized Householder matrices in a linear RNN, explicitly for group word problems ($S_3$, $S_5$). Products of Householders and products of Givens rotations sweep essentially the same orthogonal families, so at the level of "input-dependent orthogonal products per token for state tracking," DeltaProduct owns the published claim. Its related-work section states that Givens rotations have previously appeared in RNN state updates only in *non-linear* RNNs with fixed (token-independent) weights — indirect confirmation that the token-dependent Givens-in-a-scan slot was open as of that writing. Proves that $n_h$ Householder factors per token solve any group word problem over permutations of at most $n_h + 1$ elements in one layer. https://arxiv.org/abs/2502.10297

### AUSSM: Adaptive Unitary SSMs (Karuvally, Nowak, Keller, Amo Alonso, Sejnowski, Siegelmann, July 2025) — arXiv:2507.05238

Closest in spirit among work *not* currently cited; the main citation gap to close. Introduces skew-symmetric, input-dependent recurrence achieving unitary evolution in an SSM, and proves via algebraic automata theory that it simulates solvable-group automata at precision logarithmically bounded in input length — the same group-theoretic territory as the rotation rung. Differences from `GivensMinGRU`: parameterization is a skew-symmetric generator rather than a Givens mesh, backbone is an SSM/Mamba hybrid rather than minGRU, and the contribution is expressivity proofs rather than a matched-state trainability result. Practical training uses a separable convolution formulation with a CUDA implementation. https://arxiv.org/abs/2507.05238

## Adjacent published work (each misses on a key axis)

### RotRNN (Biegun et al., 2024) — arXiv:2407.07239

Rotation-matrix transitions in a linear recurrent model, but the rotations are **input-independent** (LRU-style, aimed at long-sequence stability and a clean normalization scheme, not non-commutative state tracking). https://arxiv.org/abs/2407.07239

### Rotational Unit of Memory (RUM) (Dangovski et al., 2017; TACL 2019) — arXiv:1710.09537

Genuine prior art for *input-dependent rotations* in an RNN, but a nonlinear sequential architecture: the rotation is derived from vector geometry (an embedded input/target pair) rather than angle heads, there is no parallel scan, and no state-tracking framing. https://arxiv.org/abs/1710.09537

### Fixed-weight orthogonal/unitary RNNs (2016–2018)

EUNN (Jing et al., ICML 2017; already cited — the source of the brick-wall factorization), uRNN (Arjovsky et al. 2016), GORU, and Householder-parameterized RNNs (Mhammedi et al. 2017). The mesh factorizations exist here, but always with token-independent weights; the angles are learned constants, not functions of $x_t$.

### PD-SSM (Terzic et al., NeurIPS 2025) — arXiv:2509.22284

Transition matrices parameterized as (column one-hot) × (complex diagonal), emulating any $N$-state FSA with one layer of dimension $N$ under a parallel scan. Same goal, discrete permutation-like family instead of continuous rotations. https://arxiv.org/abs/2509.22284

### BD-LRU (Feb 2026) — arXiv:2602.12021

Input-dependent dense block-diagonal transitions with L1-normalized selective gates; solves $S_3$/$S_4$/$S_5$ permutation tasks. A direct competitor to the block-rotation approach that arrives at coupled blocks without orthogonality. Also proposes higher-order recurrences (H-LRU). https://arxiv.org/html/2602.12021

### Revisiting Bi-Linear State Transitions (2025) — arXiv:2505.21749

Theory showing block-diagonal 2D-rotation-block transitions represent modular addition and hence any cyclic group; formalizes part of the rotation rung's story from the bilinear-RNN direction. https://arxiv.org/abs/2505.21749

### Subgroups of U(d) Induce Natural RNN and Transformer Architectures (Feb 2026) — arXiv:2602.18417

General framework placing hidden states on closed subgroups of $U(d)$ with tangent-space updates; subgroup choice acts as a drop-in replacement for state space, projection, and update map. Adjacent framing worth a skim before any writeup. https://arxiv.org/pdf/2602.18417

## Theory backbone (expressivity limits)

Already cited: Merrill, Petty & Sabharwal, *The Illusion of State in State-Space Models* (ICML 2024, arXiv:2404.08819) — the $\mathrm{TC}^0$ ceiling; Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues* (ICLR 2025, arXiv:2411.12537) — the SignedMinGRU rung. Newer companion: *The Expressive Limits of Diagonal SSMs for State-Tracking* (arXiv:2603.01959) — diagonal SSMs track abelian but not non-abelian groups at finite precision. https://arxiv.org/pdf/2603.01959

## Novelty assessment

| claim | status |
|---|---|
| Input-dependent non-diagonal orthogonal transitions enable non-abelian state tracking in a parallel scan | Published (DeltaProduct, AUSSM, PD-SSM, Grazzi et al.) |
| Per-token-angle brick-wall Givens mesh as the transition of a scan-parallel linear RNN | Not found in any publication |
| Matched-state trainability: $SO(8)$ Givens products fit far more reliably than 2D rotations (8/12 vs 1/12) at equal per-token state | Not found; DeltaProduct and AUSSM argue expressivity, and DeltaProduct's headline reliability rides on larger state (quantified in `TECHNICAL_REPORT.md` §4.5) |
| Rounds ablation isolating commutativity-breaking coupling, not block size, as the lever (0/12 → 6/12 → 8/12) | Not found |

If written up, the citation additions are AUSSM (must), plus RotRNN, RUM, and PD-SSM to round out related work.
