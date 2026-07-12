"""minGRU from Feng et al., "Were RNNs All We Needed?" (arXiv:2410.01201).

The minGRU removes the hidden-state dependency from the GRU's gates
(which also eliminates the reset gate) and drops the tanh range
restriction on the candidate state:

    z_t = sigmoid(Linear_z(x_t))
    h~_t = Linear_h(x_t)
    h_t = (1 - z_t) * h_{t-1} + z_t * h~_t

Because z_t and h~_t depend only on x_t, the recurrence is a first-order
linear scan h_t = a_t * h_{t-1} + b_t with a_t = (1 - z_t) and
b_t = z_t * h~_t, computable in parallel over the sequence dimension.

This module implements the paper's log-space parallel scan (Appendix B),
which is the numerically stable form the authors recommend, plus a
`step()` method for O(1)-memory recurrent inference. The log-space
parameterization applies g(x) = x + 0.5 (x >= 0) or sigmoid(x) (x < 0)
to the candidate and initial states so all quantities are positive and
logs are well-defined; `step()` applies the same g so the two paths are
numerically equivalent. Note this constrains hidden states to be
positive — a property of the paper's log-space variant, not of the
"vanilla" minGRU in Appendix A.

Parameter count: O(2 * d_h * d_x) vs. O(3 * d_h * (d_x + d_h)) for a
standard GRU.

`MinGRU`, `SignedMinGRU`, and `RotationMinGRU` are each kept atomic (one
scan layer, one sequence-mixing mechanism). `MinGRUBlock` wraps any one
of them, chosen via `mixer="log"|"signed"|"rotation"` (plus
`mixer_kwargs` for per-mixer config such as `coupled=True` or a custom
`snap` grid), in the standard pre-norm residual template (LN -> mixer ->
residual, then LN -> MLP -> residual), which supplies the inter-layer
nonlinear mixing: layer l's gates condition on layer l-1's hidden
states, and the MLP provides cross-channel interaction that a diagonal
scan cannot. `MinGRUStack` stacks N blocks under a single `mixer`
selection and supports both parallel training-mode forward and
O(1)-memory streaming via step().
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def g(x: torch.Tensor) -> torch.Tensor:
    """Continuous positive-valued activation from the paper's Appendix B.

    ``g(x) = x + 0.5`` for ``x >= 0``, ``sigmoid(x)`` for ``x < 0``.
    Continuous at 0 (both branches equal 0.5) and strictly positive
    everywhere.

    Parameters
    ----------
    x : torch.Tensor
        Pre-activation values, any shape.

    Returns
    -------
    torch.Tensor
        ``g(x)``, same shape as ``x``, values in ``(0, inf)``.
    """
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x: torch.Tensor) -> torch.Tensor:
    """``log(g(x))``, stable and with a usable gradient everywhere.

    Parameters
    ----------
    x : torch.Tensor
        Pre-activation values, any shape.

    Returns
    -------
    torch.Tensor
        ``log(g(x))``, same shape as ``x``.

    Notes
    -----
    The ``x >= 0`` branch must not compute ``(x + 0.5).log()`` directly:
    since ``torch.where`` evaluates both branches, ``x < -0.5`` would
    produce NaNs that poison gradients even when unselected. The paper
    guards this with relu, but ``relu'(0) = 0`` kills the gradient at
    exactly ``x = 0`` (outside the true subdifferential ``[0.5, 2]``),
    deadening any zero-initialized parameter fed through ``log_g``. The
    nested ``where`` keeps the unselected branch a finite constant while
    the selected branch at ``x = 0`` is plain ``x``, giving gradient
    ``1/g(0) = 2``.
    """
    safe_x = torch.where(x >= 0, x, torch.zeros_like(x))
    return torch.where(x >= 0, (safe_x + 0.5).log(), -F.softplus(-x))


def parallel_scan_log(log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
    """Heinsen (2023) parallel scan in log-space.

    Solves ``h_t = a_t * h_{t-1} + b_t`` for all ``t`` simultaneously,
    given ``log(a_t)`` and ``log(b_t)``, assuming ``a_t, b_t > 0``.

    Parameters
    ----------
    log_coeffs : torch.Tensor
        Shape ``(B, T, D)``. ``log(a_t)`` for ``t = 1..T``.
    log_values : torch.Tensor
        Shape ``(B, T + 1, D)``. Slot 0 is ``log(h_0)`` — the scan's
        initial value, i.e. ``v_0 <- h_0`` in Heinsen's formulation;
        slots ``1..T`` are ``log(b_t)``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, T, D)``. The states ``h_1..h_T``.
    """
    # a*_t = sum_{i<=t} log(a_i), padded with a*_0 = 0
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]


class MinGRU(nn.Module):
    """minGRU layer: log-space parallel-scan training, recurrent inference.

    Parameters
    ----------
    input_size : int
        Dimensionality of the inputs ``x_t``.
    hidden_size : int
        Dimensionality of the hidden states ``h_t``.
    bias : bool, default=True
        Whether the two linear maps carry bias terms.
    learnable_h0 : bool, default=False
        If True, the module owns a learned initial state, stored as an
        unconstrained pre-activation and mapped through ``g`` internally
        (so positivity is by construction). Used whenever ``forward()``
        or ``step()`` receive no explicit state. Zero-init gives
        ``g(0) = 0.5``, matching the fixed default.

    Notes
    -----
    Shapes: ``forward`` maps ``x (B, T, input_size)`` with optional
    ``h_0 (B, 1, hidden_size)`` to ``(B, T, hidden_size)``; ``step``
    maps ``x_t (B, input_size)`` with optional ``h_prev
    (B, hidden_size)`` to ``(B, hidden_size)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        learnable_h0: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h0_pre: nn.Parameter | None = (
            nn.Parameter(torch.zeros(1, 1, hidden_size)) if learnable_h0 else None
        )

    def _default_log_h0(self, B: int, x: torch.Tensor) -> torch.Tensor:
        if self.h0_pre is not None:
            return log_g(self.h0_pre).expand(B, 1, self.hidden_size)
        return torch.full(
            (B, 1, self.hidden_size), 0.5, dtype=x.dtype, device=x.device
        ).log()

    def forward(self, x: torch.Tensor, h_0: torch.Tensor | None = None) -> torch.Tensor:
        """Parallel (training-mode) forward over a full sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape ``(B, T, input_size)``.
        h_0 : torch.Tensor, optional
            Initial hidden state, shape ``(B, 1, hidden_size)``, as a
            REAL state — i.e. an output of ``forward()``/``step()``,
            non-negative (exact zeros from fp16/bf16 underflow are
            clamped). Do NOT pass a pre-activation; ``g`` is not applied
            here (this differs from the paper's reference code, which
            treats ``h_0`` as a pre-activation). Defaults to
            ``g(0) = 0.5``, matching ``step()``'s default. For
            chunked/TBPTT training, carry ``h_0 = prev_out[:, -1:]``
            (detach as appropriate).

        Returns
        -------
        torch.Tensor
            All hidden states ``h_1..h_T``, shape
            ``(B, T, hidden_size)``.

        Raises
        ------
        RuntimeError
            If ``h_0`` contains strictly negative entries (the signature
            of pre-activation misuse). Raised device-side via
            ``torch._assert_async``; on CUDA the error surfaces at the
            next sync point rather than at this call.
        """
        B = x.size(0)
        if h_0 is None:
            log_h_0 = self._default_log_h0(B, x)
        else:
            # Strictly negative entries signal misuse (e.g. passing a
            # pre-activation). Checked device-side: no host sync on
            # CUDA, so chunked-forward loops don't stall per chunk.
            torch._assert_async(
                (h_0 >= 0).all(),
                "h_0 must be a non-negative real hidden state "
                "(an output of forward()/step()), not a pre-activation.",
            )
            # Exact zeros are legitimate: valid small states underflow
            # to 0 in fp16/bf16. Clamp to the dtype's smallest normal
            # so log() stays finite.
            log_h_0 = h_0.clamp_min(torch.finfo(h_0.dtype).tiny).log()

        k = self.linear_z(x)                    # pre-activation of z
        log_z = -F.softplus(-k)                 # log(sigmoid(k)) = log(z)
        log_coeffs = -F.softplus(k)             # log(1 - sigmoid(k)) = log(1 - z)
        log_tilde_h = log_g(self.linear_h(x))
        return parallel_scan_log(
            log_coeffs,
            torch.cat([log_h_0, log_z + log_tilde_h], dim=1),
        )

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        """Single recurrent step for streaming/token-by-token inference.

        Numerically equivalent to ``forward()`` one timestep at a time
        (``g`` is applied to the candidate to match the log-space
        parameterization). ``h_prev`` is a real hidden state — the same
        convention as ``forward()``'s ``h_0``.

        Parameters
        ----------
        x_t : torch.Tensor
            Input at the current timestep, shape ``(B, input_size)``.
        h_prev : torch.Tensor, optional
            Previous hidden state, shape ``(B, hidden_size)`` — an
            output of ``step()`` or ``forward()``. Defaults to
            ``g(0) = 0.5`` (or the learned initial state if
            ``learnable_h0``), matching ``forward()``'s default.

        Returns
        -------
        torch.Tensor
            New hidden state ``h_t``, shape ``(B, hidden_size)``.
        """
        if h_prev is None:
            if self.h0_pre is not None:
                h_prev = g(self.h0_pre)[:, 0].expand(x_t.size(0), self.hidden_size)
            else:
                h_prev = x_t.new_full((x_t.size(0), self.hidden_size), 0.5)
        z = torch.sigmoid(self.linear_z(x_t))
        h_tilde = g(self.linear_h(x_t))
        return (1 - z) * h_prev + z * h_tilde

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, hidden_size={self.hidden_size}"


def linear_scan(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear-space associative scan for ``h_t = a_t * h_{t-1} + b_t``.

    Hillis-Steele doubling over the segment-composition monoid
    ``(A1, B1) o (A2, B2) = (A2*A1, A2*B1 + B2)``: O(T log T) work,
    O(log T) depth, pure torch ops, differentiable. Unlike
    ``parallel_scan_log``, coefficients may be NEGATIVE — this is what
    enables signed transition eigenvalues (see ``SignedMinGRU``).

    Parameters
    ----------
    a : torch.Tensor
        Shape ``(B, T, D)``. Transition coefficients ``a_t``; stability
        expects ``|a_t| <= 1`` but is not enforced.
    b : torch.Tensor
        Shape ``(B, T, D)``. Additive inputs ``b_t``.

    Returns
    -------
    tuple of torch.Tensor
        ``(A, Bc)``, each ``(B, T, D)``, where
        ``h_t = A_t * h_0 + Bc_t`` with ``A_t`` the running product of
        ``a`` and ``Bc_t`` the ``h_0 = 0`` solution of the recurrence.

    Notes
    -----
    Hillis-Steele is work-inefficient (O(T log T) vs O(T) for a
    work-efficient Blelloch scan) and retains O(log T) full ``(B, T, D)``
    tensors for autograd. That is a deliberate simplicity-over-efficiency
    choice: the overhead is negligible at the sequence lengths this repo
    targets (T <= 256), and no stable ``torch.associative_scan`` primitive
    exists to lean on. Revisit if sequences grow large.
    """
    T = a.size(1)
    A, Bc = a, b
    offset = 1
    while offset < T:
        A_prev = F.pad(A, (0, 0, offset, 0), value=1.0)[:, :T]
        B_prev = F.pad(Bc, (0, 0, offset, 0), value=0.0)[:, :T]
        Bc = A * B_prev + Bc
        A = A * A_prev
        offset *= 2
    return A, Bc


def matrix_scan(M: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Associative scan for ``h_t = M_t @ h_{t-1} + b_t``, 2x2 block transitions.

    The non-commutative generalization of ``linear_scan``: transitions
    are 2x2 matrices composed by matrix multiplication instead of
    scalars multiplied together, so the running product depends on
    order. This is what lets a mixer built on ``matrix_scan``
    (``RotationMinGRU``) express non-abelian state tracking that a
    diagonal/commutative scan (``linear_scan``, ``parallel_scan_log``)
    provably cannot. Same Hillis-Steele doubling scheme as
    ``linear_scan``, over the affine 2x2 monoid
    ``(A1, B1) o (A2, B2) = (A2 @ A1, A2 @ B1 + B2)``: O(T log T) work,
    O(log T) depth, pure torch ops, differentiable. The identity
    padding is the 2x2 identity matrix (not 1.0), and composition order
    is ``A_current @ A_earlier`` — matrix multiplication does not
    commute, unlike ``linear_scan``'s scalar product.

    Parameters
    ----------
    M : torch.Tensor
        Shape ``(B, T, n, 2, 2)``. Per-block transition matrices
        ``M_t``; stability expects ``M_t``'s spectral norm ``<= 1`` but
        this is not enforced.
    b : torch.Tensor
        Shape ``(B, T, n, 2)``. Additive inputs ``b_t``, viewed as
        ``n`` 2-vectors per timestep.

    Returns
    -------
    tuple of torch.Tensor
        ``(A, Bc)``, shapes ``(B, T, n, 2, 2)`` and ``(B, T, n, 2)``,
        each aligned to ``t``, where ``h_t = A_t @ h_0 + Bc_t`` with
        ``A_t`` the running matrix product of ``M`` and ``Bc_t`` the
        ``h_0 = 0`` solution of the recurrence.

    Notes
    -----
    Same simplicity-over-efficiency tradeoff as ``linear_scan``:
    Hillis-Steele is work-inefficient (O(T log T) vs O(T)) and retains
    O(log T) full ``(B, T, n, 2, 2)`` tensors for autograd. Fine at
    this repo's target sequence lengths (T <= 256); revisit if
    sequences grow large.
    """
    B_, T, n = M.shape[:3]
    eye = torch.eye(2, dtype=M.dtype, device=M.device)
    A, Bc = M, b
    offset = 1
    while offset < T:
        pad_A = eye.expand(B_, offset, n, 2, 2)
        A_prev = torch.cat([pad_A, A[:, : T - offset]], dim=1)
        B_prev = torch.cat([b.new_zeros(B_, offset, n, 2), Bc[:, : T - offset]], dim=1)
        Bc = torch.einsum("btnij,btnj->btni", A, B_prev) + Bc
        A = A @ A_prev
        offset *= 2
    return A, Bc


class SignedMinGRU(nn.Module):
    """minGRU variant with signed diagonal transitions (linear-space scan).

    The recurrence is ``h_t = a_t * h_{t-1} + z_t * h~_t`` with
    ``a_t`` ranging over ``(-1, 1)`` instead of ``(0, 1)``. Motivated by
    Merrill, Petty & Sabharwal (2024): still diagonal, hence commutative
    and TC0, but negative eigenvalues restore per-layer parity /
    sign-alternation dynamics (the mechanism identified by Grazzi et al.,
    2025).

    Two parameterizations of ``a_t`` are available:

    - ``coupled=False`` (default): ``a_t = tanh(Linear_s(x_t))``, the
      eigenvalue decoupled from the update gate ``z_t``. This is the
      experimentally superior form (parity accuracy @1024:
      0.61 -> 0.996, 6-seed mean, current-env) and is now the default.
    - ``coupled=True``: ``a_t = (1 - z_t) * tanh(Linear_s(x_t))``, the
      original parameterization (eigenvalue coupled to the update gate,
      as in the vanilla minGRU's ``(1 - z_t)`` retention coefficient).
      Bit-exact reproduction of the pre-promotion class: identical
      parameter shapes and construction order (``linear_z``,
      ``linear_h``, ``linear_s``), so identical seeds give identical
      weights.

    When the sign head saturates positive, the coupled form reduces to
    the vanilla (Appendix A) minGRU's retention dynamics (``a = 1 - z``);
    the decoupled form instead saturates to ``a = 1``, a perfect
    integrator — reaching the interval boundary is exactly what the
    coupling prevents, and why the decoupled form length-generalizes.

    States are unconstrained reals: no ``g``, no positivity checks, no
    underflow handling. Same forward/step API and shapes as ``MinGRU``;
    ``h_0`` is any real state.

    Parameters
    ----------
    input_size : int
        Dimensionality of the inputs ``x_t``.
    hidden_size : int
        Dimensionality of the hidden states ``h_t``.
    bias : bool, default=True
        Whether the three linear maps carry bias terms.
    learnable_h0 : bool, default=False
        If True, the module owns a learned initial state (an
        unconstrained parameter used directly — no ``g``). Zero-init
        matches the fixed default ``h_0 = 0``.
    coupled : bool, default=False
        If True, use the legacy coupled eigenvalue
        ``a_t = (1 - z_t) * tanh(Linear_s(x_t))``. If False (default),
        use the decoupled eigenvalue ``a_t = tanh(Linear_s(x_t))``.

    Notes
    -----
    Parameter count is 3 linear heads vs. MinGRU's 2 (the extra sign
    head) — account for this in parameter-matched comparisons.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        learnable_h0: bool = False,
        coupled: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.coupled = coupled
        self.linear_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_s = nn.Linear(input_size, hidden_size, bias=bias)
        self.h0: nn.Parameter | None = (
            nn.Parameter(torch.zeros(1, 1, hidden_size)) if learnable_h0 else None
        )

    def _coeffs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.sigmoid(self.linear_z(x))
        tanh_s = torch.tanh(self.linear_s(x))
        a = (1 - z) * tanh_s if self.coupled else tanh_s
        b = z * self.linear_h(x)
        return a, b

    def forward(self, x: torch.Tensor, h_0: torch.Tensor | None = None) -> torch.Tensor:
        """Parallel forward over a full sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape ``(B, T, input_size)``.
        h_0 : torch.Tensor, optional
            Initial hidden state, shape ``(B, 1, hidden_size)``. Any
            real values. Defaults to zeros (or the learned initial
            state if ``learnable_h0``).

        Returns
        -------
        torch.Tensor
            All hidden states ``h_1..h_T``, shape
            ``(B, T, hidden_size)``.
        """
        if h_0 is None:
            h_0 = (
                self.h0.expand(x.size(0), 1, self.hidden_size)
                if self.h0 is not None
                else x.new_zeros(x.size(0), 1, self.hidden_size)
            )
        a, b = self._coeffs(x)
        A, Bc = linear_scan(a, b)
        return A * h_0 + Bc

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        """Single recurrent step; same real-state convention as forward().

        Parameters
        ----------
        x_t : torch.Tensor
            Input at the current timestep, shape ``(B, input_size)``.
        h_prev : torch.Tensor, optional
            Previous hidden state, shape ``(B, hidden_size)``. Defaults
            to zeros (or the learned initial state).

        Returns
        -------
        torch.Tensor
            New hidden state, shape ``(B, hidden_size)``.
        """
        if h_prev is None:
            h_prev = (
                self.h0[:, 0].expand(x_t.size(0), self.hidden_size)
                if self.h0 is not None
                else x_t.new_zeros(x_t.size(0), self.hidden_size)
            )
        a, b = self._coeffs(x_t)
        return a * h_prev + b

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"coupled={self.coupled}"
        )


class RotationMinGRU(nn.Module):
    """minGRU variant with 2x2 block rotation transitions (non-diagonal).

    State is ``n = hidden_size / 2`` independent planar (2D) blocks.
    Per block, the recurrence is a full 2x2 affine map instead of a
    scalar one:

        M_t = R(theta_t) @ diag(1, tanh(u_t))
        h_t = M_t @ h_{t-1} + b_t,     b_t = z_t * Linear_h(x_t)

    (``h_{t-1}``, ``h_t``, ``b_t`` viewed as 2-vectors per block).
    Unlike ``MinGRU`` and ``SignedMinGRU``, per-block transitions do
    not commute (2x2 rotation/reflection matrices form a non-abelian
    group under composition), so this mixer — with a non-commutative
    parallel scan, ``matrix_scan`` — can represent state-tracking
    automata over non-abelian groups that a diagonal (commutative)
    scan provably cannot. D3 (isomorphic to S3, the smallest
    non-abelian group) embeds in O(2), so one layer of this mixer can
    represent the S3 running product exactly; see
    ``experiments/SUMMARY.md`` for the mechanism verification
    (per-block matrices extracted from a trained model satisfy the D3
    composition table to ~1e-4).

    Angle snapping (``snap``): with ``snap`` set, ``theta_t`` is
    quantized per block to an exact multiple of ``2*pi/K`` via a
    straight-through estimator — forward uses the snapped angle,
    gradient passes through the pre-snap "soft" angle unchanged. ``K``
    is cycled across blocks from the ``snap`` tuple (block ``j`` uses
    ``snap[j % len(snap)]``). This manufactures attractors at exact
    group elements, the same way ``tanh``'s asymptote manufactures an
    attractor at eigenvalue -1 for ``SignedMinGRU``: without snapping,
    plain rotation angles have no attractor and drift with sequence
    length (error compounds with T). The snap grid must contain the
    group being tracked: choose ``K`` values whose rotations
    (``2*pi/K``) generate, or coincide with, the target group's
    rotation subgroup, or the exact automaton is not representable on
    the grid at all (e.g. tracking Z/5 needs a multiple of 5 in
    ``snap``). The default ``snap=(2, 3, 4, 6)`` was chosen for the
    D3/S3 task; other state-tracking targets need their own grid.
    ``snap=None`` gives continuous (unsnapped) rotations — a
    legitimate, documented ladder rung, but angles then drift under
    length generalization since there is no attractor at the task's
    true transition angle; use only when exact length generalization
    is not required.

    Depth: validated at L=1 ONLY. Stacking these mixers breaks STE
    snap training (the straight-through discontinuity compounds across
    layers) — do not assume depth helps here the way it does for the
    diagonal mixers.

    Training protocol: the exact automaton is reachable but is NOT a
    stable attractor of standard training — runs wander in and out of
    it during optimization. The validated protocol is best-checkpoint
    selection by validation accuracy at a length LONGER than the
    training length (e.g. T=128 when training at T=64; not one of the
    eventual test lengths), evaluated over the full step budget instead
    of early-stopping, plus a retry-on-flag rule: a best validation
    score at that checkpoint length below 1.0 flags the run as failed
    (this perfectly separated good from bad seeds in the recorded
    evidence). See ``experiments/SUMMARY.md`` for the full protocol,
    per-seed success rate, and mechanism verification.

    Excludes refuted experiment-loop mechanisms: no full orthogonality
    constraint (``ortho``), no grid-attraction regularizer (``reg``),
    no post-hoc projection/ablation masks. All were tried and either
    hurt length generalization or were redundant with the best-val
    selection protocol above; see ``experiments/SUMMARY.md`` rounds 5
    and 8.

    ``h_0`` is an unconditional learnable parameter (no
    ``learnable_h0`` flag, unlike the module's other two mixers):
    ``h_0 = 0`` has no orbit under the group action (a fixed point
    cannot demonstrate state tracking), and a state vector lying on a
    reflection axis collapses reflections onto rotations. A random
    nonzero learned vector avoids both failure modes.

    Parameters
    ----------
    input_size : int
        Dimensionality of the inputs ``x_t``.
    hidden_size : int
        Dimensionality of the hidden states ``h_t``; must be even
        (``hidden_size = 2 * n_blocks``).
    bias : bool, default=True
        Whether the four linear maps carry bias terms.
    snap : tuple of int, or None, default=(2, 3, 4, 6)
        Per-block angle-snap grid orders ``K`` (cycled across blocks);
        each block's angle snaps to multiples of ``2*pi/K``. ``None``
        disables snapping (continuous rotations; see above).

    Raises
    ------
    ValueError
        If ``hidden_size`` is odd.

    Notes
    -----
    Parameter count is 4 linear heads (z, h, theta, u) vs.
    ``SignedMinGRU``'s 3 — account for this in parameter-matched
    comparisons. Same forward/step shapes as the module's other
    mixers: ``forward`` maps ``x (B, T, input_size)`` with optional
    ``h_0 (B, 1, hidden_size)`` to ``(B, T, hidden_size)``; ``step``
    maps ``x_t (B, input_size)`` with optional ``h_prev
    (B, hidden_size)`` to ``(B, hidden_size)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        snap: tuple[int, ...] | None = (2, 3, 4, 6),
    ):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError(
                f"RotationMinGRU requires an even hidden_size (got {hidden_size}); "
                "state is n = hidden_size / 2 planar 2D blocks."
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_blocks = hidden_size // 2
        self.snap = snap
        self.linear_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_theta = nn.Linear(input_size, self.n_blocks, bias=bias)
        self.linear_u = nn.Linear(input_size, self.n_blocks, bias=bias)
        self.h0 = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.5)
        if snap is not None:
            self.register_buffer(
                "snap_step",
                torch.tensor(
                    [2 * math.pi / snap[j % len(snap)] for j in range(self.n_blocks)]
                ),
            )

    def _coeffs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-block transition matrix and injection; shared by forward/step.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., input_size)`` — ``forward`` passes
            ``(B, T, input_size)``, ``step`` passes ``(B,
            input_size)``; both work unchanged since ``nn.Linear`` and
            the elementwise ops here broadcast uniformly over leading
            dims, so this single helper serves both call paths and
            they cannot drift apart.

        Returns
        -------
        tuple of torch.Tensor
            ``M``, shape ``(..., n_blocks, 2, 2)``: the (possibly
            snapped) transition ``R(theta_t) @ diag(1, tanh(u_t))``.
            ``b``, shape ``(..., n_blocks, 2)``: the injection
            ``z_t * Linear_h(x_t)``, reshaped into ``n_blocks``
            2-vectors.
        """
        theta = self.linear_theta(x)
        if self.snap is not None:
            snapped = torch.round(theta / self.snap_step) * self.snap_step
            # STE: forward uses the snapped angle; gradient passes
            # through the pre-snap angle unchanged.
            theta = theta + (snapped - theta).detach()
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        d = torch.tanh(self.linear_u(x))
        # R(theta) @ diag(1, d) = [[cos, -sin*d], [sin, cos*d]]
        row0 = torch.stack([cos_t, -sin_t * d], dim=-1)
        row1 = torch.stack([sin_t, cos_t * d], dim=-1)
        M = torch.stack([row0, row1], dim=-2)

        z = torch.sigmoid(self.linear_z(x))
        b = z * self.linear_h(x)
        b = b.reshape(*b.shape[:-1], self.n_blocks, 2)
        return M, b

    def forward(self, x: torch.Tensor, h_0: torch.Tensor | None = None) -> torch.Tensor:
        """Parallel forward over a full sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape ``(B, T, input_size)``.
        h_0 : torch.Tensor, optional
            Initial hidden state, shape ``(B, 1, hidden_size)``. Any
            real values (reshaped into ``n_blocks`` 2-vectors
            internally). Defaults to the learned initial state.

        Returns
        -------
        torch.Tensor
            All hidden states ``h_1..h_T``, shape
            ``(B, T, hidden_size)``.
        """
        B, T, _ = x.shape
        if h_0 is None:
            h_0 = self.h0.expand(B, 1, self.hidden_size)
        h0_blocks = h_0.reshape(B, self.n_blocks, 2)
        M, b = self._coeffs(x)
        A, Bc = matrix_scan(M, b)
        h = torch.einsum("btnij,bnj->btni", A, h0_blocks) + Bc
        return h.reshape(B, T, self.hidden_size)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        """Single recurrent step; same real-state convention as forward().

        Computed from the same ``_coeffs`` helper ``forward()`` uses
        (applied per-step instead of over the full sequence), so the
        two paths cannot drift apart — mirrors how ``SignedMinGRU``
        shares ``_coeffs`` between its ``forward``/``step``.

        Parameters
        ----------
        x_t : torch.Tensor
            Input at the current timestep, shape ``(B, input_size)``.
        h_prev : torch.Tensor, optional
            Previous hidden state, shape ``(B, hidden_size)``. Defaults
            to the learned initial state.

        Returns
        -------
        torch.Tensor
            New hidden state, shape ``(B, hidden_size)``.
        """
        B = x_t.size(0)
        if h_prev is None:
            h_prev = self.h0.expand(B, 1, self.hidden_size)[:, 0]
        h_prev_blocks = h_prev.reshape(B, self.n_blocks, 2)
        M, b = self._coeffs(x_t)
        h = torch.einsum("bnij,bnj->bni", M, h_prev_blocks) + b
        return h.reshape(B, self.hidden_size)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"snap={self.snap}"
        )


class MinGRUBlock(nn.Module):
    """Pre-norm residual block: LN -> minGRU -> +x, then LN -> MLP -> +x.

    The minGRU is the (linear-in-state) sequence mixer; the MLP is the
    position-wise channel mixer. Because the scan's transition is
    diagonal, cross-channel interaction happens only here and in the
    next block's input projections.

    Parameters
    ----------
    d_model : int
        Residual stream width (the minGRU maps ``d_model -> d_model``).
    mlp_expansion : int, default=4
        Hidden width multiplier for the MLP. ``0`` disables the MLP
        sub-block entirely (scan-only block).
    dropout : float, default=0.0
        Applied after the minGRU output and inside the MLP.
    learnable_h0 : bool, default=False
        Routed to the block's mixer when ``mixer`` is ``"log"`` or
        ``"signed"`` (see ``MinGRU`` / ``SignedMinGRU``). Not accepted
        by ``"rotation"``: ``RotationMinGRU``'s ``h_0`` is an intrinsic
        learned parameter with no ``learnable_h0`` flag, so this
        argument is silently unused when ``mixer="rotation"``.
    mixer : {"log", "signed", "rotation"}, default="log"
        Selects the sequence mixer: ``MinGRU`` (log-space parallel
        scan), ``SignedMinGRU`` (signed diagonal transitions), or
        ``RotationMinGRU`` (2x2 block rotations). Any other value
        raises ``ValueError``.
    mixer_kwargs : dict, optional
        Extra constructor kwargs forwarded to the selected mixer class
        (e.g. ``{"coupled": True}`` for ``"signed"``, or
        ``{"snap": (2, 3, 5)}`` for ``"rotation"``).
    """

    _MIXER_CLASSES: dict[str, type[nn.Module]] = {
        "log": MinGRU,
        "signed": SignedMinGRU,
        "rotation": RotationMinGRU,
    }

    def __init__(
        self,
        d_model: int,
        mlp_expansion: int = 4,
        dropout: float = 0.0,
        learnable_h0: bool = False,
        mixer: str = "log",
        mixer_kwargs: dict | None = None,
    ):
        super().__init__()
        if mixer not in self._MIXER_CLASSES:
            raise ValueError(
                f"unknown mixer {mixer!r}; expected one of "
                f"{sorted(self._MIXER_CLASSES)}"
            )
        mixer_kwargs = dict(mixer_kwargs) if mixer_kwargs else {}
        self.norm1 = nn.LayerNorm(d_model)
        if mixer == "rotation":
            # RotationMinGRU's h_0 is intrinsic; it takes no
            # learnable_h0 kwarg (see class docstring).
            self.mingru = RotationMinGRU(d_model, d_model, **mixer_kwargs)
        else:
            self.mingru = self._MIXER_CLASSES[mixer](
                d_model, d_model, learnable_h0=learnable_h0, **mixer_kwargs
            )
        self.drop = nn.Dropout(dropout)
        if mlp_expansion > 0:
            self.norm2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_expansion * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_expansion * d_model, d_model),
            )
        else:
            self.norm2 = None
            self.mlp = None

    def forward(
        self, x: torch.Tensor, h_0: torch.Tensor | None = None, return_state: bool = False
    ):
        """Parallel forward over a full sequence.

        Parameters
        ----------
        x : torch.Tensor
            Residual-stream input, shape ``(B, T, d_model)``.
        h_0 : torch.Tensor, optional
            This block's real minGRU state carried from a previous
            chunk, shape ``(B, 1, d_model)``; see ``MinGRU.forward``.
        return_state : bool, default=False
            If True, also return the block's final minGRU state for
            the next chunk.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Output ``(B, T, d_model)``; if ``return_state``, a tuple of
            the output and the final minGRU state ``(B, 1, d_model)``.
        """
        h_seq = self.mingru(self.norm1(x), h_0)
        x = x + self.drop(h_seq)
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        if return_state:
            return x, h_seq[:, -1:]
        return x

    @torch.no_grad()
    def step(
        self, x_t: torch.Tensor, h_prev: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single streaming step.

        Parameters
        ----------
        x_t : torch.Tensor
            Residual-stream input at time ``t``, shape ``(B, d_model)``.
        h_prev : torch.Tensor or None
            This block's minGRU state from ``t-1``, shape
            ``(B, d_model)``; None at the first step.

        Returns
        -------
        tuple of torch.Tensor
            The block output and the new minGRU state, each
            ``(B, d_model)``.
        """
        h = self.mingru.step(self.norm1(x_t), h_prev)
        x_t = x_t + h
        if self.mlp is not None:
            x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, h


class MinGRUStack(nn.Module):
    """N stacked MinGRUBlocks with input projection and final norm.

    Parameters
    ----------
    input_size : int
        Dimensionality of the raw inputs ``x_t``.
    d_model : int
        Residual stream width.
    n_layers : int
        Number of blocks.
    mlp_expansion : int, default=4
        Per-block MLP expansion; ``0`` gives scan-only blocks.
    dropout : float, default=0.0
        Per-block dropout.
    learnable_h0 : bool, default=False
        Passed through to every block; routed to the block's mixer
        when ``mixer`` is ``"log"`` or ``"signed"``, unused for
        ``"rotation"`` (see ``MinGRUBlock``).
    mixer : {"log", "signed", "rotation"}, default="log"
        Sequence mixer used by every block; see ``MinGRUBlock``. Any
        other value raises ``ValueError``.
    mixer_kwargs : dict, optional
        Extra constructor kwargs forwarded to every block's mixer; see
        ``MinGRUBlock``.

    Notes
    -----
    Shapes: ``forward`` maps ``(B, T, input_size)`` to
    ``(B, T, d_model)``; ``step`` maps ``(B, input_size)`` and a state
    (a list of ``n_layers`` tensors of shape ``(B, d_model)``) to the
    output ``(B, d_model)`` and the updated state.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_layers: int,
        mlp_expansion: int = 4,
        dropout: float = 0.0,
        learnable_h0: bool = False,
        mixer: str = "log",
        mixer_kwargs: dict | None = None,
    ):
        super().__init__()
        self.in_proj = (
            nn.Linear(input_size, d_model) if input_size != d_model else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            MinGRUBlock(
                d_model, mlp_expansion, dropout, learnable_h0, mixer, mixer_kwargs
            )
            for _ in range(n_layers)
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor | None] | None = None,
        return_state: bool = False,
    ):
        """Parallel training-mode forward.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape ``(B, T, input_size)``.
        state : list of (torch.Tensor or None), optional
            Per-block minGRU states from a previous chunk — as returned
            with ``return_state=True``, or from streaming ``step()``
            after unsqueezing each entry to ``(B, 1, d_model)``. For
            TBPTT, detach the returned state before feeding it to the
            next chunk.
        return_state : bool, default=False
            If True, also return the per-block final states.

        Returns
        -------
        torch.Tensor or tuple
            Output ``(B, T, d_model)``; if ``return_state``, a tuple of
            the output and a list of ``n_layers`` states, each
            ``(B, 1, d_model)``.
        """
        if state is None:
            state = self.init_state()
        x = self.in_proj(x)
        new_state = []
        for block, h_0 in zip(self.blocks, state):
            if return_state:
                x, h_last = block(x, h_0, return_state=True)
                new_state.append(h_last)
            else:
                x = block(x, h_0)
        out = self.norm_out(x)
        if return_state:
            return out, new_state
        return out

    def init_state(self) -> list[None]:
        """Fresh streaming state.

        Returns
        -------
        list of None
            One slot per block; pass to ``step()`` at the first
            timestep.
        """
        return [None] * len(self.blocks)

    @torch.no_grad()
    def step(
        self, x_t: torch.Tensor, state: list[torch.Tensor | None]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Streaming step.

        Total cached state is ``n_layers * d_model`` per sample.

        Parameters
        ----------
        x_t : torch.Tensor
            Input at the current timestep, shape ``(B, input_size)``.
        state : list of (torch.Tensor or None)
            From ``init_state()`` or a previous ``step()``.

        Returns
        -------
        tuple
            The output, shape ``(B, d_model)``, and the updated state
            (a list of ``n_layers`` tensors, each ``(B, d_model)``).
        """
        x_t = self.in_proj(x_t)
        new_state = []
        for block, h_prev in zip(self.blocks, state):
            x_t, h = block.step(x_t, h_prev)
            new_state.append(h)
        return self.norm_out(x_t), new_state


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D_in, D_h = 4, 128, 32, 64

    m = MinGRU(D_in, D_h)
    x = torch.randn(B, T, D_in)

    # Parallel path
    h_par = m(x)

    # Sequential path
    h = None
    hs = []
    for t in range(T):
        h = m.step(x[:, t], h)
        hs.append(h)
    h_seq = torch.stack(hs, dim=1)

    err = (h_par - h_seq).abs().max().item()
    print(f"parallel vs sequential max abs diff: {err:.3e}")
    assert err < 1e-4, "parallel scan does not match sequential recurrence"

    # Gradients flow through the parallel path
    loss = m(x).sum()
    loss.backward()
    assert m.linear_z.weight.grad is not None
    print("gradcheck ok; param count:", sum(p.numel() for p in m.parameters()))

    # --- Stack: parallel vs streaming equivalence ---
    stack = MinGRUStack(D_in, d_model=D_h, n_layers=3).eval()
    with torch.no_grad():
        y_par = stack(x)

    state = stack.init_state()
    ys = []
    for t in range(T):
        y_t, state = stack.step(x[:, t], state)
        ys.append(y_t)
    y_seq = torch.stack(ys, dim=1)

    err = (y_par - y_seq).abs().max().item()
    print(f"stack parallel vs streaming max abs diff: {err:.3e}")
    assert err < 1e-4

    # --- Chunked forward equivalence (TBPTT-style state carry) ---
    with torch.no_grad():
        h_full = m(x)
        h_a = m(x[:, : T // 2])
        h_b = m(x[:, T // 2 :], h_0=h_a[:, -1:])
        h_chunked = torch.cat([h_a, h_b], dim=1)
    err = (h_full - h_chunked).abs().max().item()
    print(f"MinGRU chunked vs full max abs diff: {err:.3e}")
    assert err < 1e-4, "chunked forward with carried h_0 must match full forward"

    with torch.no_grad():
        y_a, carry = stack(x[:, : T // 2], return_state=True)
        y_b = stack(x[:, T // 2 :], state=carry)
        y_chunked = torch.cat([y_a, y_b], dim=1)
    err = (y_par - y_chunked).abs().max().item()
    print(f"stack chunked vs full max abs diff: {err:.3e}")
    assert err < 1e-4

    loss = MinGRUStack(D_in, D_h, 3)(x).sum()
    loss.backward()
    print("stack gradcheck ok; param count:", sum(p.numel() for p in stack.parameters()))

    # --- learnable_h0: grads must reach the init param FROM ZERO-INIT ---
    # (relu-based log_g had grad exactly 0 at x=0, silently deadening the
    # default-initialized parameter; this guards the fix.)
    xg = torch.zeros(3, requires_grad=True)
    log_g(xg).sum().backward()
    assert torch.allclose(xg.grad, torch.full_like(xg, 2.0)), (
        f"log_g grad at 0 should be 1/g(0)=2, got {xg.grad}"
    )

    m2 = MinGRU(D_in, D_h, learnable_h0=True)  # h0_pre at zero-init
    m2(x).sum().backward()
    assert m2.h0_pre.grad is not None and m2.h0_pre.grad.abs().sum() > 0, (
        "learnable_h0 is dead at zero-init"
    )
    print("learnable_h0 trains from zero-init: ok")

    with torch.no_grad():
        m2.h0_pre.normal_()  # also verify consistency off the default
    m2 = m2.eval()
    with torch.no_grad():
        h_par2 = m2(x)
        h = None
        hs = [h := m2.step(x[:, t], h) for t in range(T)]
    err = (h_par2 - torch.stack(hs, dim=1)).abs().max().item()
    print(f"learnable_h0 parallel vs streaming max abs diff: {err:.3e}")
    assert err < 1e-4

    # --- h_0 validation: underflowed zeros ok, negatives raise ---
    with torch.no_grad():
        h_carry = m(x)[:, -1:]
        h_carry[0, 0, 0] = 0.0  # simulate fp16/bf16 underflow of a valid state
        out = m(x, h_0=h_carry)
        assert torch.isfinite(out).all(), "zero-underflowed state must not produce inf/nan"
        try:
            m(x, h_0=h_carry - 1.0)  # negative: pre-activation-style misuse
            raise AssertionError("negative h_0 should have raised")
        except RuntimeError:
            pass
    print("h_0 validation: zeros clamped, negatives raise: ok")

    # --- SignedMinGRU (decoupled, default): parallel vs sequential, chunked carry ---
    ms = SignedMinGRU(D_in, D_h).eval()
    assert not ms.coupled, "default SignedMinGRU must be decoupled (coupled=False)"
    with torch.no_grad():
        h_par = ms(x)
        h = None
        hs = [h := ms.step(x[:, t], h) for t in range(T)]
    err = (h_par - torch.stack(hs, dim=1)).abs().max().item()
    print(f"signed (decoupled) parallel vs sequential max abs diff: {err:.3e}")
    assert err < 1e-4

    with torch.no_grad():
        h_a = ms(x[:, : T // 2])
        h_b = ms(x[:, T // 2 :], h_0=h_a[:, -1:])  # negative states are legal here
        assert (h_a[:, -1:] < 0).any(), "expected some negative signed states"
    err = (h_par - torch.cat([h_a, h_b], dim=1)).abs().max().item()
    print(f"signed (decoupled) chunked vs full max abs diff: {err:.3e}")
    assert err < 1e-4

    # --- SignedMinGRU(coupled=True): legacy parameterization regression ---
    msc = SignedMinGRU(D_in, D_h, coupled=True).eval()
    assert msc.coupled
    with torch.no_grad():
        h_par_c = msc(x)
        h = None
        hs_c = [h := msc.step(x[:, t], h) for t in range(T)]
    err = (h_par_c - torch.stack(hs_c, dim=1)).abs().max().item()
    print(f"signed (coupled) parallel vs sequential max abs diff: {err:.3e}")
    assert err < 1e-4

    with torch.no_grad():
        h_a_c = msc(x[:, : T // 2])
        h_b_c = msc(x[:, T // 2 :], h_0=h_a_c[:, -1:])
    err = (h_par_c - torch.cat([h_a_c, h_b_c], dim=1)).abs().max().item()
    print(f"signed (coupled) chunked vs full max abs diff: {err:.3e}")
    assert err < 1e-4

    # Construction determinism: identical seeds -> bit-identical coupled
    # outputs. NOTE: this checks the class against itself (guards against
    # RNG-consuming or order-dependent construction changes); bit-exactness
    # vs the PRE-promotion class is guaranteed by the unchanged parameter
    # construction order (linear_z, linear_h, linear_s) and the op-for-op
    # identical coupled _coeffs — verified against git history at promotion
    # time, not re-checkable in-repo without the old class.
    torch.manual_seed(123)
    msc_ref = SignedMinGRU(D_in, D_h, coupled=True)
    torch.manual_seed(123)
    msc_new = SignedMinGRU(D_in, D_h, coupled=True)
    with torch.no_grad():
        err = (msc_ref(x) - msc_new(x)).abs().max().item()
    print(f"signed (coupled) construction determinism max abs diff: {err:.3e}")
    assert err == 0.0, "identical seeds must give bit-identical coupled outputs"

    sstack = MinGRUStack(D_in, D_h, 3, mixer="signed").eval()
    with torch.no_grad():
        y_par = sstack(x)
        state = sstack.init_state()
        ys = []
        for t in range(T):
            y_t, state = sstack.step(x[:, t], state)
            ys.append(y_t)
    err = (y_par - torch.stack(ys, dim=1)).abs().max().item()
    print(f"signed stack parallel vs streaming max abs diff: {err:.3e}")
    assert err < 1e-4

    loss = MinGRUStack(D_in, D_h, 3, mixer="signed")(x).sum()
    loss.backward()
    print("signed stack gradcheck ok")

    # --- matrix_scan vs brute-force sequential recurrence (signed 2x2 coeffs) ---
    torch.manual_seed(0)
    Bm, Tm, Nm = 3, 17, 5
    Mm = torch.randn(Bm, Tm, Nm, 2, 2) * 0.5  # unconstrained signed entries
    bm = torch.randn(Bm, Tm, Nm, 2)
    h0m = torch.randn(Bm, Nm, 2)
    Am, Bcm = matrix_scan(Mm, bm)
    h_scan = torch.einsum("btnij,bnj->btni", Am, h0m) + Bcm
    h = h0m
    hs = []
    for t in range(Tm):
        h = torch.einsum("bnij,bnj->bni", Mm[:, t], h) + bm[:, t]
        hs.append(h)
    h_seq = torch.stack(hs, dim=1)
    err = (h_scan - h_seq).abs().max().item()
    print(f"matrix_scan vs sequential max abs diff: {err:.3e}")
    assert err < 1e-4, "matrix_scan does not match sequential recurrence"

    Mm.requires_grad_(True)
    Am, Bcm = matrix_scan(Mm, bm)
    Bcm.sum().backward()
    assert Mm.grad is not None and torch.isfinite(Mm.grad).all()
    print("matrix_scan gradcheck-lite ok")

    # --- RotationMinGRU: parallel vs sequential, chunked carry ---
    torch.manual_seed(0)
    mr = RotationMinGRU(D_in, D_h).eval()
    with torch.no_grad():
        h_par = mr(x)
        h = None
        hs = [h := mr.step(x[:, t], h) for t in range(T)]
    err = (h_par - torch.stack(hs, dim=1)).abs().max().item()
    print(f"rotation parallel vs sequential max abs diff: {err:.3e}")
    assert err < 1e-4

    with torch.no_grad():
        h_a = mr(x[:, : T // 2])
        h_b = mr(x[:, T // 2 :], h_0=h_a[:, -1:])
    err = (h_par - torch.cat([h_a, h_b], dim=1)).abs().max().item()
    print(f"rotation chunked vs full max abs diff: {err:.3e}")
    assert err < 1e-4

    # --- RotationMinGRU: snapped angles land exactly on the grid ---
    with torch.no_grad():
        M_used, _ = mr._coeffs(x)
    theta_used = torch.atan2(M_used[..., 1, 0], M_used[..., 0, 0])
    ratio = theta_used / mr.snap_step
    dev = (ratio - torch.round(ratio)).abs().max().item()
    print(f"rotation snapped angle grid deviation: {dev:.3e}")
    assert dev < 1e-4, "snapped angles must land exactly on the grid"

    # --- RotationMinGRU: gradients reach all four heads and h0 ---
    mr_grad = RotationMinGRU(D_in, D_h)
    mr_grad(x).sum().backward()
    for name, p in [
        ("linear_z", mr_grad.linear_z.weight),
        ("linear_h", mr_grad.linear_h.weight),
        ("linear_theta", mr_grad.linear_theta.weight),
        ("linear_u", mr_grad.linear_u.weight),
        ("h0", mr_grad.h0),
    ]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} received no gradient"
    print("rotation gradcheck ok: all four heads + h0 receive gradient")

    # --- RotationMinGRU: odd hidden_size raises ---
    try:
        RotationMinGRU(D_in, D_h + 1)
        raise AssertionError("odd hidden_size should have raised ValueError")
    except ValueError:
        pass
    print("rotation odd hidden_size raises: ok")

    # --- MinGRUStack: mixer="rotation" parallel vs streaming equivalence ---
    torch.manual_seed(0)
    rstack = MinGRUStack(D_in, D_h, 3, mixer="rotation").eval()
    with torch.no_grad():
        y_par_r = rstack(x)
        state = rstack.init_state()
        ys = []
        for t in range(T):
            y_t, state = rstack.step(x[:, t], state)
            ys.append(y_t)
    err = (y_par_r - torch.stack(ys, dim=1)).abs().max().item()
    print(f"rotation stack parallel vs streaming max abs diff: {err:.3e}")
    assert err < 1e-4

    # --- MinGRUStack: mixer="rotation" chunked vs full (state carry) ---
    with torch.no_grad():
        y_a_r, carry_r = rstack(x[:, : T // 2], return_state=True)
        y_b_r = rstack(x[:, T // 2 :], state=carry_r)
        y_chunked_r = torch.cat([y_a_r, y_b_r], dim=1)
    err = (y_par_r - y_chunked_r).abs().max().item()
    print(f"rotation stack chunked vs full max abs diff: {err:.3e}")
    assert err < 1e-4

    loss = MinGRUStack(D_in, D_h, 3, mixer="rotation")(x).sum()
    loss.backward()
    print("rotation stack gradcheck ok")

    # --- MinGRUBlock/MinGRUStack: unknown mixer raises ValueError ---
    try:
        MinGRUBlock(D_h, mixer="bogus")
        raise AssertionError("unknown mixer should have raised ValueError")
    except ValueError:
        pass
    try:
        MinGRUStack(D_in, D_h, 3, mixer="bogus")
        raise AssertionError("unknown mixer should have raised ValueError")
    except ValueError:
        pass
    print("unknown mixer raises ValueError: ok")

    # --- MinGRUStack: mixer_kwargs={"coupled": True} trains through the stack ---
    # (regression for the coupled=True path: Task 1 covered SignedMinGRU in
    # isolation but not coupled=True reached via the stack's mixer selector.)
    cstack = MinGRUStack(D_in, D_h, 3, mixer="signed", mixer_kwargs={"coupled": True})
    assert all(block.mingru.coupled for block in cstack.blocks)
    loss = cstack(x).sum()
    loss.backward()
    for i, block in enumerate(cstack.blocks):
        grad = block.mingru.linear_s.weight.grad
        assert grad is not None and grad.abs().sum() > 0, (
            f"block {i} linear_s received no gradient under mixer_kwargs={{'coupled': True}}"
        )
    print("stack mixer='signed', mixer_kwargs={'coupled': True} gradcheck ok")
