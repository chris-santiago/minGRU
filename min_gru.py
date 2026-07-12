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

`MinGRU` is kept atomic (one scan layer). `MinGRUBlock` wraps it in the
standard pre-norm residual template (LN -> minGRU -> residual, then
LN -> MLP -> residual), which supplies the inter-layer nonlinear mixing:
layer l's gates condition on layer l-1's hidden states, and the MLP
provides cross-channel interaction that the diagonal scan cannot.
`MinGRUStack` stacks N blocks and supports both parallel training-mode
forward and O(1)-memory streaming via step().
"""

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
        Passed through to the block's ``MinGRU``; see ``MinGRU``.
    """

    def __init__(
        self,
        d_model: int,
        mlp_expansion: int = 4,
        dropout: float = 0.0,
        learnable_h0: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mingru = MinGRU(d_model, d_model, learnable_h0=learnable_h0)
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
        Passed through to every block's ``MinGRU``; see ``MinGRU``.

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
    ):
        super().__init__()
        self.in_proj = (
            nn.Linear(input_size, d_model) if input_size != d_model else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            MinGRUBlock(d_model, mlp_expansion, dropout, learnable_h0)
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
