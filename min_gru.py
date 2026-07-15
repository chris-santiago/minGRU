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

`MinGRU`, `SignedMinGRU`, `RotationMinGRU`, and `GivensMinGRU` are each
kept atomic (one scan layer, one sequence-mixing mechanism).
`MinGRUBlock` wraps any one of them, chosen via
`mixer="log"|"signed"|"rotation"|"givens"` (plus `mixer_kwargs` for
per-mixer config such as `coupled=True`, a custom `snap` grid, or
`block_size`/`rounds`), in the standard pre-norm residual template
(LN -> mixer ->
residual, then LN -> MLP -> residual), which supplies the inter-layer
nonlinear mixing: layer l's gates condition on layer l-1's hidden
states, and the MLP provides cross-channel interaction that a diagonal
scan cannot. `MinGRUStack` stacks N blocks under a single `mixer`
selection and supports both parallel training-mode forward and
O(1)-memory streaming via step().
"""

import math
import warnings
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

Decay = Literal["fixed", "learnable"] | None


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


# --- Shared time-decay machinery -------------------------------------------
#
# Exponential time decay (``gamma = exp(-lambda * f(delta_t))``) is threaded
# through all three mixers below. The pieces are factored into small
# module-level functions rather than duplicated per class: the
# squeeze/clamp/log1p pipeline, the decay/delta_t pairing contract, and
# decay-parameter construction are byte-for-byte identical across
# ``MinGRU``, ``SignedMinGRU``, and ``RotationMinGRU``; only how ``gamma``
# is folded into that mixer's transition coefficients differs (log-space
# additive for ``MinGRU``, multiplicative for the other two), so that part
# stays in each class's own coefficient computation.


# Cap applied to +inf entries during delta_t sanitization (see
# _normalize_delta_t). Empirically, an unclamped +inf overflows MinGRU's
# log-space scan to NaN; 1e10 was verified (by the __main__ NaN/inf test)
# to stay finite end-to-end for all three mixers, at every decay_rate the
# self-test suite exercises.
_DELTA_T_POSINF_CAP = 1e10

# Constructor kwarg names that configure a mixer's decay behavior (see
# _init_decay). MinGRUStack's decay_layers="last" strips exactly these
# keys from mixer_kwargs for every block but the final one, giving
# last-layer-only decay.
_DECAY_MIXER_KWARGS = ("decay", "decay_rate", "log1p_delta")


def _normalize_delta_t(
    delta_t: torch.Tensor, canonical_ndim: int, log1p_delta: bool = False
) -> tuple[torch.Tensor, bool]:
    """Normalize a raw ``delta_t`` tensor into the shared decay pipeline's form.

    Implements the pipeline in order: squeeze a trailing singleton
    dimension down to ``canonical_ndim`` dims, then sanitize
    unconditionally and branch-free (``NaN`` -> 0 — ``NaN < 0`` is False,
    so a negative-only clamp would silently let it propagate; ``+inf`` ->
    ``_DELTA_T_POSINF_CAP``, a large finite value — an unbounded ``+inf``
    overflows the log-space scan to NaN; ``-inf`` and any remaining
    negative value -> 0 via the trailing clamp), then optionally apply
    ``log1p``.

    The sanitization itself never inspects tensor content on the host —
    it is unconditional and applies to every element — so it costs no
    CUDA sync, the same no-host-sync posture as ``MinGRU.forward``'s
    ``torch._assert_async`` h_0 check. The *warn-once courtesy message*
    is a separate concern: deciding whether to warn requires a boolean
    reduction over the tensor, which IS a host sync, so that inspection
    is performed ONLY when ``delta_t`` is already on CPU (free there).
    On CUDA, negative/NaN/infinite timestamps are silently sanitized
    with no warning — a deliberate throughput/courtesy tradeoff, not an
    oversight. Warn-once STATE is not owned here — a pure function has
    nowhere to persist it across calls — so this only reports whether
    the (CPU-only) inspection found invalid entries; the caller holds a
    ``_warned_negative_dt`` flag on its own module instance and decides
    whether to actually emit the warning.

    Parameters
    ----------
    delta_t : torch.Tensor
        Shape ``(..., )`` matching ``canonical_ndim`` dims exactly, or
        with one extra trailing size-1 dim to squeeze (e.g. ``forward``
        passes ``canonical_ndim=2`` for ``(B, T)`` or ``(B, T, 1)``;
        ``step`` passes ``canonical_ndim=1`` for ``(B,)`` or ``(B, 1)``).
    canonical_ndim : int
        The expected number of dims after squeezing.
    log1p_delta : bool, default=False
        If True, apply ``log1p`` after sanitizing.

    Returns
    -------
    tuple of (torch.Tensor, bool)
        The normalized ``delta_t`` (squeezed; NaN, +/-inf, and negative
        entries sanitized to finite non-negative values; ``log1p``
        applied if configured), and — CPU only — whether any negative,
        ``NaN``, or infinite entries were found in the raw input
        (always False on non-CPU devices, since that inspection is
        skipped there to avoid a host sync).

    Raises
    ------
    ValueError
        If ``delta_t``'s shape doesn't match ``canonical_ndim`` dims
        (optionally plus one trailing size-1 dim).
    """
    if delta_t.dim() == canonical_ndim + 1:
        if delta_t.size(-1) != 1:
            raise ValueError(
                f"delta_t with {delta_t.dim()} dims must have a trailing "
                f"size-1 dimension to squeeze (got shape "
                f"{tuple(delta_t.shape)})"
            )
        delta_t = delta_t.squeeze(-1)
    elif delta_t.dim() != canonical_ndim:
        raise ValueError(
            f"delta_t must have {canonical_ndim} dims (or "
            f"{canonical_ndim + 1} with a trailing size-1 dim to squeeze); "
            f"got shape {tuple(delta_t.shape)}"
        )
    has_invalid = False
    if delta_t.device.type == "cpu":
        has_invalid = bool(((delta_t < 0) | delta_t.isnan() | delta_t.isinf()).any())
    delta_t = delta_t.nan_to_num(nan=0.0, posinf=_DELTA_T_POSINF_CAP).clamp(min=0)
    if log1p_delta:
        delta_t = torch.log1p(delta_t)
    return delta_t, has_invalid


def _validate_delta_t_pairing(decay: Decay, delta_t: torch.Tensor | None) -> None:
    """Enforce the decay/``delta_t`` pairing contract.

    Parameters
    ----------
    decay : {"fixed", "learnable", None}
        The mixer's decay mode.
    delta_t : torch.Tensor or None
        The ``delta_t`` argument passed to this call.

    Raises
    ------
    ValueError
        If decay is enabled but ``delta_t`` is omitted, or ``delta_t``
        is supplied but decay is disabled.
    """
    if decay is not None and delta_t is None:
        raise ValueError(f"decay={decay!r} requires delta_t to be provided.")
    if decay is None and delta_t is not None:
        raise ValueError(
            "delta_t was provided but decay is disabled (decay=None); "
            "construct the mixer with decay='fixed' or decay='learnable' "
            "to use delta_t."
        )


def _warn_once_invalid_delta_t(module: nn.Module, has_invalid: bool) -> None:
    """Emit the invalid-``delta_t`` warning once per module instance.

    ``has_invalid`` is only ever True when ``_normalize_delta_t`` ran its
    (CPU-only) content inspection and found negative, ``NaN``, or
    infinite entries; it is always False on non-CPU devices, so this
    never warns on CUDA — the same no-host-sync posture as
    ``MinGRU.forward``'s ``torch._assert_async`` h_0 check.

    Parameters
    ----------
    module : nn.Module
        The calling mixer instance; the warned flag is stored as
        ``module._warned_negative_dt``.
    has_invalid : bool
        Whether negative, ``NaN``, or infinite entries were found by
        ``_normalize_delta_t``.
    """
    if has_invalid and not module._warned_negative_dt:
        warnings.warn(
            "delta_t contains negative, NaN, or infinite entries (likely "
            "out-of-order or corrupted timestamps); sanitizing to finite, "
            "non-negative values.",
            stacklevel=3,
        )
        module._warned_negative_dt = True


class DecayMixin:
    """Exponential time-decay contract shared by minGRU's diagonal and
    rotation mixers: ``gamma = exp(-lambda * f(delta_t))``.

    A subclass opts in via ``class Foo(DecayMixin, nn.Module)`` and
    calls ``self._init_decay(decay, decay_rate, log1p_delta,
    num_channels)`` as the LAST statement of its own ``__init__``,
    after every one of its own parameters/buffers is constructed. That
    ordering — combined with ``_init_decay`` never drawing RNG in any
    mode — is what keeps ``decay=None`` construction/state_dict layout
    and same-seed weight values bit-identical to the pre-decay module,
    and keeps ``decay="learnable"``'s extra ``rho`` parameter from
    perturbing the RNG stream consumed by the subclass's own
    ``nn.Linear`` layers (see ``_init_decay``).

    Attributes
    ----------
    decay : {"fixed", "learnable", None}
        The mixer's decay mode; set by ``_init_decay``.
    decay_rate : float
        Fixed value, or the learnable rate's init target; set by
        ``_init_decay``.
    log1p_delta : bool
        Whether ``delta_t`` is passed through ``log1p`` before scaling
        by ``lambda``; set by ``_init_decay``, read by
        ``_prepare_decay``.
    _warned_negative_dt : bool
        Whether the once-per-instance invalid-``delta_t`` warning (see
        ``_prepare_decay``) has already fired; set by ``_init_decay``,
        mutated by ``_warn_once_invalid_delta_t``.
    decay_rate_buf : torch.Tensor
        Registered by ``_init_decay`` ONLY when ``decay == "fixed"``:
        scalar buffer holding ``lambda`` directly.
    rho : nn.Parameter
        Registered by ``_init_decay`` ONLY when ``decay ==
        "learnable"``: one raw pre-softplus rate per channel,
        ``lambda = softplus(rho)``.

    This is the single place the attribute contract is declared; do
    not reorder attribute assignment or buffer/parameter registration
    inside ``_init_decay`` — every state_dict-order and bit-identity
    self-test in this module's ``__main__`` depends on these names and
    their construction order.
    """

    decay: Decay
    decay_rate: float
    log1p_delta: bool
    _warned_negative_dt: bool
    rho: nn.Parameter
    decay_rate_buf: torch.Tensor

    def _init_decay(
        self,
        decay: Decay,
        decay_rate: float,
        log1p_delta: bool,
        num_channels: int,
    ) -> None:
        """Validate and construct this mixer's decay parameters/buffers.

        Must be called AFTER all of the subclass's other parameters/
        buffers are constructed (last line of its ``__init__``); see
        the class docstring for why.

        Parameters
        ----------
        decay : {"fixed", "learnable", None}
            Decay mode.
        decay_rate : float
            Fixed value, or the learnable rate's init target
            (``softplus(rho) == decay_rate`` at construction).
        log1p_delta : bool
            Stored for use by ``_prepare_decay``/``_normalize_delta_t``
            at call time.
        num_channels : int
            Number of decay-rate channels: ``hidden_size`` for
            ``MinGRU``/``SignedMinGRU``, ``n_blocks`` for
            ``RotationMinGRU``.

        Raises
        ------
        ValueError
            If ``decay`` is not one of ``None``, ``"fixed"``,
            ``"learnable"``; or if decay is enabled and ``decay_rate <=
            0``. ``lambda`` must be strictly positive in both modes for
            ``gamma = exp(-lambda * f(delta_t)) in (0, 1]`` to hold for
            all ``delta_t >= 0`` (spec §7): a non-positive fixed
            ``decay_rate`` is used directly as ``lambda`` and would let
            ``gamma`` reach or exceed 1 (amplification); a non-positive
            ``decay_rate`` in learnable mode has no valid ``rho``
            (``math.log(math.expm1(x))`` is undefined for ``x <= 0``)
            and would otherwise surface as an opaque ``math domain
            error``.
        """
        if decay not in (None, "fixed", "learnable"):
            raise ValueError(
                f"decay must be None, 'fixed', or 'learnable' (got {decay!r})"
            )
        if decay is not None and decay_rate <= 0:
            raise ValueError(
                f"decay_rate must be > 0 when decay={decay!r} is enabled (got "
                f"{decay_rate!r}); lambda = decay_rate (fixed) or "
                "softplus(rho) == decay_rate at init (learnable) must be "
                "strictly positive so gamma = exp(-lambda * f(delta_t)) stays "
                "in (0, 1] and never amplifies."
            )
        self.decay = decay
        self.decay_rate = decay_rate
        self.log1p_delta = log1p_delta
        self._warned_negative_dt = False
        if decay == "fixed":
            self.register_buffer("decay_rate_buf", torch.full((), decay_rate))
        elif decay == "learnable":
            # softplus(rho) == decay_rate at init; math.log(math.expm1(.))
            # is a computed constant (no RNG), so this cannot perturb
            # bit-identity for the decay=None path or the RNG stream
            # other params consume.
            rho_init = math.log(math.expm1(decay_rate))
            self.rho = nn.Parameter(torch.full((num_channels,), rho_init))

    def _decay_lambda(self) -> torch.Tensor:
        """Per-channel decay rate ``lambda`` for a mixer with decay enabled.

        Returns
        -------
        torch.Tensor
            ``lambda``, always non-negative: the fixed scalar buffer, or
            ``softplus(rho)`` per channel (shape ``(num_channels,)``).
        """
        if self.decay == "fixed":
            return self.decay_rate_buf
        return F.softplus(self.rho)

    def _decay_gamma(self, dt: torch.Tensor) -> torch.Tensor:
        """Multiplicative decay factor ``gamma = exp(-lambda * dt)``.

        Shared by every mixer's multiplicative decay-application site
        (``MinGRU.step``, ``SignedMinGRU._coeffs``,
        ``RotationMinGRU._coeffs`` — previously byte-identical
        ``torch.exp(-_decay_lambda(self) * dt.unsqueeze(-1))`` lines,
        now this one method). ``MinGRU.forward``'s parallel path applies
        decay additively in log-space instead (``log_coeffs -
        lambda * dt``, no exp/log round-trip) and does not call this
        method.

        Parameters
        ----------
        dt : torch.Tensor
            Already-normalized ``delta_t`` (see ``_prepare_decay``).

        Returns
        -------
        torch.Tensor
            ``gamma``, with one trailing dim inserted into ``dt``'s
            shape so it broadcasts against a ``(..., num_channels)``
            (or ``(..., num_channels, 1, 1)``, after the caller's own
            further unsqueezing) transition coefficient.
        """
        return torch.exp(-self._decay_lambda() * dt.unsqueeze(-1))

    def _prepare_decay(
        self, delta_t: torch.Tensor | None, canonical_ndim: int
    ) -> torch.Tensor | None:
        """Validate, normalize, and warn for this mixer's ``delta_t`` input.

        Enforces the decay/``delta_t`` pairing contract (``ValueError``
        both directions via ``_validate_delta_t_pairing``), then —
        only when decay is enabled, i.e. ``delta_t is not None`` —
        normalizes it via ``_normalize_delta_t`` and emits the
        once-per-instance invalid-entries warning (CPU only; see
        ``_normalize_delta_t``). Shared by every mixer's
        ``forward``/``step`` so the pairing/normalization/warning
        logic cannot drift between classes or between the two call
        paths.

        Parameters
        ----------
        delta_t : torch.Tensor or None
            Raw ``delta_t``, or None if decay is disabled.
        canonical_ndim : int
            2 for ``forward`` (``(B, T)``), 1 for ``step`` (``(B,)``).

        Returns
        -------
        torch.Tensor or None
            Normalized ``delta_t``, or None (decay disabled —
            guaranteed by the pairing check to mean ``delta_t`` was
            also None).
        """
        _validate_delta_t_pairing(self.decay, delta_t)
        if delta_t is None:
            return None
        dt, has_invalid = _normalize_delta_t(delta_t, canonical_ndim, self.log1p_delta)
        _warn_once_invalid_delta_t(self, has_invalid)
        return dt


class MinGRU(DecayMixin, nn.Module):
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
    decay : {"fixed", "learnable", None}, default=None
        Exponential time decay of the carried state, driven by
        ``delta_t``: ``gamma = exp(-lambda * f(delta_t))`` multiplies
        the log-space transition coefficient (``log(1 - z)``) additively
        in log-space, i.e. ``log(gamma * (1 - z)) = log(1 - z) -
        lambda * f(delta_t)`` — no new exp/log round-trip. ``None``
        disables decay: ``delta_t`` must then be omitted, and behavior
        is bit-identical to the module without this feature.
        ``"fixed"``: ``lambda = decay_rate``, a scalar buffer uniform
        across channels. ``"learnable"``: ``lambda = softplus(rho)``,
        one ``rho`` per hidden channel, initialized so
        ``lambda == decay_rate`` at construction. ``delta_t = 0`` gives
        ``gamma = 1`` exactly (no decay), with no special-casing of
        ``t = 0``: callers pass ``delta_t = 0`` at a true sequence
        start, and chunked calls pass the real gap at the chunk
        boundary to match a full forward exactly.
    decay_rate : float, default=1.0
        Fixed decay rate, or the learnable rate's init target.
    log1p_delta : bool, default=False
        If True, ``delta_t`` is passed through ``log1p`` before scaling
        by ``lambda`` (compresses large gaps). See ``_normalize_delta_t``.
        Note: negative/NaN/inf ``delta_t`` entries are always sanitized
        to finite non-negative values; the courtesy warning about them
        fires on CPU only (on CUDA they are fixed silently — no host
        sync, same rationale as the ``h_0`` async validation).

    Raises
    ------
    ValueError
        If ``decay`` is not one of ``None``, ``"fixed"``, ``"learnable"``
        (at construction); or if ``decay`` is enabled without
        ``delta_t``, or ``delta_t`` is given without decay enabled (at
        call time, in ``forward``/``step``).

    Notes
    -----
    Shapes: ``forward`` maps ``x (B, T, input_size)`` with optional
    ``h_0 (B, 1, hidden_size)`` to ``(B, T, hidden_size)``; ``step``
    maps ``x_t (B, input_size)`` with optional ``h_prev
    (B, hidden_size)`` to ``(B, hidden_size)``. ``delta_t`` follows the
    same optionality: ``(B, T)`` or ``(B, T, 1)`` for ``forward``,
    ``(B,)`` or ``(B, 1)`` for ``step``. Invalid ``delta_t`` entries
    (negative/``NaN``/``inf``) are always sanitized to finite,
    non-negative values; the courtesy warning about them is CPU-only
    (silent on CUDA, avoiding a host sync) — the sanitizing clamp
    itself always applies, on every device.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        learnable_h0: bool = False,
        decay: Decay = None,
        decay_rate: float = 1.0,
        log1p_delta: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h0_pre: nn.Parameter | None = (
            nn.Parameter(torch.zeros(1, 1, hidden_size)) if learnable_h0 else None
        )
        self._init_decay(decay, decay_rate, log1p_delta, hidden_size)

    def _default_log_h0(self, B: int, x: torch.Tensor) -> torch.Tensor:
        if self.h0_pre is not None:
            return log_g(self.h0_pre).expand(B, 1, self.hidden_size)
        return torch.full(
            (B, 1, self.hidden_size), 0.5, dtype=x.dtype, device=x.device
        ).log()

    def forward(
        self,
        x: torch.Tensor,
        h_0: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        delta_t : torch.Tensor, optional
            Time gaps preceding each event, shape ``(B, T)`` or
            ``(B, T, 1)``; required iff ``decay`` is enabled (see class
            docstring). ``delta_t[:, t]`` is the gap preceding event
            ``t``; ``delta_t = 0`` means no decay at that step.

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
        ValueError
            If ``decay`` is enabled without ``delta_t``, or ``delta_t``
            is given without decay enabled.
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
        dt = self._prepare_decay(delta_t, canonical_ndim=2)
        if dt is not None:
            # log(gamma * (1 - z)) = log(1 - z) - lambda * f(dt); additive
            # in log-space, so no new exp/log round-trip. (The one
            # decay-application site that stays log-additive rather than
            # going through DecayMixin._decay_gamma's multiplicative form.)
            log_coeffs = log_coeffs - self._decay_lambda() * dt.unsqueeze(-1)
        log_tilde_h = log_g(self.linear_h(x))
        return parallel_scan_log(
            log_coeffs,
            torch.cat([log_h_0, log_z + log_tilde_h], dim=1),
        )

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        delta_t : torch.Tensor, optional
            Time gap preceding this event, shape ``(B,)`` or ``(B, 1)``;
            required iff ``decay`` is enabled.

        Returns
        -------
        torch.Tensor
            New hidden state ``h_t``, shape ``(B, hidden_size)``.

        Raises
        ------
        ValueError
            If ``decay`` is enabled without ``delta_t``, or ``delta_t``
            is given without decay enabled.
        """
        if h_prev is None:
            if self.h0_pre is not None:
                h_prev = g(self.h0_pre)[:, 0].expand(x_t.size(0), self.hidden_size)
            else:
                h_prev = x_t.new_full((x_t.size(0), self.hidden_size), 0.5)
        z = torch.sigmoid(self.linear_z(x_t))
        h_tilde = g(self.linear_h(x_t))
        a = 1 - z
        dt = self._prepare_decay(delta_t, canonical_ndim=1)
        if dt is not None:
            a = self._decay_gamma(dt) * a
        return a * h_prev + z * h_tilde

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
    choice: the overhead is negligible at the short training lengths
    this repo targets (probes train at T=64), and no stable
    ``torch.associative_scan`` primitive exists to lean on. Revisit for
    long-sequence training.
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
    the short training lengths this repo targets (probes train at
    T=64); revisit for long-sequence training.
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


def matrix_affine_scan(
    A: torch.Tensor, Bm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Associative scan for ``H_t = A_t @ H_{t-1} + B_t``, k x k transitions.

    The k x k generalization of ``matrix_scan``: transitions are square
    ``k x k`` matrices (instead of ``matrix_scan``'s fixed 2x2 blocks)
    and injections are rectangular ``k x v`` matrix-valued states
    (instead of ``matrix_scan``'s 2-vectors, i.e. ``v = 1``). Both scan
    the same affine monoid
    ``(A1, B1) o (A2, B2) = (A2 @ A1, A2 @ B1 + B2)`` by the same
    Hillis-Steele doubling scheme; ``matrix_scan`` is exactly the
    ``k = 2, v = 1`` special case, and ``matrix_affine_scan`` is what
    the ``k > 2`` block-rotation mixer (``GivensMinGRU``, with the state
    as a ``v = 1`` column) and matrix-valued delta-rule states run on.
    O(T log T) work, O(log T) depth, pure torch ops, differentiable.
    Composition order is ``A_current @ A_earlier`` — matrix
    multiplication does not commute — so, as in ``matrix_scan``, the
    running product depends on order.

    Parameters
    ----------
    A : torch.Tensor
        Shape ``(B, T, n, k, k)``. Per-block square transitions ``A_t``;
        stability expects each ``A_t``'s spectral norm ``<= 1`` but this
        is not enforced.
    Bm : torch.Tensor
        Shape ``(B, T, n, k, v)``. Additive injections ``B_t``, viewed
        as ``n`` ``k x v`` matrices per timestep (``v = 1`` recovers the
        vector state of ``matrix_scan``).

    Returns
    -------
    tuple of torch.Tensor
        ``(Abar, Bbar)``, shapes ``(B, T, n, k, k)`` and
        ``(B, T, n, k, v)``, each aligned to ``t``, where
        ``H_t = Abar_t @ H_0 + Bbar_t`` with ``Abar_t`` the running
        matrix product of ``A`` and ``Bbar_t`` the ``H_0 = 0`` solution
        of the recurrence.

    Notes
    -----
    Same simplicity-over-efficiency tradeoff as ``matrix_scan`` /
    ``linear_scan``: Hillis-Steele is work-inefficient (O(T log T) vs
    O(T)) and retains O(log T) full ``(B, T, n, k, k)`` tensors for
    autograd. Fine at the short training lengths this repo targets
    (probes train at T=64); revisit for long-sequence training. Kept as
    a distinct helper beside ``matrix_scan`` (rather than a
    generalization of it) so the recorded 2x2 rotation evidence's
    floating-point path is preserved byte-for-byte.
    """
    Ab, Bb = A, Bm
    offset, T = 1, A.size(1)
    while offset < T:
        newA = torch.einsum("btnij,btnjk->btnik", Ab[:, offset:], Ab[:, :-offset])
        newB = (
            torch.einsum("btnij,btnjv->btniv", Ab[:, offset:], Bb[:, :-offset])
            + Bb[:, offset:]
        )
        Ab = torch.cat([Ab[:, :offset], newA], dim=1)
        Bb = torch.cat([Bb[:, :offset], newB], dim=1)
        offset *= 2
    return Ab, Bb


class SignedMinGRU(DecayMixin, nn.Module):
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
    decay : {"fixed", "learnable", None}, default=None
        Exponential time decay of the carried state:
        ``a_decayed = gamma * a`` with
        ``gamma = exp(-lambda * f(delta_t))`` — magnitude decays, sign
        is preserved, applied identically for both ``coupled`` values.
        ``None`` disables decay: ``delta_t`` must then be omitted, and
        behavior (including the ``coupled=True`` bit-exact reproduction
        guarantee) is bit-identical to the module without this feature.
        ``"fixed"``: ``lambda = decay_rate``, a scalar buffer.
        ``"learnable"``: ``lambda = softplus(rho)``, one ``rho`` per
        hidden channel, initialized so ``lambda == decay_rate`` at
        construction. ``delta_t = 0`` gives ``gamma = 1`` exactly, with
        no special-casing of ``t = 0``.
    decay_rate : float, default=1.0
        Fixed decay rate, or the learnable rate's init target.
    log1p_delta : bool, default=False
        If True, ``delta_t`` is passed through ``log1p`` before scaling
        by ``lambda``. See ``_normalize_delta_t``.
        Note: negative/NaN/inf ``delta_t`` entries are always sanitized
        to finite non-negative values; the courtesy warning about them
        fires on CPU only (on CUDA they are fixed silently — no host
        sync, same rationale as the ``h_0`` async validation).

    Raises
    ------
    ValueError
        If ``decay`` is not one of ``None``, ``"fixed"``, ``"learnable"``
        (at construction); or if ``decay`` is enabled without
        ``delta_t``, or ``delta_t`` is given without decay enabled (at
        call time, in ``forward``/``step``).

    Notes
    -----
    Parameter count is 3 linear heads vs. MinGRU's 2 (the extra sign
    head) — account for this in parameter-matched comparisons. Invalid
    ``delta_t`` entries (negative/``NaN``/``inf``) are always sanitized
    to finite, non-negative values; the courtesy warning about them is
    CPU-only (silent on CUDA, avoiding a host sync) — the sanitizing
    clamp itself always applies, on every device.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        learnable_h0: bool = False,
        coupled: bool = False,
        decay: Decay = None,
        decay_rate: float = 1.0,
        log1p_delta: bool = False,
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
        self._init_decay(decay, decay_rate, log1p_delta, hidden_size)

    def _coeffs(
        self, x: torch.Tensor, dt: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transition coefficient and injection; shared by forward/step.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., input_size)``.
        dt : torch.Tensor, optional
            Already-normalized ``delta_t`` (see ``_prepare_decay``),
            shape matching ``x``'s leading dims. None if decay is
            disabled.

        Returns
        -------
        tuple of torch.Tensor
            ``a`` (possibly decayed) and ``b``, each ``(..., hidden_size)``.
        """
        z = torch.sigmoid(self.linear_z(x))
        tanh_s = torch.tanh(self.linear_s(x))
        a = (1 - z) * tanh_s if self.coupled else tanh_s
        b = z * self.linear_h(x)
        if dt is not None:
            a = self._decay_gamma(dt) * a
        return a, b

    def forward(
        self,
        x: torch.Tensor,
        h_0: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Parallel forward over a full sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape ``(B, T, input_size)``.
        h_0 : torch.Tensor, optional
            Initial hidden state, shape ``(B, 1, hidden_size)``. Any
            real values. Defaults to zeros (or the learned initial
            state if ``learnable_h0``).
        delta_t : torch.Tensor, optional
            Time gaps preceding each event, shape ``(B, T)`` or
            ``(B, T, 1)``; required iff ``decay`` is enabled.

        Returns
        -------
        torch.Tensor
            All hidden states ``h_1..h_T``, shape
            ``(B, T, hidden_size)``.

        Raises
        ------
        ValueError
            If ``decay`` is enabled without ``delta_t``, or ``delta_t``
            is given without decay enabled.
        """
        if h_0 is None:
            h_0 = (
                self.h0.expand(x.size(0), 1, self.hidden_size)
                if self.h0 is not None
                else x.new_zeros(x.size(0), 1, self.hidden_size)
            )
        dt = self._prepare_decay(delta_t, canonical_ndim=2)
        a, b = self._coeffs(x, dt)
        A, Bc = linear_scan(a, b)
        return A * h_0 + Bc

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single recurrent step; same real-state convention as forward().

        Computed from the same ``_coeffs`` helper ``forward()`` uses, so
        decayed and non-decayed dynamics cannot drift between the two
        call paths.

        Parameters
        ----------
        x_t : torch.Tensor
            Input at the current timestep, shape ``(B, input_size)``.
        h_prev : torch.Tensor, optional
            Previous hidden state, shape ``(B, hidden_size)``. Defaults
            to zeros (or the learned initial state).
        delta_t : torch.Tensor, optional
            Time gap preceding this event, shape ``(B,)`` or ``(B, 1)``;
            required iff ``decay`` is enabled.

        Returns
        -------
        torch.Tensor
            New hidden state, shape ``(B, hidden_size)``.

        Raises
        ------
        ValueError
            If ``decay`` is enabled without ``delta_t``, or ``delta_t``
            is given without decay enabled.
        """
        if h_prev is None:
            h_prev = (
                self.h0[:, 0].expand(x_t.size(0), self.hidden_size)
                if self.h0 is not None
                else x_t.new_zeros(x_t.size(0), self.hidden_size)
            )
        dt = self._prepare_decay(delta_t, canonical_ndim=1)
        a, b = self._coeffs(x_t, dt)
        return a * h_prev + b

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"coupled={self.coupled}"
        )


class RotationMinGRU(DecayMixin, nn.Module):
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

    Depth: single-layer use is the recorded baseline; stacks holding
    rotation blocks (with or without other mixer types) are validated
    at L=2 on the S3 probe under the best-val@128 protocol, deeper is
    untested. The straight-through discontinuity can compound across
    rotation layers, so multi-rotation stacks warn at construction —
    see ``MinGRUStack`` and the README's Rotation variant section.

    Training protocol: the exact automaton is reachable but is NOT a
    stable attractor of standard training — runs wander in and out of
    it during optimization. The validated protocol is best-checkpoint
    selection by validation accuracy at a length LONGER than the
    training length (e.g. T=128 when training at T=64; not one of the
    eventual test lengths), evaluated over the full step budget instead
    of early-stopping, plus a retry-on-flag rule: a best validation
    score at that checkpoint length below 1.0 flags the run as failed.
    The flag is one-directional: a sub-1.0 score reliably marks a bad
    run, but a perfect checkpoint score does not guarantee exact length
    generalization (in the recorded evidence every seed passed the
    checkpoint yet most still decayed at the longest lengths). See
    ``experiments/SUMMARY.md`` for the full protocol, per-seed success
    rate, and mechanism verification.

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
    decay : {"fixed", "learnable", None}, default=None
        Exponential time decay of the carried state: ``M_decayed =
        gamma * M`` per block, with ``gamma = exp(-lambda *
        f(delta_t))``. Scalar decay commutes with the rotation/
        reflection group action, so the composed transition's ANGLE is
        unaffected (stays exactly on the snap grid) — only amplitude
        fades. ``None`` disables decay: ``delta_t`` must then be
        omitted, and behavior is bit-identical to the module without
        this feature. ``"fixed"``: ``lambda = decay_rate``, a scalar
        buffer. ``"learnable"``: ``lambda = softplus(rho)``, one
        ``rho`` per block (``n_blocks`` channels), initialized so
        ``lambda == decay_rate`` at construction. ``delta_t = 0`` gives
        ``gamma = 1`` exactly, with no special-casing of ``t = 0``.
    decay_rate : float, default=1.0
        Fixed decay rate, or the learnable rate's init target.
    log1p_delta : bool, default=False
        If True, ``delta_t`` is passed through ``log1p`` before scaling
        by ``lambda``. See ``_normalize_delta_t``.
        Note: negative/NaN/inf ``delta_t`` entries are always sanitized
        to finite non-negative values; the courtesy warning about them
        fires on CPU only (on CUDA they are fixed silently — no host
        sync, same rationale as the ``h_0`` async validation).

    Raises
    ------
    ValueError
        If ``hidden_size`` is odd, or ``decay`` is not one of ``None``,
        ``"fixed"``, ``"learnable"`` (at construction); or if ``decay``
        is enabled without ``delta_t``, or ``delta_t`` is given without
        decay enabled (at call time, in ``forward``/``step``).

    Notes
    -----
    Parameter count is 4 linear heads (z, h, theta, u) vs.
    ``SignedMinGRU``'s 3 — account for this in parameter-matched
    comparisons. Same forward/step shapes as the module's other
    mixers: ``forward`` maps ``x (B, T, input_size)`` with optional
    ``h_0 (B, 1, hidden_size)`` to ``(B, T, hidden_size)``; ``step``
    maps ``x_t (B, input_size)`` with optional ``h_prev
    (B, hidden_size)`` to ``(B, hidden_size)``. ``delta_t`` follows the
    same optionality: ``(B, T)`` or ``(B, T, 1)`` for ``forward``,
    ``(B,)`` or ``(B, 1)`` for ``step``. Invalid ``delta_t`` entries
    (negative/``NaN``/``inf``) are always sanitized to finite,
    non-negative values; the courtesy warning about them is CPU-only
    (silent on CUDA, avoiding a host sync) — the sanitizing clamp
    itself always applies, on every device.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        snap: tuple[int, ...] | None = (2, 3, 4, 6),
        decay: Decay = None,
        decay_rate: float = 1.0,
        log1p_delta: bool = False,
    ):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError(
                f"RotationMinGRU requires an even hidden_size (got {hidden_size}); "
                "state is n = hidden_size / 2 planar 2D blocks."
            )
        if snap is not None and (
            len(snap) == 0 or any(k < 1 for k in snap)
        ):
            raise ValueError(
                f"snap must be None or a non-empty tuple of positive ints (got {snap!r})"
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
        self._init_decay(decay, decay_rate, log1p_delta, self.n_blocks)

    def _coeffs(
        self, x: torch.Tensor, dt: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        dt : torch.Tensor, optional
            Already-normalized ``delta_t`` (see ``_prepare_decay``),
            shape matching ``x``'s leading dims. None if decay is
            disabled.

        Returns
        -------
        tuple of torch.Tensor
            ``M``, shape ``(..., n_blocks, 2, 2)``: the (possibly
            snapped, possibly decayed) transition
            ``gamma * R(theta_t) @ diag(1, tanh(u_t))``. Decay scales
            the whole block matrix by a positive scalar per
            ``(..., n_blocks)``, so ``atan2`` of the matrix entries
            recovers the exact (snapped) angle unchanged — only
            amplitude is affected. ``b``, shape ``(..., n_blocks, 2)``:
            the injection ``z_t * Linear_h(x_t)`` (never decayed),
            reshaped into ``n_blocks`` 2-vectors.
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
        if dt is not None:
            M = self._decay_gamma(dt).unsqueeze(-1).unsqueeze(-1) * M

        z = torch.sigmoid(self.linear_z(x))
        b = z * self.linear_h(x)
        b = b.reshape(*b.shape[:-1], self.n_blocks, 2)
        return M, b

    def forward(
        self,
        x: torch.Tensor,
        h_0: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Parallel forward over a full sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape ``(B, T, input_size)``.
        h_0 : torch.Tensor, optional
            Initial hidden state, shape ``(B, 1, hidden_size)``. Any
            real values (reshaped into ``n_blocks`` 2-vectors
            internally). Defaults to the learned initial state.
        delta_t : torch.Tensor, optional
            Time gaps preceding each event, shape ``(B, T)`` or
            ``(B, T, 1)``; required iff ``decay`` is enabled.

        Returns
        -------
        torch.Tensor
            All hidden states ``h_1..h_T``, shape
            ``(B, T, hidden_size)``.

        Raises
        ------
        ValueError
            If ``decay`` is enabled without ``delta_t``, or ``delta_t``
            is given without decay enabled.
        """
        B, T, _ = x.shape
        if h_0 is None:
            h_0 = self.h0.expand(B, 1, self.hidden_size)
        h0_blocks = h_0.reshape(B, self.n_blocks, 2)
        dt = self._prepare_decay(delta_t, canonical_ndim=2)
        M, b = self._coeffs(x, dt)
        A, Bc = matrix_scan(M, b)
        h = torch.einsum("btnij,bnj->btni", A, h0_blocks) + Bc
        return h.reshape(B, T, self.hidden_size)

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        delta_t : torch.Tensor, optional
            Time gap preceding this event, shape ``(B,)`` or ``(B, 1)``;
            required iff ``decay`` is enabled.

        Returns
        -------
        torch.Tensor
            New hidden state, shape ``(B, hidden_size)``.

        Raises
        ------
        ValueError
            If ``decay`` is enabled without ``delta_t``, or ``delta_t``
            is given without decay enabled.
        """
        B = x_t.size(0)
        if h_prev is None:
            h_prev = self.h0.expand(B, 1, self.hidden_size)[:, 0]
        h_prev_blocks = h_prev.reshape(B, self.n_blocks, 2)
        dt = self._prepare_decay(delta_t, canonical_ndim=1)
        M, b = self._coeffs(x_t, dt)
        h = torch.einsum("bnij,bnj->bni", M, h_prev_blocks) + b
        return h.reshape(B, self.hidden_size)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"snap={self.snap}"
        )


class GivensMinGRU(DecayMixin, nn.Module):
    """minGRU variant with k-dim block-rotation transitions (non-diagonal).

    The ``k > 2`` generalization of ``RotationMinGRU``'s ``snap=None``
    regime. State is ``n_blocks = hidden_size / block_size`` independent
    blocks of ``k = block_size`` dims (64 state elements total at the
    repo's ``d_model = 512``: the same per-token state as every other
    promoted mixer). Per block and token the transition is a product of
    ``rounds`` layers of Givens rotations on fixed brick-wall planes,

        M_t = G_{rounds-1} @ ... @ G_1 @ G_0,
        h_t = M_t @ h_{t-1} + b_t,     b_t = sigmoid(z_t) * Linear_h(x_t)

    (``h_{t-1}``, ``h_t``, ``b_t`` viewed as ``k``-vectors per block).
    Each round ``r`` rotates a set of disjoint coordinate planes by
    input-dependent angles from one linear head: even rounds pair planes
    ``(0,1),(2,3),...``; odd rounds the staggered
    ``(1,2),(3,4),...,(k-1,0)``. Every ``G_r`` is orthogonal with
    determinant ``+1`` by construction, so ``M_t`` is exactly
    special-orthogonal — continuous, with no straight-through snap.
    The brick-wall plane layout is the standard rectangular mesh from
    the orthogonal/unitary-RNN literature (EUNN: Jing et al. 2017,
    arXiv:1612.05231; mesh design: Clements et al. 2016) built from
    Givens plane rotations (Golub & Van Loan, *Matrix Computations*);
    the departure here is input-dependent angles per token.

    Where ``RotationMinGRU`` manufactures attractors at exact group
    elements by snapping 2x2 angles, ``GivensMinGRU`` deliberately does
    not: each per-token map lives on a ``rounds * block_size / 2``-angle
    submanifold of ``SO(k)`` — 12 of ``SO(8)``'s 28 dimensions at the
    defaults; products of enough Givens rotations (about ``k - 1``
    brick-wall rounds) generate all of ``SO(k)``, so three rounds is a
    deliberate budget rather than full per-token coverage — giving
    richer non-abelian per-token maps at matched state capacity, at the
    cost of having no attractor at any particular
    transition — angles drift under length generalization exactly as
    ``RotationMinGRU(snap=None)``'s do. Like the other matrix-transition
    mixers it runs on a non-commutative parallel scan
    (``matrix_affine_scan``, the ``k``-dim generalization of
    ``matrix_scan``), so it can represent non-abelian state tracking a
    diagonal (commutative) scan provably cannot.

    Depth: unlike ``RotationMinGRU``, stacks holding multiple Givens
    blocks construct WITHOUT a warning — the multi-rotation
    ``UserWarning`` names straight-through snap compounding, which the
    continuous (unsnapped) Givens transition does not have (see
    ``MinGRUStack``).

    ``h_0`` is an unconditional learnable parameter (no ``learnable_h0``
    flag, same convention and rationale as ``RotationMinGRU``):
    ``h_0 = 0`` has no orbit under the group action, so a zero initial
    state cannot demonstrate state tracking; a random nonzero learned
    vector avoids that failure mode.

    Parameters
    ----------
    input_size : int
        Dimensionality of the inputs ``x_t``.
    hidden_size : int
        Dimensionality of the hidden states ``h_t``; must be an integer
        multiple of ``block_size`` (``hidden_size = n_blocks *
        block_size``).
    bias : bool, default=True
        Whether the three linear maps carry bias terms.
    block_size : int, default=8
        Per-block state dimension ``k``; must be even and divide
        ``hidden_size``. The default keeps the standard 64-element
        per-token state at the repo's ``d_model``.
    rounds : int, default=3
        Number of brick-wall Givens layers composed per transition; must
        be at least 1. Full ``SO(k)`` coverage would take
        ``k * (k - 1) / 2`` rotations (about ``k - 1`` brick-wall
        rounds); the default three rounds span a 12-angle submanifold of
        ``SO(8)`` and are the evidence-validated budget.
    decay : {"fixed", "learnable", None}, default=None
        Exponential time decay of the carried state: ``M_decayed =
        gamma * M`` per block, with ``gamma = exp(-lambda *
        f(delta_t))``. A scalar ``gamma`` commutes with the orthogonal
        block action, so the composed transition's rotation is
        unaffected — only amplitude fades — identical decay semantics to
        ``RotationMinGRU``. ``None`` disables decay: ``delta_t`` must
        then be omitted, and the forward computation is bit-identical to
        the module without this feature (no ``gamma = 1`` multiply on
        the disabled-decay path). ``"fixed"``: ``lambda = decay_rate``,
        a scalar buffer. ``"learnable"``: ``lambda = softplus(rho)``,
        one ``rho`` per block (``n_blocks`` channels), initialized so
        ``lambda == decay_rate`` at construction. ``delta_t = 0`` gives
        ``gamma = 1`` exactly.
    decay_rate : float, default=1.0
        Fixed decay rate, or the learnable rate's init target.
    log1p_delta : bool, default=False
        If True, ``delta_t`` is passed through ``log1p`` before scaling
        by ``lambda``. See ``_normalize_delta_t``.

    Raises
    ------
    ValueError
        If ``hidden_size`` is not a multiple of ``block_size``, or
        ``block_size`` is odd (at construction); or if ``decay`` is not
        one of ``None``, ``"fixed"``, ``"learnable"`` (at construction);
        or if ``decay`` is enabled without ``delta_t``, or ``delta_t``
        is given without decay enabled (at call time, in
        ``forward``/``step``).

    Notes
    -----
    Same forward/step shapes as the module's other mixers: ``forward``
    maps ``x (B, T, input_size)`` with optional ``h_0 (B, 1,
    hidden_size)`` to ``(B, T, hidden_size)``; ``step`` maps ``x_t (B,
    input_size)`` with optional ``h_prev (B, hidden_size)`` to ``(B,
    hidden_size)``. ``delta_t`` follows the same optionality and shapes
    as ``RotationMinGRU`` (``(B, T)`` or ``(B, T, 1)`` for ``forward``;
    ``(B,)`` or ``(B, 1)`` for ``step``), with the identical
    decay/``delta_t`` pairing contract (``ValueError`` both directions,
    see ``_validate_delta_t_pairing``) and CPU-only invalid-entry
    warning. Parameter count is 3 linear heads (theta, z, h): the angle
    head emits ``n_blocks * rounds * (block_size / 2)`` angles.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        block_size: int = 8,
        rounds: int = 3,
        decay: Decay = None,
        decay_rate: float = 1.0,
        log1p_delta: bool = False,
    ):
        super().__init__()
        if block_size % 2 != 0:
            raise ValueError(
                f"GivensMinGRU requires an even block_size (got {block_size}); "
                "brick-wall Givens rounds pair the k state dims into k/2 "
                "disjoint planes."
            )
        if hidden_size % block_size != 0:
            raise ValueError(
                f"GivensMinGRU requires hidden_size ({hidden_size}) divisible by "
                f"block_size ({block_size}); state is n_blocks = hidden_size / "
                "block_size blocks of block_size dims."
            )
        if rounds < 1:
            raise ValueError(
                f"GivensMinGRU requires rounds >= 1 (got {rounds}); with no "
                "Givens layers every transition is the identity and the mixer "
                "cannot mix state."
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = block_size
        self.n_blocks = hidden_size // block_size
        self.rounds = rounds
        half = block_size // 2
        # RNG-draw order (the bit-identity seam vs the lab evidence
        # generator): linear_theta, linear_z, linear_h, h0, then the
        # angle-plane buffers (no RNG), then decay params last via
        # _init_decay. Do not reorder the three Linear heads or h0.
        self.linear_theta = nn.Linear(input_size, self.n_blocks * rounds * half, bias=bias)
        self.linear_z = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h0 = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.5)
        for r in range(rounds):
            start = r % 2
            i_idx = torch.arange(start, block_size, 2) % block_size
            j_idx = (i_idx + 1) % block_size
            self.register_buffer(f"_pi{r}", i_idx)
            self.register_buffer(f"_pj{r}", j_idx)
        self._init_decay(decay, decay_rate, log1p_delta, self.n_blocks)

    def _coeffs(
        self, x: torch.Tensor, dt: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-block transition matrix and injection; shared by forward/step.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., input_size)`` — ``forward`` passes
            ``(B, T, input_size)``, ``step`` passes ``(B, input_size)``;
            both work unchanged since ``nn.Linear`` and the elementwise
            ops here broadcast uniformly over leading dims, so this
            single helper serves both call paths and they cannot drift
            apart.
        dt : torch.Tensor, optional
            Already-normalized ``delta_t`` (see ``_prepare_decay``),
            shape matching ``x``'s leading dims. None if decay is
            disabled.

        Returns
        -------
        tuple of torch.Tensor
            ``M``, shape ``(..., n_blocks, k, k)``: the (possibly
            decayed) special-orthogonal transition, the product of
            ``rounds`` brick-wall Givens layers. Decay scales the whole
            block matrix by a positive scalar per ``(..., n_blocks)``,
            leaving the rotation unchanged — only amplitude is affected.
            ``b``, shape ``(..., n_blocks, k)``: the injection
            ``sigmoid(z_t) * Linear_h(x_t)`` (never decayed), reshaped
            into ``n_blocks`` ``k``-vectors.
        """
        lead = x.shape[:-1]
        half = self.k // 2
        theta = self.linear_theta(x).view(*lead, self.n_blocks, self.rounds, half)
        eye = torch.eye(self.k, dtype=x.dtype, device=x.device)
        M = eye.expand(*lead, self.n_blocks, self.k, self.k).clone()
        for r in range(self.rounds):
            i_idx = getattr(self, f"_pi{r}")
            j_idx = getattr(self, f"_pj{r}")
            cos = torch.cos(theta[..., r, :]).unsqueeze(-1)  # (..., n, half, 1)
            sin = torch.sin(theta[..., r, :]).unsqueeze(-1)
            rows_i = M[..., i_idx, :]
            rows_j = M[..., j_idx, :]
            M = M.clone()
            M[..., i_idx, :] = cos * rows_i - sin * rows_j
            M[..., j_idx, :] = sin * rows_i + cos * rows_j
        if dt is not None:
            M = self._decay_gamma(dt).unsqueeze(-1).unsqueeze(-1) * M
        z = torch.sigmoid(self.linear_z(x))
        b = (z * self.linear_h(x)).reshape(*lead, self.n_blocks, self.k)
        return M, b

    def forward(
        self,
        x: torch.Tensor,
        h_0: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Parallel forward over a full sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape ``(B, T, input_size)``.
        h_0 : torch.Tensor, optional
            Initial hidden state, shape ``(B, 1, hidden_size)`` (reshaped
            into ``n_blocks`` ``k``-vectors internally). Defaults to the
            learned initial state.
        delta_t : torch.Tensor, optional
            Time gaps preceding each event, shape ``(B, T)`` or
            ``(B, T, 1)``; required iff ``decay`` is enabled.

        Returns
        -------
        torch.Tensor
            All hidden states ``h_1..h_T``, shape
            ``(B, T, hidden_size)``.

        Raises
        ------
        ValueError
            If ``decay`` is enabled without ``delta_t``, or ``delta_t``
            is given without decay enabled.
        """
        B, T, _ = x.shape
        if h_0 is None:
            h_0 = self.h0.expand(B, 1, self.hidden_size)
        h0_blocks = h_0.reshape(B, self.n_blocks, self.k)
        dt = self._prepare_decay(delta_t, canonical_ndim=2)
        M, b = self._coeffs(x, dt)
        Abar, Bbar = matrix_affine_scan(M, b.unsqueeze(-1))
        h = torch.einsum("btnij,bnj->btni", Abar, h0_blocks) + Bbar.squeeze(-1)
        return h.reshape(B, T, self.hidden_size)

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single recurrent step; same real-state convention as forward().

        Computed from the same ``_coeffs`` helper ``forward()`` uses
        (applied per-step instead of over the full sequence), so the two
        paths cannot drift apart — mirrors ``RotationMinGRU``.

        Parameters
        ----------
        x_t : torch.Tensor
            Input at the current timestep, shape ``(B, input_size)``.
        h_prev : torch.Tensor, optional
            Previous hidden state, shape ``(B, hidden_size)``. Defaults
            to the learned initial state.
        delta_t : torch.Tensor, optional
            Time gap preceding this event, shape ``(B,)`` or ``(B, 1)``;
            required iff ``decay`` is enabled.

        Returns
        -------
        torch.Tensor
            New hidden state, shape ``(B, hidden_size)``.

        Raises
        ------
        ValueError
            If ``decay`` is enabled without ``delta_t``, or ``delta_t``
            is given without decay enabled.
        """
        B = x_t.size(0)
        if h_prev is None:
            h_prev = self.h0.expand(B, 1, self.hidden_size)[:, 0]
        h_prev_blocks = h_prev.reshape(B, self.n_blocks, self.k)
        dt = self._prepare_decay(delta_t, canonical_ndim=1)
        M, b = self._coeffs(x_t, dt)
        h = torch.einsum("bnij,bnj->bni", M, h_prev_blocks) + b
        return h.reshape(B, self.hidden_size)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"block_size={self.k}, rounds={self.rounds}"
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
        by ``"rotation"`` or ``"givens"``: ``RotationMinGRU``'s and
        ``GivensMinGRU``'s ``h_0`` is an intrinsic learned parameter
        with no ``learnable_h0`` flag, so this argument is silently
        unused for those mixers.
    mixer : {"log", "signed", "rotation", "givens"}, default="log"
        Selects the sequence mixer: ``MinGRU`` (log-space parallel
        scan), ``SignedMinGRU`` (signed diagonal transitions),
        ``RotationMinGRU`` (2x2 block rotations), or ``GivensMinGRU``
        (k-dim block rotations built from brick-wall Givens rounds). Any
        other value raises ``ValueError``.
    mixer_kwargs : dict, optional
        Extra constructor kwargs forwarded to the selected mixer class
        (e.g. ``{"coupled": True}`` for ``"signed"``,
        ``{"snap": (2, 3, 5)}`` for ``"rotation"``, or
        ``{"block_size": 8, "rounds": 3}`` for ``"givens"``); pass
        ``decay``, ``decay_rate``, ``log1p_delta`` here to enable time
        decay (see
        ``MinGRU``/``SignedMinGRU``/``RotationMinGRU``/``GivensMinGRU``).

    Notes
    -----
    ``delta_t`` threads through ``forward``/``step`` straight to the
    mixer, unchanged; the block performs no decay-mode validation of
    its own. The mixer's own decay/``delta_t`` pairing contract
    (``ValueError`` both directions, see ``_validate_delta_t_pairing``)
    is what fires whether the block is used standalone or inside a
    ``MinGRUStack``.
    """

    # name -> (class, accepts_learnable_h0). RotationMinGRU's and
    # GivensMinGRU's h_0 are intrinsic (see their class docstrings), so
    # the flag is not forwarded to them.
    _MIXER_CLASSES: dict[str, tuple[type[nn.Module], bool]] = {
        "log": (MinGRU, True),
        "signed": (SignedMinGRU, True),
        "rotation": (RotationMinGRU, False),
        "givens": (GivensMinGRU, False),
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
        mixer_cls, accepts_h0 = self._MIXER_CLASSES[mixer]
        if accepts_h0:
            mixer_kwargs["learnable_h0"] = learnable_h0
        self.mingru = mixer_cls(d_model, d_model, **mixer_kwargs)
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
        self,
        x: torch.Tensor,
        h_0: torch.Tensor | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parallel forward over a full sequence.

        Always returns ``(output, state)``, matching ``step()`` and the
        ``nn.GRU`` convention; discard the state when not carrying it.

        Parameters
        ----------
        x : torch.Tensor
            Residual-stream input, shape ``(B, T, d_model)``.
        h_0 : torch.Tensor, optional
            This block's real minGRU state carried from a previous
            chunk, shape ``(B, 1, d_model)``; see ``MinGRU.forward``.
        delta_t : torch.Tensor, optional
            Time gaps preceding each event, shape ``(B, T)`` or
            ``(B, T, 1)``; forwarded to this block's mixer unchanged.
            Required iff the mixer's ``decay`` is enabled — the mixer's
            own pairing contract raises ``ValueError`` otherwise (see
            ``_validate_delta_t_pairing``); this method adds no
            validation of its own.

        Returns
        -------
        tuple of torch.Tensor
            The output ``(B, T, d_model)`` and the block's final minGRU
            state ``(B, 1, d_model)`` for the next chunk.

        Raises
        ------
        ValueError
            If the mixer's ``decay`` is enabled without ``delta_t``, or
            ``delta_t`` is given without decay enabled.
        """
        h_seq = self.mingru(self.norm1(x), h_0, delta_t)
        x = x + self.drop(h_seq)
        if self.mlp is not None:
            x = x + self.mlp(self.norm2(x))
        return x, h_seq[:, -1:]

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | None,
        delta_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single streaming step.

        Parameters
        ----------
        x_t : torch.Tensor
            Residual-stream input at time ``t``, shape ``(B, d_model)``.
        h_prev : torch.Tensor or None
            This block's minGRU state from ``t-1``, shape
            ``(B, d_model)``; None at the first step.
        delta_t : torch.Tensor, optional
            Time gap preceding this event, shape ``(B,)`` or ``(B, 1)``;
            forwarded to this block's mixer unchanged. Same
            required-iff-decay-enabled contract as ``forward``.

        Returns
        -------
        tuple of torch.Tensor
            The block output and the new minGRU state, each
            ``(B, d_model)``.

        Raises
        ------
        ValueError
            If the mixer's ``decay`` is enabled without ``delta_t``, or
            ``delta_t`` is given without decay enabled.
        """
        h = self.mingru.step(self.norm1(x_t), h_prev, delta_t)
        x_t = x_t + h
        if self.mlp is not None:
            x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, h


def _resolve_stack_mixer_spec(
    mixer: str | list[str], mixer_kwargs: dict | None, n_layers: int
) -> tuple[list[str], dict[str, dict | None]]:
    """Normalize ``MinGRUStack``'s ``mixer``/``mixer_kwargs`` pair.

    The kwargs schema is decided entirely by the *type* of ``mixer``:
    a ``str`` always means a flat ``mixer_kwargs`` dict (one type, used
    by every block); a ``list[str]`` always means ``mixer_kwargs`` is
    ``None`` or a dict keyed by mixer type name. ``mixer_kwargs`` is
    never inspected to guess which schema was intended -- a dict whose
    shape doesn't match the schema implied by ``mixer``'s type raises
    ``ValueError`` describing both schemas, rather than being silently
    reinterpreted.

    Parameters
    ----------
    mixer : str or list of str
        See ``MinGRUStack.__init__``.
    mixer_kwargs : dict, optional
        See ``MinGRUStack.__init__``.
    n_layers : int
        Expected length of ``mixer`` when it is a list.

    Returns
    -------
    tuple of (list of str, dict)
        ``mixer_list``: one mixer name per block, length ``n_layers``.
        ``kwargs_by_type``: each mixer type name present in
        ``mixer_list`` mapped to its resolved flat kwargs dict (or
        ``None``).

    Raises
    ------
    ValueError
        If ``mixer`` is a list with length != ``n_layers``; if a list
        entry is not a valid mixer name; if ``mixer`` is a str and
        ``mixer_kwargs`` looks type-keyed (a key matches a mixer type
        name); if ``mixer`` is a list and ``mixer_kwargs`` has a key
        that is not a valid mixer name or names a type absent from
        ``mixer``; if ``mixer`` is neither a str nor a list.

    Notes
    -----
    An unknown ``str`` mixer name is deliberately NOT validated here --
    that check is left to ``MinGRUBlock``'s own constructor, exactly as
    before this function existed, so the string-mixer construction path
    raises from the same call site, in the same position relative to
    ``in_proj`` construction, as prior versions.
    """
    valid_names = MinGRUBlock._MIXER_CLASSES
    if isinstance(mixer, str):
        if mixer_kwargs:
            type_keyed_hits = sorted(k for k in mixer_kwargs if k in valid_names)
            if type_keyed_hits:
                raise ValueError(
                    f"mixer={mixer!r} is a single mixer name, so mixer_kwargs "
                    "must be a flat dict of that mixer's constructor kwargs "
                    "(e.g. {'coupled': True}); got key(s) matching a mixer "
                    f"type name instead ({type_keyed_hits!r}), which is the "
                    "schema for a list mixer (mixer_kwargs keyed by type, "
                    "e.g. {'signed': {...}, 'rotation': {...}})."
                )
        return [mixer] * n_layers, {mixer: mixer_kwargs}

    if isinstance(mixer, list):
        if len(mixer) != n_layers:
            raise ValueError(
                f"mixer list length ({len(mixer)}) must equal n_layers "
                f"({n_layers})"
            )
        unknown = sorted({name for name in mixer if name not in valid_names})
        if unknown:
            raise ValueError(
                f"unknown mixer name(s) {unknown!r} in mixer list; expected "
                f"one of {sorted(valid_names)}"
            )
        types_in_stack = set(mixer)
        if mixer_kwargs is None:
            kwargs_by_type: dict[str, dict | None] = {
                name: None for name in types_in_stack
            }
        else:
            bad_keys = sorted(
                k
                for k in mixer_kwargs
                if k not in valid_names or k not in types_in_stack
            )
            if bad_keys:
                raise ValueError(
                    "mixer_kwargs keys must name a mixer type present in "
                    f"mixer={mixer!r} (one of {sorted(types_in_stack)}); got "
                    f"unexpected key(s) {bad_keys!r}. For a list mixer, "
                    "mixer_kwargs is None or a dict keyed by mixer type name "
                    "(e.g. {'signed': {...}}), not a flat kwargs dict."
                )
            kwargs_by_type = {name: mixer_kwargs.get(name) for name in types_in_stack}
        return list(mixer), kwargs_by_type

    raise ValueError(f"mixer must be a str or list[str] (got {type(mixer).__name__})")


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
        ``"rotation"`` and ``"givens"`` (see ``MinGRUBlock``).
    mixer : str or list of str, default="log"
        Sequence mixer for the blocks; one of ``{"log", "signed",
        "rotation", "givens"}`` (see ``MinGRUBlock``). A single ``str`` uses that
        mixer for every block, bit-identical to prior versions: same
        construction order, same RNG consumption, same state_dict
        keys. Unknown ``str`` values raise ``ValueError`` (from
        ``MinGRUBlock``, unchanged). A ``list[str]`` of length
        ``n_layers`` gives one mixer name per block, in order, so a
        single ``"rotation"`` block can live inside a deeper stack of
        ``"signed"``/``"log"`` blocks; a length mismatch or an unknown
        name anywhere in the list raises ``ValueError`` naming the
        valid set. More than one ``"rotation"`` entry emits exactly one
        ``UserWarning`` per construction (see Notes) and then proceeds;
        a single ``"rotation"`` entry in a mixed stack does not warn.
    mixer_kwargs : dict, optional
        Schema is decided by the *type* of ``mixer``, never by
        inspecting ``mixer_kwargs`` itself:

        - ``mixer: str`` -> the current flat dict, forwarded as-is to
          every block's mixer constructor (e.g. ``{"coupled": True}``
          for ``"signed"``).
        - ``mixer: list[str]`` -> ``None``, or a dict keyed by mixer
          type name, e.g. ``{"signed": {"coupled": True}, "rotation":
          {"snap": (2, 3, 6)}}``; each value is that type's flat
          kwargs dict, applied to every block of that type (two blocks
          of the same type share one config -- no per-block-index
          overrides). A key that is not a valid mixer name, or names a
          type absent from ``mixer``, raises ``ValueError``.

        A flat dict alongside a list ``mixer``, or a type-keyed dict
        alongside a str ``mixer``, raises ``ValueError`` describing
        both schemas.
    decay_layers : {"all", "last"}, default="all"
        Which blocks receive the decay-related keys
        (``decay``, ``decay_rate``, ``log1p_delta``) from each block's
        RESOLVED ``mixer_kwargs`` (the type-specific dict when
        ``mixer`` is a list). ``"all"`` (default, uniform decay)
        applies them to every block unchanged. ``"last"`` strips those
        three keys, by position, from every block except the final one
        (position ``n_layers - 1``) -- whatever that block's mixer
        type. Any other value raises ``ValueError`` at construction. If
        a block's resolved kwargs carry no decay keys, ``"last"`` is a
        harmless no-op for it (nothing to strip).

        Trap in mixed stacks: ``decay_layers`` is purely positional, so
        under ``"last"`` the final block keeps its decay kwargs
        whatever type it is -- if that happens to be the ``"rotation"``
        block, decay lands there and every ``"signed"``/``"log"`` block
        is stripped, regardless of where you put the decay keys in
        ``mixer_kwargs``. Prefer per-type ``mixer_kwargs`` (put decay
        keys only under the type(s) you want decayed) to place decay
        deliberately in a mixed stack; reserve ``decay_layers="last"``
        for homogeneous (single-``str``-mixer) stacks.

    Notes
    -----
    Shapes: ``forward`` maps ``(B, T, input_size)`` to
    ``(B, T, d_model)``; ``step`` maps ``(B, input_size)`` and a state
    (a list of ``n_layers`` tensors of shape ``(B, d_model)``) to the
    output ``(B, d_model)`` and the updated state -- uniform across
    mixer types, so mixed stacks use the same ``forward``/``step``
    contract as homogeneous ones. Both also accept an optional
    ``delta_t`` (same shapes as the mixers' ``delta_t``): it is routed
    to a given block only if that block's mixer has decay enabled
    (checked via the mixer's own ``decay`` attribute), so a
    ``decay_layers="last"`` stack silently gives ``delta_t`` only to
    its final block. If NO block in the stack has decay enabled,
    passing ``delta_t`` raises ``ValueError`` (the mode-error rule);
    if at least one block decays, that block's own pairing contract
    raises if ``delta_t`` was left out.

    More than one ``"rotation"`` block in ``mixer`` triggers exactly
    one ``UserWarning`` per construction: the straight-through snap
    discontinuity can compound across rotation layers, and multi-
    rotation stacks are validated only at L=2 on the S3 probe (deeper
    is untested — see the README's Rotation variant section, "Depth:
    L=2 is validated"); construction proceeds regardless. A single
    ``"rotation"`` block in a mixed stack does not warn. Multiple
    ``"givens"`` blocks do NOT warn: the warning is specific to the
    straight-through snap discontinuity, and ``GivensMinGRU`` is
    continuous (unsnapped), so it has no such discontinuity to compound.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_layers: int,
        mlp_expansion: int = 4,
        dropout: float = 0.0,
        learnable_h0: bool = False,
        mixer: str | list[str] = "log",
        mixer_kwargs: dict | None = None,
        decay_layers: str = "all",
    ):
        super().__init__()
        if decay_layers not in ("all", "last"):
            raise ValueError(
                f"decay_layers must be 'all' or 'last' (got {decay_layers!r})"
            )
        self.decay_layers = decay_layers
        self.in_proj = (
            nn.Linear(input_size, d_model) if input_size != d_model else nn.Identity()
        )
        # Normalize mixer/mixer_kwargs to a per-block mixer list and a
        # kwargs dict keyed by mixer type name (a str mixer collapses to
        # a single-entry list/dict, so the rest of __init__ is generic
        # over both forms). See _resolve_stack_mixer_spec for the
        # ValueError cases (length mismatch, unknown name, schema
        # mismatch between mixer's type and mixer_kwargs's shape).
        mixer_list, kwargs_by_type = _resolve_stack_mixer_spec(
            mixer, mixer_kwargs, n_layers
        )
        # "last": non-final blocks get their resolved kwargs with decay
        # keys stripped, by position, whatever that block's mixer type;
        # the final block keeps its resolved kwargs unchanged. "all"
        # (default) passes every block's resolved kwargs unchanged,
        # exactly matching pre-decay_layers / pre-list-mixer
        # construction order and values for non-decay, string-mixer
        # configs.
        stripped_by_type = {
            name: (
                {k: v for k, v in kw.items() if k not in _DECAY_MIXER_KWARGS}
                if decay_layers == "last" and kw
                else kw
            )
            for name, kw in kwargs_by_type.items()
        }
        n_rotation_blocks = mixer_list.count("rotation")
        if n_rotation_blocks > 1:
            warnings.warn(
                "stack contains more than one 'rotation' block "
                f"({n_rotation_blocks} of {n_layers}): the straight-through "
                "snap discontinuity can compound across rotation layers; "
                "multi-rotation stacks are validated only at L=2 on the S3 "
                "probe, deeper is untested (see the README's Rotation "
                "variant section, 'Depth: L=2 is validated'); proceeding "
                "anyway.",
                stacklevel=2,
            )
        blocks = []
        for i in range(n_layers):
            is_last_block = i == n_layers - 1
            name = mixer_list[i]
            block_kwargs = (
                kwargs_by_type[name] if is_last_block else stripped_by_type[name]
            )
            blocks.append(
                MinGRUBlock(
                    d_model, mlp_expansion, dropout, learnable_h0, name, block_kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(d_model)

    def _check_delta_t_has_decay(self, delta_t: torch.Tensor | None) -> None:
        """Raise the mode-error ``ValueError`` if no block in this stack decays.

        Shared by ``forward``/``step``. When at least one block's mixer
        has decay enabled, that block's own pairing contract (via
        ``_validate_delta_t_pairing`` inside its mixer) is what raises
        for a missing ``delta_t`` — this check only covers the case
        where no mixer in the stack would ever see ``delta_t`` to
        validate it against.

        Parameters
        ----------
        delta_t : torch.Tensor or None
            The ``delta_t`` argument passed to ``forward``/``step``.

        Raises
        ------
        ValueError
            If ``delta_t`` is not None but no block's mixer has decay
            enabled.
        """
        if not any(block.mingru.decay is not None for block in self.blocks):
            _validate_delta_t_pairing(None, delta_t)

    def forward(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor | None] | None = None,
        delta_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Parallel training-mode forward.

        Always returns ``(output, state)``, matching ``step()`` and the
        ``nn.GRU`` convention; discard the state when not carrying it.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape ``(B, T, input_size)``.
        state : list of (torch.Tensor or None), optional
            Per-block minGRU states from a previous chunk — as returned
            by a prior ``forward()``, or from streaming ``step()`` after
            unsqueezing each entry to ``(B, 1, d_model)``. For TBPTT,
            detach the returned state before feeding it to the next
            chunk.
        delta_t : torch.Tensor, optional
            Time gaps preceding each event, shape ``(B, T)`` or
            ``(B, T, 1)``. Routed only to blocks whose mixer has decay
            enabled (see class Notes); required iff at least one block
            decays, and rejected with ``ValueError`` if no block decays.

        Returns
        -------
        tuple
            The output ``(B, T, d_model)`` and a list of ``n_layers``
            per-block final states, each ``(B, 1, d_model)``.

        Raises
        ------
        ValueError
            If ``delta_t`` is given but no block's mixer has decay
            enabled, or a decayed block's mixer requires ``delta_t``
            and it was omitted.
        """
        if state is None:
            state = self.init_state()
        self._check_delta_t_has_decay(delta_t)
        x = self.in_proj(x)
        new_state = []
        for block, h_0 in zip(self.blocks, state):
            block_dt = delta_t if block.mingru.decay is not None else None
            x, h_last = block(x, h_0, block_dt)
            new_state.append(h_last)
        return self.norm_out(x), new_state

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
        self,
        x_t: torch.Tensor,
        state: list[torch.Tensor | None],
        delta_t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Streaming step.

        Total cached state is ``n_layers * d_model`` per sample.

        Parameters
        ----------
        x_t : torch.Tensor
            Input at the current timestep, shape ``(B, input_size)``.
        state : list of (torch.Tensor or None)
            From ``init_state()`` or a previous ``step()``.
        delta_t : torch.Tensor, optional
            Time gap preceding this event, shape ``(B,)`` or ``(B, 1)``.
            Same per-block routing and ``ValueError`` contract as
            ``forward``.

        Returns
        -------
        tuple
            The output, shape ``(B, d_model)``, and the updated state
            (a list of ``n_layers`` tensors, each ``(B, d_model)``).

        Raises
        ------
        ValueError
            If ``delta_t`` is given but no block's mixer has decay
            enabled, or a decayed block's mixer requires ``delta_t``
            and it was omitted.
        """
        self._check_delta_t_has_decay(delta_t)
        x_t = self.in_proj(x_t)
        new_state = []
        for block, h_prev in zip(self.blocks, state):
            block_dt = delta_t if block.mingru.decay is not None else None
            x_t, h = block.step(x_t, h_prev, block_dt)
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
        y_par, _ = stack(x)

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
        y_a, carry = stack(x[:, : T // 2])
        y_b, _ = stack(x[:, T // 2 :], state=carry)
        y_chunked = torch.cat([y_a, y_b], dim=1)
    err = (y_par - y_chunked).abs().max().item()
    print(f"stack chunked vs full max abs diff: {err:.3e}")
    assert err < 1e-4

    loss = MinGRUStack(D_in, D_h, 3)(x)[0].sum()
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
        y_par, _ = sstack(x)
        state = sstack.init_state()
        ys = []
        for t in range(T):
            y_t, state = sstack.step(x[:, t], state)
            ys.append(y_t)
    err = (y_par - torch.stack(ys, dim=1)).abs().max().item()
    print(f"signed stack parallel vs streaming max abs diff: {err:.3e}")
    assert err < 1e-4

    loss = MinGRUStack(D_in, D_h, 3, mixer="signed")(x)[0].sum()
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
        y_par_r, _ = rstack(x)
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
        y_a_r, carry_r = rstack(x[:, : T // 2])
        y_b_r, _ = rstack(x[:, T // 2 :], state=carry_r)
        y_chunked_r = torch.cat([y_a_r, y_b_r], dim=1)
    err = (y_par_r - y_chunked_r).abs().max().item()
    print(f"rotation stack chunked vs full max abs diff: {err:.3e}")
    assert err < 1e-4

    loss = MinGRUStack(D_in, D_h, 3, mixer="rotation")(x)[0].sum()
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
    loss = cstack(x)[0].sum()
    loss.backward()
    for i, block in enumerate(cstack.blocks):
        grad = block.mingru.linear_s.weight.grad
        assert grad is not None and grad.abs().sum() > 0, (
            f"block {i} linear_s received no gradient under mixer_kwargs={{'coupled': True}}"
        )
    print("stack mixer='signed', mixer_kwargs={'coupled': True} gradcheck ok")

    # =====================================================================
    # Task 1 (time-aware decay): per-mixer self-tests (spec section 9.1)
    # =====================================================================
    import subprocess
    import types
    import warnings as _warnings

    # Commit immediately before the time-decay work started (verified above,
    # via `git log`/`git diff main --stat`, to be HEAD == main at review
    # time): the true pre-extension min_gru.py, loaded from git history so
    # the decay=None bit-identity check below is a real regression against
    # the actual prior module, not a self-consistency check of the new
    # module against itself.
    _PRE_EXTENSION_COMMIT = "4a0f8c85b9cca0d11b605400d6bb183dc079e935"

    def _load_pre_extension_module() -> types.ModuleType:
        """Load the pre-extension min_gru.py from git history.

        Raises
        ------
        subprocess.CalledProcessError
            If `git show` fails (e.g. the commit isn't reachable — a
            shallow clone that doesn't include it).
        FileNotFoundError, OSError
            If `git` itself isn't available, or the repo/.git isn't
            present (e.g. a vendored copy of this single file with no
            surrounding git checkout).
        """
        src = subprocess.run(
            ["git", "show", f"{_PRE_EXTENSION_COMMIT}:min_gru.py"],
            cwd=__import__("pathlib").Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        mod = types.ModuleType("_min_gru_pre_extension")
        exec(compile(src, "<min_gru_pre_extension>", "exec"), mod.__dict__)
        return mod

    # This module is meant to be usable as a standalone vendored file (no
    # surrounding git checkout required, e.g. a tarball export or shallow
    # clone), so the git-history regression below is best-effort: on any
    # git/subprocess failure, skip only the four bit-identity checks loudly
    # and let the rest of the suite run.
    try:
        _pre = _load_pre_extension_module()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as _git_err:
        _pre = None
        print(
            "pre-extension bit-identity check SKIPPED: git history unavailable "
            f"({type(_git_err).__name__}: {_git_err}) — run from a full clone "
            "to exercise it"
        )

    def _check_bit_identical_vs_pre_extension(
        cls_name: str, ctor_kwargs: dict, D_in: int, D_h: int, seed: int, x: torch.Tensor
    ) -> None:
        """decay=None must reproduce the pre-extension class bit-for-bit."""
        old_cls = getattr(_pre, cls_name)
        new_cls = globals()[cls_name]
        torch.manual_seed(seed)
        m_old = old_cls(D_in, D_h, **ctor_kwargs)
        torch.manual_seed(seed)
        m_new = new_cls(D_in, D_h, decay=None, **ctor_kwargs)
        with torch.no_grad():
            err = (m_old(x) - m_new(x)).abs().max().item()
        assert err == 0.0, (
            f"{cls_name}{ctor_kwargs}: decay=None must be bit-identical to the "
            f"pre-extension module (diff {err})"
        )
        print(f"{cls_name}{ctor_kwargs} decay=None bit-identical to pre-extension: ok")

    if _pre is not None:
        _check_bit_identical_vs_pre_extension("MinGRU", {}, D_in, D_h, seed=101, x=x)
        _check_bit_identical_vs_pre_extension(
            "SignedMinGRU", {"coupled": False}, D_in, D_h, seed=102, x=x
        )
        _check_bit_identical_vs_pre_extension(
            "SignedMinGRU", {"coupled": True}, D_in, D_h, seed=103, x=x
        )
        _check_bit_identical_vs_pre_extension("RotationMinGRU", {}, D_in, D_h, seed=104, x=x)

        # Task 2: MinGRUStack's new delta_t/decay_layers plumbing must not
        # perturb construction RNG order or output for a decay-off,
        # decay_layers="all" (default) stack — the pre-extension
        # MinGRUStack lacks both parameters entirely.
        torch.manual_seed(105)
        stack_old = _pre.MinGRUStack(D_in, D_h, 3, mixer="signed")
        torch.manual_seed(105)
        stack_new = MinGRUStack(D_in, D_h, 3, mixer="signed")
        with torch.no_grad():
            err = (stack_old(x)[0] - stack_new(x)[0]).abs().max().item()
        assert err == 0.0, (
            "MinGRUStack decay-off, decay_layers='all' must be bit-identical "
            f"to the pre-extension module (diff {err})"
        )
        print("MinGRUStack decay-off decay_layers='all' bit-identical to pre-extension: ok")

    def _check_decay_suite(
        cls_name: str, ctor_kwargs: dict, D_in: int, D_h: int, B: int, T: int, seed: int
    ) -> nn.Module:
        """Per-mixer x {fixed, learnable}: delta_t=0, parallel-vs-step, chunked-vs-full.

        Returns the last constructed (learnable-mode) instance, for any
        further mixer-specific probing (e.g. rotation's angle-grid check).
        """
        cls = globals()[cls_name]
        m_decay = None
        for mode in ("fixed", "learnable"):
            torch.manual_seed(seed)
            x_local = torch.randn(B, T, D_in)
            torch.manual_seed(seed)
            m_none = cls(D_in, D_h, decay=None, **ctor_kwargs).eval()
            torch.manual_seed(seed)
            m_decay = cls(D_in, D_h, decay=mode, decay_rate=1.0, **ctor_kwargs).eval()

            # delta_t = 0 -> gamma = 1 exactly: matches the no-decay path
            # on identical weights, bit-for-bit.
            with torch.no_grad():
                out_none = m_none(x_local)
                out_zero = m_decay(x_local, delta_t=torch.zeros(B, T))
            err = (out_none - out_zero).abs().max().item()
            print(f"{cls_name} ({mode}) delta_t=0 vs no-decay max abs diff: {err:.3e}")
            assert err == 0.0, f"{cls_name} ({mode}): delta_t=0 must give gamma=1 exactly"

            # Parallel forward vs iterated step, random positive delta_t.
            torch.manual_seed(seed + 1)
            dt = torch.rand(B, T) * 3.0 + 1e-2  # strictly positive gaps
            with torch.no_grad():
                h_par = m_decay(x_local, delta_t=dt)
                h = None
                hs = []
                for t in range(T):
                    h = m_decay.step(x_local[:, t], h, delta_t=dt[:, t])
                    hs.append(h)
                h_seq = torch.stack(hs, dim=1)
            err = (h_par - h_seq).abs().max().item()
            print(f"{cls_name} ({mode}) parallel vs step (decay) max abs diff: {err:.3e}")
            assert err < 1e-4

            # Chunked vs full, with a real (nonzero) gap at the chunk boundary.
            Th = T // 2
            assert dt[:, Th].min().item() > 0, "test setup: boundary gap must be nonzero"
            with torch.no_grad():
                h_full = m_decay(x_local, delta_t=dt)
                h_a = m_decay(x_local[:, :Th], delta_t=dt[:, :Th])
                h_b = m_decay(x_local[:, Th:], h_0=h_a[:, -1:], delta_t=dt[:, Th:])
                h_chunked = torch.cat([h_a, h_b], dim=1)
            err = (h_full - h_chunked).abs().max().item()
            print(
                f"{cls_name} ({mode}) chunked vs full (nonzero boundary gap) "
                f"max abs diff: {err:.3e}"
            )
            assert err < 1e-4

            if mode == "learnable":
                loss = m_decay(x_local, delta_t=dt).sum()
                loss.backward()
                assert m_decay.rho.grad is not None and m_decay.rho.grad.abs().sum() > 0, (
                    f"{cls_name}: rho received no gradient"
                )
                print(f"{cls_name} (learnable): rho gradcheck ok")
        return m_decay

    _check_decay_suite("MinGRU", {}, D_in, D_h, B=4, T=128, seed=201)
    _check_decay_suite("SignedMinGRU", {"coupled": False}, D_in, D_h, B=4, T=128, seed=202)
    _check_decay_suite("SignedMinGRU", {"coupled": True}, D_in, D_h, B=4, T=128, seed=203)
    mr_decay_learnable = _check_decay_suite(
        "RotationMinGRU", {}, D_in, D_h, B=4, T=128, seed=204
    )

    # --- RotationMinGRU: snapped angles remain exact grid multiples under active decay ---
    torch.manual_seed(205)
    x_rot = torch.randn(4, 128, D_in)
    dt_rot = torch.rand(4, 128) * 2.0 + 1e-2
    with torch.no_grad():
        M_used, _ = mr_decay_learnable._coeffs(x_rot, dt_rot)
    theta_used = torch.atan2(M_used[..., 1, 0], M_used[..., 0, 0])
    ratio = theta_used / mr_decay_learnable.snap_step
    dev = (ratio - torch.round(ratio)).abs().max().item()
    print(f"rotation (decay=learnable) snapped angle grid deviation under decay: {dev:.3e}")
    assert dev < 1e-4, "snapped angles must remain exact grid multiples under active decay"

    # --- decay/delta_t pairing: ValueError both directions, all three mixers ---
    def _check_delta_t_value_errors(cls_name: str, ctor_kwargs: dict, D_in: int, D_h: int) -> None:
        cls = globals()[cls_name]
        x_local = torch.randn(2, 5, D_in)
        m_none = cls(D_in, D_h, decay=None, **ctor_kwargs)
        try:
            m_none(x_local, delta_t=torch.rand(2, 5))
            raise AssertionError(
                f"{cls_name}: delta_t without decay should have raised ValueError"
            )
        except ValueError:
            pass
        m_decay = cls(D_in, D_h, decay="fixed", **ctor_kwargs)
        try:
            m_decay(x_local)
            raise AssertionError(
                f"{cls_name}: decay enabled without delta_t should have raised ValueError"
            )
        except ValueError:
            pass
        print(f"{cls_name}: both delta_t/decay ValueError modes: ok")

    _check_delta_t_value_errors("MinGRU", {}, D_in, D_h)
    _check_delta_t_value_errors("SignedMinGRU", {}, D_in, D_h)
    _check_delta_t_value_errors("RotationMinGRU", {}, D_in, D_h)

    # --- invalid decay string raises ValueError at construction ---
    for _cls_name in ("MinGRU", "SignedMinGRU", "RotationMinGRU"):
        _cls = globals()[_cls_name]
        try:
            _cls(D_in, D_h, decay="bogus")
            raise AssertionError(
                f"{_cls_name}: invalid decay string should have raised ValueError"
            )
        except ValueError:
            pass
    print("invalid decay string raises ValueError: ok")

    # --- non-positive decay_rate raises ValueError, both modes (gamma in
    # (0, 1] requires lambda > 0; a non-positive fixed decay_rate would let
    # gamma amplify, and a non-positive learnable decay_rate has no valid
    # rho init, otherwise surfacing as an opaque math domain error) ---
    for _mode in ("fixed", "learnable"):
        for _bad_rate in (0.0, -1.0):
            try:
                MinGRU(D_in, D_h, decay=_mode, decay_rate=_bad_rate)
                raise AssertionError(
                    f"MinGRU: decay={_mode!r}, decay_rate={_bad_rate} should "
                    "have raised ValueError"
                )
            except ValueError:
                pass
    print("non-positive decay_rate raises ValueError, both modes: ok")

    # --- negative delta_t: warn once per instance, then clamp to 0 exactly ---
    torch.manual_seed(206)
    x_neg = torch.randn(3, 6, D_in)
    m_neg = MinGRU(D_in, D_h, decay="fixed").eval()
    dt_neg = torch.rand(3, 6) + 0.5
    dt_neg[0, 2] = -4.0
    dt_clamped = dt_neg.clone()
    dt_clamped[0, 2] = 0.0
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        with torch.no_grad():
            out_neg = m_neg(x_neg, delta_t=dt_neg)
            out_neg_again = m_neg(x_neg, delta_t=dt_neg)  # 2nd call: no new warning
        neg_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
    assert len(neg_warnings) == 1, (
        f"expected exactly one negative-delta_t warning (warn-once), got {len(neg_warnings)}"
    )
    with torch.no_grad():
        out_clamped = m_neg(x_neg, delta_t=dt_clamped)
    err = (out_neg - out_clamped).abs().max().item()
    assert err == 0.0, "negative delta_t must clamp to 0, exactly matching the zero-gap output"
    assert torch.isfinite(out_neg).all() and torch.isfinite(out_neg_again).all()
    print("negative delta_t: warn-once + exact clamp-to-0: ok")

    # --- log1p_delta=True: per mixer, delta_t=0 exact, parallel-vs-step ok,
    # and it must actually change outputs vs log1p_delta=False on nonzero delta_t ---
    def _check_log1p_delta(
        cls_name: str, ctor_kwargs: dict, D_in: int, D_h: int, B: int, T: int, seed: int
    ) -> None:
        cls = globals()[cls_name]
        torch.manual_seed(seed)
        x_local = torch.randn(B, T, D_in)
        torch.manual_seed(seed)
        m_none = cls(D_in, D_h, decay=None, **ctor_kwargs).eval()
        torch.manual_seed(seed)
        m_log1p = cls(
            D_in, D_h, decay="fixed", decay_rate=1.0, log1p_delta=True, **ctor_kwargs
        ).eval()
        torch.manual_seed(seed)
        m_plain = cls(
            D_in, D_h, decay="fixed", decay_rate=1.0, log1p_delta=False, **ctor_kwargs
        ).eval()

        # delta_t = 0 -> log1p(0) = 0 -> gamma = 1 exactly, same as log1p_delta=False.
        with torch.no_grad():
            out_none = m_none(x_local)
            out_zero = m_log1p(x_local, delta_t=torch.zeros(B, T))
        err = (out_none - out_zero).abs().max().item()
        assert err == 0.0, f"{cls_name} (log1p_delta=True): delta_t=0 must give gamma=1 exactly"

        # Parallel vs step equivalence still holds under log1p_delta.
        torch.manual_seed(seed + 1)
        dt = torch.rand(B, T) * 3.0 + 1e-2
        with torch.no_grad():
            h_par = m_log1p(x_local, delta_t=dt)
            h = None
            hs = []
            for t in range(T):
                h = m_log1p.step(x_local[:, t], h, delta_t=dt[:, t])
                hs.append(h)
            h_seq = torch.stack(hs, dim=1)
        err = (h_par - h_seq).abs().max().item()
        assert err < 1e-4, f"{cls_name} (log1p_delta=True): parallel vs step max diff {err}"

        # log1p_delta actually changes the decay applied on the same nonzero delta_t
        # (log1p(dt) != dt for dt > 0, so the two must diverge on identical weights).
        with torch.no_grad():
            out_log1p = m_log1p(x_local, delta_t=dt)
            out_plain = m_plain(x_local, delta_t=dt)
        diff = (out_log1p - out_plain).abs().max().item()
        assert diff > 1e-6, (
            f"{cls_name}: log1p_delta=True should differ from log1p_delta=False "
            "on the same nonzero delta_t"
        )
        print(
            f"{cls_name} log1p_delta=True: delta_t=0 exact, parallel-vs-step "
            f"ok ({err:.3e}), differs from log1p_delta=False (diff {diff:.3e})"
        )

    _check_log1p_delta("MinGRU", {}, D_in, D_h, B=4, T=64, seed=301)
    _check_log1p_delta("SignedMinGRU", {"coupled": False}, D_in, D_h, B=4, T=64, seed=302)
    _check_log1p_delta("SignedMinGRU", {"coupled": True}, D_in, D_h, B=4, T=64, seed=303)
    _check_log1p_delta("RotationMinGRU", {}, D_in, D_h, B=4, T=64, seed=304)

    # --- _normalize_delta_t: trailing-singleton squeeze path, both call shapes,
    # plus the malformed-shape ValueError ---
    torch.manual_seed(310)
    x_sq = torch.randn(3, 10, D_in)
    m_sq = MinGRU(D_in, D_h, decay="fixed").eval()
    dt_2d = torch.rand(3, 10) + 0.1

    # forward: (B, T) vs (B, T, 1) must be bit-identical after squeeze.
    with torch.no_grad():
        out_2d = m_sq(x_sq, delta_t=dt_2d)
        out_3d = m_sq(x_sq, delta_t=dt_2d.unsqueeze(-1))
    err = (out_2d - out_3d).abs().max().item()
    assert err == 0.0, "delta_t (B, T, 1) must be bit-identical to (B, T) after squeeze"

    # step: (B,) vs (B, 1) must be bit-identical after squeeze.
    dt_t0 = dt_2d[:, 0]
    with torch.no_grad():
        h_1d = m_sq.step(x_sq[:, 0], None, delta_t=dt_t0)
        h_2d = m_sq.step(x_sq[:, 0], None, delta_t=dt_t0.unsqueeze(-1))
    err = (h_1d - h_2d).abs().max().item()
    assert err == 0.0, "step delta_t (B, 1) must be bit-identical to (B,) after squeeze"

    # malformed shape: an extra trailing dim that isn't a squeezable size-1.
    try:
        m_sq(x_sq, delta_t=torch.rand(3, 10, 2))
        raise AssertionError("malformed delta_t shape (B, T, 2) should have raised ValueError")
    except ValueError:
        pass
    print(
        "delta_t trailing-singleton squeeze (forward (B,T,1), step (B,1)) + "
        "malformed-shape ValueError: ok"
    )

    # --- NaN / +inf delta_t: sanitized to finite outputs on all three mixers
    # (log-space MinGRU especially), and the CPU warn-once fires for them ---
    def _check_nan_inf_delta_t(cls_name: str, ctor_kwargs: dict, D_in: int, D_h: int) -> None:
        cls = globals()[cls_name]
        torch.manual_seed(311)
        x_local = torch.randn(3, 12, D_in)
        m = cls(D_in, D_h, decay="fixed", **ctor_kwargs).eval()
        dt = torch.rand(3, 12) + 0.1
        dt[0, 3] = float("nan")
        dt[1, 5] = float("inf")
        with _warnings.catch_warnings(record=True) as rec:
            _warnings.simplefilter("always")
            with torch.no_grad():
                out = m(x_local, delta_t=dt)
            fired = any(issubclass(w.category, UserWarning) for w in rec)
        assert torch.isfinite(out).all(), (
            f"{cls_name}: NaN/+inf delta_t must produce finite outputs, got {out}"
        )
        assert fired, f"{cls_name}: CPU warn-once must fire for NaN/+inf delta_t"
        print(f"{cls_name}: NaN/+inf delta_t -> finite outputs, CPU warn fires: ok")

    _check_nan_inf_delta_t("MinGRU", {}, D_in, D_h)
    _check_nan_inf_delta_t("SignedMinGRU", {}, D_in, D_h)
    _check_nan_inf_delta_t("RotationMinGRU", {}, D_in, D_h)

    # =====================================================================
    # Task 2 (time-aware decay): Block/Stack delta_t threading + decay_layers
    # (spec section 9.2)
    # =====================================================================

    # --- decayed stack (signed, learnable): parallel vs streaming, random delta_t ---
    torch.manual_seed(401)
    decayed_stack = MinGRUStack(
        D_in, D_h, 3, mixer="signed",
        mixer_kwargs={"decay": "learnable", "decay_rate": 1.0},
    ).eval()
    dt_stack = torch.rand(B, T) * 2.0 + 1e-2
    with torch.no_grad():
        y_par, _ = decayed_stack(x, delta_t=dt_stack)
        state = decayed_stack.init_state()
        ys = []
        for t in range(T):
            y_t, state = decayed_stack.step(x[:, t], state, delta_t=dt_stack[:, t])
            ys.append(y_t)
        y_seq = torch.stack(ys, dim=1)
    err = (y_par - y_seq).abs().max().item()
    print(f"decayed stack (signed, learnable) parallel vs streaming max abs diff: {err:.3e}")
    assert err < 1e-4

    # --- decayed stack: chunked vs full, nonzero boundary gap ---
    Th = T // 2
    assert dt_stack[:, Th].min().item() > 0, "test setup: boundary gap must be nonzero"
    with torch.no_grad():
        y_a, carry = decayed_stack(x[:, :Th], delta_t=dt_stack[:, :Th])
        y_b, _ = decayed_stack(x[:, Th:], state=carry, delta_t=dt_stack[:, Th:])
        y_chunked = torch.cat([y_a, y_b], dim=1)
    err = (y_par - y_chunked).abs().max().item()
    print(f"decayed stack chunked vs full (nonzero boundary gap) max abs diff: {err:.3e}")
    assert err < 1e-4

    # --- decay_layers="last": rho present/grad in EXACTLY the final block ---
    torch.manual_seed(402)
    last_stack = MinGRUStack(
        D_in, D_h, 3, mixer="signed",
        mixer_kwargs={"decay": "learnable", "decay_rate": 1.0},
        decay_layers="last",
    )
    assert all(block.mingru.decay is None for block in last_stack.blocks[:-1]), (
        "decay_layers='last' must strip decay from every block but the final one"
    )
    assert last_stack.blocks[-1].mingru.decay == "learnable", (
        "decay_layers='last' must keep decay on the final block"
    )
    for i, block in enumerate(last_stack.blocks[:-1]):
        assert not hasattr(block.mingru, "rho"), (
            f"block {i} must have no decay params under decay_layers='last'"
        )
    dt_last = torch.rand(B, T) * 2.0 + 1e-2
    loss = last_stack(x, delta_t=dt_last)[0].sum()
    loss.backward()
    final_rho_grad = last_stack.blocks[-1].mingru.rho.grad
    assert final_rho_grad is not None and final_rho_grad.abs().sum() > 0, (
        "final block's rho must receive gradient under decay_layers='last'"
    )
    print("decay_layers='last': rho present/grad in exactly the final block: ok")

    # --- decay_layers="last": streaming step() matches parallel forward ---
    # (mixed per-block routing: earlier blocks get no delta_t, final block
    # gets the real per-step gap — the one routing shape forward-only
    # tests cannot see.)
    last_eval = last_stack.eval()
    with torch.no_grad():
        y_par_last, _ = last_eval(x, delta_t=dt_last)
        state = last_eval.init_state()
        ys = []
        for t in range(T):
            y_t, state = last_eval.step(x[:, t], state, delta_t=dt_last[:, t])
            ys.append(y_t)
    err = (y_par_last - torch.stack(ys, dim=1)).abs().max().item()
    print(f"decay_layers='last' parallel vs streaming max abs diff: {err:.3e}")
    assert err < 1e-4

    # --- decay_layers="last" with NO decay keys: harmless no-op ---
    noop_stack = MinGRUStack(D_in, D_h, 2, mixer="signed", decay_layers="last")
    assert all(block.mingru.decay is None for block in noop_stack.blocks)
    with torch.no_grad():
        _ = noop_stack(x)
    print("decay_layers='last' with no decay keys: harmless no-op: ok")

    # --- decay_layers="last" with n_layers=1: sole block is the final block ---
    single_last = MinGRUStack(
        D_in, D_h, 1, mixer="signed",
        mixer_kwargs={"decay": "fixed", "decay_rate": 1.0},
        decay_layers="last",
    )
    assert single_last.blocks[0].mingru.decay == "fixed", (
        "n_layers=1 under decay_layers='last' must keep decay on the sole block"
    )
    print("decay_layers='last' with n_layers=1: decay kept on sole block: ok")

    # --- decay_layers="all": rho grads reach every block ---
    torch.manual_seed(403)
    all_stack = MinGRUStack(
        D_in, D_h, 3, mixer="signed",
        mixer_kwargs={"decay": "learnable", "decay_rate": 1.0},
        decay_layers="all",
    )
    assert all(block.mingru.decay == "learnable" for block in all_stack.blocks)
    dt_all = torch.rand(B, T) * 2.0 + 1e-2
    loss = all_stack(x, delta_t=dt_all)[0].sum()
    loss.backward()
    for i, block in enumerate(all_stack.blocks):
        grad = block.mingru.rho.grad
        assert grad is not None and grad.abs().sum() > 0, (
            f"block {i} rho received no gradient under decay_layers='all'"
        )
    print("decay_layers='all': rho grads reach every block: ok")

    # --- delta_t rejection when no block in the stack has decay ---
    plain_stack = MinGRUStack(D_in, D_h, 3, mixer="signed")
    try:
        plain_stack(x, delta_t=torch.rand(B, T))
        raise AssertionError(
            "stack with no decayed blocks should reject delta_t with ValueError"
        )
    except ValueError:
        pass
    state0 = plain_stack.init_state()
    try:
        plain_stack.step(x[:, 0], state0, delta_t=torch.rand(B))
        raise AssertionError(
            "stack.step with no decayed blocks should reject delta_t with ValueError"
        )
    except ValueError:
        pass
    print("stack delta_t rejection when no block has decay: ok")

    # --- invalid decay_layers string raises ValueError at construction ---
    try:
        MinGRUStack(D_in, D_h, 3, decay_layers="bogus")
        raise AssertionError("invalid decay_layers should have raised ValueError")
    except ValueError:
        pass
    print("invalid decay_layers string raises ValueError: ok")

    # --- rotation-mixer decayed stack: parallel vs streaming ---
    torch.manual_seed(404)
    rot_decayed_stack = MinGRUStack(
        D_in, D_h, 3, mixer="rotation",
        mixer_kwargs={"decay": "learnable", "decay_rate": 1.0},
    ).eval()
    dt_rot_stack = torch.rand(B, T) * 2.0 + 1e-2
    with torch.no_grad():
        y_par_rd, _ = rot_decayed_stack(x, delta_t=dt_rot_stack)
        state = rot_decayed_stack.init_state()
        ys = []
        for t in range(T):
            y_t, state = rot_decayed_stack.step(x[:, t], state, delta_t=dt_rot_stack[:, t])
            ys.append(y_t)
        y_seq_rd = torch.stack(ys, dim=1)
    err = (y_par_rd - y_seq_rd).abs().max().item()
    print(f"decayed stack (rotation, learnable) parallel vs streaming max abs diff: {err:.3e}")
    assert err < 1e-4

    # =====================================================================
    # Task 3 (heterogeneous stacks): list-mixer construction, kwargs schema,
    # multi-rotation warning, mixed-stack equivalence (spec section 9.1)
    # =====================================================================

    # --- valid 2-layer mixed construction: signed -> rotation ---
    torch.manual_seed(501)
    hetero2 = MinGRUStack(D_in, D_h, 2, mixer=["signed", "rotation"]).eval()
    assert isinstance(hetero2.blocks[0].mingru, SignedMinGRU)
    assert isinstance(hetero2.blocks[1].mingru, RotationMinGRU)
    with torch.no_grad():
        _ = hetero2(x)
    print("2-layer mixed stack (signed, rotation): construction + forward ok")

    # --- valid 3-layer mixed construction, with per-type mixer_kwargs
    # (coupled signed + a custom snap grid on rotation) ---
    torch.manual_seed(502)
    hetero3 = MinGRUStack(
        D_in, D_h, 3, mixer=["signed", "signed", "rotation"],
        mixer_kwargs={"signed": {"coupled": True}, "rotation": {"snap": (2, 3, 6)}},
    ).eval()
    assert all(block.mingru.coupled for block in hetero3.blocks[:2])
    assert hetero3.blocks[2].mingru.snap == (2, 3, 6)
    with torch.no_grad():
        _ = hetero3(x)
    print("3-layer mixed stack (signed, signed, rotation) + per-type mixer_kwargs: ok")

    # --- mixer list length mismatch raises ValueError ---
    try:
        MinGRUStack(D_in, D_h, 3, mixer=["signed", "rotation"])
        raise AssertionError("mixer list length mismatch should have raised ValueError")
    except ValueError:
        pass
    print("mixer list length mismatch raises ValueError: ok")

    # --- unknown mixer name in list raises ValueError ---
    try:
        MinGRUStack(D_in, D_h, 2, mixer=["signed", "bogus"])
        raise AssertionError("unknown mixer name in list should have raised ValueError")
    except ValueError:
        pass
    print("unknown mixer name in list raises ValueError: ok")

    # --- flat dict with list mixer raises ValueError (the dict's key isn't a
    # valid mixer name, exactly the same structural check as "unknown/absent
    # type key" below applied to a fully flat dict) ---
    try:
        MinGRUStack(
            D_in, D_h, 2, mixer=["signed", "rotation"], mixer_kwargs={"coupled": True}
        )
        raise AssertionError("flat dict with list mixer should have raised ValueError")
    except ValueError:
        pass
    print("flat dict with list mixer raises ValueError: ok")

    # --- type-keyed dict with str mixer raises ValueError ---
    try:
        MinGRUStack(
            D_in, D_h, 2, mixer="signed", mixer_kwargs={"signed": {"coupled": True}}
        )
        raise AssertionError("type-keyed dict with str mixer should have raised ValueError")
    except ValueError:
        pass
    print("type-keyed dict with str mixer raises ValueError: ok")

    # --- mixer_kwargs key naming a type absent from the list raises ValueError
    # ("rotation" is a globally valid mixer name, but not present in this
    # particular mixer list) ---
    try:
        MinGRUStack(
            D_in, D_h, 2, mixer=["signed", "signed"],
            mixer_kwargs={"rotation": {"snap": (3,)}},
        )
        raise AssertionError(
            "mixer_kwargs key naming a type absent from the list should have "
            "raised ValueError"
        )
    except ValueError:
        pass
    print("mixer_kwargs key naming a type absent from the list raises ValueError: ok")

    # --- mixer of an unsupported type (neither str nor list) raises ValueError ---
    try:
        MinGRUStack(D_in, D_h, 2, mixer=("signed", "rotation"))
        raise AssertionError("mixer of an unsupported type should have raised ValueError")
    except ValueError:
        pass
    print("mixer of an unsupported type (tuple) raises ValueError: ok")

    # --- warn-once: exactly one UserWarning for a rotation x2 mixed stack ---
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        MinGRUStack(D_in, D_h, 3, mixer=["rotation", "signed", "rotation"])
        rot_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
    assert len(rot_warnings) == 1, (
        f"expected exactly one multi-rotation UserWarning, got {len(rot_warnings)}"
    )
    print("multi-rotation (rotation x2) mixed stack: exactly one UserWarning: ok")

    # --- no warning for a single-rotation mixed stack ---
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        MinGRUStack(D_in, D_h, 3, mixer=["signed", "signed", "rotation"])
        rot_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
    assert len(rot_warnings) == 0, (
        f"single-rotation mixed stack must not warn, got {len(rot_warnings)}"
    )
    print("single-rotation mixed stack: zero UserWarning: ok")

    # --- mixed stack (no decay): parallel vs streaming, chunked vs full ---
    torch.manual_seed(503)
    mixed_stack = MinGRUStack(D_in, D_h, 3, mixer=["signed", "rotation", "signed"]).eval()
    with torch.no_grad():
        y_par_m, _ = mixed_stack(x)
        state = mixed_stack.init_state()
        ys = []
        for t in range(T):
            y_t, state = mixed_stack.step(x[:, t], state)
            ys.append(y_t)
        y_seq_m = torch.stack(ys, dim=1)
    err = (y_par_m - y_seq_m).abs().max().item()
    print(f"mixed stack (signed, rotation, signed) parallel vs streaming max abs diff: {err:.3e}")
    assert err < 1e-4

    with torch.no_grad():
        y_a_m, carry_m = mixed_stack(x[:, : T // 2])
        y_b_m, _ = mixed_stack(x[:, T // 2 :], state=carry_m)
        y_chunked_m = torch.cat([y_a_m, y_b_m], dim=1)
    err = (y_par_m - y_chunked_m).abs().max().item()
    print(f"mixed stack (signed, rotation, signed) chunked vs full max abs diff: {err:.3e}")
    assert err < 1e-4

    loss = mixed_stack(x)[0].sum()
    loss.backward()
    print("mixed stack (signed, rotation, signed) gradcheck ok")

    # --- mixed stack, decay on ONE type only (per-type mixer_kwargs): parallel
    # vs streaming, chunked vs full, delta_t reaches only the decayed type's
    # block(s) ---
    torch.manual_seed(504)
    mixed_decay_stack = MinGRUStack(
        D_in, D_h, 3, mixer=["signed", "rotation", "signed"],
        mixer_kwargs={"signed": {"decay": "learnable", "decay_rate": 1.0}},
    ).eval()
    assert mixed_decay_stack.blocks[0].mingru.decay == "learnable"
    assert mixed_decay_stack.blocks[2].mingru.decay == "learnable"
    assert mixed_decay_stack.blocks[1].mingru.decay is None, (
        "rotation block must not receive decay kwargs meant for 'signed'"
    )
    dt_mixed = torch.rand(B, T) * 2.0 + 1e-2
    with torch.no_grad():
        y_par_md, _ = mixed_decay_stack(x, delta_t=dt_mixed)
        state = mixed_decay_stack.init_state()
        ys = []
        for t in range(T):
            y_t, state = mixed_decay_stack.step(x[:, t], state, delta_t=dt_mixed[:, t])
            ys.append(y_t)
        y_seq_md = torch.stack(ys, dim=1)
    err = (y_par_md - y_seq_md).abs().max().item()
    print(
        "mixed stack, decay on 'signed' only: parallel vs streaming max abs "
        f"diff: {err:.3e}"
    )
    assert err < 1e-4

    Th_m = T // 2
    assert dt_mixed[:, Th_m].min().item() > 0, "test setup: boundary gap must be nonzero"
    with torch.no_grad():
        y_a_md, carry_md = mixed_decay_stack(x[:, :Th_m], delta_t=dt_mixed[:, :Th_m])
        y_b_md, _ = mixed_decay_stack(
            x[:, Th_m:], state=carry_md, delta_t=dt_mixed[:, Th_m:]
        )
        y_chunked_md = torch.cat([y_a_md, y_b_md], dim=1)
    err = (y_par_md - y_chunked_md).abs().max().item()
    print(
        "mixed stack, decay on 'signed' only: chunked vs full (nonzero "
        f"boundary gap) max abs diff: {err:.3e}"
    )
    assert err < 1e-4

    loss = mixed_decay_stack(x, delta_t=dt_mixed)[0].sum()
    loss.backward()
    assert mixed_decay_stack.blocks[0].mingru.rho.grad is not None
    assert mixed_decay_stack.blocks[2].mingru.rho.grad is not None
    print("mixed stack, decay on 'signed' only: gradcheck ok")

    # delta_t reaches only decay-enabled blocks: the rotation block's mixer
    # has decay=None, so MinGRUStack.forward/step's per-block routing (see
    # `_check_delta_t_has_decay` / the `block.mingru.decay is not None` gate)
    # must never pass it delta_t even though the stack-level call carries one.
    decay_enabled_blocks = [
        i for i, b in enumerate(mixed_decay_stack.blocks) if b.mingru.decay is not None
    ]
    assert decay_enabled_blocks == [0, 2], (
        f"expected delta_t routing to blocks [0, 2] only, got {decay_enabled_blocks}"
    )
    print("mixed stack: delta_t reaches only decay-enabled blocks ([0, 2]): ok")

    # =====================================================================
    # GivensMinGRU promotion: Givens-specific algebra + generic decay
    # contract pickup (spec section 9.1). GivensMinGRU is a standard
    # DecayMixin subclass, so the per-mixer decay/delta_t helpers defined
    # above (_check_decay_suite, _check_delta_t_value_errors,
    # _check_log1p_delta, _check_nan_inf_delta_t) exercise it with NO
    # special-casing — invoked here by class name exactly as for the other
    # mixers. The remaining checks cover the Givens-only algebra (exact
    # special-orthogonality of the transition, the k-dim matrix_affine_scan
    # vs an explicit sequential recurrence) and the multi-'givens'
    # zero-warning construction contract.
    # =====================================================================

    # --- transition is exactly special-orthogonal: M @ M^T = I, det = +1 ---
    # Tolerances at least as tight as the lab evidence generator's
    # (orthogonality 1e-5, determinant 1e-4).
    torch.manual_seed(601)
    mg = GivensMinGRU(D_in, D_h).eval()  # block_size=8, rounds=3, decay=None
    x_g = torch.randn(B, T, D_in)
    k_g = mg.k
    with torch.no_grad():
        M_g, b_g = mg._coeffs(x_g)  # (B, T, n_blocks, k, k), (B, T, n_blocks, k)
        MMt = M_g @ M_g.transpose(-1, -2)
        orth_err = (MMt - torch.eye(k_g)).abs().max().item()
        det_err = (torch.linalg.det(M_g) - 1.0).abs().max().item()
    print(f"givens transition orthogonality (M @ M^T - I) max abs diff: {orth_err:.3e}")
    print(f"givens transition det - 1 max abs diff: {det_err:.3e}")
    assert orth_err < 1e-5, "Givens transition must be orthogonal (M @ M^T = I)"
    assert det_err < 1e-4, "Givens transition must have determinant +1"

    # --- parallel scan (forward) vs explicit sequential recurrence on the
    # SAME per-token coeffs (validates the matrix_affine_scan wiring) ---
    with torch.no_grad():
        h_par = mg(x_g)
        h_prev = mg.h0.expand(B, 1, D_h).reshape(B, mg.n_blocks, k_g)
        hs = []
        for t in range(T):
            h_prev = torch.einsum("bnij,bnj->bni", M_g[:, t], h_prev) + b_g[:, t]
            hs.append(h_prev.reshape(B, D_h))
        h_seq = torch.stack(hs, dim=1)
    err = (h_par - h_seq).abs().max().item()
    print(f"givens parallel scan vs explicit sequential recurrence max abs diff: {err:.3e}")
    assert err < 1e-4, "matrix_affine_scan forward does not match sequential recurrence"

    # --- forward == step (no decay), and chunked h_0 carry ---
    with torch.no_grad():
        h = None
        hs = [h := mg.step(x_g[:, t], h) for t in range(T)]
    err = (h_par - torch.stack(hs, dim=1)).abs().max().item()
    print(f"givens parallel vs step (no decay) max abs diff: {err:.3e}")
    assert err < 1e-4

    with torch.no_grad():
        h_a = mg(x_g[:, : T // 2])
        h_b = mg(x_g[:, T // 2 :], h_0=h_a[:, -1:])
    err = (h_par - torch.cat([h_a, h_b], dim=1)).abs().max().item()
    print(f"givens chunked vs full (h_0 carry, no decay) max abs diff: {err:.3e}")
    assert err < 1e-4

    # --- generic decay-contract pickup (no special-casing): the same
    # per-mixer helpers used for MinGRU/SignedMinGRU/RotationMinGRU above.
    # _check_decay_suite covers forward == step WITH decay (fixed and
    # learnable) and chunked h_0 carry WITH decay, plus the rho gradcheck. ---
    _check_decay_suite("GivensMinGRU", {}, D_in, D_h, B=4, T=128, seed=602)
    _check_delta_t_value_errors("GivensMinGRU", {}, D_in, D_h)
    _check_log1p_delta("GivensMinGRU", {}, D_in, D_h, B=4, T=64, seed=603)
    _check_nan_inf_delta_t("GivensMinGRU", {}, D_in, D_h)
    try:
        GivensMinGRU(D_in, D_h, decay="bogus")
        raise AssertionError("GivensMinGRU: invalid decay string should have raised ValueError")
    except ValueError:
        pass
    print("givens: generic decay-contract pickup (no special-casing): ok")

    # --- construction ValueError cases: indivisible hidden_size, odd
    # block_size, and the registry naming 'givens' in the unknown-mixer message ---
    try:
        GivensMinGRU(D_in, D_h + 4, block_size=8)  # 68 % 8 != 0
        raise AssertionError("indivisible hidden_size should have raised ValueError")
    except ValueError:
        pass
    try:
        GivensMinGRU(D_in, D_h, block_size=3)  # odd block_size
        raise AssertionError("odd block_size should have raised ValueError")
    except ValueError:
        pass
    try:
        GivensMinGRU(D_in, D_h, rounds=0)  # no Givens layers: identity mixer
        raise AssertionError("rounds=0 should have raised ValueError")
    except ValueError:
        pass
    print(
        "givens construction ValueError (indivisible hidden_size, odd "
        "block_size, rounds=0): ok"
    )

    try:
        MinGRUBlock(D_h, mixer="not_a_mixer")
        raise AssertionError("unknown mixer should have raised ValueError")
    except ValueError as _e:
        assert "givens" in str(_e), "unknown-mixer ValueError should list 'givens' as valid"
    print("unknown-mixer ValueError lists 'givens': ok")

    # --- multi-'givens' stacks construct with ZERO warnings (Givens is
    # continuous — no straight-through snap to compound), and the existing
    # multi-'rotation' warn-once is unchanged when givens is also present ---
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        MinGRUStack(D_in, D_h, 3, mixer=["givens", "signed", "givens"])
        MinGRUStack(D_in, D_h, 3, mixer="givens")
        giv_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
    assert len(giv_warnings) == 0, (
        f"multi-'givens' stacks must not warn, got {len(giv_warnings)}"
    )
    print("multi-'givens' + homogeneous 'givens' stacks: zero UserWarning: ok")

    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        MinGRUStack(D_in, D_h, 3, mixer=["rotation", "givens", "rotation"])
        rot_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
    assert len(rot_warnings) == 1, (
        f"multi-rotation warn-once must be unchanged with givens present, got {len(rot_warnings)}"
    )
    print("multi-rotation warn-once unchanged with givens present: ok")

    # --- MinGRUStack mixer="givens": parallel vs streaming, chunked vs full ---
    # The parallel and streaming paths are algebraically identical; the gap
    # is pure float32 accumulation. A single Givens block's k=8 transition is
    # a product of 3 brick-wall rounds of 8x8 matmuls (vs rotation's single
    # 2x2), so per-block scan error runs ~20x rotation's (~2e-5 vs ~1e-6),
    # and 3 stacked layers over T=128 compound it to a few e-4. In float64
    # the same test collapses to ~1e-13, confirming there is no logic drift.
    # The brief-mandated 1e-4 bound is kept where it belongs (the mixer-level
    # scan-vs-sequential and forward-vs-step checks above, which pass at
    # ~1e-5); this deep-stack aggregate gets a wider, documented bound.
    _GIVENS_STACK_TOL = 1e-3
    torch.manual_seed(604)
    gstack = MinGRUStack(D_in, D_h, 3, mixer="givens").eval()
    with torch.no_grad():
        y_par_g, _ = gstack(x)
        state = gstack.init_state()
        ys = []
        for t in range(T):
            y_t, state = gstack.step(x[:, t], state)
            ys.append(y_t)
    err = (y_par_g - torch.stack(ys, dim=1)).abs().max().item()
    print(f"givens stack parallel vs streaming max abs diff: {err:.3e}")
    assert err < _GIVENS_STACK_TOL

    with torch.no_grad():
        y_a_g, carry_g = gstack(x[:, : T // 2])
        y_b_g, _ = gstack(x[:, T // 2 :], state=carry_g)
    err = (y_par_g - torch.cat([y_a_g, y_b_g], dim=1)).abs().max().item()
    print(f"givens stack chunked vs full max abs diff: {err:.3e}")
    assert err < _GIVENS_STACK_TOL

    loss = MinGRUStack(D_in, D_h, 3, mixer="givens")(x)[0].sum()
    loss.backward()
    print("givens stack gradcheck ok")

    # --- gradients reach all three heads + h0, all finite and nonzero ---
    mg_grad = GivensMinGRU(D_in, D_h)
    mg_grad(x_g).sum().backward()
    for _name, _p in [
        ("linear_theta", mg_grad.linear_theta.weight),
        ("linear_z", mg_grad.linear_z.weight),
        ("linear_h", mg_grad.linear_h.weight),
        ("h0", mg_grad.h0),
    ]:
        assert _p.grad is not None and torch.isfinite(_p.grad).all() and _p.grad.abs().sum() > 0, (
            f"givens {_name} received no/non-finite gradient"
        )
    print("givens gradcheck ok: all three heads + h0 receive finite gradient")

    # --- matrix_affine_scan gradcheck-lite: finite grads through the k-dim
    # scan (the k>2 analogue of the matrix_scan gradcheck-lite above) ---
    torch.manual_seed(605)
    A_s = torch.randn(2, 9, 3, 4, 4) * 0.5  # (B, T, n, k, k), unconstrained
    B_s = torch.randn(2, 9, 3, 4, 1)  # (B, T, n, k, v=1)
    A_s.requires_grad_(True)
    B_s.requires_grad_(True)
    Abar_s, Bbar_s = matrix_affine_scan(A_s, B_s)
    (Abar_s.sum() + Bbar_s.sum()).backward()
    assert A_s.grad is not None and torch.isfinite(A_s.grad).all()
    assert B_s.grad is not None and torch.isfinite(B_s.grad).all()
    print("matrix_affine_scan gradcheck-lite ok")

    # =====================================================================
    # Structural guard: DecayMixin._init_decay constructs a mixer's decay
    # parameter/buffer LAST, for every mixer registered in
    # MinGRUBlock._MIXER_CLASSES — generic over the table (not enumerating
    # MinGRU/SignedMinGRU/RotationMinGRU by name), so a future mixer added
    # there gets the construct-last invariant checked automatically.
    #
    # "Last" here means last among the mixer's OWN directly-registered
    # parameters (decay="learnable" -> checked via ``self._parameters``) or
    # buffers (decay="fixed" -> checked via ``self._buffers``) -- NOT last
    # in the flattened, recursive ``state_dict()`` key list. PyTorch's
    # ``nn.Module.state_dict()`` always emits a module's own direct
    # parameters, then its own direct buffers, then recurses into child
    # modules (here, each mixer's ``linear_*`` submodules) -- so a mixer's
    # own keys (decay entries included) necessarily precede its Linear
    # submodules' "linear_*.weight"/"linear_*.bias" keys in the flat list,
    # regardless of __init__ call order; that ordering is not something
    # decay construction order can or needs to control. Checking
    # ``self._parameters``/``self._buffers`` directly is the literal,
    # verifiable form of "constructed last" -- exactly what "last line of
    # __init__" governs.
    # =====================================================================
    for _mixer_name, (_mixer_cls, _) in MinGRUBlock._MIXER_CLASSES.items():
        _m_fixed = _mixer_cls(D_in, D_h, decay="fixed", decay_rate=1.0)
        _fixed_buf_keys = list(_m_fixed._buffers.keys())
        assert _fixed_buf_keys and _fixed_buf_keys[-1] == "decay_rate_buf", (
            f"{_mixer_name} ({_mixer_cls.__name__}, decay='fixed'): expected "
            f"'decay_rate_buf' last among direct buffers, got {_fixed_buf_keys!r}"
        )

        _m_learnable = _mixer_cls(D_in, D_h, decay="learnable", decay_rate=1.0)
        _learnable_param_keys = list(_m_learnable._parameters.keys())
        assert _learnable_param_keys and _learnable_param_keys[-1] == "rho", (
            f"{_mixer_name} ({_mixer_cls.__name__}, decay='learnable'): expected "
            f"'rho' last among direct parameters, got {_learnable_param_keys!r}"
        )
    print(
        "generic decay-constructed-last check (all _MIXER_CLASSES entries, "
        "fixed buffer / learnable parameter): ok"
    )
