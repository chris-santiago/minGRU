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

References
----------
Feng et al., "Were RNNs All We Needed?", arXiv:2410.01201.
Heinsen (2023) -- the log-space parallel scan used by the log-space
minGRU's parallel training-mode forward.
"""

import math
import os
import warnings
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# Public surface (additive metadata, Phase-4 packaging prep): the four
# mixers, the block/stack wrappers, and the four scan functions -- exactly
# what the README documents as importable (e.g. `from min_gru import
# MinGRUStack`). Everything else (dispatch internals, decay machinery,
# `g`/`log_g`, `DecayMixin`) is implementation detail, not part of the
# documented public API.
__all__ = [
    "MinGRU",
    "SignedMinGRU",
    "RotationMinGRU",
    "GivensMinGRU",
    "MinGRUBlock",
    "MinGRUStack",
    "parallel_scan_log",
    "linear_scan",
    "matrix_scan",
    "matrix_affine_scan",
]

Decay = Literal["fixed", "learnable"] | None

# --- Triton scan dispatch seam ----------------------------------------------
#
# MINGRU_SCAN selects which implementation backs the four scan functions
# below (mirrors probes.py's MAX_STEPS/CKPT env-var idiom: read via
# os.environ.get at call time, not cached at import time):
#   "auto" (default) -- CUDA-resident inputs with an importable, available
#       Triton kernel registered for the op run that kernel; everything
#       else (CPU tensors, missing/unavailable triton_scans, no kernel
#       registered yet for the op) falls back to the eager implementation
#       below, unchanged. A CUDA-input fallback warns exactly once per
#       process, naming the reason.
#   "eager" -- always the eager implementation; triton_scans is never
#       imported.
#   "triton" -- always a Triton kernel; raises rather than silently
#       falling back to eager if one isn't available.
# See the ``mingru.triton_scans`` module docstring (and the project docs
# site) for the full dispatch contract and shape envelope. triton_scans is
# imported lazily, only from within this function, so importing/running
# min_gru.py never attempts it.
#
# Disclosure: for `parallel_scan_log` specifically (the `MinGRU`/
# `SignedMinGRU` path), "auto" routing the fwd+bwd of a training step to
# the Triton kernel on CUDA is measured SLOWER than eager at benchmarked
# shapes (0.70-0.80x eager; see experiments/bench/scan_bench.md) -- the
# forward-only kernel is roughly on par with eager, but the backward's
# autograd-through-recomputation reruns the eager forward under
# `enable_grad`, doubling the forward cost. `MINGRU_SCAN=eager` is the
# right choice for training the base mixer today; this is disclosure
# only, the routing itself is unchanged.
_VALID_SCAN_MODES = ("auto", "eager", "triton")
_warned_scan_fallback = False
_warned_angle_fallback = False


def _resolve_scan_mode(is_cuda: bool) -> str | None:
    """Validate ``MINGRU_SCAN`` and apply the eager/CPU-``auto`` short-circuit.

    Shared by ``_dispatch_scan``, ``_angle_scan_should_try``, and
    ``_dispatch_angle_scan`` -- the identical "read ``MINGRU_SCAN``,
    validate it, then skip Triton for ``eager`` or CPU-resident ``auto``"
    check used to be duplicated across the first two; hoisting it here lets
    ``_dispatch_angle_scan`` call it directly too, so it no longer has to
    (silently) trust that ``_angle_scan_should_try`` already ran first --
    see ``_dispatch_angle_scan``'s docstring.

    Parameters
    ----------
    is_cuda : bool
        Whether this call's tensor input is CUDA-resident.

    Returns
    -------
    str or None
        The validated mode (``"auto"`` or ``"triton"``) if a Triton path
        should be attempted; ``None`` if the caller should just run its
        existing eager implementation (``MINGRU_SCAN=eager``, or ``auto``
        with a CPU-resident input -- CPU behavior is unchanged: no import
        attempted, no warning).

    Raises
    ------
    ValueError
        If ``MINGRU_SCAN`` is not one of ``"auto"``, ``"eager"``, ``"triton"``.
    """
    mode = os.environ.get("MINGRU_SCAN", "auto")
    if mode not in _VALID_SCAN_MODES:
        raise ValueError(
            f"MINGRU_SCAN must be one of {_VALID_SCAN_MODES} (got {mode!r})"
        )
    if mode == "eager":
        return None
    if mode == "auto" and not is_cuda:
        # CPU behavior is unchanged: no import attempted, no warning.
        return None
    return mode


def _resolve_triton_dispatch(
    mode: str, op_label: str, call
) -> tuple[object | None, str | None]:
    """Shared ``triton_scans`` resolution: lazy import, per-op availability
    check + invocation via ``call``, and ``ScanFallback`` handling.

    Both ``_dispatch_scan`` and ``_dispatch_angle_scan`` need the identical
    "lazily import ``triton_scans``, confirm there's a usable Triton path
    for this op, invoke it, treat a declined/failed kernel exactly like any
    other unavailability reason" flow, differing only in which op-specific
    availability check and kernel call to make -- this is the one hoisted
    copy of that machinery, parameterized by ``call``.

    Parameters
    ----------
    mode : str
        The value a caller's own ``_resolve_scan_mode`` call already
        validated and returned. Always ``"auto"`` or ``"triton"`` here --
        this function is only ever reached after ruling out ``"eager"``
        and CPU-resident ``"auto"`` (never called with any other value).
    op_label : str
        Human-readable name of the op/path for the ``MINGRU_SCAN=triton``
        error message (e.g. ``repr(scan_name)``, or a fixed phrase for the
        angle-fused path).
    call : Callable[[module], object]
        Given the imported ``triton_scans`` module, perform this op's own
        availability check(s) and the actual kernel invocation, raising
        ``triton_scans.ScanFallback`` (never a bare exception) if no usable
        Triton path exists for this call -- out-of-envelope shape, no
        kernel registered yet, or a launch/compile failure are all the
        same "declined" signal to this function.

    Returns
    -------
    tuple of (result, reason)
        ``(kernel_result, None)`` if a Triton path ran; ``(None, reason)``
        if it didn't (only possible when ``mode == "auto"`` -- ``"triton"``
        raises directly below instead of returning a reason). The caller
        owns the ``mode == "auto"`` warn-once call itself (rather than this
        helper doing it), so each ``warnings.warn``'s ``stacklevel`` keeps
        pointing at that caller's own original call site, unaffected by
        this helper's stack depth.

    Raises
    ------
    RuntimeError
        If ``mode == "triton"`` but no Triton path is usable for this call
        (never a silent downgrade to eager).
    """
    reason = None
    try:
        from mingru import triton_scans
    except ImportError as exc:
        reason = f"triton_scans not importable: {exc}"
    else:
        try:
            return call(triton_scans), None
        except triton_scans.ScanFallback as exc:
            reason = str(exc)

    if mode == "triton":
        raise RuntimeError(
            f"MINGRU_SCAN=triton requested for {op_label} but Triton is "
            f"unavailable: {reason}"
        )
    return None, reason


def _dispatch_scan(name: str, *args: torch.Tensor):
    """Resolve ``MINGRU_SCAN`` and run a Triton scan kernel if selected.

    Called as a guard at the top of each of the four scan functions
    (``parallel_scan_log``, ``linear_scan``, ``matrix_scan``,
    ``matrix_affine_scan``), before any of their existing eager
    computation. A non-``None`` return is the scan's final result (the
    caller returns it immediately, skipping the eager code below); a
    ``None`` return means "fall through to the unchanged eager
    implementation".

    Parameters
    ----------
    name : str
        The scan function's own name, used to look up its Triton
        implementation in ``triton_scans.SCAN_IMPLS`` and to name the op
        in fallback warnings and errors.
    *args : torch.Tensor
        The scan function's positional tensor arguments, forwarded as-is
        to the Triton implementation when one runs. ``args[0]`` is
        inspected for ``.is_cuda`` to decide device residency.

    Returns
    -------
    tuple, torch.Tensor, or None
        The Triton kernel's result if one ran; ``None`` to signal the
        eager fallback.

    Raises
    ------
    ValueError
        If ``MINGRU_SCAN`` is set to something other than ``"auto"``,
        ``"eager"``, or ``"triton"``.
    RuntimeError
        If ``MINGRU_SCAN=triton`` but no Triton implementation is
        available or registered for ``name`` (never a silent downgrade
        to eager).
    """
    mode = _resolve_scan_mode(args[0].is_cuda)
    if mode is None:
        return None

    def _call(triton_scans):
        status = triton_scans.available()
        if status is not True:
            raise triton_scans.ScanFallback(str(status))
        if name not in triton_scans.SCAN_IMPLS:
            raise triton_scans.ScanFallback(f"no Triton kernel registered for {name!r} yet")
        return triton_scans.SCAN_IMPLS[name](*args)

    result, reason = _resolve_triton_dispatch(mode, repr(name), _call)
    if reason is None:
        return result

    # mode == "auto" with CUDA-resident inputs and no usable Triton path:
    # warn once per process, naming the reason, then fall back to eager.
    global _warned_scan_fallback
    if not _warned_scan_fallback:
        warnings.warn(
            f"MINGRU_SCAN=auto fell back to the eager scan implementation "
            f"for {name!r} despite CUDA inputs: {reason}",
            stacklevel=3,
        )
        _warned_scan_fallback = True
    return None


def _angle_scan_should_try(is_cuda: bool) -> bool:
    """Cheap pre-check: should the angle-fused module path be attempted?

    Thin wrapper over ``_resolve_scan_mode`` -- returns ``False`` (run the
    unchanged eager mixer forward) for ``eager``, or for ``auto`` with CPU
    inputs, WITHOUT importing ``triton_scans``. Keeping this gate ahead of
    any head assembly means the recorded CPU evidence path stays
    byte-identical and import-free. ``True`` means assemble the angle heads
    and call ``_dispatch_angle_scan`` (which independently re-validates via
    the same ``_resolve_scan_mode`` call -- see its docstring -- so this
    pre-check is a throughput optimization for the CPU/eager path, not a
    correctness dependency ``_dispatch_angle_scan`` relies on).

    Parameters
    ----------
    is_cuda : bool
        Whether the mixer's input is CUDA-resident.

    Returns
    -------
    bool
        ``True`` to attempt the angle-fused kernel, ``False`` to stay eager.

    Raises
    ------
    ValueError
        If ``MINGRU_SCAN`` is not one of ``auto``/``eager``/``triton``.
    """
    return _resolve_scan_mode(is_cuda) is not None


def _dispatch_angle_scan(
    theta: torch.Tensor,
    scale: torch.Tensor,
    gamma: torch.Tensor,
    b: torch.Tensor,
    h0: torch.Tensor,
    perm: torch.Tensor,
    sgn: torch.Tensor,
    p2p: torch.Tensor,
    *,
    has_scale: int,
    has_decay: int,
):
    """Run the angle-fused kernel, or return ``None`` to fall back to eager.

    ``has_scale``/``has_decay`` are keyword-only: both are 0/1 flags with no
    positional meaning a reader could infer from a bare ``1, 0`` at a call
    site (unlike ``theta``/``scale``/etc., whose names the tensor shapes
    already suggest) -- keyword-only makes every call site self-documenting
    and immune to an accidental argument-order swap between the two flags.

    The module-forward analogue of ``_dispatch_scan``, sharing both of its
    hoisted helpers. Both current call sites (``GivensMinGRU``,
    ``RotationMinGRU``) invoke this only after ``_angle_scan_should_try``
    already returned ``True`` (mode is ``auto`` with CUDA inputs, or
    ``triton``) so the eager path never pays for assembling the angle
    heads -- but this function does NOT depend on that ordering for
    correctness: it re-validates ``MINGRU_SCAN`` itself via
    ``_resolve_scan_mode(theta.is_cuda)``, exactly like ``_dispatch_scan``
    does with its own tensor argument, so a direct/out-of-order call is
    just as safe (redundant, not incorrect) as the current call sites.
    Shape/envelope validation and the launch-failure funnel live in
    ``triton_scans.angle_scan_impl`` (mirroring how ``_dispatch_scan``
    leaves those checks to each ``SCAN_IMPLS`` entry) -- this function
    catches only ``triton_scans.ScanFallback``, never a bare exception,
    exactly like ``_dispatch_scan``.

    Returns
    -------
    torch.Tensor or None
        The states ``h`` ``(B, T, n, k)`` if the kernel ran; ``None`` to fall
        through to the unchanged eager mixer forward.

    Raises
    ------
    ValueError
        If ``MINGRU_SCAN`` is not one of ``"auto"``, ``"eager"``, ``"triton"``.
    RuntimeError
        If ``MINGRU_SCAN=triton`` but the angle-fused kernel is unavailable
        (never a silent downgrade).
    """
    mode = _resolve_scan_mode(theta.is_cuda)
    if mode is None:
        return None

    def _call(triton_scans):
        status = triton_scans.available()
        if status is not True:
            raise triton_scans.ScanFallback(str(status))
        if not hasattr(triton_scans, "angle_scan_impl"):
            raise triton_scans.ScanFallback("no angle-fused Triton kernel registered")
        return triton_scans.angle_scan_impl(
            theta, scale, gamma, b, h0, perm, sgn, p2p,
            has_scale=has_scale, has_decay=has_decay,
        )

    result, reason = _resolve_triton_dispatch(
        mode, "the angle-fused mixer path", _call
    )
    if reason is None:
        return result

    global _warned_angle_fallback
    if not _warned_angle_fallback:
        warnings.warn(
            "MINGRU_SCAN=auto fell back to the eager mixer forward "
            f"(angle-fused path) despite CUDA inputs: {reason}",
            stacklevel=3,
        )
        _warned_angle_fallback = True
    return None


def _cached_plane_meta_per_device(obj, attr: str, device: torch.device, build):
    """Return ``build()``'s ``(perm, sgn, p2p)`` tensors on ``device``, cached.

    Shared by ``GivensMinGRU._angle_plane_meta`` and
    ``RotationMinGRU._angle_plane_meta`` -- both mixers cache small constant
    plane-metadata tensors for the angle-fused kernel on an instance attribute
    keyed by device, rebuilding only on a cache miss (e.g. the module's first
    call on a given device, or after a ``.to(device)`` move). ``build`` returns
    the CPU tensors fresh on each cache miss; this data is constant (no RNG),
    so caching it never perturbs the eager path or the parameter RNG stream.

    Parameters
    ----------
    obj : object
        The mixer instance (``self``); the cache lives on ``getattr(obj, attr)``.
    attr : str
        Instance attribute name to cache the ``(perm, sgn, p2p)`` tuple under.
    device : torch.device
        Target device for the cached tensors.
    build : Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        Zero-arg callable returning fresh CPU ``(perm, sgn, p2p)`` tensors.

    Returns
    -------
    tuple of torch.Tensor
        ``(perm, sgn, p2p)`` on ``device``.
    """
    cache = getattr(obj, attr, None)
    if cache is not None and cache[0].device == device:
        return cache
    meta = tuple(t.to(device) for t in build())
    setattr(obj, attr, meta)
    return meta


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

    References
    ----------
    Heinsen (2023), "Efficient Parallelization of a Ubiquitous Sequential
    Computation" -- the log-space cumulative-sum / logcumsumexp scan.

    Notes
    -----
    Under ``MINGRU_SCAN=auto`` (the default) on CUDA, this op dispatches to
    a Triton kernel that is measured SLOWER than eager for a full forward+
    backward training step at benchmarked shapes (0.70-0.80x eager; see
    ``experiments/bench/scan_bench.md``), even though its forward-only
    latency is roughly on par with eager. ``MINGRU_SCAN=eager`` is the
    right choice for training ``MinGRU``/``SignedMinGRU`` today.

    Examples
    --------
    >>> import torch
    >>> from mingru import parallel_scan_log
    >>> B, T, D = 2, 5, 3
    >>> log_coeffs = -torch.rand(B, T, D)          # log(a_t), a_t in (0, 1)
    >>> log_values = -torch.rand(B, T + 1, D)      # log(h_0) and log(b_t)
    >>> h = parallel_scan_log(log_coeffs, log_values)
    >>> tuple(h.shape)
    (2, 5, 3)
    """
    result = _dispatch_scan("parallel_scan_log", log_coeffs, log_values)
    if result is not None:
        return result
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
# log-space scan to NaN; 1e10 was verified (by the NaN/inf test in the
# repo-root `min_gru.py` evidence driver's `__main__` selftest -- run via
# `python min_gru.py` from a checkout, not shipped in the wheel)
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
    self-test in the repo-root ``min_gru.py`` evidence driver's
    ``__main__`` (run via ``python min_gru.py`` from a checkout) depends
    on these names and their construction order.
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
            all ``delta_t >= 0``: a non-positive fixed
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

    References
    ----------
    Feng et al., "Were RNNs All We Needed?", arXiv:2410.01201.
    Heinsen (2023) -- the log-space parallel scan used by ``forward``.

    Examples
    --------
    >>> import torch
    >>> from mingru import MinGRU
    >>> layer = MinGRU(input_size=32, hidden_size=64)
    >>> x = torch.randn(4, 128, 32)
    >>> h = layer(x)                       # parallel training-mode forward
    >>> tuple(h.shape)
    (4, 128, 64)
    >>> h_t = layer.step(x[:, 0])          # one O(1)-memory streaming step
    >>> tuple(h_t.shape)
    (4, 64)
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

    Examples
    --------
    >>> import torch
    >>> from mingru import linear_scan
    >>> a = torch.rand(2, 5, 3) * 2 - 1        # a_t in (-1, 1), signed
    >>> b = torch.randn(2, 5, 3)
    >>> A, Bc = linear_scan(a, b)
    >>> tuple(A.shape), tuple(Bc.shape)
    ((2, 5, 3), (2, 5, 3))
    """
    result = _dispatch_scan("linear_scan", a, b)
    if result is not None:
        return result
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

    Examples
    --------
    >>> import torch
    >>> from mingru import matrix_scan
    >>> B, T, n = 2, 5, 4
    >>> M = torch.eye(2).expand(B, T, n, 2, 2)     # identity transitions
    >>> b = torch.randn(B, T, n, 2)
    >>> A, Bc = matrix_scan(M, b)
    >>> tuple(A.shape), tuple(Bc.shape)
    ((2, 5, 4, 2, 2), (2, 5, 4, 2))
    """
    result = _dispatch_scan("matrix_scan", M, b)
    if result is not None:
        return result
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

    Examples
    --------
    >>> import torch
    >>> from mingru import matrix_affine_scan
    >>> B, T, n, k, v = 2, 5, 4, 8, 1
    >>> A = torch.eye(k).expand(B, T, n, k, k)     # identity transitions
    >>> Bm = torch.randn(B, T, n, k, v)
    >>> Abar, Bbar = matrix_affine_scan(A, Bm)
    >>> tuple(Abar.shape), tuple(Bbar.shape)
    ((2, 5, 4, 8, 8), (2, 5, 4, 8, 1))
    """
    result = _dispatch_scan("matrix_affine_scan", A, Bm)
    if result is not None:
        return result
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

    References
    ----------
    Merrill, Petty & Sabharwal (2024) -- expressivity gains from signed
    (negative-eigenvalue) diagonal linear recurrences.
    Grazzi et al. (2025) -- sign-alternation / parity dynamics enabled by
    negative transition eigenvalues.

    Examples
    --------
    >>> import torch
    >>> from mingru import SignedMinGRU
    >>> layer = SignedMinGRU(input_size=32, hidden_size=64)  # decoupled default
    >>> x = torch.randn(4, 128, 32)
    >>> tuple(layer(x).shape)
    (4, 128, 64)
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
    group elements -- exactly for the snapped angle; the scale channel
    ``tanh(u_t)`` reaches a group element (+-1) only as the ``u``-head
    saturates, so full-map exactness is empirical -- the same way
    ``tanh``'s asymptote manufactures an
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

    References
    ----------
    Merrill, Petty & Sabharwal (2024) and Grazzi et al. (2025) -- state
    tracking over non-abelian groups needs non-commutative (matrix)
    transitions, which a diagonal scan cannot express.

    Examples
    --------
    >>> import torch
    >>> from mingru import RotationMinGRU
    >>> layer = RotationMinGRU(input_size=32, hidden_size=64)  # snap=(2,3,4,6)
    >>> x = torch.randn(4, 128, 32)
    >>> tuple(layer(x).shape)
    (4, 128, 64)
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
        fused = self._angle_fused_forward(x, h0_blocks, dt)
        if fused is not None:
            return fused
        M, b = self._coeffs(x, dt)
        A, Bc = matrix_scan(M, b)
        h = torch.einsum("btnij,bnj->btni", A, h0_blocks) + Bc
        return h.reshape(B, T, self.hidden_size)

    def _angle_plane_meta(self, device: torch.device):
        """The single-plane ``(perm, sgn, p2p)`` metadata for the angle kernel.

        ``RotationMinGRU`` is always ``k=2``, ``rounds=1``, one plane -- the
        metadata is a fixed constant, but still cached per device (mirrors
        ``GivensMinGRU._angle_plane_meta``'s caching, via the shared
        ``_cached_plane_meta_per_device`` helper) rather than rebuilt on every
        ``forward`` call.
        """

        def build():
            return (
                torch.tensor([[1, 0]], dtype=torch.int32),
                torch.tensor([[-1.0, 1.0]], dtype=torch.float32),
                torch.tensor([[0, 0]], dtype=torch.int32),
            )

        return _cached_plane_meta_per_device(self, "_angle_meta_cache", device, build)

    def _angle_heads(
        self, x: torch.Tensor, dt: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble the angle-fused kernel's per-step heads.

        Pure extraction of ``_angle_fused_forward``'s head-assembly math
        (no CUDA gate, no kernel dispatch) -- lets a CPU-runnable selftest
        (the repo-root ``min_gru.py`` evidence driver's ``__main__``, "CPU
        lockstep guard" section) call this directly and cross-check its
        reconstructed transition against ``_coeffs``'s ``(M, b)``, the
        MAINTENANCE lockstep this method's caller-side comment warns about.

        MAINTENANCE: this head derivation (``linear_theta``, snap STE,
        ``linear_u`` -> ``tanh``, ``linear_z``/``linear_h`` -> ``b``) must be
        kept byte-for-byte in lockstep with ``RotationMinGRU._coeffs``.
        ``_coeffs`` is frozen for evidence invariance (its eager op sequence
        must not be reordered or refactored -- see the class/module
        docstrings), so it is re-derived here rather than shared; any future
        change to ``_coeffs``'s head math must be mirrored in this method by
        hand. Two checks catch divergence: the GPU-only module-level
        angle-fused parity selftest (``triton_scans._run_angle_fused_parity``)
        and the CPU-runnable lockstep assertion in the repo-root
        ``min_gru.py`` evidence driver's ``__main__`` (run via ``python
        min_gru.py`` from a checkout, needs no GPU/Triton, so ordinary CI
        and the GPU-less Phase-4 wheel CI both catch drift too).

        Returns
        -------
        tuple of torch.Tensor
            ``theta`` ``(B, T, n_blocks, 1, 1)`` post-snap; ``scale`` =
            ``tanh(u_t)`` ``(B, T, n_blocks)``; ``gamma`` ``(B, T, n_blocks)``
            (all ones if ``dt`` is None); ``b`` ``(B, T, n_blocks, 2)``.
        """
        B, T, _ = x.shape
        n = self.n_blocks
        theta = self.linear_theta(x)
        if self.snap is not None:
            snapped = torch.round(theta / self.snap_step) * self.snap_step
            theta = theta + (snapped - theta).detach()
        theta = theta.view(B, T, n, 1, 1)
        scale = torch.tanh(self.linear_u(x))  # d_t, (B, T, n)
        b = (torch.sigmoid(self.linear_z(x)) * self.linear_h(x)).reshape(B, T, n, 2)
        if dt is not None:
            gamma = self._decay_gamma(dt).expand(B, T, n)
        else:
            gamma = torch.ones(B, T, n, dtype=x.dtype, device=x.device)
        return theta, scale, gamma, b

    def _angle_fused_forward(
        self, x: torch.Tensor, h0_blocks: torch.Tensor, dt: torch.Tensor | None
    ) -> torch.Tensor | None:
        """CUDA angle-fused fast path for ``forward`` (returns None off-path).

        On CUDA under ``MINGRU_SCAN=auto``/``triton`` this assembles the raw
        rotation angle, the ``tanh(u)`` scale channel, injection, decay, and
        ``h_0`` (via ``_angle_heads``) and routes them through the
        angle-fused Triton kernel, never materializing the 2x2 transition
        matrices ``matrix_scan`` is defined over. On CPU or under ``eager``
        it returns ``None`` without importing ``triton_scans``, so
        ``forward`` runs its unchanged eager path (the recorded evidence path
        stays byte-identical). The snap STE stays inside ``_angle_heads``
        upstream -- the kernel consumes POST-snap angles.

        The reversible backward reads every ``h_{t-1}`` exactly from the
        stored forward output -- never by dividing out ``gamma``/``tanh(u)``.
        An earlier design reversed across a multi-step checkpoint by dividing,
        which is unsound here: the per-block scale ``diag(1, tanh(u))`` is
        near-singular whenever ``tanh(u_t) ~ 0`` (typical at init), so that
        division would amplify roundoff past the grad tolerance. The user's
        ruling (Task-5 report, "Fix round 2") rejected division-based reversal
        for both mixers, so ``GivensMinGRU`` now uses the same exact-recompute
        backward.
        """
        if not _angle_scan_should_try(x.is_cuda):
            return None
        B, T, _ = x.shape
        theta, scale, gamma, b = self._angle_heads(x, dt)
        has_decay = 1 if dt is not None else 0
        perm, sgn, p2p = self._angle_plane_meta(x.device)
        h = _dispatch_angle_scan(
            theta, scale, gamma, b, h0_blocks, perm, sgn, p2p,
            has_scale=1, has_decay=has_decay,
        )
        if h is None:
            return None
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

    References
    ----------
    Jing et al. (2017), "Tunable Efficient Unitary Neural Networks (EUNN)",
    arXiv:1612.05231 -- the brick-wall mesh of plane rotations.
    Clements et al. (2016) -- the rectangular (brick-wall) mesh design.
    Golub & Van Loan, *Matrix Computations* -- Givens plane rotations.

    Examples
    --------
    >>> import torch
    >>> from mingru import GivensMinGRU
    >>> layer = GivensMinGRU(input_size=32, hidden_size=64)  # block_size=8, rounds=3
    >>> x = torch.randn(4, 128, 32)
    >>> tuple(layer(x).shape)
    (4, 128, 64)
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
        fused = self._angle_fused_forward(x, h0_blocks, dt)
        if fused is not None:
            return fused
        M, b = self._coeffs(x, dt)
        Abar, Bbar = matrix_affine_scan(M, b.unsqueeze(-1))
        h = torch.einsum("btnij,bnj->btni", Abar, h0_blocks) + Bbar.squeeze(-1)
        return h.reshape(B, T, self.hidden_size)

    def _angle_plane_meta(self, device: torch.device):
        """Per-round ``(perm, sgn, p2p)`` plane metadata for the angle kernel.

        Derived (via the shared ``_cached_plane_meta_per_device`` helper, and
        cached per device) from the registered brick-wall plane buffers
        ``_pi{r}``/``_pj{r}``: ``perm[r, p]`` is position ``p``'s partner
        within round ``r``'s planes, ``sgn[r, p]`` is ``-1`` for the first
        (``i``) member and ``+1`` for the second (``j``), and ``p2p[r, p]``
        is the index of ``p``'s plane among the round's ``k/2`` angles.
        Constant data (no RNG), so building it here never perturbs the eager
        path or the parameter RNG stream.
        """

        def build():
            R, k = self.rounds, self.k
            perm = torch.empty(R, k, dtype=torch.int32)
            sgn = torch.empty(R, k, dtype=torch.float32)
            p2p = torch.empty(R, k, dtype=torch.int32)
            for r in range(R):
                i_idx = getattr(self, f"_pi{r}").cpu()
                j_idx = getattr(self, f"_pj{r}").cpu()
                for m in range(len(i_idx)):
                    i, j = int(i_idx[m]), int(j_idx[m])
                    perm[r, i], perm[r, j] = j, i
                    sgn[r, i], sgn[r, j] = -1.0, 1.0
                    p2p[r, i] = m
                    p2p[r, j] = m
            return perm, sgn, p2p

        return _cached_plane_meta_per_device(self, "_angle_meta_cache", device, build)

    def _angle_heads(
        self, x: torch.Tensor, dt: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble the angle-fused kernel's per-step heads.

        Pure extraction of ``_angle_fused_forward``'s head-assembly math (no
        CUDA gate, no kernel dispatch) -- lets a CPU-runnable selftest
        (the repo-root ``min_gru.py`` evidence driver's ``__main__``, "CPU
        lockstep guard" section) call this directly and cross-check its
        reconstructed transition against ``_coeffs``'s ``(M, b)``, the
        MAINTENANCE lockstep this method's caller-side comment warns about.

        MAINTENANCE: this head derivation (``linear_theta`` view,
        ``linear_z``/``linear_h`` -> ``b``) must be kept byte-for-byte in
        lockstep with ``GivensMinGRU._coeffs``. ``_coeffs`` is frozen for
        evidence invariance (its eager op sequence must not be reordered or
        refactored -- see the class/module docstrings), so it is re-derived
        here rather than shared; any future change to ``_coeffs``'s head math
        must be mirrored in this method by hand. Two checks catch
        divergence: the GPU-only module-level angle-fused parity selftest
        (``triton_scans._run_angle_fused_parity``) and the CPU-runnable
        lockstep assertion in the repo-root ``min_gru.py`` evidence driver's
        ``__main__`` (run via ``python min_gru.py`` from a checkout, needs no
        GPU/Triton, so ordinary CI and the GPU-less Phase-4 wheel CI both
        catch drift too).

        Returns
        -------
        tuple of torch.Tensor
            ``theta`` ``(B, T, n_blocks, rounds, k // 2)``; ``scale`` = all
            ones ``(B, T, n_blocks)`` (the Givens transition has no per-block
            scale; kept only for the angle-fused kernel's shared calling
            convention with ``RotationMinGRU``); ``gamma`` ``(B, T,
            n_blocks)`` (all ones if ``dt`` is None); ``b`` ``(B, T,
            n_blocks, k)``.
        """
        B, T, _ = x.shape
        n, k, R = self.n_blocks, self.k, self.rounds
        theta = self.linear_theta(x).view(B, T, n, R, k // 2)
        b = (torch.sigmoid(self.linear_z(x)) * self.linear_h(x)).reshape(B, T, n, k)
        if dt is not None:
            gamma = self._decay_gamma(dt).expand(B, T, n)
        else:
            gamma = torch.ones(B, T, n, dtype=x.dtype, device=x.device)
        scale = torch.ones(B, T, n, dtype=x.dtype, device=x.device)
        return theta, scale, gamma, b

    def _angle_fused_forward(
        self, x: torch.Tensor, h0_blocks: torch.Tensor, dt: torch.Tensor | None
    ) -> torch.Tensor | None:
        """CUDA angle-fused fast path for ``forward`` (returns None off-path).

        On CUDA under ``MINGRU_SCAN=auto``/``triton`` this assembles the raw
        per-round Givens angles, injection, decay, and ``h_0`` (via
        ``_angle_heads``) and routes them through the angle-fused Triton
        kernel, never materializing or scanning the ``k x k`` transition
        matrices ``matrix_affine_scan`` is defined over. On CPU or under
        ``eager`` it returns ``None`` without importing ``triton_scans`` (the
        recorded evidence path stays byte-identical). The Givens transition
        has no per-block scale, so ``has_scale`` is 0.

        The reversible backward reads every ``h_{t-1}`` exactly from the
        stored forward output -- never by inverting the forward step. An
        earlier design reversed across a multi-step checkpoint (``C=64``,
        exploiting the Givens transition's exact orthogonality) by applying
        the inverse rotation and dividing by ``gamma``; a blind CPU probe
        showed that division still amplifies roundoff by roughly
        ``gamma^{-chunklen}``, which blew past the grad tolerance even at this
        class's own default ``decay_rate=1.0`` (see the Task-5 report,
        "Fix round 2"). The user's ruling rejected division-based reversal
        entirely, so this mixer now uses the same exact stored-state backward
        as ``RotationMinGRU``, with no checkpoint-interval knob left to
        re-enable reversal.
        """
        if not _angle_scan_should_try(x.is_cuda):
            return None
        B, T, _ = x.shape
        theta, scale, gamma, b = self._angle_heads(x, dt)
        has_decay = 1 if dt is not None else 0
        perm, sgn, p2p = self._angle_plane_meta(x.device)
        h = _dispatch_angle_scan(
            theta, scale, gamma, b, h0_blocks, perm, sgn, p2p,
            has_scale=0, has_decay=has_decay,
        )
        if h is None:
            return None
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

    Examples
    --------
    >>> import torch
    >>> from mingru import MinGRUBlock
    >>> block = MinGRUBlock(d_model=64, mixer="signed")
    >>> x = torch.randn(4, 128, 64)
    >>> out, state = block(x)              # always returns (output, state)
    >>> tuple(out.shape), tuple(state.shape)
    ((4, 128, 64), (4, 1, 64))
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

    Examples
    --------
    >>> import torch
    >>> from mingru import MinGRUStack
    >>> model = MinGRUStack(input_size=32, d_model=64, n_layers=3, mixer="log")
    >>> x = torch.randn(4, 128, 32)
    >>> out, state = model(x)             # state: one entry per block
    >>> tuple(out.shape), len(state)
    ((4, 128, 64), 3)
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
