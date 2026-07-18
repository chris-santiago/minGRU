"""Publication-stats engine for the matched-state round's verdict table.

Two-sided Fisher exact (exact hypergeometric enumeration, stdlib
``math.comb``, no scipy), composer parameter-count formulas, and
per-arm ledger-row aggregation (fit counts, mean acc, fit-only acc,
threshold-robustness) -- the pure computation ``scripts/run_matched_state
.py``'s ``report`` subcommand prints as the spec section 6 /
TECHNICAL_REPORT section 4.4 verdict table. Hoisted out of that single
caller (design review S3) mirroring the ``scripts/_bench_env.py``
precedent from this round: a small, stdlib-only helper module, not a
dumping-ground ``utils.py``. Stdlib-only, no torch dependency.

Ledger-row shape assumed by ``arm_stats``: the unchanged ``run_arm``
schema from ``experiments/hetero_lab.py`` (``round``, ``task``,
``variant``, ``layers``, ``seed``, ``steps``, ``acc`` keyed by T as a
string, ``secs``, ``max_steps``, ``ckpt.{step, val128}``, ``config``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

FIT_THRESHOLD = 0.99
ROBUSTNESS_THRESHOLDS = (0.98, 0.99, 0.995)
ACC_LENGTHS = (64, 256, 512, 1024)
FIT_ONLY_LENGTHS = (512, 1024)


# --- Fisher exact (stdlib, exact hypergeometric enumeration) --------------


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]].

    Exact hypergeometric enumeration (stdlib ``math.comb``, no scipy),
    per the round's Global Constraint. Sums the hypergeometric
    probability of every table sharing the observed table's marginals
    whose probability is <= the observed table's probability (the
    standard two-sided definition; a small relative tolerance absorbs
    floating-point rounding at the boundary).
    """
    row1, row2 = a + b, c + d
    col1 = a + c
    n = row1 + row2
    denom = math.comb(n, col1)

    def _prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / denom

    lo, hi = max(0, col1 - row2), min(row1, col1)
    p_observed = _prob(a)
    tolerance = p_observed * 1e-7 + 1e-12
    return sum(_prob(x) for x in range(lo, hi + 1) if _prob(x) <= p_observed + tolerance)


# --- composer parameter counts ---------------------------------------------


def _linear_params(in_features: int, out_features: int, bias: bool = True) -> int:
    """``nn.Linear`` parameter count: ``in*out`` weights + ``out`` bias."""
    return in_features * out_features + (out_features if bias else 0)


def givens_composer_params(
    hidden_size: int = 64, block_size: int = 8, rounds: int = 3, input_size: int = 64
) -> int:
    """``GivensMinGRU`` parameter count, from ``experiments/hetero_lab.py``'s
    ``GivensMinGRU.__init__`` (mirrored by the packaged ``mingru.GivensMinGRU``,
    bit-identical per the lab's bridge selftest): ``linear_theta`` (bias,
    out = n_blocks*rounds*(block_size//2)) + ``linear_z`` + ``linear_h``
    (both ``hidden_size -> hidden_size``, bias) + ``h0`` (``hidden_size``
    free parameters, no weight/bias structure). At the recorded arm's
    config (block_size=8, rounds=3, hidden_size=64) this must reproduce
    the recorded 14,624 (TECHNICAL_REPORT section 4.4); callers should
    treat any drift from that value as a discrepancy to report, not to
    silently accept.
    """
    n_blocks = hidden_size // block_size
    half = block_size // 2
    theta_out = n_blocks * rounds * half
    return (
        _linear_params(input_size, theta_out)  # linear_theta
        + 2 * _linear_params(input_size, hidden_size)  # linear_z, linear_h
        + hidden_size  # h0
    )


def delta_composer_params(
    n_heads: int, nh: int, d_k: int, d_v: int, input_size: int = 64, hidden_size: int = 64
) -> int:
    """``DeltaMinGRU`` parameter count, from ``mingru.min_gru.DeltaMinGRU
    .__init__``: ``linear_q`` + ``nh``-length ``linear_k``/``linear_v``/
    ``linear_beta`` ModuleLists (each linear per micro-step, bias) +
    ``out_proj``. At the recorded delta@64 config (n_heads=1, nh=2,
    d_k=d_v=8) this must reproduce the recorded 3,306 (TECHNICAL_REPORT
    section 4.4 / next-round design doc); callers should treat any drift
    from that value as a discrepancy to report, not to silently accept.
    """
    return (
        _linear_params(input_size, n_heads * d_k)  # linear_q
        + nh * _linear_params(input_size, n_heads * d_k)  # linear_k[j]
        + nh * _linear_params(input_size, n_heads * d_v)  # linear_v[j]
        + nh * _linear_params(input_size, n_heads)  # linear_beta[j]
        + _linear_params(n_heads * d_v, hidden_size)  # out_proj
    )


# --- per-arm ledger-row aggregation -----------------------------------------


@dataclass
class ArmStats:
    seeds: int
    fits: int
    mean_acc: dict[int, float | None]
    fit_only_acc: dict[int, float | None]
    robustness: dict[float, int]


def arm_stats(rows: list[dict[str, Any]]) -> ArmStats:
    """Fit counts, mean acc, fit-only acc, threshold-robustness -- all
    computed from ``rows`` (never hand-transcribed), matching
    TECHNICAL_REPORT section 4.4's definitions: a fit is the selected
    checkpoint's val@128 >= threshold; acc@T is the mean over ALL rows;
    fit-only acc@T is the mean over fitting rows only.
    """
    n = len(rows)
    val128s = [row["ckpt"]["val128"] for row in rows]
    fit_rows = [row for row in rows if row["ckpt"]["val128"] >= FIT_THRESHOLD]

    def _mean_acc(subset: list[dict[str, Any]], t: int) -> float | None:
        if not subset:
            return None
        return sum(row["acc"][str(t)] for row in subset) / len(subset)

    return ArmStats(
        seeds=n,
        fits=len(fit_rows),
        mean_acc={t: _mean_acc(rows, t) for t in ACC_LENGTHS},
        fit_only_acc={t: _mean_acc(fit_rows, t) for t in FIT_ONLY_LENGTHS},
        robustness={th: sum(1 for v in val128s if v >= th) for th in ROBUSTNESS_THRESHOLDS},
    )
