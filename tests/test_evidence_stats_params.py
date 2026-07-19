"""Regression tests for ``scripts/_evidence_stats.py``'s stats
parameterization (2026-07-19 benchmark-round design spec section 6,
"Stats parameterization").

``arm_stats`` now accepts ``(metric_key, threshold, direction,
robustness)`` so the accepted-benchmark round's tasks -- whose fit
metrics live under different ``ckpt`` keys, and whose pendulum task is
"lower is better" (MSE) rather than "higher is better" (accuracy) --
can reuse the same engine. The load-bearing guarantee this file pins:
calling with no new arguments must reproduce exactly what the S3
matched-state round's hardcoded ``FIT_THRESHOLD``/``ROBUSTNESS_
THRESHOLDS``/``val128``/``>=`` behavior always computed -- checked both
against a captured ``run_matched_state.py report`` fixture (byte-
identical) and against direct ``arm_stats`` calls on synthetic rows.

``scripts/`` is not an importable package (see ``tests/test_run_matched
_state.py``'s docstring for the same rationale), so the module is
loaded by file path via ``importlib``, exactly as that sibling test
file does.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def _load_by_path(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


_evidence_stats = _load_by_path("_evidence_stats", "_evidence_stats.py")

arm_stats = _evidence_stats.arm_stats
FIT_THRESHOLD = _evidence_stats.FIT_THRESHOLD
ROBUSTNESS_THRESHOLDS = _evidence_stats.ROBUSTNESS_THRESHOLDS


def _row(seed: int, val128: float, acc64: float, acc256: float, acc512: float, acc1024: float):
    """A minimal synthetic ledger row -- only the fields ``arm_stats`` reads."""
    return {
        "round": "synthetic-round",
        "seed": seed,
        "acc": {"64": acc64, "256": acc256, "512": acc512, "1024": acc1024},
        "ckpt": {"step": 100, "val128": val128},
    }


def _mse_row(seed: int, val_mse: float):
    """A minimal synthetic MSE-style ledger row. ``arm_stats`` unconditionally
    computes ``mean_acc``/``fit_only_acc`` over ``ACC_LENGTHS`` regardless of
    ``metric_key`` (that dict's shape is out of this task's parameterization
    scope -- see the module docstring), so placeholder ``acc`` T-keys are
    still required even though these tests only exercise fit counting and
    robustness."""
    return {
        "round": "synthetic-mse-round",
        "seed": seed,
        "acc": {"64": 0.0, "256": 0.0, "512": 0.0, "1024": 0.0},
        "ckpt": {"step": 100, "val_mse": val_mse},
    }


# --- default-argument behavior: byte-identical to today's hardcoded S3 ----


def test_arm_stats_defaults_match_explicit_s3_values():
    # Calling with no new arguments must be indistinguishable from calling
    # with the S3 values spelled out explicitly.
    rows = [
        _row(0, val128=1.00, acc64=1.0, acc256=0.9, acc512=0.8, acc1024=0.7),
        _row(1, val128=0.50, acc64=0.5, acc256=0.4, acc512=0.3, acc1024=0.2),
        _row(2, val128=0.99, acc64=0.99, acc256=0.95, acc512=0.9, acc1024=0.85),
    ]

    defaulted = arm_stats(rows)
    explicit = arm_stats(
        rows,
        metric_key="val128",
        threshold=FIT_THRESHOLD,
        direction="ge",
        robustness=ROBUSTNESS_THRESHOLDS,
    )

    assert defaulted == explicit
    assert defaulted.fits == 2
    assert defaulted.robustness == {0.98: 2, 0.99: 2, 0.995: 1}


def test_report_output_byte_identical_to_captured_baseline():
    # The captured baseline was recorded from `uv run python
    # scripts/run_matched_state.py report` before this task's
    # parameterization; per the task brief, existing callers pass
    # nothing new, so the report subcommand's output must not move by a
    # single byte.
    fixture = Path(__file__).resolve().parent / "fixtures" / "run_matched_state_report.txt"
    baseline = fixture.read_text()

    result = subprocess.run(
        [sys.executable, str(_SCRIPTS_DIR / "run_matched_state.py"), "report"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == baseline


# --- direction="le": MSE-style "lower is better" fit counting -------------


def test_le_direction_fit_counting_for_mse_style_metric():
    rows = [
        _mse_row(0, val_mse=0.05),  # fits: below tau
        _mse_row(1, val_mse=0.20),  # does not fit: above tau
        _mse_row(2, val_mse=0.10),  # fits: exactly at tau (boundary is inclusive)
    ]
    tau = 0.10

    stats = arm_stats(rows, metric_key="val_mse", threshold=tau, direction="le")

    assert stats.seeds == 3
    assert stats.fits == 2  # seeds 0 and 2 clear <= tau; seed 1 does not


def test_le_direction_robustness_triple_orders_opposite_of_ge():
    # Pendulum-style robustness triple {1.25*tau, tau, 0.8*tau}: the
    # loosest (largest) threshold admits the most fits under "le", the
    # opposite ordering from a "ge" accuracy-style triple.
    tau = 0.10
    rows = [_mse_row(i, val_mse=v) for i, v in enumerate([0.07, 0.09, 0.11, 0.15])]

    stats = arm_stats(
        rows,
        metric_key="val_mse",
        threshold=tau,
        direction="le",
        robustness=(1.25 * tau, tau, 0.8 * tau),
    )

    assert stats.robustness[1.25 * tau] == 3  # 0.07, 0.09, 0.11 <= 0.125
    assert stats.robustness[tau] == 2  # 0.07, 0.09 <= 0.10
    assert stats.robustness[0.8 * tau] == 1  # only 0.07 <= 0.08


def test_unknown_direction_raises_value_error():
    rows = [_mse_row(0, val_mse=0.05)]
    try:
        arm_stats(rows, metric_key="val_mse", threshold=0.1, direction="lt")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "lt" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown fit direction")
