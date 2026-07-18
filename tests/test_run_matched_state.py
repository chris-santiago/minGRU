"""Regression tests for ``scripts/run_matched_state.py`` and its hoisted
``scripts/_evidence_stats.py`` publication-stats engine.

``scripts/`` is not an importable package (see ``tests/test_scaling_probe
.py``'s docstring for the same rationale), so both modules are loaded
directly by file path via ``importlib`` -- separate from
``test_scaling_probe.py`` (a concurrent task's file, not touched here).

Covers only the stdlib-pure logic (no subprocess spawning, no torch):
- ``_evidence_stats.fisher_exact_two_sided`` against four
  TECHNICAL_REPORT-recorded contingency tables.
- ``_evidence_stats.givens_composer_params``/``delta_composer_params``
  against the recorded 14,624 / 3,306 / 25,480 figures.
- ``_evidence_stats.arm_stats`` on synthetic ledger rows (fit counts,
  mean accs, fit-only means, threshold-robustness), including the
  0-fits and empty-row-set edge cases.
- ``run_matched_state._parse_peak_rss_bytes`` on a real and a malformed
  ``/usr/bin/time -l`` stderr capture.
- ``run_matched_state._find_existing_seeds`` (stays in
  ``run_matched_state.py`` -- it's runner-specific, not part of the
  stats engine) on a synthetic ledger file, including the
  no-false-positive-on-other-rounds and empty-ledger cases.

The `report` subcommand's markdown assembly and the pooled `run`
subcommand's subprocess orchestration are exercised instead by the
task's manual dry-run/report verification (spawning real child
processes, or a full ledger round-trip, is the point there; it does not
belong in a fast unit suite).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _load_by_path(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPTS_DIR / file_name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


_evidence_stats = _load_by_path("_evidence_stats", "_evidence_stats.py")
run_matched_state = _load_by_path("run_matched_state", "run_matched_state.py")

fisher_exact_two_sided = _evidence_stats.fisher_exact_two_sided
givens_composer_params = _evidence_stats.givens_composer_params
delta_composer_params = _evidence_stats.delta_composer_params
arm_stats = _evidence_stats.arm_stats

_parse_peak_rss_bytes = run_matched_state._parse_peak_rss_bytes
_find_existing_seeds = run_matched_state._find_existing_seeds
_Arm = run_matched_state._Arm


def _row(seed: int, val128: float, acc64: float, acc256: float, acc512: float, acc1024: float):
    """A minimal synthetic ledger row -- only the fields ``arm_stats`` reads."""
    return {
        "round": "synthetic-round",
        "seed": seed,
        "acc": {"64": acc64, "256": acc256, "512": acc512, "1024": acc1024},
        "ckpt": {"step": 100, "val128": val128},
    }


# --- fisher_exact_two_sided: four TECHNICAL_REPORT-recorded contingency
# tables (section 4.4), each cross-checked here to 1e-9 -------------------


def test_fisher_delta64_vs_givens64_small_delta_contrast():
    # Givens 8/12 vs small-delta 4/12 -- TECHNICAL_REPORT section 4.4's
    # "suggestive only" cross-mechanism comparison, p ~= 0.22.
    p = fisher_exact_two_sided(8, 4, 4, 8)
    assert abs(p - 0.2203467551) < 1e-9


def test_fisher_givens64_vs_rotation_continuous():
    # Givens 8/12 vs continuous 2D rotation 1/12 -- section 4.4 finding 2,
    # p ~= 0.0094.
    p = fisher_exact_two_sided(8, 4, 1, 11)
    assert abs(p - 0.0094225333) < 1e-9


def test_fisher_identical_tables_is_one():
    # Identical fit rates (8/12 vs 8/12) must be non-significant, p == 1.0.
    p = fisher_exact_two_sided(8, 4, 8, 4)
    assert abs(p - 1.0) < 1e-9


def test_fisher_rounds_endpoint_separation():
    # rounds=3 (8/12) vs rounds=1 (0/12) -- section 4.4's rounds-ablation
    # endpoint separation, p ~= 0.0013.
    p = fisher_exact_two_sided(8, 4, 0, 12)
    assert abs(p - 0.0013460762) < 1e-9


def test_fisher_two_sided_is_symmetric():
    # Swapping the two rows (which arm is "a" vs "c") must not change the
    # two-sided p-value.
    p_ab = fisher_exact_two_sided(8, 4, 1, 11)
    p_ba = fisher_exact_two_sided(1, 11, 8, 4)
    assert abs(p_ab - p_ba) < 1e-9


# --- composer parameter-count formulas ------------------------------------


def test_givens_composer_params_matches_recorded_14624():
    # TECHNICAL_REPORT section 4.4's recorded GivensMinGRU (givens8
    # composer) parameter count, at the recorded arm's config.
    assert givens_composer_params(hidden_size=64, block_size=8, rounds=3) == 14_624


def test_delta64_composer_params_matches_recorded_3306():
    # TECHNICAL_REPORT section 4.4's recorded matched-64-state delta
    # composer parameter count (the delta@64-matched arm's config).
    assert delta_composer_params(n_heads=1, nh=2, d_k=8, d_v=8) == 3_306


def test_delta1024_composer_params_matches_recorded_25480():
    # TECHNICAL_REPORT section 4.3's recorded `DeltaNetMixer nh=2 (4
    # heads)` parameter count -- an independent cross-check since the
    # delta@1024 arm's config (n_heads=4, nh=2, d_k=d_v=16) mirrors it.
    assert delta_composer_params(n_heads=4, nh=2, d_k=16, d_v=16) == 25_480


# --- arm_stats: synthetic ledger rows --------------------------------------


def test_arm_stats_exact_counts_and_means():
    rows = [
        _row(0, val128=1.00, acc64=1.0, acc256=0.9, acc512=0.8, acc1024=0.7),
        _row(1, val128=0.50, acc64=0.5, acc256=0.4, acc512=0.3, acc1024=0.2),
        _row(2, val128=0.99, acc64=0.99, acc256=0.95, acc512=0.9, acc1024=0.85),
    ]
    stats = arm_stats(rows)

    assert stats.seeds == 3
    assert stats.fits == 2  # val128 >= 0.99: seed 0 (1.00) and seed 2 (0.99)

    assert stats.mean_acc[64] == (1.0 + 0.5 + 0.99) / 3
    assert stats.mean_acc[256] == (0.9 + 0.4 + 0.95) / 3
    assert stats.mean_acc[512] == (0.8 + 0.3 + 0.9) / 3
    assert stats.mean_acc[1024] == (0.7 + 0.2 + 0.85) / 3

    # fit-only means: seeds 0 and 2 only (seed 1 does not fit)
    assert stats.fit_only_acc[512] == (0.8 + 0.9) / 2
    assert stats.fit_only_acc[1024] == (0.7 + 0.85) / 2

    # threshold-robustness: val128s are {1.00, 0.50, 0.99}
    assert stats.robustness[0.98] == 2  # 1.00, 0.99 both >= 0.98
    assert stats.robustness[0.99] == 2  # 1.00, 0.99 both >= 0.99
    assert stats.robustness[0.995] == 1  # only 1.00 >= 0.995 (0.99 < 0.995)


def test_arm_stats_zero_fits():
    rows = [
        _row(0, val128=0.60, acc64=0.6, acc256=0.5, acc512=0.4, acc1024=0.3),
        _row(1, val128=0.40, acc64=0.4, acc256=0.3, acc512=0.2, acc1024=0.1),
    ]
    stats = arm_stats(rows)

    assert stats.seeds == 2
    assert stats.fits == 0
    assert stats.robustness == {0.98: 0, 0.99: 0, 0.995: 0}
    # mean_acc is still computed over all (fitless) rows...
    assert stats.mean_acc[64] == (0.6 + 0.4) / 2
    # ...but fit-only means are undefined with no fitting rows.
    assert stats.fit_only_acc[512] is None
    assert stats.fit_only_acc[1024] is None


def test_arm_stats_empty_row_set():
    stats = arm_stats([])

    assert stats.seeds == 0
    assert stats.fits == 0
    assert stats.robustness == {0.98: 0, 0.99: 0, 0.995: 0}
    assert stats.mean_acc == {64: None, 256: None, 512: None, 1024: None}
    assert stats.fit_only_acc == {512: None, 1024: None}


# --- _parse_peak_rss_bytes: real and malformed `/usr/bin/time -l` stderr -


def test_parse_peak_rss_bytes_real_macos_output():
    # Verbatim shape of `/usr/bin/time -l`'s report on macOS (see the
    # module docstring's capture convention) -- the value is already in
    # bytes on this platform, parsed as-is.
    stderr = (
        "        0.11 real         0.00 user         0.00 sys\n"
        "             1196032  maximum resident set size\n"
        "                   0  average shared memory size\n"
        "                 211  page reclaims\n"
    )
    assert _parse_peak_rss_bytes(stderr) == 1196032


def test_parse_peak_rss_bytes_missing_line_returns_none():
    # No "maximum resident set size" line at all (e.g. a spawn failure
    # never reached `time`'s report, or a non-macOS platform where the
    # child ran unwrapped) -- must return None, not raise or fabricate 0.
    stderr = "FAILED to spawn child (['uv', 'run', ...]): [Errno 2] No such file or directory\n"
    assert _parse_peak_rss_bytes(stderr) is None


def test_parse_peak_rss_bytes_empty_string_returns_none():
    assert _parse_peak_rss_bytes("") is None


# --- _find_existing_seeds: synthetic ledger vs requested cells -----------


def _write_ledger(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "lab_results.jsonl"
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_find_existing_seeds_reports_exact_conflicts(tmp_path):
    arm_a = _Arm("a", "hetero-loop-20-pd64", "hetero-pd64")
    arm_b = _Arm("b", "hetero-loop-21-pd1024", "hetero-pd1024")
    ledger = _write_ledger(
        tmp_path,
        [
            {"round": "hetero-loop-20-pd64", "seed": 0, "acc": {}, "ckpt": {"val128": 1.0}},
            {"round": "hetero-loop-21-pd1024", "seed": 1, "acc": {}, "ckpt": {"val128": 1.0}},
        ],
    )
    cells = [(arm_a, 0), (arm_a, 1), (arm_b, 0), (arm_b, 1)]

    conflicts = _find_existing_seeds(ledger, cells)

    assert set(conflicts) == {("hetero-loop-20-pd64", 0), ("hetero-loop-21-pd1024", 1)}


def test_find_existing_seeds_no_false_positive_on_other_rounds(tmp_path):
    # Rows for an unrelated round (the recorded givens arm) at the SAME
    # seeds as the requested cells must never register as a conflict --
    # a duplicate check keyed only on `seed` (ignoring `round`) would
    # false-positive here and block a legitimate first run.
    arm_a = _Arm("a", "hetero-loop-20-pd64", "hetero-pd64")
    ledger = _write_ledger(
        tmp_path,
        [
            {"round": "hetero-loop-17-sg8", "seed": 0, "acc": {}, "ckpt": {"val128": 1.0}},
            {"round": "hetero-loop-17-sg8", "seed": 1, "acc": {}, "ckpt": {"val128": 1.0}},
        ],
    )
    cells = [(arm_a, 0), (arm_a, 1)]

    conflicts = _find_existing_seeds(ledger, cells)

    assert conflicts == []


def test_find_existing_seeds_empty_ledger(tmp_path):
    arm_a = _Arm("a", "hetero-loop-20-pd64", "hetero-pd64")
    ledger = tmp_path / "does_not_exist.jsonl"
    cells = [(arm_a, 0), (arm_a, 1)]

    conflicts = _find_existing_seeds(ledger, cells)

    assert conflicts == []
