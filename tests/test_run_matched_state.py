"""Regression tests for ``scripts/run_matched_state.py``'s pure helpers.

``scripts/`` is not an importable package (see ``tests/test_scaling_probe
.py``'s docstring for the same rationale), so this module is loaded
directly by file path via ``importlib`` -- a separate module from
``test_scaling_probe.py`` (a concurrent task's file, not touched here).

Covers only the stdlib-pure logic (no subprocess spawning, no torch,
no ledger I/O beyond in-memory strings/tuples): the Fisher exact
implementation against four TECHNICAL_REPORT-recorded contingency
tables, the ``/usr/bin/time -l`` peak-RSS parser on a real and a
malformed stderr capture, and the composer parameter-count formulas
against the recorded 14,624 / 3,306 / 25,480 figures. The `report`
subcommand's stats aggregation (``_arm_stats``) and the pooled `run`
subcommand's subprocess orchestration are exercised instead by the
task's manual dry-run/report verification (spawning real child
processes there is the point; it does not belong in a fast unit
suite).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_matched_state.py"
_spec = importlib.util.spec_from_file_location("run_matched_state", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
run_matched_state = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("run_matched_state", run_matched_state)
_spec.loader.exec_module(run_matched_state)

_fisher_exact_two_sided = run_matched_state._fisher_exact_two_sided
_parse_peak_rss_bytes = run_matched_state._parse_peak_rss_bytes
_givens_composer_params = run_matched_state._givens_composer_params
_delta_composer_params = run_matched_state._delta_composer_params


# --- _fisher_exact_two_sided: four TECHNICAL_REPORT-recorded contingency
# tables (section 4.4), each cross-checked here to 1e-9 -------------------


def test_fisher_delta64_vs_givens64_small_delta_contrast():
    # Givens 8/12 vs small-delta 4/12 -- TECHNICAL_REPORT section 4.4's
    # "suggestive only" cross-mechanism comparison, p ~= 0.22.
    p = _fisher_exact_two_sided(8, 4, 4, 8)
    assert abs(p - 0.2203467551) < 1e-9


def test_fisher_givens64_vs_rotation_continuous():
    # Givens 8/12 vs continuous 2D rotation 1/12 -- section 4.4 finding 2,
    # p ~= 0.0094.
    p = _fisher_exact_two_sided(8, 4, 1, 11)
    assert abs(p - 0.0094225333) < 1e-9


def test_fisher_identical_tables_is_one():
    # Identical fit rates (8/12 vs 8/12) must be non-significant, p == 1.0.
    p = _fisher_exact_two_sided(8, 4, 8, 4)
    assert abs(p - 1.0) < 1e-9


def test_fisher_rounds_endpoint_separation():
    # rounds=3 (8/12) vs rounds=1 (0/12) -- section 4.4's rounds-ablation
    # endpoint separation, p ~= 0.0013.
    p = _fisher_exact_two_sided(8, 4, 0, 12)
    assert abs(p - 0.0013460762) < 1e-9


def test_fisher_two_sided_is_symmetric():
    # Swapping the two rows (which arm is "a" vs "c") must not change the
    # two-sided p-value.
    p_ab = _fisher_exact_two_sided(8, 4, 1, 11)
    p_ba = _fisher_exact_two_sided(1, 11, 8, 4)
    assert abs(p_ab - p_ba) < 1e-9


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


# --- composer parameter-count formulas ------------------------------------


def test_givens_composer_params_matches_recorded_14624():
    # TECHNICAL_REPORT section 4.4's recorded GivensMinGRU (givens8
    # composer) parameter count, at the recorded arm's config.
    assert _givens_composer_params(hidden_size=64, block_size=8, rounds=3) == 14_624


def test_delta64_composer_params_matches_recorded_3306():
    # TECHNICAL_REPORT section 4.4's recorded matched-64-state delta
    # composer parameter count (the delta@64-matched arm's config).
    assert _delta_composer_params(n_heads=1, nh=2, d_k=8, d_v=8) == 3_306


def test_delta1024_composer_params_matches_recorded_25480():
    # TECHNICAL_REPORT section 4.3's recorded `DeltaNetMixer nh=2 (4
    # heads)` parameter count -- an independent cross-check since the
    # delta@1024 arm's config (n_heads=4, nh=2, d_k=d_v=16) mirrors it.
    assert _delta_composer_params(n_heads=4, nh=2, d_k=16, d_v=16) == 25_480
