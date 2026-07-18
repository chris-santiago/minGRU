"""Regression tests for ``scripts/gpu_check.py``'s ``_extract_last`` (Task 6).

``scripts/`` is not an importable package (no ``__init__.py`` -- see
``tests/test_scaling_probe.py``'s identical note), so this module is
loaded directly by file path via ``importlib``.

These tests exercise ``_extract_last`` in complete isolation: crafted
strings only, no subprocess, no ``lightning_sdk``, no torch. They pin the
specific regression a live-job smoke test cannot reproduce on demand: a
truncated or malformed ``MINGRU_GPU_PROBE_RESULT`` line in a job's fetched
logs (interleaved with keepalive heartbeats and clone/checkout noise) must
degrade to "no match", never raise ``json.JSONDecodeError`` -- an uncaught
raise here would propagate out of ``_finish_delta_probe`` and crash the
CLI instead of reporting the clear "no result line found" error the task
brief requires (and, worse, could leave a half-written artifact on disk if
it happened mid-write instead of before any write is attempted).

Mirrors ``tests/test_scaling_probe.py``'s test structure for
``scaling_probe.py``'s own ``_extract_last`` (same line-marker
extraction/malformed-line-tolerance pattern, duplicated per
``scripts/gpu_check.py``'s ``_extract_last`` docstring's
``DUPLICATION-PENDING`` note) -- kept as a parallel, independent test file
rather than merged, since the two scripts are independent job-runner entry
points.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "gpu_check.py"
_spec = importlib.util.spec_from_file_location("gpu_check", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
gpu_check = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("gpu_check", gpu_check)
_spec.loader.exec_module(gpu_check)

_extract_last = gpu_check._extract_last
_RESULT_PREFIX = gpu_check._DELTA_PROBE_RESULT_PREFIX


def test_no_matching_line_returns_none():
    text = "[keepalive] Fri Jul 18 00:00:00 UTC 2026\nCloning into '/tmp/minGRU'...\n"
    assert _extract_last(_RESULT_PREFIX, text) is None


def test_empty_text_returns_none():
    assert _extract_last(_RESULT_PREFIX, "") is None


def test_well_formed_line_parses():
    text = f'{_RESULT_PREFIX}{{"env": {{"torch_version": "2.8.0"}}, "shapes": []}}\n'
    assert _extract_last(_RESULT_PREFIX, text) == {
        "env": {"torch_version": "2.8.0"},
        "shapes": [],
    }


def test_well_formed_line_amid_keepalive_and_progress_noise():
    text = (
        "gpu_delta_probe: env {...}\n"
        "  running pd1024_T64 (B=128, T=64)...\n"
        "[keepalive] Fri Jul 18 00:05:00 UTC 2026\n"
        "    eager=0.0123s floor=0.0050s compile=0.0060s (ok) headroom=2.46\n"
        f'{_RESULT_PREFIX}{{"env": {{}}, "shapes": [{{"label": "pd1024_T64"}}]}}\n'
    )
    assert _extract_last(_RESULT_PREFIX, text) == {
        "env": {},
        "shapes": [{"label": "pd1024_T64"}],
    }


def test_truncated_line_degrades_to_none_not_raise():
    # Simulates a job killed (idle timeout / OOM / preemption) mid-print():
    # the JSON payload is cut off partway through. Must not raise
    # json.JSONDecodeError.
    text = f'{_RESULT_PREFIX}{{"env": {{"torch_version": "2.8.0"}}, "shap'
    assert _extract_last(_RESULT_PREFIX, text) is None


def test_malformed_line_does_not_clobber_earlier_valid_line():
    # An earlier matching line parses fine; a later matching line (same
    # prefix) is malformed. The malformed line is skipped rather than
    # overwriting the result with a parse failure -- the last
    # successfully-parsed match wins, and the function never raises.
    text = (
        f'{_RESULT_PREFIX}{{"env": {{}}, "shapes": [1]}}\n'
        f'{_RESULT_PREFIX}{{"env": {{}}, "shapes": [2], "truncated'
    )
    assert _extract_last(_RESULT_PREFIX, text) == {"env": {}, "shapes": [1]}


def test_prefix_must_match_exactly_not_substring():
    # A line that merely contains the prefix midway through must not match
    # (startswith, not "in") -- guards against a false match on stray log
    # noise that happens to embed the marker text.
    text = f'noise before {_RESULT_PREFIX}{{"env": {{}}, "shapes": []}}\n'
    assert _extract_last(_RESULT_PREFIX, text) is None
