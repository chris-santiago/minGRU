"""Regression tests for ``scripts/scaling_probe.py``'s ``_extract_last``.

``scripts/`` is not an importable package (no ``__init__.py`` -- this
repo's ``pyproject.toml`` ``[tool.pytest.ini_options].pythonpath`` only
adds ``src`` and the repo root, for the ``mingru``/``min_gru``/
``triton_scans`` import surfaces), so this module is loaded directly by
file path via ``importlib``.

These tests exercise ``_extract_last`` in complete isolation: crafted
strings only, no subprocess, no torch. They pin the specific regression a
subprocess-timing smoke test cannot reproduce on demand -- a truncated or
malformed ``MINGRU_PROBE_HEADER``/``MINGRU_PROBE_RESULT`` line (most
likely in practice from a ``SIGKILL`` landing mid-``print()`` when the
parent kills a hung child on timeout, e.g. the givens@4096 wall case)
must degrade to "no match", never raise ``json.JSONDecodeError`` -- an
uncaught raise there would propagate out of ``_run_one_config`` into
``run()``'s grid loop and abort the whole probe before the artifact is
written, discarding every already-collected row over one truncated line.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "scaling_probe.py"
_spec = importlib.util.spec_from_file_location("scaling_probe", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
scaling_probe = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("scaling_probe", scaling_probe)
_spec.loader.exec_module(scaling_probe)

_extract_last = scaling_probe._extract_last
_HEADER_PREFIX = scaling_probe._HEADER_PREFIX


def test_no_matching_line_returns_none():
    assert _extract_last(_HEADER_PREFIX, "some unrelated output\nmore lines\n") is None


def test_empty_text_returns_none():
    assert _extract_last(_HEADER_PREFIX, "") is None


def test_well_formed_line_parses():
    text = f'{_HEADER_PREFIX}{{"label": "x", "state_elements": 64}}\n'
    assert _extract_last(_HEADER_PREFIX, text) == {"label": "x", "state_elements": 64}


def test_truncated_line_degrades_to_none_not_raise():
    # Simulates a SIGKILL landing mid-print(): the JSON payload is cut off
    # partway through. Must not raise json.JSONDecodeError.
    text = f'{_HEADER_PREFIX}{{"label": "givens_dh4096", "state_ele'
    assert _extract_last(_HEADER_PREFIX, text) is None


def test_malformed_line_does_not_clobber_earlier_valid_line():
    # An earlier matching line parses fine; a later matching line (same
    # prefix) is malformed. The malformed line is skipped rather than
    # overwriting the result with a parse failure -- the last
    # successfully-parsed match wins, and the function never raises.
    text = f'{_HEADER_PREFIX}{{"label": "a"}}\n{_HEADER_PREFIX}{{"label": "b", "truncated'
    assert _extract_last(_HEADER_PREFIX, text) == {"label": "a"}


def test_non_matching_lines_interleaved_with_malformed_and_valid():
    text = (
        "child stdout noise\n"
        f"{_HEADER_PREFIX}not even json at all\n"
        "more noise\n"
        f'{_HEADER_PREFIX}{{"label": "final", "ok": true}}\n'
    )
    assert _extract_last(_HEADER_PREFIX, text) == {"label": "final", "ok": True}


def test_prefix_must_match_exactly_not_substring():
    # A line that merely contains the prefix midway through must not match
    # (startswith, not "in") -- guards against a false match on stray
    # stdout that happens to embed the marker text.
    text = f'noise before {_HEADER_PREFIX}{{"label": "x"}}\n'
    assert _extract_last(_HEADER_PREFIX, text) is None
