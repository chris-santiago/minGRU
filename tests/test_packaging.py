"""Tests for the packaging surface: ``__all__``, version, lazy Triton import,
and driver re-export identity.

CPU-only. The lazy-import invariant is checked in a fresh subprocess so an
earlier test importing ``mingru.triton_scans`` cannot mask it.

Sections
--------
1. __all__ / __dir__ -- eager API only, no Triton names leak in
2. __version__ -- present and semver-shaped
3. py.typed -- the typing marker ships inside the package
4. Lazy Triton import -- ``import mingru`` leaves the Triton module out of
   sys.modules; touching a Triton name pulls it in
5. Driver re-export identity -- ``min_gru.X is mingru.min_gru.X``
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import mingru
from mingru import min_gru as packaged_min_gru

SEED = 42

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
# The eleven Triton-backed names must stay lazy: none may appear in the
# eager public surface.
_TRITON_NAMES = (
    "available",
    "ScanFallback",
    "SCAN_IMPLS",
    "angle_scan_impl",
    "affine_scan_fwd",
    "linear_scan_fwd",
    "parallel_scan_log_fwd",
    "affine_scan_bwd",
    "linear_scan_bwd",
    "angle_scan_fwd",
    "angle_scan_bwd",
)


# ===========================================================================
# 1. __all__ / __dir__
# ===========================================================================


class TestPublicAll:
    def test_all_is_eager_api_plus_version(self):
        assert set(mingru.__all__) == set(packaged_min_gru.__all__) | {"__version__"}

    def test_no_triton_name_in_all(self):
        for name in _TRITON_NAMES:
            assert name not in mingru.__all__

    def test_no_triton_name_in_dir(self):
        listing = dir(mingru)
        for name in _TRITON_NAMES:
            assert name not in listing

    def test_all_eager_names_resolve(self):
        for name in packaged_min_gru.__all__:
            assert getattr(mingru, name) is getattr(packaged_min_gru, name)

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            _ = mingru.does_not_exist


# ===========================================================================
# 2. __version__
# ===========================================================================


class TestVersion:
    def test_version_present(self):
        assert isinstance(mingru.__version__, str)
        assert mingru.__version__

    def test_version_is_semver_shaped(self):
        assert re.match(r"^\d+\.\d+\.\d+", mingru.__version__)


# ===========================================================================
# 3. py.typed
# ===========================================================================


class TestTypingMarker:
    def test_py_typed_ships_in_package(self):
        marker = Path(mingru.__file__).resolve().parent / "py.typed"
        assert marker.is_file()


# ===========================================================================
# 4. Lazy Triton import
# ===========================================================================


class TestLazyImport:
    def _run(self, snippet: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_SRC_DIR), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        return subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_import_mingru_does_not_import_triton(self):
        snippet = (
            "import sys, mingru\n"
            "assert 'mingru.triton_scans' not in sys.modules, "
            "'triton_scans imported eagerly'\n"
        )
        result = self._run(snippet)
        assert result.returncode == 0, result.stderr

    def test_import_star_leaves_triton_unimported(self):
        snippet = (
            "import sys\n"
            "exec('from mingru import *')\n"
            "assert 'mingru.triton_scans' not in sys.modules\n"
        )
        result = self._run(snippet)
        assert result.returncode == 0, result.stderr

    def test_touching_triton_name_imports_module(self):
        snippet = (
            "import sys, mingru\n"
            "assert 'mingru.triton_scans' not in sys.modules\n"
            "_ = mingru.ScanFallback\n"
            "assert 'mingru.triton_scans' in sys.modules, "
            "'attribute access did not trigger lazy import'\n"
        )
        result = self._run(snippet)
        assert result.returncode == 0, result.stderr


# ===========================================================================
# 5. Driver re-export identity
# ===========================================================================


class TestDriverIdentity:
    def test_root_driver_reexports_by_identity(self):
        # The root evidence driver is importable because pyproject's pytest
        # pythonpath includes the repo root ('.').
        import min_gru as driver

        for name in packaged_min_gru.__all__:
            assert getattr(driver, name) is getattr(packaged_min_gru, name), name

    def test_driver_all_matches_package_min_gru(self):
        import min_gru as driver

        assert list(driver.__all__) == list(packaged_min_gru.__all__)
