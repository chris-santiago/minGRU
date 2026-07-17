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
6. Triton ``__all__`` drift guard -- the one place we import
   ``mingru.triton_scans`` directly, to hold ``mingru._TRITON_EXPORTS`` and
   ``triton_scans.__all__`` in sync
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
# The Triton-backed names must stay lazy: none may appear in the eager public
# surface. Read off the package's own registry (rather than hand-copied here)
# so this test and ``mingru._TRITON_EXPORTS`` cannot silently drift apart;
# ``TestTritonAllDriftGuard`` closes the loop back to ``triton_scans.__all__``.
_TRITON_NAMES = mingru._TRITON_EXPORTS


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


# ===========================================================================
# 6. Triton __all__ drift guard
# ===========================================================================


class TestTritonAllDriftGuard:
    """Import ``mingru.triton_scans`` directly and pin it to ``_TRITON_EXPORTS``.

    This is the ONE legitimate place in the suite to import the Triton module
    eagerly: the test exists precisely to guard ``mingru._TRITON_EXPORTS`` (the
    hand-maintained lazy registry that every other name check derives from)
    against drifting from ``triton_scans.__all__`` (the module's own export
    list). Every other test keeps the module lazy (see ``TestLazyImport``).

    Requires torch>=2.8, under which ``mingru.triton_scans`` imports on every
    platform. Whether a local Triton install is present only changes
    ``triton_scans.__all__`` between its three unconditional names and the full
    eleven; both cases must satisfy the invariants below.
    """

    def test_triton_all_is_subset_of_exports(self):
        from mingru import triton_scans

        assert set(triton_scans.__all__) <= set(mingru._TRITON_EXPORTS)

    def test_unconditional_names_always_exported(self):
        from mingru import triton_scans

        assert {"available", "ScanFallback", "SCAN_IMPLS"} <= set(triton_scans.__all__)
