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
7. Frozen-AST gate narrowing -- ``check_frozen_ast.py``'s per-name check (a)
   still fails on a mutated certified definition or module constant, passes
   on the real tree, and tolerates a mutation to a non-frozen definition
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import mingru
from mingru import min_gru as packaged_min_gru

SEED = 42

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
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
        env["PYTHONPATH"] = os.pathsep.join([str(_SRC_DIR), env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        )
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


# ===========================================================================
# 7. Frozen-AST gate narrowing
# ===========================================================================

# Files the gate script reads from the working tree (see
# ``scripts/check_frozen_ast.py``'s ``PKG_MIN_GRU``/``ROOT_MIN_GRU_DRIVER``/
# ``ROOT_TRITON_SCANS_DRIVER``). Listed explicitly, rather than copying the
# whole repo, so the mutation harness below stays cheap and self-contained.
_GATE_INPUT_FILES = (
    Path("scripts") / "check_frozen_ast.py",
    Path("src") / "mingru" / "min_gru.py",
    Path("min_gru.py"),
    Path("triton_scans.py"),
)


def _copy_gate_inputs(dest: Path) -> None:
    """Copy just the files the frozen-AST gate reads, including uncommitted edits.

    Not a ``git clone``: that would only see committed history, missing
    whatever this task just wrote to ``check_frozen_ast.py`` itself. Not a
    full ``shutil.copytree`` of the repo either: the working tree carries
    multi-GB untracked directories (data, venvs) that have nothing to do
    with the gate. Git history access (for the pinned baseline commit) is
    provided separately via the ``GIT_DIR`` env var in
    ``_run_frozen_ast_gate``, so ``.git`` need not be copied.
    """
    for rel in _GATE_INPUT_FILES:
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO_ROOT / rel, dst)


def _mutate_top_level_node(min_gru_path: Path, name: str) -> None:
    """Make a content-changing edit to a named top-level def/class or module constant.

    The gate compares AST dumps (docstrings stripped), so any
    content-changing edit -- not just a behavior-changing one -- is enough
    to exercise it, whether ``name`` is on the certified surface (should
    flip the gate) or not (should not); the file need not remain importable,
    only parseable. Function/class bodies get a no-op ``pass`` prepended;
    module-level ``Assign``/``AnnAssign`` statements have no body to insert
    into, so their RHS value is replaced with a dummy sentinel instead.
    """
    tree = ast.parse(min_gru_path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            node.body.insert(0, ast.Pass())
            break
        is_assign_target = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        )
        is_ann_assign_target = (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        )
        if is_assign_target or is_ann_assign_target:
            node.value = ast.Constant(value="__mutated_for_test__")
            break
    else:
        raise AssertionError(
            f"{name!r} not found as a top-level definition or module constant in {min_gru_path}"
        )
    ast.fix_missing_locations(tree)
    min_gru_path.write_text(ast.unparse(tree))


def _run_frozen_ast_gate(
    repo_root: Path, *, git_dir: Path | None = None
) -> subprocess.CompletedProcess:
    """Run the gate script rooted at ``repo_root``.

    ``git_dir``, when given, points ``git`` at a different repository's
    object database while the script still reads its input files from
    ``repo_root`` -- lets an isolated copy of just the gate's input files
    (see ``_copy_gate_inputs``) reuse the real repo's commit history
    without copying ``.git``.
    """
    env = dict(os.environ)
    if git_dir is not None:
        env["GIT_DIR"] = str(git_dir)
    return subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "check_frozen_ast.py")],
        capture_output=True,
        text=True,
        env=env,
    )


class TestFrozenAstGateNarrowing:
    """The gate's narrowing to a certified per-name surface still bites.

    Proves four things: it passes on the real tree (the narrowing didn't
    just start always-passing), it fails when a certified definition is
    mutated in an isolated copy (the narrowing isn't a no-op), it fails the
    same way for a certified module constant (the surface isn't just
    defs/classes), and it tolerates a mutation to a non-frozen definition
    (the narrowing's whole point -- e.g. Task 2 adding a mixer seam to
    ``MinGRUBlock``).
    """

    def test_gate_passes_on_the_real_tree(self):
        result = _run_frozen_ast_gate(_REPO_ROOT)
        assert result.returncode == 0, result.stderr

    def test_gate_fails_when_a_certified_definition_is_mutated(self, tmp_path):
        repo_copy = tmp_path / "repo"
        _copy_gate_inputs(repo_copy)
        _mutate_top_level_node(repo_copy / "src" / "mingru" / "min_gru.py", "linear_scan")

        result = _run_frozen_ast_gate(repo_copy, git_dir=_REPO_ROOT / ".git")

        assert result.returncode != 0
        assert "linear_scan" in result.stderr

    def test_gate_fails_when_a_certified_module_constant_is_mutated(self, tmp_path):
        repo_copy = tmp_path / "repo"
        _copy_gate_inputs(repo_copy)
        # _VALID_SCAN_MODES is on _FROZEN_MODULE_CONSTANTS: `_resolve_scan_mode`
        # (a certified dispatch helper) reads it, so it must be frozen too.
        _mutate_top_level_node(repo_copy / "src" / "mingru" / "min_gru.py", "_VALID_SCAN_MODES")

        result = _run_frozen_ast_gate(repo_copy, git_dir=_REPO_ROOT / ".git")

        assert result.returncode != 0
        assert "_VALID_SCAN_MODES" in result.stderr

    def test_gate_tolerates_change_to_non_frozen_definition(self, tmp_path):
        repo_copy = tmp_path / "repo"
        _copy_gate_inputs(repo_copy)
        # MinGRUBlock is not on FROZEN_DEFINITIONS: Task 2 adds a matrix-state
        # seam to its forward path, so the narrowed gate must not flag edits
        # here the way it flags edits to a certified definition.
        _mutate_top_level_node(repo_copy / "src" / "mingru" / "min_gru.py", "MinGRUBlock")

        result = _run_frozen_ast_gate(repo_copy, git_dir=_REPO_ROOT / ".git")

        assert result.returncode == 0, result.stderr
