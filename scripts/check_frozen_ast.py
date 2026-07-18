#!/usr/bin/env python3
"""Frozen-AST gate for the Phase-4 packaging move of ``min_gru.py``/``triton_scans.py``.

The Phase-4 packaging invariant (design spec section 7) is that the eager
library moved verbatim into ``src/mingru/`` and each module's ``__main__``
selftest block relocated verbatim into its repo-root evidence driver.
"Verbatim" is enforced at the level of *executable content*: the abstract
syntax tree with docstrings stripped (comments are already absent from the
AST). Three checks, each a hard stop (nonzero exit) on violation:

(a) **Library freeze (narrowed to the certified evidence surface).**
    ``src/mingru/min_gru.py`` is no longer frozen whole-file -- the delta-mixer
    design (spec section 6, "Frozen-AST gate contract") unfreezes the file for
    growth, since the whole-file check was always a stronger-than-necessary
    proxy for what the replication evidence actually certifies. Instead, each
    definition named in ``FROZEN_DEFINITIONS`` (the four scan ops, the four
    pre-existing mixer classes, ``DecayMixin``, the Triton-dispatch helpers,
    and the five free functions the certified mixer forward paths call at
    runtime -- ``g``, ``log_g``, ``_normalize_delta_t``,
    ``_validate_delta_t_pairing``, ``_warn_once_invalid_delta_t``) and each
    module-level assignment named in ``_FROZEN_MODULE_CONSTANTS`` (constants
    and shared warn-once flags the frozen definitions read --
    ``_DELTA_T_POSINF_CAP``, ``_VALID_SCAN_MODES``, ``_warned_scan_fallback``,
    ``_warned_angle_fallback``, ``Decay``) must be executable-identical, *by
    name*, to its counterpart in `the frozen baseline
    (`FROZEN_BASELINE:min_gru.py`)`, the single permitted delta being the
    dispatch-import retarget (``import triton_scans`` -> ``from mingru import
    triton_scans``). Definitions and module constants outside that surface --
    including new top-level definitions -- may change or be added freely;
    unfreezing the file this way does not invalidate the evidence the frozen
    surface certifies.

(b) **min_gru selftest relocation.** The ``__main__`` block of the root
    ``min_gru.py`` driver must be executable-identical to the ``__main__``
    block of `the frozen baseline (`FROZEN_BASELINE:min_gru.py`)` modulo the two permitted
    driver-contract adaptations (spec section 6): a leading import header, and
    the module-name string ``"triton_scans"`` -> ``"mingru.triton_scans"`` in
    the ``sys.modules`` pops/asserts and their messages.

(c) **triton_scans selftest relocation.** The ``__main__`` block of the root
    ``triton_scans.py`` driver must be executable-identical to the
    ``__main__`` block of the pinned ``FROZEN_BASELINE`` revision of
    ``triton_scans.py`` modulo the same
    driver-contract adaptations; this block has no ``sys.modules`` "triton_scans"
    string to substitute, so only the leading import header may differ.

Static only: parses ASTs, never imports torch or the package, so it runs on a
bare interpreter with no dependencies beyond git.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

# The evidence-frozen baseline: the last pre-packaging commit, where
# min_gru.py / triton_scans.py were the monolithic root modules whose eager
# behavior the replication evidence certifies. Pinned to an immutable SHA
# (NOT a moving ref like `main`): after the packaging branch merges, `main`'s
# root files are the evidence drivers, and comparing against them would be
# self-referential.
FROZEN_BASELINE = "d67002ad588f2fb8e19cadfa62a8d9af50e6e927"


REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_MIN_GRU = REPO_ROOT / "src" / "mingru" / "min_gru.py"
ROOT_MIN_GRU_DRIVER = REPO_ROOT / "min_gru.py"
ROOT_TRITON_SCANS_DRIVER = REPO_ROOT / "triton_scans.py"

# The dispatch-import retarget, expressed as a canonicalization: rewriting the
# packaged form back to the pre-move form makes the two trees comparable.
RETARGET_FROM = ("mingru", "triton_scans")  # `from mingru import triton_scans`
RETARGET_TO = "triton_scans"  # `import triton_scans`
# The relocated-selftest module-name adaptation, reversed for comparison.
RELOCATED_MODULE_NAME = "mingru.triton_scans"
ORIGINAL_MODULE_NAME = "triton_scans"

# The certified evidence surface (design spec section 6, "Frozen-AST gate
# contract"): only these top-level definitions in src/mingru/min_gru.py are
# held AST-identical to the pinned baseline. Everything else in the file --
# MinGRUBlock, MinGRUStack, and any new top-level definitions -- is free to
# change; the replication evidence was only ever certified for this surface,
# so narrowing the check to it doesn't invalidate what came before.
_FROZEN_SCAN_OPS = ("parallel_scan_log", "linear_scan", "matrix_scan", "matrix_affine_scan")
_FROZEN_MIXER_CLASSES = ("MinGRU", "SignedMinGRU", "RotationMinGRU", "GivensMinGRU")
_FROZEN_DECAY_MIXIN = ("DecayMixin",)
# The Triton-dispatch internals (module docstring's "dispatch internals"
# grouping): resolving which backend to use and dispatching to it.
_FROZEN_DISPATCH_HELPERS = (
    "_resolve_scan_mode",
    "_resolve_triton_dispatch",
    "_dispatch_scan",
    "_angle_scan_should_try",
    "_dispatch_angle_scan",
    "_cached_plane_meta_per_device",
)
# Free functions the certified mixer forward paths call at runtime: leaving
# these out would let their bodies change silently under an AST-identical
# mixer class, which is exactly the gap the whole-file freeze existed to
# close. Part of the certified surface even though they aren't scan ops,
# mixer classes, DecayMixin, or dispatch internals themselves.
_FROZEN_EVIDENCE_HELPERS = (
    "g",
    "log_g",
    "_normalize_delta_t",
    "_validate_delta_t_pairing",
    "_warn_once_invalid_delta_t",
)
FROZEN_DEFINITIONS = (
    _FROZEN_SCAN_OPS
    + _FROZEN_MIXER_CLASSES
    + _FROZEN_DECAY_MIXIN
    + _FROZEN_DISPATCH_HELPERS
    + _FROZEN_EVIDENCE_HELPERS
)

# Module-level assignments the frozen definitions above read (constants and
# the shared warn-once flags): identical gap to the evidence helpers above --
# an AST-identical frozen function/class can still change behavior if a
# constant it references is free to drift. Compared as whole assignment
# statements (Assign/AnnAssign), not just their values.
_FROZEN_MODULE_CONSTANTS = (
    "_DELTA_T_POSINF_CAP",
    "_VALID_SCAN_MODES",
    "_warned_scan_fallback",
    "_warned_angle_fallback",
    "Decay",
)


def _git_show(ref_path: str) -> str:
    return subprocess.run(
        ["git", "show", ref_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _strip_docstrings(node: ast.AST) -> ast.AST:
    """Remove module/class/function leading-string-literal docstrings, in place."""
    for child in ast.walk(node):
        if isinstance(child, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(child, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                child.body = body[1:]
    return node


def _find_main_block(tree: ast.Module) -> ast.If:
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            return node
    raise SystemExit('FAIL: no `if __name__ == "__main__"` block found')


def _is_main_block(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def _canonicalize_retarget(tree: ast.AST) -> None:
    """Rewrite `from mingru import triton_scans` back to `import triton_scans`."""
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for i, node in enumerate(body):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == RETARGET_FROM[0]
                and node.level == 0
                and len(node.names) == 1
                and node.names[0].name == RETARGET_FROM[1]
                and node.names[0].asname is None
            ):
                body[i] = ast.Import(names=[ast.alias(name=RETARGET_TO, asname=None)])


def _reverse_module_name_strings(node: ast.AST) -> None:
    """Rewrite "mingru.triton_scans" -> "triton_scans" in every str constant."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            child.value = child.value.replace(RELOCATED_MODULE_NAME, ORIGINAL_MODULE_NAME)


def _strip_header(stmts: list[ast.stmt], count: int) -> list[ast.stmt]:
    """Drop the driver's leading import-header statements.

    Takes an explicit count rather than stripping every leading
    ``Import``/``ImportFrom`` generically: the *original* (pre-move)
    ``__main__`` block itself may open with its own import statement (e.g.
    triton_scans's relocated block starts with ``import min_gru as
    _min_gru_lockstep``), so a generic prefix-strip would eat into the
    verbatim body instead of stopping at the header boundary.
    """
    header, body = stmts[:count], stmts[count:]
    if not all(isinstance(s, (ast.Import, ast.ImportFrom)) for s in header):
        raise SystemExit(
            f"FAIL: expected the driver's first {count} statements to be its "
            "import header, but found a non-import statement -- header count "
            "is out of sync with the driver source"
        )
    return body


def _top_level_definitions(tree: ast.Module) -> dict[str, ast.stmt]:
    """Map top-level function/class names to their nodes."""
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _top_level_module_constants(tree: ast.Module) -> dict[str, ast.stmt]:
    """Map module-level single-target assignment names to their statement nodes.

    Covers ``Assign`` (``NAME = value``) and ``AnnAssign`` (``NAME: T =
    value``); only single-target assigns are indexed, since none of the
    frozen constants use tuple/starred targets.
    """
    constants: dict[str, ast.stmt] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            constants[node.targets[0].id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            constants[node.target.id] = node
    return constants


def _dump(stmts: list[ast.stmt]) -> str:
    module = ast.Module(body=stmts, type_ignores=[])
    _strip_docstrings(module)
    return ast.dump(module, annotate_fields=True, include_attributes=False, indent=2)


def _first_divergence(a: str, b: str) -> str:
    la, lb = a.splitlines(), b.splitlines()
    for idx, (x, y) in enumerate(zip(la, lb, strict=False)):
        if x != y:
            return (
                f"  first divergence at dump line {idx + 1}:\n"
                f"    main:   {x.strip()}\n    local:  {y.strip()}"
            )
    if len(la) != len(lb):
        return f"  dumps differ in length: main={len(la)} lines, local={len(lb)} lines"
    return "  (dumps differ but no line-level divergence located)"


def _check_frozen_names(
    names: tuple[str, ...],
    main_index: dict[str, ast.stmt],
    pkg_index: dict[str, ast.stmt],
    *,
    kind: str,
) -> list[str]:
    """Compare each certified name's AST dump between baseline and packaged indices.

    Shared by both halves of check (a): certified definitions
    (``FROZEN_DEFINITIONS`` against ``_top_level_definitions``) and certified
    module constants (``_FROZEN_MODULE_CONSTANTS`` against
    ``_top_level_module_constants``) -- same missing/diverging-name error
    shape either way, just a different ``kind`` label and node index.
    """
    errors: list[str] = []
    for name in names:
        if name not in main_index:
            errors.append(
                f"library freeze: internal error -- certified {kind} {name!r} "
                "not found in the frozen baseline itself; the frozen-surface "
                "list is out of sync with `FROZEN_BASELINE:min_gru.py`"
            )
            continue
        if name not in pkg_index:
            errors.append(
                f"library freeze: certified {kind} {name!r} is missing from src/mingru/min_gru.py"
            )
            continue
        main_dump = _dump([main_index[name]])
        pkg_dump = _dump([pkg_index[name]])
        if main_dump != pkg_dump:
            errors.append(
                f"library freeze: {kind} {name!r} in src/mingru/min_gru.py "
                "diverges from the frozen baseline (`FROZEN_BASELINE:min_gru.py`) "
                "beyond the permitted dispatch retarget.\n" + _first_divergence(main_dump, pkg_dump)
            )
    return errors


def check_library_freeze() -> list[str]:
    """(a) each certified definition/constant == its baseline counterpart, modulo the retarget.

    Narrowed from whole-file equality (spec section 6): only the definitions
    named in ``FROZEN_DEFINITIONS`` and the module-level assignments named in
    ``_FROZEN_MODULE_CONSTANTS`` are compared. New top-level definitions
    (e.g. a new mixer class) and edits to non-surface definitions/constants
    (e.g. ``MinGRUBlock``) are permitted and simply not inspected here.
    """
    errors: list[str] = []
    main_tree = ast.parse(_git_show(f"{FROZEN_BASELINE}:min_gru.py"))
    pkg_tree = ast.parse(PKG_MIN_GRU.read_text())

    if any(_is_main_block(n) for n in pkg_tree.body):
        errors.append(
            "library freeze: src/mingru/min_gru.py still contains a __main__ "
            "block (it must relocate to the root driver)"
        )

    _canonicalize_retarget(pkg_tree)

    errors += _check_frozen_names(
        FROZEN_DEFINITIONS,
        _top_level_definitions(main_tree),
        _top_level_definitions(pkg_tree),
        kind="definition",
    )
    errors += _check_frozen_names(
        _FROZEN_MODULE_CONSTANTS,
        _top_level_module_constants(main_tree),
        _top_level_module_constants(pkg_tree),
        kind="module constant",
    )
    return errors


def check_selftest_relocation(
    *, main_ref: str, driver_path: Path, driver_name: str, header_import_count: int
) -> list[str]:
    """One driver's __main__ == its pre-move main __main__, modulo the §6 adaptations.

    Parameters
    ----------
    main_ref : str
        The ``git show`` path for the pre-move module, e.g. ``f"{FROZEN_BASELINE}:min_gru.py"``.
    driver_path : Path
        The repo-root evidence driver to check, e.g. ``ROOT_MIN_GRU_DRIVER``.
    driver_name : str
        Human-readable label for error messages, e.g. ``"min_gru.py"``.
    header_import_count : int
        Number of top-level import statements in the driver's ``__main__``
        import header, to strip before comparison (see ``_strip_header``).
    """
    errors: list[str] = []
    main_tree = ast.parse(_git_show(main_ref))
    driver_tree = ast.parse(driver_path.read_text())

    main_block = _find_main_block(main_tree)
    driver_block = _find_main_block(driver_tree)

    # Adaptation (a): strip the leading import header the driver prepends.
    driver_body = _strip_header(list(driver_block.body), header_import_count)
    # Adaptation (b): reverse the module-name string substitution (a no-op if
    # the block has no "mingru.triton_scans" string, e.g. the triton_scans
    # driver's own relocated block).
    driver_module = ast.Module(body=driver_body, type_ignores=[])
    _reverse_module_name_strings(driver_module)

    main_dump = _dump(list(main_block.body))
    driver_dump = _dump(driver_module.body)
    if main_dump != driver_dump:
        errors.append(
            f"selftest relocation: root {driver_name} __main__ block diverges "
            f"from `git show {main_ref}` __main__ beyond the permitted import "
            "header + module-name-string adaptations.\n" + _first_divergence(main_dump, driver_dump)
        )
    return errors


def main() -> int:
    errors: list[str] = []
    errors += check_library_freeze()
    errors += check_selftest_relocation(
        main_ref=f"{FROZEN_BASELINE}:min_gru.py",
        driver_path=ROOT_MIN_GRU_DRIVER,
        driver_name="min_gru.py",
        # `import os` / `import torch` / `import torch.nn as nn` /
        # `from mingru.min_gru import (...)`.
        header_import_count=4,
    )
    errors += check_selftest_relocation(
        main_ref=f"{FROZEN_BASELINE}:triton_scans.py",
        driver_path=ROOT_TRITON_SCANS_DRIVER,
        driver_name="triton_scans.py",
        # `import torch` / `import torch.nn.functional as F` /
        # `from mingru.triton_scans import (...)`. NOT 4: the original
        # block's own first statement is itself an import
        # (`import min_gru as _min_gru_lockstep`), which must stay in the
        # verbatim body, not be swallowed as header.
        header_import_count=3,
    )

    if errors:
        print("FROZEN-AST CHECK FAILED:", file=sys.stderr)
        for err in errors:
            print(f"\n- {err}", file=sys.stderr)
        return 1

    print("frozen-AST check OK:")
    print(
        f"  (a) {len(FROZEN_DEFINITIONS)} certified definitions + "
        f"{len(_FROZEN_MODULE_CONSTANTS)} certified module constants in "
        "src/mingru/min_gru.py frozen"
    )
    print("      vs the pinned baseline (only the dispatch retarget differs); other")
    print("      definitions/constants, including new ones, are unrestricted")
    print("  (b) root min_gru.py __main__ selftest relocated verbatim vs the pinned baseline")
    print("      (only the import header + module-name strings differ)")
    print("  (c) root triton_scans.py __main__ selftest relocated verbatim vs the pinned baseline")
    print("      (only the import header differs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
