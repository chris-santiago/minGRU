#!/usr/bin/env bash
# Release-readiness wheel proof for mingru-scans.
#
# Builds the wheel + sdist, installs the wheel into a THROWAWAY venv (torch>=2.8,
# CPU), runs an import/shape smoke, then the full pytest suite against the
# INSTALLED package -- never the src/ checkout. Exits nonzero on any failure.
#
# Installed-vs-checkout mechanism (the subtle part):
#   pyproject sets `pythonpath = ["src", "."]` so the dev suite resolves `mingru`
#   from src/ and the root evidence drivers from the repo root. That is exactly
#   wrong for a wheel proof: src/ would shadow the installed package and hide
#   packaging bugs (missing files, bad metadata, broken lazy imports). This
#   script runs the venv's own pytest with `-o pythonpath=` (override to empty),
#   so nothing is injected and `mingru` resolves from the venv site-packages.
#   The smoke step asserts `mingru.__file__` lives under site-packages to prove
#   it.
#
# Deselected against the installed package:
#   tests/test_packaging.py::TestDriverIdentity -- `min_gru.X is mingru.min_gru.X`
#   is a repo-CHECKOUT invariant. The root evidence driver is never shipped in
#   the wheel (src-layout excludes it), so the identity is meaningless here; and
#   importing the driver would insert src/ ahead of site-packages, defeating the
#   whole point of testing the wheel. Its checkout-world coverage runs in the
#   normal `pytest` invocation from the repo (pythonpath includes ".").
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Throwaway venv under a temp dir; always cleaned up, even on failure.
TMP_ROOT="$(mktemp -d)"
VENV_DIR="$TMP_ROOT/venv"
cleanup() { rm -rf "$TMP_ROOT"; }
trap cleanup EXIT

echo "==> Building wheel + sdist (uv build)"
rm -rf dist
uv build

WHEEL="$(ls dist/*.whl)"
SDIST="$(ls dist/*.tar.gz)"
test -f "$WHEEL" || { echo "no wheel built" >&2; exit 1; }
test -f "$SDIST" || { echo "no sdist built" >&2; exit 1; }
echo "    wheel: $WHEEL"
echo "    sdist: $SDIST"

echo "==> Creating fresh venv (python 3.12)"
uv venv --python 3.12 "$VENV_DIR"
VENV_PY="$VENV_DIR/bin/python"

# CPU torch: native on macOS PyPI wheels; the CPU index keeps Linux CI off the
# CUDA build. `--torch-backend cpu` (uv >= 0.6) resolves the CPU variant on both.
echo "==> Installing wheel + pytest into the fresh venv (CPU torch)"
uv pip install --python "$VENV_PY" --torch-backend cpu "$WHEEL" pytest

EXPECTED_VERSION="$(grep -E '^__version__' src/mingru/__init__.py | sed -E 's/.*"([^"]+)".*/\1/')"
echo "==> Import + shape smoke (installed package, src/ NOT on path); expecting version ${EXPECTED_VERSION}"
EXPECTED_VERSION="$EXPECTED_VERSION" "$VENV_PY" - <<'PY'
import os
import mingru
import torch

# Prove `mingru` came from the installed wheel, not a src/ checkout.
assert "site-packages" in mingru.__file__, mingru.__file__
# Prove the installed wheel carries the current source version (guards against a
# stale/mismatched wheel). Derived from src/mingru/__init__.py, so it tracks the
# version bump automatically instead of hardcoding a value that goes stale each release.
_expected = os.environ["EXPECTED_VERSION"]
assert mingru.__version__ == _expected, f"{mingru.__version__} != {_expected}"

from mingru import MinGRU

model = MinGRU(input_size=16, hidden_size=16)
x = torch.randn(2, 5, 16)
y = model(x)
assert y.shape == (2, 5, 16), y.shape
print("smoke OK:", mingru.__file__, mingru.__version__)
PY

echo "==> Full pytest suite against the INSTALLED package"
"$VENV_PY" -m pytest tests/ -q \
  -o pythonpath= \
  --deselect tests/test_packaging.py::TestDriverIdentity

echo "==> check_wheel: PASS"
