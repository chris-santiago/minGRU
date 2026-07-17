"""Tests for the ``MINGRU_SCAN`` dispatch seam (CPU-only).

No Triton kernel ever runs here: on CPU ``available()`` reports "CUDA not
available", so every attempted-Triton path resolves to either the eager
fallback (``auto``) or a ``RuntimeError`` (``triton``). The warn-once
``auto`` fallback branch is only reachable with a CUDA-resident input, so it
is exercised by driving the private ``_dispatch_scan``/``_dispatch_angle_scan``
helpers with a stand-in object whose ``.is_cuda`` is True -- the only
CPU-runnable way to cover that branch, which is exactly the envelope/
availability rejection path.

Sections
--------
1. eager -- forces the eager path, never imports mingru.triton_scans
2. auto on CPU -- stays eager, never imports mingru.triton_scans
3. triton on CPU -- raises RuntimeError naming the unavailability reason
4. invalid MINGRU_SCAN -- raises ValueError
5. auto warn-once fallback -- warns exactly once, then returns to eager
"""

from __future__ import annotations

import sys
import warnings

import pytest
import torch

from mingru import linear_scan
from mingru import min_gru as mg

SEED = 42
_TRITON_MODULE = "mingru.triton_scans"


class _FakeCudaTensor:
    """Minimal stand-in reporting CUDA residency for the dispatch guards.

    The dispatch helpers inspect only ``.is_cuda`` before the CPU
    ``available()`` check declines the Triton path, so no real tensor data
    is ever touched on this branch.
    """

    is_cuda = True


@pytest.fixture(autouse=True)
def _isolate_dispatch(monkeypatch):
    """Keep every test order-independent.

    Drops any cached ``mingru.triton_scans`` import, resets the process-global
    warn-once flags, and clears ``MINGRU_SCAN`` so each test sets its own.
    """
    monkeypatch.delenv("MINGRU_SCAN", raising=False)
    monkeypatch.setattr(mg, "_warned_scan_fallback", False)
    monkeypatch.setattr(mg, "_warned_angle_fallback", False)
    sys.modules.pop(_TRITON_MODULE, None)
    yield
    sys.modules.pop(_TRITON_MODULE, None)


# ===========================================================================
# 1. eager
# ===========================================================================


class TestEagerMode:
    def test_eager_produces_result(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "eager")
        a = torch.randn(2, 5, 3)
        b = torch.randn(2, 5, 3)
        A, Bc = linear_scan(a, b)
        assert A.shape == (2, 5, 3)

    def test_eager_never_imports_triton(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "eager")
        linear_scan(torch.randn(2, 5, 3), torch.randn(2, 5, 3))
        assert _TRITON_MODULE not in sys.modules


# ===========================================================================
# 2. auto on CPU
# ===========================================================================


class TestAutoModeCPU:
    def test_auto_default_stays_eager(self):
        # MINGRU_SCAN unset -> defaults to auto.
        A, Bc = linear_scan(torch.randn(2, 5, 3), torch.randn(2, 5, 3))
        assert A.shape == (2, 5, 3)
        assert _TRITON_MODULE not in sys.modules

    def test_auto_explicit_never_imports_triton(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "auto")
        linear_scan(torch.randn(2, 5, 3), torch.randn(2, 5, 3))
        assert _TRITON_MODULE not in sys.modules


# ===========================================================================
# 3. triton on CPU
# ===========================================================================


class TestTritonModeCPU:
    def test_triton_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "triton")
        with pytest.raises(RuntimeError) as exc:
            linear_scan(torch.randn(2, 5, 3), torch.randn(2, 5, 3))
        assert "triton" in str(exc.value).lower()

    def test_triton_names_the_reason(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "triton")
        with pytest.raises(RuntimeError) as exc:
            linear_scan(torch.randn(2, 5, 3), torch.randn(2, 5, 3))
        assert "unavailable" in str(exc.value).lower()


# ===========================================================================
# 4. invalid MINGRU_SCAN
# ===========================================================================


class TestInvalidMode:
    def test_invalid_value_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "turbo")
        with pytest.raises(ValueError):
            linear_scan(torch.randn(2, 5, 3), torch.randn(2, 5, 3))

    def test_invalid_value_never_imports_triton(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "turbo")
        with pytest.raises(ValueError):
            linear_scan(torch.randn(2, 5, 3), torch.randn(2, 5, 3))
        assert _TRITON_MODULE not in sys.modules


# ===========================================================================
# 5. auto warn-once fallback (envelope / availability rejection)
# ===========================================================================


class TestAutoWarnOnceFallback:
    def test_scan_fallback_warns_once_and_returns_eager_signal(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "auto")
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            first = mg._dispatch_scan("linear_scan", _FakeCudaTensor())
            second = mg._dispatch_scan("linear_scan", _FakeCudaTensor())
        # None means "fall through to the eager implementation".
        assert first is None
        assert second is None
        user_warnings = [w for w in records if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "eager" in str(user_warnings[0].message)

    def test_angle_scan_fallback_warns_once(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "auto")
        theta = _FakeCudaTensor()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            first = mg._dispatch_angle_scan(
                theta, None, None, None, None, None, None, None,
                has_scale=1, has_decay=0,
            )
            second = mg._dispatch_angle_scan(
                theta, None, None, None, None, None, None, None,
                has_scale=1, has_decay=0,
            )
        assert first is None
        assert second is None
        user_warnings = [w for w in records if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "eager" in str(user_warnings[0].message)

    def test_triton_mode_uses_the_same_fallback_seam(self, monkeypatch):
        """MINGRU_SCAN=triton on a CUDA-residency stand-in still raises."""
        monkeypatch.setenv("MINGRU_SCAN", "triton")
        with pytest.raises(RuntimeError):
            mg._dispatch_scan("linear_scan", _FakeCudaTensor())
