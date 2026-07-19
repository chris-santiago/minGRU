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
6. DeltaMinGRU dispatch seam -- same auto/eager/triton contract, mirrored
   for ``_dispatch_delta_scan``/``_delta_scan_should_try``, plus the ``auto``
   shape-gate region (``T >= 16 * chunk_size`` AND ``d_k <= 16``): in-region
   auto engages, out-of-region auto stays eager silently (no warning),
   triton-mode ignores the region
7. DeltaMinGRU envelope violations -- each ``_delta_envelope_reason``
   rejection carries a distinct reason string (CPU-testable directly, since
   it takes plain descriptors rather than tensors)
8. DeltaMinGRU seam layout round trip -- value-level pin for the spec
   section 6 permute seam: a fake ``triton_scans.delta_scan_impl`` (driven
   on CPU via ``MINGRU_SCAN=triton``, which skips the CPU-auto short
   circuit) records exactly the ``Q``/``K``/``V``/``beta``/``H0`` tensors
   ``_dispatch_delta_scan`` assembles, and per-coordinate-unique marker
   values pin both the inbound permute and the outbound restore +
   reshape + ``out_proj``
"""

from __future__ import annotations

import sys
import warnings

import pytest
import torch

import mingru
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
    monkeypatch.setattr(mg, "_warned_delta_fallback", False)
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
                theta,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                has_scale=1,
                has_decay=0,
            )
            second = mg._dispatch_angle_scan(
                theta,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                has_scale=1,
                has_decay=0,
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


# ===========================================================================
# 6. DeltaMinGRU dispatch seam
# ===========================================================================
#
# Mirrors sections 1-5, but for the module-forward fast path
# (`_dispatch_delta_scan`/`_delta_scan_should_try`) instead of the four scan
# ops. `_dispatch_delta_scan`'s `_call` closure checks `triton_scans.available()`
# before ever touching K/V/beta/H0 (see its docstring), so a `_FakeCudaTensor`
# `q` with `None` for the other four positional args is enough to drive the
# CPU-declined branch, exactly like `TestAutoWarnOnceFallback`'s angle-scan
# case above.


class TestDeltaEagerMode:
    def test_eager_never_tries_the_kernel(self, monkeypatch):
        """The pre-gate declines even for an in-region CUDA input under eager."""
        monkeypatch.setenv("MINGRU_SCAN", "eager")
        assert mg._delta_scan_should_try(True, T=1024, chunk_size=16, d_k=16) is False

    def test_eager_forward_never_imports_triton(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "eager")
        torch.manual_seed(SEED)
        layer = mg.DeltaMinGRU(input_size=6, hidden_size=8, n_heads=2, nh=1)
        x = torch.randn(2, 5, 6)
        y = layer(x)
        assert y.shape == (2, 5, 8)
        assert _TRITON_MODULE not in sys.modules


class TestDeltaAutoModeCPU:
    def test_auto_default_stays_eager(self):
        # MINGRU_SCAN unset -> defaults to auto; CPU input -> stays eager.
        assert mg._delta_scan_should_try(False, T=1024, chunk_size=16, d_k=16) is False
        torch.manual_seed(SEED)
        layer = mg.DeltaMinGRU(input_size=6, hidden_size=8, n_heads=2, nh=1)
        x = torch.randn(2, 5, 6)
        y = layer(x)
        assert y.shape == (2, 5, 8)
        assert _TRITON_MODULE not in sys.modules


class TestDeltaAutoRegionGate:
    """The ``auto`` shape-gate: engage only in the measured win region.

    ``auto`` engages the DeltaMinGRU kernel only when ``T >= 16 * chunk_size``
    AND ``d_k <= 16`` (the L4 win region; see ``_delta_scan_should_try`` and
    experiments/bench/gpu_delta_probe.md). Out-of-region ``auto`` stays eager
    SILENTLY -- that is the designed default, not a degradation, so no fallback
    warning fires. Explicit ``triton`` skips the region gate entirely. The
    gate is unit-testable on CPU because ``_delta_scan_should_try`` takes
    ``is_cuda`` as a plain bool -- passing ``True`` simulates a CUDA input
    without a device (the same stand-in trick the warn-once tests use for
    ``_FakeCudaTensor``).
    """

    def test_in_region_auto_engages(self, monkeypatch):
        # Boundary of both conditions: T == 16 * chunk_size, d_k == 16.
        monkeypatch.setenv("MINGRU_SCAN", "auto")
        assert mg._delta_scan_should_try(True, T=16 * 16, chunk_size=16, d_k=16) is True

    def test_out_of_region_short_sequence_stays_eager_silently(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "auto")
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            # T one below the 16-chunk floor -> out of region.
            engaged = mg._delta_scan_should_try(True, T=16 * 16 - 1, chunk_size=16, d_k=16)
        assert engaged is False
        assert records == []  # designed default -> no warning

    def test_out_of_region_wide_dk_stays_eager_silently(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "auto")
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            # Long sequence but d_k above the region cap -> out of region.
            engaged = mg._delta_scan_should_try(True, T=1024, chunk_size=16, d_k=32)
        assert engaged is False
        assert records == []

    def test_triton_mode_ignores_region_full_envelope(self, monkeypatch):
        """``triton`` drives the full envelope: the region gate is skipped."""
        monkeypatch.setenv("MINGRU_SCAN", "triton")
        # An out-of-region shape (short T, wide d_k) still attempts under triton,
        # regardless of CUDA residency (triton fails loud downstream, never gates).
        assert mg._delta_scan_should_try(True, T=64, chunk_size=64, d_k=64) is True
        assert mg._delta_scan_should_try(False, T=64, chunk_size=64, d_k=64) is True


class TestDeltaTritonModeCPU:
    def test_triton_raises_runtime_error_naming_delta_path(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "triton")
        q = _FakeCudaTensor()
        with pytest.raises(RuntimeError) as exc:
            mg._dispatch_delta_scan(q, None, None, None, None, chunk_size=64)
        message = str(exc.value).lower()
        assert "delta" in message
        assert "unavailable" in message


class TestDeltaAutoWarnOnceFallback:
    def test_delta_scan_fallback_warns_once_and_returns_eager_signal(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "auto")
        q = _FakeCudaTensor()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            first = mg._dispatch_delta_scan(q, None, None, None, None, chunk_size=64)
            second = mg._dispatch_delta_scan(q, None, None, None, None, chunk_size=64)
        # None means "fall through to the eager _forward_chunked".
        assert first is None
        assert second is None
        user_warnings = [w for w in records if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "eager" in str(user_warnings[0].message)
        assert "delta" in str(user_warnings[0].message).lower()

    def test_delta_and_angle_fallbacks_warn_independently(self, monkeypatch):
        """Each fused path owns its own warn-once flag; one firing doesn't mask the other."""
        monkeypatch.setenv("MINGRU_SCAN", "auto")
        q = _FakeCudaTensor()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            mg._dispatch_delta_scan(q, None, None, None, None, chunk_size=64)
            mg._dispatch_angle_scan(
                q, None, None, None, None, None, None, None, has_scale=1, has_decay=0
            )
        user_warnings = [w for w in records if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 2


# ===========================================================================
# 7. DeltaMinGRU envelope violations
# ===========================================================================
#
# `_delta_envelope_reason` (mingru.triton_scans) takes plain descriptors --
# dtypes, ints, a bool -- rather than tensors, so every branch is reachable
# from a CPU-only test: passing `is_cuda=True` with in-envelope values for
# every earlier check is what lets a later check (e.g. `d_k` membership) be
# exercised, since a real CPU tensor would trip the CUDA check first (see the
# module's own docstring). Imported locally per test (not at module scope),
# matching ``TestTritonAllDriftGuard`` in ``tests/test_packaging.py``: the
# only place in the suite that imports ``mingru.triton_scans`` directly,
# which requires torch>=2.8 and is one of the suite's recorded pin-specific
# gaps rather than something every test file should risk at collection time.


class TestDeltaEnvelopeReasons:
    _BASE = dict(
        is_cuda=True,
        q_dtype=torch.float32,
        k_dtype=torch.float32,
        v_dtype=torch.float32,
        beta_dtype=torch.float32,
        h0_dtype=torch.float32,
        d_k=16,
        d_v=16,
        chunk_size=64,
        nh=1,
    )

    @classmethod
    def _reason(cls, **overrides):
        from mingru import triton_scans

        kwargs = dict(cls._BASE)
        kwargs.update(overrides)
        return triton_scans._delta_envelope_reason(**kwargs)

    def test_in_envelope_is_none(self):
        assert self._reason() is None

    def test_non_cuda(self):
        reason = self._reason(is_cuda=False)
        assert reason is not None
        assert "CUDA" in reason

    def test_q_dtype_not_fp32(self):
        reason = self._reason(q_dtype=torch.float16)
        assert "Q dtype" in reason

    def test_k_dtype_not_fp32(self):
        reason = self._reason(k_dtype=torch.float16)
        assert "K dtype" in reason

    def test_v_dtype_not_fp32(self):
        reason = self._reason(v_dtype=torch.float16)
        assert "V dtype" in reason

    def test_beta_dtype_not_fp32(self):
        reason = self._reason(beta_dtype=torch.float16)
        assert "beta dtype" in reason

    def test_h0_dtype_not_fp32(self):
        reason = self._reason(h0_dtype=torch.float16)
        assert "H0 dtype" in reason

    def test_dk_neq_dv(self):
        reason = self._reason(d_k=16, d_v=8)
        assert "d_k == d_v" in reason

    def test_dk_outside_envelope_set(self):
        reason = self._reason(d_k=17, d_v=17)
        assert "d_k must be in" in reason

    def test_chunk_size_too_large(self):
        reason = self._reason(chunk_size=65)
        assert "chunk_size must be" in reason

    def test_nh_times_chunk_size_too_large(self):
        reason = self._reason(nh=3, chunk_size=64)  # 3 * 64 = 192 > 128
        assert "nh * chunk_size" in reason

    def test_every_violation_reason_is_distinct(self):
        reasons = [
            self._reason(is_cuda=False),
            self._reason(q_dtype=torch.float16),
            self._reason(k_dtype=torch.float16),
            self._reason(v_dtype=torch.float16),
            self._reason(beta_dtype=torch.float16),
            self._reason(h0_dtype=torch.float16),
            self._reason(d_k=16, d_v=8),
            self._reason(d_k=17, d_v=17),
            self._reason(chunk_size=65),
            self._reason(nh=3, chunk_size=64),
        ]
        assert len(set(reasons)) == len(reasons)


# ===========================================================================
# 8. DeltaMinGRU seam layout round trip (value-level pin)
# ===========================================================================
#
# Section 6 covers control flow (auto/eager/triton branching); this section
# pins the seam's actual data movement. Shape-only assertions can't catch a
# permute regression that keeps the shape right but scrambles which value
# lands where (e.g. swapping the ``T``/``n_heads`` axes when both keep the
# same size at some other call site), so every dim here is pairwise distinct
# (B=2, T=7, n_heads=3, nh=4, d_k=5, d_v=6) and every coordinate carries a
# unique marker value, so a mis-permuted axis is caught even when the
# resulting shape happens to be legal.
#
# ``MINGRU_SCAN=triton`` is used (not ``auto`` + a CUDA stand-in) because it
# skips ``_resolve_scan_mode``'s CPU-``auto`` short circuit unconditionally
# (see its body: the ``mode == "auto" and not is_cuda`` check never fires
# for ``"triton"``), so the seam runs end-to-end on a real CPU tensor -- the
# same trick ``TestTritonModeCPU`` already relies on above.
#
# ``DeltaMinGRU._coeffs`` is monkeypatched to emit deterministic
# per-coordinate-unique markers (real ``_coeffs`` output is unpredictable
# Linear-layer noise, not useful for pinning *which* coordinate moved
# *where*); the fake ``triton_scans.delta_scan_impl`` stand-in is installed
# via ``sys.modules`` exactly like ``TestDeltaEnvelopeReasons``' local-import
# idiom expects it to be reachable, following ``_isolate_dispatch``'s
# existing CPU-only mocked-availability convention.


class TestDeltaSeamLayoutRoundTrip:
    B, T, N_HEADS, NH, D_K, D_V = 2, 7, 3, 4, 5, 6

    def _marker_coeffs(self, x):
        """Deterministic ``_coeffs`` stand-in: every coordinate gets a unique value.

        Ignores ``x`` entirely -- shapes come from the class-level dims above,
        not from any real projection, so the values calling code receives are
        exactly identifiable, not incidental floating-point noise.
        """
        B, T, n_heads, nh, d_k, d_v = self.B, self.T, self.N_HEADS, self.NH, self.D_K, self.D_V
        q = torch.arange(B * T * n_heads * d_k, dtype=torch.float32).reshape(B, T, n_heads, d_k)
        ks, vs, betas = [], [], []
        for j in range(nh):
            ks.append(
                10_000 * (j + 1)
                + torch.arange(B * T * n_heads * d_k, dtype=torch.float32).reshape(
                    B, T, n_heads, d_k
                )
            )
            vs.append(
                20_000 * (j + 1)
                + torch.arange(B * T * n_heads * d_v, dtype=torch.float32).reshape(
                    B, T, n_heads, d_v
                )
            )
            betas.append(
                30_000 * (j + 1)
                + torch.arange(B * T * n_heads, dtype=torch.float32).reshape(B, T, n_heads)
            )
        return q, ks, vs, betas

    def test_layout_round_trip_value_level(self, monkeypatch):
        monkeypatch.setenv("MINGRU_SCAN", "triton")
        B, T, n_heads, nh, d_k, d_v = self.B, self.T, self.N_HEADS, self.NH, self.D_K, self.D_V

        torch.manual_seed(SEED)
        layer = mg.DeltaMinGRU(
            input_size=3,
            hidden_size=n_heads * d_v,
            n_heads=n_heads,
            nh=nh,
            d_k=d_k,
            d_v=d_v,
        )
        monkeypatch.setattr(layer, "_coeffs", self._marker_coeffs)
        q_expected, ks_expected, vs_expected, betas_expected = self._marker_coeffs(None)

        # A marker H0 too, so a future bug that permutes H0's own d_k/d_v
        # axes (it isn't permuted today -- forward's H is passed straight
        # through -- but nothing stops a future edit from adding one) would
        # also be caught, not just assumed correct because it's zero.
        h0_flat = torch.arange(B * n_heads * d_k * d_v, dtype=torch.float32).reshape(
            B, 1, n_heads * d_k * d_v
        )
        h0_expected = h0_flat.reshape(B, n_heads, d_k, d_v)

        received = {}

        class _FakeScanFallback(Exception):
            pass

        class _FakeTritonScans:
            ScanFallback = _FakeScanFallback

            @staticmethod
            def available():
                return True

            @staticmethod
            def delta_scan_impl(Q, K, V, beta, H0, *, chunk_size):
                received["Q"] = Q
                received["K"] = K
                received["V"] = V
                received["beta"] = beta
                received["H0"] = H0
                Bq, nhq, Tq, dkq = Q.shape
                dvq = V.shape[-1]
                y = torch.arange(Bq * nhq * Tq * dvq, dtype=torch.float32).reshape(Bq, nhq, Tq, dvq)
                received["y_from_kernel"] = y
                H_T = torch.zeros(Bq, nhq, dkq, dvq)
                return y, H_T

        # `from mingru import triton_scans` (inside `_resolve_triton_dispatch`)
        # resolves via `getattr(mingru_pkg, "triton_scans")` first, which an
        # earlier real import in this same process (section 7's local
        # `from mingru import triton_scans`) may already have cached on the
        # `mingru` package object -- overwriting only `sys.modules` would
        # then be silently ignored. Patch both so the fake wins regardless
        # of test execution order.
        monkeypatch.setitem(sys.modules, _TRITON_MODULE, _FakeTritonScans)
        monkeypatch.setattr(mingru, "triton_scans", _FakeTritonScans, raising=False)

        x = torch.randn(B, T, 3)
        out = layer(x, h_0=h0_flat)

        # -- (a) inbound layouts: spec section 6 shapes + per-coordinate values --
        Q, K, V, beta, H0 = (
            received["Q"],
            received["K"],
            received["V"],
            received["beta"],
            received["H0"],
        )
        assert tuple(Q.shape) == (B, n_heads, T, d_k)
        assert tuple(K.shape) == (B, n_heads, T, nh, d_k)
        assert tuple(V.shape) == (B, n_heads, T, nh, d_v)
        assert tuple(beta.shape) == (B, n_heads, T, nh)
        assert tuple(H0.shape) == (B, n_heads, d_k, d_v)
        assert torch.equal(H0, h0_expected)

        for b in range(B):
            for h in range(n_heads):
                for t in range(T):
                    assert torch.equal(Q[b, h, t], q_expected[b, t, h])
                    for j in range(nh):
                        assert torch.equal(K[b, h, t, j], ks_expected[j][b, t, h])
                        assert torch.equal(V[b, h, t, j], vs_expected[j][b, t, h])
                        assert torch.equal(beta[b, h, t, j], betas_expected[j][b, t, h])

        # -- (b) outbound restore + reshape + out_proj --
        y_restored = received["y_from_kernel"].permute(0, 2, 1, 3)  # (B, T, n_heads, d_v)
        y_flat = y_restored.reshape(B, T, n_heads * d_v)
        expected_out = layer.out_proj(y_flat)
        assert torch.equal(out, expected_out)
