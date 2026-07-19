"""One-shot L4 diagnostic: does the fused delta backward engage, per envelope class?

Ground-truth instrument for the Task 6b SMEM tuning loop. Three successive
analytic shared-memory models each missed an envelope class (the compiler's
operand staging is not reliably predictable from first principles), so this
probe replaces estimation with measurement: for every envelope class
(d_k x (nh, chunk_size)), it drives ``delta_scan_impl`` forward + backward on
tiny tensors and reports, per class:

- whether the fused backward ENGAGED or fell back (with the parsed
  ``Required: <bytes>`` from the OutOfResources message when it didn't), and
- the actual compiled shared-memory size (``metadata.shared``) of every delta
  kernel newly compiled for that class, straight from the Triton JIT cache.

Output is one ``MINGRU_SMEM <json>`` marked line per class plus a final
``MINGRU_SMEM_DONE <json>`` summary -- the marked-line transport the repo's
other job scripts use. OOM fallbacks are the *finding*, not a failure: the
probe exits 0 unless something unexpected breaks, so one cheap job always
yields the full table.

Run inside a Lightning job (foreground-only command chain, no keepalive):
``cd /tmp/minGRU && python scripts/delta_smem_probe.py``
"""

from __future__ import annotations

import json
import re
import sys
import warnings

sys.path.insert(0, "src")

import torch  # noqa: E402

from mingru import triton_scans as ts  # noqa: E402

# Every (nh, chunk_size) family the envelope admits at its M ceiling, crossed
# with every envelope d_k. T = ragged multi-chunk (3 chunks incl. a partial
# tail) so the probe compiles the same worst-path variants the parity grid hits.
_FAMILIES = ((1, 64), (2, 64), (4, 32))
_DKS = (4, 8, 16, 32, 64)
_KERNELS = (
    "_delta_prepass_kernel",
    "_delta_state_kernel",
    "_delta_readout_kernel",
    "_delta_bwd_prepass_kernel",
    "_delta_bwd_state_kernel",
    "_delta_bwd_grad_kernel",
)

_REQUIRED_RE = re.compile(r"Required: (\d+)")


def _cache_snapshot(kern) -> dict[object, int]:
    """Map every compiled-cache entry of one JIT kernel to its SMEM bytes.

    Defensive against Triton cache-layout differences: unknown shapes yield
    an empty map rather than a crash (the engage/fallback column never
    depends on this introspection).
    """
    out: dict[object, int] = {}
    cache = getattr(kern, "cache", None)
    if not isinstance(cache, dict):
        return out
    for per_device in cache.values():
        if not isinstance(per_device, dict):
            continue
        for key, compiled in per_device.items():
            meta = getattr(compiled, "metadata", None)
            shared = getattr(meta, "shared", None)
            if isinstance(shared, int):
                out[(id(per_device), key)] = shared
    return out


def _probe_class(d_k: int, nh: int, chunk_size: int) -> dict:
    B, n_heads, d_v = 2, 2, d_k
    T = 2 * chunk_size + chunk_size // 4
    torch.manual_seed(0)
    dev = "cuda"
    Q = torch.randn(B, n_heads, T, d_k, device=dev, requires_grad=True)
    K = torch.randn(B, n_heads, T, nh, d_k, device=dev, requires_grad=True)
    V = torch.randn(B, n_heads, T, nh, d_v, device=dev, requires_grad=True)
    beta = torch.rand(B, n_heads, T, nh, device=dev, requires_grad=True)
    H0 = torch.randn(B, n_heads, d_k, d_v, device=dev, requires_grad=True)

    before = {name: _cache_snapshot(getattr(ts, name)) for name in _KERNELS}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y, H_T = ts.delta_scan_impl(Q, K, V, beta, H0, chunk_size=chunk_size)
        (y.square().sum() + H_T.square().sum()).backward()
    torch.cuda.synchronize()

    fallback = [
        str(w.message) for w in caught if issubclass(w.category, ts._DeltaBackwardFallbackWarning)
    ]
    required = None
    if fallback:
        match = _REQUIRED_RE.search(fallback[0])
        required = int(match.group(1)) if match else None

    row = {
        "d_k": d_k,
        "nh": nh,
        "chunk_size": chunk_size,
        "M": nh * chunk_size,
        "T": T,
        "engaged": not fallback,
        "oom_required_bytes": required,
        "fallback_reason": fallback[0] if fallback else None,
        "kernel_smem_bytes": {},
    }
    for name in _KERNELS:
        after = _cache_snapshot(getattr(ts, name))
        new = [shared for key, shared in after.items() if key not in before[name]]
        if new:
            row["kernel_smem_bytes"][name] = max(new)
    return row


def main() -> int:
    assert torch.cuda.is_available(), "SMEM probe needs a CUDA device"
    status = ts.available()
    assert status is True, f"triton kernels unavailable: {status}"
    hw_limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    rows = []
    for d_k in _DKS:
        for nh, chunk_size in _FAMILIES:
            row = _probe_class(d_k, nh, chunk_size)
            rows.append(row)
            print("MINGRU_SMEM " + json.dumps(row), flush=True)
    summary = {
        "hw_limit_optin_bytes": hw_limit,
        "classes": len(rows),
        "engaged": sum(r["engaged"] for r in rows),
        "fallbacks": [
            {k: r[k] for k in ("d_k", "nh", "chunk_size", "oom_required_bytes")}
            for r in rows
            if not r["engaged"]
        ],
    }
    print("MINGRU_SMEM_DONE " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
