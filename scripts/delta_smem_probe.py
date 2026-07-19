"""One-shot L4 diagnostic: per-forward-kernel SMEM for the delta path, per class.

Standalone diagnostic script, not a ``scripts/gpu_check.py --job`` mode: it
has no CLI/argparse surface of its own and is invoked directly via an in-job
command (see ``experiments/index.md``'s "in-job" tag and the run instructions
below), not dispatched through ``gpu_check.py``'s job-mode table. Its primary
historical role was the fused-backward SMEM-fitting campaign (six tuning
iterations reaching full envelope engagement under the shared-memory limit,
but ultimately reverted for speed — see ``experiments/EXPERIMENTS.md`` round
``gpu-delta-kernel-01``); it is retained here as a forward-kernel SMEM
instrument, since the shipping backward is the torch-composed reverse-chunk
loop and no longer has a fused-kernel SMEM budget to measure.

Ground-truth instrument for the delta forward-trio SMEM budget. Three
successive analytic shared-memory models each missed an envelope class (the
compiler's operand staging is not reliably predictable from first principles),
so this probe replaces estimation with measurement: for every envelope class
(d_k x (nh, chunk_size)), it drives ``delta_scan_impl`` forward + backward on
tiny tensors and reports, per class:

- whether the forward kernels ENGAGED (launched) or the delta path declined /
  failed (with the parsed ``Required: <bytes>`` from an OutOfResources message
  when present), and
- the actual compiled shared-memory size (``metadata.shared``) of every delta
  FORWARD kernel newly compiled for that class, straight from the Triton JIT
  cache.

The backward is now the torch-composed reverse-chunk loop (the fused Triton
backward was measured 8-12x slower than compile on L4 and reverted -- see the
``mingru.triton_scans`` module docstring), so this probe covers the forward
kernels only; "engaged" means the forward launched.

Output is one ``MINGRU_SMEM <json>`` marked line per class plus a final
``MINGRU_SMEM_DONE <json>`` summary -- the marked-line transport the repo's
other job scripts use. A declined/failed class is the *finding*, not a
failure: the probe exits 0 unless something unexpected breaks, so one cheap
job always yields the full table.

Run inside a Lightning job (foreground-only command chain, no keepalive):
``cd /tmp/minGRU && python scripts/delta_smem_probe.py``
"""

from __future__ import annotations

import json
import re
import sys

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
)

_REQUIRED_RE = re.compile(r"Required: (\d+)")


def _iter_compiled_caches(kern):
    """Yield each per-device ``{key: CompiledKernel}`` dict of a JIT kernel.

    Handles two Triton cache layouts, newest first: triton >= 3.3 keeps the
    compiled cache in ``JITFunction.device_caches``, a dict whose values are
    TUPLES ``(compiled_dict, ...)`` (the old ``JITFunction.cache`` dict of
    dicts is retained only as a compatibility shim on some builds and reads
    back empty on 3.4 -- the reason the first probe saw nothing). Falls back
    to the legacy ``cache`` dict-of-dicts on < 3.3. Anything unrecognized is
    skipped, never raised: the engage/fallback column never depends on this.
    """
    device_caches = getattr(kern, "device_caches", None)
    if isinstance(device_caches, dict):
        for entry in device_caches.values():
            # 3.4: entry is a tuple whose first element is the compiled dict.
            compiled = entry[0] if isinstance(entry, tuple) and entry else entry
            if isinstance(compiled, dict):
                yield compiled
        return
    cache = getattr(kern, "cache", None)
    if isinstance(cache, dict):
        for per_device in cache.values():
            if isinstance(per_device, dict):
                yield per_device


def _cache_snapshot(kern) -> dict[object, int]:
    """Map every compiled-cache entry of one JIT kernel to its SMEM bytes.

    Defensive against Triton cache-layout differences: unknown shapes yield
    an empty map rather than a crash (the engage/fallback column never
    depends on this introspection).
    """
    out: dict[object, int] = {}
    for per_device in _iter_compiled_caches(kern):
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

    # "engaged" == the forward kernels launched. The delta path raises
    # ScanFallback (or a kernel launch/compile error) if the forward declines;
    # the backward is the torch-composed loop and is not probed here.
    engaged = True
    fallback_reason = None
    required = None
    try:
        y, H_T = ts.delta_scan_impl(Q, K, V, beta, H0, chunk_size=chunk_size)
        (y.square().sum() + H_T.square().sum()).backward()
        torch.cuda.synchronize()
    except Exception as exc:  # ScanFallback or a kernel launch/compile failure
        engaged = False
        fallback_reason = f"{type(exc).__name__}: {exc}"
        match = _REQUIRED_RE.search(str(exc))
        required = int(match.group(1)) if match else None

    row = {
        "d_k": d_k,
        "nh": nh,
        "chunk_size": chunk_size,
        "M": nh * chunk_size,
        "T": T,
        "engaged": engaged,
        "oom_required_bytes": required,
        "fallback_reason": fallback_reason,
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
