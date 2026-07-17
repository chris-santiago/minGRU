# Triton kernels

The `mingru.triton_scans` surface, imported lazily on first access (see the [Triton scans explanation](../explanation/triton-scans.md) for kernel design and the [dispatch how-to](../how-to/control-scan-dispatch.md) for `MINGRU_SCAN`). Importing `mingru` alone does not import this module; touching any name below does. The kernel wrappers require a CUDA build of PyTorch (`torch >= 2.8`); the availability probe, fallback signal, and impl registry are present on every build.

## Availability and registry

::: mingru.triton_scans.available

::: mingru.triton_scans.ScanFallback

::: mingru.triton_scans.SCAN_IMPLS

## Kernel entry points

::: mingru.triton_scans.angle_scan_impl

::: mingru.triton_scans.affine_scan_fwd

::: mingru.triton_scans.affine_scan_bwd

::: mingru.triton_scans.linear_scan_fwd

::: mingru.triton_scans.linear_scan_bwd

::: mingru.triton_scans.parallel_scan_log_fwd

::: mingru.triton_scans.angle_scan_fwd

::: mingru.triton_scans.angle_scan_bwd
