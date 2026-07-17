"""minGRU-scans: parallel-scan minGRU variants with an optional Triton backend.

The eager library (the four mixers ``MinGRU``/``SignedMinGRU``/
``RotationMinGRU``/``GivensMinGRU``, the ``MinGRUBlock``/``MinGRUStack``
wrappers, and the four scan functions) is re-exported eagerly from
:mod:`mingru.min_gru`. The Triton kernel surface (:mod:`mingru.triton_scans`)
is exposed *lazily* via :pep:`562` module ``__getattr__``: ``import mingru``
never imports the Triton module, and only touching a Triton-backed name
(e.g. ``mingru.ScanFallback``, ``mingru.angle_scan_impl``) triggers its
import. This keeps ``import mingru`` working wherever the eager library works
(including torch releases older than 2.8 -- the repository's recorded
evidence environment runs the eager library under torch 2.5.1). Once
torch>=2.8 is installed,
:mod:`mingru.triton_scans` imports successfully and its three unconditional
names (``available``, ``ScanFallback``, ``SCAN_IMPLS``) resolve regardless of
platform; the other eight names (the raw kernel wrappers -- e.g.
``angle_scan_impl`` and each scan op's ``*_fwd``/``*_bwd`` pair) additionally
require a working Triton install, and so only resolve where Triton itself is
installable (currently: torch's Linux CUDA wheels). On macOS, Windows, or a
Linux CPU-only install, touching one of those eight raises
``AttributeError`` -- the import of :mod:`mingru.triton_scans` itself still
succeeds; only the attribute lookup inside it fails.

``__all__`` (and thus ``__dir__`` and ``from mingru import *``) lists only
the eager API: every Triton-module name -- including ``available``,
``ScanFallback``, and ``SCAN_IMPLS``, which exist on every Triton-importable
build -- resolves through this module's ``__getattr__`` by importing
:mod:`mingru.triton_scans`, and that import raises ``ImportError`` outright
below the module's torch>=2.8 floor (i.e. on any pre-2.8 torch).
Since ``__all__``/``__dir__``/``import *`` must resolve on *every*
build the eager API supports, no Triton name -- gated or not -- belongs in
them; all are still reachable individually via ``mingru.<name>`` attribute
access, which imports :mod:`mingru.triton_scans` lazily on first touch (the
three unconditional names resolve there on any torch>=2.8 install; the
other eight additionally need Triton, per above).
"""

from .min_gru import *  # noqa: F401,F403 -- eager public API re-export
from .min_gru import __all__ as _min_gru_all

__version__ = "0.1.0"

# Names of :mod:`mingru.triton_scans` reachable lazily via `__getattr__`,
# listed statically so resolving this tuple need not import the Triton module
# (the point of the PEP 562 lazy path). All are deliberately kept out of
# ``__all__`` below; the module docstring above is the authoritative account
# of why, and of the per-platform resolution rules. In short: the first three
# (``available``/``ScanFallback``/``SCAN_IMPLS``) are unconditional on
# torch>=2.8, the remaining eight (raw kernel wrappers) additionally require a
# working Triton install.
_TRITON_EXPORTS = (
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

__all__ = [*_min_gru_all, "__version__"]


def __getattr__(name: str) -> object:
    """Lazily resolve Triton-backed names (PEP 562).

    Importing ``mingru`` must not import ``mingru.triton_scans``; the first
    access to any Triton public name imports it on demand and returns the
    attribute.
    """
    if name in _TRITON_EXPORTS:
        from . import triton_scans

        return getattr(triton_scans, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
