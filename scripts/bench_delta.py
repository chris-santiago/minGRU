"""CPU bench: DeltaMinGRU's three forward paths, uncontended fwd+bwd (Task 4).

Spec: ``.claude/output/specs/2026-07-17-delta-mixer-design.md`` section 9.9
("A CPU bench row ... records chunked-WY faster than the naive affine-scan
path") and section 10 ("The efficiency claim is validated by measurement
... never asserted"). Intent ledger statement 4: a recorded CPU bench row
showing the chunked-WY path faster than the naive affine-scan path at
B=128,T=64 and at B=128,T=1024, entered per evidence discipline -- this is
what lets the docs correction (Task 5) claim an efficient parallel delta
path honestly.

Three ``DeltaMinGRU`` arms, all built from the *same* seeded instance and
the *same* input tensor per shape (identical weights, identical data;
only the computation path differs), so the comparison is apples-to-apples:

``sequential``
    The hand-rolled per-token, per-micro-step recurrence
    (``H <- (I - beta k k^T) H + beta k v^T``) mirroring
    ``tests/test_mixers.py``'s ``_ref_delta_forward`` -- correct by
    construction (the lab-certified math) but ``O(T * nh)`` sequential
    Python-loop steps. Reimplemented locally rather than imported from
    ``tests/`` (this repo's convention: tests stay out of the runtime
    import path); the reimplementation is a direct transcription with no
    new logic.

``naive_affine_scan``
    Composes each token's ``nh`` micro-steps into a per-token affine map
    and reduces with the frozen ``matrix_affine_scan``, mirroring
    ``tests/test_mixers.py``'s ``_ref_affine_delta_forward`` -- the
    "measured-slow" path the chunked-WY form is designed to avoid (it
    materializes a ``d_k x d_k`` transition matrix per token).

``chunked_wy``
    ``DeltaMinGRU.forward`` itself -- the shipped chunked-WY UT-transform
    path.

A fourth, unrelated arm rides along for cross-mixer context:

``givens`` (cross-mixer reference, NOT part of the delta three-way
comparison or its agreement gate)
    The packaged ``mingru.GivensMinGRU`` (``block_size=8, rounds=3``, same
    ``input_size=hidden_size=64``, default ``MINGRU_SCAN`` -- eager on
    CPU), timed the same way at the same two shapes. Because this whole
    script runs under the evidence pin (torch 2.5.1, see above), this
    arm's number is directly comparable to the previously-recorded
    Givens CPU cost (``experiments/EXPERIMENTS.md``'s ``hetero-loop-17/18``
    round, 0.961s at the lab shape) -- same torch, same machine, not
    merely contextual. It computes a genuinely different function on a
    separately-built module (own weights, own state shape) than the
    delta arms, so it is deliberately excluded from
    ``_verify_arms_agree``: an agreement check across mixer families
    would be meaningless (they are not supposed to agree).

Before any timing, ``_verify_arms_agree`` forward-compares the three
*delta* arms (pairwise, ``atol=1e-5``) at the full bench config's
``T=64`` shape and at a small ragged shape (``B=4, T=13,
chunk_size=5``) that exercises chunk-boundary logic the ``T=64``
config's ``chunk_size=64`` (>= T) never does. This is a self-check on
this *script's* own arm implementations (the sequential/naive
reimplementations could themselves have a bug, and a bug that makes one
arm compute a different, cheaper function would produce a
meaningless-but-plausible-looking timing win) -- it is not a substitute
for ``tests/test_mixers.py``'s dual-oracle correctness suite, which is
what actually certifies ``DeltaMinGRU.forward``. On any disagreement
the gate prints every pairwise max-abs-diff, then the whole script
exits nonzero before running a single timed iteration -- a benchmark of
arms that don't compute the same function is not evidence of anything.
``T=1024`` is deliberately skipped here (the naive arm's cost there is
why the timing section itself takes several minutes; a third agreement
check at that shape would add a lot of runtime for very little
incremental confidence given ``T=64`` + the ragged shape already
exercise both the non-degenerate multi-chunk path and chunk boundaries,
and ``tests/test_mixers.py`` already parametrizes its own oracles up to
``T=1024``).

Timing methodology: single-process, uncontended (run this script alone,
nothing else competing for CPU), one discarded warmup iteration then the
min of 3 timed forward+backward iterations per (arm, shape) cell --
matching this repo's existing bench convention (see the "Efficiency,
measured" paragraphs in ``experiments/EXPERIMENTS.md``'s
``hetero-loop-17/18`` round). Each iteration zeros gradients, runs the
arm's forward, sums the output to a scalar loss, and calls ``.backward()``
-- ``time.perf_counter()`` brackets exactly that (no CUDA events: this is
a CPU-only bench, wall-clock is the right instrument here).

Writes ``experiments/bench/delta_paths.md`` (table + config + torch
version + CPU info) and a ``delta_paths.json`` companion (same rows,
machine-readable), matching the ``experiments/bench/*.{json,md}`` pairing
``scripts/bench_scans.py`` already established.

Evidence-pin requirement (user decision, supersedes an earlier two-table
draft of this script): this bench is recorded under this repository's
evidence-pin environment, ``torch==2.5.1`` -- the SAME torch version (and
machine) as the previously-recorded ``GivensMinGRU``/sequential-delta
numbers in ``experiments/EXPERIMENTS.md``'s ``hetero-loop-17/18`` round
(0.961s / 0.179s at the lab shape), so every number this script produces
is directly comparable to that recorded evidence, not merely
contextual. The packaged ``mingru`` distribution's declared
``torch>=2.8`` floor (``pyproject.toml``) is install/packaging metadata
for the Triton GPU kernel surface (``mingru.triton_scans``, gated
separately, see ``src/mingru/__init__.py``'s module docstring); the
*eager* path this bench exercises (every arm here, including the new
``DeltaMinGRU``, which uses nothing newer than
``torch.linalg.solve_triangular``) runs under the pin by design, matching
how every other recorded evidence command in this repo already runs
(``uv run --python 3.12 --with 'torch==2.5.1' python ...``, see
``experiments/EXPERIMENTS.md``). ``run()`` asserts
``torch.__version__ == "2.5.1"`` and refuses to write any artifact
otherwise -- a bench recorded under the wrong torch would misrepresent
comparability to the historical rows it's meant to sit beside.

Import-path note: unlike a plain ``pip install -e .`` invocation (where
``from mingru import DeltaMinGRU`` resolves via site-packages
regardless of CWD), the evidence-pin invocation above does not
necessarily sync/install this project (its ``pyproject.toml`` pins
``torch>=2.8``, a real conflict with the ``torch==2.5.1`` pin). This
script therefore mirrors the root ``min_gru.py`` evidence driver's own
bootstrap: it inserts ``src/`` onto ``sys.path`` ahead of site-packages
(derived from ``__file__``, not the CWD) before importing ``mingru``, so
the import resolves from the source tree either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

# See the module docstring's "Import-path note": insert `src/` onto
# `sys.path` ahead of site-packages, mirroring the root `min_gru.py`
# evidence driver's own `sys.path.insert(0, str(Path(__file__).resolve()
# .parent / "src"))` -- so `from mingru import ...` below resolves from
# the source tree whether or not this project is pip/uv-installed. Must
# happen before the `mingru` import; every stdlib/third-party import below
# that line is consequently not "at the top of the file" in the strict
# PEP 8 sense (ruff's E402), which is why this file carries a narrow,
# named `pyproject.toml` lint exemption for exactly that rule, matching
# the precedent already set for the root `min_gru.py`/`triton_scans.py`
# evidence drivers.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

from mingru import DeltaMinGRU, GivensMinGRU
from mingru.min_gru import matrix_affine_scan

_OUT_DIR = _REPO_ROOT / "experiments" / "bench"
_OUT_MD = _OUT_DIR / "delta_paths.md"
_OUT_JSON = _OUT_DIR / "delta_paths.json"

_EVIDENCE_PIN_TORCH_VERSION = "2.5.1"

# Bench config (stated verbatim in the artifact per the brief). A moderate
# multi-head, multi-micro-step DeltaProduct configuration (nh=2 exercises
# the micro-step loop the chunk WY transform batches; n_heads=4 exercises
# the head-batched einsums) -- not the DeltaNet default (nh=1), so the
# naive arm's per-token d_k x d_k transition composition (the "nh" loop in
# ``_naive_affine_scan_forward`` below) is actually exercised more than
# once per token.
_INPUT_SIZE = 64
_HIDDEN_SIZE = 64
_N_HEADS = 4
_NH = 2
_D_K = 16
_D_V = 16
_CHUNK_SIZE = 64  # DeltaMinGRU's default
_SEED = 0

# Cross-mixer reference arm config: mirrors the recorded `hetero-loop-17/18`
# GivensMinGRU config (block_size=8, rounds=3) at the same input/hidden
# size as the delta arms above, so this run's number sits next to that
# round's 0.961s-at-the-lab-shape figure with only the environment (torch
# version, machine) differing, not the model config.
_GIVENS_BLOCK_SIZE = 8
_GIVENS_ROUNDS = 3
_GIVENS_ARM_NAME = "GivensMinGRU (packaged, cross-mixer reference)"

_WARMUP = 1
_REPS = 3

_SHAPES = ({"label": "T=64", "B": 128, "T": 64}, {"label": "T=1024", "B": 128, "T": 1024})

# Small ragged shape for the pre-timing agreement gate only (never timed):
# T=13 is not a multiple of chunk_size=5, so the chunked-WY arm's final
# chunk is ragged (C=3 < chunk_size) -- exercising the chunk-boundary path
# the T=64 config's chunk_size=64 (>= T, a single whole chunk) never does.
_RAGGED_SHAPE = {"label": "ragged (chunk-boundary)", "B": 4, "T": 13}
_RAGGED_CHUNK_SIZE = 5
_VERIFY_ATOL = 1e-5


# --- local oracle reimplementations (mirrors tests/test_mixers.py) --------


def _sequential_forward(layer: DeltaMinGRU, x: torch.Tensor) -> torch.Tensor:
    """Per-token, per-micro-step recurrence; mirrors ``_ref_delta_forward``.

    ``step`` itself is ``@torch.no_grad()`` by design (matches the other
    mixers), so it cannot anchor a fwd+bwd timing -- this is the same
    recurrence, differentiable, driven token-by-token exactly as ``step``
    would be in a training loop.
    """
    B_, T_, _ = x.shape
    n_heads, d_k, d_v, nh = layer.n_heads, layer.d_k, layer.d_v, layer.nh
    q = layer.linear_q(x).view(B_, T_, n_heads, d_k)
    ks = [F.normalize(lin(x).view(B_, T_, n_heads, d_k), dim=-1) for lin in layer.linear_k]
    vs = [lin(x).view(B_, T_, n_heads, d_v) for lin in layer.linear_v]
    betas = [2 * torch.sigmoid(lin(x)) for lin in layer.linear_beta]
    H = x.new_zeros(B_, n_heads, d_k, d_v)
    ys = []
    for t in range(T_):
        for j in range(nh):
            k, v, beta = ks[j][:, t], vs[j][:, t], betas[j][:, t]
            kH = torch.einsum("bhk,bhkv->bhv", k, H)
            beta_ = beta[..., None, None]
            H = H - beta_ * torch.einsum("bhk,bhv->bhkv", k, kH)
            H = H + beta_ * torch.einsum("bhk,bhv->bhkv", k, v)
        ys.append(torch.einsum("bhk,bhkv->bhv", q[:, t], H))
    y = torch.stack(ys, dim=1).reshape(B_, T_, n_heads * d_v)
    return layer.out_proj(y)


def _naive_affine_scan_forward(layer: DeltaMinGRU, x: torch.Tensor) -> torch.Tensor:
    """Per-token affine maps reduced by ``matrix_affine_scan``; mirrors
    ``_ref_affine_delta_forward``.

    Materializes a ``d_k x d_k`` transition per token (the composition of
    the token's ``nh`` micro-steps) -- exactly the "measured-slow" path
    the chunked-WY ``forward`` is designed to avoid.
    """
    B_, T_, _ = x.shape
    n_heads, d_k, d_v, nh = layer.n_heads, layer.d_k, layer.d_v, layer.nh
    q = layer.linear_q(x).view(B_, T_, n_heads, d_k)
    ks = [F.normalize(lin(x).view(B_, T_, n_heads, d_k), dim=-1) for lin in layer.linear_k]
    vs = [lin(x).view(B_, T_, n_heads, d_v) for lin in layer.linear_v]
    betas = [2 * torch.sigmoid(lin(x)) for lin in layer.linear_beta]

    eye = torch.eye(d_k, dtype=x.dtype).expand(B_, T_, n_heads, d_k, d_k)
    A = eye
    Bm = x.new_zeros(B_, T_, n_heads, d_k, d_v)
    for j in range(nh):
        k, v, beta = ks[j], vs[j], betas[j]
        beta_ = beta[..., None, None]
        kkT = torch.einsum("ntha,nthb->nthab", k, k)
        A_j = eye - beta_ * kkT
        B_j = beta_ * torch.einsum("ntha,nthv->nthav", k, v)
        A = torch.einsum("nthab,nthbc->nthac", A_j, A)
        Bm = torch.einsum("nthab,nthbv->nthav", A_j, Bm) + B_j

    _, Bbar = matrix_affine_scan(A, Bm)
    y = torch.einsum("ntha,nthav->nthv", q, Bbar).reshape(B_, T_, n_heads * d_v)
    return layer.out_proj(y)


@dataclass(frozen=True)
class _Arm:
    """One bench arm: a display name and the forward callable it times.

    ``forward`` is typed over ``nn.Module`` (not ``DeltaMinGRU``
    specifically) so the same dataclass also describes the cross-mixer
    ``GivensMinGRU`` reference arm, which is timed the same way but is
    not one of the three ``_ARMS`` below (see module docstring).
    """

    name: str
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor]


_ARMS = (
    _Arm("sequential step-loop (oracle)", _sequential_forward),
    _Arm("naive affine-scan reduction (oracle)", _naive_affine_scan_forward),
    _Arm("chunked-WY (shipped forward)", lambda layer, x: layer(x)),
)


# --- timing -----------------------------------------------------------------


def _time_fwd_bwd(
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    layer: nn.Module,
    x: torch.Tensor,
    warmup: int,
    reps: int,
) -> float:
    """Min-of-``reps`` wall-clock seconds for one forward+backward iteration.

    ``layer`` is typed over ``nn.Module`` (not ``DeltaMinGRU``
    specifically) since this same helper times the cross-mixer
    ``GivensMinGRU`` reference arm too.

    Gradients (on both the module's parameters and ``x``, which carries
    ``requires_grad=True``) are cleared before every iteration -- including
    warmup -- so repeated backward calls never accumulate across
    iterations and each timed window measures only that iteration's own
    forward+backward cost.
    """
    times: list[float] = []
    for i in range(warmup + reps):
        layer.zero_grad(set_to_none=True)
        x.grad = None
        start = time.perf_counter()
        loss = forward(layer, x).sum()
        loss.backward()
        elapsed = time.perf_counter() - start
        if i >= warmup:
            times.append(elapsed)
    return min(times)


def _build_layer_and_input(
    shape: dict, chunk_size: int = _CHUNK_SIZE
) -> tuple[DeltaMinGRU, torch.Tensor]:
    """One seeded ``DeltaMinGRU`` + input for ``shape``, shared across the delta arms.

    Same seed for every shape (only ``B``/``T``/``chunk_size`` vary) and
    the identical ``layer``/``x`` pair is reused by every delta arm -- so
    the three arms time (or, in the agreement gate, forward-compare) the
    same weights on the same data, differing only in computation path.
    ``chunk_size`` defaults to the timed bench config but is overridable
    for the agreement gate's ragged shape (``chunk_size=5``, see
    ``_RAGGED_CHUNK_SIZE``) -- ``chunked_wy``'s output must not depend on
    it (chunk size is a performance-only knob per ``DeltaMinGRU``'s own
    docstring), so varying it here is exactly what makes the ragged check
    a meaningful chunk-boundary exercise rather than a repeat of ``T=64``.
    """
    torch.manual_seed(_SEED)
    layer = DeltaMinGRU(
        _INPUT_SIZE,
        _HIDDEN_SIZE,
        n_heads=_N_HEADS,
        nh=_NH,
        d_k=_D_K,
        d_v=_D_V,
        chunk_size=chunk_size,
    )
    x = torch.randn(shape["B"], shape["T"], _INPUT_SIZE, requires_grad=True)
    return layer, x


def _build_givens_and_input(shape: dict) -> tuple[GivensMinGRU, torch.Tensor]:
    """One seeded ``GivensMinGRU`` + input for ``shape`` -- the cross-mixer reference arm.

    Independent of the delta arms' ``layer``/``x`` (different mixer,
    different parameter shapes) but seeded and shaped the same way for
    this script's own run-to-run reproducibility.
    """
    torch.manual_seed(_SEED)
    layer = GivensMinGRU(
        _INPUT_SIZE, _HIDDEN_SIZE, block_size=_GIVENS_BLOCK_SIZE, rounds=_GIVENS_ROUNDS
    )
    x = torch.randn(shape["B"], shape["T"], _INPUT_SIZE, requires_grad=True)
    return layer, x


# --- pre-timing agreement gate ----------------------------------------------


def _verify_arms_agree() -> bool:
    """Forward-only pairwise agreement check for the three delta arms.

    Runs *before* any timing (see ``run``): the timing loop shares one
    ``layer``/``x`` per shape across all three delta arms specifically so
    a timing win can't be an artifact of one arm silently computing a
    different (cheaper) function than the others -- this gate is what
    actually establishes that precondition, rather than just asserting
    it. Checked at two shapes: the full bench config's ``T=64`` (a single
    whole chunk, since ``chunk_size=64 >= T``) and a small ragged shape
    (``B=4, T=13, chunk_size=5``, three chunks with a ragged final chunk
    of size 3) that exercises chunk-boundary logic ``T=64`` never
    reaches. ``T=1024`` is deliberately not checked here (module
    docstring explains why); ``GivensMinGRU`` is deliberately not
    included (a different mixer computing a different function -- an
    agreement check across mixer families would be meaningless).

    Every pairwise max-abs-diff is printed (pass or fail) for both
    shapes; returns ``True`` iff all six comparisons (3 pairs x 2
    shapes) are within ``_VERIFY_ATOL``.
    """
    checks = (
        (_SHAPES[0], _CHUNK_SIZE),  # T=64, the full bench config's chunk_size
        (_RAGGED_SHAPE, _RAGGED_CHUNK_SIZE),
    )
    all_ok = True
    for shape, chunk_size in checks:
        layer, x = _build_layer_and_input(shape, chunk_size)
        with torch.no_grad():
            outputs = {arm.name: arm.forward(layer, x) for arm in _ARMS}
        names = [arm.name for arm in _ARMS]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                diff = (outputs[a] - outputs[b]).abs().max().item()
                ok = diff <= _VERIFY_ATOL
                all_ok = all_ok and ok
                status = "OK" if ok else "MISMATCH"
                print(
                    f"  [{status}] {shape['label']} (chunk_size={chunk_size}): "
                    f"{a} vs {b}: max abs diff {diff:.3e}"
                )
    return all_ok


# --- reporting ----------------------------------------------------------------


def _cpu_info() -> str:
    """Best-effort CPU model string; falls back to ``platform.processor()``.

    ``platform.processor()`` alone is often uninformative (empty string,
    or a bare architecture tag like ``arm``) -- ``sysctl``/``/proc/cpuinfo``
    give the actual model name where available. Never raises: this is
    metadata for the artifact, not something the bench should depend on.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return out.stdout.strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _render_markdown(rows: list[dict], meta: dict) -> str:
    lines = [
        "# DeltaMinGRU forward-path bench (chunked-WY vs naive affine-scan vs sequential)",
        "",
        "Uncontended CPU forward+backward timing (1 discarded warmup, min of "
        f"{meta['reps']} timed iterations), per spec section 9.9 / intent ledger "
        "statement 4 -- validates the chunked-WY efficiency claim by measurement.",
        "",
        f"Config: input_size={meta['config']['input_size']}, "
        f"hidden_size={meta['config']['hidden_size']}, "
        f"n_heads={meta['config']['n_heads']}, nh={meta['config']['nh']}, "
        f"d_k={meta['config']['d_k']}, d_v={meta['config']['d_v']}, "
        f"chunk_size={meta['config']['chunk_size']}, seed={meta['config']['seed']}. "
        f"GivensMinGRU reference arm: block_size={meta['givens_config']['block_size']}, "
        f"rounds={meta['givens_config']['rounds']}, same input_size/hidden_size.",
        "",
        f"torch {meta['torch_version']} (evidence pin, deliberately chosen -- "
        "see below), CPU: "
        f"{meta['cpu_info']} ({meta['num_threads']} torch threads), "
        f"{meta['platform']}, commit {meta['git_commit']}, "
        f"generated {meta['timestamp']}.",
        "",
        f"Environment note: this run is pinned to torch=="
        f"{_EVIDENCE_PIN_TORCH_VERSION} (asserted at runtime; the script "
        "refuses to write this artifact under any other torch version) so "
        "every number below sits on the same torch version and machine as "
        "the previously-recorded evidence in `experiments/EXPERIMENTS.md`'s "
        "`hetero-loop-17/18` round (GivensMinGRU 0.961s, sequential delta16 "
        "0.179s at the lab shape) -- directly comparable, not merely "
        "contextual. The packaged `mingru` distribution's declared "
        "`torch>=2.8` floor (`pyproject.toml`) is install/packaging "
        "metadata for the separately-gated Triton GPU kernel surface; the "
        "eager CPU path every arm below exercises runs under the pin by "
        "design.",
        "",
        "Agreement gate (forward-only, before any timing): all pairwise "
        "comparisons among the three delta arms passed at atol=1e-5, at "
        "T=64 (full bench config) and at a ragged shape (B=4, T=13, "
        "chunk_size=5) -- see console output for the per-pair max abs "
        "diffs. GivensMinGRU is excluded from this gate (a different "
        "mixer computing a different function).",
        "",
        "| arm | T=64 fwd+bwd (s) | T=1024 fwd+bwd (s) |",
        "| --- | --- | --- |",
    ]
    by_arm: dict[str, dict] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[row["shape_label"]] = row["seconds"]
    arm_names_in_order = [arm.name for arm in _ARMS] + [_GIVENS_ARM_NAME]
    for name in arm_names_in_order:
        times = by_arm[name]
        lines.append(f"| {name} | {times['T=64']:.4f} | {times['T=1024']:.4f} |")

    naive = by_arm["naive affine-scan reduction (oracle)"]
    chunked = by_arm["chunked-WY (shipped forward)"]
    sequential = by_arm["sequential step-loop (oracle)"]
    givens = by_arm[_GIVENS_ARM_NAME]
    ratio_t64 = naive["T=64"] / chunked["T=64"]
    ratio_t1024 = naive["T=1024"] / chunked["T=1024"]
    seq_ratio_t64 = sequential["T=64"] / chunked["T=64"]
    seq_ratio_t1024 = sequential["T=1024"] / chunked["T=1024"]
    givens_ratio_t64 = givens["T=64"] / chunked["T=64"]
    givens_ratio_t1024 = givens["T=1024"] / chunked["T=1024"]

    lines += [
        "",
        "Ratios (chunked-WY speedup, delta arms only):",
        f"- vs naive affine-scan: {ratio_t64:.2f}x at T=64, {ratio_t1024:.2f}x at T=1024",
        f"- vs sequential step-loop: {seq_ratio_t64:.2f}x at T=64, "
        f"{seq_ratio_t1024:.2f}x at T=1024",
        "",
        "Cross-mixer reference (not part of the delta comparison above): "
        f"GivensMinGRU / chunked-WY time ratio {givens_ratio_t64:.2f}x at T=64, "
        f"{givens_ratio_t1024:.2f}x at T=1024 (>1 means GivensMinGRU is slower "
        "than chunked-WY DeltaMinGRU on this run; not a like-for-like "
        "comparison -- different mixer, different math -- included only for "
        "same-environment context against the torch-2.5.1-era 0.961s figure).",
        "",
        f"Acceptance (spec section 9.9, delta arms only): chunked-WY beats naive "
        f"affine-scan at both shapes -- "
        f"{'PASS' if ratio_t64 > 1 and ratio_t1024 > 1 else 'FAIL'}.",
    ]
    return "\n".join(lines) + "\n"


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def run() -> int:
    """Verify torch pin + arm agreement, time every cell, write the artifact.

    Returns 0 iff (a) torch is exactly the evidence-pin version, (b) the
    pre-timing agreement gate passes (the three delta arms compute the
    same function), and (c) the chunked-WY arm beats the naive
    affine-scan arm at both shapes (spec section 9.9's acceptance
    criterion) -- a nonzero return means one of: the wrong torch version
    (no artifact written), the gate caught a real disagreement (timing
    never ran), or the recorded row does NOT show the required speedup
    (artifact IS written -- a CONCERN, not hidden).
    """
    print(f"torch {torch.__version__} (evidence-pin requirement: =={_EVIDENCE_PIN_TORCH_VERSION})")
    if torch.__version__ != _EVIDENCE_PIN_TORCH_VERSION:
        print(
            f"FAILED: this bench must run under torch=={_EVIDENCE_PIN_TORCH_VERSION} "
            "(the repository's evidence-pin environment, matching the "
            "previously-recorded GivensMinGRU/sequential-delta rows in "
            f"experiments/EXPERIMENTS.md) but found torch=={torch.__version__} "
            "-- refusing to write an artifact that would misrepresent "
            "comparability to that recorded evidence. Re-run via, e.g., "
            "`uv run --no-project --with torch==2.5.1 python "
            "scripts/bench_delta.py`."
        )
        return 1

    print("\n=== agreement gate (forward-only, before any timing) ===")
    if not _verify_arms_agree():
        print(
            "\nFAILED: delta arm outputs disagree above atol="
            f"{_VERIFY_ATOL:g} -- aborting before any timing (a benchmark "
            "of arms that don't compute the same function is not evidence "
            "of anything)."
        )
        return 1
    print("agreement gate: all delta arm pairs agree within tolerance\n")

    rows: list[dict] = []
    for shape in _SHAPES:
        print(f"shape {shape['label']} (B={shape['B']}, T={shape['T']}):")
        layer, x = _build_layer_and_input(shape)
        for arm in _ARMS:
            seconds = _time_fwd_bwd(arm.forward, layer, x, _WARMUP, _REPS)
            print(f"  [ok] {arm.name}: {seconds:.4f}s")
            rows.append(
                {
                    "arm": arm.name,
                    "shape_label": shape["label"],
                    "B": shape["B"],
                    "T": shape["T"],
                    "seconds": seconds,
                }
            )

        givens, gx = _build_givens_and_input(shape)
        givens_seconds = _time_fwd_bwd(lambda m, xx: m(xx), givens, gx, _WARMUP, _REPS)
        print(f"  [ok] {_GIVENS_ARM_NAME}: {givens_seconds:.4f}s")
        rows.append(
            {
                "arm": _GIVENS_ARM_NAME,
                "shape_label": shape["label"],
                "B": shape["B"],
                "T": shape["T"],
                "seconds": givens_seconds,
            }
        )

    meta = {
        "config": {
            "input_size": _INPUT_SIZE,
            "hidden_size": _HIDDEN_SIZE,
            "n_heads": _N_HEADS,
            "nh": _NH,
            "d_k": _D_K,
            "d_v": _D_V,
            "chunk_size": _CHUNK_SIZE,
            "seed": _SEED,
        },
        "givens_config": {
            "block_size": _GIVENS_BLOCK_SIZE,
            "rounds": _GIVENS_ROUNDS,
        },
        "warmup": _WARMUP,
        "reps": _REPS,
        "torch_version": torch.__version__,
        "cpu_info": _cpu_info(),
        "num_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "git_commit": _git_commit_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
    _OUT_MD.write_text(_render_markdown(rows, meta))
    print(f"\nwrote {_OUT_JSON} and {_OUT_MD}")

    by_arm: dict[str, dict] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[row["shape_label"]] = row["seconds"]
    naive = by_arm["naive affine-scan reduction (oracle)"]
    chunked = by_arm["chunked-WY (shipped forward)"]
    beats_both = chunked["T=64"] < naive["T=64"] and chunked["T=1024"] < naive["T=1024"]
    if not beats_both:
        print(
            "CONCERN: chunked-WY does not beat naive affine-scan at both shapes "
            f"(T=64: chunked={chunked['T=64']:.4f}s naive={naive['T=64']:.4f}s; "
            f"T=1024: chunked={chunked['T=1024']:.4f}s naive={naive['T=1024']:.4f}s)"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
