"""In-job campaign loop for the GPU 36-seed round (Task 2).

Runs INSIDE a Lightning GPU job (submitted by ``scripts/gpu_check.py --job
hetero36``, Task 3), never invoked directly by a developer machine as the
evidence source of truth -- the ``--device cpu`` / ``--arms`` / ``--seeds`` /
``--steps`` flags below exist for local smoke testing only. See
``.claude/output/specs/2026-07-18-gpu36-round-design.md`` sections 4, 6, 7
and ``.git/sdd/gpu36/task-2-brief.md`` for the full design.

Arm matrix (spec section 6, exact)
-----------------------------------
Three arms, each 36 seeds (0-35), 1600 steps, ``--device cuda``, run
in-process against ``experiments/hetero_lab.py``'s ``run_arm`` (one process
amortizes ``torch.compile``/Inductor caches across the 72 delta seeds):

=========  =======================  ============  =============================
arm key    round                    model          backend
=========  =======================  ============  =============================
``sg8``    hetero-gpu36-sg8         hetero-pg8     ``MINGRU_SCAN=triton`` (forced)
``pd64``   hetero-gpu36-pd64        hetero-pd64    ``--compile``, scan unset
``pd1024`` hetero-gpu36-pd1024      hetero-pd1024  ``--compile``, scan unset
=========  =======================  ============  =============================

Each mixer runs its best-practical GPU path (user decision, spec section 8):
givens on the packaged Triton scan, delta on ``torch.compile``. The two
backends are intentionally different -- this round measures "each mixer as a
package user would run it on GPU", not a backend-controlled comparison.

Pre-flight (fail-loud, before seed 0)
--------------------------------------
On ``--device cuda`` (the production path), before touching seed 0:

1. ``torch.cuda.is_available()`` -- clear ``SystemExit`` if not.
2. ``torch.__version__`` starts with ``2.8.`` (this round's pinned CUDA
   stratum) -- clear ``SystemExit`` if not.
3. If the ``sg8`` arm is selected: one tiny forward+backward through a
   ``hetero-pg8``-config ``GivensMinGRU`` (``block_size=8, rounds=3``,
   matching ``hetero_lab``'s ``pgivens8`` factory exactly) on CUDA under
   ``MINGRU_SCAN=triton``. The packaged scan dispatch
   (``src/mingru/min_gru.py``'s ``_dispatch_scan``) itself raises
   ``SystemExit``-worthy ``RuntimeError`` if no usable Triton path exists --
   that raise IS the gate; this probe only guarantees it runs before seed 0.
4. If the ``pd64`` arm is selected: one tiny ``torch.compile``
   forward+backward through a ``pdelta64``-config ``DeltaMinGRU``
   (``nh=2, n_heads=1, d_k=8, d_v=8``) on CUDA, ``MINGRU_SCAN`` unset.
5. If the ``pd1024`` arm is selected: the same probe at the ``pdelta1024``
   config (``nh=2, n_heads=4, d_k=16, d_v=16`` -- matching ``hetero_lab``'s
   ``pdelta1024`` factory exactly). Kept as a separate gate from 4 rather
   than assumed-covered by the pd64 probe: the multi-head chunked-
   recurrence code path (``n_heads=4``) is materially different from the
   single-head case, and on the production arm order (sg8, pd64, pd1024)
   an Inductor failure specific to that shape would otherwise surface only
   after ~72 of 108 paid seeds. Either compile probe failing raises
   uncaught -- never a silent eager fallback (spec section 7).

Gates 3-5 only run for arms actually present in the *selected* ``--arms``
subset, so a narrow smoke invocation (e.g. ``--arms pd64``) never pays for
an unrelated probe. The production default (``--arms`` = all three) always
exercises all three.

On ``--device cpu`` (smoke only) every pre-flight gate above is skipped
-- CUDA/Triton/compile engagement has no meaning on CPU -- and no
``MINGRU_LAB_ENV`` line is printed (see below).

Row transport (spec section 6)
--------------------------------
Every seed is run via ``hetero_lab.run_arm`` with ``dry_run=True`` (the
lab's no-append path -- ``experiments/lab_results.jsonl`` is never written
from inside the job; ``scripts/gpu_check.py --job hetero36`` appends rows
locally, post-dedup, after the job completes). Per completed seed this
prints exactly one ``MINGRU_LAB_ROW <row json>`` line (the row dict
``run_arm`` returns) plus a human-readable progress line carrying that
seed's wall seconds -- the job's container filesystem dies with the job, so
these stdout lines are the sole evidence transport. ``hetero_lab.run_arm``
also unconditionally prints the raw row JSON itself before returning it
(unmarked, no prefix); at 108-seed production scale that near-duplicate
line every seed is stdout noise the job log shouldn't carry, so the
``run_arm`` call is wrapped in ``contextlib.redirect_stdout`` into a
discarded buffer -- only this module's own ``MINGRU_LAB_ROW``-prefixed line
and progress line (both printed OUTSIDE the redirected block) reach the
job log.

Cross-task env contract (orchestrator-pinned)
------------------------------------------------
After pre-flight passes on ``--device cuda``, exactly one
``MINGRU_LAB_ENV <json>`` line is printed with
``{torch, cuda_device_name, device_capability, triton, scan_mode_givens,
compile_delta, platform, timestamp, git_commit}`` -- Task 3's submitter
extracts this line for the sidecar env block
(``experiments/bench/gpu36_env.json``). ``scan_mode_givens``/
``compile_delta`` describe this campaign's FIXED per-arm backend contract
(always ``"triton"`` / ``True``), not which arms this particular invocation
happened to select via ``--arms``.

Import-path note: ``experiments/hetero_lab.py`` is not on ``sys.path`` by
default (``experiments/`` has no ``__init__.py``); ``_import_hetero_lab``
inserts it before importing, function-scoped (mirrors ``scripts/
gpu_delta_probe.py``'s ``_import_delta_mingru`` convention) so no
module-level import in this file needs a path-surgery exemption. Importing
``hetero_lab`` cascades its own ``sys.path`` inserts (repo root, then via
the root ``min_gru.py`` driver, ``src/``), so by the time
``_import_mixer_classes`` runs, ``from mingru import ...`` resolves
directly without redoing that bootstrap.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TORCH_STRATUM_PIN = "2.8."
_ROW_PREFIX = "MINGRU_LAB_ROW "
_ENV_PREFIX = "MINGRU_LAB_ENV "


@dataclass(frozen=True)
class _Arm:
    key: str
    round: str
    model: str
    compile: bool
    scan_mode: str | None  # value MINGRU_SCAN is forced to; None = unset


# Arm matrix, spec section 6 -- order matches the spec's listing
# (sg8, pd64, pd1024) and is this module's canonical default ``--arms`` order.
_ARMS: tuple[_Arm, ...] = (
    _Arm(
        key="sg8", round="hetero-gpu36-sg8", model="hetero-pg8", compile=False, scan_mode="triton"
    ),
    _Arm(key="pd64", round="hetero-gpu36-pd64", model="hetero-pd64", compile=True, scan_mode=None),
    _Arm(
        key="pd1024",
        round="hetero-gpu36-pd1024",
        model="hetero-pd1024",
        compile=True,
        scan_mode=None,
    ),
)
_ARM_BY_KEY: dict[str, _Arm] = {arm.key: arm for arm in _ARMS}

# Derived campaign-contract facts for _env_block's MINGRU_LAB_ENV payload,
# always read off the FULL arm matrix above -- never off a particular
# invocation's --arms subset -- so they can never independently drift from
# _ARMS the way two separately hardcoded literals could (design-review
# finding: "scan_mode_givens"/"compile_delta" used to be their own string/
# bool literals inside _env_block, duplicating facts already structural
# here). Values are unchanged from before this derivation ("triton" / True).
_GIVENS_SCAN_MODE: str | None = _ARM_BY_KEY["sg8"].scan_mode
_ALL_DELTA_ARMS_COMPILE: bool = all(_ARM_BY_KEY[k].compile for k in ("pd64", "pd1024"))

_DEFAULT_SEEDS: tuple[int, ...] = tuple(range(36))
_DEFAULT_STEPS = 1600


def _import_hetero_lab() -> Any:
    """Import ``experiments/hetero_lab.py`` by inserting ``experiments/``
    onto ``sys.path`` first (it is not a package). Function-scoped -- see
    the module docstring's "Import-path note".
    """
    experiments_dir = _REPO_ROOT / "experiments"
    if str(experiments_dir) not in sys.path:
        sys.path.insert(0, str(experiments_dir))
    import hetero_lab

    return hetero_lab


def _import_mixer_classes() -> tuple[type, type]:
    """Return ``(GivensMinGRU, DeltaMinGRU)`` from the packaged ``mingru``
    distribution, for the pre-flight backend-engagement probes.

    Must be called after ``_import_hetero_lab`` (see the module docstring's
    "Import-path note" -- that import's side effects put ``mingru`` on
    ``sys.path``); this function does not redo the bootstrap itself.
    """
    from mingru import DeltaMinGRU, GivensMinGRU

    return GivensMinGRU, DeltaMinGRU


@contextmanager
def _scan_env(mode: str | None) -> Iterator[None]:
    """Set ``os.environ["MINGRU_SCAN"]`` to ``mode`` (or unset it if
    ``mode`` is None) for the duration of the block, restoring whatever was
    there beforehand on exit -- including on an exception, so one arm's
    backend setting can never leak into the next arm's seeds even if a seed
    raises.
    """
    prior = os.environ.get("MINGRU_SCAN")
    if mode is None:
        os.environ.pop("MINGRU_SCAN", None)
    else:
        os.environ["MINGRU_SCAN"] = mode
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("MINGRU_SCAN", None)
        else:
            os.environ["MINGRU_SCAN"] = prior


def _assert_cuda_and_torch_version() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "pre-flight FAILED: --device cuda requested but CUDA is not "
            f"available (torch.cuda.is_available() is False, torch=="
            f"{torch.__version__}). Run this campaign inside the CUDA-enabled "
            "Lightning job (scripts/gpu_check.py --job hetero36); for local "
            "smoke testing without a GPU use --device cpu."
        )
    if not torch.__version__.startswith(_TORCH_STRATUM_PIN):
        raise SystemExit(
            "pre-flight FAILED: this round's CUDA stratum is pinned to torch "
            f"{_TORCH_STRATUM_PIN}x (spec section 4.4 stratum discipline), "
            f"found torch=={torch.__version__}. Use the job's pinned "
            "container image (pytorch/pytorch:2.8.0-...)."
        )


# GivensMinGRU kwargs for the sg8 arm's engagement probe, matching
# hetero_lab's pgivens8 factory exactly (experiments/hetero_lab.py
# LOCAL_FACTORIES: block_size=8, rounds=3 -- the same config the hetero-pg8
# model trains on). Named the same way as _PD64_DELTA_KWARGS/
# _PD1024_DELTA_KWARGS below so a drift test can introspect the real
# factory's constructed module the same way (see
# tests/test_gpu_hetero_campaign.py's
# test_givens_engagement_probe_config_matches_hetero_lab_factory).
_PGIVENS8_KWARGS: dict[str, int] = {"block_size": 8, "rounds": 3}


def _assert_givens_triton_engages(givens_cls: type) -> None:
    """Force ``MINGRU_SCAN=triton`` and run one tiny forward+backward
    through a ``hetero-pg8``-config (``_PGIVENS8_KWARGS``) ``GivensMinGRU``
    on CUDA.

    The packaged scan dispatch raises if no usable Triton path exists under
    ``MINGRU_SCAN=triton`` (``src/mingru/min_gru.py``'s ``_dispatch_scan``)
    -- that raise IS the gate; this function adds no assertion beyond
    letting it propagate, and restores ``MINGRU_SCAN`` afterward regardless
    of outcome.
    """
    with _scan_env("triton"):
        mixer = givens_cls(64, 64, **_PGIVENS8_KWARGS).to("cuda")
        x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
        mixer(x).sum().backward()


# DeltaMinGRU kwargs for the two delta arms' compile-smoke probes, matching
# hetero_lab's pdelta64/pdelta1024 factories exactly (experiments/
# hetero_lab.py LOCAL_FACTORIES). Kept as two separate constants/probe
# calls (not one "delta compile smoke" gate) because the pd1024 config's
# n_heads=4 exercises a materially different multi-head chunked-recurrence
# code path than pd64's n_heads=1 -- see the module docstring's pre-flight
# item 5.
_PD64_DELTA_KWARGS: dict[str, int] = {"nh": 2, "n_heads": 1, "d_k": 8, "d_v": 8}
_PD1024_DELTA_KWARGS: dict[str, int] = {"nh": 2, "n_heads": 4, "d_k": 16, "d_v": 16}


def _assert_delta_compile_succeeds(delta_cls: type, kwargs: dict[str, int]) -> None:
    """Tiny ``torch.compile`` forward+backward through a ``DeltaMinGRU``
    built with ``kwargs`` (one of ``_PD64_DELTA_KWARGS``/
    ``_PD1024_DELTA_KWARGS``) on CUDA, ``MINGRU_SCAN`` unset (delta arms
    never set it). A compile or first-call failure raises uncaught --
    fail-loud, no silent eager fallback (spec section 7).
    """
    with _scan_env(None):
        layer = delta_cls(64, 64, **kwargs).to("cuda")
        compiled = torch.compile(layer)
        x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
        compiled(x).sum().backward()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # best-effort provenance field, never fatal
        return None


def _env_block() -> dict[str, Any]:
    try:
        import triton

        triton_version: str | None = triton.__version__
    except ImportError:
        triton_version = None
    return {
        "torch": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "triton": triton_version,
        # Fixed campaign backend contract (spec section 6), derived from the
        # full _ARMS matrix (module-level constants above), not from "what
        # this invocation's --arms subset happened to run" -- see module
        # docstring.
        "scan_mode_givens": _GIVENS_SCAN_MODE,
        "compile_delta": _ALL_DELTA_ARMS_COMPILE,
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }


def run_preflight(device: str, arms: list[_Arm]) -> dict[str, Any] | None:
    """Run the fail-loud pre-flight gates and return the env block (``None``
    on ``--device cpu``, where every gate is skipped -- see the module
    docstring). Raises ``SystemExit`` with a clear message on any failure.
    """
    if device != "cuda":
        print("pre-flight: --device cpu, skipping CUDA/Triton/compile gates", flush=True)
        return None

    print("pre-flight: CUDA + torch version...", flush=True)
    _assert_cuda_and_torch_version()
    print(f"pre-flight: CUDA + torch=={torch.__version__} ok", flush=True)

    givens_cls, delta_cls = _import_mixer_classes()
    selected_keys = {arm.key for arm in arms}

    if "sg8" in selected_keys:
        print("pre-flight: givens-triton engagement probe...", flush=True)
        _assert_givens_triton_engages(givens_cls)
        print("pre-flight: givens-triton engagement ok", flush=True)

    if "pd64" in selected_keys:
        print("pre-flight: delta compile smoke (pd64)...", flush=True)
        _assert_delta_compile_succeeds(delta_cls, _PD64_DELTA_KWARGS)
        print("pre-flight: delta compile smoke (pd64) ok", flush=True)

    if "pd1024" in selected_keys:
        print("pre-flight: delta compile smoke (pd1024)...", flush=True)
        _assert_delta_compile_succeeds(delta_cls, _PD1024_DELTA_KWARGS)
        print("pre-flight: delta compile smoke (pd1024) ok", flush=True)

    env = _env_block()
    print(_ENV_PREFIX + json.dumps(env), flush=True)
    return env


def _lab_args(arm: _Arm, seed: int, steps: int, device: str, ckpt_t: int) -> argparse.Namespace:
    """Build the ``argparse.Namespace`` ``hetero_lab.run_arm`` expects for
    one seed of one arm -- every intervention flag left at ``main()``'s
    documented default (this campaign runs the plain hetero stacks, no
    training-side interventions), ``dry_run=True`` so the clone's ledger is
    never written (rows are transported via stdout only, see the module
    docstring's "Row transport" section).
    """
    return argparse.Namespace(
        round=arm.round,
        task="S3-hier",
        model=arm.model,
        seed=seed,
        steps=steps,
        soft_warmup=0,
        soft_blend=0,
        commit_lambda=0.0,
        commit_ramp=800,
        grad_noise=0.0,
        identity_warmup=0,
        curriculum="",
        ckpt_t=ckpt_t,
        dry_run=True,
        require_torch=None,
        device=device,
        compile=arm.compile,
    )


def run_campaign(
    arms: list[_Arm], seeds: list[int], steps: int, device: str
) -> dict[str, Any] | None:
    """Run the full campaign loop: pre-flight, then each arm's seeds
    in-process against ``hetero_lab.run_arm``, printing one
    ``MINGRU_LAB_ROW`` line plus a progress line per completed seed.
    Returns the pre-flight env block (``None`` on ``--device cpu``).
    """
    hetero_lab = _import_hetero_lab()
    env = run_preflight(device, arms)

    for arm in arms:
        with _scan_env(arm.scan_mode):
            for seed in seeds:
                t0 = time.perf_counter()
                ns = _lab_args(arm, seed, steps, device, hetero_lab.CKPT_T)
                # run_arm prints the raw row JSON itself before returning it
                # (unmarked); discard that here so only the MINGRU_LAB_ROW
                # line below (and the progress line after it) reach the job
                # log -- see the module docstring's "Row transport" section.
                # NOTE: if run_arm ever grows a mid-training diagnostic
                # print, this redirect would swallow that too -- fine today
                # (its only print is the final row line), but worth
                # revisiting if that changes.
                with redirect_stdout(io.StringIO()):
                    row = hetero_lab.run_arm(ns)
                wall = time.perf_counter() - t0
                print(_ROW_PREFIX + json.dumps(row), flush=True)
                val_key = f"val{ns.ckpt_t}"
                print(
                    f"[{arm.round}] seed={seed} done in {wall:.1f}s "
                    f"(selected step={row['steps']}, ckpt {val_key}="
                    f"{row['ckpt'].get(val_key)})",
                    flush=True,
                )
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=sorted(_ARM_BY_KEY),
        default=[arm.key for arm in _ARMS],
        help="arm subset to run (default: all three, spec order sg8/pd64/pd1024)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="seed subset to run (default: 0-35, the full round)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=_DEFAULT_STEPS,
        help=f"training steps per seed (default {_DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="cuda (default, the production path) or cpu (local smoke only -- "
        "skips every pre-flight gate, see module docstring)",
    )
    args = parser.parse_args(argv)

    arms = [_ARM_BY_KEY[key] for key in args.arms]
    seeds = list(_DEFAULT_SEEDS) if args.seeds is None else args.seeds

    run_campaign(arms, seeds, args.steps, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
