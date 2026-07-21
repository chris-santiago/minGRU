"""In-job campaign loop for the accepted-benchmark validation round (Task 5).

Runs INSIDE a Lightning GPU job (submitted by a benchmarks job mode on
``scripts/gpu_check.py``, a later task), never invoked directly by a
developer machine as the evidence source of truth -- the ``--device cpu`` /
``--tasks`` / ``--arms`` / ``--seeds`` / ``--steps`` flags below exist for
local smoke testing only. See ``.claude/output/specs/2026-07-19-benchmark-
round-design.md`` sections 4, 6, 7 and ``.git/sdd/task-5-brief.md`` for the
full design.

Task x arm matrix (spec section 1/6)
-------------------------------------
Four tasks (``s5``, ``mqar``, ``psmnist``, ``pendulum``), each run by
default against ``experiments/benchmark_lab.py``'s ``MATRIX_ARMS`` (spec
section 6 "Arms" fixed ``log``/``signed``/``rotation``/``givens``/
``delta``; ``signed-rotation`` was added by amendment, six total; a second
amendment then added ``signed-givens``/``signed-delta``, eight total --
every matrix arm runs on all four tasks, no per-task arm subset). Unlike
the GPU 36-seed round's ``gpu_hetero_campaign.py``,
no arm here forces a distinct backend (no per-arm ``MINGRU_SCAN``/
``torch.compile`` split) -- every arm of a task runs under the same
``--device`` this invocation was given; the round's Global Constraints fix
the training budget per task, not a per-arm backend.

A third amendment (2026-07-20) added ``PROBE_ARMS`` (``signed-rotation-k5``,
``signed-delta-nh3``, ``signed-delta-nh4``) -- an S5-only follow-up probe,
selectable via ``ARM_REGISTRY`` (``MATRIX_ARMS`` union ``PROBE_ARMS``) but
NOT in the default ``--arms`` list: probe arms run only when named
explicitly (e.g. ``--arms signed-rotation-k5 signed-delta-nh3
signed-delta-nh4``).

A fourth amendment (2026-07-20, "standard GRU control arm") added ``gru``
to ``MATRIX_ARMS``, nine total -- a depth-matched (2-layer, d_model=64)
classical ``nn.GRU`` external control, run on all four tasks like every
other matrix arm. Unlike ``PROBE_ARMS``, this IS default-``--arms`` matrix
expansion: no special-casing was needed here, since this module's
``--arms`` default and choices both already read ``experiments.benchmark_lab
.MATRIX_ARMS``/``.ARM_REGISTRY`` directly rather than hardcoding an arm
list.

A fifth amendment (2026-07-20, "gru-large grounding reference") added
``REF_ARMS`` (``gru-large``, a literature-scale hidden-256 2-layer
``nn.GRU``) -- an explicitly NON-matched reference, selectable via
``ARM_REGISTRY`` (``MATRIX_ARMS`` union ``PROBE_ARMS`` union ``REF_ARMS``)
but NOT in the default ``--arms`` list, same convention as ``PROBE_ARMS``:
a ref arm runs only when named explicitly (e.g. ``--arms gru-large``) and
writes under its own distinct ledger round tag
(``experiments.benchmark_tasks.BENCH_REF_ROUND_TAGS``), resolved by
``_round_tag_for_arm`` alongside probe-arm routing.

Round tags: ``bench-s5-02``, ``bench-mqar-02``, ``bench-psmnist-02``,
``bench-pendulum-02`` -- one per task, independent of which MATRIX arms
that task's cell selects, read from ``experiments.benchmark_tasks
.BENCH_ROUND_TAGS`` (the single source of truth this module and
``scripts/report_benchmarks.py`` both bind to). Bumped from the spec's
original ``-01`` tags at the pre-matrix technical review (2026-07-19): the
``-01`` tags are now the recorded pilot/calibration population
(heterogeneous per-seed training budgets), so the seed matrices land under
clean ``-02`` tags that can't dedup-collide with pilot rows or get pooled
into the same statistics by the report -- see ``BENCH_ROUND_TAGS``'s own
comment for the full rationale. A probe arm's rows land under a distinct
tag instead, ``bench-s5-probe-01`` (``experiments.benchmark_tasks
.BENCH_PROBE_ROUND_TAGS``); a ref arm's rows land under yet another
distinct tag, ``bench-psmnist-ref-01`` (``experiments.benchmark_tasks
.BENCH_REF_ROUND_TAGS``) -- ``_round_tag_for_arm`` resolves which tag each
``(task, arm)`` cell writes under, per arm, inside ``run_campaign``'s loop,
so neither a probe nor a ref arm can ever dedup-collide with or get pooled
into the matrix population's statistics.

Pre-flight (fail-loud, before seed 0, ``--device cuda`` only)
----------------------------------------------------------------
1. ``torch.cuda.is_available()`` -- clear ``SystemExit`` if not.
2. ``torch.__version__`` starts with ``2.8.`` (this round's pinned L4
   stratum, matching ``gpu_hetero_campaign.py``'s pin) -- clear
   ``SystemExit`` if not.
3. ``experiments/benchmark_lab.py --selftest`` run as a subprocess (tiny
   model, few steps, all four tasks, dry-run only -- a wiring smoke test).
   A non-zero exit raises ``SystemExit`` with the subprocess's stderr
   attached; this blocks a broken lab from burning a seed matrix (spec
   section 10). Run once, unconditionally, regardless of the invocation's
   ``--tasks``/``--arms`` subset -- it is a single wiring gate over the
   whole lab, not a per-arm engagement probe like the hetero round's
   compile/triton gates. This gate always runs its tiny models on CPU (a
   backend-independent sanity check) with ``MINGRU_SCAN`` explicitly
   unset in the subprocess's environment, regardless of what the parent
   job process has it set to (see ``_run_selftest_gate``'s docstring) --
   the production job exports ``MINGRU_SCAN=triton`` job-wide for the
   real evidence runs, and that export must not reach this CPU-only gate.

On ``--device cpu`` (smoke only) every pre-flight gate above is skipped --
mirrors ``gpu_hetero_campaign.py``'s convention -- and no ``MINGRU_LAB_ENV``
line is printed.

Row transport (spec section 6)
--------------------------------
Every seed is run via ``experiments.benchmark_lab.run_arm`` with
``dry_run=True`` (the lab's no-append path -- ``experiments/
lab_results.jsonl`` is never written from inside the job; the local finish
handler, a later task, appends rows post-dedup after the job completes).
Per completed seed this prints exactly one ``MINGRU_LAB_ROW <row json>``
line plus a human-readable progress line -- the job's container filesystem
dies with the job, so these stdout lines are the sole evidence transport.
``run_arm`` also unconditionally prints the raw row JSON itself before
returning it (unmarked, no prefix, mirroring ``hetero_lab.run_arm``'s same
behavior); that near-duplicate line is discarded via
``contextlib.redirect_stdout`` into a throwaway buffer -- only this
module's own ``MINGRU_LAB_ROW``-prefixed line and progress line (both
printed OUTSIDE the redirected block) reach the job log.

Cross-task env contract
--------------------------
After pre-flight passes on ``--device cuda``, exactly one
``MINGRU_LAB_ENV <json>`` line is printed with ``{torch, cuda_device_name,
device_capability, triton, platform, timestamp, git_commit}`` for a later
job runner's sidecar env block. No ``scan_mode_*``/``compile_*`` fields
(unlike ``gpu_hetero_campaign.py``'s env block) -- this round fixes no
per-arm backend contract to describe.

Import-path note: ``experiments/benchmark_lab.py`` imports itself via
dotted ``experiments.benchmark_tasks``/``experiments.benchmark_lab`` names
(unlike ``experiments/hetero_lab.py``, imported as a bare top-level module
by ``gpu_hetero_campaign.py``), so ``_import_benchmark_lab`` here inserts
the REPO ROOT onto ``sys.path`` (not ``experiments/`` itself) and imports
``experiments.benchmark_lab`` as a dotted module -- the same convention
``tests/test_benchmark_lab.py`` already uses.
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
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))  # `experiments.*` (namespace package, no __init__.py)
from experiments.benchmark_tasks import (  # noqa: E402
    BENCH_ARM_ROUND_OVERRIDES,
    BENCH_PROBE_ROUND_TAGS,
    BENCH_REF_ROUND_TAGS,
    BENCH_ROUND_TAGS,
)

_BENCHMARK_LAB_PATH = _REPO_ROOT / "experiments" / "benchmark_lab.py"
_TORCH_STRATUM_PIN = "2.8."
_ROW_PREFIX = "MINGRU_LAB_ROW "
_ENV_PREFIX = "MINGRU_LAB_ENV "

# Round tags: this module owns `_ROUND_TAGS` as its own name (existing
# convention -- `report_benchmarks.py` binds its own `ROUND_TAGS` name to
# the same mapping), but both read the single `BENCH_ROUND_TAGS` source of
# truth in `experiments/benchmark_tasks.py` rather than each hardcoding an
# independently-editable copy -- see that mapping's own comment for the
# `-01` -> `-02` bump rationale.
_ROUND_TAGS: dict[str, str] = BENCH_ROUND_TAGS

# Probe round tags (S5-only follow-up probe, `experiments.benchmark_lab
# .PROBE_ARMS`): a distinct source-of-truth mapping, same binding
# convention as `_ROUND_TAGS` above, so a probe arm's rows never land
# under the matrix `-02` tag -- see `_round_tag_for_arm` below and
# `BENCH_PROBE_ROUND_TAGS`'s own comment.
_PROBE_ROUND_TAGS: dict[str, str] = BENCH_PROBE_ROUND_TAGS

# Reference round tags (gru-large grounding reference, `experiments
# .benchmark_lab.REF_ARMS`): same distinct source-of-truth-mapping
# convention as `_PROBE_ROUND_TAGS` above, so a ref arm's rows never land
# under the matrix `-02` tag either -- see `_round_tag_for_arm` below and
# `BENCH_REF_ROUND_TAGS`'s own comment.
_REF_ROUND_TAGS: dict[str, str] = BENCH_REF_ROUND_TAGS

# Per-arm round-tag correction overrides (design spec, `.claude/output/
# specs/2026-07-21-round-tag-override-design.md`): the write-side half of
# the paired override rule -- `scripts/report_benchmarks.py`'s
# `_source_round_for_arm` is the read-side half, both resolving the same
# `arm -> {task -> correction round tag}` map so the two paths cannot
# diverge (spec section 6). Same binding convention as `_ROUND_TAGS`/
# `_PROBE_ROUND_TAGS`/`_REF_ROUND_TAGS` above: this module owns its own
# name, bound directly to the single source of truth in
# `experiments/benchmark_tasks.py`, never a hardcoded copy.
_ARM_ROUND_OVERRIDES: dict[str, dict[str, str]] = BENCH_ARM_ROUND_OVERRIDES


def _round_tag_for_arm(task_name: str, arm: str, probe_arms: dict, ref_arms: dict) -> str:
    """Ledger ``round`` tag for one ``(task_name, arm)`` cell.

    The override map (``_ARM_ROUND_OVERRIDES``) is consulted first: if
    ``arm`` has a correction tag registered for ``task_name``, that tag
    wins outright, ahead of every other branch below (spec section 6,
    "write-side resolver contract"). Otherwise: the probe tag
    (``_PROBE_ROUND_TAGS``) when ``arm`` is one of ``probe_arms``
    (``experiments.benchmark_lab.PROBE_ARMS``); the reference tag
    (``_REF_ROUND_TAGS``) when ``arm`` is one of ``ref_arms``
    (``experiments.benchmark_lab.REF_ARMS``); else the task's matrix tag
    (``_ROUND_TAGS``) -- the same per-task tag every matrix arm has always
    used. ``probe_arms``/``ref_arms`` are disjoint (a matrix arm is never
    in either), so checking probe membership first is not a priority
    choice, just an arbitrary but stable order. A probe/ref arm requested
    against a task with no entry in its own round-tag mapping AND no
    override entry (every task except S5 for probes; every task except
    psMNIST for refs, per this round's scope) fails loud rather than
    silently falling back to the matrix tag, which would pollute the clean
    `-02` matrix population with a differently-configured arm's rows."""
    override_tag = _ARM_ROUND_OVERRIDES.get(arm, {}).get(task_name)
    if override_tag is not None:
        return override_tag
    if arm in probe_arms:
        if task_name not in _PROBE_ROUND_TAGS:
            raise ValueError(
                f"probe arm {arm!r} has no round tag for task {task_name!r} -- "
                f"BENCH_PROBE_ROUND_TAGS only covers {sorted(_PROBE_ROUND_TAGS)}; "
                "this probe round does not run that task"
            )
        return _PROBE_ROUND_TAGS[task_name]
    if arm in ref_arms:
        if task_name not in _REF_ROUND_TAGS:
            raise ValueError(
                f"ref arm {arm!r} has no round tag for task {task_name!r} -- "
                f"BENCH_REF_ROUND_TAGS only covers {sorted(_REF_ROUND_TAGS)}; "
                "this reference arm does not run that task"
            )
        return _REF_ROUND_TAGS[task_name]
    return _ROUND_TAGS[task_name]


def _import_benchmark_lab() -> Any:
    """Import ``experiments.benchmark_lab`` by inserting the repo root onto
    ``sys.path`` (see the module docstring's "Import-path note").
    Function-scoped, mirroring ``gpu_hetero_campaign.py``'s
    ``_import_hetero_lab`` convention.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import experiments.benchmark_lab as benchmark_lab

    return benchmark_lab


def _resolve_seeds(task: Any, seeds: list[int] | None) -> list[int]:
    """Seed subset for one task: ``seeds`` override if given (applies
    uniformly across every selected task, local-smoke convention), else the
    task's own seed-matrix size (spec section 2: 36 seeds for s5/mqar/
    pendulum, 12 for psmnist) -- ``TaskSpec.seeds``.
    """
    return list(range(task.seeds)) if seeds is None else list(seeds)


def _resolve_eval_every(task: Any, steps: int | None) -> int | None:
    """Checkpoint-selection cadence to pass alongside a ``--steps``
    override: ``None`` (the task's own cadence, untouched) unless ``steps``
    overrides ``TaskSpec.budget.steps`` for a step-based task, in which case
    the cadence shrinks to ``min(steps, task.budget.eval_every)``.

    ``--steps`` is a local-smoke-only override (production runs the full
    frozen budget with no override, spec section 7 "no per-arm tuning");
    without this shrink, a small ``--steps`` smoke run's step range would
    never reach the full-scale ``eval_every`` (e.g. ``steps=20`` against the
    default ``eval_every=100``), so no checkpoint would ever be selected and
    ``run_arm``'s ``_require_checkpoint`` guard would raise. Shrinking to
    the minimum guarantees a checkpoint is selected at (or before) the last
    step for any ``steps >= 1``. ``None`` for epoch-based tasks (``psmnist``,
    whose ``eval_every`` is already ``None``) -- unaffected by this shrink.
    """
    if steps is None or task.budget.eval_every is None:
        return None
    return min(steps, task.budget.eval_every)


def _run_arm_kwargs(
    round_tag: str, task: Any, arm: str, seed: int, steps: int | None, device: str
) -> dict[str, Any]:
    """The keyword arguments passed to ``benchmark_lab.run_arm`` for one
    (task, arm, seed) cell -- ``dry_run`` is always ``True`` (jobs never
    write the ledger, spec section 7); ``steps`` overrides
    ``TaskSpec.budget.steps`` only for step-based tasks and is silently
    unused by ``run_arm`` for psmnist (epoch-based -- ``run_arm`` never
    reads its ``steps`` parameter for a ``last_step`` task). ``eval_every``
    is shrunk alongside a ``--steps`` override (see
    ``_resolve_eval_every``).
    """
    return dict(
        round_tag=round_tag,
        task=task,
        arm=arm,
        seed=seed,
        steps=steps,
        eval_every=_resolve_eval_every(task, steps),
        device=device,
        dry_run=True,
    )


def _assert_cuda_and_torch_version() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "pre-flight FAILED: --device cuda requested but CUDA is not "
            f"available (torch.cuda.is_available() is False, torch=="
            f"{torch.__version__}). Run this campaign inside the CUDA-enabled "
            "Lightning job; for local smoke testing without a GPU use "
            "--device cpu."
        )
    if not torch.__version__.startswith(_TORCH_STRATUM_PIN):
        raise SystemExit(
            "pre-flight FAILED: this round's L4 stratum is pinned to torch "
            f"{_TORCH_STRATUM_PIN}x (spec section 4/CLAUDE.md stratum "
            f"discipline), found torch=={torch.__version__}. Use the job's "
            "pinned container image."
        )


def _run_selftest_gate() -> None:
    """Run ``experiments/benchmark_lab.py --selftest`` as a subprocess
    (module docstring pre-flight item 3): a non-zero exit raises
    ``SystemExit`` with the subprocess's stderr attached, blocking a broken
    lab from burning a seed matrix (spec section 10). Subprocessing (rather
    than importing and calling the lab's private ``_selftest`` function
    directly) exercises the exact documented CLI entry point instead of
    reaching across a module-private boundary.

    A hung selftest (``subprocess.TimeoutExpired``) or a broken interpreter
    invocation (``OSError``) is caught and re-raised as the same clean
    ``SystemExit`` shape every other pre-flight gate in this module uses,
    rather than letting a raw subprocess exception traceback escape.

    The selftest is a backend-independent CPU sanity gate -- it builds and
    trains tiny models on ``device="cpu"`` regardless of what device this
    campaign is targeting. The production job exports ``MINGRU_SCAN=triton``
    job-wide (so the campaign's real evidence runs dispatch to the Triton
    kernel), but that export must not reach this subprocess: the Triton
    dispatch path fails loud on a CPU tensor (`"Pointer argument ... cannot
    be accessed from Triton (cpu tensor?)"`), which would trip this gate on
    a perfectly healthy lab. The subprocess therefore runs with a copy of
    the parent environment with ``MINGRU_SCAN`` removed (not set to a
    value -- unsetting it is the neutral state, letting the packaged
    dispatch fall back to its own ``auto`` default, which never engages
    Triton on CPU per the ``MINGRU_SCAN`` contract). The campaign's own
    evidence-bearing seed runs (``run_campaign``'s calls to
    ``benchmark_lab.run_arm``) are unaffected -- they run in-process, not
    through this subprocess, and correctly inherit the job's
    ``MINGRU_SCAN=triton`` export untouched.
    """
    selftest_env = os.environ.copy()
    selftest_env.pop("MINGRU_SCAN", None)
    try:
        result = subprocess.run(
            [sys.executable, str(_BENCHMARK_LAB_PATH), "--selftest"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env=selftest_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            "pre-flight FAILED: experiments/benchmark_lab.py --selftest timed out "
            f"after {exc.timeout}s"
        ) from exc
    except OSError as exc:
        raise SystemExit(
            f"pre-flight FAILED: experiments/benchmark_lab.py --selftest could not "
            f"be launched: {exc}"
        ) from exc

    print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        raise SystemExit(
            "pre-flight FAILED: experiments/benchmark_lab.py --selftest exited "
            f"{result.returncode}; stderr:\n{result.stderr}"
        )


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
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }


def run_preflight(device: str) -> dict[str, Any] | None:
    """Run the fail-loud pre-flight gates and return the env block (``None``
    on ``--device cpu``, where every gate is skipped -- see the module
    docstring). Raises ``SystemExit`` with a clear message on any failure.
    """
    if device != "cuda":
        print("pre-flight: --device cpu, skipping CUDA/selftest gates", flush=True)
        return None

    print("pre-flight: CUDA + torch version...", flush=True)
    _assert_cuda_and_torch_version()
    print(f"pre-flight: CUDA + torch=={torch.__version__} ok", flush=True)

    print("pre-flight: benchmark_lab selftest...", flush=True)
    _run_selftest_gate()
    print("pre-flight: benchmark_lab selftest ok", flush=True)

    env = _env_block()
    print(_ENV_PREFIX + json.dumps(env), flush=True)
    return env


def run_campaign(
    tasks: list[str], arms: list[str], seeds: list[int] | None, steps: int | None, device: str
) -> dict[str, Any] | None:
    """Run the full campaign loop: pre-flight, then each task's arms x
    seeds in-process against ``benchmark_lab.run_arm``, printing one
    ``MINGRU_LAB_ROW`` line plus a progress line per completed seed.
    Returns the pre-flight env block (``None`` on ``--device cpu``).
    """
    benchmark_lab = _import_benchmark_lab()
    env = run_preflight(device)

    for task_name in tasks:
        task = benchmark_lab.TASKS[task_name]
        task_seeds = _resolve_seeds(task, seeds)
        for arm in arms:
            round_tag = _round_tag_for_arm(
                task_name, arm, benchmark_lab.PROBE_ARMS, benchmark_lab.REF_ARMS
            )
            for seed in task_seeds:
                t0 = time.perf_counter()
                kwargs = _run_arm_kwargs(round_tag, task, arm, seed, steps, device)
                # run_arm prints the raw row JSON itself before returning it
                # (unmarked); discard that here so only the MINGRU_LAB_ROW
                # line below (and the progress line after it) reach the job
                # log -- see the module docstring's "Row transport" section.
                with redirect_stdout(io.StringIO()):
                    row = benchmark_lab.run_arm(**kwargs)
                wall = time.perf_counter() - t0
                print(_ROW_PREFIX + json.dumps(row), flush=True)
                print(
                    f"[{round_tag}] arm={arm} seed={seed} done in {wall:.1f}s "
                    f"(selected {row['ckpt']})",
                    flush=True,
                )
    return env


def main(argv: list[str] | None = None) -> int:
    benchmark_lab = _import_benchmark_lab()

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(benchmark_lab.TASKS),
        default=list(benchmark_lab.TASKS),
        help="task subset (default: all four, spec order s5/mqar/psmnist/pendulum)",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=sorted(benchmark_lab.ARM_REGISTRY),
        default=list(benchmark_lab.MATRIX_ARMS),
        help="arm subset (default: the nine MATRIX_ARMS -- "
        "log/signed/rotation/signed-rotation/givens/delta/signed-givens/signed-delta/gru; "
        "the three S5-only PROBE_ARMS -- signed-rotation-k5/signed-delta-nh3/"
        "signed-delta-nh4 -- and the psMNIST-only REF_ARMS grounding reference "
        "-- gru-large -- are choosable but never run unless named explicitly, "
        "and write under their own bench-s5-probe-01 / bench-psmnist-ref-01 "
        "round tags, not the matrix tag)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="seed subset applied to every selected task (default: each task's own "
        "seed-matrix size, spec section 2: 36 for s5/mqar/pendulum, 12 for psmnist)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override each step-based task's Budget.steps (ignored by psmnist, "
        "which is epoch-based)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="cuda (default, the production path) or cpu (local smoke only -- "
        "skips every pre-flight gate, see module docstring)",
    )
    args = parser.parse_args(argv)

    run_campaign(args.tasks, args.arms, args.seeds, args.steps, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
