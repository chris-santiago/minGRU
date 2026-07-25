"""Run the GPU conformance suite on a Lightning AI batch job.

Submits an ephemeral containerized job (lightning_sdk ``Job``) that clones this
repository at a given commit, verifies the CUDA/Triton toolchain, and runs the
full parity sweep (``scripts/bench_scans.py --check`` -- the 590-row
CPU-vs-GPU conformance matrix). The job's exit code is this script's exit
code, so ``nox -s gpu`` is a one-command GPU gate. The container dies with
the job: no idle studio, no auto-sleep races, no keepalives.

Authentication (create under lightning.ai Settings -> Keys, export before
running):
  LIGHTNING_USER_ID, LIGHTNING_API_KEY

Required configuration:
  MINGRU_LIGHTNING_TEAMSPACE   e.g. "owner/teamspace" (or pass --teamspace)

Optional overrides:
  --machine   Lightning machine name (default L4 -- the tier the committed
              conformance artifacts were produced on)
  --image     container image (default: the pytorch/pytorch devel image
              matching the library's torch floor; -devel ships the compiler
              toolchain and Python headers Triton's C launcher needs)
  --ref       git commit/branch to test (default: current HEAD SHA)
  --repo      repository URL (default: this repo's origin)
  --bench     also run --bench after --check (longer, more credits)
  --dry-run   print the fully-resolved job spec and exit without submitting

Cost note: L4 jobs consume paid credits; the free tier covers T4 only
(``--machine T4`` works for a smoke run, but the committed evidence envelope
was validated on L4). ``--interruptible`` uses spot pricing.

Job modes (``--job``, default ``check``, existing invocations unaffected):

``check`` (default)
  The conformance sweep described above -- unchanged.

``delta-probe`` (Task 6, extended Task 5)
  Runs ``scripts/gpu_delta_probe.py`` instead: the CUDA fusion-headroom
  probe for ``DeltaMinGRU``'s chunked-WY forward (profiles eager vs. its
  matmul-FLOP floor vs. ``torch.compile`` vs. the Triton chunked-WY kernel
  under ``MINGRU_SCAN=triton`` -- the round's central speed-bar evidence,
  spec section 9.1 acceptance criterion 1 -- see that script's module
  docstring and ``.git/sdd/task-6-brief.md``). No keepalive heartbeat:
  the Lightning idle auto-shutdown applies to interactive studios, not
  jobs, and a backgrounded loop outlives a studio-mode job's command
  chain (see ``build_delta_probe_command``). After the job completes,
  this script fetches its logs, extracts the last ``MINGRU_GPU_PROBE_RESULT``
  line, and writes ``experiments/bench/gpu_delta_probe.json``/``.md``
  locally -- the job's container filesystem dies with the job, so stdout
  is the only transport. A missing/malformed result line is a clear error
  exit; no partial artifact is ever written.

``hetero36`` (GPU 36-seed round, Task 3)
  Runs ``scripts/gpu_hetero_campaign.py`` instead: the 3-arm x 36-seed
  S3-hier round (packaged givens on the Triton scan path vs. the two
  packaged delta configs under ``torch.compile`` -- see
  ``.claude/output/specs/2026-07-18-gpu36-round-design.md`` sections 4
  and 6). Same clone/checkout/triton-install steps and foreground-only,
  no-keepalive command chain as ``delta-probe`` (see
  ``build_delta_probe_command``'s docstring for why). After the job
  completes, this script fetches its logs, extracts EVERY
  ``MINGRU_LAB_ROW`` line (one per completed seed, guarded parse --
  malformed lines skipped, never crash). A row that parses as JSON but is
  the wrong SHAPE (not an object, an unrecognized ``round``, or a
  non-``int``/``bool`` ``seed``) is shape-invalid: it's warned about on
  stderr and counted (``rows_skipped_invalid`` in the sidecar), never
  appended and never a crash. Within the surviving batch, a duplicate
  ``(round, seed)`` resolves last-N-wins (spec §6), counted separately as
  ``rows_deduped_in_batch``. The remaining rows are dedup-checked by
  ``(round, seed)`` against the local ``experiments/lab_results.jsonl``
  for the three hetero36 round names, and only the new rows are appended
  (preserving log order -- a retried job's already-appended rows dedup
  out, making retries idempotent). Writes
  ``experiments/bench/gpu36_env.json`` (the job's single
  ``MINGRU_LAB_ENV`` line plus extraction/dedup counts and per-seed wall
  seconds, reconciling ``rows_extracted == rows_appended +
  rows_skipped_duplicate + rows_skipped_invalid + rows_deduped_in_batch``).
  Rows or the env line missing/malformed entirely is a clear error exit;
  nothing is appended or written in that case.

``benchmarks`` (accepted-benchmark validation round, Task 6, consumes Task 5)
  Runs ``scripts/gpu_benchmark_campaign.py`` instead: the multi-task x
  multi-arm benchmark validation round (S5, MQAR, psMNIST, pendulum -- see
  ``.claude/output/specs/2026-07-19-benchmark-round-design.md`` sections 4
  and 6, plus the amended sixth ``signed-rotation`` arm recorded in the
  intent ledger's Amendments -- and churn, the real-world event-sequence
  task added by ``.claude/output/specs/2026-07-25-churn-benchmark-design.md``).
  Same clone/checkout/triton-install preamble as ``hetero36``, plus a
  ``torchvision`` install (psMNIST downloads MNIST inside the job), a
  ``pandas`` install when the job's tasks include churn (its loader
  downloads and parses the Rosbank churn csv.gz files inside the job -- see
  ``build_benchmarks_command``), and an ``export MINGRU_SCAN=triton`` ahead
  of the campaign invocation (evidence-phase-gate amendment, "Triton
  everywhere" -- every seed matrix in this round runs under a uniform
  Triton scan backend on L4; the ``MINGRU_SCAN`` contract itself is
  unchanged, so a task/arm combination with no Triton path fails loud
  rather than downgrading silently). ``--tasks``/``--arms``/``--seeds``
  pass through to the campaign script's own CLI when given (default: its
  own defaults, every task/arm/seed); the production command NEVER passes
  ``--steps`` -- that flag is a local-smoke-only override that shrinks the
  campaign's eval cadence, not a production budget knob (spec section 7:
  "no per-arm tuning"; the committed ``TaskSpec`` budgets are frozen).
  Foreground-only, no-keepalive command chain -- see
  ``build_delta_probe_command``'s docstring for why.

  ``--shards N`` (default 1, Task 6 of the churn round) submits ``N``
  benchmarks jobs up front instead of one -- each running every requested
  arm against a disjoint, exhaustive slice of the seed pool (the task's own
  seed count, or ``--seeds`` if given) -- then waits for and collects each
  sequentially through this exact same finish-handler path (see
  ``_run_sharded_benchmarks``/``_resolve_shard_seed_pool``/
  ``_shard_seed_lists``). Submission is concurrent; waiting is sequential
  (acceptable per the round's constraints); every shard's command chain is
  still its own single foreground, no-keepalive chain -- ``--shards``
  changes how many job commands run, never whether any of them background
  a step. For the churn round's 36-seed x 9-arm matrix, ``--shards 6``
  gives job ``k`` seeds ``6k..6k+5``, all nine arms. Each task's per-shard
  ``bench_<task>_env.json`` sidecar is merged onto the prior shard's
  already-written sidecar rather than overwritten (see
  ``_merge_benchmarks_sidecar``), so the file left after the last shard
  discloses the whole round's timings, not just the last shard's.

  After the job completes, this script fetches its logs, extracts every
  ``MINGRU_LAB_ROW`` line (guarded parse, malformed lines skipped -- see
  ``_extract_all``) and the single ``MINGRU_LAB_ENV`` line (guarded,
  last-wins -- see ``_extract_last``). Either rows entirely absent or the
  env line missing/malformed is a clear error exit with nothing appended or
  written, mirroring ``hetero36``'s contract. Unlike ``hetero36``, this
  round's dedup key is the four-field ``(round, task, variant, seed)``
  (spec section 6's marker-line protocol) -- a task's arms and 36 (or
  12, psMNIST) seeds all share one round tag, so ``variant`` (the arm) is
  load-bearing in the key. Rows are bucketed by their own ``task`` field
  (dict rows only, non-empty string ``task`` only -- a row that isn't a
  dict, or is a dict with a missing/empty ``task``, can't be attributed to
  any task and is dropped with a warning, never appended anywhere) and each
  task's bucket is dedup-appended to the
  ledger and written to its own ``experiments/bench/bench_<task>_env.json``
  sidecar (one per task present in the parsed rows), mirroring
  ``gpu36_env.json``'s shape except ``per_seed_wall_secs`` is keyed by
  ``{variant: {seed: secs}}`` rather than ``{round: {seed: secs}}`` -- a
  task's rows span many arms at the same seed, so seed alone would collide
  across arms the way it never could in ``hetero36`` (there, one round
  already implied exactly one arm).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))  # `experiments.*` (namespace package, no __init__.py)
from experiments.benchmark_tasks import (  # noqa: E402
    BENCH_ARM_ROUND_OVERRIDES,
    BENCH_PROBE_ROUND_TAGS,
    BENCH_REF_ROUND_TAGS,
    BENCH_ROUND_TAGS,
    TASKS,
)

DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
DEFAULT_MACHINE = "L4"
JOB_NAME = "mingru-gpu-check"

# --- delta-probe job mode (Task 6) --------------------------------------

_DELTA_PROBE_JOB_NAME = "mingru-gpu-delta-probe"
_DELTA_PROBE_RESULT_PREFIX = "MINGRU_GPU_PROBE_RESULT "
_DELTA_PROBE_OUT_DIR = _REPO_ROOT / "experiments" / "bench"
_DELTA_PROBE_JSON = _DELTA_PROBE_OUT_DIR / "gpu_delta_probe.json"
_DELTA_PROBE_MD = _DELTA_PROBE_OUT_DIR / "gpu_delta_probe.md"

# --- hetero36 job mode (Task 3) -----------------------------------------

_HETERO36_JOB_NAME = "mingru-gpu-hetero36"
_HETERO36_ROW_PREFIX = "MINGRU_LAB_ROW "
_HETERO36_ENV_PREFIX = "MINGRU_LAB_ENV "
# Spec §6 arm matrix: the only round names the submitting/dedup path
# recognizes as this round's rows.
_HETERO36_ROUNDS = ("hetero-gpu36-sg8", "hetero-gpu36-pd64", "hetero-gpu36-pd1024")
_LEDGER_PATH = _REPO_ROOT / "experiments" / "lab_results.jsonl"
_HETERO36_SIDECAR = _DELTA_PROBE_OUT_DIR / "gpu36_env.json"

# --- benchmarks job mode (Task 6, consumes Task 5) ----------------------

_BENCHMARKS_JOB_NAME = "mingru-gpu-benchmarks"
_BENCHMARKS_ROW_PREFIX = "MINGRU_LAB_ROW "
_BENCHMARKS_ENV_PREFIX = "MINGRU_LAB_ENV "
# The round names this round's submitting/dedup path recognizes as its own
# rows. The frozen `-01` tags (spec's original Global Constraints) are now
# the recorded pilot/calibration population (heterogeneous per-seed
# budgets, see `experiments.benchmark_tasks.BENCH_ROUND_TAGS`'s comment) --
# kept here hardcoded, forever, so old pilot job logs/sidecars stay
# parseable -- UNIONED with the current `-02` matrix tags, read from
# `BENCH_ROUND_TAGS` (the single source of truth `gpu_benchmark_campaign
# .py`'s `_ROUND_TAGS` and `report_benchmarks.py`'s `ROUND_TAGS` also bind
# to, so all three never risk drifting out of sync on the live tag), the
# S5-only probe tag(s) from `BENCH_PROBE_ROUND_TAGS` (same source-of-
# truth binding as `gpu_benchmark_campaign.py`'s `_PROBE_ROUND_TAGS`), AND
# the psMNIST-only reference tag(s) from `BENCH_REF_ROUND_TAGS`
# (`gpu_benchmark_campaign.py`'s `_REF_ROUND_TAGS`, the `gru-large`
# grounding reference), AND the per-arm correction tags from
# `BENCH_ARM_ROUND_OVERRIDES` (design spec, `.claude/output/specs/
# 2026-07-21-round-tag-override-design.md` §4 Ingest: the override tags
# join this allow-list so corrected rows are accepted, while every prior
# tag -- including the now-superseded `bench-*-02` rotation-hetero rows
# and the `bench-s5-probe-01` row -- stays parseable) -- the finish
# handler's dedup key already includes `variant` (`_valid_benchmarks_key`),
# so a probe, ref, or corrected arm's rows dedup correctly on their own
# round tag without any further change here. One round per task,
# independent of which arms/seeds that task's cell selects.
#
# `_BENCHMARKS_ROUNDS_PILOT` additionally hardcodes `bench-churn-01` (Task
# 6, `.claude/output/specs/2026-07-25-churn-benchmark-design.md` §6 "Round
# tags"): unlike the four `-01` tags above, churn's pilot tag was never a
# heterogeneous-budget population under `BENCH_ROUND_TAGS["churn"]`
# (`bench-churn-02`) -- it's churn's own from-the-start pilot tag (see
# `experiments.benchmark_tasks.BENCH_ROUND_TAGS`'s comment on the
# `"churn"` entry) -- but it gets the identical hardcoded treatment here
# for the identical reason: a frozen tag this allow-list must keep
# accepting forever so pilot job logs/sidecars stay parseable, even though
# `BENCH_ROUND_TAGS` (unioned in below) only ever carries churn's current
# `-02` matrix tag.
_BENCHMARKS_ROUNDS_PILOT = (
    "bench-s5-01",
    "bench-mqar-01",
    "bench-psmnist-01",
    "bench-pendulum-01",
    "bench-churn-01",
)
_BENCHMARKS_ROUNDS = (
    _BENCHMARKS_ROUNDS_PILOT
    + tuple(BENCH_ROUND_TAGS.values())
    + tuple(BENCH_PROBE_ROUND_TAGS.values())
    + tuple(BENCH_REF_ROUND_TAGS.values())
    + tuple(
        tag
        for per_task_tags in BENCH_ARM_ROUND_OVERRIDES.values()
        for tag in per_task_tags.values()
    )
)


def _sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def _job_preamble_steps(repo: str, ref: str) -> list[str]:
    """Shared job-shell preamble: clone, checkout, CUDA assert, triton install.

    Used by all three job builders (check, delta-probe, hetero36) to initialize
    the container before running their respective workloads. Each builder appends
    its own final step(s) to this list and joins them with `` && ``.
    """
    return [
        "set -eux",  # job shell is dash: -o pipefail unsupported (no pipes used)
        (
            "python -c 'import torch; assert torch.cuda.is_available(), "
            '"no CUDA device"; print(torch.__version__)\''
        ),
        "(python -c 'import triton' || pip install --no-cache-dir triton)",
        f"git clone --filter=blob:none {repo} /tmp/minGRU",
        f"cd /tmp/minGRU && git checkout --detach {ref}",
    ]


def _fetch_job_logs(job: Any) -> tuple[str | None, Exception | None]:
    """Best-effort ``job.logs`` fetch: ``(logs, None)`` or ``(None, exc)``.

    Log retrieval can fail independently of the job's own run status (SDK
    transient errors, etc.); every call site treats that as a soft
    failure to report rather than a crash. Rule-of-three hoist: this
    try/except was identically duplicated across ``_finish_delta_probe``,
    ``_finish_hetero36``, and ``main()``'s check-mode failure path.
    Returns the exception instead of swallowing it so each call site can
    keep its own (slightly different) existing message text -- callers
    check ``exc is not None`` first, which also guarantees ``logs`` is a
    ``str`` on the success path.
    """
    try:
        return job.logs, None
    except Exception as exc:  # log retrieval is best-effort
        return None, exc


def build_command(repo: str, ref: str, bench: bool) -> str:
    steps = _job_preamble_steps(repo, ref)
    steps.append("cd /tmp/minGRU && python scripts/bench_scans.py --check")
    if bench:
        steps.append("cd /tmp/minGRU && python scripts/bench_scans.py --bench")
    return " && ".join(steps)


def build_delta_probe_command(repo: str, ref: str) -> str:
    """Job-shell command for ``--job delta-probe``: clone + probe run.

    Uses the shared clone/checkout/triton-install preamble (Task 6
    brief: ``torch.compile``'s Inductor needs triton on GPU, same as the
    parity suite), running ``scripts/gpu_delta_probe.py`` instead of the
    parity suite.

    Deliberately NO keepalive heartbeat: the Lightning tier's 10-minute
    idle shutdown applies to interactive studio sessions, not to jobs --
    a job runs its command to completion regardless of log quietness
    (user clarification, 2026-07-18). The first delta-probe run carried a
    detached ``while true ... &`` heartbeat anyway, and on a studio-mode
    job that backgrounded loop OUTLIVED the finished command chain, kept
    the job "running" so ``job.wait()`` never returned, and kept the
    machine billing until stopped via the SDK. Jobs must keep their
    command chain foreground-only.
    """
    steps = _job_preamble_steps(repo, ref)
    steps.append("cd /tmp/minGRU && python scripts/gpu_delta_probe.py")
    return " && ".join(steps)


def build_hetero36_command(repo: str, ref: str) -> str:
    """Job-shell command for ``--job hetero36``: clone + 36-seed campaign run.

    Uses the shared clone/checkout/triton-install preamble (``torch.compile``'s
    Inductor needs triton on GPU here too), running ``scripts/gpu_hetero_campaign.py``
    instead -- no extra args, the campaign script's own CLI covers the full
    3x36 arm matrix (spec §5/§6: ``cd /tmp/minGRU && python
    scripts/gpu_hetero_campaign.py``). Deliberately NO keepalive heartbeat,
    foreground-only command chain -- see ``build_delta_probe_command``'s
    docstring for why a backgrounded loop outlives a studio-mode job's
    command chain and keeps it billing.
    """
    steps = _job_preamble_steps(repo, ref)
    steps.append("cd /tmp/minGRU && python scripts/gpu_hetero_campaign.py")
    return " && ".join(steps)


def build_benchmarks_command(
    repo: str,
    ref: str,
    tasks: list[str] | None,
    arms: list[str] | None,
    seeds: list[int] | None,
) -> str:
    """Job-shell command for ``--job benchmarks``: clone + benchmark campaign run.

    Uses the shared clone/checkout/triton-install preamble, then installs
    ``torchvision`` (Task 6 brief -- the psMNIST task downloads MNIST inside
    the job; no other job mode needs it, so this install is appended here
    rather than folded into ``_job_preamble_steps`` itself, keeping every
    other job mode's preamble unchanged). When ``tasks`` includes ``churn``
    (or is ``None``, meaning every task -- churn included), a second
    ``pip install --no-cache-dir pandas`` step is appended the same way
    (churn round, Task 6 of this round: ``RosbankLoader`` lazy-imports
    pandas inside the job, mirroring the ``torchvision``/psMNIST
    precedent -- see ``experiments/benchmark_tasks.py``'s ``RosbankLoader``
    docstring). Then runs ``scripts/gpu_benchmark_campaign.py`` with
    ``--tasks``/``--arms``/``--seeds`` passthrough when given (omitted
    entirely when ``None``, so the campaign script's own defaults -- every
    task/arm/seed -- apply).
    Deliberately NEVER passes ``--steps``: that flag is the campaign
    script's local-smoke-only override (shrinks its eval cadence via
    ``_resolve_eval_every``) -- production budgets are the committed
    ``TaskSpec`` values (spec section 7: "no per-arm tuning"). Foreground-
    only, no keepalive -- see ``build_delta_probe_command``'s docstring for
    why a backgrounded loop outlives a studio-mode job's command chain and
    keeps it billing.

    Exports ``MINGRU_SCAN=triton`` ahead of the campaign invocation (user
    decision at the evidence-phase gate, recorded in the intent ledger's
    Amendments -- "Triton everywhere"): every seed matrix in this round
    runs under a uniform Triton scan backend on L4, and every ledger row
    this campaign emits records that backend in its own ``config.scan``
    field (``benchmark_lab.run_arm`` reads ``MINGRU_SCAN`` from the
    process environment). This is a job-command-only default -- local/
    campaign-script defaults are untouched (``MINGRU_SCAN`` unset locally
    still resolves to the scan dispatcher's own ``auto`` default). The
    ``MINGRU_SCAN`` contract itself is unchanged here: ``triton`` fails
    loud with a reason rather than silently downgrading (see
    ``src/mingru/triton_scans.py``), so a task/arm combination this round
    turns out to have no Triton path for is a clear in-job failure, not a
    silent fallback.
    """
    steps = _job_preamble_steps(repo, ref)
    steps.append("pip install --no-cache-dir torchvision")
    if tasks is None or "churn" in tasks:
        steps.append("pip install --no-cache-dir pandas")
    steps.append("export MINGRU_SCAN=triton")
    command = "cd /tmp/minGRU && python scripts/gpu_benchmark_campaign.py"
    if tasks:
        command += " --tasks " + " ".join(tasks)
    if arms:
        command += " --arms " + " ".join(arms)
    if seeds:
        command += " --seeds " + " ".join(str(s) for s in seeds)
    steps.append(command)
    return " && ".join(steps)


def _resolve_shard_seed_pool(tasks: list[str] | None, seeds: list[int] | None) -> list[int]:
    """The full seed pool ``--shards`` splits into disjoint per-job slices.

    Mirrors ``gpu_benchmark_campaign.py``'s own ``_resolve_seeds`` seed-
    source convention (an explicit ``--seeds`` override applies uniformly
    to every selected task; otherwise each task's own ``TaskSpec.seeds``
    count) so a shard's ``--seeds`` passthrough selects exactly the seeds
    the unsharded run would have used, just partitioned.

    An explicit ``seeds`` list is returned as given (sharding partitions
    whatever seeds were named). Without one, every task in ``tasks`` (or
    every registered task, when ``tasks`` is ``None``) must share the same
    seed count -- sharding a mixed-count task set would need a different
    slice per task, which the campaign script's flat ``--seeds`` passthrough
    (one list shared by every task in the job) can't express. A mismatch
    raises ``ValueError`` rather than silently sharding by one task's count
    and running the wrong seeds for another. An unrecognized task name in
    ``tasks`` also raises ``ValueError`` (rather than an unguarded
    ``TASKS[name]`` bare ``KeyError``): ``main`` only catches ``ValueError``
    around the shard-resolution call, so a ``--tasks`` typo must degrade to
    the module's standard "error: ..." + exit 2 pattern, not an uncaught
    traceback.
    """
    if seeds is not None:
        return list(seeds)
    task_names = tasks if tasks is not None else list(TASKS)
    unknown = [name for name in task_names if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s) {unknown} (known tasks: {sorted(TASKS)})")
    seed_counts = {TASKS[name].seeds for name in task_names}
    if len(seed_counts) != 1:
        counts = {name: TASKS[name].seeds for name in task_names}
        raise ValueError(
            f"cannot shard tasks with differing seed counts ({counts}) "
            "without an explicit --seeds list"
        )
    return list(range(seed_counts.pop()))


def _shard_seed_lists(pool: list[int], shards: int) -> list[list[int]]:
    """Split ``pool`` into ``shards`` disjoint, exhaustive, contiguous slices.

    Job ``k`` of ``shards`` gets ``pool[k * size : (k + 1) * size]`` --
    e.g. the churn round's 36-seed pool split 6 ways gives job ``k`` seeds
    ``6k..6k+5`` (Task 6 brief). Requires ``pool`` to divide evenly by
    ``shards``: rejected loudly (``ValueError``) rather than silently
    dropping seeds or distributing them unevenly, since a dropped seed
    would silently shrink the matrix below its committed size.
    """
    if shards < 1:
        raise ValueError(f"--shards must be >= 1, got {shards}")
    if len(pool) % shards != 0:
        raise ValueError(
            f"cannot evenly split {len(pool)} seeds into {shards} shards "
            f"({len(pool)} % {shards} != 0)"
        )
    size = len(pool) // shards
    return [pool[i * size : (i + 1) * size] for i in range(shards)]


def _extract_last(prefix: str, text: str) -> dict[str, Any] | None:
    """Parse the JSON payload of the last well-formed ``prefix``-prefixed line.

    ``text`` is the job's fetched logs, which may interleave
    clone/checkout output and the probe's own
    progress lines around the single ``MINGRU_GPU_PROBE_RESULT`` line
    this looks for. A malformed matching line (or none at all) degrades
    to ``None`` -- this function never raises -- so a caller can treat
    "no result" as a clear, reportable error rather than a crash.

    DUPLICATION-PENDING: this is the same pattern as
    ``scripts/scaling_probe.py``'s ``_extract_last`` (line-marker
    extraction with malformed-line tolerance), duplicated here rather
    than imported since the two scripts are independent job-runner
    entry points with no existing shared module that owns this logic
    (``scripts/_bench_env.py`` is the closest candidate) -- flagged for
    the orchestrator to hoist into a shared helper if a third site
    appears.
    """
    parsed: dict[str, Any] | None = None
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            parsed = json.loads(line[len(prefix) :])
        except (json.JSONDecodeError, ValueError):
            continue
    return parsed


def _extract_all(prefix: str, text: str) -> list[Any]:
    """Parse the JSON payload of every well-formed ``prefix``-prefixed line.

    Sibling to ``_extract_last`` for job modes -- like ``hetero36`` --
    that transport MANY marked rows per job rather than a single terminal
    result: each completed seed prints its own ``MINGRU_LAB_ROW <json>``
    line, and all of them must survive extraction, not just the last.
    Same guarded-parse contract as ``_extract_last``: a malformed (not
    valid JSON) matching line is skipped, never raised, and the returned
    list preserves the order payloads appeared in the log (later
    dedup/append logic relies on this to make retries idempotent without
    reordering rows). Return type is deliberately ``list[Any]``, not
    ``list[dict]``: this function only guards PARSEABILITY, not SHAPE --
    a line's payload can be well-formed JSON that isn't a row at all
    (``MINGRU_LAB_ROW [1, 2, 3]`` parses to a ``list``, not a ``dict``).
    Callers that need row shape (a dict with the hetero36 round/seed
    keys) apply that guard themselves -- see ``_valid_hetero36_key``.
    """
    payloads: list[Any] = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payloads.append(json.loads(line[len(prefix) :]))
        except (json.JSONDecodeError, ValueError):
            continue
    return payloads


def _fmt_probe_cell(value: Any, spec: str = ".4f") -> str:
    """Render a possibly-``None``/non-numeric probe field as a table cell."""
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def _fmt_bytes_mb(value: Any) -> str:
    """Render a possibly-``None`` byte count as MB (1 decimal place)."""
    return _fmt_probe_cell(value / 1e6, ".1f") if isinstance(value, (int, float)) else "n/a"


def _fmt_bar_judgment(value: Any) -> str:
    """Render a three-state bar-judgment field (``True``/``False``/``None``) as a table cell.

    ``None`` means "not judged" (e.g. the vs-compile bar when that shape's
    compile arm failed), not "failed" -- rendered distinctly from ``False``
    so a reader never mistakes "can't judge" for "judged and failed".
    """
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "n/a"


def _render_delta_probe_markdown(result: dict[str, Any]) -> str:
    """Render the delta-probe JSON artifact as Markdown (Task 5/6).

    Two tables: the first carries every arm's step-time medians (eager,
    floor, compile, triton) plus the speed-bar judgment (spec section 9.1
    acceptance criterion 1) per shape; the second carries the eager-vs-
    triton peak-memory comparison and its own bar judgment (spec section 7
    memory invariant) -- split out rather than crammed into the first
    table's already-wide row, since MB figures and the PASS/FAIL/n/a bar
    cells read as their own self-contained comparison.
    """
    env = result.get("env", {})
    lines = [
        "# CUDA fusion-headroom probe: DeltaMinGRU chunked-WY (Task 6)",
        "",
        "Eager chunked-WY forward+backward vs. its matmul-FLOP floor "
        "(approximate, see each row's `floor_method`) vs. `torch.compile` "
        "vs. the Triton chunked-WY kernel, per `.git/sdd/task-6-brief.md` / "
        "`scripts/gpu_delta_probe.py`. This is a GPU evidence stratum: "
        "nothing below is comparable to the pinned-CPU rows in "
        "`experiments/lab_results.jsonl` / `EXPERIMENTS.md`.",
        "",
        f"Env: torch {env.get('torch_version')} (CUDA {env.get('cuda_version')}), "
        f"device {env.get('cuda_device_name')} (capability "
        f"{env.get('device_capability')}), triton "
        f"{env.get('triton_version')}, platform {env.get('platform')}, "
        f"B={env.get('batch_size')}, warmup={env.get('warmup_steps')}, "
        f"timed steps={env.get('timed_steps')}, generated "
        f"{env.get('timestamp')}.",
        "",
        "| shape | config | B | T | eager median (s) | floor (s, approx) | "
        "compile median (s) | compile status | triton median (s) | "
        "headroom (eager/floor) | compile recovered | triton/compile | "
        "triton/eager | bar met |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.get("shapes", []):
        cfg_str = ", ".join(f"{k}={v}" for k, v in row.get("live_config", {}).items())
        lines.append(
            f"| {row.get('label')} | {row.get('config_name')}: {cfg_str} | "
            f"{row.get('B')} | {row.get('T')} | "
            f"{_fmt_probe_cell(row.get('eager_step_secs_median'))} | "
            f"{_fmt_probe_cell(row.get('floor_step_secs'))} | "
            f"{_fmt_probe_cell(row.get('compile_step_secs_median'))} | "
            f"{row.get('compile_status')} | "
            f"{_fmt_probe_cell(row.get('triton_step_secs_median'))} | "
            f"{_fmt_probe_cell(row.get('headroom_eager_over_floor'), '.2f')} | "
            f"{_fmt_probe_cell(row.get('compile_recovered_fraction'), '.2%')} | "
            f"{_fmt_probe_cell(row.get('triton_vs_compile_ratio'), '.2f')} | "
            f"{_fmt_probe_cell(row.get('triton_vs_eager_ratio'), '.2f')} | "
            f"{_fmt_bar_judgment(row.get('bar_met'))} |"
        )
    lines += [
        "",
        "`floor` rows are an explicit approximation of the dominant "
        "GEMM/triangular-solve contractions only, scaled by the standard "
        "3x fwd-GEMM convention -- see each shape's `floor_method` in the "
        "JSON artifact for the full disclosure. A `compile status` other "
        "than `ok` means Inductor failed on that shape; see the JSON "
        "artifact's `compile_error` for that row. `bar met` is the spec "
        "section 9.1 speed bar (triton fwd+bwd median <= 1.2x the recorded "
        "compile median AND <= the recorded eager median, both judged on "
        "this same run's own medians) -- `n/a` means the vs-compile leg "
        "couldn't be judged because that shape's compile arm failed, not "
        "that the bar failed.",
        "",
        "| shape | eager peak mem (MB) | triton peak mem (MB) | "
        "triton/eager peak mem | memory bar met |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result.get("shapes", []):
        lines.append(
            f"| {row.get('label')} | "
            f"{_fmt_bytes_mb(row.get('eager_peak_mem_bytes'))} | "
            f"{_fmt_bytes_mb(row.get('triton_peak_mem_bytes'))} | "
            f"{_fmt_probe_cell(row.get('triton_vs_eager_peak_mem_ratio'), '.2f')} | "
            f"{_fmt_bar_judgment(row.get('memory_bar_met'))} |"
        )
    lines += [
        "",
        "`memory bar met` is the spec section 7 memory invariant "
        "(triton-path peak training memory <= eager-path peak training "
        "memory), judged per shape from `eager_peak_mem_bytes`/"
        "`triton_peak_mem_bytes` in the JSON artifact.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_delta_probe_artifact(result: dict[str, Any]) -> None:
    _DELTA_PROBE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _DELTA_PROBE_JSON.write_text(json.dumps(result, indent=2) + "\n")
    _DELTA_PROBE_MD.write_text(_render_delta_probe_markdown(result))


def _finish_delta_probe(job: Any, ok: bool) -> int:
    """Post-``job.wait()`` handling for ``--job delta-probe``.

    Fetches the job's logs (best-effort -- see ``_fetch_job_logs``),
    extracts the last ``MINGRU_GPU_PROBE_RESULT`` line, and writes the
    local artifact. A missing/malformed result line is a clear error
    exit -- never a partial artifact write, per the task brief.
    """
    logs, exc = _fetch_job_logs(job)
    if exc is not None:
        print(f"error: could not fetch logs: {exc}", file=sys.stderr)
        return 1
    assert logs is not None  # guaranteed by _fetch_job_logs's exc is None contract
    result = _extract_last(_DELTA_PROBE_RESULT_PREFIX, logs)
    if result is None:
        print(
            "error: no well-formed MINGRU_GPU_PROBE_RESULT line found in job "
            "logs -- not writing a partial artifact. Log tail:\n"
            + "\n".join(logs.splitlines()[-40:]),
            file=sys.stderr,
        )
        return 1
    _write_delta_probe_artifact(result)
    print(f"wrote {_DELTA_PROBE_JSON} and {_DELTA_PROBE_MD}")
    return 0 if ok else 1


def _valid_hetero36_key(row: Any, rounds: tuple[str, ...]) -> tuple[str, int] | None:
    """Return ``row``'s ``(round, seed)`` key if shape-valid, else ``None``.

    Guards the shape every dedup/append/ledger-read path below depends on.
    A decoded JSON payload can be well-formed JSON and still be the wrong
    SHAPE -- ``MINGRU_LAB_ROW [1, 2, 3]`` parses fine but isn't a row at
    all, and a dict missing ``round``/``seed`` would otherwise be appended
    to the ledger under a garbage ``(None, None)`` key that then crashes
    every later read of that ledger. This function is the single choke
    point that prevents both: ``row`` must be a ``dict``, its ``round``
    must be one of ``rounds`` (the three hetero36 round names -- an
    unrecognized round is never this round's data), and its ``seed`` must
    be an ``int`` (``bool`` is excluded despite being an ``int`` subclass
    in Python, since a JSON ``true``/``false`` is never a valid seed).
    """
    if not isinstance(row, dict):
        return None
    round_name = row.get("round")
    seed = row.get("seed")
    if round_name not in rounds:
        return None
    if not isinstance(seed, int) or isinstance(seed, bool):
        return None
    return round_name, seed


def _valid_benchmarks_key(row: Any, rounds: tuple[str, ...]) -> tuple[str, str, str, int] | None:
    """Return ``row``'s ``(round, task, variant, seed)`` key if shape-valid, else ``None``.

    Sibling to ``_valid_hetero36_key`` for the benchmarks round's four-field
    dedup key (spec section 6's marker-line protocol): a task's arms
    and many seeds all share one round tag, so ``variant`` (the arm) must
    be part of the key or two different arms at the same seed would
    collide. Same guard shape as ``_valid_hetero36_key``: ``row`` must be a
    ``dict``, its ``round`` must be one of ``rounds`` (the four benchmarks
    round names), its ``task``/``variant`` must be non-empty strings, and
    its ``seed`` must be an ``int`` excluding ``bool``.
    """
    if not isinstance(row, dict):
        return None
    round_name = row.get("round")
    task = row.get("task")
    variant = row.get("variant")
    seed = row.get("seed")
    if round_name not in rounds:
        return None
    if not isinstance(task, str) or not task or not isinstance(variant, str) or not variant:
        return None
    if not isinstance(seed, int) or isinstance(seed, bool):
        return None
    return round_name, task, variant, seed


_RowKey = tuple[Any, ...]


def _dedup_batch_last_wins_by_key(
    rows: list[Any], key_fn: Callable[[Any], _RowKey | None]
) -> tuple[list[dict[str, Any]], int, int]:
    """Resolve one extracted batch to last-N-wins per ``key_fn(row)``.

    Generic core shared by every job mode's dedup path -- hetero36's
    ``(round, seed)`` key and the benchmarks round's four-field
    ``(round, task, variant, seed)`` key are both call sites (see
    ``_valid_hetero36_key``/``_valid_benchmarks_key``), parameterized here
    rather than duplicated per mode. Spec §6: "extraction is last-N-wins
    per <key>". Returns ``(deduped_rows, rows_skipped_invalid,
    rows_deduped_in_batch)``: a shape-invalid row (``key_fn`` returns
    ``None``) never participates -- it's dropped and counted rather than
    crashing or silently entering the ledger under a garbage key. Valid
    rows are folded into a dict keyed by ``key_fn(row)`` IN LOG ORDER: a
    plain ``dict``'s iteration order is the position of a key's FIRST
    occurrence, but assigning to an already-present key still overwrites
    its value -- so ``dict.values()`` afterward yields exactly one row per
    key, in first-occurrence order, holding each key's LAST-seen row.
    ``rows_deduped_in_batch`` counts every row that LOST that overwrite
    (one less than the number of rows sharing a key) -- without it, a
    losing row would vanish from the sidecar's accounting: the caller's
    ``len(rows) == appended + skipped_duplicate + skipped_invalid +
    deduped_in_batch`` reconciliation depends on this count existing.
    """
    by_key: dict[_RowKey, dict[str, Any]] = {}
    skipped_invalid = 0
    deduped_in_batch = 0
    for row in rows:
        key = key_fn(row)
        if key is None:
            skipped_invalid += 1
            continue
        if key in by_key:
            deduped_in_batch += 1
        by_key[key] = row
    return list(by_key.values()), skipped_invalid, deduped_in_batch


def _existing_keys_by_key(
    ledger_path: Path, key_fn: Callable[[Any], _RowKey | None]
) -> set[_RowKey]:
    """Keys already present in the local ledger, per ``key_fn``.

    Generic core shared by every job mode's dedup path -- see
    ``_dedup_batch_last_wins_by_key``'s docstring for why this is
    parameterized rather than hardcoded to one mode's key shape. Guarded
    per-line parse (a malformed ledger line is skipped, never raised)
    mirrors the extraction contract this module already applies to job
    logs; shape-invalid lines (non-dict JSON, or a dict for which
    ``key_fn`` returns ``None``) are skipped the same way -- a garbage line
    already in the ledger must never crash every future run that reads it
    back. A ledger that doesn't exist yet (fresh checkout, or a
    ``tmp_path`` fixture in tests) is treated as "no existing rows".
    """
    keys: set[_RowKey] = set()
    if not ledger_path.exists():
        return keys
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        key = key_fn(row)
        if key is not None:
            keys.add(key)
    return keys


class _AppendResult(NamedTuple):
    appended: int
    skipped_duplicate: int
    skipped_invalid: int
    deduped_in_batch: int


def _append_rows_by_key(
    ledger_path: Path, rows: list[Any], key_fn: Callable[[Any], _RowKey | None]
) -> _AppendResult:
    """Append ``rows`` whose ``key_fn(row)`` isn't already in the ledger.

    Generic core shared by every job mode's dedup-append path -- see
    ``_dedup_batch_last_wins_by_key``'s docstring for why this is
    parameterized. Shape-invalid rows (``key_fn`` returns ``None``) are
    dropped and counted rather than appended or crashing the batch. The
    surviving batch is first resolved last-N-wins per key (spec §6, see
    ``_dedup_batch_last_wins_by_key``); only then is each remaining row
    checked against the ledger's existing keys -- a key already present in
    the ledger still skips unconditionally, keeping retries idempotent.
    Log order (of the deduped batch) is preserved in both the returned
    counts and the appended lines. The four returned counts reconcile
    exactly against ``len(rows)``: every input row is either appended,
    skipped as an existing-ledger duplicate, skipped as shape-invalid, or
    deduped away by a later same-batch occurrence.
    """
    batch, skipped_invalid, deduped_in_batch = _dedup_batch_last_wins_by_key(rows, key_fn)
    existing = _existing_keys_by_key(ledger_path, key_fn)
    appended_lines: list[str] = []
    appended = 0
    skipped_duplicate = 0
    for row in batch:
        key = key_fn(row)
        assert key is not None  # batch already filtered by _dedup_batch_last_wins_by_key
        if key in existing:
            skipped_duplicate += 1
            continue
        appended_lines.append(json.dumps(row))
        existing.add(key)
        appended += 1
    if appended_lines:
        with ledger_path.open("a") as f:
            for line in appended_lines:
                f.write(line + "\n")
    return _AppendResult(appended, skipped_duplicate, skipped_invalid, deduped_in_batch)


def _append_new_rows(ledger_path: Path, rows: list[Any], rounds: tuple[str, ...]) -> _AppendResult:
    """Append ``rows`` whose ``(round, seed)`` isn't already in the ledger.

    Thin wrapper over ``_append_rows_by_key`` binding hetero36's
    ``(round, seed)`` key (see ``_valid_hetero36_key``).
    """
    return _append_rows_by_key(ledger_path, rows, lambda row: _valid_hetero36_key(row, rounds))


def _append_benchmarks_rows(
    ledger_path: Path, rows: list[Any], rounds: tuple[str, ...]
) -> _AppendResult:
    """Append ``rows`` whose ``(round, task, variant, seed)`` isn't already
    in the ledger.

    Thin wrapper over ``_append_rows_by_key`` binding the benchmarks
    round's four-field key (see ``_valid_benchmarks_key``).
    """
    return _append_rows_by_key(ledger_path, rows, lambda row: _valid_benchmarks_key(row, rounds))


def _build_hetero36_sidecar(
    env: dict[str, Any],
    rows: list[Any],
    result: _AppendResult,
    rounds: tuple[str, ...],
) -> dict[str, Any]:
    """Assemble the ``gpu36_env.json`` sidecar payload (spec §6).

    ``per_seed_wall_secs`` is keyed ``{round: {seed: secs}}`` from every
    shape-valid extracted row's own ``secs`` field -- over ALL extracted
    rows, not just the newly-appended ones, so a retried job's sidecar
    still reports the full observed wall-cost picture even though
    duplicate rows themselves aren't re-appended to the ledger (a losing
    intra-batch duplicate's ``secs`` is naturally overwritten here too,
    by the same last-N-wins rule as the ledger append). A shape-invalid
    row (see ``_valid_hetero36_key``) or one missing ``secs`` is skipped
    from this map rather than raising.

    The four ``rows_*`` counts reconcile exactly: ``rows_extracted ==
    rows_appended + rows_skipped_duplicate + rows_skipped_invalid +
    rows_deduped_in_batch`` -- no extracted row is ever silently
    unaccounted for.
    """
    per_seed_wall_secs: dict[str, dict[str, float]] = {}
    for row in rows:
        key = _valid_hetero36_key(row, rounds)
        secs = row.get("secs") if isinstance(row, dict) else None
        if key is None or secs is None:
            continue
        round_name, seed = key
        per_seed_wall_secs.setdefault(round_name, {})[str(seed)] = secs
    return {
        "env": env,
        "rows_extracted": len(rows),
        "rows_appended": result.appended,
        "rows_skipped_duplicate": result.skipped_duplicate,
        "rows_skipped_invalid": result.skipped_invalid,
        "rows_deduped_in_batch": result.deduped_in_batch,
        "per_seed_wall_secs": per_seed_wall_secs,
    }


def _finish_hetero36(job: Any, ok: bool) -> int:
    """Post-``job.wait()`` handling for ``--job hetero36``.

    Fetches the job's logs (best-effort -- see ``_fetch_job_logs``),
    extracts every ``MINGRU_LAB_ROW`` line (guarded, malformed lines
    skipped -- see ``_extract_all``) and the single ``MINGRU_LAB_ENV``
    line (guarded, last-wins -- see ``_extract_last``). Either rows
    entirely absent or the env line missing/malformed is a clear error
    exit with NOTHING appended or written -- both checks run before any
    ledger mutation or sidecar write, so a bad log never produces a
    partial ledger append. On success, resolves the batch last-N-wins
    per ``(round, seed)`` (counted as ``rows_deduped_in_batch``), drops
    shape-invalid rows with a stderr warning (counted as
    ``rows_skipped_invalid``, never appended, never a crash -- see
    ``_valid_hetero36_key``), dedups the rest against
    ``experiments/lab_results.jsonl`` for the three hetero36 round names,
    appends only the new ones (preserving log order), and writes the env
    sidecar. The sidecar's counts reconcile exactly against the raw
    extracted count -- see ``_build_hetero36_sidecar``.
    """
    logs, exc = _fetch_job_logs(job)
    if exc is not None:
        print(f"error: could not fetch logs: {exc}", file=sys.stderr)
        return 1
    assert logs is not None  # guaranteed by _fetch_job_logs's exc is None contract

    rows = _extract_all(_HETERO36_ROW_PREFIX, logs)
    if not rows:
        print(
            "error: no well-formed MINGRU_LAB_ROW lines found in job logs -- "
            "not appending or writing a sidecar. Log tail:\n" + "\n".join(logs.splitlines()[-40:]),
            file=sys.stderr,
        )
        return 1

    env = _extract_last(_HETERO36_ENV_PREFIX, logs)
    if env is None:
        print(
            "error: no well-formed MINGRU_LAB_ENV line found in job logs -- "
            "not appending or writing a sidecar. Log tail:\n" + "\n".join(logs.splitlines()[-40:]),
            file=sys.stderr,
        )
        return 1

    result = _append_new_rows(_LEDGER_PATH, rows, _HETERO36_ROUNDS)
    if result.skipped_invalid:
        print(
            f"warning: skipped {result.skipped_invalid} shape-invalid "
            "MINGRU_LAB_ROW row(s) (non-dict payload, unrecognized round, "
            "or non-int seed) -- not appended to the ledger",
            file=sys.stderr,
        )
    sidecar = _build_hetero36_sidecar(env, rows, result, _HETERO36_ROUNDS)
    _DELTA_PROBE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _HETERO36_SIDECAR.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(
        f"appended {result.appended} new row(s), skipped {result.skipped_duplicate} "
        f"duplicate(s), {result.skipped_invalid} invalid, and "
        f"{result.deduped_in_batch} intra-batch duplicate(s); wrote {_HETERO36_SIDECAR}"
    )
    return 0 if ok else 1


def _build_benchmarks_sidecar(
    env: dict[str, Any],
    task: str,
    rows: list[Any],
    result: _AppendResult,
) -> dict[str, Any]:
    """Assemble one task's ``bench_<task>_env.json`` sidecar payload (spec §6).

    ``rows`` is already this task's bucket (see ``_finish_benchmarks``), not
    the full job's extracted batch -- so every count here is scoped to
    ``task``, and ``rows_extracted == rows_appended + rows_skipped_duplicate
    + rows_skipped_invalid + rows_deduped_in_batch`` reconciles per file,
    mirroring ``_build_hetero36_sidecar``'s reconciliation invariant.

    ``per_variant_seed_wall_secs`` is keyed ``{variant: {seed: secs}}``,
    NOT ``{round: {seed: secs}}`` like ``_build_hetero36_sidecar`` -- one
    task's rows span multiple arms (variants) at the same seed, and a
    task's round tag is fixed (one round per task), so keying by round
    first would collide every arm sharing a seed onto the same sub-map.
    Keying by variant first is the direct fix; seed alone was never a
    unique key here the way it was for hetero36 (there, one round already
    implied exactly one arm). A shape-invalid row (see
    ``_valid_benchmarks_key``) or one missing ``secs`` is skipped from this
    map rather than raising.
    """
    per_variant_seed_wall_secs: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        variant = row.get("variant")
        seed = row.get("seed")
        secs = row.get("secs")
        if not isinstance(variant, str) or not isinstance(seed, int) or isinstance(seed, bool):
            continue
        if secs is None:
            continue
        per_variant_seed_wall_secs.setdefault(variant, {})[str(seed)] = secs
    return {
        "env": env,
        "task": task,
        "rows_extracted": len(rows),
        "rows_appended": result.appended,
        "rows_skipped_duplicate": result.skipped_duplicate,
        "rows_skipped_invalid": result.skipped_invalid,
        "rows_deduped_in_batch": result.deduped_in_batch,
        "per_variant_seed_wall_secs": per_variant_seed_wall_secs,
    }


def _bench_sidecar_path(task: str) -> Path:
    return _DELTA_PROBE_OUT_DIR / f"bench_{task}_env.json"


def _read_benchmarks_sidecar(sidecar_path: Path) -> dict[str, Any] | None:
    """Best-effort guarded read of a previously-written per-task sidecar.

    ``--shards`` (Task 6 hardening): each shard's ``_finish_benchmarks``
    call only sees that shard's own rows, so a later shard needs to read
    back an earlier shard's already-written sidecar before merging onto it
    (see ``_merge_benchmarks_sidecar``). Guarded parse -- a missing file,
    unreadable JSON, or a JSON value that isn't an object all degrade to
    ``None`` (never raise) -- mirrors the house convention already used for
    ledger/log parsing elsewhere in this module (e.g.
    ``_existing_keys_by_key``): a malformed or absent prior artifact is
    "nothing to merge onto", not a crash.
    """
    if not sidecar_path.exists():
        return None
    try:
        parsed = json.loads(sidecar_path.read_text())
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _int_field(payload: dict[str, Any], key: str) -> int:
    """``payload[key]`` if a non-bool ``int``, else ``0``.

    Guards ``_merge_benchmarks_sidecar``'s count-summing against a prior
    sidecar that's structurally a JSON object but has a malformed/missing
    count field (hand-edited, truncated, or from a future schema version)
    -- summing must never raise on a field that isn't the ``int`` it
    contracts to.
    """
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _merge_benchmarks_sidecar(
    existing: dict[str, Any] | None, fresh: dict[str, Any]
) -> dict[str, Any]:
    """Union ``fresh`` (this shard's own sidecar payload) onto ``existing``
    (an earlier shard's already-written sidecar for the same task), or
    return ``fresh`` unchanged when there's no valid prior sidecar.

    ``--shards`` (Task 6 hardening): writing each shard's sidecar wholesale
    would leave only the LAST-finishing shard's ``per_variant_seed_wall_secs``
    on disk, silently covering a fraction of the round's timing disclosure
    (rows themselves are unaffected -- the ledger append is already
    idempotent dedup-by-key). Two-level union of
    ``per_variant_seed_wall_secs`` (per-variant, then per-seed) so seeds
    from every shard accumulate rather than overwrite. The four
    reconciliation counts (``rows_extracted``/``rows_appended``/
    ``rows_skipped_duplicate``/``rows_skipped_invalid``/
    ``rows_deduped_in_batch``) are SUMMED across shards too, not just the
    timing map -- leaving them at one shard's value while the timing map
    reflects every shard would break ``_build_benchmarks_sidecar``'s own
    documented reconciliation invariant on the merged artifact and actively
    mislead a reader about how many rows the round covers. ``env``/``task``
    are taken from ``fresh`` (identical across shards under the round's
    single-stratum invariant, spec section 7). A malformed
    ``per_variant_seed_wall_secs`` in ``existing`` (not a dict -- hand-
    edited, truncated, or from a future schema version) degrades to "no
    prior timings to merge" rather than raising, the same guard class as
    ``_int_field``'s count-field guard above.
    """
    if existing is None:
        return fresh

    existing_secs = existing.get("per_variant_seed_wall_secs", {})
    if not isinstance(existing_secs, dict):
        existing_secs = {}
    merged_secs: dict[str, dict[str, float]] = {}
    for variant, seed_secs in existing_secs.items():
        if isinstance(seed_secs, dict):
            merged_secs[variant] = dict(seed_secs)
    for variant, seed_secs in fresh["per_variant_seed_wall_secs"].items():
        merged_secs.setdefault(variant, {}).update(seed_secs)

    return {
        **fresh,
        "rows_extracted": _int_field(existing, "rows_extracted") + fresh["rows_extracted"],
        "rows_appended": _int_field(existing, "rows_appended") + fresh["rows_appended"],
        "rows_skipped_duplicate": (
            _int_field(existing, "rows_skipped_duplicate") + fresh["rows_skipped_duplicate"]
        ),
        "rows_skipped_invalid": (
            _int_field(existing, "rows_skipped_invalid") + fresh["rows_skipped_invalid"]
        ),
        "rows_deduped_in_batch": (
            _int_field(existing, "rows_deduped_in_batch") + fresh["rows_deduped_in_batch"]
        ),
        "per_variant_seed_wall_secs": merged_secs,
    }


def _row_task(row: Any) -> str | None:
    """The row's ``task`` bucket key, or ``None`` if it can't be attributed.

    The single guard every task-bucketing site in ``_finish_benchmarks``
    shares (the ``tasks_present`` set, the ``unattributed`` list, and each
    task's own row bucket) -- ``row`` must be a ``dict`` and its ``task``
    field a non-empty string (an empty string would otherwise route to a
    malformed ``bench__env.json`` sidecar path). Deriving this once and
    reusing it everywhere, rather than re-deriving the same guard per call
    site, is what keeps the three sites from drifting out of sync -- an
    earlier version re-derived an ``isinstance(row, dict)``-less filter for
    the per-task row bucket and crashed with ``AttributeError`` on a
    non-dict ``MINGRU_LAB_ROW`` payload (quality review REQUIRED FIX 1).
    """
    if not isinstance(row, dict):
        return None
    task = row.get("task")
    return task if isinstance(task, str) and task else None


def _finish_benchmarks(job: Any, ok: bool, *, merge_sidecar: bool = False) -> int:
    """Post-``job.wait()`` handling for ``--job benchmarks``.

    Fetches the job's logs (best-effort -- see ``_fetch_job_logs``),
    extracts every ``MINGRU_LAB_ROW`` line (guarded, malformed lines
    skipped -- see ``_extract_all``) and the single ``MINGRU_LAB_ENV`` line
    (guarded, last-wins -- see ``_extract_last``). Either rows entirely
    absent or the env line missing/malformed is a clear error exit with
    NOTHING appended or written -- both checks run before any ledger
    mutation or sidecar write, mirroring ``_finish_hetero36``'s contract.

    Unlike ``_finish_hetero36``, one job covers every task/arm/seed the
    campaign was given, and this round's dedup key is the four-field
    ``(round, task, variant, seed)`` (spec section 6) rather than
    ``(round, seed)`` -- a task's arms and many seeds share one round
    tag, so ``variant`` (the arm) must be in the key. Rows are bucketed by
    their own ``task`` field via ``_row_task`` (a row that isn't a dict, or
    is a dict with no non-empty string ``task``, can't be attributed to any
    task; it's dropped with a stderr warning, never appended anywhere --
    see the module docstring's "benchmarks" job-mode section). Each task's
    bucket is independently
    dedup-appended to the ledger (see ``_append_benchmarks_rows``) and
    written to its own ``experiments/bench/bench_<task>_env.json`` sidecar
    (one per task present in the parsed rows), since dedup keys never cross
    task boundaries (each round tag names exactly one task) -- resolving
    per-task bucket is equivalent to one global pass over all rows, up to
    ordering. Each task's OWN row order is preserved (log order within the
    bucket), but the ledger's cross-task ordering is bucket order, not the
    original interleaved log order -- spec section 4 states merges across
    tasks are order-independent (tasks may even run as separate parallel
    jobs), so only membership and per-task relative order are guaranteed,
    unlike ``_finish_hetero36``'s single round-agnostic global pass.

    ``merge_sidecar`` (default ``False``, Task 6 hardening for ``--shards``):
    when ``False`` (every non-sharded call, including the single-job
    ``--job benchmarks`` path -- unchanged, byte-identical to before this
    flag existed), each task's sidecar is written wholesale from THIS
    call's own rows only, exactly as always. When ``True`` (every call from
    ``_run_sharded_benchmarks``), each task's sidecar is instead merged onto
    whatever a prior shard already wrote for that task (see
    ``_read_benchmarks_sidecar``/``_merge_benchmarks_sidecar``) before being
    written whole -- otherwise a multi-shard round's sidecar would silently
    retain only the LAST-finishing shard's rows/timings, covering a
    fraction of the round.
    """
    logs, exc = _fetch_job_logs(job)
    if exc is not None:
        print(f"error: could not fetch logs: {exc}", file=sys.stderr)
        return 1
    assert logs is not None  # guaranteed by _fetch_job_logs's exc is None contract

    rows = _extract_all(_BENCHMARKS_ROW_PREFIX, logs)
    if not rows:
        print(
            "error: no well-formed MINGRU_LAB_ROW lines found in job logs -- "
            "not appending or writing a sidecar. Log tail:\n" + "\n".join(logs.splitlines()[-40:]),
            file=sys.stderr,
        )
        return 1

    env = _extract_last(_BENCHMARKS_ENV_PREFIX, logs)
    if env is None:
        print(
            "error: no well-formed MINGRU_LAB_ENV line found in job logs -- "
            "not appending or writing a sidecar. Log tail:\n" + "\n".join(logs.splitlines()[-40:]),
            file=sys.stderr,
        )
        return 1

    tasks_present = sorted({t for row in rows if (t := _row_task(row)) is not None})
    unattributed = [row for row in rows if _row_task(row) is None]
    if unattributed:
        print(
            f"warning: {len(unattributed)} MINGRU_LAB_ROW row(s) had no attributable "
            "non-empty string 'task' field (non-dict payload, or a dict with a "
            "missing/empty 'task') -- skipped entirely, not appended to any task's "
            "ledger rows or sidecar",
            file=sys.stderr,
        )

    _DELTA_PROBE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task in tasks_present:
        task_rows = [row for row in rows if _row_task(row) == task]
        result = _append_benchmarks_rows(_LEDGER_PATH, task_rows, _BENCHMARKS_ROUNDS)
        if result.skipped_invalid:
            print(
                f"warning: task {task}: skipped {result.skipped_invalid} shape-invalid "
                "MINGRU_LAB_ROW row(s) (unrecognized round, non-string variant, or "
                "non-int seed) -- not appended to the ledger",
                file=sys.stderr,
            )
        sidecar = _build_benchmarks_sidecar(env, task, task_rows, result)
        sidecar_path = _bench_sidecar_path(task)
        if merge_sidecar:
            sidecar = _merge_benchmarks_sidecar(_read_benchmarks_sidecar(sidecar_path), sidecar)
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
        print(
            f"[{task}] appended {result.appended} new row(s), skipped "
            f"{result.skipped_duplicate} duplicate(s), {result.skipped_invalid} invalid, "
            f"and {result.deduped_in_batch} intra-batch duplicate(s); wrote {sidecar_path}"
        )
    return 0 if ok else 1


class _PreflightFailure(Exception):
    """Raised by ``_resolve_submission_context`` when a submission prereq fails.

    The error message is already printed to stderr at the raise site (same
    text the single-job path always printed); this carries only the exit
    code so both call sites (single-job and ``--shards``) can ``return`` it
    without re-deriving or re-printing anything.
    """

    def __init__(self, exit_code: int) -> None:
        super().__init__(f"preflight check failed (exit {exit_code})")
        self.exit_code = exit_code


def _resolve_submission_context(args: argparse.Namespace) -> tuple[Any, str | None, str]:
    """Validate teamspace/credentials and resolve ``(machine, owner, ts_name)``.

    Shared by the single-job and ``--shards`` sharded submission paths
    (Task 6): teamspace presence, ``LIGHTNING_USER_ID``/``LIGHTNING_API_KEY``
    credentials, ``--machine`` name resolution against the SDK's ``Machine``
    enum, and the ``"owner/teamspace"`` splitting were previously inlined
    once directly in ``main`` -- pulled out here rather than duplicated a
    second time for the sharded path, which would otherwise drift the two
    copies apart the way ad hoc duplication always does. Raises
    ``_PreflightFailure`` (message already printed to stderr) rather than
    returning a mixed success/error shape.
    """
    if not args.teamspace:
        print(
            "error: no teamspace (set MINGRU_LIGHTNING_TEAMSPACE or pass --teamspace)",
            file=sys.stderr,
        )
        raise _PreflightFailure(2)
    for var in ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY"):
        if not os.environ.get(var):
            print(f"error: {var} not set (lightning.ai Settings -> Keys)", file=sys.stderr)
            raise _PreflightFailure(2)

    from lightning_sdk import Machine

    try:
        machine = getattr(Machine, args.machine)
    except AttributeError:
        names = [m for m in dir(Machine) if not m.startswith("_")]
        print(f"error: unknown machine {args.machine!r}; available: {names}", file=sys.stderr)
        raise _PreflightFailure(2) from None

    # The SDK takes the teamspace NAME plus a separate org=/user= owner kwarg;
    # our config uses the UI's "owner/teamspace" form. Split it and try the
    # owner as an org first, then as a user.
    if "/" in args.teamspace:
        owner, ts_name = args.teamspace.split("/", 1)
    else:
        owner, ts_name = None, args.teamspace
    return machine, owner, ts_name


def _submit_job(
    args: argparse.Namespace,
    machine: Any,
    owner: str | None,
    ts_name: str,
    name: str,
    command: str,
) -> Any:
    """Submit one Lightning job (studio or image mode); return its handle.

    Factored out of ``main`` so ``--shards`` (Task 6) can submit several
    jobs -- one per disjoint seed slice -- through the exact same studio-
    vs-image and org/user teamspace-owner-fallback logic the single-job
    path already used, parameterized by ``name``/``command`` per call
    rather than duplicated per shard. Raises ``RuntimeError`` (never prints
    or exits itself) when both the org and user teamspace-owner attempts
    fail, so every call site -- single-job or per-shard -- reports the
    failure in its own existing style.
    """
    from lightning_sdk import Job, Studio

    if args.studio:
        # Explicit if/else rather than `**({"org": owner} if owner else {})`
        # (the original inline call's shape): identical runtime behavior
        # (org is passed only when `owner` is truthy), but avoids a pyright
        # false-positive -- once `owner` carries an explicit `str | None`
        # parameter annotation (needed here since this is now a shared
        # helper, not inlined in `main`), pyright can no longer special-case
        # the literal dict's ``"org"`` key against `Studio.__init__`'s
        # `org` parameter specifically, and instead validates the unpacked
        # dict's value type against every other keyword parameter in the
        # signature too (including the unrelated bool-typed `create_ok`/
        # `disable_secrets`), reporting a spurious `reportArgumentType`.
        if owner:
            studio = Studio(args.studio, teamspace=ts_name, org=owner)
        else:
            studio = Studio(args.studio, teamspace=ts_name)
        return studio.run_job(
            name=name, machine=machine, command=command, interruptible=args.interruptible
        )

    def _run(**owner_kw):
        return Job.run(
            name=name,
            teamspace=ts_name,
            machine=machine,
            image=args.image,
            command=command,
            interruptible=args.interruptible,
            **owner_kw,
        )

    if owner is None:
        return _run()
    try:
        return _run(org=owner)
    except Exception as org_exc:
        try:
            return _run(user=owner)
        except Exception as user_exc:
            raise RuntimeError(
                f"teamspace resolution failed as org ({org_exc}) and as user ({user_exc})"
            ) from user_exc


def _run_sharded_benchmarks(
    args: argparse.Namespace,
    machine: Any,
    owner: str | None,
    ts_name: str,
    commands: list[str],
    job_name_prefix: str,
    ref: str,
) -> int:
    """Submit ``len(commands)`` benchmarks jobs concurrently, wait/collect sequentially.

    Task 6 (``--shards``): submission is concurrent -- every job is
    submitted via ``_submit_job`` in this loop, before any ``job.wait()``,
    so no shard's submission blocks another's -- but waiting and per-job
    collection stay strictly sequential ("submission concurrent, waiting
    sequential is acceptable"). Each shard's rows are independently
    dedup-appended through the exact same ``_finish_benchmarks`` path the
    single-job run uses, so retrying any subset of shards stays idempotent
    the same way retrying a single job already does. Calls
    ``_finish_benchmarks(..., merge_sidecar=True)`` for every shard (Task 6
    hardening): each shard's own per-task sidecar is merged onto whatever an
    earlier shard already wrote for that task, rather than clobbering it, so
    the sidecar left on disk after the last shard reflects the WHOLE round's
    ``per_variant_seed_wall_secs`` and row counts, not just the
    last-finishing shard's slice (see ``_merge_benchmarks_sidecar``). A
    shard that fails to even submit is reported and treated as a failed
    shard (not a crash); the remaining shards still submit and run.
    ``KeyboardInterrupt`` is handled in BOTH windows this function can be
    interrupted in, not just the wait loop: during submission, every
    already-submitted shard (i.e. every non-``None`` entry already in
    ``jobs``) is stopped before re-raising; during waiting, the in-flight
    shard AND every not-yet-waited shard are stopped before re-raising --
    either way mirroring the single-job path's "avoid orphaned billing"
    contract across all concurrently-running shards, not just the one
    directly being awaited.
    """
    jobs: list[Any] = []
    try:
        for i, command in enumerate(commands):
            name = f"{job_name_prefix}-{ref[:7]}-shard{i}of{len(commands)}"
            try:
                job = _submit_job(args, machine, owner, ts_name, name, command)
            except RuntimeError as exc:
                print(f"error: shard {i}: {exc}", file=sys.stderr)
                jobs.append(None)
                continue
            print(f"submitted shard {i}: {job.name}")
            jobs.append(job)
    except KeyboardInterrupt:
        print(
            "interrupted during submission -- stopping every already-submitted "
            "shard to avoid orphaned billing",
            file=sys.stderr,
        )
        for submitted in jobs:
            if submitted is not None:
                submitted.stop()
        raise

    overall_rc = 0
    for i, job in enumerate(jobs):
        if job is None:
            overall_rc = 2
            continue
        print(f"waiting on shard {i}: {job.name}...")
        try:
            job.wait()
        except KeyboardInterrupt:
            print(
                "interrupted -- stopping this and remaining shards to avoid orphaned billing",
                file=sys.stderr,
            )
            job.stop()
            for remaining in jobs[i + 1 :]:
                if remaining is not None:
                    remaining.stop()
            raise
        status = str(job.status)
        print(f"shard {i} status: {status}")
        ok = "completed" in status.lower() or "succe" in status.lower()
        rc = _finish_benchmarks(job, ok, merge_sidecar=True)
        if rc != 0:
            overall_rc = rc
    return overall_rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--machine", default=os.environ.get("MINGRU_GPU_MACHINE", DEFAULT_MACHINE))
    ap.add_argument("--image", default=os.environ.get("MINGRU_GPU_IMAGE", DEFAULT_IMAGE))
    ap.add_argument("--ref", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--teamspace", default=os.environ.get("MINGRU_LIGHTNING_TEAMSPACE"))
    ap.add_argument(
        "--studio",
        default=os.environ.get("MINGRU_LIGHTNING_STUDIO"),
        help="run via Studio.run_job (the studio's warm machine pool + env snapshot) "
        "instead of a container-image batch job -- attaches much faster",
    )
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--interruptible", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--job",
        choices=["check", "delta-probe", "hetero36", "benchmarks"],
        default="check",
        help="job mode: 'check' (default, existing behavior unchanged), "
        "'delta-probe' (Task 6 CUDA fusion-headroom probe), 'hetero36' "
        "(Task 3 GPU 36-seed round campaign), or 'benchmarks' (Task 6 "
        "accepted-benchmark validation round) -- see the module docstring's "
        "'Job modes' section",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="'benchmarks' job mode only: task subset passthrough to "
        "gpu_benchmark_campaign.py's --tasks (default: its own default, all four)",
    )
    ap.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="'benchmarks' job mode only: arm subset passthrough to "
        "gpu_benchmark_campaign.py's --arms (default: its own default, every arm)",
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="'benchmarks' job mode only: seed subset passthrough to "
        "gpu_benchmark_campaign.py's --seeds (default: each task's own seed-matrix size)",
    )
    ap.add_argument(
        "--shards",
        type=int,
        default=1,
        help="'benchmarks' job mode only: submit this many jobs concurrently, "
        "each running every requested arm against a disjoint slice of the "
        "seed pool (the task's own seed count, or --seeds if given), then "
        "wait/collect each sequentially; default 1 (unsharded, existing "
        "single-job behavior)",
    )
    args = ap.parse_args()

    if args.shards < 1:
        print(f"error: --shards must be >= 1, got {args.shards}", file=sys.stderr)
        return 2
    if args.shards > 1 and args.job != "benchmarks":
        print("error: --shards > 1 is only supported for --job benchmarks", file=sys.stderr)
        return 2

    ref = args.ref or _sh(["git", "rev-parse", "HEAD"])
    repo = args.repo or _sh(["git", "remote", "get-url", "origin"])

    # `--shards` (Task 6) branches into its own fully self-contained
    # workflow -- command building, spec printing, dry-run, preflight, and
    # submission -- with its own `return`, rather than sharing a
    # `command`/`spec` binding with the single-job path below via a second,
    # separately-re-checked conditional. Pyright's possibly-unbound check
    # (correctly) can't prove two independent `if` guards on different
    # variables stay correlated; one guard with an early return sidesteps
    # the ambiguity entirely and keeps the unsharded path exactly as it
    # was (unchanged control flow, byte-identical output).
    if args.job == "benchmarks" and args.shards > 1:
        job_name_prefix = _BENCHMARKS_JOB_NAME
        try:
            pool = _resolve_shard_seed_pool(args.tasks, args.seeds)
            shard_commands = [
                build_benchmarks_command(repo, ref, args.tasks, args.arms, shard_seeds)
                for shard_seeds in _shard_seed_lists(pool, args.shards)
            ]
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        specs = [
            {
                "name": f"{job_name_prefix}-{ref[:7]}-shard{i}of{len(shard_commands)}",
                "teamspace": args.teamspace,
                "machine": args.machine,
                "mode": f"studio:{args.studio}" if args.studio else f"image:{args.image}",
                "interruptible": args.interruptible,
                "command": shard_command,
            }
            for i, shard_command in enumerate(shard_commands)
        ]
        for i, shard_spec in enumerate(specs):
            print(f"--- shard {i} ---")
            for k, v in shard_spec.items():
                print(f"{k}: {v}")
        if args.dry_run:
            print("\n--dry-run: not submitting.")
            return 0

        try:
            machine, owner, ts_name = _resolve_submission_context(args)
        except _PreflightFailure as exc:
            return exc.exit_code
        return _run_sharded_benchmarks(
            args, machine, owner, ts_name, shard_commands, job_name_prefix, ref
        )

    # --- single-job path (unsharded): unchanged behavior ---
    if args.job == "delta-probe":
        command = build_delta_probe_command(repo, ref)
        job_name_prefix = _DELTA_PROBE_JOB_NAME
    elif args.job == "hetero36":
        command = build_hetero36_command(repo, ref)
        job_name_prefix = _HETERO36_JOB_NAME
    elif args.job == "benchmarks":
        command = build_benchmarks_command(repo, ref, args.tasks, args.arms, args.seeds)
        job_name_prefix = _BENCHMARKS_JOB_NAME
    else:
        command = build_command(repo, ref, args.bench)
        job_name_prefix = JOB_NAME

    spec = {
        "name": f"{job_name_prefix}-{ref[:7]}",
        "teamspace": args.teamspace,
        "machine": args.machine,
        "mode": f"studio:{args.studio}" if args.studio else f"image:{args.image}",
        "interruptible": args.interruptible,
        "command": command,
    }
    for k, v in spec.items():
        print(f"{k}: {v}")

    if args.dry_run:
        print("\n--dry-run: not submitting.")
        return 0

    try:
        machine, owner, ts_name = _resolve_submission_context(args)
    except _PreflightFailure as exc:
        return exc.exit_code

    try:
        job = _submit_job(args, machine, owner, ts_name, spec["name"], command)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"submitted: {job.name}; waiting...")
    try:
        job.wait()
    except KeyboardInterrupt:
        print("interrupted -- stopping job to avoid orphaned billing", file=sys.stderr)
        job.stop()
        raise
    status = str(job.status)
    print(f"status: {status}")
    ok = "completed" in status.lower() or "succe" in status.lower()
    if args.job == "delta-probe":
        return _finish_delta_probe(job, ok)
    if args.job == "hetero36":
        return _finish_hetero36(job, ok)
    if args.job == "benchmarks":
        return _finish_benchmarks(job, ok)
    if not ok:
        logs, exc = _fetch_job_logs(job)
        if exc is not None:
            print(f"(could not fetch logs: {exc})", file=sys.stderr)
        else:
            print(logs)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
