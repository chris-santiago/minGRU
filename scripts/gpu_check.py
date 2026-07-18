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
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
DEFAULT_MACHINE = "L4"
JOB_NAME = "mingru-gpu-check"

# --- delta-probe job mode (Task 6) --------------------------------------

_DELTA_PROBE_JOB_NAME = "mingru-gpu-delta-probe"
_DELTA_PROBE_RESULT_PREFIX = "MINGRU_GPU_PROBE_RESULT "
_REPO_ROOT = Path(__file__).resolve().parent.parent
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


def _existing_round_seed_pairs(ledger_path: Path, rounds: tuple[str, ...]) -> set[tuple[str, int]]:
    """(round, seed) pairs already present in the local ledger for ``rounds``.

    Guarded per-line parse (a malformed ledger line is skipped, never
    raised) mirrors the extraction contract this module already applies to
    job logs; shape-invalid lines (non-dict JSON, or a dict that fails
    ``_valid_hetero36_key``) are skipped the same way -- a garbage line
    already in the ledger must never crash every future run that reads it
    back. A ledger that doesn't exist yet (fresh checkout, or a
    ``tmp_path`` fixture in tests) is treated as "no existing rows".
    """
    pairs: set[tuple[str, int]] = set()
    if not ledger_path.exists():
        return pairs
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        key = _valid_hetero36_key(row, rounds)
        if key is not None:
            pairs.add(key)
    return pairs


def _dedup_batch_last_wins(
    rows: list[Any], rounds: tuple[str, ...]
) -> tuple[list[dict[str, Any]], int, int]:
    """Resolve one extracted batch to last-N-wins per ``(round, seed)``.

    Spec §6: "extraction is last-N-wins per (round, seed)". Returns
    ``(deduped_rows, rows_skipped_invalid, rows_deduped_in_batch)``: a
    shape-invalid row (see ``_valid_hetero36_key``) never participates --
    it's dropped and counted rather than crashing or silently entering
    the ledger under a garbage key. Valid rows are folded into a dict
    keyed by ``(round, seed)`` IN LOG ORDER: a plain ``dict``'s iteration
    order is the position of a key's FIRST occurrence, but assigning to
    an already-present key still overwrites its value -- so
    ``dict.values()`` afterward yields exactly one row per key, in
    first-occurrence order, holding each key's LAST-seen row.
    ``rows_deduped_in_batch`` counts every row that LOST that
    overwrite (one less than the number of rows sharing a key) -- without
    it, a losing row would vanish from the sidecar's accounting: the
    caller's ``len(rows) == appended + skipped_duplicate + skipped_invalid
    + deduped_in_batch`` reconciliation depends on this count existing.
    """
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    skipped_invalid = 0
    deduped_in_batch = 0
    for row in rows:
        key = _valid_hetero36_key(row, rounds)
        if key is None:
            skipped_invalid += 1
            continue
        if key in by_key:
            deduped_in_batch += 1
        by_key[key] = row
    return list(by_key.values()), skipped_invalid, deduped_in_batch


class _AppendResult(NamedTuple):
    appended: int
    skipped_duplicate: int
    skipped_invalid: int
    deduped_in_batch: int


def _append_new_rows(ledger_path: Path, rows: list[Any], rounds: tuple[str, ...]) -> _AppendResult:
    """Append ``rows`` whose ``(round, seed)`` isn't already in the ledger.

    Shape-invalid rows (see ``_valid_hetero36_key``) are dropped and
    counted rather than appended or crashing the batch. The surviving
    batch is first resolved last-N-wins per ``(round, seed)`` (spec §6,
    see ``_dedup_batch_last_wins``); only then is each remaining row
    checked against the ledger's existing keys for ``rounds`` -- a key
    already present in the ledger still skips unconditionally, keeping
    retries idempotent. Log order (of the deduped batch) is preserved in
    both the returned counts and the appended lines. The four returned
    counts reconcile exactly against ``len(rows)``: every input row is
    either appended, skipped as an existing-ledger duplicate, skipped as
    shape-invalid, or deduped away by a later same-batch occurrence.
    """
    batch, skipped_invalid, deduped_in_batch = _dedup_batch_last_wins(rows, rounds)
    existing = _existing_round_seed_pairs(ledger_path, rounds)
    appended_lines: list[str] = []
    appended = 0
    skipped_duplicate = 0
    for row in batch:
        key = (row["round"], row["seed"])
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
        choices=["check", "delta-probe", "hetero36"],
        default="check",
        help="job mode: 'check' (default, existing behavior unchanged), "
        "'delta-probe' (Task 6 CUDA fusion-headroom probe), or 'hetero36' "
        "(Task 3 GPU 36-seed round campaign) -- see the module docstring's "
        "'Job modes' section",
    )
    args = ap.parse_args()

    ref = args.ref or _sh(["git", "rev-parse", "HEAD"])
    repo = args.repo or _sh(["git", "remote", "get-url", "origin"])
    if args.job == "delta-probe":
        command = build_delta_probe_command(repo, ref)
        job_name_prefix = _DELTA_PROBE_JOB_NAME
    elif args.job == "hetero36":
        command = build_hetero36_command(repo, ref)
        job_name_prefix = _HETERO36_JOB_NAME
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

    if not args.teamspace:
        print(
            "error: no teamspace (set MINGRU_LIGHTNING_TEAMSPACE or pass --teamspace)",
            file=sys.stderr,
        )
        return 2
    for var in ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY"):
        if not os.environ.get(var):
            print(f"error: {var} not set (lightning.ai Settings -> Keys)", file=sys.stderr)
            return 2

    from lightning_sdk import Job, Machine, Studio

    try:
        machine = getattr(Machine, args.machine)
    except AttributeError:
        names = [m for m in dir(Machine) if not m.startswith("_")]
        print(f"error: unknown machine {args.machine!r}; available: {names}", file=sys.stderr)
        return 2

    # The SDK takes the teamspace NAME plus a separate org=/user= owner kwarg;
    # our config uses the UI's "owner/teamspace" form. Split it and try the
    # owner as an org first, then as a user.
    if "/" in args.teamspace:
        owner, ts_name = args.teamspace.split("/", 1)
    else:
        owner, ts_name = None, args.teamspace

    def _run(**owner_kw):
        return Job.run(
            name=spec["name"],
            teamspace=ts_name,
            machine=machine,
            image=args.image,
            command=command,
            interruptible=args.interruptible,
            **owner_kw,
        )

    if args.studio:
        studio = Studio(
            args.studio,
            teamspace=ts_name,
            **({"org": owner} if owner else {}),
        )
        job = studio.run_job(
            name=spec["name"],
            machine=machine,
            command=command,
            interruptible=args.interruptible,
        )
    elif owner is None:
        job = _run()
    else:
        try:
            job = _run(org=owner)
        except Exception as org_exc:
            try:
                job = _run(user=owner)
            except Exception as user_exc:
                print(f"error: teamspace resolution failed as org ({org_exc}) "
                      f"and as user ({user_exc})", file=sys.stderr)
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
    if not ok:
        logs, exc = _fetch_job_logs(job)
        if exc is not None:
            print(f"(could not fetch logs: {exc})", file=sys.stderr)
        else:
            print(logs)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
