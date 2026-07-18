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

``delta-probe`` (Task 6)
  Runs ``scripts/gpu_delta_probe.py`` instead: the CUDA fusion-headroom
  probe for ``DeltaMinGRU``'s chunked-WY forward (profiles eager vs. its
  matmul-FLOP floor vs. ``torch.compile``, the speedup-worth-it judgment
  for building a Triton chunked-WY kernel -- see that script's module
  docstring and ``.git/sdd/task-6-brief.md``). The job command is
  additionally prefixed with a dash-compatible keepalive heartbeat (the
  probe's five shapes x three arms can run past the Lightning tier's
  10-minute idle auto-shutdown without one). After the job completes,
  this script fetches its logs, extracts the last ``MINGRU_GPU_PROBE_RESULT``
  line, and writes ``experiments/bench/gpu_delta_probe.json``/``.md``
  locally -- the job's container filesystem dies with the job, so stdout
  is the only transport. A missing/malformed result line is a clear error
  exit; no partial artifact is ever written.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def _sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def build_command(repo: str, ref: str, bench: bool) -> str:
    steps = [
        "set -eux",  # job shell is dash: -o pipefail unsupported (no pipes used)
        (
            "python -c 'import torch; assert torch.cuda.is_available(), "
            '"no CUDA device"; print(torch.__version__)\''
        ),
        "(python -c 'import triton' || pip install --no-cache-dir triton)",
        f"git clone --filter=blob:none {repo} /tmp/minGRU",
        f"cd /tmp/minGRU && git checkout --detach {ref}",
        "cd /tmp/minGRU && python scripts/bench_scans.py --check",
    ]
    if bench:
        steps.append("cd /tmp/minGRU && python scripts/bench_scans.py --bench")
    return " && ".join(steps)


def build_delta_probe_command(repo: str, ref: str) -> str:
    """Job-shell command for ``--job delta-probe``: keepalive + clone + probe run.

    Same clone/checkout/triton-install steps as ``build_command`` (Task 6
    brief: ``torch.compile``'s Inductor needs triton on GPU, same as the
    parity suite), plus a dash-compatible keepalive heartbeat prefixed at
    the front -- the user's Lightning tier auto-shuts an idle job down
    after 10 minutes of inactivity; the probe's five shapes x three arms
    (each doing several seconds of CUDA-event-timed warmup/timed steps)
    can run long enough between log-visible progress lines for that to
    fire without one, so the heartbeat prints every ~5 minutes -- and runs
    ``scripts/gpu_delta_probe.py`` instead of the parity suite.
    """
    keepalive = '( while true; do echo "[keepalive] $(date -u)"; sleep 300; done & )'
    steps = [
        "set -eux",  # job shell is dash: -o pipefail unsupported (no pipes used)
        keepalive,
        (
            "python -c 'import torch; assert torch.cuda.is_available(), "
            '"no CUDA device"; print(torch.__version__)\''
        ),
        "(python -c 'import triton' || pip install --no-cache-dir triton)",
        f"git clone --filter=blob:none {repo} /tmp/minGRU",
        f"cd /tmp/minGRU && git checkout --detach {ref}",
        "cd /tmp/minGRU && python scripts/gpu_delta_probe.py",
    ]
    return " && ".join(steps)


def _extract_last(prefix: str, text: str) -> dict[str, Any] | None:
    """Parse the JSON payload of the last well-formed ``prefix``-prefixed line.

    ``text`` is the job's fetched logs, which may interleave the
    keepalive heartbeat, clone/checkout output, and the probe's own
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


def _fmt_probe_cell(value: Any, spec: str = ".4f") -> str:
    """Render a possibly-``None``/non-numeric probe field as a table cell."""
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def _render_delta_probe_markdown(result: dict[str, Any]) -> str:
    env = result.get("env", {})
    lines = [
        "# CUDA fusion-headroom probe: DeltaMinGRU chunked-WY (Task 6)",
        "",
        "Eager chunked-WY forward+backward vs. its matmul-FLOP floor "
        "(approximate, see each row's `floor_method`) vs. `torch.compile`, "
        "per `.git/sdd/task-6-brief.md` / `scripts/gpu_delta_probe.py`. "
        "This is a GPU evidence stratum: nothing below is comparable to "
        "the pinned-CPU rows in `experiments/lab_results.jsonl` / "
        "`EXPERIMENTS.md`.",
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
        "compile median (s) | compile status | headroom (eager/floor) | "
        "compile recovered |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
            f"{_fmt_probe_cell(row.get('headroom_eager_over_floor'), '.2f')} | "
            f"{_fmt_probe_cell(row.get('compile_recovered_fraction'), '.2%')} |"
        )
    lines += [
        "",
        "`floor` rows are an explicit approximation of the dominant "
        "GEMM/triangular-solve contractions only, scaled by the standard "
        "3x fwd-GEMM convention -- see each shape's `floor_method` in the "
        "JSON artifact for the full disclosure. A `compile status` other "
        "than `ok` means Inductor failed on that shape; see the JSON "
        "artifact's `compile_error` for that row.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_delta_probe_artifact(result: dict[str, Any]) -> None:
    _DELTA_PROBE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _DELTA_PROBE_JSON.write_text(json.dumps(result, indent=2) + "\n")
    _DELTA_PROBE_MD.write_text(_render_delta_probe_markdown(result))


def _finish_delta_probe(job: Any, ok: bool) -> int:
    """Post-``job.wait()`` handling for ``--job delta-probe``.

    Fetches the job's logs (best-effort, mirroring the existing
    ``check``-mode log fetch), extracts the last
    ``MINGRU_GPU_PROBE_RESULT`` line, and writes the local artifact.
    A missing/malformed result line is a clear error exit -- never a
    partial artifact write, per the task brief.
    """
    try:
        logs = job.logs
    except Exception as exc:  # log retrieval is best-effort, mirrors check mode
        print(f"error: could not fetch logs: {exc}", file=sys.stderr)
        return 1
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
        choices=["check", "delta-probe"],
        default="check",
        help="job mode: 'check' (default, existing behavior unchanged) or "
        "'delta-probe' (Task 6 CUDA fusion-headroom probe -- see the module "
        "docstring's 'Job modes' section)",
    )
    args = ap.parse_args()

    ref = args.ref or _sh(["git", "rev-parse", "HEAD"])
    repo = args.repo or _sh(["git", "remote", "get-url", "origin"])
    if args.job == "delta-probe":
        command = build_delta_probe_command(repo, ref)
        job_name_prefix = _DELTA_PROBE_JOB_NAME
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
    if not ok:
        try:
            print(job.logs)
        except Exception as exc:  # log retrieval is best-effort
            print(f"(could not fetch logs: {exc})", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
