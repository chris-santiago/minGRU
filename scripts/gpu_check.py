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
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
DEFAULT_MACHINE = "L4"
JOB_NAME = "mingru-gpu-check"


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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--machine", default=os.environ.get("MINGRU_GPU_MACHINE", DEFAULT_MACHINE))
    ap.add_argument("--image", default=os.environ.get("MINGRU_GPU_IMAGE", DEFAULT_IMAGE))
    ap.add_argument("--ref", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--teamspace", default=os.environ.get("MINGRU_LIGHTNING_TEAMSPACE"))
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--interruptible", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ref = args.ref or _sh(["git", "rev-parse", "HEAD"])
    repo = args.repo or _sh(["git", "remote", "get-url", "origin"])
    command = build_command(repo, ref, args.bench)

    spec = {
        "name": f"{JOB_NAME}-{ref[:7]}",
        "teamspace": args.teamspace,
        "machine": args.machine,
        "image": args.image,
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

    from lightning_sdk import Job, Machine

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

    if owner is None:
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
    if not ok:
        try:
            print(job.logs)
        except Exception as exc:  # log retrieval is best-effort
            print(f"(could not fetch logs: {exc})", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
