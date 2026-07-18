"""Matched-state round: pooled seed runner + verdict-table stats (Task 2).

Spec: ``.claude/output/specs/2026-07-18-matched-state-round-design.md``
section 4 ("A runner script executes the 24 seed runs as subprocesses
through a worker pool"), section 6 (cost sidecar + verdict-table
contracts), section 7 (invariants), and acceptance criteria 3-4. Intent
ledger statements 1, 2, 5, 6: this is the tool that produces the round's
24-seed evidence under the torch==2.5.1 pin and computes the verdict
table's statistics FROM ledger rows, never by hand-transcription.

Two subcommands:

``run``
    Executes the 2x12 seed matrix -- rounds ``hetero-loop-20-pd64``
    (model ``hetero-pd64``) and ``hetero-loop-21-pd1024`` (model
    ``hetero-pd1024``), seeds 0-11 -- as child processes through a
    worker pool (default 6 workers). Each child is::

        uv run --no-project --with torch==2.5.1 python \
            experiments/hetero_lab.py --round <round> --model <model> \
            --seed <s> --steps 1600 --require-torch 2.5.1

    wrapped in ``/usr/bin/time -l`` (macOS) to capture peak RSS from its
    stderr report, with ``OMP_NUM_THREADS=1``/``MKL_NUM_THREADS=1`` set
    in the child's environment (1 torch thread per pool worker, per spec
    section 7's cost-evidence-separation invariant). Before spawning the
    pool, the runner first runs the lab's ``--selftest`` (the delta
    bridge gate) as its own child and refuses to start the seed matrix
    if it fails. Per-child wall time (labeled *contended*, not evidence
    -- the pool runs children concurrently) and peak RSS are written to
    the cost sidecar (``experiments/bench/matched_state_cost.json``,
    spec section 6 schema); ``hetero_lab.py`` remains the only writer of
    ``experiments/lab_results.jsonl`` -- this runner only shells out to
    it and never computes or appends a row itself.

``report``
    Reads ``experiments/lab_results.jsonl``, selects rows by round name
    (the recorded givens arm ``hetero-loop-17-sg8`` plus the two new
    arms this round's ``run`` subcommand produces), and prints the spec
    section 6 / TECHNICAL_REPORT section 4.4 verdict table as markdown:
    fit counts, mean acc@{64,256,512,1024}, fit-only acc@{512,1024},
    threshold-robustness at {0.98, 0.99, 0.995}, two-sided Fisher exact
    (exact hypergeometric enumeration, stdlib ``math.comb`` -- no scipy)
    for each new arm against the recorded givens arm, and composer
    parameter counts computed arithmetically from the ``nn.Linear``
    in/out/bias formulas (see ``_givens_composer_params``/
    ``_delta_composer_params``) rather than hardcoded. A round with zero
    ledger rows (e.g. before ``run`` has produced any) renders as an
    explicit "0 rows" line, not a crash.

This script is stdlib-only (no torch import): it only ever shells out to
``experiments/hetero_lab.py`` for anything that needs torch, so no
``src/``-onto-``sys.path`` bootstrap or ruff E402 exemption is needed
here (unlike ``scripts/bench_delta.py``/``scripts/scaling_probe.py``,
which import the packaged ``mingru`` distribution directly).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HETERO_LAB = _REPO_ROOT / "experiments" / "hetero_lab.py"
_RESULTS = _REPO_ROOT / "experiments" / "lab_results.jsonl"
_SIDECAR = _REPO_ROOT / "experiments" / "bench" / "matched_state_cost.json"

_EVIDENCE_PIN_TORCH_VERSION = "2.5.1"
_DEFAULT_STEPS = 1600
_DEFAULT_WORKERS = 6
_ALL_SEEDS = tuple(range(12))
_FIT_THRESHOLD = 0.99
_ROBUSTNESS_THRESHOLDS = (0.98, 0.99, 0.995)
_ACC_LENGTHS = (64, 256, 512, 1024)
_FIT_ONLY_LENGTHS = (512, 1024)

_RECORDED_GIVENS_ROUND = "hetero-loop-17-sg8"


@dataclass(frozen=True)
class _Arm:
    """One matched-state arm: ledger round name + lab ``--model`` value."""

    label: str
    round: str
    model: str


_ARMS: tuple[_Arm, ...] = (
    _Arm("delta@64-matched", "hetero-loop-20-pd64", "hetero-pd64"),
    _Arm("delta@1024", "hetero-loop-21-pd1024", "hetero-pd1024"),
)


# --- run: child invocation ---------------------------------------------


def _uv_child_argv(round_: str, model: str, seed: int, steps: int, dry_run: bool) -> list[str]:
    """The exact pinned invocation shape (spec section 4 / task brief)."""
    argv = [
        "uv",
        "run",
        "--no-project",
        "--with",
        f"torch=={_EVIDENCE_PIN_TORCH_VERSION}",
        "python",
        str(_HETERO_LAB),
        "--round",
        round_,
        "--model",
        model,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--require-torch",
        _EVIDENCE_PIN_TORCH_VERSION,
    ]
    if dry_run:
        argv.append("--dry-run")
    return argv


def _selftest_argv() -> list[str]:
    return [
        "uv",
        "run",
        "--no-project",
        "--with",
        f"torch=={_EVIDENCE_PIN_TORCH_VERSION}",
        "python",
        str(_HETERO_LAB),
        "--selftest",
    ]


def _run_selftest_gate() -> bool:
    """Run the lab's ``--selftest`` (delta bridge included) as a child.

    The runner refuses to start the seed matrix if this fails (spec
    section 4: "refuses to start if the bridge selftest fails") -- a
    failing bridge means the new arms would not be training the function
    the recorded delta arms trained, so no evidence run is worth paying
    for.
    """
    print("=== bridge selftest gate ===")
    try:
        proc = subprocess.run(_selftest_argv(), cwd=_REPO_ROOT, capture_output=True, text=True)
    except (OSError, FileNotFoundError) as exc:
        # Spawn-level failure (e.g. `uv` not on PATH) -- this runs before
        # the pool starts, so there is no partial-run data to lose; a
        # clean SystemExit with the real cause beats a bare traceback.
        raise SystemExit(
            f"FAILED to spawn the selftest child ({' '.join(_selftest_argv())}): "
            f"{exc}. Check that `uv` (and, on macOS, `/usr/bin/time`) are on PATH."
        ) from exc
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    ok = proc.returncode == 0
    print(f"selftest gate: {'PASS' if ok else 'FAIL'} (exit {proc.returncode})")
    return ok


_MAXRSS_RE = re.compile(r"(\d+)\s+maximum resident set size", re.IGNORECASE)


def _parse_peak_rss_bytes(time_stderr: str) -> int | None:
    """Peak RSS in bytes from ``/usr/bin/time -l``'s stderr report.

    macOS reports this field in bytes already (unlike Linux ``time -v``,
    which reports kilobytes) -- spec section 4/7's macOS capture
    convention. Returns ``None`` if the line is missing (e.g. not
    running on macOS, or the child was killed before ``time`` could
    report), which the caller surfaces rather than fabricating a value.
    """
    match = _MAXRSS_RE.search(time_stderr)
    return int(match.group(1)) if match else None


def _timed_child_argv(argv: list[str]) -> list[str]:
    """Wrap ``argv`` in ``/usr/bin/time -l`` for peak-RSS capture.

    Only implemented for macOS (this repo's evidence-pin machine, spec
    section 4's explicit capture convention); on any other platform the
    child runs unwrapped and peak RSS is reported as ``None`` (see
    ``_parse_peak_rss_bytes``) rather than silently guessing a Linux
    equivalent this round never asked for.
    """
    if platform.system() == "Darwin":
        return ["/usr/bin/time", "-l", *argv]
    return argv


@dataclass
class _ChildResult:
    round: str
    seed: int
    wall_secs_contended: float
    peak_rss_bytes: int | None
    exit_code: int
    stdout: str = field(repr=False)
    stderr: str = field(repr=False)


def _run_one_child(arm: _Arm, seed: int, steps: int, dry_run: bool) -> _ChildResult:
    """Run one (round, seed) cell; wall time measured around the whole
    (possibly pool-contended) subprocess call, per spec section 4/7.

    No per-child timeout is enforced here (unlike ``scripts/scaling_probe.py``'s
    per-config wall-clock timeout): a training child's cost is fundamentally
    open-ended (up to the 1600-step budget, not a fixed-shape probe step), so
    a fixed timeout would either be too short for a legitimate slow seed or
    too long to bound anything useful -- accepted as a documented tradeoff,
    not an oversight.

    A spawn-level failure (``OSError``/``FileNotFoundError`` -- e.g. ``uv``
    or ``/usr/bin/time`` missing from PATH) is caught here and converted into
    a normal ``_ChildResult`` (``exit_code=-1``, the error text as
    ``stderr``) rather than propagating out of this pool worker: letting it
    propagate would make ``future.result()`` re-raise in ``_run_cmd``'s
    collection loop, and the ``ThreadPoolExecutor`` context manager's default
    ``shutdown(wait=True)`` would then let every OTHER already-submitted
    training child (up to ~24 CPU-expensive runs) finish before the
    exception surfaces -- while the sidecar write further down is never
    reached, discarding every already-collected ``_ChildResult``. Converting
    to a normal result lets the pool finish and the sidecar always get
    written, with this cell's failure visible in its own row instead.
    """
    argv = _timed_child_argv(_uv_child_argv(arm.round, arm.model, seed, steps, dry_run))
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    start = time.perf_counter()
    try:
        proc = subprocess.run(argv, cwd=_REPO_ROOT, env=env, capture_output=True, text=True)
    except (OSError, FileNotFoundError) as exc:
        wall = time.perf_counter() - start
        return _ChildResult(
            round=arm.round,
            seed=seed,
            wall_secs_contended=round(wall, 3),
            peak_rss_bytes=None,
            exit_code=-1,
            stdout="",
            stderr=f"FAILED to spawn child ({' '.join(argv)}): {exc}",
        )
    wall = time.perf_counter() - start
    return _ChildResult(
        round=arm.round,
        seed=seed,
        wall_secs_contended=round(wall, 3),
        peak_rss_bytes=_parse_peak_rss_bytes(proc.stderr),
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _find_existing_seeds(
    results_path: Path, cells: list[tuple[_Arm, int]]
) -> list[tuple[str, int]]:
    """(round, seed) pairs among ``cells`` that already have a ledger row.

    ``experiments/lab_results.jsonl`` is append-only (spec section 7), so a
    duplicate row cannot be cleaned up after the fact -- this guard runs
    before any non-dry-run pool starts (spec/acceptance criterion 3: "12
    rows per round, no duplicates") so a retry after a partial failure is
    safe by construction: the caller sees exactly which (round, seed) pairs
    already exist and can re-run with ``--seeds`` limited to what's missing.
    """
    by_round = _load_rows_by_round(results_path)
    return [
        (arm.round, seed)
        for arm, seed in cells
        if seed in {row["seed"] for row in by_round.get(arm.round, [])}
    ]


def _run_cmd(args: argparse.Namespace) -> int:
    steps = args.steps if args.steps is not None else _DEFAULT_STEPS
    seeds = sorted(set(args.seeds)) if args.seeds else list(_ALL_SEEDS)

    if not _run_selftest_gate():
        print(
            "FAILED: bridge selftest did not pass -- refusing to start the "
            "seed matrix (no child runs, no sidecar written). Fix the "
            "bridge and re-run.",
            file=sys.stderr,
        )
        return 1

    cells = [(arm, seed) for arm in _ARMS for seed in seeds]

    if not args.dry_run:
        conflicts = _find_existing_seeds(_RESULTS, cells)
        if conflicts:
            conflict_set = set(conflicts)
            safe_seeds = [
                s for s in seeds if not any((arm.round, s) in conflict_set for arm in _ARMS)
            ]
            pairs_str = ", ".join(f"({r}, seed={s})" for r, s in sorted(conflicts))
            safe_seeds_str = (
                " ".join(str(s) for s in safe_seeds)
                if safe_seeds
                else "<none -- every requested seed already has a row for at least one arm>"
            )
            print(
                f"FAILED: {_RESULTS} already has ledger rows for these (round, "
                f"seed) pairs: {pairs_str}. The ledger is append-only, so "
                "duplicates cannot be cleaned up after the fact -- refusing "
                "to start (no child runs, no sidecar written). Re-run with "
                f"`--seeds {safe_seeds_str}` to run only the missing seeds.",
                file=sys.stderr,
            )
            return 1

    print(
        f"\n=== seed matrix: {len(cells)} run(s) across {len(_ARMS)} arm(s) x "
        f"{len(seeds)} seed(s), pool size {args.workers}, steps={steps}, "
        f"dry_run={args.dry_run} ==="
    )

    results: list[_ChildResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_run_one_child, arm, seed, steps, args.dry_run) for arm, seed in cells
        ]
        for future in futures:
            result = future.result()
            results.append(result)
            status = "ok" if result.exit_code == 0 else f"FAILED (exit {result.exit_code})"
            print(
                f"  [{status}] {result.round} seed={result.seed} "
                f"wall={result.wall_secs_contended:.2f}s(contended) "
                f"peak_rss={result.peak_rss_bytes}"
            )
            if result.stdout.strip():
                print(f"    stdout: {result.stdout.strip().splitlines()[-1]}")
            if result.exit_code != 0 and result.stderr.strip():
                tail = result.stderr.strip().splitlines()[-5:]
                print(f"    stderr tail: {tail}", file=sys.stderr)

    n_failed = sum(1 for r in results if r.exit_code != 0)
    if n_failed:
        print(f"\nCONCERN: {n_failed}/{len(results)} child run(s) exited nonzero.")

    if args.dry_run:
        print("\ndry-run: sidecar not written (per --dry-run contract).")
        return 1 if n_failed else 0

    sidecar = {
        "env": {
            "torch": _EVIDENCE_PIN_TORCH_VERSION,
            "num_threads": 1,
            "pool_size": args.workers,
            "platform": platform.platform(),
        },
        "runs": [
            {
                "round": r.round,
                "seed": r.seed,
                "wall_secs_contended": r.wall_secs_contended,
                "peak_rss_bytes": r.peak_rss_bytes,
                "exit_code": r.exit_code,
            }
            for r in results
        ],
    }
    _SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    _SIDECAR.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"\nwrote {_SIDECAR}")
    return 1 if n_failed else 0


# --- report: composer parameter counts ----------------------------------


def _linear_params(in_features: int, out_features: int, bias: bool = True) -> int:
    """``nn.Linear`` parameter count: ``in*out`` weights + ``out`` bias."""
    return in_features * out_features + (out_features if bias else 0)


def _givens_composer_params(
    hidden_size: int = 64, block_size: int = 8, rounds: int = 3, input_size: int = 64
) -> int:
    """``GivensMinGRU`` parameter count, from ``experiments/hetero_lab.py``'s
    ``GivensMinGRU.__init__`` (mirrored by the packaged ``mingru.GivensMinGRU``,
    bit-identical per the lab's bridge selftest): ``linear_theta`` (bias,
    out = n_blocks*rounds*(block_size//2)) + ``linear_z`` + ``linear_h``
    (both ``hidden_size -> hidden_size``, bias) + ``h0`` (``hidden_size``
    free parameters, no weight/bias structure). At the recorded arm's
    config (block_size=8, rounds=3, hidden_size=64) this must reproduce
    the recorded 14,624 (TECHNICAL_REPORT section 4.4); callers should
    treat any drift from that value as a discrepancy to report, not to
    silently accept.
    """
    n_blocks = hidden_size // block_size
    half = block_size // 2
    theta_out = n_blocks * rounds * half
    return (
        _linear_params(input_size, theta_out)  # linear_theta
        + 2 * _linear_params(input_size, hidden_size)  # linear_z, linear_h
        + hidden_size  # h0
    )


def _delta_composer_params(
    n_heads: int, nh: int, d_k: int, d_v: int, input_size: int = 64, hidden_size: int = 64
) -> int:
    """``DeltaMinGRU`` parameter count, from ``mingru.min_gru.DeltaMinGRU
    .__init__``: ``linear_q`` + ``nh``-length ``linear_k``/``linear_v``/
    ``linear_beta`` ModuleLists (each linear per micro-step, bias) +
    ``out_proj``. At the recorded delta@64 config (n_heads=1, nh=2,
    d_k=d_v=8) this must reproduce the recorded 3,306 (TECHNICAL_REPORT
    section 4.4 / next-round design doc); callers should treat any drift
    from that value as a discrepancy to report, not to silently accept.
    """
    return (
        _linear_params(input_size, n_heads * d_k)  # linear_q
        + nh * _linear_params(input_size, n_heads * d_k)  # linear_k[j]
        + nh * _linear_params(input_size, n_heads * d_v)  # linear_v[j]
        + nh * _linear_params(input_size, n_heads)  # linear_beta[j]
        + _linear_params(n_heads * d_v, hidden_size)  # out_proj
    )


# Recorded values this round's arms must reproduce (TECHNICAL_REPORT
# section 4.4); computed here, never hardcoded into the table -- see
# ``_param_discrepancy_notes``.
_RECORDED_GIVENS_PARAMS = 14_624
_RECORDED_DELTA64_PARAMS = 3_306


def _param_discrepancy_notes() -> list[str]:
    """Flag (not fudge) any drift between the computed formulas above and
    the recorded literature values they are supposed to reproduce."""
    notes = []
    givens = _givens_composer_params()
    delta64 = _delta_composer_params(n_heads=1, nh=2, d_k=8, d_v=8)
    if givens != _RECORDED_GIVENS_PARAMS:
        notes.append(
            f"DISCREPANCY: computed givens@64 composer params = {givens:,}, "
            f"recorded TECHNICAL_REPORT value = {_RECORDED_GIVENS_PARAMS:,}"
        )
    if delta64 != _RECORDED_DELTA64_PARAMS:
        notes.append(
            f"DISCREPANCY: computed delta@64 composer params = {delta64:,}, "
            f"recorded value = {_RECORDED_DELTA64_PARAMS:,}"
        )
    return notes


# --- report: Fisher exact (stdlib, exact hypergeometric enumeration) ----


def _fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]].

    Exact hypergeometric enumeration (stdlib ``math.comb``, no scipy),
    per the round's Global Constraint. Sums the hypergeometric
    probability of every table sharing the observed table's marginals
    whose probability is <= the observed table's probability (the
    standard two-sided definition; a small relative tolerance absorbs
    floating-point rounding at the boundary).
    """
    row1, row2 = a + b, c + d
    col1 = a + c
    n = row1 + row2
    denom = math.comb(n, col1)

    def _prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / denom

    lo, hi = max(0, col1 - row2), min(row1, col1)
    p_observed = _prob(a)
    tolerance = p_observed * 1e-7 + 1e-12
    return sum(_prob(x) for x in range(lo, hi + 1) if _prob(x) <= p_observed + tolerance)


# --- report: ledger stats -------------------------------------------------


def _load_rows_by_round(results_path: Path) -> dict[str, list[dict[str, Any]]]:
    by_round: dict[str, list[dict[str, Any]]] = {}
    if not results_path.exists():
        return by_round
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_round.setdefault(row["round"], []).append(row)
    return by_round


@dataclass
class _ArmStats:
    seeds: int
    fits: int
    mean_acc: dict[int, float | None]
    fit_only_acc: dict[int, float | None]
    robustness: dict[float, int]


def _arm_stats(rows: list[dict[str, Any]]) -> _ArmStats:
    """Fit counts, mean acc, fit-only acc, threshold-robustness -- all
    computed from ``rows`` (never hand-transcribed), matching
    TECHNICAL_REPORT section 4.4's definitions: a fit is the selected
    checkpoint's val@128 >= threshold; acc@T is the mean over ALL rows;
    fit-only acc@T is the mean over fitting rows only.
    """
    n = len(rows)
    val128s = [row["ckpt"]["val128"] for row in rows]
    fit_rows = [row for row in rows if row["ckpt"]["val128"] >= _FIT_THRESHOLD]

    def _mean_acc(subset: list[dict[str, Any]], t: int) -> float | None:
        if not subset:
            return None
        return sum(row["acc"][str(t)] for row in subset) / len(subset)

    return _ArmStats(
        seeds=n,
        fits=len(fit_rows),
        mean_acc={t: _mean_acc(rows, t) for t in _ACC_LENGTHS},
        fit_only_acc={t: _mean_acc(fit_rows, t) for t in _FIT_ONLY_LENGTHS},
        robustness={th: sum(1 for v in val128s if v >= th) for th in _ROBUSTNESS_THRESHOLDS},
    )


def _fmt_acc(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _fmt_fit_only(stats: _ArmStats) -> str:
    if stats.seeds == 0:
        return "n/a"
    parts = [_fmt_acc(stats.fit_only_acc[t]) for t in _FIT_ONLY_LENGTHS]
    return " / ".join(parts)


def _report_cmd(args: argparse.Namespace) -> int:
    by_round = _load_rows_by_round(args.results)

    row_specs = [
        ("givens@64 (recorded, `hetero-loop-17-sg8`)", _RECORDED_GIVENS_ROUND, None),
        ("delta@64-matched (`hetero-loop-20-pd64`)", "hetero-loop-20-pd64", "delta64"),
        ("delta@1024 (`hetero-loop-21-pd1024`)", "hetero-loop-21-pd1024", "delta1024"),
    ]

    lines = [
        f"# Matched-state round: verdict-table stats (computed from `{args.results}`)",
        "",
        "| config | seeds | fits | acc@64 | acc@256 | acc@512 | acc@1024 | fit-only @512/@1024 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    all_stats: dict[str, _ArmStats] = {}
    for label, round_name, key in row_specs:
        rows = by_round.get(round_name, [])
        stats = _arm_stats(rows)
        if key is not None:
            all_stats[key] = stats
        fits_str = f"{stats.fits}/{stats.seeds}" if stats.seeds else "0/0"
        lines.append(
            f"| {label} | {stats.seeds} | {fits_str} | "
            f"{_fmt_acc(stats.mean_acc[64])} | {_fmt_acc(stats.mean_acc[256])} | "
            f"{_fmt_acc(stats.mean_acc[512])} | {_fmt_acc(stats.mean_acc[1024])} | "
            f"{_fmt_fit_only(stats)} |"
        )
        if not rows:
            lines.append(f"  (0 rows found for round `{round_name}`)")

    givens_stats = _arm_stats(by_round.get(_RECORDED_GIVENS_ROUND, []))

    lines += ["", "## Threshold-robustness (fits at {0.98, 0.99, 0.995})", ""]
    lines += ["| config | 0.98 | 0.99 | 0.995 |", "| --- | --- | --- |"]
    for label, round_name, _key in row_specs:
        rows = by_round.get(round_name, [])
        stats = _arm_stats(rows)
        n = stats.seeds
        cells = " | ".join(
            f"{stats.robustness[th]}/{n}" if n else "n/a" for th in _ROBUSTNESS_THRESHOLDS
        )
        lines.append(f"| {label} | {cells} |")

    lines += ["", "## Two-sided Fisher exact vs recorded givens@64", ""]
    for label, _round_name, key in row_specs:
        if key is None:
            continue
        stats = all_stats[key]
        if stats.seeds == 0 or givens_stats.seeds == 0:
            lines.append(f"- {label} vs givens@64: n/a (one arm has 0 rows)")
            continue
        p = _fisher_exact_two_sided(
            stats.fits,
            stats.seeds - stats.fits,
            givens_stats.fits,
            givens_stats.seeds - givens_stats.fits,
        )
        lines.append(
            f"- {label} ({stats.fits}/{stats.seeds}) vs givens@64 "
            f"({givens_stats.fits}/{givens_stats.seeds}): p = {p:.4g}"
        )

    givens_params = _givens_composer_params()
    delta64_params = _delta_composer_params(n_heads=1, nh=2, d_k=8, d_v=8)
    delta1024_params = _delta_composer_params(n_heads=4, nh=2, d_k=16, d_v=16)
    lines += [
        "",
        "## Composer parameter counts (computed arithmetically, not hardcoded)",
        "",
        f"- givens@64 (`block_size=8, rounds=3, hidden_size=64`): {givens_params:,}",
        f"- delta@64-matched (`n_heads=1, nh=2, d_k=d_v=8`): {delta64_params:,}",
        f"- delta@1024 (`n_heads=4, nh=2, d_k=d_v=16`): {delta1024_params:,}",
    ]
    lines += _param_discrepancy_notes() or [
        "- (computed values match the recorded literature figures)"
    ]

    print("\n".join(lines))
    return 0


# --- CLI ------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="execute the pooled 2x12 seed matrix")
    run_p.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help=f"pool size (default {_DEFAULT_WORKERS})",
    )
    run_p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="subset of seeds to run (default: all of 0-11); for smoke/partial use",
    )
    run_p.add_argument(
        "--steps", type=int, default=None, help=f"override steps per run (default {_DEFAULT_STEPS})"
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="pass --dry-run through to the lab (no ledger rows); sidecar not written",
    )
    run_p.set_defaults(func=_run_cmd)

    report_p = sub.add_parser("report", help="print the verdict-table stats as markdown")
    report_p.add_argument(
        "--results",
        type=Path,
        default=_RESULTS,
        help="path to lab_results.jsonl (default: experiments/lab_results.jsonl)",
    )
    report_p.set_defaults(func=_report_cmd)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
