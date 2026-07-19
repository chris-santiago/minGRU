"""Benchmark round report generator (spec:
``.claude/output/specs/2026-07-19-benchmark-round-design.md``, §4 "Fit and
statistics" / "Reporting", §6 ledger row + stats parameterization
contracts).

Reads ``experiments/lab_results.jsonl`` for the four bench round tags
(``bench-s5-01``, ``bench-mqar-01``, ``bench-psmnist-01``,
``bench-pendulum-01``) and regenerates ``experiments/bench/bench_<task>
.json``/``.md`` per task, whole, every run -- never hand-edited (module
docstring convention shared with every other ``experiments/bench/*``
generator; see ``experiments/index.md``).

Per task, per arm (``experiments.benchmark_lab.ARM_REGISTRY``:
log/signed/rotation/givens/delta):

- fit count + threshold-robustness triple, judged on the task's own
  ``fit_metric``/``fit_threshold``/``fit_direction``/``robustness``
  (read from ``experiments.benchmark_tasks.TASKS``, never hardcoded here
  -- S5/MQAR/psMNIST are "ge" accuracy-style, pendulum is "le" MSE-style).
- generalization accuracy, both raw (mean over ALL rows) and fit-only
  (mean over fitting rows only) -- docs framing rule (CLAUDE.md /
  spec §7): never state generalization accuracy without both.
  ``scripts/_evidence_stats.py``'s ``arm_stats`` cannot compute this
  directly here: it unconditionally reads ``row["acc"][str(t)]`` for its
  own hardcoded ``ACC_LENGTHS``/``FIT_ONLY_LENGTHS`` (64/256/512/1024,
  512/1024), which is the S3 matched-state round's key shape, not this
  round's -- S5's eval keys are ``"T256"``/``"T512"``/``"T1024"``,
  MQAR's are ``"T256_p16"``/``"T256_p32"``, psMNIST's is a single
  ``"test"``, and pendulum's is empty (spec §4: no post-selection
  generalization sweep beyond the fit metric itself). Forcing four
  different key shapes through one hardcoded-key aggregator would be
  more parameter creep than the aggregator's own docstring already
  invites (see its "out of this task's parameterization scope" note in
  ``tests/test_evidence_stats_params.py``) -- so this module computes
  the generalization table directly from each task's own rows (whatever
  keys their ``acc`` dicts actually carry, sorted for determinism) and
  only imports ``fisher_exact_two_sided`` from ``_evidence_stats``,
  which needed no such task-specific shape.
- two-sided Fisher exact vs the ``log`` arm (spec §4/§8 "log as the
  Fisher reference arm"; ``_evidence_stats.fisher_exact_two_sided``, no
  scipy).
- per-arm parameter count: computed by constructing the task's actual
  model (``experiments.benchmark_lab.build_model``) and summing
  ``model.parameters()``, NOT a hand-written arithmetic formula per arm
  (unlike ``_evidence_stats.py``'s existing ``givens_composer_params``/
  ``delta_composer_params``, which count only an isolated composer cell
  at a fixed input/hidden size). This round's model shape genuinely
  varies by task (vocab/head size differs per task) and by arm (the four
  decay-capable arms get extra ``DecayMixin`` parameters on the pendulum
  task, delta does not -- spec §4/``benchmark_lab.py``'s module
  docstring); reproducing that exact conditional shape in three more
  formulas (log/signed/rotation) would risk precisely the drift
  ``_evidence_stats.py``'s own docstring already flags for its two
  existing ones, for no accuracy benefit -- ``build_model`` already is
  the single source of truth for what each arm actually trains. Params
  are reported per arm, never equalized (spec §4/§8/§7 docs framing
  rule).

Completeness readout: which ``(arm, seed)`` cells are present vs the
planned ``range(TaskSpec.seeds)`` matrix (arms from ``ARM_REGISTRY``,
seed counts from ``TASKS``) -- consumed by a later task that verifies the
seed matrices landed in full (task brief).

Usage::

    uv run python scripts/report_benchmarks.py
    uv run python scripts/report_benchmarks.py --results <path> --out-dir <dir>  # tests

Not stdlib-only (unlike ``scripts/run_matched_state.py``/``_bench_env
.py``): computing param counts imports the packaged ``mingru``
distribution through ``experiments.benchmark_lab``, which imports torch.
That's fine here -- ``build_model`` never touches data, a GPU, or
randomness-sensitive training state, so this is a structural count, not a
``torch==2.5.1``-pinned evidence measurement; the project's normal
(unpinned) ``uv run`` environment is what runs this script.
"""

from __future__ import annotations

import argparse
import json
import operator
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch  # env disclosure only (torch.__version__); build_model already needs it

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))  # `experiments.*` (namespace package, no __init__.py)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))  # sibling `_evidence_stats`/`_bench_env` imports

from _bench_env import git_commit_sha  # noqa: E402
from _evidence_stats import fisher_exact_two_sided  # noqa: E402
from experiments.benchmark_lab import ARM_REGISTRY, build_model  # noqa: E402
from experiments.benchmark_tasks import TASKS, TaskSpec  # noqa: E402

_RESULTS = _REPO_ROOT / "experiments" / "lab_results.jsonl"
_BENCH_DIR = _REPO_ROOT / "experiments" / "bench"

# Round tags fixed by the round's Global Constraints (spec §6/§7).
ROUND_TAGS: dict[str, str] = {
    "s5": "bench-s5-01",
    "mqar": "bench-mqar-01",
    "psmnist": "bench-psmnist-01",
    "pendulum": "bench-pendulum-01",
}

# "log as the Fisher reference arm" (spec §4/§8): a family-validation round
# contrasts every arm against the vanilla minGRU baseline.
FISHER_REFERENCE_ARM = "log"

_DIRECTION_CMP: dict[str, Callable[[float, float], bool]] = {"ge": operator.ge, "le": operator.le}


# --------------------------------------------------------------- ledger I/O
def _load_all_rows(path: Path) -> list[dict[str, Any]]:
    """Every well-formed JSON line in ``path``; a blank or malformed line is
    skipped, never raised. Only ``scripts/gpu_check.py``'s
    ``_existing_keys_by_key`` already applies this same guarded-parse
    contract (blank lines skipped AND a ``try/except`` around
    ``json.loads``); ``experiments/benchmark_lab.py``'s ``_row_exists`` and
    ``scripts/run_matched_state.py``'s ``_load_rows_by_round`` only skip
    blank lines -- a malformed line would raise ``json.JSONDecodeError``
    straight out of either of those two. This function is strictly more
    defensive than those two precedents, not a mirror of them.

    DUPLICATION-PENDING: the "read a JSONL ledger line-by-line, skip blank
    lines" leaf loop itself is duplicated across all three of those call
    sites (with or without the malformed-JSON guard); flagged for the
    orchestrator to hoist into a shared ledger-reading helper rather than
    done here, since none of those three files is in this task's footprint.
    """
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return rows


def _rows_for_task(all_rows: list[dict[str, Any]], task_name: str) -> dict[str, list[dict]]:
    """``all_rows`` filtered to ``task_name``'s round tag, grouped by arm
    (``variant``). Every ``ARM_REGISTRY`` key is present (possibly with an
    empty list) so a 0-row arm still appears in the completeness readout;
    a row whose ``variant`` isn't a recognized arm is silently dropped
    (not this round's data)."""
    round_tag = ROUND_TAGS[task_name]
    by_arm: dict[str, list[dict]] = {arm: [] for arm in ARM_REGISTRY}
    for row in all_rows:
        if row.get("round") != round_tag or row.get("task") != task_name:
            continue
        variant = row.get("variant")
        if variant in by_arm:
            by_arm[variant].append(row)
    return by_arm


# ------------------------------------------------------------- fit/robustness
def _fit_rows(
    rows: list[dict[str, Any]], metric_key: str, threshold: float, direction: str
) -> list[dict[str, Any]]:
    cmp = _DIRECTION_CMP[direction]
    return [row for row in rows if cmp(row["ckpt"][metric_key], threshold)]


def _robustness_counts(
    rows: list[dict[str, Any]], metric_key: str, direction: str, robustness: tuple[float, ...]
) -> dict[str, int]:
    cmp = _DIRECTION_CMP[direction]
    return {
        str(th): sum(1 for row in rows if cmp(row["ckpt"][metric_key], th)) for th in robustness
    }


# ------------------------------------------------------- generalization table
def _acc_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Every distinct ``acc`` key across ``rows``, sorted for determinism.

    Data-driven rather than a per-task hardcoded list: S5 rows carry
    ``"T256"``/``"T512"``/``"T1024"``, MQAR rows carry
    ``"T256_p16"``/``"T256_p32"``, psMNIST rows carry a single
    ``"test"``, and pendulum rows carry none (empty ``acc`` dict -- no
    post-selection generalization sweep beyond the fit metric, spec §4).
    """
    keys: set[str] = set()
    for row in rows:
        keys.update((row.get("acc") or {}).keys())
    return sorted(keys)


def _mean_acc_by_key(rows: list[dict[str, Any]], key: str) -> float | None:
    """Mean of ``row["acc"][key]`` over the subset of ``rows`` that carry
    ``key`` (skips rows missing it rather than raising -- tolerant of the
    heterogeneous per-task ``acc`` shapes above); ``None`` if no row has it."""
    values = [row["acc"][key] for row in rows if key in (row.get("acc") or {})]
    return sum(values) / len(values) if values else None


def _generalization_tables(
    rows: list[dict[str, Any]], fit_rows: list[dict[str, Any]]
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """``(mean_acc, fit_only_acc)`` -- raw (over ALL rows) and fit-only
    (over fitting rows only), per acc key observed in ``rows`` (docs
    framing rule: never state generalization accuracy without both)."""
    keys = _acc_keys(rows)
    mean_acc = {k: _mean_acc_by_key(rows, k) for k in keys}
    fit_only_acc = {k: _mean_acc_by_key(fit_rows, k) for k in keys}
    return mean_acc, fit_only_acc


# --------------------------------------------------------------- completeness
def _missing_seeds(rows: list[dict[str, Any]], seeds_planned: int) -> tuple[list[int], list[int]]:
    """``(present_seeds, missing_seeds)`` vs the planned ``range(seeds_planned)``
    matrix (spec §2 seed matrices; the lab driver's own seed convention,
    e.g. ``run_matched_state.py``'s ``_ALL_SEEDS = tuple(range(12))``)."""
    present = sorted({row["seed"] for row in rows if isinstance(row.get("seed"), int)})
    missing = sorted(set(range(seeds_planned)) - set(present))
    return present, missing


# --------------------------------------------------------------- param counts
def _arm_param_count(task: TaskSpec, arm: str) -> int:
    """Full per-arm model parameter count for ``task`` (embedding/head +
    composer + decay wiring where applicable), via
    ``experiments.benchmark_lab.build_model`` -- see module docstring for
    why this is preferred over hand-written formulas here."""
    model = build_model(task, arm)
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------- stratum check
def _stratum_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinct ``(device, torch, scan, compile)`` stratum combinations
    observed across ``rows``' ``config`` blocks (spec §7 / CLAUDE.md:
    "Evidence strata are never mixed silently"). More than one distinct
    combination for the same task's rows is a genuine anomaly this report
    surfaces for a human to resolve, not something it silently accepts or
    fixes."""
    seen: set[tuple[Any, Any, Any, Any]] = set()
    for row in rows:
        cfg = row.get("config") or {}
        seen.add((cfg.get("device"), cfg.get("torch"), cfg.get("scan"), cfg.get("compile")))
    return [
        {"device": d, "torch": t, "scan": s, "compile": c}
        for d, t, s, c in sorted(seen, key=lambda tup: tuple(str(x) for x in tup))
    ]


# --------------------------------------------------------------- per-arm report
@dataclass
class ArmReport:
    arm: str
    seeds_present: int
    seeds_planned: int
    present_seeds: list[int]
    missing_seeds: list[int]
    fits: int
    robustness: dict[str, int]
    mean_acc: dict[str, float | None]
    fit_only_acc: dict[str, float | None]
    params: int


def _build_arm_report(task: TaskSpec, arm: str, rows: list[dict[str, Any]]) -> ArmReport:
    fit_rows = _fit_rows(rows, task.fit_metric, task.fit_threshold, task.fit_direction)
    mean_acc, fit_only_acc = _generalization_tables(rows, fit_rows)
    present, missing = _missing_seeds(rows, task.seeds)
    return ArmReport(
        arm=arm,
        seeds_present=len(rows),
        seeds_planned=task.seeds,
        present_seeds=present,
        missing_seeds=missing,
        fits=len(fit_rows),
        robustness=_robustness_counts(rows, task.fit_metric, task.fit_direction, task.robustness),
        mean_acc=mean_acc,
        fit_only_acc=fit_only_acc,
        params=_arm_param_count(task, arm),
    )


def _fisher_vs_reference(arm_reports: dict[str, ArmReport]) -> dict[str, dict[str, Any]]:
    """Two-sided Fisher exact for every arm vs ``FISHER_REFERENCE_ARM``
    (spec §4/§8). An arm with 0 rows -- or a 0-row reference arm -- reports
    ``n/a`` rather than a crash (mirrors ``run_matched_state.py``'s
    "0 rows found" convention)."""
    reference = arm_reports[FISHER_REFERENCE_ARM]
    out: dict[str, dict[str, Any]] = {}
    for arm, rep in arm_reports.items():
        if arm == FISHER_REFERENCE_ARM:
            continue
        if rep.seeds_present == 0 or reference.seeds_present == 0:
            out[arm] = {"p": None, "note": "n/a (one arm has 0 rows)"}
            continue
        p = fisher_exact_two_sided(
            rep.fits,
            rep.seeds_present - rep.fits,
            reference.fits,
            reference.seeds_present - reference.fits,
        )
        out[arm] = {
            "p": p,
            "fits": f"{rep.fits}/{rep.seeds_present}",
            "reference_fits": f"{reference.fits}/{reference.seeds_present}",
        }
    return out


# ------------------------------------------------------------------ report
def build_task_report(task_name: str, all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the full ``bench_<task>.json`` payload for ``task_name``
    from ``all_rows`` (already-loaded ledger rows, any round) -- pure
    function of its inputs, no file I/O, so tests can pass synthetic rows
    directly."""
    task = TASKS[task_name]
    by_arm_rows = _rows_for_task(all_rows, task_name)
    arm_reports = {arm: _build_arm_report(task, arm, rows) for arm, rows in by_arm_rows.items()}
    all_task_rows = [row for rows in by_arm_rows.values() for row in rows]
    return {
        "task": task_name,
        "round": ROUND_TAGS[task_name],
        "fit_metric": task.fit_metric,
        "fit_threshold": task.fit_threshold,
        "fit_direction": task.fit_direction,
        "robustness_thresholds": list(task.robustness),
        "fisher_reference_arm": FISHER_REFERENCE_ARM,
        "arms": {arm: asdict(rep) for arm, rep in arm_reports.items()},
        "fisher_vs_reference": _fisher_vs_reference(arm_reports),
        "stratum_labels": _stratum_labels(all_task_rows),
        "env": {
            "torch": torch.__version__,
            "git_commit": git_commit_sha(),
            "generated": datetime.now(timezone.utc).isoformat(),
        },
    }


# ------------------------------------------------------------------ markdown
def _fmt_acc(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _fmt_threshold(value: float) -> str:
    """Display-only rounding for a robustness threshold (e.g. the
    pendulum's ``0.8 * PENDULUM_TAU`` carries float noise as
    ``0.0024000000000000002``): cosmetic only -- the JSON payload and the
    ``rep["robustness"]`` dict key (built from the unrounded ``str(th)``)
    are untouched, so this never risks a lookup mismatch."""
    return f"{value:.6g}"


def render_markdown(report: dict[str, Any]) -> str:
    """Render one task's ``build_task_report`` payload as Markdown --
    fits/generalization table, threshold-robustness, Fisher-vs-reference,
    and a completeness section (present/missing seeds per arm).

    The "Env" line (``report["env"]``) discloses the report-GENERATION
    environment -- the torch build/commit/timestamp this run of
    ``scripts/report_benchmarks.py`` itself used to construct models for
    param counting -- matching the convention every other
    ``experiments/bench/*.md`` artifact opens with (e.g.
    ``delta_paths.md``, ``gpu_delta_probe.md``). It is deliberately
    separate from "Stratum(s) observed" below, which discloses the
    device/torch/scan/compile combinations recorded on the actual
    TRAINING rows -- the two describe different machines/runs and must
    not be conflated (CLAUDE.md: never mix strata silently)."""
    arms = report["arms"]
    fit_dir_symbol = ">=" if report["fit_direction"] == "ge" else "<="
    env = report["env"]
    lines = [
        f"# Benchmark round: {report['task']} (`{report['round']}`)",
        "",
        f"Fit metric: `ckpt.{report['fit_metric']}` {fit_dir_symbol} "
        f"{report['fit_threshold']} (robustness triple: "
        f"{', '.join(_fmt_threshold(t) for t in report['robustness_thresholds'])}). Fisher "
        f"reference arm: `{report['fisher_reference_arm']}`. Computed from "
        "`experiments/lab_results.jsonl` (rows matching this round's tag); "
        "regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.",
        "",
        f"Env: torch {env['torch']}, commit {env['git_commit']}, generated {env['generated']}.",
        "",
    ]
    strata = report["stratum_labels"]
    if strata:
        strata_str = "; ".join(
            f"device={s['device']}, torch={s['torch']}, scan={s['scan']}, compile={s['compile']}"
            for s in strata
        )
        anomaly = (
            "" if len(strata) == 1 else " -- MULTIPLE DISTINCT STRATA OBSERVED (never mix silently)"
        )
        lines.append(f"Stratum(s) observed: {strata_str}{anomaly}")
    else:
        lines.append("Stratum(s) observed: none (0 rows).")
    lines.append("")

    acc_keys = sorted({k for rep in arms.values() for k in rep["mean_acc"]})
    lines += ["## Fits and generalization accuracy (raw / fit-only)", ""]
    if acc_keys:
        acc_header = " | ".join(f"acc@{k} (raw/fit-only)" for k in acc_keys)
        lines.append(f"| arm | seeds (present/planned) | fits | {acc_header} | params |")
        lines.append("| --- | --- | --- |" + " --- |" * len(acc_keys) + " --- |")
    else:
        lines.append("| arm | seeds (present/planned) | fits | params |")
        lines.append("| --- | --- | --- | --- |")
    for arm, rep in arms.items():
        fits_str = f"{rep['fits']}/{rep['seeds_present']}" if rep["seeds_present"] else "0/0"
        seeds_cell = f"{rep['seeds_present']}/{rep['seeds_planned']}"
        if acc_keys:
            acc_cells = " | ".join(
                f"{_fmt_acc(rep['mean_acc'].get(k))} / {_fmt_acc(rep['fit_only_acc'].get(k))}"
                for k in acc_keys
            )
            lines.append(f"| {arm} | {seeds_cell} | {fits_str} | {acc_cells} | {rep['params']:,} |")
        else:
            lines.append(f"| {arm} | {seeds_cell} | {fits_str} | {rep['params']:,} |")
        if rep["seeds_present"] == 0:
            lines.append(f"  (0 rows found for arm `{arm}`)")

    thresholds = report["robustness_thresholds"]
    threshold_header = " | ".join(_fmt_threshold(t) for t in thresholds)
    lines += ["", "## Threshold-robustness", "", f"| arm | {threshold_header} |"]
    lines.append("| --- |" + " --- |" * len(thresholds))
    for arm, rep in arms.items():
        n = rep["seeds_present"]
        # dict key must be the exact `str(th)` build_task_report used (see
        # `_robustness_counts`) -- `_fmt_threshold` above is display-only.
        cells = " | ".join(f"{rep['robustness'][str(th)]}/{n}" if n else "n/a" for th in thresholds)
        lines.append(f"| {arm} | {cells} |")

    lines += ["", f"## Two-sided Fisher exact vs `{report['fisher_reference_arm']}`", ""]
    for arm, info in report["fisher_vs_reference"].items():
        if info.get("p") is None:
            lines.append(f"- {arm} vs {report['fisher_reference_arm']}: {info.get('note', 'n/a')}")
        else:
            lines.append(
                f"- {arm} ({info['fits']}) vs {report['fisher_reference_arm']} "
                f"({info['reference_fits']}): p = {info['p']:.4g}"
            )

    lines += ["", "## Completeness (present vs planned seed matrix)", ""]
    for arm, rep in arms.items():
        seeds_cell = f"{rep['seeds_present']}/{rep['seeds_planned']}"
        if rep["missing_seeds"]:
            lines.append(f"- {arm}: {seeds_cell} present; missing seeds: {rep['missing_seeds']}")
        else:
            lines.append(f"- {arm}: {seeds_cell} present; complete")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- write
def write_reports(all_rows: list[dict[str, Any]], out_dir: Path) -> dict[str, dict[str, Any]]:
    """Build and write ``bench_<task>.{json,md}`` for every task in
    ``TASKS``, regenerated whole. Returns the built reports (keyed by
    task name) so ``main`` can print the completeness summary without
    re-reading the just-written files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for task_name in TASKS:
        report = build_task_report(task_name, all_rows)
        reports[task_name] = report
        (out_dir / f"bench_{task_name}.json").write_text(json.dumps(report, indent=2) + "\n")
        (out_dir / f"bench_{task_name}.md").write_text(render_markdown(report))
    return reports


def _print_completeness_summary(reports: dict[str, dict[str, Any]]) -> None:
    print("Completeness readout (present/planned seeds per arm):")
    for task_name, report in reports.items():
        for arm, rep in report["arms"].items():
            flag = f"  MISSING {rep['missing_seeds']}" if rep["missing_seeds"] else ""
            seeds_cell = f"{rep['seeds_present']:>3}/{rep['seeds_planned']:<3}"
            print(f"  {task_name:>9}/{arm:<8} {seeds_cell}{flag}")


# ------------------------------------------------------------------------ CLI
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=_RESULTS, help="path to lab_results.jsonl (default: repo's)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_BENCH_DIR, help="output dir for bench_<task>.{json,md}"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    all_rows = _load_all_rows(args.results)
    reports = write_reports(all_rows, args.out_dir)
    for task_name in reports:
        print(f"wrote {args.out_dir / f'bench_{task_name}.json'} and bench_{task_name}.md")
    _print_completeness_summary(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
