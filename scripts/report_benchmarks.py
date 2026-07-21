"""Benchmark round report generator (spec:
``.claude/output/specs/2026-07-19-benchmark-round-design.md``, §4 "Fit and
statistics" / "Reporting", §6 ledger row + stats parameterization
contracts).

Reads ``experiments/lab_results.jsonl`` for the four current bench round
tags (``bench-s5-02``, ``bench-mqar-02``, ``bench-psmnist-02``,
``bench-pendulum-02`` -- read from ``experiments.benchmark_tasks
.BENCH_ROUND_TAGS``, the single source of truth this module and
``scripts/gpu_benchmark_campaign.py`` both bind to) and regenerates
``experiments/bench/bench_<task>.json``/``.md`` per task, whole, every run
-- never hand-edited (module docstring convention shared with every other
``experiments/bench/*`` generator; see ``experiments/index.md``). The
``-01`` tags are the recorded pilot/calibration population (heterogeneous
per-seed budgets) and are deliberately excluded here -- see
``BENCH_ROUND_TAGS``'s own comment for the ``-01`` -> ``-02`` bump
rationale; ``scripts/gpu_check.py`` is the one place that still recognizes
both generations, for old pilot job logs/sidecars. The S5-only probe round
(``bench-s5-probe-01``, ``experiments.benchmark_tasks
.BENCH_PROBE_ROUND_TAGS``) is a separate population entirely -- this
module's matched ``-02`` per-task accounting below reads only
``experiments.benchmark_lab.MATRIX_ARMS`` (never ``ARM_REGISTRY``, which
also includes the three S5-only ``PROBE_ARMS`` and the psMNIST-only
``REF_ARMS``), so the matched ``-02`` reports this module writes never
show probe or reference arms and never change shape because of them. The
probe population gets its own separate report -- see the "further, later
addition" paragraph below.

A separate, later addition (Amendments, 2026-07-20 "gru-large grounding
reference" entry) regenerates ``experiments/bench/bench_<task>_ref.json``/
``.md`` for every task in ``experiments.benchmark_tasks
.BENCH_REF_ROUND_TAGS`` (currently just ``psmnist``): a REFERENCE-labeled
report for ``experiments.benchmark_lab.REF_ARMS`` rows (e.g. ``gru-large``)
-- distinct filenames, distinct round tag, no Fisher-vs-``log`` comparison
(a ref arm's rows live under a different training budget than the matched
population `FISHER_REFERENCE_ARM` is judged against, so comparing them
would silently mix two strata into one statistic -- CLAUDE.md: "evidence
strata are never mixed silently"). This is purely additive: the four
canonical ``bench_<task>.{json,md}`` files and their generation code path
are untouched by it, so the matched ``-02`` accounting/regression check
stays byte-identical.

A further, later addition (task-11a, S5 design-correction probe write-up)
regenerates ``experiments/bench/bench_s5_probe.json``/``.md`` for every
task in ``experiments.benchmark_tasks.BENCH_PROBE_ROUND_TAGS`` (currently
just ``s5``): a PROBE-labeled report for ``experiments.benchmark_lab
.PROBE_ARMS`` rows (``signed-rotation-k5``/``signed-delta-nh3``/
``signed-delta-nh4``, round tag ``bench-s5-probe-01``) -- a descriptive
design-correction population, not a competing arm, so it shares the
REFERENCE report's no-Fisher, no-matched-accounting treatment (both flow
through the same ``_build_population_task_report``/
``_render_population_markdown`` path below; only the round tags, arm
registry, filenames, and banner text differ per population). Also purely
additive: the matched ``bench_<task>.{json,md}`` and the REFERENCE
``bench_<task>_ref.{json,md}`` files and their generation code paths are
untouched by it.

Per task, per arm (``experiments.benchmark_lab.MATRIX_ARMS``:
log/signed/rotation/signed-rotation/givens/delta/signed-givens/signed-delta/gru
-- ``gru`` is the depth-matched classical ``nn.GRU`` external control arm,
added by a fourth amendment; not a ``MinGRUStack`` mixer, but a normal
``MATRIX_ARMS`` row like the other eight, so it flows through this
module's ``build_model``-based param counting and completeness accounting
with no special-casing needed here):

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
planned ``range(TaskSpec.seeds)`` matrix (arms from ``MATRIX_ARMS``,
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
from collections.abc import Callable, Iterable
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
from experiments.benchmark_lab import MATRIX_ARMS, PROBE_ARMS, REF_ARMS, build_model  # noqa: E402
from experiments.benchmark_tasks import (  # noqa: E402
    BENCH_ARM_ROUND_OVERRIDES,
    BENCH_PROBE_ROUND_TAGS,
    BENCH_REF_ROUND_TAGS,
    BENCH_ROUND_TAGS,
    TASKS,
    TaskSpec,
)

_RESULTS = _REPO_ROOT / "experiments" / "lab_results.jsonl"
_BENCH_DIR = _REPO_ROOT / "experiments" / "bench"

# Current seed-matrix population's round tags -- this module owns
# `ROUND_TAGS` as its own name (existing convention -- `gpu_benchmark_
# campaign.py` binds its own `_ROUND_TAGS` name to the same mapping), but
# both read the single `BENCH_ROUND_TAGS` source of truth in
# `experiments/benchmark_tasks.py` rather than each hardcoding an
# independently-editable copy -- see that mapping's own comment for the
# `-01` -> `-02` bump rationale.
ROUND_TAGS: dict[str, str] = BENCH_ROUND_TAGS

# Reference-arm round tags (gru-large grounding reference, module
# docstring's "separate, later addition" paragraph) -- this module owns
# `REF_ROUND_TAGS` as its own name, same binding convention as `ROUND_TAGS`
# above, bound directly to `experiments.benchmark_tasks
# .BENCH_REF_ROUND_TAGS` (the single source of truth
# `scripts/gpu_benchmark_campaign.py`'s `_REF_ROUND_TAGS` also binds to).
REF_ROUND_TAGS: dict[str, str] = BENCH_REF_ROUND_TAGS

# S5 design-correction probe round tags (task-11a, module docstring's
# "further, later addition" paragraph) -- this module owns `PROBE_ROUND_TAGS`
# as its own name, same binding convention as `ROUND_TAGS`/`REF_ROUND_TAGS`
# above, bound directly to `experiments.benchmark_tasks
# .BENCH_PROBE_ROUND_TAGS` (the single source of truth
# `scripts/gpu_benchmark_campaign.py`'s round-tag resolution also reads).
PROBE_ROUND_TAGS: dict[str, str] = BENCH_PROBE_ROUND_TAGS

# Per-arm round-tag correction overrides (design spec, `.claude/output/
# specs/2026-07-21-round-tag-override-design.md`): the read-side half of
# the paired override rule -- `scripts/gpu_benchmark_campaign.py`'s
# `_round_tag_for_arm` is the write-side half, both resolving the same
# `arm -> {task -> correction round tag}` map so the two paths cannot
# diverge (spec §6). Same binding convention as `ROUND_TAGS`/`REF_ROUND_TAGS`
# /`PROBE_ROUND_TAGS` above: this module owns its own name, bound directly
# to the single source of truth in `experiments/benchmark_tasks.py`, never
# a hardcoded copy.
ARM_ROUND_OVERRIDES: dict[str, dict[str, str]] = BENCH_ARM_ROUND_OVERRIDES

# "log as the Fisher reference arm" (spec §4/§8): a family-validation round
# contrasts every arm against the vanilla minGRU baseline.
FISHER_REFERENCE_ARM = "log"


def _source_round_for_arm(task_name: str, arm: str, primary_tag: str) -> str:
    """The round tag one arm's rows for ``task_name`` are read from: the
    arm's correction tag if ``ARM_ROUND_OVERRIDES`` registers one for
    ``(arm, task_name)``, else ``primary_tag`` unchanged (spec §6, "shared
    read-side resolver"). The paired write-side half is
    ``gpu_benchmark_campaign.py``'s ``_round_tag_for_arm``, whose override
    check is the same ``ARM_ROUND_OVERRIDES.get(arm, {}).get(task_name)``
    lookup -- the two sides must resolve identically so a correction round
    is written and read under the same tag. Both ``_rows_for_task`` (primary
    = the matrix tag) and ``_rows_for_population`` (primary = the
    population's own tag) call this once per arm rather than inlining the
    lookup, so the resolution rule cannot drift between the two selectors.
    With an empty override map this always returns ``primary_tag``
    (behavior-neutral)."""
    return ARM_ROUND_OVERRIDES.get(arm, {}).get(task_name, primary_tag)


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
    (``variant``). Every ``MATRIX_ARMS`` key is present (possibly with an
    empty list) so a 0-row arm still appears in the completeness readout;
    a row whose ``variant`` isn't a recognized MATRIX arm is silently
    dropped (not this round's data).

    Deliberately ``MATRIX_ARMS`` here, NOT ``experiments.benchmark_lab
    .ARM_REGISTRY`` (matrix arms unioned with the S5-only ``PROBE_ARMS``,
    e.g. ``signed-rotation-k5``): this ``-02`` matrix report's planned-arm
    accounting must stay exactly the nine clean seed-matrix arms --
    widening it to ``ARM_REGISTRY`` would make every ``-02`` report show
    the three probe arms as permanently "0/seeds missing" (they write
    under a distinct probe round tag, `BENCH_PROBE_ROUND_TAGS`, never
    this task's `-02` tag, so they can never have rows here regardless).
    See `experiments.benchmark_lab.PROBE_ARMS`'s own comment.

    Each arm's source round is resolved through ``_source_round_for_arm``
    (primary = the matrix tag, ``ROUND_TAGS[task_name]``): an arm with a
    registered correction override (e.g. ``signed-rotation``) reads its
    rows from its override tag instead, while every other arm reads from
    the matrix tag unchanged -- arm *presence* still comes from iterating
    ``MATRIX_ARMS``, only the round each arm's rows are drawn from
    differs."""
    primary_tag = ROUND_TAGS[task_name]
    by_arm: dict[str, list[dict]] = {arm: [] for arm in MATRIX_ARMS}
    source_round: dict[str, str] = {
        arm: _source_round_for_arm(task_name, arm, primary_tag) for arm in MATRIX_ARMS
    }
    for row in all_rows:
        variant = row.get("variant")
        if variant not in by_arm:
            continue
        if row.get("round") != source_round[variant] or row.get("task") != task_name:
            continue
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


def _assert_one_row_per_seed(
    task_name: str, arm: str, rows: list[dict[str, Any]], present: list[int]
) -> None:
    """Fail loud when ``rows``' count doesn't match its distinct-seed count
    (pre-matrix technical review, item 7): a ledger dedup failure or a
    genuine duplicate seed would silently inflate this arm's Fisher-exact
    denominator -- ``rep.seeds_present`` feeds ``_fisher_vs_reference``
    directly (``rep.seeds_present - rep.fits`` is the non-fit count in the
    2x2 contingency table) -- without ever surfacing as a wrong p-value a
    reader could catch by inspection. This is a data-integrity check, not a
    report annotation: it raises rather than rendering a silently-wrong
    table (CLAUDE.md: "Numbers are transcribed from artifacts... reviewers
    now check cell-by-cell")."""
    if len(rows) == len(present):
        return
    seed_counts: dict[int, int] = {}
    for row in rows:
        seed = row.get("seed")
        if isinstance(seed, int):
            seed_counts[seed] = seed_counts.get(seed, 0) + 1
    duplicates = {seed: n for seed, n in seed_counts.items() if n > 1}
    raise ValueError(
        f"{task_name}/{arm}: {len(rows)} row(s) but only {len(present)} distinct "
        f"seed(s) -- duplicate seed rows would silently inflate the Fisher-exact "
        f"denominator; duplicate seeds: {duplicates}. Fix the ledger (or the "
        "upstream dedup path in scripts/gpu_check.py) before regenerating this "
        "report."
    )


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
    _assert_one_row_per_seed(task.name, arm, rows, present)
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
def _overridden_arm_sources(
    task_name: str, arms: Iterable[str], primary_tag: str
) -> dict[str, str]:
    """``{overridden arm: source round}`` for every arm in ``arms`` whose
    resolved source round (`_source_round_for_arm`) differs from
    ``primary_tag`` -- the disclosure record (design spec §4 "Disclosure",
    §6 "disclosure record"). Empty when no arm in ``arms`` is overridden
    for ``task_name`` (e.g. the empty-override-map case, or a task/arm set
    with no registered correction). Shared by both `build_task_report`
    (arms = ``MATRIX_ARMS``, primary = the matrix tag) and
    `_build_population_task_report` (arms = ``pop.arms``, primary = the
    population's own tag) so the disclosure rule cannot drift between the
    two payload builders, mirroring how both already share
    `_source_round_for_arm` itself."""
    return {
        arm: source
        for arm in arms
        if (source := _source_round_for_arm(task_name, arm, primary_tag)) != primary_tag
    }


def build_task_report(task_name: str, all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the full ``bench_<task>.json`` payload for ``task_name``
    from ``all_rows`` (already-loaded ledger rows, any round) -- pure
    function of its inputs, no file I/O, so tests can pass synthetic rows
    directly."""
    task = TASKS[task_name]
    primary_tag = ROUND_TAGS[task_name]
    by_arm_rows = _rows_for_task(all_rows, task_name)
    arm_reports = {arm: _build_arm_report(task, arm, rows) for arm, rows in by_arm_rows.items()}
    all_task_rows = [row for rows in by_arm_rows.values() for row in rows]
    return {
        "task": task_name,
        "round": primary_tag,
        "fit_metric": task.fit_metric,
        "fit_threshold": task.fit_threshold,
        "fit_direction": task.fit_direction,
        "robustness_thresholds": list(task.robustness),
        "fisher_reference_arm": FISHER_REFERENCE_ARM,
        "arms": {arm: asdict(rep) for arm, rep in arm_reports.items()},
        "fisher_vs_reference": _fisher_vs_reference(arm_reports),
        "stratum_labels": _stratum_labels(all_task_rows),
        "arm_round_overrides": _overridden_arm_sources(task_name, MATRIX_ARMS, primary_tag),
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


def _render_stratum_line(stratum_labels: list[dict[str, Any]]) -> str:
    """ "Stratum(s) observed: ..." line, shared by the matched and the
    reference report renderers (both carry the same ``stratum_labels``
    shape, `_stratum_labels`'s output)."""
    if not stratum_labels:
        return "Stratum(s) observed: none (0 rows)."
    strata_str = "; ".join(
        f"device={s['device']}, torch={s['torch']}, scan={s['scan']}, compile={s['compile']}"
        for s in stratum_labels
    )
    anomaly = (
        ""
        if len(stratum_labels) == 1
        else " -- MULTIPLE DISTINCT STRATA OBSERVED (never mix silently)"
    )
    return f"Stratum(s) observed: {strata_str}{anomaly}"


def _render_provenance_lines(report: dict[str, Any]) -> list[str]:
    """Mixed-provenance disclosure line(s) (design spec §4 "Disclosure",
    §6 "disclosure record"), shared by `render_markdown` and
    `_render_population_markdown` so the provenance block is never inlined
    twice. Returns ``[]`` -- no line rendered -- when
    ``report["arm_round_overrides"]`` is empty (the common case today: no
    overridden arm present), so a report with no overridden arm renders
    exactly as before this key existed. Otherwise returns one line naming
    each overridden arm, its correction round, and the primary round every
    other arm in this report was read from."""
    overrides = report["arm_round_overrides"]
    if not overrides:
        return []
    primary = report["round"]
    corrections = "; ".join(f"{arm} <- `{source}`" for arm, source in overrides.items())
    return [
        "",
        f"Provenance: {corrections} (other arms: `{primary}`) -- this report mixes rounds; "
        "see `arm_round_overrides` in the JSON payload.",
    ]


def _render_fits_table_lines(arms: dict[str, dict[str, Any]]) -> list[str]:
    """ "Fits and generalization accuracy" table, shared by the matched and
    the reference report renderers (both carry the same per-arm
    ``ArmReport``-shaped dict, `_build_arm_report`'s output)."""
    lines: list[str] = []
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
    return lines


def _render_robustness_lines(arms: dict[str, dict[str, Any]], thresholds: list[float]) -> list[str]:
    """ "Threshold-robustness" table, shared by the matched and the
    reference report renderers."""
    threshold_header = " | ".join(_fmt_threshold(t) for t in thresholds)
    lines = ["", "## Threshold-robustness", "", f"| arm | {threshold_header} |"]
    lines.append("| --- |" + " --- |" * len(thresholds))
    for arm, rep in arms.items():
        n = rep["seeds_present"]
        # dict key must be the exact `str(th)` build_task_report used (see
        # `_robustness_counts`) -- `_fmt_threshold` above is display-only.
        cells = " | ".join(f"{rep['robustness'][str(th)]}/{n}" if n else "n/a" for th in thresholds)
        lines.append(f"| {arm} | {cells} |")
    return lines


def _render_completeness_lines(arms: dict[str, dict[str, Any]]) -> list[str]:
    """ "Completeness (present vs planned seed matrix)" section, shared by
    the matched and the reference report renderers."""
    lines = ["", "## Completeness (present vs planned seed matrix)", ""]
    for arm, rep in arms.items():
        seeds_cell = f"{rep['seeds_present']}/{rep['seeds_planned']}"
        if rep["missing_seeds"]:
            lines.append(f"- {arm}: {seeds_cell} present; missing seeds: {rep['missing_seeds']}")
        else:
            lines.append(f"- {arm}: {seeds_cell} present; complete")
    return lines


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
        _render_stratum_line(report["stratum_labels"]),
    ]
    lines += _render_provenance_lines(report)
    lines.append("")

    lines += _render_fits_table_lines(arms)
    lines += _render_robustness_lines(arms, report["robustness_thresholds"])

    lines += ["", f"## Two-sided Fisher exact vs `{report['fisher_reference_arm']}`", ""]
    for arm, info in report["fisher_vs_reference"].items():
        if info.get("p") is None:
            lines.append(f"- {arm} vs {report['fisher_reference_arm']}: {info.get('note', 'n/a')}")
        else:
            lines.append(
                f"- {arm} ({info['fits']}) vs {report['fisher_reference_arm']} "
                f"({info['reference_fits']}): p = {info['p']:.4g}"
            )

    lines += _render_completeness_lines(arms)
    lines.append("")
    return "\n".join(lines) + "\n"


# ----------------------------------------- non-matched population reports
# `REF_ARMS` (gru-large grounding reference) and `PROBE_ARMS` (S5
# design-correction probe) are both explicitly NON-matched populations --
# never in the matched `-02` seed-matrix accounting, never a Fisher-vs-`log`
# comparison (module docstring's "separate, later addition"/"further, later
# addition" paragraphs). `_AuxPopulation` bundles the handful of things that
# actually differ between the two (round tags, arm registry, raise-loud
# names, banner text) so `_build_population_task_report`/
# `_render_population_markdown` below are the single build/render path both
# `build_ref_task_report`/`render_ref_markdown` and `build_probe_task_report`/
# `render_probe_markdown` flow through, rather than two independently-
# drifting near-copies of the same population-shape logic.
@dataclass(frozen=True)
class _AuxPopulation:
    # Set True in the payload ("reference"/"probe"); also the word used in
    # the "rows matching this <payload_key> round's tag" prose sentence
    # below (`_render_population_markdown`).
    payload_key: str
    arms_name: str  # "REF_ARMS" / "PROBE_ARMS" -- raise-loud message only
    # "BENCH_REF_ROUND_TAGS" / "BENCH_PROBE_ROUND_TAGS" -- raise-loud message only
    round_tags_name: str
    round_tags: dict[str, str]
    arms: dict[str, tuple[str | list[str], dict | None]]
    banner_title: str  # render_*_markdown's H1 suffix, e.g. "REFERENCE arm(s)"
    # Explanatory paragraph under the H1; a callable because the REFERENCE
    # variant embeds the matched `ROUND_TAGS[task]` dynamically.
    banner_note: Callable[[dict[str, Any]], str]


def _ref_banner_note(report: dict[str, Any]) -> str:
    return (
        f"**Non-matched reference/grounding arm(s) -- NOT part of the matched "
        f"`{ROUND_TAGS[report['task']]}` seed-matrix accounting.** No Fisher-exact "
        "comparison is computed here: this population runs under its own training "
        "budget and round tag, distinct from the matched arms (CLAUDE.md: evidence "
        "strata are never mixed silently)."
    )


def _probe_banner_note(report: dict[str, Any]) -> str:
    return (
        f"**S5 design-correction probe arm(s) -- NOT part of the matched "
        f"`{ROUND_TAGS[report['task']]}` seed-matrix accounting.** Tests whether two "
        "suspected experiment-design artifacts -- `signed-rotation`'s missing K=5 snap "
        "order and `signed-delta`'s low nh=2 product count -- rather than a genuine "
        "mechanism limit, explain S5's 0/36 rows for those families (see "
        "`experiments.benchmark_lab.PROBE_ARMS`'s own comment). No Fisher-exact "
        "comparison is computed here: this is a descriptive design-correction "
        "population, not a competing arm judged against `log`."
    )


_REF_POPULATION = _AuxPopulation(
    payload_key="reference",
    arms_name="REF_ARMS",
    round_tags_name="BENCH_REF_ROUND_TAGS",
    round_tags=REF_ROUND_TAGS,
    arms=REF_ARMS,
    banner_title="REFERENCE arm(s)",
    banner_note=_ref_banner_note,
)

_PROBE_POPULATION = _AuxPopulation(
    payload_key="probe",
    arms_name="PROBE_ARMS",
    round_tags_name="BENCH_PROBE_ROUND_TAGS",
    round_tags=PROBE_ROUND_TAGS,
    arms=PROBE_ARMS,
    banner_title="PROBE arm(s)",
    banner_note=_probe_banner_note,
)


def _rows_for_population(
    all_rows: list[dict[str, Any]], task_name: str, pop: _AuxPopulation
) -> dict[str, list[dict]]:
    """``all_rows`` filtered to ``task_name``'s round tag under ``pop``'s
    population (`REF_ARMS` or `PROBE_ARMS`), grouped by arm -- the shared
    non-matched-population counterpart to `_rows_for_task`. Every
    ``pop.arms`` key is present (possibly with an empty list); a row whose
    ``variant`` isn't a recognized arm for this population is silently
    dropped, mirroring `_rows_for_task`'s contract.

    Each arm's source round is resolved through ``_source_round_for_arm``
    (primary = ``pop.round_tags[task_name]``, the population's own tag):
    an arm with a registered correction override (e.g.
    ``signed-rotation-k5`` on ``s5``) reads its rows from its override tag
    instead, while every other arm in ``pop.arms`` reads from the
    population tag unchanged -- mirroring `_rows_for_task`'s per-arm
    resolution. Callers must resolve ``task_name not in pop.round_tags``
    (the raise-loud guard) before calling this, since ``pop.round_tags
    [task_name]`` is read as the primary tag here."""
    primary_tag = pop.round_tags[task_name]
    by_arm: dict[str, list[dict]] = {arm: [] for arm in pop.arms}
    source_round: dict[str, str] = {
        arm: _source_round_for_arm(task_name, arm, primary_tag) for arm in pop.arms
    }
    for row in all_rows:
        variant = row.get("variant")
        if variant not in by_arm:
            continue
        if row.get("round") != source_round[variant] or row.get("task") != task_name:
            continue
        by_arm[variant].append(row)
    return by_arm


def _build_population_task_report(
    task_name: str, all_rows: list[dict[str, Any]], pop: _AuxPopulation
) -> dict[str, Any]:
    """Assemble the ``bench_<task>_{ref,probe}.json`` payload for
    ``task_name``'s ``pop.arms`` rows -- the shared build path behind
    `build_ref_task_report` and `build_probe_task_report`. Deliberately no
    Fisher-vs-``log`` comparison for either non-matched population -- see
    each public wrapper's own docstring for its population-specific
    rationale."""
    if task_name not in pop.round_tags:
        raise ValueError(
            f"no {pop.arms_name} round tag registered for task {task_name!r} -- "
            f"{pop.round_tags_name} only covers {sorted(pop.round_tags)}"
        )
    task = TASKS[task_name]
    primary_tag = pop.round_tags[task_name]
    by_arm_rows = _rows_for_population(all_rows, task_name, pop)
    arm_reports = {arm: _build_arm_report(task, arm, rows) for arm, rows in by_arm_rows.items()}
    all_task_rows = [row for rows in by_arm_rows.values() for row in rows]
    return {
        "task": task_name,
        "round": primary_tag,
        pop.payload_key: True,  # non-matched arm(s) -- never in the matched -02 accounting
        "fit_metric": task.fit_metric,
        "fit_threshold": task.fit_threshold,
        "fit_direction": task.fit_direction,
        "robustness_thresholds": list(task.robustness),
        "arms": {arm: asdict(rep) for arm, rep in arm_reports.items()},
        "stratum_labels": _stratum_labels(all_task_rows),
        "arm_round_overrides": _overridden_arm_sources(task_name, pop.arms, primary_tag),
        "env": {
            "torch": torch.__version__,
            "git_commit": git_commit_sha(),
            "generated": datetime.now(timezone.utc).isoformat(),
        },
    }


def build_ref_task_report(task_name: str, all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the ``bench_<task>_ref.json`` payload for ``task_name``'s
    ``REF_ARMS`` rows (e.g. ``gru-large`` on psMNIST) -- an explicitly
    NON-matched grounding reference, never the matched population
    `build_task_report` reports (module docstring's "separate, later
    addition" paragraph).

    Deliberately no Fisher-vs-``log`` comparison here: a ref arm runs
    under its own training budget (`experiments.benchmark_lab
    .REF_ARM_BUDGETS`), distinct from the matched arms' frozen
    ``TaskSpec.budget`` that `FISHER_REFERENCE_ARM`'s matched-population
    fit rate is itself computed under -- comparing them would silently mix
    two different strata into one Fisher-exact statistic (CLAUDE.md:
    "evidence strata are never mixed silently")."""
    return _build_population_task_report(task_name, all_rows, _REF_POPULATION)


def build_probe_task_report(task_name: str, all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the ``bench_s5_probe.json`` payload for ``task_name``'s
    ``PROBE_ARMS`` rows (``signed-rotation-k5``/``signed-delta-nh3``/
    ``signed-delta-nh4``) -- an explicitly NON-matched, descriptive
    design-correction population, never the matched population
    `build_task_report` reports (module docstring's "further, later
    addition" paragraph; the `REF_ARMS` counterpart is `build_ref_task_
    report`, sharing this function's `_build_population_task_report` path).

    Deliberately no Fisher-vs-``log`` comparison here either: unlike
    `REF_ARMS`, the probe arms run under the SAME matched ``TaskSpec.budget``
    (no ``PROBE_ARM_BUDGETS`` override exists), but the probe is a
    diagnostic for whether `signed-rotation`/`signed-delta`'s matrix configs
    themselves were under-specified (see `experiments.benchmark_lab
    .PROBE_ARMS`'s own comment) -- a descriptive design-correction readout,
    not a claim to be tested against `log`'s matched fit rate."""
    return _build_population_task_report(task_name, all_rows, _PROBE_POPULATION)


def _render_population_markdown(report: dict[str, Any], pop: _AuxPopulation) -> str:
    """Render one task's `_build_population_task_report` payload as
    Markdown -- the shared render path behind `render_ref_markdown` and
    `render_probe_markdown`: same fits/robustness/completeness sections as
    `render_markdown` (shared via `_render_fits_table_lines`/
    `_render_robustness_lines`/`_render_completeness_lines`), but with a
    population-specific banner up top and no Fisher-vs-reference section."""
    arms = report["arms"]
    fit_dir_symbol = ">=" if report["fit_direction"] == "ge" else "<="
    env = report["env"]
    lines = [
        f"# Benchmark round: {report['task']} {pop.banner_title} (`{report['round']}`)",
        "",
        pop.banner_note(report),
        "",
        f"Fit metric: `ckpt.{report['fit_metric']}` {fit_dir_symbol} "
        f"{report['fit_threshold']} (robustness triple: "
        f"{', '.join(_fmt_threshold(t) for t in report['robustness_thresholds'])}). Computed "
        f"from `experiments/lab_results.jsonl` (rows matching this {pop.payload_key} round's "
        "tag); regenerated whole by `scripts/report_benchmarks.py`, never hand-edited.",
        "",
        f"Env: torch {env['torch']}, commit {env['git_commit']}, generated {env['generated']}.",
        "",
        _render_stratum_line(report["stratum_labels"]),
    ]
    lines += _render_provenance_lines(report)
    lines.append("")
    lines += _render_fits_table_lines(arms)
    lines += _render_robustness_lines(arms, report["robustness_thresholds"])
    lines += _render_completeness_lines(arms)
    lines.append("")
    return "\n".join(lines) + "\n"


def render_ref_markdown(report: dict[str, Any]) -> str:
    """Render one task's ``build_ref_task_report`` payload as Markdown --
    same fits/robustness/completeness sections as `render_markdown`
    (shared via `_render_fits_table_lines`/`_render_robustness_lines`/
    `_render_completeness_lines`), but with an explicit REFERENCE banner
    up top and no Fisher-vs-reference section (see `build_ref_task_report`'s
    docstring for why)."""
    return _render_population_markdown(report, _REF_POPULATION)


def render_probe_markdown(report: dict[str, Any]) -> str:
    """Render one task's ``build_probe_task_report`` payload as Markdown --
    mirrors `render_ref_markdown` via the shared `_render_population_markdown`
    path, with an explicit PROBE banner instead of the REFERENCE banner and
    no Fisher-vs-reference section (see `build_probe_task_report`'s
    docstring for why)."""
    return _render_population_markdown(report, _PROBE_POPULATION)


# --------------------------------------------------------------------- write
def write_reports(all_rows: list[dict[str, Any]], out_dir: Path) -> dict[str, dict[str, Any]]:
    """Build and write ``bench_<task>.{json,md}`` for every task in
    ``TASKS``, regenerated whole. Returns the built reports (keyed by
    task name) so ``main`` can print the completeness summary without
    re-reading the just-written files.

    Matched-population output ONLY -- untouched by the reference-report
    addition (`write_ref_reports` below), which writes distinct
    ``bench_<task>_ref.{json,md}`` filenames from a distinct function, so
    this function's return shape/byte output for the four canonical tasks
    is exactly what it was before `REF_ARMS` existed (regression check:
    the matched `-02` reports stay byte-identical)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    for task_name in TASKS:
        report = build_task_report(task_name, all_rows)
        reports[task_name] = report
        (out_dir / f"bench_{task_name}.json").write_text(json.dumps(report, indent=2) + "\n")
        (out_dir / f"bench_{task_name}.md").write_text(render_markdown(report))
    return reports


def write_ref_reports(all_rows: list[dict[str, Any]], out_dir: Path) -> dict[str, dict[str, Any]]:
    """Build and write ``bench_<task>_ref.{json,md}`` for every task in
    ``REF_ROUND_TAGS`` (currently just ``psmnist``), regenerated whole --
    the `REF_ARMS` (e.g. ``gru-large``) counterpart to `write_reports`.
    Distinct filenames from the matched ``bench_<task>.{json,md}`` pair, so
    calling this never touches the matched output."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_reports: dict[str, dict[str, Any]] = {}
    for task_name in REF_ROUND_TAGS:
        report = build_ref_task_report(task_name, all_rows)
        ref_reports[task_name] = report
        (out_dir / f"bench_{task_name}_ref.json").write_text(json.dumps(report, indent=2) + "\n")
        (out_dir / f"bench_{task_name}_ref.md").write_text(render_ref_markdown(report))
    return ref_reports


def write_probe_reports(all_rows: list[dict[str, Any]], out_dir: Path) -> dict[str, dict[str, Any]]:
    """Build and write ``bench_<task>_probe.{json,md}`` for every task in
    ``PROBE_ROUND_TAGS`` (currently just ``s5``), regenerated whole -- the
    `PROBE_ARMS` counterpart to `write_ref_reports`. Distinct filenames
    from both the matched ``bench_<task>.{json,md}`` pair and the
    reference ``bench_<task>_ref.{json,md}`` pair, so calling this never
    touches either."""
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_reports: dict[str, dict[str, Any]] = {}
    for task_name in PROBE_ROUND_TAGS:
        report = build_probe_task_report(task_name, all_rows)
        probe_reports[task_name] = report
        (out_dir / f"bench_{task_name}_probe.json").write_text(json.dumps(report, indent=2) + "\n")
        (out_dir / f"bench_{task_name}_probe.md").write_text(render_probe_markdown(report))
    return probe_reports


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
    ref_reports = write_ref_reports(all_rows, args.out_dir)
    for task_name in ref_reports:
        print(
            f"wrote {args.out_dir / f'bench_{task_name}_ref.json'} and "
            f"bench_{task_name}_ref.md (REFERENCE, non-matched)"
        )
    probe_reports = write_probe_reports(all_rows, args.out_dir)
    for task_name in probe_reports:
        print(
            f"wrote {args.out_dir / f'bench_{task_name}_probe.json'} and "
            f"bench_{task_name}_probe.md (PROBE, S5-only)"
        )
    _print_completeness_summary(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
