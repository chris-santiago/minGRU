"""Tests for `scripts/report_benchmarks.py`: the benchmark round report
generator (spec `.claude/output/specs/2026-07-19-benchmark-round-design.md`
§4 "Fit and statistics" / "Reporting", §6; task brief `.git/sdd/task-7-
brief.md`).

`scripts/` is not an importable package (no `__init__.py` -- see
`tests/test_evidence_stats_params.py`'s identical note), so the module is
loaded by file path via `importlib`, exactly as that sibling test file does.

Sections
--------
1. `_load_all_rows` -- guarded JSONL parse (blank/malformed lines skipped).
2. Fit counting / robustness -- "ge" (S5/MQAR/psMNIST-style) and "le"
   (pendulum MSE-style) directions, read from the real `TaskSpec` registry
   rather than hardcoded here.
3. Generalization tables -- per-task acc-key shapes (S5's `T256`/`T512`/
   `T1024`, MQAR's `T256_p16`/`T256_p32`, psMNIST's single `test`,
   pendulum's none), raw vs fit-only.
4. Fisher-vs-`log` -- computed value sanity, `n/a` degradation when an arm
   (or the reference arm) has 0 rows.
5. Completeness readout -- present/missing seeds vs the planned matrix.
6. Param counts -- positive, per-arm, per-task (not equalized).
7. End-to-end `write_reports`/`main` -- JSON + MD written to a `tmp_path`
   dir from a synthetic ledger fixture, 0-row round renders without a
   crash, real ledger regeneration doesn't raise.
8. Regression: `-02` matrix accounting untouched by the S5-only probe round
   (`PROBE_ARMS`) -- the planned arm set stays exactly the nine
   `MATRIX_ARMS` (the eight `MinGRUStack` mixer arms plus `gru`, the
   classical control arm added by a fourth amendment), and rows carrying
   either the probe round tag or a probe-only variant never surface in a
   `-02` report.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from experiments.benchmark_lab import ARM_REGISTRY, DECAY_CAPABLE_ARMS, MATRIX_ARMS, PROBE_ARMS
from experiments.benchmark_tasks import BENCH_PROBE_ROUND_TAGS, BENCH_ROUND_TAGS, TASKS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "report_benchmarks.py"
_spec = importlib.util.spec_from_file_location("report_benchmarks", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
report_benchmarks = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("report_benchmarks", report_benchmarks)
_spec.loader.exec_module(report_benchmarks)

ROUND_TAGS = report_benchmarks.ROUND_TAGS
FISHER_REFERENCE_ARM = report_benchmarks.FISHER_REFERENCE_ARM
build_task_report = report_benchmarks.build_task_report
render_markdown = report_benchmarks.render_markdown
write_reports = report_benchmarks.write_reports
main = report_benchmarks.main
_load_all_rows = report_benchmarks._load_all_rows


def _s5_row(seed: int, arm: str, val128: float, t256: float, t512: float, t1024: float) -> dict:
    return {
        "round": ROUND_TAGS["s5"],
        "task": "s5",
        "variant": arm,
        "layers": 2,
        "seed": seed,
        "steps": 1600,
        "acc": {"T256": t256, "T512": t512, "T1024": t1024},
        "secs": 12.3,
        "ckpt": {"step": 1600, "val128": val128},
        "config": {"device": "cuda", "torch": "2.8.0+cu128", "scan": None, "compile": False},
    }


def _pendulum_row(seed: int, arm: str, val_mse: float) -> dict:
    return {
        "round": ROUND_TAGS["pendulum"],
        "task": "pendulum",
        "variant": arm,
        "layers": 2,
        "seed": seed,
        "steps": 1600,
        "acc": {},
        "secs": 9.9,
        "ckpt": {"step": 1600, "val_mse": val_mse},
        "config": {"device": "cuda", "torch": "2.8.0+cu128"},
    }


def _psmnist_row(seed: int, arm: str, val_acc: float, test_acc: float) -> dict:
    return {
        "round": ROUND_TAGS["psmnist"],
        "task": "psmnist",
        "variant": arm,
        "layers": 2,
        "seed": seed,
        "steps": 5,
        "acc": {"test": test_acc},
        "secs": 40.0,
        "ckpt": {"epoch": 5, "val_acc": val_acc},
        "config": {"budget": {}, "permutation_seed": 20260719},
    }


# ----------------------------------------------------------- 0. round tags
def test_round_tags_bind_to_the_shared_bench_round_tags_source():
    # This module owns `ROUND_TAGS` as its own name, but binds it directly
    # to `experiments.benchmark_tasks.BENCH_ROUND_TAGS` -- the single source
    # of truth `scripts/gpu_benchmark_campaign.py` also reads -- rather than
    # hardcoding an independently-editable copy.
    assert ROUND_TAGS == BENCH_ROUND_TAGS
    assert ROUND_TAGS == {
        "s5": "bench-s5-02",
        "mqar": "bench-mqar-02",
        "psmnist": "bench-psmnist-02",
        "pendulum": "bench-pendulum-02",
    }


# --------------------------------------------------------------- 1. ledger I/O
def test_load_all_rows_skips_blank_and_malformed_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    good = _pendulum_row(0, "log", 0.001)
    path.write_text(json.dumps(good) + "\n\n" + "{not json\n" + json.dumps(good) + "\n")

    rows = _load_all_rows(path)

    assert rows == [good, good]


def test_load_all_rows_missing_file_returns_empty_list(tmp_path):
    assert _load_all_rows(tmp_path / "does_not_exist.jsonl") == []


# ------------------------------------------------------- 2. fit / robustness
def test_ge_direction_fit_counting_matches_taskspec_threshold():
    task = TASKS["s5"]
    rows = [
        _s5_row(0, "log", val128=0.995, t256=0.9, t512=0.8, t1024=0.7),  # fits
        _s5_row(1, "log", val128=0.50, t256=0.4, t512=0.3, t1024=0.2),  # does not fit
        _s5_row(2, "log", val128=task.fit_threshold, t256=1.0, t512=1.0, t1024=1.0),  # boundary
    ]

    report = build_task_report("s5", rows)

    assert report["fit_metric"] == "val128"
    assert report["fit_threshold"] == task.fit_threshold
    log_rep = report["arms"]["log"]
    assert log_rep["seeds_present"] == 3
    assert log_rep["fits"] == 2  # seed 0 and the boundary seed 2


def test_le_direction_fit_counting_for_pendulum_mse():
    task = TASKS["pendulum"]
    tau = task.fit_threshold
    rows = [
        _pendulum_row(0, "log", val_mse=tau * 0.5),  # fits: well below tau
        _pendulum_row(1, "log", val_mse=tau * 2.0),  # does not fit
        _pendulum_row(2, "log", val_mse=tau),  # fits: boundary is inclusive
    ]

    report = build_task_report("pendulum", rows)

    assert report["fit_direction"] == "le"
    assert report["arms"]["log"]["fits"] == 2


def test_robustness_triple_read_from_taskspec_not_hardcoded():
    task = TASKS["psmnist"]
    rows = [_psmnist_row(i, "log", val_acc=v, test_acc=v) for i, v in enumerate([0.89, 0.91, 0.93])]

    report = build_task_report("psmnist", rows)

    assert report["robustness_thresholds"] == list(task.robustness)
    log_robustness = report["arms"]["log"]["robustness"]
    # 0.89 clears 0.88 only; 0.91 clears 0.88/0.90; 0.93 clears all three.
    assert log_robustness[str(task.robustness[0])] == 3  # >= 0.88
    assert log_robustness[str(task.robustness[1])] == 2  # >= 0.90
    assert log_robustness[str(task.robustness[2])] == 1  # >= 0.92


# --------------------------------------------------------- 3. generalization
def test_s5_generalization_keys_are_T_prefixed_and_raw_vs_fit_only_differ():
    task = TASKS["s5"]
    rows = [
        _s5_row(0, "log", val128=1.0, t256=0.9, t512=0.8, t1024=0.7),  # fits
        _s5_row(1, "log", val128=0.1, t256=0.1, t512=0.1, t1024=0.1),  # does not fit
    ]
    assert 0.1 < task.fit_threshold <= 1.0

    report = build_task_report("s5", rows)
    log_rep = report["arms"]["log"]

    assert set(log_rep["mean_acc"]) == {"T256", "T512", "T1024"}
    assert log_rep["mean_acc"]["T256"] == (0.9 + 0.1) / 2  # raw: over all rows
    assert log_rep["fit_only_acc"]["T256"] == 0.9  # fit-only: over the one fitting row


def test_mqar_generalization_keys_carry_pair_count_suffix():
    rows = [
        {
            "round": ROUND_TAGS["mqar"],
            "task": "mqar",
            "variant": "log",
            "layers": 2,
            "seed": 0,
            "steps": 1600,
            "acc": {"T256_p16": 0.5, "T256_p32": 0.4},
            "secs": 5.0,
            "ckpt": {"step": 1600, "val_qacc": 0.99},
            "config": {},
        }
    ]

    report = build_task_report("mqar", rows)

    assert set(report["arms"]["log"]["mean_acc"]) == {"T256_p16", "T256_p32"}


def test_psmnist_generalization_key_is_test():
    rows = [_psmnist_row(0, "log", val_acc=0.95, test_acc=0.94)]

    report = build_task_report("psmnist", rows)

    assert report["arms"]["log"]["mean_acc"] == {"test": 0.94}


def test_pendulum_has_no_generalization_keys():
    rows = [_pendulum_row(0, "log", val_mse=0.001)]

    report = build_task_report("pendulum", rows)

    assert report["arms"]["log"]["mean_acc"] == {}
    assert report["arms"]["log"]["fit_only_acc"] == {}


def test_generalization_mean_tolerates_rows_missing_a_key():
    # A row without a given acc key (e.g. a partial/older row) contributes
    # to no key it lacks, rather than crashing the whole arm's table.
    rows = [
        _s5_row(0, "log", val128=1.0, t256=0.9, t512=0.8, t1024=0.7),
        {**_s5_row(1, "log", val128=1.0, t256=0.5, t512=0.5, t1024=0.5), "acc": {"T256": 0.5}},
    ]

    report = build_task_report("s5", rows)
    mean_acc = report["arms"]["log"]["mean_acc"]

    assert mean_acc["T256"] == (0.9 + 0.5) / 2
    assert mean_acc["T512"] == 0.8  # only row 0 carries T512


# ------------------------------------------------------------------- 4. Fisher
def test_fisher_vs_log_n_a_when_an_arm_has_zero_rows():
    rows = [_s5_row(0, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)]

    report = build_task_report("s5", rows)

    assert report["fisher_vs_reference"]["signed"]["p"] is None
    assert "n/a" in report["fisher_vs_reference"]["signed"]["note"]


def test_fisher_vs_log_computed_when_both_arms_have_fits_and_non_fits():
    task = TASKS["s5"]
    rows = [_s5_row(i, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0) for i in range(3)]
    rows += [_s5_row(i, "log", val128=0.0, t256=0.0, t512=0.0, t1024=0.0) for i in range(3, 6)]
    rows += [_s5_row(i, "signed", val128=1.0, t256=1.0, t512=1.0, t1024=1.0) for i in range(6, 8)]
    rows += [_s5_row(i, "signed", val128=0.0, t256=0.0, t512=0.0, t1024=0.0) for i in range(8, 12)]
    assert task.fit_threshold <= 1.0

    report = build_task_report("s5", rows)
    info = report["fisher_vs_reference"]["signed"]

    assert info["fits"] == "2/6"
    assert info["reference_fits"] == "3/6"
    assert 0.0 <= info["p"] <= 1.0


def test_log_arm_itself_is_not_in_fisher_vs_reference():
    rows = [_s5_row(0, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)]
    report = build_task_report("s5", rows)
    assert FISHER_REFERENCE_ARM not in report["fisher_vs_reference"]


# --------------------------------------------------- 4b. duplicate-seed guard
def test_duplicate_seed_rows_raise_instead_of_inflating_fisher_denominator():
    """Two rows sharing a seed for the same arm (a ledger dedup failure, or
    a genuine duplicate) must raise rather than silently inflate
    `seeds_present`/the Fisher-exact denominator (pre-matrix technical
    review, item 7)."""
    rows = [
        _s5_row(0, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0),
        _s5_row(0, "log", val128=0.5, t256=0.5, t512=0.5, t1024=0.5),  # duplicate seed 0
    ]
    with pytest.raises(ValueError, match="distinct seed"):
        build_task_report("s5", rows)


def test_distinct_seeds_across_arms_do_not_trip_the_duplicate_guard():
    """The same seed appearing once for two DIFFERENT arms is not a
    duplicate (each arm's rows are checked independently) -- a legitimate
    shape every task's ledger has by construction (every arm shares the
    seed matrix)."""
    rows = [
        _s5_row(0, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0),
        _s5_row(0, "signed", val128=0.5, t256=0.5, t512=0.5, t1024=0.5),
    ]
    report = build_task_report("s5", rows)  # must not raise
    assert report["arms"]["log"]["seeds_present"] == 1
    assert report["arms"]["signed"]["seeds_present"] == 1


# --------------------------------------------------------------- 5. completeness
def test_completeness_reports_present_and_missing_seeds():
    task = TASKS["psmnist"]
    rows = [
        _psmnist_row(0, "log", val_acc=0.95, test_acc=0.94),
        _psmnist_row(2, "log", val_acc=0.95, test_acc=0.94),
    ]

    report = build_task_report("psmnist", rows)
    log_rep = report["arms"]["log"]

    assert log_rep["seeds_planned"] == task.seeds
    assert log_rep["present_seeds"] == [0, 2]
    assert log_rep["missing_seeds"] == [s for s in range(task.seeds) if s not in (0, 2)]


def test_completeness_all_arms_present_even_with_zero_rows():
    report = build_task_report("s5", [])
    assert set(report["arms"]) == set(MATRIX_ARMS)
    for rep in report["arms"].values():
        assert rep["seeds_present"] == 0
        assert rep["missing_seeds"] == list(range(TASKS["s5"].seeds))


# --------------------------------------------------------------------- 6. params
def test_param_counts_are_positive_and_vary_by_task_and_arm():
    report_s5 = build_task_report("s5", [])
    report_psmnist = build_task_report("psmnist", [])

    for arm in MATRIX_ARMS:
        assert report_s5["arms"][arm]["params"] > 0
        assert report_psmnist["arms"][arm]["params"] > 0

    # Different tasks have different vocab/head sizes -> different param
    # counts for the same arm; params are reported per arm, never equalized
    # across arms within a task.
    assert report_s5["arms"]["log"]["params"] != report_psmnist["arms"]["log"]["params"]
    log_params = {report_s5["arms"][arm]["params"] for arm in MATRIX_ARMS}
    assert len(log_params) == len(MATRIX_ARMS)  # every arm's count is distinct


def test_delta_arm_excludes_decay_wiring_on_pendulum():
    # DeltaMinGRU rejects decay/delta_t unconditionally (frozen contract);
    # the other four (decay-capable) arms get extra DecayMixin parameters
    # for the pendulum task specifically. This is an existing
    # `build_model`/`DECAY_CAPABLE_ARMS` invariant this report's param
    # counts must reflect, not something this test invents.
    build_task_report("pendulum", [])  # exercises build_model for every arm without raising
    assert "delta" not in DECAY_CAPABLE_ARMS
    assert "log" in DECAY_CAPABLE_ARMS


# ------------------------------------------------------------------ 7. end-to-end
def test_write_reports_creates_json_and_md_for_every_task(tmp_path):
    rows = [_s5_row(0, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)]

    reports = write_reports(rows, tmp_path)

    assert set(reports) == set(TASKS)
    for task_name in TASKS:
        json_path = tmp_path / f"bench_{task_name}.json"
        md_path = tmp_path / f"bench_{task_name}.md"
        assert json_path.exists()
        assert md_path.exists()
        on_disk = json.loads(json_path.read_text())
        assert on_disk == reports[task_name]


def test_zero_row_round_renders_explicit_0_rows_note_not_a_crash():
    report = build_task_report("mqar", [])
    md = render_markdown(report)
    assert "0 rows found for arm `log`" in md
    assert "Stratum(s) observed: none (0 rows)." in md


def test_main_regenerates_against_the_real_ledger_without_raising(tmp_path):
    # Exercises the real experiments/lab_results.jsonl (whatever bench-round
    # rows exist, including zero) through the full CLI entry point --
    # writing to a tmp_path out-dir so the real experiments/bench/ artifacts
    # are never touched by the test suite.
    exit_code = main(["--out-dir", str(tmp_path)])
    assert exit_code == 0
    for task_name in TASKS:
        assert (tmp_path / f"bench_{task_name}.json").exists()
        assert (tmp_path / f"bench_{task_name}.md").exists()


def test_regenerating_twice_is_byte_identical_given_same_rows(tmp_path):
    rows = [
        _s5_row(0, "log", val128=1.0, t256=0.9, t512=0.8, t1024=0.7),
        _s5_row(0, "signed", val128=0.5, t256=0.4, t512=0.3, t1024=0.2),
    ]
    out_a, out_b = tmp_path / "a", tmp_path / "b"

    write_reports(rows, out_a)
    write_reports(rows, out_b)

    for task_name in TASKS:
        # env.generated timestamps differ between the two runs -- strip
        # before comparing, everything else must be byte-identical.
        a = json.loads((out_a / f"bench_{task_name}.json").read_text())
        b = json.loads((out_b / f"bench_{task_name}.json").read_text())
        a["env"]["generated"] = b["env"]["generated"] = None
        assert a == b


# ------------------------- 8. regression: -02 matrix accounting untouched --
# (S5-only probe round, PROBE_ARMS -- Amendments, 2026-07-20 entry): the
# `-02` matrix reports' planned-arm accounting must stay exactly the nine
# clean seed-matrix arms (the eight MinGRUStack mixer arms plus `gru`, the
# classical control arm added by a fourth amendment), never widened by the
# three probe arms added to `experiments.benchmark_lab.ARM_REGISTRY`
# alongside them.
def test_matrix_arms_planned_set_is_exactly_the_nine_matrix_arms():
    assert set(MATRIX_ARMS) == {
        "log",
        "signed",
        "rotation",
        "rotation-hetero",
        "givens",
        "delta",
        "signed-givens",
        "signed-delta",
        "gru",
    }
    assert len(MATRIX_ARMS) == 9
    # PROBE_ARMS is disjoint and joins ARM_REGISTRY, but never MATRIX_ARMS.
    assert set(PROBE_ARMS) & set(MATRIX_ARMS) == set()
    assert set(ARM_REGISTRY) == set(MATRIX_ARMS) | set(PROBE_ARMS)
    assert len(ARM_REGISTRY) == 12


def test_probe_round_rows_never_appear_in_a_minus02_matrix_report():
    """A row tagged under the probe round (`BENCH_PROBE_ROUND_TAGS["s5"]`)
    with a probe-arm `variant` must never surface in the `-02` matrix
    report: `_rows_for_task` filters by round tag FIRST (the probe tag
    never equals `ROUND_TAGS["s5"]`), so the row is dropped before variant
    matching is even attempted."""
    probe_row = _s5_row(0, "rotation-hetero-k5", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    probe_row["round"] = BENCH_PROBE_ROUND_TAGS["s5"]
    matrix_row = _s5_row(0, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)

    report = build_task_report("s5", [probe_row, matrix_row])

    assert set(report["arms"]) == set(MATRIX_ARMS)
    assert "rotation-hetero-k5" not in report["arms"]
    assert report["arms"]["log"]["seeds_present"] == 1


def test_unrecognized_probe_variant_under_the_matrix_round_tag_is_silently_dropped():
    """A row that somehow carried the MATRIX `-02` round tag but a
    probe-only `variant` (never emitted by the campaign -- `PROBE_ARMS`
    always writes under the distinct probe tag) is still silently dropped,
    matching `_rows_for_task`'s documented "unrecognized variant" contract
    -- it must not crash the report or fabricate a ninth arm column."""
    stray_row = _s5_row(0, "signed-delta-nh3", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    matrix_row = _s5_row(0, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)

    report = build_task_report("s5", [stray_row, matrix_row])

    assert set(report["arms"]) == set(MATRIX_ARMS)
    assert "signed-delta-nh3" not in report["arms"]
    assert report["arms"]["log"]["seeds_present"] == 1
