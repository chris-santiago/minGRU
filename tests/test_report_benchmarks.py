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
9. `REF_ARMS` grounding reference (`gru-large`, fifth amendment) -- a
   distinct `build_ref_task_report`/`render_ref_markdown`/
   `write_ref_reports` path writing `bench_<task>_ref.{json,md}`, never
   the matched `bench_<task>.{json,md}` pair; no Fisher-vs-`log` section;
   rows carrying either the ref round tag or a ref-only variant never
   surface in a `-02` report, mirroring section 8's probe regression.
10. `PROBE_ARMS` S5 design-correction probe (task-11a) -- the mirror-image
    `build_probe_task_report`/`render_probe_markdown`/`write_probe_reports`
    path writing `bench_s5_probe.{json,md}`, sharing section 9's
    `_build_population_task_report`/`_render_population_markdown` helper;
    no Fisher-vs-`log` section; a PROBE banner instead of REFERENCE; only
    `s5` has a registered round tag.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from experiments.benchmark_lab import (
    ARM_REGISTRY,
    DECAY_CAPABLE_ARMS,
    MATRIX_ARMS,
    PROBE_ARMS,
    REF_ARMS,
)
from experiments.benchmark_tasks import (
    BENCH_ARM_ROUND_OVERRIDES,
    BENCH_PROBE_ROUND_TAGS,
    BENCH_REF_ROUND_TAGS,
    BENCH_ROUND_TAGS,
    TASKS,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "report_benchmarks.py"
_spec = importlib.util.spec_from_file_location("report_benchmarks", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
report_benchmarks = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("report_benchmarks", report_benchmarks)
_spec.loader.exec_module(report_benchmarks)

ROUND_TAGS = report_benchmarks.ROUND_TAGS
REF_ROUND_TAGS = report_benchmarks.REF_ROUND_TAGS
PROBE_ROUND_TAGS = report_benchmarks.PROBE_ROUND_TAGS
FISHER_REFERENCE_ARM = report_benchmarks.FISHER_REFERENCE_ARM
build_task_report = report_benchmarks.build_task_report
build_ref_task_report = report_benchmarks.build_ref_task_report
build_probe_task_report = report_benchmarks.build_probe_task_report
render_markdown = report_benchmarks.render_markdown
render_ref_markdown = report_benchmarks.render_ref_markdown
render_probe_markdown = report_benchmarks.render_probe_markdown
write_reports = report_benchmarks.write_reports
write_ref_reports = report_benchmarks.write_ref_reports
write_probe_reports = report_benchmarks.write_probe_reports
main = report_benchmarks.main
_load_all_rows = report_benchmarks._load_all_rows
_source_round_for_arm = report_benchmarks._source_round_for_arm
ARM_ROUND_OVERRIDES = report_benchmarks.ARM_ROUND_OVERRIDES


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
        "churn": "bench-churn-02",
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
    # DeltaMinGRU now supports decay (Gated-DeltaNet-style), but the delta
    # benchmark arm is deliberately excluded from the decay-capable set here
    # (decay-arm benchmark evidence is a deferred round); the other four
    # (decay-capable) arms get extra DecayMixin parameters
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
        "signed-rotation",
        "givens",
        "delta",
        "signed-givens",
        "signed-delta",
        "gru",
    }
    assert len(MATRIX_ARMS) == 9
    # PROBE_ARMS and REF_ARMS are each disjoint from MATRIX_ARMS (and from
    # each other) and join ARM_REGISTRY, but never MATRIX_ARMS.
    assert set(PROBE_ARMS) & set(MATRIX_ARMS) == set()
    assert set(REF_ARMS) & set(MATRIX_ARMS) == set()
    assert set(ARM_REGISTRY) == set(MATRIX_ARMS) | set(PROBE_ARMS) | set(REF_ARMS)
    assert len(REF_ARMS) == 1
    assert len(ARM_REGISTRY) == 13


def test_probe_round_rows_never_appear_in_a_minus02_matrix_report():
    """A row tagged under the probe round (`BENCH_PROBE_ROUND_TAGS["s5"]`)
    with a probe-arm `variant` must never surface in the `-02` matrix
    report: `_rows_for_task` filters by round tag FIRST (the probe tag
    never equals `ROUND_TAGS["s5"]`), so the row is dropped before variant
    matching is even attempted."""
    probe_row = _s5_row(0, "signed-rotation-k5", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    probe_row["round"] = BENCH_PROBE_ROUND_TAGS["s5"]
    matrix_row = _s5_row(0, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)

    report = build_task_report("s5", [probe_row, matrix_row])

    assert set(report["arms"]) == set(MATRIX_ARMS)
    assert "signed-rotation-k5" not in report["arms"]
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


# --------------------------- 9. REF_ARMS grounding reference (gru-large) --
# (psMNIST-only reference round, REF_ARMS -- Amendments, 2026-07-20
# "gru-large grounding reference" entry): a literature-scale gru-large arm,
# reported through a DISTINCT build_ref_task_report/render_ref_markdown/
# write_ref_reports path -- bench_<task>_ref.{json,md}, never the matched
# bench_<task>.{json,md} pair -- so the matched -02 accounting stays
# byte-identical regardless of whether REF_ARMS rows exist.
def test_ref_round_tags_match_bench_ref_round_tags():
    assert REF_ROUND_TAGS == BENCH_REF_ROUND_TAGS
    assert REF_ROUND_TAGS == {"psmnist": "bench-psmnist-ref-01"}


def test_gru_large_ref_row_never_appears_in_a_minus02_matrix_report():
    """A row tagged under the reference round (`BENCH_REF_ROUND_TAGS
    ["psmnist"]`) with the `gru-large` variant must never surface in the
    `-02` matrix report: `_rows_for_task` filters by round tag FIRST (the
    ref tag never equals `ROUND_TAGS["psmnist"]`), so the row is dropped
    before variant matching is even attempted -- mirroring the probe-round
    regression test above."""
    ref_row = _psmnist_row(0, "gru-large", val_acc=0.93, test_acc=0.92)
    ref_row["round"] = BENCH_REF_ROUND_TAGS["psmnist"]
    matrix_row = _psmnist_row(0, "log", val_acc=0.8, test_acc=0.78)

    report = build_task_report("psmnist", [ref_row, matrix_row])

    assert set(report["arms"]) == set(MATRIX_ARMS)
    assert "gru-large" not in report["arms"]
    assert report["arms"]["log"]["seeds_present"] == 1


def test_unrecognized_ref_variant_under_the_matrix_round_tag_is_silently_dropped():
    """A row that somehow carried the MATRIX `-02` round tag but the
    ref-only `gru-large` variant (never emitted by the campaign --
    `REF_ARMS` always writes under the distinct ref tag) is still silently
    dropped -- must not crash the report or fabricate a tenth arm column."""
    stray_row = _psmnist_row(0, "gru-large", val_acc=0.93, test_acc=0.92)
    matrix_row = _psmnist_row(0, "log", val_acc=0.8, test_acc=0.78)

    report = build_task_report("psmnist", [stray_row, matrix_row])

    assert set(report["arms"]) == set(MATRIX_ARMS)
    assert "gru-large" not in report["arms"]
    assert report["arms"]["log"]["seeds_present"] == 1


def test_build_ref_task_report_reads_only_the_ref_round_and_ref_arms():
    """`build_ref_task_report` is the mirror image of the regression tests
    above: a matched `-02` row for `gru-large` (wrong round) and a matched
    `log` row under the ref tag (wrong arm) must both be excluded from the
    reference report -- only a `gru-large` row under the ref tag counts."""
    wrong_round = _psmnist_row(0, "gru-large", val_acc=0.93, test_acc=0.92)  # matched -02 tag
    wrong_arm = _psmnist_row(1, "log", val_acc=0.8, test_acc=0.78)
    wrong_arm["round"] = BENCH_REF_ROUND_TAGS["psmnist"]
    real_ref_row = _psmnist_row(2, "gru-large", val_acc=0.93, test_acc=0.92)
    real_ref_row["round"] = BENCH_REF_ROUND_TAGS["psmnist"]

    report = build_ref_task_report("psmnist", [wrong_round, wrong_arm, real_ref_row])

    assert set(report["arms"]) == set(REF_ARMS)
    assert report["arms"]["gru-large"]["seeds_present"] == 1
    assert report["arms"]["gru-large"]["present_seeds"] == [2]


def test_build_ref_task_report_has_no_fisher_section():
    """Deliberately no Fisher-vs-`log` comparison in the reference report
    (`build_ref_task_report`'s docstring): a ref arm's rows run under a
    distinct training budget from the matched population `log`'s fit rate
    is computed under, so comparing them would silently mix two strata."""
    report = build_ref_task_report("psmnist", [])
    assert "fisher_vs_reference" not in report
    assert "fisher_reference_arm" not in report
    assert report["reference"] is True


def test_build_ref_task_report_raises_for_a_task_with_no_ref_round_tag():
    for task_name in ("s5", "mqar", "pendulum"):
        with pytest.raises(ValueError, match="no REF_ARMS round tag"):
            build_ref_task_report(task_name, [])


def test_render_ref_markdown_labels_reference_and_omits_fisher_section():
    report = build_ref_task_report("psmnist", [])
    md = render_ref_markdown(report)
    assert "REFERENCE arm(s)" in md
    assert "NOT part of the matched" in md
    assert "gru-large" in md
    assert "Fisher exact" not in md
    assert "0 rows found for arm `gru-large`" in md


def test_write_ref_reports_writes_distinct_filenames_from_the_matched_pair(tmp_path):
    """`write_ref_reports` must write `bench_<task>_ref.{json,md}` --
    distinct filenames from `write_reports`'s `bench_<task>.{json,md}` --
    so calling it never touches (or collides with) the matched output."""
    ref_row = _psmnist_row(0, "gru-large", val_acc=0.93, test_acc=0.92)
    ref_row["round"] = BENCH_REF_ROUND_TAGS["psmnist"]

    matched_reports = write_reports([], tmp_path)
    ref_reports = write_ref_reports([ref_row], tmp_path)

    assert set(ref_reports) == set(REF_ROUND_TAGS)
    assert (tmp_path / "bench_psmnist_ref.json").exists()
    assert (tmp_path / "bench_psmnist_ref.md").exists()
    # The matched files are untouched by the ref write (still 0-row, since
    # matched_reports was built from an empty ledger).
    on_disk_matched = json.loads((tmp_path / "bench_psmnist.json").read_text())
    assert on_disk_matched == matched_reports["psmnist"]
    assert on_disk_matched["arms"]["log"]["seeds_present"] == 0


# ------------------------- 10. PROBE_ARMS S5 design-correction probe -----
# (task-11a): the mirror-image `build_probe_task_report`/
# `render_probe_markdown`/`write_probe_reports` path, sharing section 9's
# `_build_population_task_report`/`_render_population_markdown` helper --
# writes `bench_s5_probe.{json,md}`, never the matched `bench_s5.{json,md}`
# or the reference `bench_psmnist_ref.{json,md}` pair.
def test_probe_round_tags_match_bench_probe_round_tags():
    assert PROBE_ROUND_TAGS == BENCH_PROBE_ROUND_TAGS
    assert PROBE_ROUND_TAGS == {"s5": "bench-s5-probe-01"}


def test_build_probe_task_report_reads_only_the_probe_round_and_probe_arms():
    """Mirrors `test_build_ref_task_report_reads_only_the_ref_round_and_ref_arms`:
    a matched `-02` row for a probe arm (wrong round) and a matched `log`
    row under the probe tag (wrong arm) must both be excluded -- only a
    probe-arm row under the probe tag counts."""
    wrong_round = _s5_row(0, "signed-delta-nh4", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    wrong_arm = _s5_row(1, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    wrong_arm["round"] = BENCH_PROBE_ROUND_TAGS["s5"]
    real_probe_row = _s5_row(2, "signed-delta-nh4", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    real_probe_row["round"] = BENCH_PROBE_ROUND_TAGS["s5"]

    report = build_probe_task_report("s5", [wrong_round, wrong_arm, real_probe_row])

    assert set(report["arms"]) == set(PROBE_ARMS)
    assert report["arms"]["signed-delta-nh4"]["seeds_present"] == 1
    assert report["arms"]["signed-delta-nh4"]["present_seeds"] == [2]


def test_build_probe_task_report_has_no_fisher_section():
    """Deliberately no Fisher-vs-`log` comparison in the probe report
    either (`build_probe_task_report`'s docstring): a descriptive
    design-correction population, not a competing arm judged against
    `log`'s matched fit rate."""
    report = build_probe_task_report("s5", [])
    assert "fisher_vs_reference" not in report
    assert "fisher_reference_arm" not in report
    assert report["probe"] is True


def test_build_probe_task_report_raises_for_a_task_with_no_probe_round_tag():
    for task_name in ("mqar", "psmnist", "pendulum"):
        with pytest.raises(ValueError, match="no PROBE_ARMS round tag"):
            build_probe_task_report(task_name, [])


def test_probe_round_matches_the_three_probe_arms_exactly():
    report = build_probe_task_report("s5", [])
    assert set(report["arms"]) == {"signed-rotation-k5", "signed-delta-nh3", "signed-delta-nh4"}


def test_render_probe_markdown_labels_probe_and_omits_fisher_section():
    report = build_probe_task_report("s5", [])
    md = render_probe_markdown(report)
    assert "PROBE arm(s)" in md
    assert "NOT part of the matched" in md
    assert "bench-s5-probe-01" in md
    assert "signed-delta-nh3" in md
    assert "Fisher exact" not in md
    assert "0 rows found for arm `signed-rotation-k5`" in md


def test_write_probe_reports_writes_distinct_filenames_from_matched_and_ref(tmp_path):
    """`write_probe_reports` must write `bench_<task>_probe.{json,md}` --
    distinct filenames from both `write_reports`'s `bench_<task>.{json,md}`
    and `write_ref_reports`'s `bench_<task>_ref.{json,md}` -- so calling it
    never touches (or collides with) either."""
    probe_row = _s5_row(0, "signed-delta-nh4", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    probe_row["round"] = BENCH_PROBE_ROUND_TAGS["s5"]
    ref_row = _psmnist_row(0, "gru-large", val_acc=0.93, test_acc=0.92)
    ref_row["round"] = BENCH_REF_ROUND_TAGS["psmnist"]

    matched_reports = write_reports([], tmp_path)
    ref_reports = write_ref_reports([ref_row], tmp_path)
    probe_reports = write_probe_reports([probe_row], tmp_path)

    assert set(probe_reports) == set(PROBE_ROUND_TAGS)
    assert (tmp_path / "bench_s5_probe.json").exists()
    assert (tmp_path / "bench_s5_probe.md").exists()
    # The matched and ref files are untouched by the probe write.
    on_disk_matched = json.loads((tmp_path / "bench_s5.json").read_text())
    assert on_disk_matched == matched_reports["s5"]
    assert on_disk_matched["arms"]["log"]["seeds_present"] == 0
    on_disk_ref = json.loads((tmp_path / "bench_psmnist_ref.json").read_text())
    assert on_disk_ref == ref_reports["psmnist"]


def test_main_writes_the_probe_report_alongside_matched_and_ref(tmp_path):
    """End-to-end: `main` must write `bench_s5_probe.{json,md}` in the
    same run as the matched and reference reports, from the real ledger."""
    exit_code = main(["--out-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "bench_s5_probe.json").exists()
    assert (tmp_path / "bench_s5_probe.md").exists()


# ------------------ 11. round-tag correction override (read-side resolver) --
# (design spec `.claude/output/specs/2026-07-21-round-tag-override-design
# .md`, §6 "shared read-side resolver"): `_source_round_for_arm` is the
# READ half of a paired write/read override rule -- `gpu_benchmark_campaign
# .py`'s `_round_tag_for_arm` is the WRITE half, both resolving the same
# `BENCH_ARM_ROUND_OVERRIDES` map with the identical override-first rule.
# `signed-rotation` (all four matrix tasks) and `signed-rotation-k5` (S5
# only) are the only two arms populated with a correction tag right now;
# every other arm/task pair falls through to the primary tag unchanged.
def test_source_round_for_arm_returns_the_override_tag_when_registered():
    rotation_tags = BENCH_ARM_ROUND_OVERRIDES["signed-rotation"]
    k5_tags = BENCH_ARM_ROUND_OVERRIDES["signed-rotation-k5"]
    assert _source_round_for_arm("s5", "signed-rotation", "primary-tag") == rotation_tags["s5"]
    assert _source_round_for_arm("mqar", "signed-rotation", "primary-tag") == rotation_tags["mqar"]
    assert _source_round_for_arm("s5", "signed-rotation-k5", "primary-tag") == k5_tags["s5"]
    # The override tag must differ from the primary tag passed in, or the
    # test could pass by coincidence rather than by routing.
    assert rotation_tags["s5"] != "primary-tag"


def test_source_round_for_arm_falls_through_to_primary_tag_when_unregistered():
    """An arm with no override entry (`log`) and an overridden arm on a
    task it has no override for (`signed-rotation-k5` on `mqar`, which only
    has an `s5` entry) both fall through to `primary_tag` unchanged."""
    assert _source_round_for_arm("s5", "log", "primary-tag") == "primary-tag"
    assert _source_round_for_arm("mqar", "signed-rotation-k5", "primary-tag") == "primary-tag"


def test_source_round_for_arm_empty_override_map_is_behavior_neutral(monkeypatch):
    """With `ARM_ROUND_OVERRIDES` empty, every `(task, arm)` resolves to
    `primary_tag` -- the mechanism must be inert until a correction round
    is declared (spec §6 contract, §7 "empty/absent override is
    behavior-neutral")."""
    monkeypatch.setattr(report_benchmarks, "ARM_ROUND_OVERRIDES", {})
    for task_name, arm in (
        ("s5", "signed-rotation"),
        ("mqar", "signed-rotation"),
        ("psmnist", "signed-rotation"),
        ("pendulum", "signed-rotation"),
        ("s5", "signed-rotation-k5"),
        ("s5", "log"),
    ):
        assert _source_round_for_arm(task_name, arm, "primary-tag") == "primary-tag"


def test_matrix_report_sources_the_overridden_arm_from_its_correction_tag():
    """`build_task_report` (the matrix path, primary = `ROUND_TAGS`) must
    read `signed-rotation`'s rows from its correction tag, not the primary
    `-02` matrix tag: a row under the correction tag counts, a row for the
    same arm under the superseded primary tag does not. A same-tag
    non-overridden arm (`log`) is unaffected by any of this."""
    corrected = _s5_row(0, "signed-rotation", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    corrected["round"] = BENCH_ARM_ROUND_OVERRIDES["signed-rotation"]["s5"]
    superseded = _s5_row(1, "signed-rotation", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    superseded["round"] = ROUND_TAGS["s5"]  # old primary tag -- must not count
    unaffected = _s5_row(0, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)

    report = build_task_report("s5", [corrected, superseded, unaffected])

    assert report["arms"]["signed-rotation"]["seeds_present"] == 1
    assert report["arms"]["signed-rotation"]["present_seeds"] == [0]
    assert report["arms"]["log"]["seeds_present"] == 1


def test_population_report_sources_the_overridden_probe_arm_from_its_correction_tag():
    """`build_probe_task_report` (the population path, primary =
    `PROBE_ROUND_TAGS`) must read `signed-rotation-k5`'s S5 rows from its
    correction tag, mirroring the matrix-path test above. A same-tag
    non-overridden probe arm (`signed-delta-nh4`) is unaffected."""
    corrected = _s5_row(0, "signed-rotation-k5", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    corrected["round"] = BENCH_ARM_ROUND_OVERRIDES["signed-rotation-k5"]["s5"]
    superseded = _s5_row(1, "signed-rotation-k5", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    superseded["round"] = BENCH_PROBE_ROUND_TAGS["s5"]  # old primary probe tag -- must not count
    unaffected = _s5_row(0, "signed-delta-nh4", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    unaffected["round"] = BENCH_PROBE_ROUND_TAGS["s5"]

    report = build_probe_task_report("s5", [corrected, superseded, unaffected])

    assert report["arms"]["signed-rotation-k5"]["seeds_present"] == 1
    assert report["arms"]["signed-rotation-k5"]["present_seeds"] == [0]
    assert report["arms"]["signed-delta-nh4"]["seeds_present"] == 1


def test_probe_raise_contract_still_green_under_the_override_fall_through():
    """The population raise-for-unregistered-task guard
    (`_build_population_task_report`) must still run before any override
    use: `signed-rotation-k5` has no override entry for mqar/psmnist/
    pendulum, and those tasks have no `PROBE_ROUND_TAGS` entry either, so
    `build_probe_task_report` must still raise for them (spec §7
    "fall-through preserves fail-loud"; mirrors the existing test at
    `test_build_probe_task_report_raises_for_a_task_with_no_probe_round_tag`)."""
    for task_name in ("mqar", "psmnist", "pendulum"):
        with pytest.raises(ValueError, match="no PROBE_ARMS round tag"):
            build_probe_task_report(task_name, [])


# --------------------------- 12. disclosure (payload key + provenance line) --
# (design spec §4 "Disclosure", §6 "disclosure record"): a report whose arm
# set includes an overridden arm must record `arm_round_overrides` in its
# JSON payload and render a provenance line naming the arm, its correction
# round, and the primary round every other arm came from; a report with no
# overridden arm must serialize/render exactly as before (empty payload
# key, no provenance line). Both cases must discriminate -- a test that
# passes regardless of whether the override is disclosed proves nothing.
def test_matrix_report_discloses_the_overridden_arm_present_case():
    """PRESENT case: `signed-rotation` (a real, currently-populated
    override) is in `MATRIX_ARMS`, so a matrix report's
    `arm_round_overrides` must be non-empty and name it, and the rendered
    Markdown must carry a provenance line naming the arm, its correction
    round, and the primary `-02` round."""
    row = _s5_row(0, "signed-rotation", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    row["round"] = BENCH_ARM_ROUND_OVERRIDES["signed-rotation"]["s5"]

    report = build_task_report("s5", [row])

    assert report["arm_round_overrides"] == {"signed-rotation": row["round"]}
    md = render_markdown(report)
    assert "signed-rotation" in md
    assert row["round"] in md
    assert ROUND_TAGS["s5"] in md
    assert "Provenance" in md


def test_matrix_report_discloses_nothing_absent_case(monkeypatch):
    """ABSENT case: with the override map emptied (no overridden arm in
    the arm set), `arm_round_overrides` must be empty and the rendered
    Markdown must carry no provenance line -- output otherwise unchanged
    from the pre-disclosure shape."""
    monkeypatch.setattr(report_benchmarks, "ARM_ROUND_OVERRIDES", {})
    row = _s5_row(0, "log", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)

    report = build_task_report("s5", [row])

    assert report["arm_round_overrides"] == {}
    md = render_markdown(report)
    assert "Provenance" not in md


def test_probe_report_discloses_the_overridden_arm_present_case():
    """PRESENT case for the population-report path: `signed-rotation-k5`
    is a real, currently-populated override on `s5`, so the probe report's
    `arm_round_overrides` must be non-empty and name it, and the rendered
    Markdown must carry a provenance line."""
    row = _s5_row(0, "signed-rotation-k5", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    row["round"] = BENCH_ARM_ROUND_OVERRIDES["signed-rotation-k5"]["s5"]

    report = build_probe_task_report("s5", [row])

    assert report["arm_round_overrides"] == {"signed-rotation-k5": row["round"]}
    md = render_probe_markdown(report)
    assert "signed-rotation-k5" in md
    assert row["round"] in md
    assert "Provenance" in md


def test_probe_report_discloses_nothing_absent_case(monkeypatch):
    """ABSENT case for the population-report path: with the override map
    emptied, `signed-delta-nh4` (no override entry regardless) resolves
    from the population's own probe tag, so `arm_round_overrides` must be
    empty and the rendered Markdown must carry no provenance line."""
    monkeypatch.setattr(report_benchmarks, "ARM_ROUND_OVERRIDES", {})
    row = _s5_row(0, "signed-delta-nh4", val128=1.0, t256=1.0, t512=1.0, t1024=1.0)
    row["round"] = BENCH_PROBE_ROUND_TAGS["s5"]

    report = build_probe_task_report("s5", [row])

    assert report["arm_round_overrides"] == {}
    md = render_probe_markdown(report)
    assert "Provenance" not in md
