"""Regression tests for ``scripts/gpu_benchmark_campaign.py`` (accepted-
benchmark validation round, in-job campaign script, Task 5).

``scripts/`` is not an importable package (no ``__init__.py`` -- see
``tests/test_gpu_hetero_campaign.py``'s identical note), so this module is
loaded directly by file path via ``importlib``.

Split into two groups, mirroring ``test_gpu_hetero_campaign.py``:

- Pure, CUDA-free logic: the round-tag mapping (Global Constraints
  exactness), ``_resolve_seeds``'s per-task default vs. override behavior,
  ``_resolve_eval_every``'s steps-override shrink, the ``_run_arm_kwargs``
  namespace ``benchmark_lab.run_arm`` consumes (``dry_run`` always
  ``True``), and ``_run_selftest_gate``'s failure paths (non-zero exit,
  ``TimeoutExpired``, ``OSError``) plus its ``MINGRU_SCAN``-stripping
  environment fix (the production job exports ``MINGRU_SCAN=triton``
  job-wide; the selftest subprocess must not inherit it -- see
  ``_run_selftest_gate``'s docstring) via a monkeypatched ``subprocess.run``
  -- these never reach ``_run_selftest_gate`` through the end-to-end tests
  below (``--device cpu`` skips pre-flight entirely; ``--device cuda`` trips
  the CUDA assert before the selftest gate on this CUDA-less test machine),
  so they need direct unit coverage. Also pure: ``_round_tag_for_arm``'s
  probe-vs-matrix routing (S5-only probe round, ``BENCH_PROBE_ROUND_TAGS``)
  and ``main``'s default ``--arms`` (``MATRIX_ARMS`` only, probe arms
  excluded unless named explicitly).
- End-to-end subprocess checks per the task's stated verification: a tiny
  ``--device cpu`` smoke run emits a valid ``MINGRU_LAB_ROW`` line and never
  touches ``experiments/lab_results.jsonl``, and the default (``--device
  cuda``) invocation fails fast at pre-flight with a clear, non-traceback
  message on this CUDA-less test machine. A second smoke run confirms the
  three probe arms emit rows tagged ``bench-s5-probe-01``, never
  ``bench-s5-02``. These import torch (via the subprocess) and are slower,
  so they're kept to a small, fixed number of cases.

``_round_tag_for_arm``'s probe routing above also covers the ``gru-large``
reference arm's routing to ``bench-psmnist-ref-01`` (``REF_ARMS``,
``BENCH_REF_ROUND_TAGS`` -- Amendments, 2026-07-20 "gru-large grounding
reference" entry), same pattern as the S5-only probe round.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from experiments.benchmark_lab import ARM_REGISTRY, MATRIX_ARMS, PROBE_ARMS, REF_ARMS
from experiments.benchmark_tasks import TASKS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "gpu_benchmark_campaign.py"

_spec = importlib.util.spec_from_file_location("gpu_benchmark_campaign", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
gpu_benchmark_campaign = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("gpu_benchmark_campaign", gpu_benchmark_campaign)
_spec.loader.exec_module(gpu_benchmark_campaign)

_ROUND_TAGS = gpu_benchmark_campaign._ROUND_TAGS
_PROBE_ROUND_TAGS = gpu_benchmark_campaign._PROBE_ROUND_TAGS
_REF_ROUND_TAGS = gpu_benchmark_campaign._REF_ROUND_TAGS
_resolve_seeds = gpu_benchmark_campaign._resolve_seeds
_resolve_eval_every = gpu_benchmark_campaign._resolve_eval_every
_round_tag_for_arm = gpu_benchmark_campaign._round_tag_for_arm
_run_selftest_gate = gpu_benchmark_campaign._run_selftest_gate
_run_arm_kwargs = gpu_benchmark_campaign._run_arm_kwargs

# `scripts/report_benchmarks.py`'s read-side resolver, loaded the same way
# `tests/test_report_benchmarks.py` loads its own script module -- needed
# here for the write/read resolver cross-check below
# (test_round_tag_for_arm_matches_source_round_for_arm_for_every_override_entry).
_REPORT_SCRIPT_PATH = _REPO_ROOT / "scripts" / "report_benchmarks.py"
_report_spec = importlib.util.spec_from_file_location("report_benchmarks", _REPORT_SCRIPT_PATH)
assert _report_spec is not None and _report_spec.loader is not None
report_benchmarks = importlib.util.module_from_spec(_report_spec)
sys.modules.setdefault("report_benchmarks", report_benchmarks)
_report_spec.loader.exec_module(report_benchmarks)

_source_round_for_arm = report_benchmarks._source_round_for_arm


# --- round tags: Global Constraints exactness ------------------------------


def test_round_tags_match_current_bench_round_tags():
    # Bumped -01 -> -02 at the pre-matrix technical review (2026-07-19): the
    # -01 tags are now the recorded pilot/calibration population. This
    # module's `_ROUND_TAGS` binds directly to `experiments.benchmark_tasks
    # .BENCH_ROUND_TAGS`, the single source of truth, so this test also
    # pins that the two never drift apart. `churn` (Task 4, this round's own
    # design spec) never had a heterogeneous-budget `-01` pilot population
    # under this tag -- its pilot rows land under the distinct
    # `bench-churn-01` tag from the start -- but it still gets a `-02`
    # matrix tag here, same shape as every other task.
    from experiments.benchmark_tasks import BENCH_ROUND_TAGS

    assert _ROUND_TAGS == BENCH_ROUND_TAGS
    assert _ROUND_TAGS == {
        "s5": "bench-s5-02",
        "mqar": "bench-mqar-02",
        "psmnist": "bench-psmnist-02",
        "pendulum": "bench-pendulum-02",
        "churn": "bench-churn-02",
    }


def test_round_tags_cover_every_registered_task():
    assert set(_ROUND_TAGS) == set(TASKS)


# --- probe round tags + _round_tag_for_arm: probe routing -----------------


def test_probe_round_tags_match_bench_probe_round_tags():
    from experiments.benchmark_tasks import BENCH_PROBE_ROUND_TAGS

    assert _PROBE_ROUND_TAGS == BENCH_PROBE_ROUND_TAGS
    assert _PROBE_ROUND_TAGS == {"s5": "bench-s5-probe-01", "churn": "bench-churn-probe-01"}


@pytest.mark.parametrize("arm", sorted(set(PROBE_ARMS) - {"signed-rotation-k5"}))
def test_round_tag_for_arm_routes_probe_arms_to_the_probe_tag(arm):
    # `signed-rotation-k5` is excluded here -- it has a correction override
    # for s5 (round-tag override design spec) and is covered separately by
    # test_round_tag_for_arm_routes_overridden_probe_arm_to_its_correction_tag
    # below; the other two probe arms (signed-delta-nh3/nh4) are unchanged.
    assert _round_tag_for_arm("s5", arm, PROBE_ARMS, REF_ARMS) == "bench-s5-probe-01"


def test_round_tag_for_arm_routes_overridden_probe_arm_to_its_correction_tag():
    # signed-rotation-k5 (rotation-order correction re-run) has an override
    # entry for s5 only -- BENCH_ARM_ROUND_OVERRIDES["signed-rotation-k5"]
    # -- so it routes to the probe-rotfix tag instead of the plain probe tag.
    assert (
        _round_tag_for_arm("s5", "signed-rotation-k5", PROBE_ARMS, REF_ARMS)
        == "bench-s5-probe-rotfix-01"
    )


@pytest.mark.parametrize("arm", sorted(set(MATRIX_ARMS) - {"signed-rotation"}))
def test_round_tag_for_arm_routes_matrix_arms_to_the_matrix_tag(arm):
    # `signed-rotation` is excluded here -- it has a correction override on
    # all four tasks (round-tag override design spec) and is covered
    # separately by
    # test_round_tag_for_arm_routes_overridden_matrix_arm_to_its_correction_tag
    # below; every other matrix arm is unchanged.
    for task_name in TASKS:
        assert _round_tag_for_arm(task_name, arm, PROBE_ARMS, REF_ARMS) == _ROUND_TAGS[task_name]


def test_round_tag_for_arm_routes_overridden_matrix_arm_to_its_correction_tag():
    from experiments.benchmark_tasks import BENCH_ARM_ROUND_OVERRIDES

    for task_name, expected_tag in BENCH_ARM_ROUND_OVERRIDES["signed-rotation"].items():
        assert _round_tag_for_arm(task_name, "signed-rotation", PROBE_ARMS, REF_ARMS) == (
            expected_tag
        )


def test_round_tag_for_arm_override_first_then_falls_through_to_population_routing():
    """Write-side resolver contract (design spec section 6): the override
    map is consulted before probe/ref/matrix routing, for every registered
    ``(arm, task)`` pair in the source-of-truth map -- not just the two
    arms this round happens to populate -- so this test still discriminates
    correctly if the map grows. An arm/task pair absent from the map falls
    through unchanged to population routing (here, the probe tag)."""
    from experiments.benchmark_tasks import BENCH_ARM_ROUND_OVERRIDES

    for arm, per_task in BENCH_ARM_ROUND_OVERRIDES.items():
        for task_name, override_tag in per_task.items():
            assert _round_tag_for_arm(task_name, arm, PROBE_ARMS, REF_ARMS) == override_tag

    # signed-delta-nh3 has no override entry at all -- falls through to the
    # ordinary probe-tag routing, proving the override doesn't swallow
    # arms it was never told about.
    assert _round_tag_for_arm("s5", "signed-delta-nh3", PROBE_ARMS, REF_ARMS) == "bench-s5-probe-01"


def test_round_tag_for_arm_matches_source_round_for_arm_for_every_override_entry():
    """Cross-check the paired write/read resolvers (round-tag override
    design spec section 6): the write-side `_round_tag_for_arm` (this
    module) and the read-side `report_benchmarks._source_round_for_arm`
    both consult the same `BENCH_ARM_ROUND_OVERRIDES` map, but the
    override-first lookup RULE is implemented twice -- only the data is
    shared, so nothing before this test asserted the two paths actually
    agree. For every populated (arm, task) entry, both resolvers must
    return the map's own registered tag.

    `primary_tag` is a stand-in deliberately distinct from every override
    tag: if either resolver ignored the override and fell through to its
    "no override" branch, it would return `primary_tag` (or, on the write
    side, a different population tag) instead of the override tag, and the
    equality below would fail -- this is what makes the test discriminate
    rather than trivially pass."""
    from experiments.benchmark_tasks import BENCH_ARM_ROUND_OVERRIDES

    primary_tag = "not-a-real-override-tag"
    for arm, per_task in BENCH_ARM_ROUND_OVERRIDES.items():
        for task_name, override_tag in per_task.items():
            write_side = _round_tag_for_arm(task_name, arm, PROBE_ARMS, REF_ARMS)
            read_side = _source_round_for_arm(task_name, arm, primary_tag)
            assert write_side == read_side == override_tag


def test_round_tag_for_arm_raises_for_probe_arm_on_a_task_with_no_probe_tag():
    # The probe round is S5-only (BENCH_PROBE_ROUND_TAGS has one entry) --
    # a probe arm requested against any other task must fail loud rather
    # than silently falling back to that task's matrix tag. signed-rotation-k5's
    # override entry only covers s5 (BENCH_ARM_ROUND_OVERRIDES), so mqar/
    # psmnist/pendulum still have no override and no probe tag -- the raise
    # is preserved, not weakened, by the override-first check.
    for task_name in ("mqar", "psmnist", "pendulum"):
        with pytest.raises(ValueError, match="no round tag"):
            _round_tag_for_arm(task_name, "signed-rotation-k5", PROBE_ARMS, REF_ARMS)


# --- ref round tags + _round_tag_for_arm: gru-large reference routing ------


def test_ref_round_tags_match_bench_ref_round_tags():
    from experiments.benchmark_tasks import BENCH_REF_ROUND_TAGS

    assert _REF_ROUND_TAGS == BENCH_REF_ROUND_TAGS
    assert _REF_ROUND_TAGS == {"psmnist": "bench-psmnist-ref-01", "churn": "bench-churn-ref-01"}


@pytest.mark.parametrize("arm", sorted(REF_ARMS))
def test_round_tag_for_arm_routes_ref_arms_to_the_ref_tag(arm):
    assert _round_tag_for_arm("psmnist", arm, PROBE_ARMS, REF_ARMS) == "bench-psmnist-ref-01"


def test_round_tag_for_arm_raises_for_ref_arm_on_a_task_with_no_ref_tag():
    # The reference round is psMNIST-only (BENCH_REF_ROUND_TAGS has one
    # entry) -- a ref arm requested against any other task must fail loud
    # rather than silently falling back to that task's matrix tag.
    for task_name in ("s5", "mqar", "pendulum"):
        with pytest.raises(ValueError, match="no round tag"):
            _round_tag_for_arm(task_name, "gru-large", PROBE_ARMS, REF_ARMS)


# --- _resolve_seeds: per-task default vs. override -------------------------


def test_resolve_seeds_defaults_to_each_tasks_own_seed_count():
    assert _resolve_seeds(TASKS["s5"], None) == list(range(36))
    assert _resolve_seeds(TASKS["mqar"], None) == list(range(36))
    assert _resolve_seeds(TASKS["pendulum"], None) == list(range(36))
    assert _resolve_seeds(TASKS["psmnist"], None) == list(range(12))


def test_resolve_seeds_override_applies_uniformly():
    for task in TASKS.values():
        assert _resolve_seeds(task, [0, 3, 7]) == [0, 3, 7]


# --- _resolve_eval_every: steps-override shrink ----------------------------


def test_resolve_eval_every_untouched_when_steps_not_overridden():
    assert _resolve_eval_every(TASKS["s5"], None) is None


def test_resolve_eval_every_shrinks_below_full_eval_every():
    # s5's budget.eval_every is 100; a steps=20 smoke override must shrink
    # the cadence so at least one checkpoint is selected within range.
    assert TASKS["s5"].budget.eval_every == 100
    assert _resolve_eval_every(TASKS["s5"], 20) == 20


def test_resolve_eval_every_keeps_full_cadence_when_steps_exceeds_it():
    assert _resolve_eval_every(TASKS["s5"], 500) == 100


def test_resolve_eval_every_none_for_epoch_based_task():
    # psmnist has no eval_every (checkpoints once per epoch instead) --
    # a --steps override must not fabricate one.
    assert TASKS["psmnist"].budget.eval_every is None
    assert _resolve_eval_every(TASKS["psmnist"], 20) is None


# --- _run_arm_kwargs: the kwargs dict benchmark_lab.run_arm consumes -------


def test_run_arm_kwargs_always_dry_run_no_ledger_write_path():
    for task in TASKS.values():
        for arm in ARM_REGISTRY:
            kwargs = _run_arm_kwargs("some-round", task, arm, seed=0, steps=None, device="cpu")
            assert kwargs["dry_run"] is True


def test_run_arm_kwargs_carries_round_task_arm_seed_device():
    kwargs = _run_arm_kwargs(
        "bench-s5-02", TASKS["s5"], "givens", seed=7, steps=None, device="cuda"
    )
    assert kwargs["round_tag"] == "bench-s5-02"
    assert kwargs["task"] is TASKS["s5"]
    assert kwargs["arm"] == "givens"
    assert kwargs["seed"] == 7
    assert kwargs["device"] == "cuda"


def test_run_arm_kwargs_shrinks_eval_every_alongside_steps_override():
    kwargs = _run_arm_kwargs("bench-s5-02", TASKS["s5"], "log", seed=0, steps=20, device="cpu")
    assert kwargs["steps"] == 20
    assert kwargs["eval_every"] == 20


# --- _churn_row_provenance / run_campaign: churn dataset provenance -------

_churn_row_provenance = gpu_benchmark_campaign._churn_row_provenance


def test_churn_row_provenance_matches_the_loader_and_dataset_constants():
    """Design spec section 6 row contract: `config` additionally records
    `dataset_id`, `revision`, `split_seed`, the four vocab caps, and `T`.
    Every value must equal the exact constant `CHURN_TASK`'s own loader
    factory (`make_churn_loader`) is built from -- not an independently
    hardcoded literal that could drift."""
    from experiments.benchmark_tasks import (
        CHURN_SPLIT_SEED,
        ROSBANK_CURRENCY_CAP,
        ROSBANK_DATASET_ID,
        ROSBANK_MCC_CAP,
        ROSBANK_REVISION,
        ROSBANK_T,
    )

    provenance = _churn_row_provenance()

    assert provenance == {
        "dataset_id": ROSBANK_DATASET_ID,
        "revision": ROSBANK_REVISION,
        "split_seed": CHURN_SPLIT_SEED,
        "mcc_cap": ROSBANK_MCC_CAP,
        "trx_category_cap": None,
        "channel_type_cap": None,
        "currency_cap": ROSBANK_CURRENCY_CAP,
        "T": ROSBANK_T,
    }


def _fake_churn_row(*, round_tag, task, arm, seed, steps, eval_every, device, dry_run):
    return {
        "round": round_tag,
        "task": "churn",
        "variant": arm,
        "layers": 2,
        "seed": seed,
        "steps": 1,
        "acc": {"test": 0.5},
        "secs": 0.1,
        "ckpt": {"epoch": 1, "val_auc": 0.5, "selection_val_auc": 0.5},
        "config": {"budget": {"lr": 0.001, "batch_size": 8, "epochs": 1}},
    }


def test_run_campaign_merges_churn_provenance_into_row_config(monkeypatch, capsys):
    """`run_campaign` must merge `_churn_row_provenance()` into a churn
    row's `config` AFTER `benchmark_lab.run_arm` returns (module docstring:
    "run_arm itself leaves churn's own config_extra empty ... provenance
    lands at this campaign layer") -- verified on the actual printed
    `MINGRU_LAB_ROW` line, not just the helper function in isolation, and
    the merge must not drop `run_arm`'s own `budget` sub-dict."""
    import experiments.benchmark_lab as benchmark_lab

    monkeypatch.setattr(benchmark_lab, "run_arm", _fake_churn_row)

    gpu_benchmark_campaign.run_campaign(["churn"], ["log"], [0], None, "cpu")

    out = capsys.readouterr().out
    row_lines = [
        line for line in out.splitlines() if line.startswith(gpu_benchmark_campaign._ROW_PREFIX)
    ]
    assert len(row_lines) == 1
    row = json.loads(row_lines[0][len(gpu_benchmark_campaign._ROW_PREFIX) :])
    assert row["config"]["budget"] == {"lr": 0.001, "batch_size": 8, "epochs": 1}
    for key, value in _churn_row_provenance().items():
        assert row["config"][key] == value


def test_run_campaign_does_not_add_churn_provenance_to_other_tasks(monkeypatch, capsys):
    import experiments.benchmark_lab as benchmark_lab

    def _fake_s5_row(*, round_tag, task, arm, seed, steps, eval_every, device, dry_run):
        return {
            "round": round_tag,
            "task": "s5",
            "variant": arm,
            "layers": 2,
            "seed": seed,
            "steps": 1,
            "acc": {"T256": 0.5},
            "secs": 0.1,
            "ckpt": {"step": 1, "val128": 0.5, "selection_val128": 0.5},
            "config": {"budget": {"lr": 0.003, "batch_size": 8, "steps": 1, "eval_every": 1}},
        }

    monkeypatch.setattr(benchmark_lab, "run_arm", _fake_s5_row)
    gpu_benchmark_campaign.run_campaign(["s5"], ["log"], [0], None, "cpu")

    out = capsys.readouterr().out
    row_lines = [
        line for line in out.splitlines() if line.startswith(gpu_benchmark_campaign._ROW_PREFIX)
    ]
    row = json.loads(row_lines[0][len(gpu_benchmark_campaign._ROW_PREFIX) :])
    for key in _churn_row_provenance():
        assert key not in row["config"]


# --- main(): --arms defaults to MATRIX_ARMS only (no per-task subset) -----


def test_main_default_arms_is_matrix_arms_only(monkeypatch):
    """Confirms `scripts/gpu_benchmark_campaign.py`'s `--arms` default
    (`list(benchmark_lab.MATRIX_ARMS)`, module docstring's Task x arm
    matrix section) resolves to exactly the nine matrix arms -- including
    `signed-givens`/`signed-delta`/`gru` -- with no per-task arm scoping:
    every task runs the same arm list. Neither the three `PROBE_ARMS` nor
    the `REF_ARMS` grounding reference (`gru-large`) must be in the default
    (both run only when named explicitly via `--arms`). Captures the
    arguments `main` actually passes to `run_campaign` rather than
    asserting on argparse internals directly."""
    captured: dict = {}

    def _fake_run_campaign(tasks, arms, seeds, steps, device):
        captured["tasks"] = tasks
        captured["arms"] = arms
        return None

    monkeypatch.setattr(gpu_benchmark_campaign, "run_campaign", _fake_run_campaign)
    gpu_benchmark_campaign.main(["--tasks", "s5", "--seeds", "0"])

    assert set(captured["arms"]) == set(MATRIX_ARMS)
    assert "signed-givens" in captured["arms"]
    assert "signed-delta" in captured["arms"]
    for arm in PROBE_ARMS:
        assert arm not in captured["arms"]
    for arm in REF_ARMS:
        assert arm not in captured["arms"]


def test_main_arms_choices_still_include_probe_arms_for_explicit_selection(monkeypatch):
    """Probe arms are excluded from the DEFAULT `--arms` list but must
    still be explicitly selectable -- `argparse`'s `choices=` is
    `sorted(ARM_REGISTRY)` (matrix union probe), not `MATRIX_ARMS`."""
    captured: dict = {}

    def _fake_run_campaign(tasks, arms, seeds, steps, device):
        captured["arms"] = arms
        return None

    monkeypatch.setattr(gpu_benchmark_campaign, "run_campaign", _fake_run_campaign)
    gpu_benchmark_campaign.main(
        [
            "--tasks",
            "s5",
            "--arms",
            "signed-rotation-k5",
            "signed-delta-nh3",
            "signed-delta-nh4",
            "--seeds",
            "0",
        ]
    )

    assert set(captured["arms"]) == set(PROBE_ARMS)


def test_main_arms_choices_still_include_ref_arms_for_explicit_selection(monkeypatch):
    """Same as the probe-arm test above, for the `REF_ARMS` grounding
    reference (`gru-large`): excluded from the default `--arms` list but
    explicitly selectable -- `choices=sorted(ARM_REGISTRY)` covers matrix
    union probe union ref."""
    captured: dict = {}

    def _fake_run_campaign(tasks, arms, seeds, steps, device):
        captured["arms"] = arms
        return None

    monkeypatch.setattr(gpu_benchmark_campaign, "run_campaign", _fake_run_campaign)
    gpu_benchmark_campaign.main(["--tasks", "psmnist", "--arms", "gru-large", "--seeds", "0"])

    assert set(captured["arms"]) == set(REF_ARMS)


# --- _run_selftest_gate: failure paths raise a clean SystemExit -----------


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_selftest_gate_raises_clean_systemexit_on_nonzero_exit(monkeypatch):
    def _fake_run(*args, **kwargs):
        return _FakeCompletedProcess(
            returncode=3, stdout="ok so far\n", stderr="assertion failed: boom"
        )

    monkeypatch.setattr(gpu_benchmark_campaign.subprocess, "run", _fake_run)
    with pytest.raises(SystemExit) as excinfo:
        _run_selftest_gate()
    message = str(excinfo.value)
    assert "pre-flight FAILED" in message
    assert "assertion failed: boom" in message


def test_run_selftest_gate_raises_clean_systemexit_on_timeout(monkeypatch):
    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["fake", "--selftest"], timeout=120)

    monkeypatch.setattr(gpu_benchmark_campaign.subprocess, "run", _fake_run)
    with pytest.raises(SystemExit) as excinfo:
        _run_selftest_gate()
    message = str(excinfo.value)
    assert "pre-flight FAILED" in message
    assert "timed out" in message


def test_run_selftest_gate_raises_clean_systemexit_on_oserror(monkeypatch):
    def _fake_run(*args, **kwargs):
        raise OSError("no such file or directory")

    monkeypatch.setattr(gpu_benchmark_campaign.subprocess, "run", _fake_run)
    with pytest.raises(SystemExit) as excinfo:
        _run_selftest_gate()
    message = str(excinfo.value)
    assert "pre-flight FAILED" in message
    assert "could not be launched" in message


def test_run_selftest_gate_strips_mingru_scan_from_subprocess_env_only(monkeypatch):
    # Real-world bug (L4 pilot job mingru-gpu-benchmarks-9610d4d): the
    # production job exports MINGRU_SCAN=triton job-wide, and the selftest's
    # tiny models run on CPU -- inheriting that export made the CPU tensors
    # hit the Triton dispatch, which fails loud per the MINGRU_SCAN
    # contract, tripping this gate on an otherwise-healthy lab.
    monkeypatch.setenv("MINGRU_SCAN", "triton")
    monkeypatch.setenv("MINGRU_BENCHMARK_CAMPAIGN_TEST_VAR", "kept")
    captured_env: dict[str, str] = {}

    def _fake_run(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return _FakeCompletedProcess(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gpu_benchmark_campaign.subprocess, "run", _fake_run)
    _run_selftest_gate()

    assert "MINGRU_SCAN" not in captured_env
    assert os.environ["MINGRU_SCAN"] == "triton"  # parent process env untouched
    assert captured_env.get("MINGRU_BENCHMARK_CAMPAIGN_TEST_VAR") == "kept"


# --- end-to-end: CPU smoke + CUDA-less fail-fast ---------------------------


def test_cpu_smoke_emits_valid_row_and_touches_no_ledger():
    ledger = _REPO_ROOT / "experiments" / "lab_results.jsonl"
    before = ledger.stat().st_mtime_ns if ledger.exists() else None

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--tasks",
            "s5",
            "--arms",
            "log",
            "--seeds",
            "0",
            "--steps",
            "20",
            "--device",
            "cpu",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    row_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(gpu_benchmark_campaign._ROW_PREFIX)
    ]
    assert len(row_lines) == 1, result.stdout
    row = json.loads(row_lines[0][len(gpu_benchmark_campaign._ROW_PREFIX) :])
    assert row["round"] == "bench-s5-02"
    assert row["task"] == "s5"
    assert row["variant"] == "log"
    assert row["seed"] == 0

    # Honest fit metric (item 5): the re-evaluated checkpoint metric and the
    # (possibly winner's-curse-biased) selection-time value both land in
    # ckpt, under distinct keys.
    assert "val128" in row["ckpt"]
    assert "selection_val128" in row["ckpt"]

    # benchmark_lab.run_arm's own bare (unmarked) row print must be
    # suppressed (redirect_stdout around the run_arm call) -- the row's
    # distinctive "round" value should appear exactly once in the whole
    # transcript, not twice (once bare, once MINGRU_LAB_ROW-prefixed).
    assert result.stdout.count('"round": "bench-s5-02"') == 1, result.stdout

    # no MINGRU_LAB_ENV line on --device cpu (pre-flight is skipped)
    assert not any(
        line.startswith(gpu_benchmark_campaign._ENV_PREFIX) for line in result.stdout.splitlines()
    )

    after = ledger.stat().st_mtime_ns if ledger.exists() else None
    assert before == after, "CPU smoke run must never touch the ledger"


def test_cpu_smoke_probe_arms_emit_rows_tagged_with_the_probe_round():
    """Task's stated local-smoke verification: `--arms
    signed-rotation-k5 signed-delta-nh3 signed-delta-nh4` on `s5` emits
    three rows, never the matrix `bench-s5-02` tag. Per-arm assertion
    (round-tag override design spec): `signed-rotation-k5` has a
    correction override for s5 and is tagged `bench-s5-probe-rotfix-01`;
    the other two probe arms are unchanged at `bench-s5-probe-01`."""
    ledger = _REPO_ROOT / "experiments" / "lab_results.jsonl"
    before = ledger.stat().st_mtime_ns if ledger.exists() else None

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--tasks",
            "s5",
            "--arms",
            "signed-rotation-k5",
            "signed-delta-nh3",
            "signed-delta-nh4",
            "--seeds",
            "99",
            "--steps",
            "40",
            "--device",
            "cpu",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    row_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(gpu_benchmark_campaign._ROW_PREFIX)
    ]
    assert len(row_lines) == 3, result.stdout
    rows = [json.loads(line[len(gpu_benchmark_campaign._ROW_PREFIX) :]) for line in row_lines]
    assert {row["variant"] for row in rows} == {
        "signed-rotation-k5",
        "signed-delta-nh3",
        "signed-delta-nh4",
    }
    expected_round_by_variant = {
        "signed-rotation-k5": "bench-s5-probe-rotfix-01",
        "signed-delta-nh3": "bench-s5-probe-01",
        "signed-delta-nh4": "bench-s5-probe-01",
    }
    for row in rows:
        assert row["round"] == expected_round_by_variant[row["variant"]]
        assert row["task"] == "s5"
        assert row["seed"] == 99

    after = ledger.stat().st_mtime_ns if ledger.exists() else None
    assert before == after, "CPU smoke run must never touch the ledger"


def test_default_cuda_invocation_fails_fast_without_traceback():
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--tasks",
            "s5",
            "--arms",
            "log",
            "--seeds",
            "0",
            "--steps",
            "4",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "pre-flight FAILED" in result.stderr
    assert "CUDA is not available" in result.stderr
