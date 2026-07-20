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
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from experiments.benchmark_lab import ARM_REGISTRY, MATRIX_ARMS, PROBE_ARMS
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
_resolve_seeds = gpu_benchmark_campaign._resolve_seeds
_resolve_eval_every = gpu_benchmark_campaign._resolve_eval_every
_round_tag_for_arm = gpu_benchmark_campaign._round_tag_for_arm
_run_selftest_gate = gpu_benchmark_campaign._run_selftest_gate
_run_arm_kwargs = gpu_benchmark_campaign._run_arm_kwargs


# --- round tags: Global Constraints exactness ------------------------------


def test_round_tags_match_current_bench_round_tags():
    # Bumped -01 -> -02 at the pre-matrix technical review (2026-07-19): the
    # -01 tags are now the recorded pilot/calibration population. This
    # module's `_ROUND_TAGS` binds directly to `experiments.benchmark_tasks
    # .BENCH_ROUND_TAGS`, the single source of truth, so this test also
    # pins that the two never drift apart.
    from experiments.benchmark_tasks import BENCH_ROUND_TAGS

    assert _ROUND_TAGS == BENCH_ROUND_TAGS
    assert _ROUND_TAGS == {
        "s5": "bench-s5-02",
        "mqar": "bench-mqar-02",
        "psmnist": "bench-psmnist-02",
        "pendulum": "bench-pendulum-02",
    }


def test_round_tags_cover_every_registered_task():
    assert set(_ROUND_TAGS) == set(TASKS)


# --- probe round tags + _round_tag_for_arm: probe routing -----------------


def test_probe_round_tags_match_bench_probe_round_tags():
    from experiments.benchmark_tasks import BENCH_PROBE_ROUND_TAGS

    assert _PROBE_ROUND_TAGS == BENCH_PROBE_ROUND_TAGS
    assert _PROBE_ROUND_TAGS == {"s5": "bench-s5-probe-01"}


@pytest.mark.parametrize("arm", sorted(PROBE_ARMS))
def test_round_tag_for_arm_routes_probe_arms_to_the_probe_tag(arm):
    assert _round_tag_for_arm("s5", arm, PROBE_ARMS) == "bench-s5-probe-01"


@pytest.mark.parametrize("arm", sorted(MATRIX_ARMS))
def test_round_tag_for_arm_routes_matrix_arms_to_the_matrix_tag(arm):
    for task_name in TASKS:
        assert _round_tag_for_arm(task_name, arm, PROBE_ARMS) == _ROUND_TAGS[task_name]


def test_round_tag_for_arm_raises_for_probe_arm_on_a_task_with_no_probe_tag():
    # The probe round is S5-only (BENCH_PROBE_ROUND_TAGS has one entry) --
    # a probe arm requested against any other task must fail loud rather
    # than silently falling back to that task's matrix tag.
    for task_name in ("mqar", "psmnist", "pendulum"):
        with pytest.raises(ValueError, match="no round tag"):
            _round_tag_for_arm(task_name, "rotation-hetero-k5", PROBE_ARMS)


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


# --- main(): --arms defaults to MATRIX_ARMS only (no per-task subset) -----


def test_main_default_arms_is_matrix_arms_only(monkeypatch):
    """Confirms `scripts/gpu_benchmark_campaign.py`'s `--arms` default
    (`list(benchmark_lab.MATRIX_ARMS)`, module docstring's Task x arm
    matrix section) resolves to exactly the eight matrix arms -- including
    `signed-givens`/`signed-delta` -- with no per-task arm scoping: every
    task runs the same arm list. The three `PROBE_ARMS` must NOT be in the
    default (probe arms run only when named explicitly via `--arms`).
    Captures the arguments `main` actually passes to `run_campaign` rather
    than asserting on argparse internals directly."""
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
            "rotation-hetero-k5",
            "signed-delta-nh3",
            "signed-delta-nh4",
            "--seeds",
            "0",
        ]
    )

    assert set(captured["arms"]) == set(PROBE_ARMS)


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
    rotation-hetero-k5 signed-delta-nh3 signed-delta-nh4` on `s5` emits
    three rows, all tagged `bench-s5-probe-01` -- never the matrix
    `bench-s5-02` tag."""
    ledger = _REPO_ROOT / "experiments" / "lab_results.jsonl"
    before = ledger.stat().st_mtime_ns if ledger.exists() else None

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--tasks",
            "s5",
            "--arms",
            "rotation-hetero-k5",
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
        "rotation-hetero-k5",
        "signed-delta-nh3",
        "signed-delta-nh4",
    }
    for row in rows:
        assert row["round"] == "bench-s5-probe-01"
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
