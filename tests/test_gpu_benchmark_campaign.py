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
  ``TimeoutExpired``, ``OSError``) via a monkeypatched
  ``subprocess.run`` -- these never reach ``_run_selftest_gate`` through the
  end-to-end tests below (``--device cpu`` skips pre-flight entirely;
  ``--device cuda`` trips the CUDA assert before the selftest gate on this
  CUDA-less test machine), so they need direct unit coverage.
- End-to-end subprocess checks per the task's stated verification: a tiny
  ``--device cpu`` smoke run emits a valid ``MINGRU_LAB_ROW`` line and never
  touches ``experiments/lab_results.jsonl``, and the default (``--device
  cuda``) invocation fails fast at pre-flight with a clear, non-traceback
  message on this CUDA-less test machine. These import torch (via the
  subprocess) and are slower, so they're kept to one case each.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from experiments.benchmark_lab import ARM_REGISTRY
from experiments.benchmark_tasks import TASKS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "gpu_benchmark_campaign.py"

_spec = importlib.util.spec_from_file_location("gpu_benchmark_campaign", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
gpu_benchmark_campaign = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("gpu_benchmark_campaign", gpu_benchmark_campaign)
_spec.loader.exec_module(gpu_benchmark_campaign)

_ROUND_TAGS = gpu_benchmark_campaign._ROUND_TAGS
_resolve_seeds = gpu_benchmark_campaign._resolve_seeds
_resolve_eval_every = gpu_benchmark_campaign._resolve_eval_every
_run_selftest_gate = gpu_benchmark_campaign._run_selftest_gate
_run_arm_kwargs = gpu_benchmark_campaign._run_arm_kwargs


# --- round tags: Global Constraints exactness ------------------------------


def test_round_tags_match_global_constraints():
    assert _ROUND_TAGS == {
        "s5": "bench-s5-01",
        "mqar": "bench-mqar-01",
        "psmnist": "bench-psmnist-01",
        "pendulum": "bench-pendulum-01",
    }


def test_round_tags_cover_every_registered_task():
    assert set(_ROUND_TAGS) == set(TASKS)


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
        "bench-s5-01", TASKS["s5"], "givens", seed=7, steps=None, device="cuda"
    )
    assert kwargs["round_tag"] == "bench-s5-01"
    assert kwargs["task"] is TASKS["s5"]
    assert kwargs["arm"] == "givens"
    assert kwargs["seed"] == 7
    assert kwargs["device"] == "cuda"


def test_run_arm_kwargs_shrinks_eval_every_alongside_steps_override():
    kwargs = _run_arm_kwargs("bench-s5-01", TASKS["s5"], "log", seed=0, steps=20, device="cpu")
    assert kwargs["steps"] == 20
    assert kwargs["eval_every"] == 20


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
    assert row["round"] == "bench-s5-01"
    assert row["task"] == "s5"
    assert row["variant"] == "log"
    assert row["seed"] == 0

    # benchmark_lab.run_arm's own bare (unmarked) row print must be
    # suppressed (redirect_stdout around the run_arm call) -- the row's
    # distinctive "round" value should appear exactly once in the whole
    # transcript, not twice (once bare, once MINGRU_LAB_ROW-prefixed).
    assert result.stdout.count('"round": "bench-s5-01"') == 1, result.stdout

    # no MINGRU_LAB_ENV line on --device cpu (pre-flight is skipped)
    assert not any(
        line.startswith(gpu_benchmark_campaign._ENV_PREFIX) for line in result.stdout.splitlines()
    )

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
