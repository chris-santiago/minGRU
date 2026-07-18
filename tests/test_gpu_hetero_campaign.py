"""Regression tests for ``scripts/gpu_hetero_campaign.py`` (GPU 36-seed round,
Task 2).

``scripts/`` is not an importable package (no ``__init__.py`` -- see
``tests/test_gpu_check.py``'s identical note), so this module is loaded
directly by file path via ``importlib``.

Split into two groups:

- Pure, CUDA-free logic: the arm matrix (spec section 6 exactness), the
  ``_lab_args`` namespace ``hetero_lab.run_arm`` consumes (every field it
  reads must be present, ``dry_run`` must always be ``True``), and
  ``_scan_env``'s ``os.environ["MINGRU_SCAN"]`` set/unset/restore
  semantics (including restore-on-exception). No torch import needed for
  these.
- One end-to-end subprocess check per the task's stated verification: a
  tiny ``--device cpu`` smoke run emits a valid ``MINGRU_LAB_ROW`` line and
  never touches ``experiments/lab_results.jsonl``, and the default
  (``--device cuda``) invocation fails fast at pre-flight with a clear,
  non-traceback message on this CUDA-less test machine. These import torch
  (via the subprocess) and are slower, so they're kept to one case each.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "gpu_hetero_campaign.py"

_spec = importlib.util.spec_from_file_location("gpu_hetero_campaign", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
gpu_hetero_campaign = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("gpu_hetero_campaign", gpu_hetero_campaign)
_spec.loader.exec_module(gpu_hetero_campaign)

_ARMS = gpu_hetero_campaign._ARMS
_ARM_BY_KEY = gpu_hetero_campaign._ARM_BY_KEY
_lab_args = gpu_hetero_campaign._lab_args
_scan_env = gpu_hetero_campaign._scan_env


# --- arm matrix: spec section 6 exactness ---------------------------------


def test_arm_keys_and_order_match_spec():
    assert [arm.key for arm in _ARMS] == ["sg8", "pd64", "pd1024"]


def test_arm_rounds_and_models_match_spec():
    by_key = {arm.key: (arm.round, arm.model) for arm in _ARMS}
    assert by_key == {
        "sg8": ("hetero-gpu36-sg8", "hetero-pg8"),
        "pd64": ("hetero-gpu36-pd64", "hetero-pd64"),
        "pd1024": ("hetero-gpu36-pd1024", "hetero-pd1024"),
    }


def test_givens_arm_forces_triton_and_does_not_compile():
    arm = _ARM_BY_KEY["sg8"]
    assert arm.scan_mode == "triton"
    assert arm.compile is False


def test_delta_arms_compile_with_scan_unset():
    for key in ("pd64", "pd1024"):
        arm = _ARM_BY_KEY[key]
        assert arm.compile is True
        assert arm.scan_mode is None


def test_delta_compile_probe_kwargs_match_hetero_lab_factories():
    """The pre-flight compile probes must build the exact same DeltaMinGRU
    shapes the pd64/pd1024 arms train on. This introspects the REAL
    ``hetero_lab.LOCAL_FACTORIES`` constructions (``DeltaMinGRU`` stores
    ``n_heads``/``nh``/``d_k``/``d_v`` as plain instance attributes, so no
    CUDA or forward pass is needed) rather than comparing two independently
    hardcoded dict literals -- a pure transcription check would still pass
    if the campaign's probe constants and the factory's actual config drift
    apart from each other (both wrong in the same way is undetectable by
    comparing them to a THIRD hardcoded copy written in the test).
    """
    hetero_lab = gpu_hetero_campaign._import_hetero_lab()
    d_in = d_h = 64

    pd64_layer = hetero_lab.LOCAL_FACTORIES["pdelta64"](d_in, d_h)
    probe_kwargs = gpu_hetero_campaign._PD64_DELTA_KWARGS
    assert pd64_layer.nh == probe_kwargs["nh"]
    assert pd64_layer.n_heads == probe_kwargs["n_heads"]
    assert pd64_layer.d_k == probe_kwargs["d_k"]
    assert pd64_layer.d_v == probe_kwargs["d_v"]

    pd1024_layer = hetero_lab.LOCAL_FACTORIES["pdelta1024"](d_in, d_h)
    probe_kwargs = gpu_hetero_campaign._PD1024_DELTA_KWARGS
    assert pd1024_layer.nh == probe_kwargs["nh"]
    assert pd1024_layer.n_heads == probe_kwargs["n_heads"]
    assert pd1024_layer.d_k == probe_kwargs["d_k"]
    assert pd1024_layer.d_v == probe_kwargs["d_v"]


def test_givens_engagement_probe_config_matches_hetero_lab_factory():
    """The pre-flight givens-triton engagement probe must build the exact
    same ``GivensMinGRU`` shape the sg8 arm (``hetero-pg8``) trains on.
    Introspects the REAL ``hetero_lab.LOCAL_FACTORIES["pgivens8"]``
    construction -- the packaged ``GivensMinGRU`` stores the block size as
    ``.k`` and the round count as ``.rounds`` (plain instance attributes,
    see ``src/mingru/min_gru.py``'s ``GivensMinGRU.__init__``) -- the same
    genuine-introspection standard as
    ``test_delta_compile_probe_kwargs_match_hetero_lab_factories``, guarding
    against the identical asymmetry: a second hardcoded dict transcription
    in the test would stay green even if the probe's config and the
    factory's actual config drifted apart from each other.
    """
    hetero_lab = gpu_hetero_campaign._import_hetero_lab()
    d_in = d_h = 64

    pgivens8_layer = hetero_lab.LOCAL_FACTORIES["pgivens8"](d_in, d_h)
    probe_kwargs = gpu_hetero_campaign._PGIVENS8_KWARGS
    assert pgivens8_layer.k == probe_kwargs["block_size"]
    assert pgivens8_layer.rounds == probe_kwargs["rounds"]


def test_env_block_backend_fields_derived_from_full_arm_matrix():
    # scan_mode_givens/compile_delta must be derived from the fixed _ARMS
    # matrix (design-review fix: these used to be independent hardcoded
    # literals inside _env_block), and the derivation must land on the same
    # values the MINGRU_LAB_ENV contract has always emitted.
    assert gpu_hetero_campaign._GIVENS_SCAN_MODE == _ARM_BY_KEY["sg8"].scan_mode == "triton"
    assert gpu_hetero_campaign._ALL_DELTA_ARMS_COMPILE is True
    assert gpu_hetero_campaign._ALL_DELTA_ARMS_COMPILE == all(
        _ARM_BY_KEY[k].compile for k in ("pd64", "pd1024")
    )


# --- _lab_args: the argparse.Namespace hetero_lab.run_arm consumes --------


def test_lab_args_carries_arm_seed_steps_device():
    arm = _ARM_BY_KEY["pd1024"]
    ns = _lab_args(arm, seed=7, steps=1600, device="cuda", ckpt_t=128)
    assert ns.round == "hetero-gpu36-pd1024"
    assert ns.model == "hetero-pd1024"
    assert ns.task == "S3-hier"
    assert ns.seed == 7
    assert ns.steps == 1600
    assert ns.device == "cuda"
    assert ns.compile is True
    assert ns.ckpt_t == 128


def test_lab_args_always_dry_run_no_ledger_write_path():
    for arm in _ARMS:
        ns = _lab_args(arm, seed=0, steps=4, device="cpu", ckpt_t=128)
        assert ns.dry_run is True


def test_lab_args_every_intervention_flag_run_arm_reads_is_present():
    # hetero_lab.run_arm dereferences these attributes unconditionally
    # (soft_warmup, soft_blend, commit_lambda, commit_ramp, grad_noise,
    # identity_warmup, curriculum, ckpt_t, require_torch) before ever
    # touching the campaign-specific ones -- a missing attribute would be
    # an AttributeError deep inside run_arm, not a clear campaign-side
    # error, so this test pins the full namespace contract.
    ns = _lab_args(_ARM_BY_KEY["sg8"], seed=0, steps=4, device="cpu", ckpt_t=128)
    for attr in (
        "round",
        "task",
        "model",
        "seed",
        "steps",
        "soft_warmup",
        "soft_blend",
        "commit_lambda",
        "commit_ramp",
        "grad_noise",
        "identity_warmup",
        "curriculum",
        "ckpt_t",
        "dry_run",
        "require_torch",
        "device",
        "compile",
    ):
        assert hasattr(ns, attr), f"missing {attr!r}"


def test_lab_args_off_flags_match_run_arm_defaults_no_intervention():
    # Every intervention knob off (soft_warmup=0, commit_lambda=0.0, etc.)
    # -- the campaign runs the plain hetero stacks, matching hetero_lab
    # main()'s own documented defaults, not some campaign-local variant.
    ns = _lab_args(_ARM_BY_KEY["pd64"], seed=0, steps=4, device="cpu", ckpt_t=128)
    assert ns.soft_warmup == 0
    assert ns.soft_blend == 0
    assert ns.commit_lambda == 0.0
    assert ns.commit_ramp == 800
    assert ns.grad_noise == 0.0
    assert ns.identity_warmup == 0
    assert ns.curriculum == ""
    assert ns.require_torch is None


# --- _scan_env: MINGRU_SCAN set/unset/restore ------------------------------


def test_scan_env_sets_triton_and_restores_prior_unset(monkeypatch):
    monkeypatch.delenv("MINGRU_SCAN", raising=False)
    with _scan_env("triton"):
        assert os.environ["MINGRU_SCAN"] == "triton"
    assert "MINGRU_SCAN" not in os.environ


def test_scan_env_none_unsets_and_restores_prior_value():
    os.environ["MINGRU_SCAN"] = "eager"
    try:
        with _scan_env(None):
            assert "MINGRU_SCAN" not in os.environ
        assert os.environ["MINGRU_SCAN"] == "eager"
    finally:
        os.environ.pop("MINGRU_SCAN", None)


def test_scan_env_restores_even_on_exception(monkeypatch):
    monkeypatch.delenv("MINGRU_SCAN", raising=False)
    with pytest.raises(RuntimeError):
        with _scan_env("triton"):
            assert os.environ["MINGRU_SCAN"] == "triton"
            raise RuntimeError("seed failure mid-arm")
    assert "MINGRU_SCAN" not in os.environ


def test_scan_env_nested_arm_sequence_never_leaks(monkeypatch):
    # Mirrors the campaign loop: sg8 (triton) then a delta arm (unset) --
    # the delta arm's block must see MINGRU_SCAN absent, not "triton"
    # leftover from the previous arm.
    monkeypatch.delenv("MINGRU_SCAN", raising=False)
    with _scan_env("triton"):
        assert os.environ["MINGRU_SCAN"] == "triton"
    with _scan_env(None):
        assert "MINGRU_SCAN" not in os.environ
    assert "MINGRU_SCAN" not in os.environ


# --- end-to-end: CPU smoke + CUDA-less fail-fast ---------------------------


def test_cpu_smoke_emits_valid_row_and_touches_no_ledger():
    ledger = _REPO_ROOT / "experiments" / "lab_results.jsonl"
    before = ledger.stat().st_mtime_ns if ledger.exists() else None

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--arms",
            "pd64",
            "--seeds",
            "0",
            "--steps",
            "4",
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
        if line.startswith(gpu_hetero_campaign._ROW_PREFIX)
    ]
    assert len(row_lines) == 1, result.stdout
    row = json.loads(row_lines[0][len(gpu_hetero_campaign._ROW_PREFIX) :])
    assert row["round"] == "hetero-gpu36-pd64"
    assert row["variant"] == "hetero-pd64"
    assert row["seed"] == 0

    # hetero_lab.run_arm's own bare (unmarked) row print must be suppressed
    # (redirect_stdout around the run_arm call) -- the row's distinctive
    # "round" value should appear exactly once in the whole transcript, not
    # twice (once bare, once MINGRU_LAB_ROW-prefixed).
    assert result.stdout.count('"round": "hetero-gpu36-pd64"') == 1, result.stdout

    # no MINGRU_LAB_ENV line on --device cpu (pre-flight is skipped)
    assert not any(
        line.startswith(gpu_hetero_campaign._ENV_PREFIX) for line in result.stdout.splitlines()
    )

    after = ledger.stat().st_mtime_ns if ledger.exists() else None
    assert before == after, "CPU smoke run must never touch the ledger"


def test_default_cuda_invocation_fails_fast_without_traceback():
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--arms",
            "pd64",
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
