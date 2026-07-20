"""Tests for `experiments/benchmark_lab.py`: the task-agnostic single-arm
benchmark lab driver (spec
`.claude/output/specs/2026-07-19-benchmark-round-design.md` §4/§6/§9).

Sections
--------
1. Loss-mode masking -- `masked_query` loss/accuracy touch only query
   positions; `last_step` loss/accuracy ignore every non-final position.
2. Regression eval -- `_forward_regression`'s MSE matches a manually
   computed `F.mse_loss`. Weighted-average (sum/count, not per-batch-
   averaged) accumulation across ragged batch sizes is exercised on
   `_eval_last_step_loader` (the loader-based eval path, where a real
   `Loader`'s trailing batch can be smaller than the rest) --
   `_eval_generator_task`'s batches are always uniform-size by
   construction (every draw uses the same `batch_size`), so it has no
   ragged-batch case to exercise.
3. Row schema -- `run_arm` (dry-run) emits every required top-level key
   and the task's fit-metric key inside `ckpt`, for all four tasks.
4. Ledger dedup -- a duplicate (round, task, variant, seed) is not
   appended twice.
5. Model construction -- `build_model` wires each of the six arms
   through `MinGRUStack` with the registry's mixer/mixer_kwargs, and the
   pendulum task's decay wiring matches `DECAY_CAPABLE_ARMS`.
6. Checkpoint-required guard -- `run_arm` raises rather than assembling a
   row around the sentinel metric when no checkpoint was ever selected
   (step-based: `eval_every` never divides a step in `[1, steps]`;
   epoch-based: `epochs=0`).
7. `rotation-hetero` arm (evidence-phase-gate amendment) -- registered as a
   `["rotation", "signed"]` hetero stack (mirroring probes.py's
   `minGRU-hetero-rs`), this repo's evidenced fix for rotation-stack STE
   compounding (not a snap on/off comparison -- both arms' rotation block
   snaps identically by default). It avoids `MinGRUStack`'s multi-rotation
   `UserWarning` (unlike `rotation`, unchanged, which still emits it),
   excludes pendulum decay wiring (mirrors `delta`'s feature-channel-only
   treatment), and has a param count distinct from `rotation`'s -- see
   `ARM_REGISTRY`'s comment in `benchmark_lab.py` for the full rationale,
   including the same-type reading that was tried and rejected.
8. `signed-givens`/`signed-delta` arms (second, later amendment) -- the
   promoted hetero structures (`probes.py`'s `minGRU-hetero-sg8`, GPU-re-
   evidenced as `hetero_lab.py`'s `hetero-pg8`; and `hetero_lab.py`'s
   `hetero-pd1024` at the delta mechanism's native config), run on ALL
   FOUR tasks like every other arm (no per-task arm subset exists).
   Neither is decay-capable, mirroring `rotation-hetero`'s treatment.
9. `PROBE_ARMS` (third amendment, S5-only follow-up probe) --
   `rotation-hetero-k5` (the `rotation-hetero` stack with the rotation
   block's snap grid widened to `(2, 3, 4, 5, 6)`, K=5 included; `signed`
   block untouched) and `signed-delta-nh3`/`signed-delta-nh4` (the
   `signed-delta` stack with the delta block's `nh` raised from this
   round's matrix value, 2, to 3/4). `MATRIX_ARMS`/`PROBE_ARMS` disjoint,
   `ARM_REGISTRY == MATRIX_ARMS | PROBE_ARMS`; probe arms excluded from
   `DECAY_CAPABLE_ARMS` like their non-decay hetero siblings.
10. `gru` arm (fourth amendment, "standard GRU control arm" entry) -- the
    ninth `MATRIX_ARMS` arm and the classical external control this whole
    round was missing: a depth-matched (2-layer, d_model=64) plain
    `nn.GRU`, registered with a sentinel (never-destructured) tuple value
    since it is not a `MinGRUStack` mixer. Construction per task/loss-mode
    (`GRUTokenSequenceModel`/`GRUFeatureSequenceModel`, mirroring
    `TokenSequenceModel`/`FeatureSequenceModel`'s shape contracts), the
    `run_arm` `"layers"`-field fix (`_model_layers`), pendulum's dt-as-
    feature-only wiring (no mechanical decay path exists to disable),
    decay exclusion, and a forward+backward gradient-flow smoke per loss
    mode.
11. `REF_ARMS` (fifth amendment, "gru-large grounding reference" entry) --
    `gru-large`, a literature-scale (hidden-256, 2-layer) plain `nn.GRU`,
    registered like `gru` with a sentinel tuple value but deliberately
    excluded from `MATRIX_ARMS`/`DECAY_CAPABLE_ARMS`: `ARM_REGISTRY ==
    MATRIX_ARMS | PROBE_ARMS | REF_ARMS`. `build_model` builds it at
    `GRU_LARGE_D_MODEL` (256) instead of `gru`'s `D_MODEL` (64), and
    `run_arm` applies `REF_ARM_BUDGETS[("psmnist", "gru-large")]`'s
    60-epoch override as the resolved default (an explicit `epochs=`
    argument still wins), leaving `gru`'s and every matched arm's
    budget resolution untouched.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from experiments.benchmark_lab import (
    ARM_REGISTRY,
    DECAY_CAPABLE_ARMS,
    GRU_LARGE_D_MODEL,
    MATRIX_ARMS,
    PROBE_ARMS,
    REF_ARM_BUDGETS,
    REF_ARMS,
    FeatureSequenceModel,
    GRUFeatureSequenceModel,
    GRUTokenSequenceModel,
    TokenSequenceModel,
    _eval_last_step_loader,
    _forward_dense,
    _forward_last_step,
    _forward_masked_query,
    _forward_regression,
    _model_layers,
    _row_exists,
    _SyntheticPsMNISTLoader,
    _tiny_task_overrides,
    build_model,
    run_arm,
)
from experiments.benchmark_tasks import TASKS, Budget, EvalConfig, TaskSpec, make_mqar

SEED = 0


# ---------------------------------------------------------- 1. loss masking
def test_masked_query_loss_ignores_non_query_positions():
    """Corrupting `y` at non-masked positions must not change the loss or
    accuracy count -- `_forward_masked_query` never reads `y` there."""
    torch.manual_seed(SEED)
    model = build_model(TASKS["mqar"], "log")
    model.eval()
    batch, T = 4, 32
    gen = torch.Generator().manual_seed(1)
    x, y, mask = make_mqar(batch, T, gen, num_pairs=4)
    assert mask.any() and not mask.all(), "fixture must have both masked and unmasked positions"

    with torch.no_grad():
        loss_a, correct_a, total_a = _forward_masked_query(model, x, y, mask)

        y_corrupted = y.clone()
        y_corrupted[~mask] = (y_corrupted[~mask] + 1) % 64  # any different, still in-range value
        loss_b, correct_b, total_b = _forward_masked_query(model, x, y_corrupted, mask)

    assert torch.equal(loss_a, loss_b), "loss changed after corrupting only non-masked positions"
    assert correct_a == correct_b
    assert total_a == total_b == int(mask.sum().item())


def test_last_step_loss_ignores_non_final_logits():
    """Corrupting the model's logits at every position except the last
    must not change `_forward_last_step`'s loss/accuracy -- it slices
    `[:, -1, :]` before computing anything."""
    torch.manual_seed(SEED)
    B, T, n_cls = 4, 10, 10
    x = torch.randn(B, T, n_cls)
    y = torch.randint(0, n_cls, (B,))

    class _Echo(torch.nn.Module):
        """Returns `x` reshaped as if it were per-position logits, so the
        test can freely mutate non-final positions and check invariance
        without needing a real trained model."""

        def forward(self, inp):
            return inp

    model = _Echo()
    loss_a, correct_a, total_a = _forward_last_step(model, x, y)

    x_corrupted = x.clone()
    x_corrupted[:, :-1, :] = torch.randn_like(x_corrupted[:, :-1, :]) * 100.0
    loss_b, correct_b, total_b = _forward_last_step(model, x_corrupted, y)

    assert torch.equal(loss_a, loss_b), "loss changed after corrupting only non-final positions"
    assert correct_a == correct_b
    assert total_a == total_b == B


# --------------------------------------------------------- 2. regression eval
def test_regression_eval_matches_manual_mse():
    """`_forward_regression`'s loss must equal `F.mse_loss` computed
    manually on the same model call (arm without decay, so the model_in
    concat is the only thing to get right)."""
    torch.manual_seed(SEED)
    B, T = 3, 16
    model = build_model(replace(TASKS["pendulum"]), "log")
    model.eval()
    x = torch.randn(B, T, 2)
    dt = torch.rand(B, T).clamp(min=0.01)
    dt[:, 0] = 0.0
    y = torch.randn(B, T, 2)

    with torch.no_grad():
        loss, sq_err_sum, count = _forward_regression(model, x, dt, y, "log")

        feat = torch.log1p(dt).unsqueeze(-1)
        model_in = torch.cat([x, feat], dim=-1)
        pred = model(model_in, delta_t=dt)
        manual_loss = F.mse_loss(pred, y)

    assert torch.allclose(loss, manual_loss, atol=1e-6)
    assert count == y.numel()
    assert sq_err_sum == pytest.approx(manual_loss.item() * count, rel=1e-4)


def test_eval_last_step_loader_weights_by_batch_size_not_per_batch_average():
    """`_eval_last_step_loader` must accumulate correct-count/total-count
    (sum/sum), NOT average each batch's accuracy and then average those
    averages -- the two disagree whenever batches are ragged (unequal
    size), which is exactly what a real `Loader`'s trailing batch can be.

    Batch 1: 5 rows, 4 correct (acc 0.8). Batch 2: 3 rows, 1 correct
    (acc 1/3). Weighted: (4 + 1) / (5 + 3) = 0.625. Naive per-batch
    average: (0.8 + 1/3) / 2 ~= 0.567 -- a different number, so this
    discriminates the two accumulation strategies.
    """

    class _EchoLogits(torch.nn.Module):
        """`_forward_last_step` reads `model(x)[:, -1, :]` as logits, so
        feeding pre-built one-hot-ish logits directly as `x` lets the test
        control correctness counts exactly without a trained model."""

        def forward(self, inp):
            return inp

    def _make_batch(n_correct: int, n_wrong: int):
        y_correct = torch.zeros(n_correct, dtype=torch.long)
        y_wrong = torch.ones(n_wrong, dtype=torch.long)
        y = torch.cat([y_correct, y_wrong])
        x = torch.zeros(n_correct + n_wrong, 1, 2)
        x[:n_correct, 0, 0] = 10.0  # correct rows: argmax picks class 0, matching y
        x[n_correct:, 0, 0] = 10.0  # wrong rows: argmax still picks class 0, but y is 1
        return x, y

    batch1 = _make_batch(n_correct=4, n_wrong=1)  # 5 rows, acc 0.8
    batch2 = _make_batch(n_correct=1, n_wrong=2)  # 3 rows, acc 1/3

    model = _EchoLogits()
    result = _eval_last_step_loader(model, [batch1, batch2])
    assert result == pytest.approx(5 / 8)
    assert result != pytest.approx((0.8 + 1 / 3) / 2)


def test_regression_delta_arm_gets_no_mechanical_decay():
    """The delta arm's `_forward_regression` call must pass `delta_t=None`
    to the model (DeltaMinGRU rejects `delta_t` unconditionally) -- so a
    delta-arm pendulum model must NOT be built with decay enabled."""
    model = build_model(TASKS["pendulum"], "delta")
    assert model.stack.blocks[0].mingru.decay is None
    x = torch.randn(2, 8, 2)
    dt = torch.rand(2, 8).clamp(min=0.01)
    dt[:, 0] = 0.0
    y = torch.randn(2, 8, 2)
    loss, _, _ = _forward_regression(model, x, dt, y, "delta")
    assert torch.isfinite(loss)


# ------------------------------------------------------------ 3. row schema
REQUIRED_ROW_KEYS = (
    "round",
    "task",
    "variant",
    "layers",
    "seed",
    "steps",
    "acc",
    "secs",
    "ckpt",
    "config",
)


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_row_schema_has_required_keys_and_fit_metric(task_name):
    tiny_tasks = _tiny_task_overrides()
    task = tiny_tasks[task_name]
    row = run_arm(round_tag="test-schema", task=task, arm="log", seed=0, device="cpu", dry_run=True)
    missing = [k for k in REQUIRED_ROW_KEYS if k not in row]
    assert not missing, f"row missing required key(s) {missing}"
    assert task.fit_metric in row["ckpt"], f"ckpt missing fit-metric key {task.fit_metric!r}"
    # Honest fit metric (item 5): the selection-time value is kept
    # alongside the re-evaluated fit-metric value, under a distinct key, for
    # every task/loss-mode -- never silently dropped.
    selection_key = f"selection_{task.fit_metric}"
    assert selection_key in row["ckpt"], f"ckpt missing selection key {selection_key!r}"
    assert row["task"] == task_name
    assert row["variant"] == "log"
    assert row["layers"] == 2


def test_psmnist_fit_metric_equals_selection_value_deterministic_val_split():
    """psMNIST's `val()` iterates in a fixed, non-shuffled order (no seed to
    vary) -- re-evaluating the selected checkpoint on it must reproduce the
    selection-time value exactly, not merely approximately."""
    tiny_tasks = _tiny_task_overrides()
    row = run_arm(
        round_tag="test-psmnist-fit-metric",
        task=tiny_tasks["psmnist"],
        arm="log",
        seed=0,
        device="cpu",
        dry_run=True,
    )
    assert row["ckpt"]["val_acc"] == row["ckpt"]["selection_val_acc"]


def test_step_based_fit_metric_reeval_uses_fit_eval_seed(monkeypatch):
    """The step-based honest-fit-metric re-eval (item 5, "kill the
    winner's curse") must call `_eval_generator_task` with
    `seed=FIT_EVAL_SEED` -- distinct from the checkpoint-selection loop's
    `CKPT_EVAL_SEED` and the generalization sweep's `FINAL_EVAL_SEED` --
    verified by spying on every call the whole `run_arm` invocation makes,
    not just asserting the resulting row shape."""
    import experiments.benchmark_lab as lab

    seen_seeds = []
    real_eval = lab._eval_generator_task

    def _spy(*args, **kwargs):
        seen_seeds.append(kwargs.get("seed"))
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(lab, "_eval_generator_task", _spy)
    tiny_tasks = _tiny_task_overrides()
    lab.run_arm(
        round_tag="test-fit-eval-seed",
        task=tiny_tasks["s5"],
        arm="log",
        seed=0,
        device="cpu",
        dry_run=True,
    )

    assert lab.FIT_EVAL_SEED in seen_seeds
    assert lab.CKPT_EVAL_SEED in seen_seeds  # the checkpoint-selection loop still runs
    assert lab.FINAL_EVAL_SEED in seen_seeds  # the generalization sweep still runs


def test_eval_seeds_are_mutually_distinct_and_outside_the_run_seed_range():
    """CKPT_EVAL_SEED/FINAL_EVAL_SEED/FIT_EVAL_SEED must never collide with
    each other or with any run seed in the largest seed matrix (0..35) --
    the pre-matrix technical review's item 6 (the previous 4/5 pair
    coincided with run seeds 4/5's `torch.manual_seed` init stream)."""
    import experiments.benchmark_lab as lab

    seeds = {lab.CKPT_EVAL_SEED, lab.FINAL_EVAL_SEED, lab.FIT_EVAL_SEED}
    assert len(seeds) == 3, "the three eval seeds must be pairwise distinct"
    max_run_seed = max(task.seeds for task in TASKS.values()) - 1
    for s in seeds:
        assert s > max_run_seed, f"eval seed {s} falls inside the run-seed range"


def test_pendulum_row_config_carries_tau():
    """Pendulum rows' `config` must carry the TaskSpec's own `fit_threshold`
    (tau) directly (item 3) -- a later edit to `PENDULUM_TAU` must never be
    able to silently re-judge an already-landed row against a different
    threshold than the one it actually ran under."""
    tiny_tasks = _tiny_task_overrides()
    task = tiny_tasks["pendulum"]
    row = run_arm(
        round_tag="test-pendulum-tau", task=task, arm="log", seed=0, device="cpu", dry_run=True
    )
    assert row["config"]["tau"] == task.fit_threshold


def test_non_pendulum_rows_have_no_tau_in_config():
    tiny_tasks = _tiny_task_overrides()
    for name in ("s5", "mqar", "psmnist"):
        row = run_arm(
            round_tag="test-no-tau",
            task=tiny_tasks[name],
            arm="log",
            seed=0,
            device="cpu",
            dry_run=True,
        )
        assert "tau" not in row["config"], f"{name}: config must not carry a tau field"


def test_row_schema_dry_run_does_not_append(tmp_path, monkeypatch):
    import experiments.benchmark_lab as lab

    results = tmp_path / "lab_results.jsonl"
    monkeypatch.setattr(lab, "RESULTS", str(results))
    tiny_tasks = _tiny_task_overrides()
    lab.run_arm(round_tag="test-dry-run", task=tiny_tasks["s5"], arm="log", seed=0, dry_run=True)
    assert not results.exists()


# ------------------------------------------------------------- 4. ledger dedup
def test_ledger_append_dedups_on_round_task_variant_seed(tmp_path, monkeypatch):
    import experiments.benchmark_lab as lab

    results = tmp_path / "lab_results.jsonl"
    monkeypatch.setattr(lab, "RESULTS", str(results))
    tiny_tasks = _tiny_task_overrides()
    task = tiny_tasks["s5"]

    lab.run_arm(round_tag="test-dedup", task=task, arm="log", seed=0, dry_run=False)
    assert results.exists()
    with open(results) as f:
        lines_after_first = f.readlines()
    assert len(lines_after_first) == 1

    lab.run_arm(round_tag="test-dedup", task=task, arm="log", seed=0, dry_run=False)
    with open(results) as f:
        lines_after_second = f.readlines()
    assert len(lines_after_second) == 1, (
        "re-running the same (round, task, variant, seed) must not duplicate"
    )


def test_row_exists_detects_matching_key(tmp_path, monkeypatch):
    import experiments.benchmark_lab as lab

    results = tmp_path / "lab_results.jsonl"
    monkeypatch.setattr(lab, "RESULTS", str(results))
    row = {"round": "r", "task": "s5", "variant": "log", "seed": 0, "other": 1}
    assert not _row_exists(row)
    with open(results, "w") as f:
        f.write(json.dumps(row) + "\n")
    assert _row_exists(row)
    assert not _row_exists({**row, "seed": 1})


# -------------------------------------------------------- 5. model construction
# `gru` (Amendments, 2026-07-20 "standard GRU control arm" entry) is not a
# `MinGRUStack` mixer, so it builds a distinct model class
# (`GRUTokenSequenceModel`/`GRUFeatureSequenceModel`, see section 10 below)
# -- the three generic per-loss-mode shape tests below therefore accept
# EITHER model family's class for the isinstance check (the output-shape
# assertion, the real invariant these tests exist to pin, is unaffected by
# which family built the model). The pendulum decay-wiring test excludes
# `gru` from its `ARM_REGISTRY` parametrization instead: a `gru` model has
# no `.stack` attribute at all to inspect (see section 10's own dedicated
# decay-exclusion test).
@pytest.mark.parametrize("arm", sorted(ARM_REGISTRY))
def test_build_model_token_sequence_shapes(arm):
    task = TASKS["s5"]
    model = build_model(task, arm)
    assert isinstance(model, (TokenSequenceModel, GRUTokenSequenceModel))
    x = torch.randint(0, 120, (2, 9))
    out = model(x)
    assert out.shape == (2, 9, 120)


@pytest.mark.parametrize("arm", sorted(ARM_REGISTRY))
def test_build_model_feature_sequence_last_step_shapes(arm):
    task = TASKS["psmnist"]
    model = build_model(task, arm)
    assert isinstance(model, (FeatureSequenceModel, GRUFeatureSequenceModel))
    x = torch.rand(2, 12, 1)
    out = model(x)
    assert out.shape == (2, 12, 10)


@pytest.mark.parametrize("arm", sorted(set(ARM_REGISTRY) - {"gru", "gru-large"}))
def test_build_model_pendulum_decay_wiring_matches_decay_capable_arms(arm):
    task = TASKS["pendulum"]
    model = build_model(task, arm)
    mixer = model.stack.blocks[0].mingru
    if arm in DECAY_CAPABLE_ARMS:
        assert mixer.decay == "learnable"
    else:
        assert mixer.decay is None


def test_build_model_unknown_arm_raises():
    with pytest.raises(ValueError):
        build_model(TASKS["s5"], "bogus-arm")


def test_build_model_unknown_loss_mode_raises():
    bogus = replace(TASKS["s5"], loss_mode="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_model(bogus, "log")


# ---------------------------------------------------- 6. checkpoint-required guard
def test_run_arm_raises_when_eval_every_never_divides_a_step():
    """Reproduces the reported bug directly: `eval_every > steps` means
    the training loop's `step % eval_every == 0` check never fires, so no
    checkpoint is ever taken. `run_arm` must raise rather than silently
    assemble a row around the sentinel metric (previously: `{"ckpt":
    {"step": 0, "val128": -1.0}}`)."""
    tiny_tasks = _tiny_task_overrides()
    with pytest.raises(RuntimeError, match="no checkpoint was ever selected"):
        run_arm(
            round_tag="test-sentinel-guard",
            task=tiny_tasks["s5"],
            arm="log",
            seed=0,
            steps=50,
            eval_every=100,
            device="cpu",
            dry_run=True,
        )


def test_run_arm_raises_when_eval_every_is_zero():
    """`eval_every=0` is falsy, so the step-based loop's `if eval_every
    and step % eval_every == 0` guard never fires either -- same sentinel
    hazard, different trigger."""
    tiny_tasks = _tiny_task_overrides()
    with pytest.raises(RuntimeError, match="no checkpoint was ever selected"):
        run_arm(
            round_tag="test-sentinel-guard-zero",
            task=tiny_tasks["mqar"],
            arm="log",
            seed=0,
            steps=10,
            eval_every=0,
            device="cpu",
            dry_run=True,
        )


def test_run_arm_raises_on_epoch_based_task_with_zero_epochs():
    """Same hazard on the epoch-based (psMNIST) path: `epochs=0` means
    `_train_epoch_based`'s loop body never runs, so `best_state` stays
    `None` -- must raise, not assemble a row around the sentinel."""
    tiny_tasks = _tiny_task_overrides()
    with pytest.raises(RuntimeError, match="no checkpoint was ever selected"):
        run_arm(
            round_tag="test-sentinel-guard-epochs",
            task=tiny_tasks["psmnist"],
            arm="log",
            seed=0,
            epochs=0,
            device="cpu",
            dry_run=True,
        )


# ----------------------------------------------- 7. rotation-hetero arm (amendment)
def test_rotation_hetero_arm_registered_as_rotation_signed_hetero_stack():
    """`rotation-hetero` (evidence-phase-gate amendment, sixth arm) is a
    single `"rotation"` block (default snap grid) composed with a single
    `"signed"` block, `mixer_kwargs=None` -- byte-identical to probes.py's
    `MIXER_REGISTRY["minGRU-hetero-rs"]`, this repo's evidenced fix for
    rotation-stack STE compounding. A same-type second `"rotation"`
    reading was tried first and rejected: it would be architecturally
    identical to the existing `rotation` arm (same param count, same
    per-seed training trajectory -- RotationMinGRU's `snap` is a
    registered buffer, not a parameter, and `mixer_kwargs={}` already
    resolves to the class's own default snap grid), caught by
    `tests/test_report_benchmarks.py`'s "every arm's param count is
    distinct" invariant. See `ARM_REGISTRY`'s comment in
    `benchmark_lab.py` for the full rationale."""
    assert "rotation-hetero" in ARM_REGISTRY
    mixer, kwargs = ARM_REGISTRY["rotation-hetero"]
    assert mixer == ["rotation", "signed"]
    assert kwargs is None


def test_rotation_hetero_does_not_emit_the_multi_rotation_warning_unlike_rotation():
    """`rotation` (unchanged, run "as is") builds a 2-block, single-mixer-
    type `"rotation"` stack -- `MinGRUStack`'s multi-rotation `UserWarning`
    fires unconditionally on rotation-block COUNT
    (`mixer_list.count("rotation") > 1`), so it warns every construction.
    `rotation-hetero`'s hetero stack has only ONE `"rotation"` entry
    (mixed with `"signed"`), and a single `"rotation"` entry in a mixed
    stack does not warn (see `MinGRUStack`'s docstring) -- this is the
    concrete, checkable difference between the documented broken baseline
    and this repo's evidenced fix for it."""
    task = TASKS["s5"]
    with pytest.warns(UserWarning, match="more than one 'rotation' block"):
        build_model(task, "rotation")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_model(task, "rotation-hetero")


def test_rotation_hetero_pendulum_wiring_matches_delta_feature_channel_only():
    """`rotation-hetero` is deliberately excluded from `DECAY_CAPABLE_ARMS`
    (no established repo precedent for splitting decay across a hetero
    rotation+signed stack -- see `ARM_REGISTRY`'s comment): on the
    pendulum task it must reach `dt` only via the `log1p(dt)` feature-
    concat channel, the same treatment `delta` gets, never mechanically
    through the stack's decay path."""
    assert "rotation-hetero" not in DECAY_CAPABLE_ARMS
    task = TASKS["pendulum"]
    model = build_model(task, "rotation-hetero")
    for block in model.stack.blocks:
        assert block.mingru.decay is None


def test_rotation_and_rotation_hetero_have_distinct_param_counts():
    """Direct local pin of the same invariant
    `tests/test_report_benchmarks.py::test_param_counts_are_positive_and_vary_by_task_and_arm`
    checks at the report layer: the two rotation-family arms must build
    genuinely different models, not the same architecture under two
    names."""
    task = TASKS["s5"]
    rotation_params = sum(p.numel() for p in build_model(task, "rotation").parameters())
    rotation_hetero_params = sum(
        p.numel() for p in build_model(task, "rotation-hetero").parameters()
    )
    assert rotation_params != rotation_hetero_params


# ------------------------- 8. signed-givens / signed-delta arms (2nd amendment)
def test_signed_givens_arm_registered_as_signed_givens_hetero_stack():
    """`signed-givens` (mid-matrix amendment, run on all four tasks like
    every other arm) is a single `"signed"` block composed with a single
    `"givens"` block, `mixer_kwargs=None` -- byte-identical to probes.py's
    `MIXER_REGISTRY["minGRU-hetero-sg8"]` (`(["signed", "givens"], None)`),
    the S3-hier promoted fit-rate winner, GPU-re-evidenced as
    `hetero_lab.py`'s `"hetero-pg8"` arm."""
    assert "signed-givens" in ARM_REGISTRY
    mixer, kwargs = ARM_REGISTRY["signed-givens"]
    assert mixer == ["signed", "givens"]
    assert kwargs is None


def test_signed_givens_block_composition_uses_promoted_givens_defaults():
    """`mixer_kwargs=None` must let `MinGRUStack` apply `GivensMinGRU`'s
    own class defaults (block_size=8, rounds=3) -- the same config this
    round's own `givens` arm passes explicitly -- rather than the givens
    block silently landing on some other configuration."""
    task = TASKS["s5"]
    model = build_model(task, "signed-givens")
    signed_block, givens_block = model.stack.blocks
    assert type(signed_block.mingru).__name__ == "SignedMinGRU"
    assert type(givens_block.mingru).__name__ == "GivensMinGRU"
    assert givens_block.mingru.k == 8  # block_size
    assert givens_block.mingru.rounds == 3


def test_signed_delta_arm_registered_with_native_delta_config():
    """`signed-delta` (mid-matrix amendment, run on all four tasks like
    every other arm) is a single `"signed"` block composed with a single
    `"delta"` block at its native promoted config (nh=2, n_heads=4,
    d_k=16, d_v=16 -- identical kwargs to this round's own `delta` arm),
    mirroring `hetero_lab.py`'s `"hetero-pd1024"` row (matched-state/
    GPU-36 lineage; see `ARM_REGISTRY`'s own comment for the signed-tanh
    vs packaged-`signed` provenance caveat this row does not hide).
    Unlike `signed-givens`, `mixer_kwargs` here is NOT `None`: it is a
    type-keyed dict naming only `"delta"`, so `"signed"` resolves to its
    own class defaults via the omitted-key-defaults-to-`None` rule."""
    assert "signed-delta" in ARM_REGISTRY
    mixer, kwargs = ARM_REGISTRY["signed-delta"]
    assert mixer == ["signed", "delta"]
    assert kwargs == {"delta": {"nh": 2, "n_heads": 4, "d_k": 16, "d_v": 16}}


def test_signed_delta_block_composition_matches_native_delta_config():
    task = TASKS["s5"]
    model = build_model(task, "signed-delta")
    signed_block, delta_block = model.stack.blocks
    assert type(signed_block.mingru).__name__ == "SignedMinGRU"
    assert type(delta_block.mingru).__name__ == "DeltaMinGRU"
    assert delta_block.mingru.nh == 2
    assert delta_block.mingru.n_heads == 4
    assert delta_block.mingru.d_k == 16
    assert delta_block.mingru.d_v == 16


def test_signed_delta_delta_block_kwargs_match_the_plain_delta_arm():
    """`signed-delta`'s delta block must use the EXACT same kwargs as this
    round's own `delta` arm -- both mirror the delta mechanism's promoted
    native-state config, not two independently drifted configs."""
    _, delta_kwargs = ARM_REGISTRY["delta"]
    _, hetero_kwargs = ARM_REGISTRY["signed-delta"]
    assert hetero_kwargs["delta"] == delta_kwargs


def test_signed_givens_and_signed_delta_are_excluded_from_decay_capable_arms():
    assert "signed-givens" not in DECAY_CAPABLE_ARMS
    assert "signed-delta" not in DECAY_CAPABLE_ARMS


@pytest.mark.parametrize("arm", ["signed-givens", "signed-delta"])
def test_new_hetero_arms_pendulum_wiring_matches_delta_feature_channel_only(arm):
    """Neither new arm is decay-capable (same rationale as `rotation-
    hetero`: no established repo precedent for splitting decay wiring
    across a hetero stack's two mixer types) -- on the pendulum task, `dt`
    must reach both new arms only via the `log1p(dt)` feature-concat
    channel, never mechanically through the stack's decay path."""
    task = TASKS["pendulum"]
    model = build_model(task, arm)
    for block in model.stack.blocks:
        assert block.mingru.decay is None


def test_all_matrix_arms_have_distinct_param_counts():
    """Direct local pin of the report-layer invariant
    `tests/test_report_benchmarks.py::test_param_counts_are_positive_and_vary_by_task_and_arm`:
    every arm in `MATRIX_ARMS`, including the two new hetero arms, must
    build a genuinely different model, not a duplicate of an existing one
    under a new name.

    Deliberately scoped to `MATRIX_ARMS`, NOT `ARM_REGISTRY`: `PROBE_ARMS`'s
    `rotation-hetero-k5` is BY CONSTRUCTION param-count-identical to
    `rotation-hetero` (`snap` is a registered buffer, not a parameter --
    see `test_rotation_hetero_k5_and_rotation_hetero_have_equal_param_counts`
    below), so this invariant would falsely fail if it included
    `ARM_REGISTRY`'s three probe arms."""
    task = TASKS["s5"]
    params = {
        arm: sum(p.numel() for p in build_model(task, arm).parameters()) for arm in MATRIX_ARMS
    }
    assert len(set(params.values())) == len(MATRIX_ARMS)


# ------------------------------------------------ 9. PROBE_ARMS (3rd amendment)
def test_matrix_and_probe_arms_are_disjoint_and_union_to_arm_registry():
    assert set(MATRIX_ARMS) & set(PROBE_ARMS) == set()
    assert set(MATRIX_ARMS) & set(REF_ARMS) == set()
    assert set(PROBE_ARMS) & set(REF_ARMS) == set()
    assert set(ARM_REGISTRY) == set(MATRIX_ARMS) | set(PROBE_ARMS) | set(REF_ARMS)
    assert len(MATRIX_ARMS) == 9  # eight MinGRUStack mixer arms + gru
    assert len(PROBE_ARMS) == 3


def test_rotation_hetero_k5_registered_with_widened_snap_grid_signed_untouched():
    """`rotation-hetero-k5` is the same `["rotation", "signed"]` stack as
    `rotation-hetero`, with the rotation block's snap grid widened to
    include K=5 (S5's element orders are {2, 3, 4, 5, 6}; a 5-cycle needs
    an order-5 rotation) -- the `signed` block gets no per-type override
    (omitted key resolves to its own class defaults, per `MinGRUStack`'s
    type-keyed-kwargs convention)."""
    assert "rotation-hetero-k5" in PROBE_ARMS
    mixer, kwargs = PROBE_ARMS["rotation-hetero-k5"]
    assert mixer == ["rotation", "signed"]
    assert kwargs == {"rotation": {"snap": (2, 3, 4, 5, 6)}}
    assert "signed" not in kwargs


def test_rotation_hetero_k5_block_composition_routes_snap_to_rotation_block_only():
    task = TASKS["s5"]
    model = build_model(task, "rotation-hetero-k5")
    rotation_block, signed_block = model.stack.blocks
    assert type(rotation_block.mingru).__name__ == "RotationMinGRU"
    assert type(signed_block.mingru).__name__ == "SignedMinGRU"
    assert rotation_block.mingru.snap == (2, 3, 4, 5, 6)
    # signed_block has no `snap` attribute at all -- confirms the per-type
    # kwargs routing never leaked the rotation-only override onto it.
    assert not hasattr(signed_block.mingru, "snap")


def test_rotation_hetero_k5_and_rotation_hetero_have_equal_param_counts():
    """`snap` is a registered buffer on `RotationMinGRU`, not a
    `nn.Parameter` -- widening the snap grid changes NO tensor shape, so
    `rotation-hetero-k5` and `rotation-hetero` must build models with
    IDENTICAL parameter counts. This is the documented exception to
    `test_all_matrix_arms_have_distinct_param_counts`'s invariant, by
    construction, not a bug -- see that test's own docstring."""
    task = TASKS["s5"]
    rotation_hetero_params = sum(
        p.numel() for p in build_model(task, "rotation-hetero").parameters()
    )
    rotation_hetero_k5_params = sum(
        p.numel() for p in build_model(task, "rotation-hetero-k5").parameters()
    )
    assert rotation_hetero_k5_params == rotation_hetero_params


@pytest.mark.parametrize("arm,expected_nh", [("signed-delta-nh3", 3), ("signed-delta-nh4", 4)])
def test_signed_delta_nh_probe_arms_registered_with_raised_nh(arm, expected_nh):
    """`signed-delta-nh3`/`signed-delta-nh4` are the `signed-delta` stack
    with the delta block's Householder-product count `nh` raised from
    this round's matrix value (2) to 3/4 -- `n_heads`/`d_k`/`d_v` unchanged
    from the matrix `delta`/`signed-delta` arms' own kwargs. `chunk_size=32`
    (vs the default 64) keeps `nh * chunk_size` within the delta Triton
    kernel's `_DELTA_MAX_MICROSTEPS`=128 envelope (nh=3 -> 96, nh=4 -> 128);
    chunk_size is an exact chunked-WY tiling parameter, not an expressivity
    change."""
    assert arm in PROBE_ARMS
    mixer, kwargs = PROBE_ARMS[arm]
    assert mixer == ["signed", "delta"]
    assert kwargs == {
        "delta": {"nh": expected_nh, "n_heads": 4, "d_k": 16, "d_v": 16, "chunk_size": 32}
    }
    assert expected_nh * kwargs["delta"]["chunk_size"] <= 128


@pytest.mark.parametrize("arm,expected_nh", [("signed-delta-nh3", 3), ("signed-delta-nh4", 4)])
def test_signed_delta_nh_probe_arms_block_composition_matches_registered_nh(arm, expected_nh):
    task = TASKS["s5"]
    model = build_model(task, arm)
    signed_block, delta_block = model.stack.blocks
    assert type(signed_block.mingru).__name__ == "SignedMinGRU"
    assert type(delta_block.mingru).__name__ == "DeltaMinGRU"
    assert delta_block.mingru.nh == expected_nh
    assert delta_block.mingru.n_heads == 4
    assert delta_block.mingru.d_k == 16
    assert delta_block.mingru.d_v == 16
    # chunk_size=32 must reach the built DeltaMinGRU (the attribute this
    # probe's kernel-envelope fix changed) and keep nh*chunk_size <= 128.
    assert delta_block.mingru.chunk_size == 32
    assert expected_nh * delta_block.mingru.chunk_size <= 128


def test_signed_delta_nh_probe_arms_have_distinct_param_counts():
    """Unlike `rotation-hetero-k5`'s snap widening, `nh` IS a real
    constructor-time shape parameter (more Householder reflections per
    delta step) -- `signed-delta`, `signed-delta-nh3`, and
    `signed-delta-nh4` must build three genuinely different models."""
    task = TASKS["s5"]
    params = {
        arm: sum(p.numel() for p in build_model(task, arm).parameters())
        for arm in ("signed-delta", "signed-delta-nh3", "signed-delta-nh4")
    }
    assert len(set(params.values())) == 3


@pytest.mark.parametrize("arm", ["rotation-hetero-k5", "signed-delta-nh3", "signed-delta-nh4"])
def test_probe_arms_excluded_from_decay_capable_arms(arm):
    assert arm not in DECAY_CAPABLE_ARMS
    task = TASKS["pendulum"]
    model = build_model(task, arm)
    for block in model.stack.blocks:
        assert block.mingru.decay is None


# ------------------------------------------------- 10. gru arm (4th amendment)
def test_gru_arm_registered_in_matrix_arms_as_sentinel_tuple():
    """`gru` (Amendments, 2026-07-20 "standard GRU control arm" entry) is
    registered in `MATRIX_ARMS` with a SENTINEL tuple value -- unlike
    every other arm, this value is never destructured into `(mixer,
    mixer_kwargs)`: `build_model` special-cases `arm == "gru"` before it
    ever consults `ARM_REGISTRY[arm]`'s value (see `MATRIX_ARMS`'s own
    comment in `benchmark_lab.py`). The tuple exists only so `"gru"`
    satisfies `MATRIX_ARMS`/`ARM_REGISTRY` dict membership."""
    assert "gru" in MATRIX_ARMS
    assert "gru" in ARM_REGISTRY
    mixer, kwargs = MATRIX_ARMS["gru"]
    assert mixer == "gru"
    assert kwargs is None


def test_gru_excluded_from_decay_capable_arms():
    """Plain `nn.GRU` has no decay/`delta_t` mechanism at all -- unlike the
    hetero arms (excluded for lack of an established decay-split
    convention), `gru` is excluded because there is no mechanism to wire
    up in the first place."""
    assert "gru" not in DECAY_CAPABLE_ARMS


@pytest.mark.parametrize("task_name", ["s5", "mqar"])
def test_build_model_gru_token_sequence_two_layers_and_shapes(task_name):
    """`gru` on the two token-input tasks (dense/masked_query): a 2-layer
    `nn.GRU` (depth-matched to the other eight arms' 2-block MinGRUStack),
    embedding input, per-position logits over the task's own vocab/n_cls
    shape -- `TokenSequenceModel`'s contract, built by a distinct class."""
    task = TASKS[task_name]
    model = build_model(task, "gru")
    assert isinstance(model, GRUTokenSequenceModel)
    assert model.rnn.num_layers == 2
    vocab, n_cls = model.emb.num_embeddings, model.head.out_features
    x = torch.randint(0, vocab, (2, 9))
    out = model(x)
    assert out.shape == (2, 9, n_cls)


def test_build_model_gru_feature_sequence_last_step_shapes():
    """`gru` on psMNIST (last_step): a 2-layer `nn.GRU` over the scalar
    pixel-stream input (no `log1p(dt)` feature -- psMNIST has no `dt`),
    `_forward_last_step` reads only the final position's logits --
    `FeatureSequenceModel`'s contract, built by a distinct class."""
    task = TASKS["psmnist"]
    model = build_model(task, "gru")
    assert isinstance(model, GRUFeatureSequenceModel)
    assert model.rnn.num_layers == 2
    assert model.rnn.input_size == 1  # scalar pixel input, no dt feature on psMNIST
    x = torch.rand(2, 12, 1)
    out = model(x)
    assert out.shape == (2, 12, 10)
    loss, correct, total = _forward_last_step(model, x, torch.zeros(2, dtype=torch.long))
    assert torch.isfinite(loss)
    assert total == 2


def test_build_model_gru_regression_gets_dt_as_feature_not_mechanical_decay():
    """`gru` on pendulum (regression): input_size is `in_dim + 1` (the
    `log1p(dt)` fairness-rule feature `build_model` concatenates for every
    arm) -- `gru` has no mechanical decay path, so this feature channel is
    the ONLY way `dt` reaches it, exactly like `delta`/the hetero arms
    (`gru` is excluded from `DECAY_CAPABLE_ARMS`)."""
    task = TASKS["pendulum"]
    model = build_model(task, "gru")
    assert isinstance(model, GRUFeatureSequenceModel)
    assert model.rnn.input_size == 3  # 2 state dims + 1 log1p(dt) feature
    x = torch.randn(2, 8, 2)
    dt = torch.rand(2, 8).clamp(min=0.01)
    dt[:, 0] = 0.0
    y = torch.randn(2, 8, 2)
    loss, sq_err_sum, count = _forward_regression(model, x, dt, y, "gru")
    assert torch.isfinite(loss)
    assert count == y.numel()


def test_gru_feature_sequence_model_rejects_a_non_none_delta_t():
    """Safety-net assertion pin: `GRUFeatureSequenceModel.forward` must
    reject a non-`None` `delta_t` -- plain `nn.GRU` has no decay path to
    route it through. `gru`'s exclusion from `DECAY_CAPABLE_ARMS` means
    `_forward_regression` must never actually pass one, but the model
    itself should not silently ignore a caller that does."""
    model = GRUFeatureSequenceModel(input_size=3, n_out=2)
    x = torch.randn(2, 5, 3)
    with pytest.raises(AssertionError):
        model(x, delta_t=torch.rand(2, 5))


def test_gru_row_layers_field_reports_two():
    """`run_arm`'s `"layers"` field must read `model.rnn.num_layers` for
    the `gru` arm (it has no `.stack` attribute, unlike the eight
    MinGRUStack arms) -- `_model_layers`'s branch, mirroring `probes.py`'s
    `run_one` GRU-vs-MinGRUStack split."""
    tiny_tasks = _tiny_task_overrides()
    row = run_arm(
        round_tag="test-gru-layers",
        task=tiny_tasks["s5"],
        arm="gru",
        seed=0,
        device="cpu",
        dry_run=True,
    )
    assert row["layers"] == 2
    assert _model_layers(build_model(TASKS["s5"], "gru")) == 2


def test_gru_pendulum_row_still_carries_tau():
    """Sanity: the `gru` arm doesn't disturb the pendulum row's existing
    `config["tau"]` wiring (item 3) -- that field is set from
    `task.fit_threshold` unconditionally on the pendulum task, regardless
    of which arm ran."""
    tiny_tasks = _tiny_task_overrides()
    row = run_arm(
        round_tag="test-gru-tau",
        task=tiny_tasks["pendulum"],
        arm="gru",
        seed=0,
        device="cpu",
        dry_run=True,
    )
    assert row["config"]["tau"] == tiny_tasks["pendulum"].fit_threshold


def _assert_gradients_flow(model: torch.nn.Module) -> None:
    """Shared post-backward assertion: every parameter that requires grad
    must have a gradient, and at least one gradient must be non-zero --
    catches a silently detached path without requiring an exact-value
    check per parameter."""
    any_nonzero = False
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"{name}: no gradient (path silently detached?)"
        if torch.any(p.grad != 0):
            any_nonzero = True
    assert any_nonzero, "every parameter's gradient was exactly zero"


def test_gru_forward_backward_smoke_dense_s5():
    torch.manual_seed(0)
    model = build_model(TASKS["s5"], "gru")
    x = torch.randint(0, 120, (3, 10))
    y = torch.randint(0, 120, (3, 10))
    loss, _, _ = _forward_dense(model, x, y)
    loss.backward()
    _assert_gradients_flow(model)


def test_gru_forward_backward_smoke_masked_query_mqar():
    torch.manual_seed(0)
    model = build_model(TASKS["mqar"], "gru")
    gen = torch.Generator().manual_seed(1)
    x, y, mask = make_mqar(3, 32, gen, num_pairs=4)
    loss, _, _ = _forward_masked_query(model, x, y, mask)
    loss.backward()
    _assert_gradients_flow(model)


def test_gru_forward_backward_smoke_last_step_psmnist():
    torch.manual_seed(0)
    model = build_model(TASKS["psmnist"], "gru")
    x = torch.rand(3, 10, 1)
    y = torch.randint(0, 10, (3,))
    loss, _, _ = _forward_last_step(model, x, y)
    loss.backward()
    _assert_gradients_flow(model)


def test_gru_forward_backward_smoke_regression_pendulum():
    torch.manual_seed(0)
    model = build_model(TASKS["pendulum"], "gru")
    x = torch.randn(3, 8, 2)
    dt = torch.rand(3, 8).clamp(min=0.01)
    dt[:, 0] = 0.0
    y = torch.randn(3, 8, 2)
    loss, _, _ = _forward_regression(model, x, dt, y, "gru")
    loss.backward()
    _assert_gradients_flow(model)


# --------------------------------------- 11. REF_ARMS (5th amendment: gru-large)
def test_gru_large_registered_as_sentinel_tuple_in_ref_arms():
    """`gru-large` (Amendments, 2026-07-20 "gru-large grounding reference"
    entry) is registered in `REF_ARMS` with a SENTINEL tuple value -- same
    never-destructured convention as `MATRIX_ARMS["gru"]`."""
    assert "gru-large" in REF_ARMS
    assert "gru-large" in ARM_REGISTRY
    mixer, kwargs = REF_ARMS["gru-large"]
    assert mixer == "gru-large"
    assert kwargs is None


def test_gru_large_excluded_from_matrix_arms_and_decay_capable_arms():
    """`gru-large` is an explicitly NON-matched reference: it must never
    join `MATRIX_ARMS` (so `scripts/report_benchmarks.py`'s `-02`
    completeness accounting and `scripts/gpu_benchmark_campaign.py`'s
    default `--arms` never pick it up), and -- like `gru` -- it has no
    decay/`delta_t` mechanism to enable, so it is excluded from
    `DECAY_CAPABLE_ARMS` too."""
    assert "gru-large" not in MATRIX_ARMS
    assert "gru-large" not in DECAY_CAPABLE_ARMS


@pytest.mark.parametrize("task_name", ["s5", "mqar"])
def test_build_model_gru_large_token_sequence_hidden_256_two_layers(task_name):
    """`gru-large` on the two token-input tasks: a 2-layer `nn.GRU` at
    `GRU_LARGE_D_MODEL` (256) -- embedding, GRU hidden state, and head
    input all at 256, distinct from the matched `gru` arm's 64."""
    task = TASKS[task_name]
    model = build_model(task, "gru-large")
    assert isinstance(model, GRUTokenSequenceModel)
    assert model.rnn.num_layers == 2
    assert model.emb.embedding_dim == GRU_LARGE_D_MODEL == 256
    assert model.rnn.hidden_size == GRU_LARGE_D_MODEL
    assert model.head.in_features == GRU_LARGE_D_MODEL
    vocab, n_cls = model.emb.num_embeddings, model.head.out_features
    x = torch.randint(0, vocab, (2, 9))
    out = model(x)
    assert out.shape == (2, 9, n_cls)


def test_build_model_gru_large_feature_sequence_last_step_hidden_256():
    """`gru-large` on psMNIST (last_step): a 2-layer `nn.GRU` at
    `GRU_LARGE_D_MODEL` over the scalar pixel-stream input -- no
    `log1p(dt)` feature (psMNIST has no `dt`), matching `gru`'s input
    shape but with a distinct (256, not 64) hidden width."""
    task = TASKS["psmnist"]
    model = build_model(task, "gru-large")
    assert isinstance(model, GRUFeatureSequenceModel)
    assert model.rnn.num_layers == 2
    assert model.rnn.input_size == 1
    assert model.rnn.hidden_size == GRU_LARGE_D_MODEL == 256
    x = torch.rand(2, 12, 1)
    out = model(x)
    assert out.shape == (2, 12, 10)
    loss, correct, total = _forward_last_step(model, x, torch.zeros(2, dtype=torch.long))
    assert torch.isfinite(loss)
    assert total == 2


def test_build_model_gru_large_regression_gets_dt_as_feature_not_mechanical_decay():
    """`gru-large` on pendulum (regression): `input_size` is `in_dim + 1`
    (the `log1p(dt)` fairness-rule feature), same treatment as `gru` --
    the only difference is the 256 hidden width."""
    task = TASKS["pendulum"]
    model = build_model(task, "gru-large")
    assert isinstance(model, GRUFeatureSequenceModel)
    assert model.rnn.input_size == 3  # 2 state dims + 1 log1p(dt) feature
    assert model.rnn.hidden_size == GRU_LARGE_D_MODEL
    x = torch.randn(2, 8, 2)
    dt = torch.rand(2, 8).clamp(min=0.01)
    dt[:, 0] = 0.0
    y = torch.randn(2, 8, 2)
    loss, sq_err_sum, count = _forward_regression(model, x, dt, y, "gru-large")
    assert torch.isfinite(loss)
    assert count == y.numel()


def test_gru_and_gru_large_have_distinct_param_counts_and_hidden_widths():
    """`gru` (hidden 64) and `gru-large` (hidden 256) must build genuinely
    different models, not two names for the same construction -- both a
    shape check (hidden width) and a params check (the S3+ "distinct
    param counts" invariant this round's other arm pairs are held to)."""
    task = TASKS["psmnist"]
    gru_model = build_model(task, "gru")
    gru_large_model = build_model(task, "gru-large")
    assert gru_model.rnn.hidden_size == 64
    assert gru_large_model.rnn.hidden_size == 256
    gru_params = sum(p.numel() for p in gru_model.parameters())
    gru_large_params = sum(p.numel() for p in gru_large_model.parameters())
    assert gru_params != gru_large_params


def test_gru_large_row_layers_field_reports_two():
    """`run_arm`'s `"layers"` field must read `model.rnn.num_layers` for
    `gru-large` too (it has no `.stack` attribute, same as `gru`)."""
    tiny_tasks = _tiny_task_overrides()
    row = run_arm(
        round_tag="test-gru-large-layers",
        task=tiny_tasks["s5"],
        arm="gru-large",
        seed=0,
        device="cpu",
        dry_run=True,
    )
    assert row["layers"] == 2
    assert _model_layers(build_model(TASKS["s5"], "gru-large")) == 2


def _psmnist_task_with_synthetic_loader() -> TaskSpec:
    """The REAL psMNIST `TaskSpec` (`Budget.epochs=30`, the matched value
    `REF_ARM_BUDGETS` must override for `gru-large`) with its `data`
    swapped for the synthetic stand-in loader (module docstring: "may use
    synthetic stand-in batches ... to avoid download") -- everything else
    (budget, fit_metric, thresholds) stays the real, un-tiny-ified
    `TaskSpec`, unlike `_tiny_task_overrides`'s own psmnist entry, which
    also shrinks `Budget.epochs` to 1 and would mask this override."""
    return replace(
        TASKS["psmnist"],
        data=lambda batch_size: _SyntheticPsMNISTLoader(batch_size),
        budget=replace(TASKS["psmnist"].budget, batch_size=8),
    )


def test_ref_arm_budget_override_psmnist_gru_large_runs_60_epochs():
    """`REF_ARM_BUDGETS[("psmnist", "gru-large")]` (60 epochs, lr=1e-3) is
    the resolved DEFAULT `run_arm` applies for `gru-large` on psMNIST when
    no explicit `epochs=`/`lr=` argument is given -- the matched arms'
    frozen 30-epoch budget is a different (task, arm) key and must stay
    untouched (see the sibling test below)."""
    assert REF_ARM_BUDGETS[("psmnist", "gru-large")] == {"epochs": 60, "lr": 1e-3}
    row = run_arm(
        round_tag="test-ref-budget",
        task=_psmnist_task_with_synthetic_loader(),
        arm="gru-large",
        seed=0,
        device="cpu",
        dry_run=True,
    )
    assert row["config"]["budget"]["epochs"] == 60
    assert row["config"]["budget"]["lr"] == 1e-3


def test_ref_arm_budget_override_leaves_matched_gru_untouched():
    """The SAME `(task.name, arm)` budget-resolution code path must leave
    the matched `gru` arm's frozen 30-epoch budget exactly as it was
    before `REF_ARM_BUDGETS` existed -- `gru` has no entry in that dict."""
    row = run_arm(
        round_tag="test-matched-budget",
        task=_psmnist_task_with_synthetic_loader(),
        arm="gru",
        seed=0,
        device="cpu",
        dry_run=True,
    )
    assert row["config"]["budget"]["epochs"] == 30
    assert row["config"]["budget"]["lr"] == 1e-3


def test_gru_large_forward_backward_smoke_dense_s5():
    torch.manual_seed(0)
    model = build_model(TASKS["s5"], "gru-large")
    x = torch.randint(0, 120, (3, 10))
    y = torch.randint(0, 120, (3, 10))
    loss, _, _ = _forward_dense(model, x, y)
    loss.backward()
    _assert_gradients_flow(model)


def test_gru_large_forward_backward_smoke_masked_query_mqar():
    torch.manual_seed(0)
    model = build_model(TASKS["mqar"], "gru-large")
    gen = torch.Generator().manual_seed(1)
    x, y, mask = make_mqar(3, 32, gen, num_pairs=4)
    loss, _, _ = _forward_masked_query(model, x, y, mask)
    loss.backward()
    _assert_gradients_flow(model)


def test_gru_large_forward_backward_smoke_last_step_psmnist():
    torch.manual_seed(0)
    model = build_model(TASKS["psmnist"], "gru-large")
    x = torch.rand(3, 10, 1)
    y = torch.randint(0, 10, (3,))
    loss, _, _ = _forward_last_step(model, x, y)
    loss.backward()
    _assert_gradients_flow(model)


def test_gru_large_forward_backward_smoke_regression_pendulum():
    torch.manual_seed(0)
    model = build_model(TASKS["pendulum"], "gru-large")
    x = torch.randn(3, 8, 2)
    dt = torch.rand(3, 8).clamp(min=0.01)
    dt[:, 0] = 0.0
    y = torch.randn(3, 8, 2)
    loss, _, _ = _forward_regression(model, x, dt, y, "gru-large")
    loss.backward()
    _assert_gradients_flow(model)


# ------------------------------------------------------ TaskSpec singletons
def test_canonical_tasks_registry_shape():
    assert set(TASKS) == {"s5", "mqar", "psmnist", "pendulum"}
    for name, task in TASKS.items():
        assert isinstance(task, TaskSpec)
        assert task.name == name
        assert isinstance(task.budget, Budget)
        if task.loss_mode == "last_step":
            assert task.budget.epochs is not None and task.budget.steps is None
        else:
            assert task.budget.steps is not None and task.budget.epochs is None


def test_canonical_eval_protocol_configs_are_eval_config_instances():
    for task in TASKS.values():
        for ec in task.eval_protocol:
            assert isinstance(ec, EvalConfig)
