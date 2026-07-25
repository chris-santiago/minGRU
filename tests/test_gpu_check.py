"""Regression tests for ``scripts/gpu_check.py``'s extraction/dedup path.

``scripts/`` is not an importable package (no ``__init__.py`` -- see
``tests/test_scaling_probe.py``'s identical note), so this module is
loaded directly by file path via ``importlib``.

The ``_extract_last`` tests (Task 6) exercise it in complete isolation:
crafted strings only, no subprocess, no ``lightning_sdk``, no torch. They
pin the specific regression a live-job smoke test cannot reproduce on
demand: a truncated or malformed ``MINGRU_GPU_PROBE_RESULT`` line in a
job's fetched logs (interleaved with keepalive heartbeats and
clone/checkout noise) must degrade to "no match", never raise
``json.JSONDecodeError`` -- an uncaught raise here would propagate out of
``_finish_delta_probe`` and crash the CLI instead of reporting the clear
"no result line found" error the task brief requires (and, worse, could
leave a half-written artifact on disk if it happened mid-write instead of
before any write is attempted).

Mirrors ``tests/test_scaling_probe.py``'s test structure for
``scaling_probe.py``'s own ``_extract_last`` (same line-marker
extraction/malformed-line-tolerance pattern, duplicated per
``scripts/gpu_check.py``'s ``_extract_last`` docstring's
``DUPLICATION-PENDING`` note) -- kept as a parallel, independent test file
rather than merged, since the two scripts are independent job-runner entry
points.

The ``hetero36`` tests (Task 3, plus two quality-review fix cycles) cover
the GPU 36-seed round's submitter path: ``_extract_all`` (the ALL-rows
sibling of ``_extract_last``), ``_existing_keys_by_key``/``_append_new_rows``
(shape-guarded dedup + idempotent append against a ledger, including
intra-batch last-N-wins resolution with full
appended/skipped_duplicate/skipped_invalid/deduped_in_batch count
reconciliation), and ``_finish_hetero36`` end-to-end. ``_existing_keys_by_key``
and ``_dedup_batch_last_wins_by_key`` are the generic key-parameterized core
shared with the benchmarks job mode's dedup path (Task 6) -- the
hetero36-specific ``_existing_round_seed_pairs``/``_dedup_batch_last_wins``
wrappers were removed as dead code once ``_append_new_rows`` started calling
``_append_rows_by_key`` directly (see ``_hetero36_key``). All ledger and
sidecar I/O in these tests targets ``tmp_path`` fixtures via
``monkeypatch`` on the module's ``_LEDGER_PATH``/``_HETERO36_SIDECAR``
constants -- the real ``experiments/lab_results.jsonl`` and
``experiments/bench/gpu36_env.json`` are never touched.

The ``benchmarks`` tests (Task 6) cover the accepted-benchmark validation
round's submitter path: ``_valid_benchmarks_key`` (the four-field
``(round, task, variant, seed)`` sibling of ``_valid_hetero36_key`` --
``variant`` is load-bearing here since a task's arms share one round
tag), ``_append_benchmarks_rows`` (same shape-guarded dedup contract as
``_append_new_rows``, now over the four-field key -- pinning that two arms
at the same seed are NOT deduped against each other), ``_build_benchmarks_
sidecar`` (``per_variant_seed_wall_secs`` keyed ``{variant: {seed: secs}}``,
not ``{round: {seed: secs}}``), and ``_finish_benchmarks`` end-to-end
(multi-task logs split into one sidecar per task, a row with no
attributable ``task`` field warned about and dropped). All ledger/sidecar
I/O targets ``tmp_path`` fixtures via ``monkeypatch`` on the module's
``_LEDGER_PATH``/``_DELTA_PROBE_OUT_DIR`` constants -- the real
``experiments/lab_results.jsonl`` and ``experiments/bench/bench_*_env.json``
files are never touched.

The ``build_benchmarks_command`` tests (Task 8b, evidence-phase-gate
amendment) pin the job command chain itself: ``export MINGRU_SCAN=triton``
must appear ahead of the campaign invocation ("Triton everywhere" --
intent ledger Amendments), ``--steps`` must never be passed (production
budgets are the committed ``TaskSpec`` values), the chain stays a single
foreground ``&&``-joined string, and ``--tasks``/``--arms``/``--seeds``
passthrough is unaffected by the new export step. String-only assertions
against the built command -- no subprocess, no ``lightning_sdk``.

A later amendment (2026-07-20 entry, S5-only probe round) adds
``BENCH_PROBE_ROUND_TAGS``'s single ``bench-s5-probe-01`` tag to
``_BENCHMARKS_ROUNDS`` alongside the pilot/matrix generations -- the
finish handler must accept probe rows and dedup them by the same
four-field key as matrix rows, without touching the matrix accounting
already pinned above.

A further amendment (2026-07-20, "gru-large grounding reference") adds
``BENCH_REF_ROUND_TAGS``'s single ``bench-psmnist-ref-01`` tag to
``_BENCHMARKS_ROUNDS`` the same way -- the finish handler must accept
reference rows too, again without touching the matrix or probe
accounting.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "gpu_check.py"
_spec = importlib.util.spec_from_file_location("gpu_check", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
gpu_check = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("gpu_check", gpu_check)
_spec.loader.exec_module(gpu_check)

_extract_last = gpu_check._extract_last
_RESULT_PREFIX = gpu_check._DELTA_PROBE_RESULT_PREFIX
_extract_all = gpu_check._extract_all
_ROW_PREFIX = gpu_check._HETERO36_ROW_PREFIX
_ENV_PREFIX = gpu_check._HETERO36_ENV_PREFIX
_ROUNDS = gpu_check._HETERO36_ROUNDS
build_benchmarks_command = gpu_check.build_benchmarks_command


def test_no_matching_line_returns_none():
    text = "[keepalive] Fri Jul 18 00:00:00 UTC 2026\nCloning into '/tmp/minGRU'...\n"
    assert _extract_last(_RESULT_PREFIX, text) is None


def test_empty_text_returns_none():
    assert _extract_last(_RESULT_PREFIX, "") is None


def test_well_formed_line_parses():
    text = f'{_RESULT_PREFIX}{{"env": {{"torch_version": "2.8.0"}}, "shapes": []}}\n'
    assert _extract_last(_RESULT_PREFIX, text) == {
        "env": {"torch_version": "2.8.0"},
        "shapes": [],
    }


def test_well_formed_line_amid_keepalive_and_progress_noise():
    text = (
        "gpu_delta_probe: env {...}\n"
        "  running pd1024_T64 (B=128, T=64)...\n"
        "[keepalive] Fri Jul 18 00:05:00 UTC 2026\n"
        "    eager=0.0123s floor=0.0050s compile=0.0060s (ok) headroom=2.46\n"
        f'{_RESULT_PREFIX}{{"env": {{}}, "shapes": [{{"label": "pd1024_T64"}}]}}\n'
    )
    assert _extract_last(_RESULT_PREFIX, text) == {
        "env": {},
        "shapes": [{"label": "pd1024_T64"}],
    }


def test_truncated_line_degrades_to_none_not_raise():
    # Simulates a job killed (idle timeout / OOM / preemption) mid-print():
    # the JSON payload is cut off partway through. Must not raise
    # json.JSONDecodeError.
    text = f'{_RESULT_PREFIX}{{"env": {{"torch_version": "2.8.0"}}, "shap'
    assert _extract_last(_RESULT_PREFIX, text) is None


def test_malformed_line_does_not_clobber_earlier_valid_line():
    # An earlier matching line parses fine; a later matching line (same
    # prefix) is malformed. The malformed line is skipped rather than
    # overwriting the result with a parse failure -- the last
    # successfully-parsed match wins, and the function never raises.
    text = (
        f'{_RESULT_PREFIX}{{"env": {{}}, "shapes": [1]}}\n'
        f'{_RESULT_PREFIX}{{"env": {{}}, "shapes": [2], "truncated'
    )
    assert _extract_last(_RESULT_PREFIX, text) == {"env": {}, "shapes": [1]}


def test_prefix_must_match_exactly_not_substring():
    # A line that merely contains the prefix midway through must not match
    # (startswith, not "in") -- guards against a false match on stray log
    # noise that happens to embed the marker text.
    text = f'noise before {_RESULT_PREFIX}{{"env": {{}}, "shapes": []}}\n'
    assert _extract_last(_RESULT_PREFIX, text) is None


# --- _extract_all (Task 3: hetero36 job mode) -------------------------------


def _row(round_name: str, seed: int, secs: float = 10.0) -> dict:
    return {
        "round": round_name,
        "task": "S3-hier",
        "variant": "hetero-pg8",
        "layers": 2,
        "seed": seed,
        "steps": 1600,
        "acc": {"64": 1.0},
        "secs": secs,
        "max_steps": 1600,
        "ckpt": {"step": 1600, "val128": 1.0},
        "config": {"device": "cuda", "torch": "2.8.0", "scan": "triton"},
    }


def test_extract_all_no_matching_lines_returns_empty_list():
    text = "[keepalive] Fri Jul 18 00:00:00 UTC 2026\nCloning into '/tmp/minGRU'...\n"
    assert _extract_all(_ROW_PREFIX, text) == []


def test_extract_all_empty_text_returns_empty_list():
    assert _extract_all(_ROW_PREFIX, "") == []


def test_extract_all_well_formed_multi_row_preserves_log_order():
    rows = [_row(_ROUNDS[0], 0), _row(_ROUNDS[0], 1), _row(_ROUNDS[1], 0)]
    text = "".join(f"{_ROW_PREFIX}{json.dumps(r)}\n" for r in rows)
    assert _extract_all(_ROW_PREFIX, text) == rows


def test_extract_all_skips_malformed_lines_interleaved_with_well_formed():
    # A malformed row (truncated mid-print, same failure mode as the
    # delta-probe's single-line extraction) sits between two well-formed
    # rows and interleaved with campaign progress noise. The malformed
    # line is skipped -- never raises -- and both well-formed rows survive
    # in their original order.
    good_first = _row(_ROUNDS[0], 0)
    good_second = _row(_ROUNDS[0], 1)
    text = (
        f"{_ROW_PREFIX}{json.dumps(good_first)}\n"
        "  running hetero-gpu36-sg8 seed=1 (12.3s)...\n"
        f'{_ROW_PREFIX}{{"round": "{_ROUNDS[0]}", "seed": 1, "sec\n'
        f"{_ROW_PREFIX}{json.dumps(good_second)}\n"
    )
    assert _extract_all(_ROW_PREFIX, text) == [good_first, good_second]


def test_extract_all_prefix_must_match_exactly_not_substring():
    text = f"noise before {_ROW_PREFIX}{json.dumps(_row(_ROUNDS[0], 0))}\n"
    assert _extract_all(_ROW_PREFIX, text) == []


def test_extract_all_returns_non_dict_payloads_verbatim_shape_unfiltered():
    # _extract_all only guards PARSEABILITY, not SHAPE (see its docstring)
    # -- a well-formed-JSON, wrong-shape payload like ``[1, 2, 3]`` still
    # comes back verbatim; shape validation is the row-consuming path's
    # job (``_valid_hetero36_key``), not this generic extractor's.
    text = f"{_ROW_PREFIX}[1, 2, 3]\n"
    assert _extract_all(_ROW_PREFIX, text) == [[1, 2, 3]]


# --- dedup / append against the local ledger --------------------------------


def _hetero36_key(row: Any) -> tuple[str, int] | None:
    """Bind hetero36's ``(round, seed)`` key for the generic core helpers
    (``_existing_keys_by_key``/``_dedup_batch_last_wins_by_key``) -- the
    hetero36-specific ``_existing_round_seed_pairs``/``_dedup_batch_last_wins``
    wrappers were removed as dead code once ``_append_new_rows`` started
    calling ``_append_rows_by_key`` directly; these tests now exercise the
    generic core the same way the one remaining production call site does.
    """
    return gpu_check._valid_hetero36_key(row, _ROUNDS)


def test_existing_keys_by_key_empty_when_ledger_absent(tmp_path):
    ledger = tmp_path / "lab_results.jsonl"
    assert gpu_check._existing_keys_by_key(ledger, _hetero36_key) == set()


def test_existing_keys_by_key_filters_to_named_rounds_and_skips_malformed(
    tmp_path,
):
    ledger = tmp_path / "lab_results.jsonl"
    ledger.write_text(
        json.dumps(_row(_ROUNDS[0], 0))
        + "\n"
        + json.dumps(_row("hetero-loop-21-pd1024", 0))  # a different round, not ours
        + "\n"
        + "not json at all\n"
        + json.dumps(_row(_ROUNDS[1], 3))
        + "\n"
    )
    assert gpu_check._existing_keys_by_key(ledger, _hetero36_key) == {
        (_ROUNDS[0], 0),
        (_ROUNDS[1], 3),
    }


def test_existing_keys_by_key_skips_non_dict_ledger_line_without_crash(tmp_path):
    # A shape-invalid line already sitting in the ledger (e.g. from a bug
    # in an older version of the append path) must never crash every
    # future run that reads the ledger back -- it's skipped like any
    # other malformed/shape-invalid line.
    ledger = tmp_path / "lab_results.jsonl"
    ledger.write_text(
        json.dumps([1, 2, 3])  # valid JSON, not a row
        + "\n"
        + json.dumps(_row(_ROUNDS[0], 0))
        + "\n"
    )
    assert gpu_check._existing_keys_by_key(ledger, _hetero36_key) == {(_ROUNDS[0], 0)}


def _assert_counts_reconcile(rows_count: int, result) -> None:
    # Fix-cycle-2 REQUIRED FIX 1: every extracted row is accounted for by
    # exactly one of the four counts -- appended, skipped as an
    # already-in-ledger duplicate, skipped as shape-invalid, or deduped
    # away by a later same-batch occurrence of the same (round, seed).
    assert rows_count == (
        result.appended
        + result.skipped_duplicate
        + result.skipped_invalid
        + result.deduped_in_batch
    )


def test_append_new_rows_appends_all_when_ledger_empty(tmp_path):
    ledger = tmp_path / "lab_results.jsonl"
    rows = [_row(_ROUNDS[0], 0), _row(_ROUNDS[0], 1)]
    result = gpu_check._append_new_rows(ledger, rows, _ROUNDS)
    assert (result.appended, result.skipped_duplicate, result.skipped_invalid) == (2, 0, 0)
    assert result.deduped_in_batch == 0
    _assert_counts_reconcile(len(rows), result)
    assert [json.loads(line) for line in ledger.read_text().splitlines()] == rows


def test_append_new_rows_dedups_against_existing_ledger_rows(tmp_path):
    ledger = tmp_path / "lab_results.jsonl"
    ledger.write_text(json.dumps(_row(_ROUNDS[0], 0)) + "\n")
    rows = [_row(_ROUNDS[0], 0), _row(_ROUNDS[0], 1)]  # seed 0 already in ledger
    result = gpu_check._append_new_rows(ledger, rows, _ROUNDS)
    assert (result.appended, result.skipped_duplicate, result.skipped_invalid) == (1, 1, 0)
    _assert_counts_reconcile(len(rows), result)
    lines = ledger.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[-1])["seed"] == 1


def test_append_new_rows_preserves_log_order_across_rounds(tmp_path):
    ledger = tmp_path / "lab_results.jsonl"
    rows = [_row(_ROUNDS[2], 5), _row(_ROUNDS[0], 2), _row(_ROUNDS[1], 9)]
    gpu_check._append_new_rows(ledger, rows, _ROUNDS)
    written = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert written == rows


def test_append_new_rows_retry_is_idempotent(tmp_path):
    # Simulates a retried job: the same rows are appended twice (e.g. the
    # submitter reran after a transient log-fetch failure). The second
    # pass must skip every row as a duplicate and leave the ledger
    # unchanged.
    ledger = tmp_path / "lab_results.jsonl"
    rows = [_row(_ROUNDS[0], s) for s in range(3)]
    first = gpu_check._append_new_rows(ledger, rows, _ROUNDS)
    assert (first.appended, first.skipped_duplicate, first.skipped_invalid) == (3, 0, 0)
    _assert_counts_reconcile(len(rows), first)
    second = gpu_check._append_new_rows(ledger, rows, _ROUNDS)
    assert (second.appended, second.skipped_duplicate, second.skipped_invalid) == (0, 3, 0)
    _assert_counts_reconcile(len(rows), second)
    assert len(ledger.read_text().splitlines()) == 3


def test_append_new_rows_non_dict_payload_is_skipped_invalid_not_a_crash(tmp_path):
    # REQUIRED FIX 1(a): a well-formed-JSON, wrong-shape payload
    # (``MINGRU_LAB_ROW [1, 2, 3]``) must not raise AttributeError and
    # must not abort the rest of the batch.
    ledger = tmp_path / "lab_results.jsonl"
    good = _row(_ROUNDS[0], 0)
    rows: list = [[1, 2, 3], good]
    result = gpu_check._append_new_rows(ledger, rows, _ROUNDS)
    assert (result.appended, result.skipped_duplicate, result.skipped_invalid) == (1, 0, 1)
    _assert_counts_reconcile(len(rows), result)
    assert [json.loads(line) for line in ledger.read_text().splitlines()] == [good]


def test_append_new_rows_dict_missing_keys_is_skipped_invalid_not_appended(tmp_path):
    # REQUIRED FIX 1(b): a dict missing round/seed must never be appended
    # under a garbage (None, None) key.
    ledger = tmp_path / "lab_results.jsonl"
    incomplete = {"task": "S3-hier", "secs": 1.0}  # no round, no seed
    good = _row(_ROUNDS[0], 0)
    rows = [incomplete, good]
    result = gpu_check._append_new_rows(ledger, rows, _ROUNDS)
    assert (result.appended, result.skipped_duplicate, result.skipped_invalid) == (1, 0, 1)
    _assert_counts_reconcile(len(rows), result)
    written = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert written == [good]
    assert (None, None) not in gpu_check._existing_keys_by_key(ledger, _hetero36_key)


def test_append_new_rows_wrong_round_name_is_skipped_invalid(tmp_path):
    ledger = tmp_path / "lab_results.jsonl"
    wrong_round = _row("hetero-loop-21-pd1024", 0)  # not one of _HETERO36_ROUNDS
    good = _row(_ROUNDS[0], 0)
    rows = [wrong_round, good]
    result = gpu_check._append_new_rows(ledger, rows, _ROUNDS)
    assert (result.appended, result.skipped_duplicate, result.skipped_invalid) == (1, 0, 1)
    _assert_counts_reconcile(len(rows), result)
    written = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert written == [good]


def test_append_new_rows_bool_seed_is_skipped_invalid_not_seed_one(tmp_path):
    # REQUIRED FIX 2 (fix-cycle 2): a JSON ``true``/``false`` seed must
    # never be silently accepted as the int seed 1/0 -- ``bool`` is an
    # ``int`` subclass in Python, so a naive ``isinstance(seed, int)``
    # check alone would wrongly accept it.
    ledger = tmp_path / "lab_results.jsonl"
    bool_seed = _row(_ROUNDS[0], 0)
    bool_seed["seed"] = True
    rows = [bool_seed]
    result = gpu_check._append_new_rows(ledger, rows, _ROUNDS)
    assert (result.appended, result.skipped_duplicate, result.skipped_invalid) == (0, 0, 1)
    _assert_counts_reconcile(len(rows), result)
    assert not ledger.exists()
    assert gpu_check._valid_hetero36_key(bool_seed, _ROUNDS) is None


def test_append_new_rows_intra_batch_duplicate_keeps_last_secs(tmp_path):
    # REQUIRED FIX 2 (fix-cycle 1) / spec §6: "extraction is last-N-wins
    # per (round, seed)". Two rows in the SAME batch share (round, seed)
    # but differ in secs -- the ledger must end up with the LAST
    # occurrence's value, and the loser is counted as deduped_in_batch
    # (REQUIRED FIX 1, fix-cycle 2), not silently dropped from the
    # accounting.
    ledger = tmp_path / "lab_results.jsonl"
    first_seen = _row(_ROUNDS[0], 0, secs=11.0)
    last_seen = _row(_ROUNDS[0], 0, secs=99.0)
    rows = [first_seen, last_seen]
    result = gpu_check._append_new_rows(ledger, rows, _ROUNDS)
    assert (result.appended, result.skipped_duplicate, result.skipped_invalid) == (1, 0, 0)
    assert result.deduped_in_batch == 1
    _assert_counts_reconcile(len(rows), result)
    written = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert written == [last_seen]
    assert written[0]["secs"] == 99.0


# --- _finish_hetero36 end-to-end (ledger/sidecar paths monkeypatched to
# tmp_path -- never the real experiments/ files) ----------------------------


class _FakeJob:
    def __init__(self, logs: str) -> None:
        self.logs = logs


def _patch_hetero36_paths(monkeypatch, tmp_path):
    ledger = tmp_path / "lab_results.jsonl"
    sidecar = tmp_path / "gpu36_env.json"
    monkeypatch.setattr(gpu_check, "_LEDGER_PATH", ledger)
    monkeypatch.setattr(gpu_check, "_HETERO36_SIDECAR", sidecar)
    return ledger, sidecar


def test_finish_hetero36_absent_rows_is_clear_error_and_writes_nothing(tmp_path, monkeypatch):
    ledger, sidecar = _patch_hetero36_paths(monkeypatch, tmp_path)
    job = _FakeJob(logs="no marked lines in this log at all\n")
    rc = gpu_check._finish_hetero36(job, ok=True)
    assert rc == 1
    assert not ledger.exists()
    assert not sidecar.exists()


def test_finish_hetero36_absent_env_line_is_clear_error_and_writes_nothing(tmp_path, monkeypatch):
    ledger, sidecar = _patch_hetero36_paths(monkeypatch, tmp_path)
    row = _row(_ROUNDS[0], 0)
    logs = f"{_ROW_PREFIX}{json.dumps(row)}\n"  # rows present, no MINGRU_LAB_ENV line
    job = _FakeJob(logs=logs)
    rc = gpu_check._finish_hetero36(job, ok=True)
    assert rc == 1
    assert not ledger.exists()
    assert not sidecar.exists()


def test_finish_hetero36_appends_rows_and_writes_sidecar(tmp_path, monkeypatch):
    ledger, sidecar = _patch_hetero36_paths(monkeypatch, tmp_path)
    rows = [_row(_ROUNDS[0], 0), _row(_ROUNDS[1], 0, secs=20.0)]
    env = {"torch": "2.8.0", "cuda_device_name": "NVIDIA L4"}
    logs = "".join(f"{_ROW_PREFIX}{json.dumps(r)}\n" for r in rows)
    logs += f"{_ENV_PREFIX}{json.dumps(env)}\n"
    job = _FakeJob(logs=logs)

    rc = gpu_check._finish_hetero36(job, ok=True)

    assert rc == 0
    assert [json.loads(line) for line in ledger.read_text().splitlines()] == rows
    sidecar_data = json.loads(sidecar.read_text())
    assert sidecar_data["env"] == env
    assert sidecar_data["rows_extracted"] == 2
    assert sidecar_data["rows_appended"] == 2
    assert sidecar_data["rows_skipped_duplicate"] == 0
    assert sidecar_data["rows_skipped_invalid"] == 0
    assert sidecar_data["rows_deduped_in_batch"] == 0
    assert sidecar_data["per_seed_wall_secs"][_ROUNDS[0]]["0"] == rows[0]["secs"]
    assert sidecar_data["per_seed_wall_secs"][_ROUNDS[1]]["0"] == 20.0


def test_finish_hetero36_shape_invalid_row_is_skipped_with_warning_not_crash(
    tmp_path, monkeypatch, capsys
):
    # REQUIRED FIX 1 end-to-end: a non-dict payload line sits alongside a
    # well-formed row. The batch must not crash; the invalid row is
    # skipped, counted in the sidecar, warned about on stderr, and the
    # well-formed row still gets appended.
    ledger, sidecar = _patch_hetero36_paths(monkeypatch, tmp_path)
    good = _row(_ROUNDS[0], 0)
    logs = (
        f"{_ROW_PREFIX}[1, 2, 3]\n"
        f"{_ROW_PREFIX}{json.dumps(good)}\n"
        f"{_ENV_PREFIX}{json.dumps({'torch': '2.8.0'})}\n"
    )
    job = _FakeJob(logs=logs)

    rc = gpu_check._finish_hetero36(job, ok=True)

    assert rc == 0
    written = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert written == [good]
    sidecar_data = json.loads(sidecar.read_text())
    assert sidecar_data["rows_extracted"] == 2
    assert sidecar_data["rows_appended"] == 1
    assert sidecar_data["rows_skipped_invalid"] == 1
    assert sidecar_data["rows_skipped_duplicate"] == 0
    assert sidecar_data["rows_deduped_in_batch"] == 0
    assert "shape-invalid" in capsys.readouterr().err


def test_finish_hetero36_intra_batch_duplicate_reconciles_and_ledger_keeps_last(
    tmp_path, monkeypatch
):
    # REQUIRED FIX 1 (fix-cycle 2) end-to-end: an intra-batch duplicate
    # (round, seed) with a losing row must not vanish from the sidecar's
    # accounting, and the ledger must keep the LAST occurrence.
    ledger, sidecar = _patch_hetero36_paths(monkeypatch, tmp_path)
    first_seen = _row(_ROUNDS[0], 0, secs=11.0)
    last_seen = _row(_ROUNDS[0], 0, secs=99.0)
    other = _row(_ROUNDS[1], 0)
    rows = [first_seen, last_seen, other]
    env = {"torch": "2.8.0"}
    logs = "".join(f"{_ROW_PREFIX}{json.dumps(r)}\n" for r in rows)
    logs += f"{_ENV_PREFIX}{json.dumps(env)}\n"
    job = _FakeJob(logs=logs)

    rc = gpu_check._finish_hetero36(job, ok=True)

    assert rc == 0
    written = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert written == [last_seen, other]
    sidecar_data = json.loads(sidecar.read_text())
    assert sidecar_data["rows_extracted"] == 3
    assert sidecar_data["rows_appended"] == 2
    assert sidecar_data["rows_skipped_duplicate"] == 0
    assert sidecar_data["rows_skipped_invalid"] == 0
    assert sidecar_data["rows_deduped_in_batch"] == 1
    # Full reconciliation: extracted == appended + skipped_duplicate +
    # skipped_invalid + deduped_in_batch.
    assert sidecar_data["rows_extracted"] == (
        sidecar_data["rows_appended"]
        + sidecar_data["rows_skipped_duplicate"]
        + sidecar_data["rows_skipped_invalid"]
        + sidecar_data["rows_deduped_in_batch"]
    )
    assert sidecar_data["per_seed_wall_secs"][_ROUNDS[0]]["0"] == 99.0


def test_finish_hetero36_nonzero_exit_when_job_not_ok_despite_valid_rows(tmp_path, monkeypatch):
    # A job that failed overall (e.g. timed out mid-matrix) can still have
    # produced well-formed rows/env for the seeds that did complete; those
    # are still extracted, deduped, and appended, but the exit code
    # reflects the job's own failure.
    ledger, sidecar = _patch_hetero36_paths(monkeypatch, tmp_path)
    row = _row(_ROUNDS[0], 0)
    logs = f"{_ROW_PREFIX}{json.dumps(row)}\n{_ENV_PREFIX}{json.dumps({'torch': '2.8.0'})}\n"
    job = _FakeJob(logs=logs)

    rc = gpu_check._finish_hetero36(job, ok=False)

    assert rc == 1
    assert ledger.exists()


def test_finish_hetero36_retry_is_idempotent_end_to_end(tmp_path, monkeypatch):
    ledger, sidecar = _patch_hetero36_paths(monkeypatch, tmp_path)
    rows = [_row(_ROUNDS[0], 0), _row(_ROUNDS[0], 1)]
    env = {"torch": "2.8.0"}
    logs = "".join(f"{_ROW_PREFIX}{json.dumps(r)}\n" for r in rows)
    logs += f"{_ENV_PREFIX}{json.dumps(env)}\n"
    job = _FakeJob(logs=logs)

    first_rc = gpu_check._finish_hetero36(job, ok=True)
    second_rc = gpu_check._finish_hetero36(job, ok=True)

    assert (first_rc, second_rc) == (0, 0)
    assert len(ledger.read_text().splitlines()) == 2
    sidecar_data = json.loads(sidecar.read_text())
    assert sidecar_data["rows_appended"] == 0
    assert sidecar_data["rows_skipped_duplicate"] == 2
    assert sidecar_data["rows_skipped_invalid"] == 0
    assert sidecar_data["rows_deduped_in_batch"] == 0
    assert sidecar_data["rows_extracted"] == 2


# --- _render_delta_probe_markdown (Task 5: triton arm + bar judgment) ------
#
# ``scripts/gpu_delta_probe.py``'s own arms are exercised nowhere in this
# CPU-only test suite (no CUDA) -- these tests target the renderer in
# complete isolation, on hand-crafted result dicts shaped like
# ``_run_shape``'s return value, mirroring this file's existing
# renderer-adjacent idiom (crafted strings/dicts only, no subprocess, no
# torch).

_fmt_bar_judgment = gpu_check._fmt_bar_judgment
_fmt_bytes_mb = gpu_check._fmt_bytes_mb
_render_delta_probe_markdown = gpu_check._render_delta_probe_markdown


def _delta_probe_shape_row(**overrides: Any) -> dict:
    """A ``_run_shape``-shaped row, all triton/bar fields present by default.

    ``overrides`` replaces individual fields so each test only states what
    it cares about (e.g. a failed compile arm, a failed speed bar).
    """
    row = {
        "label": "pd1024_T64",
        "config_name": "pd1024",
        "config": {
            "input_size": 64,
            "hidden_size": 64,
            "n_heads": 4,
            "nh": 2,
            "d_k": 16,
            "d_v": 16,
        },
        "live_config": {
            "n_heads": 4,
            "nh": 2,
            "d_k": 16,
            "d_v": 16,
            "chunk_size": 64,
            "state_elements": 1024,
        },
        "B": 128,
        "T": 64,
        "eager_step_secs_median": 0.0074,
        "eager_step_secs_all": [0.0074] * 10,
        "eager_peak_mem_bytes": 186_914_816,
        "floor_step_secs": 0.0028,
        "floor_forward_only_secs_all": [0.00093] * 10,
        "floor_method": "standalone GEMM/solve ops, see module docstring",
        "floor_op_inventory": [],
        "floor_suspect": False,
        "compile_step_secs_median": 0.0041,
        "compile_step_secs_all": [0.0041] * 10,
        "compile_status": "ok",
        "compile_error": None,
        "headroom_eager_over_floor": 2.6,
        "compile_recovered_fraction": 0.7163,
        "triton_step_secs_median": 0.0045,
        "triton_step_secs_all": [0.0045] * 10,
        "triton_peak_mem_bytes": 150_000_000,
        "triton_vs_eager_ratio": 0.6081,
        "triton_vs_compile_ratio": 1.0976,
        "bar_met_vs_eager": True,
        "bar_met_vs_compile": True,
        "bar_met": True,
        "triton_vs_eager_peak_mem_ratio": 0.8025,
        "memory_bar_met": True,
    }
    row.update(overrides)
    return row


def _delta_probe_result(*rows: dict) -> dict:
    return {
        "env": {
            "torch_version": "2.8.0+cu128",
            "cuda_version": "12.8",
            "cuda_device_name": "NVIDIA L4",
            "device_capability": [8, 9],
            "platform": "Linux-x86_64",
            "batch_size": 128,
            "warmup_steps": 3,
            "timed_steps": 10,
            "timestamp": "2026-07-18T09:57:13+00:00",
            "triton_version": "3.4.0",
        },
        "shapes": list(rows),
    }


def test_fmt_bar_judgment_three_states():
    assert _fmt_bar_judgment(True) == "PASS"
    assert _fmt_bar_judgment(False) == "FAIL"
    assert _fmt_bar_judgment(None) == "n/a"


def test_fmt_bytes_mb_renders_megabytes_and_none():
    assert _fmt_bytes_mb(150_000_000) == "150.0"
    assert _fmt_bytes_mb(None) == "n/a"
    assert _fmt_bytes_mb("not a number") == "n/a"


def test_render_delta_probe_markdown_includes_triton_column_and_bar_pass():
    result = _delta_probe_result(_delta_probe_shape_row())
    md = _render_delta_probe_markdown(result)
    assert "triton median (s)" in md
    assert "bar met" in md
    assert "0.0045" in md  # triton_step_secs_median
    # The pd1024_T64 row's bar_met=True renders as PASS somewhere on its row.
    row_line = next(line for line in md.splitlines() if line.startswith("| pd1024_T64"))
    assert row_line.strip().endswith("| PASS |")


def test_render_delta_probe_markdown_bar_fail_when_triton_slower_than_eager():
    row = _delta_probe_shape_row(
        triton_step_secs_median=0.02,
        triton_vs_eager_ratio=2.7,
        bar_met_vs_eager=False,
        bar_met=False,
    )
    md = _render_delta_probe_markdown(_delta_probe_result(row))
    row_line = next(line for line in md.splitlines() if line.startswith("| pd1024_T64"))
    assert row_line.strip().endswith("| FAIL |")


def test_render_delta_probe_markdown_bar_na_when_compile_arm_failed():
    # bar_met_vs_compile/bar_met are None (unjudgeable), not False -- a
    # failed compile arm must never render as a failed speed bar.
    row = _delta_probe_shape_row(
        compile_step_secs_median=None,
        compile_step_secs_all=[],
        compile_status="failed",
        compile_error="RuntimeError: backend compiler failed",
        compile_recovered_fraction=None,
        triton_vs_compile_ratio=None,
        bar_met_vs_compile=None,
        bar_met=None,
    )
    md = _render_delta_probe_markdown(_delta_probe_result(row))
    row_line = next(line for line in md.splitlines() if line.startswith("| pd1024_T64"))
    assert row_line.strip().endswith("| n/a |")
    assert "FAIL" not in row_line


def test_render_delta_probe_markdown_memory_table_present_with_bar_judgment():
    result = _delta_probe_result(_delta_probe_shape_row())
    md = _render_delta_probe_markdown(result)
    assert "peak mem (MB)" in md
    assert "memory bar met" in md
    assert "186.9" in md  # eager_peak_mem_bytes / 1e6
    assert "150.0" in md  # triton_peak_mem_bytes / 1e6


def test_render_delta_probe_markdown_memory_bar_fail_renders_fail():
    row = _delta_probe_shape_row(
        triton_peak_mem_bytes=400_000_000,
        triton_vs_eager_peak_mem_ratio=2.14,
        memory_bar_met=False,
    )
    md = _render_delta_probe_markdown(_delta_probe_result(row))
    memory_line = next(
        line for line in md.splitlines() if line.startswith("| pd1024_T64") and "400.0" in line
    )
    assert memory_line.strip().endswith("| FAIL |")


def test_render_delta_probe_markdown_missing_triton_fields_degrades_to_na_not_crash():
    # A row shaped like the OLDER (pre-Task-5) artifact -- no triton/bar
    # keys at all -- must render "n/a" for the new columns rather than
    # raising KeyError, so the renderer stays usable against an
    # un-regenerated artifact.
    old_row = {
        "label": "pd1024_T64",
        "config_name": "pd1024",
        "live_config": {"n_heads": 4, "nh": 2, "d_k": 16, "d_v": 16, "chunk_size": 64},
        "B": 128,
        "T": 64,
        "eager_step_secs_median": 0.0074,
        "eager_peak_mem_bytes": 186_914_816,
        "floor_step_secs": 0.0028,
        "compile_step_secs_median": 0.0041,
        "compile_status": "ok",
        "headroom_eager_over_floor": 2.6,
        "compile_recovered_fraction": 0.7163,
    }
    md = _render_delta_probe_markdown(_delta_probe_result(old_row))
    row_line = next(line for line in md.splitlines() if line.startswith("| pd1024_T64"))
    assert row_line.strip().endswith("| n/a |")
    memory_line = next(
        line for line in md.splitlines() if line.startswith("| pd1024_T64") and "186.9" in line
    )
    assert "n/a" in memory_line


def test_render_delta_probe_markdown_no_shapes_still_renders_headers():
    md = _render_delta_probe_markdown(_delta_probe_result())
    assert "triton median (s)" in md
    assert "memory bar met" in md


# --- benchmarks job mode (Task 6): _valid_benchmarks_key / dedup / sidecar /
# _finish_benchmarks end-to-end ---------------------------------------------

_BENCH_ROUNDS = gpu_check._BENCHMARKS_ROUNDS
_BENCH_ROW_PREFIX = gpu_check._BENCHMARKS_ROW_PREFIX
_BENCH_ENV_PREFIX = gpu_check._BENCHMARKS_ENV_PREFIX


def _bench_row(round_name: str, task: str, variant: str, seed: int, secs: float = 10.0) -> dict:
    return {
        "round": round_name,
        "task": task,
        "variant": variant,
        "layers": 2,
        "seed": seed,
        "steps": 3000,
        "acc": {"128": 1.0},
        "secs": secs,
        "ckpt": {"step": 3000, "val128": 1.0},
        "config": {"device": "cuda", "torch": "2.8.0"},
    }


# `_build_benchmarks_sidecar` takes an `_AppendResult`; the merge tests below
# only care about `_build_benchmarks_sidecar`'s row-shaped output, not any
# particular dedup outcome, so a fixed "one row appended, nothing skipped"
# result is reused across them rather than re-deriving one per call site.
_APPEND_ONE = gpu_check._AppendResult(
    appended=1, skipped_duplicate=0, skipped_invalid=0, deduped_in_batch=0
)


# --------------------------------------------- build_benchmarks_command
# Evidence-phase-gate amendment ("Triton everywhere" -- intent ledger
# Amendments): every seed matrix in this round runs under a uniform
# MINGRU_SCAN=triton backend on L4, exported ahead of the campaign
# invocation in the job command chain built here.
def test_build_benchmarks_command_exports_mingru_scan_triton():
    command = build_benchmarks_command("git@example.com/repo.git", "deadbeef", None, None, None)
    assert "export MINGRU_SCAN=triton" in command
    # The export must precede the campaign invocation it's meant to cover,
    # not merely appear somewhere in the command chain.
    assert command.index("export MINGRU_SCAN=triton") < command.index(
        "python scripts/gpu_benchmark_campaign.py"
    )


def test_build_benchmarks_command_never_passes_steps():
    # Production budgets are the committed TaskSpec values (spec section 7:
    # "no per-arm tuning") -- --steps is a local-smoke-only override this
    # job command must never pass, amendment or not.
    command = build_benchmarks_command("git@example.com/repo.git", "deadbeef", None, None, None)
    assert "--steps" not in command


def test_build_benchmarks_command_still_single_foreground_chain():
    # Job command chains are foreground-only (no backgrounded keepalive) --
    # a single `&&`-joined string, unchanged by the new export step.
    command = build_benchmarks_command("git@example.com/repo.git", "deadbeef", None, None, None)
    assert "&" not in command.replace("&&", "")
    assert "set -eux" in command


def test_build_benchmarks_command_passthrough_args_unaffected_by_triton_export():
    command = build_benchmarks_command(
        "git@example.com/repo.git", "deadbeef", ["s5"], ["log", "signed-rotation"], [0, 1]
    )
    assert "export MINGRU_SCAN=triton" in command
    assert "--tasks s5" in command
    assert "--arms log signed-rotation" in command
    assert "--seeds 0 1" in command
    assert "--steps" not in command


def test_valid_benchmarks_key_accepts_well_formed_row():
    row = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) == (_BENCH_ROUNDS[0], "s5", "log", 0)


def test_benchmarks_rounds_accepts_both_pilot_and_current_generations():
    """`_BENCHMARKS_ROUNDS` must accept BOTH the frozen `-01` pilot tags
    (old pilot job logs/sidecars must stay parseable) and the current `-02`
    tags this module reads from `experiments.benchmark_tasks
    .BENCH_ROUND_TAGS` (pre-matrix technical review, item 1)."""
    from experiments.benchmark_tasks import BENCH_ROUND_TAGS

    # `bench-churn-01` (Task 6, churn round) is hardcoded here alongside the
    # four heterogeneous-budget `-01` tags for a different reason (it's
    # churn's own from-the-start pilot tag, not a superseded population under
    # the current `BENCH_ROUND_TAGS["churn"]` matrix tag) but gets identical
    # frozen-forever allow-list treatment -- see `_BENCHMARKS_ROUNDS_PILOT`'s
    # comment in `scripts/gpu_check.py`.
    pilot_tags = (
        "bench-s5-01",
        "bench-mqar-01",
        "bench-psmnist-01",
        "bench-pendulum-01",
        "bench-churn-01",
    )
    for tag in pilot_tags:
        assert tag in _BENCH_ROUNDS
    for tag in BENCH_ROUND_TAGS.values():
        assert tag in _BENCH_ROUNDS
    # No accidental collision between the two generations (the S5-only
    # probe round, the psMNIST-only ref round, and the per-arm correction
    # overrides are three further, separately-counted generations -- see
    # test_benchmarks_rounds_also_accepts_the_probe_round,
    # test_benchmarks_rounds_also_accepts_the_ref_round, and
    # test_benchmarks_rounds_also_accepts_the_override_rounds below).
    from experiments.benchmark_tasks import (
        BENCH_ARM_ROUND_OVERRIDES,
        BENCH_PROBE_ROUND_TAGS,
        BENCH_REF_ROUND_TAGS,
    )

    n_override_tags = sum(
        len(per_task_tags) for per_task_tags in BENCH_ARM_ROUND_OVERRIDES.values()
    )
    n_generations = (
        len(pilot_tags)
        + len(BENCH_ROUND_TAGS)
        + len(BENCH_PROBE_ROUND_TAGS)
        + len(BENCH_REF_ROUND_TAGS)
        + n_override_tags
    )
    assert len(_BENCH_ROUNDS) == n_generations
    # Pinned to the design spec's populated values (4 rotfix + 1
    # probe-rotfix): this count must grow by exactly the override-tag
    # generation, not drift silently if the override map's shape changes.
    assert n_override_tags == 5


def test_benchmarks_rounds_also_accepts_the_probe_round():
    """`_BENCHMARKS_ROUNDS` must additionally accept the S5-only probe
    round tag (`experiments.benchmark_tasks.BENCH_PROBE_ROUND_TAGS`,
    Amendments 2026-07-20 entry) -- the finish handler is the one place
    that must recognize probe rows as this job mode's own data, alongside
    the pilot and matrix generations."""
    from experiments.benchmark_tasks import BENCH_PROBE_ROUND_TAGS

    assert BENCH_PROBE_ROUND_TAGS == {"s5": "bench-s5-probe-01", "churn": "bench-churn-probe-01"}
    for tag in BENCH_PROBE_ROUND_TAGS.values():
        assert tag in _BENCH_ROUNDS


def test_benchmarks_rounds_also_accepts_the_ref_round():
    """`_BENCHMARKS_ROUNDS` must additionally accept the psMNIST-only
    reference round tag (`experiments.benchmark_tasks.BENCH_REF_ROUND_TAGS`,
    "gru-large grounding reference" amendment) -- the finish handler must
    recognize `gru-large` reference rows as this job mode's own data too,
    alongside the pilot, matrix, and probe generations."""
    from experiments.benchmark_tasks import BENCH_REF_ROUND_TAGS

    assert BENCH_REF_ROUND_TAGS == {"psmnist": "bench-psmnist-ref-01", "churn": "bench-churn-ref-01"}
    for tag in BENCH_REF_ROUND_TAGS.values():
        assert tag in _BENCH_ROUNDS


def test_benchmarks_rounds_also_accepts_the_override_rounds():
    """`_BENCHMARKS_ROUNDS` must additionally accept every per-arm
    correction tag (`experiments.benchmark_tasks.BENCH_ARM_ROUND_OVERRIDES`,
    round-tag override design spec §4 Ingest) -- the finish handler must
    recognize corrected `signed-rotation`/`signed-rotation-k5` rows as this
    job mode's own data too, alongside the pilot, matrix, probe, and ref
    generations, while every superseded prior tag remains accepted
    (spec's non-collision/never-mutated invariants)."""
    from experiments.benchmark_tasks import BENCH_ARM_ROUND_OVERRIDES

    assert BENCH_ARM_ROUND_OVERRIDES == {
        "signed-rotation": {
            "s5": "bench-s5-rotfix-01",
            "mqar": "bench-mqar-rotfix-01",
            "psmnist": "bench-psmnist-rotfix-01",
            "pendulum": "bench-pendulum-rotfix-01",
        },
        "signed-rotation-k5": {
            "s5": "bench-s5-probe-rotfix-01",
        },
    }
    for per_task_tags in BENCH_ARM_ROUND_OVERRIDES.values():
        for tag in per_task_tags.values():
            assert tag in _BENCH_ROUNDS


def test_valid_benchmarks_key_accepts_a_row_carrying_an_override_tag():
    """A row whose round is a correction tag (not a pilot/matrix/probe/ref
    tag) is accepted by the ingest key/allow-list -- the concrete
    behavioral proof behind
    test_benchmarks_rounds_also_accepts_the_override_rounds above."""
    row = _bench_row("bench-s5-rotfix-01", "s5", "signed-rotation", 0)
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) == (
        "bench-s5-rotfix-01",
        "s5",
        "signed-rotation",
        0,
    )


def test_valid_benchmarks_key_accepts_a_probe_round_and_variant_row():
    row = _bench_row("bench-s5-probe-01", "s5", "signed-rotation-k5", 99)
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) == (
        "bench-s5-probe-01",
        "s5",
        "signed-rotation-k5",
        99,
    )


def test_append_benchmarks_rows_dedups_probe_round_rows_by_full_key(tmp_path):
    # Same (round, task, variant, seed) dedup contract applies to probe
    # rows as to matrix rows -- variant is still load-bearing (two
    # different probe arms at the same seed under the probe round tag are
    # not duplicates of each other).
    ledger = tmp_path / "lab_results.jsonl"
    rows = [
        _bench_row("bench-s5-probe-01", "s5", "signed-rotation-k5", 99),
        _bench_row("bench-s5-probe-01", "s5", "signed-delta-nh3", 99),
        _bench_row("bench-s5-probe-01", "s5", "signed-rotation-k5", 99),  # duplicate
    ]
    result = gpu_check._append_benchmarks_rows(ledger, rows, _BENCH_ROUNDS)
    assert (result.appended, result.skipped_duplicate, result.skipped_invalid) == (2, 0, 0)
    assert result.deduped_in_batch == 1
    written = {json.loads(line)["variant"] for line in ledger.read_text().splitlines()}
    assert written == {"signed-rotation-k5", "signed-delta-nh3"}


def test_valid_benchmarks_key_rejects_non_dict_payload():
    assert gpu_check._valid_benchmarks_key([1, 2, 3], _BENCH_ROUNDS) is None


def test_valid_benchmarks_key_rejects_unrecognized_round():
    row = _bench_row("not-a-benchmarks-round", "s5", "log", 0)
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) is None


def test_valid_benchmarks_key_rejects_non_string_variant():
    row = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)
    row["variant"] = 3  # not a string
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) is None


def test_valid_benchmarks_key_rejects_non_string_task():
    row = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)
    row["task"] = 3  # not a string
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) is None


def test_valid_benchmarks_key_rejects_bool_seed_not_seed_one():
    # Same bool/int subtlety pinned for hetero36 -- a JSON true/false must
    # never be silently accepted as the int seed 1/0.
    row = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)
    row["seed"] = True
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) is None


def test_valid_benchmarks_key_rejects_empty_string_task():
    # Quality review OPTIONAL FIX 3: the docstring promises "non-empty
    # strings" -- an empty task must be rejected, not silently accepted
    # (an empty task would otherwise route to a malformed
    # `bench__env.json` sidecar path in `_finish_benchmarks`).
    row = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)
    row["task"] = ""
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) is None


def test_valid_benchmarks_key_rejects_empty_string_variant():
    row = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)
    row["variant"] = ""
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) is None


def test_row_task_rejects_non_dict_and_empty_string_returns_task_for_valid_row():
    assert gpu_check._row_task([1, 2, 3]) is None
    assert gpu_check._row_task({"task": ""}) is None
    assert gpu_check._row_task({"task": 3}) is None
    assert gpu_check._row_task({"task": "s5"}) == "s5"


def test_append_benchmarks_rows_two_arms_same_seed_are_not_duplicates(tmp_path):
    # The load-bearing difference from hetero36's (round, seed) key: two
    # different arms (variants) at the SAME seed under the SAME round tag
    # must both be appended, not treated as duplicates of each other.
    ledger = tmp_path / "lab_results.jsonl"
    rows = [
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0),
        _bench_row(_BENCH_ROUNDS[0], "s5", "signed", 0),
    ]
    result = gpu_check._append_benchmarks_rows(ledger, rows, _BENCH_ROUNDS)
    assert (result.appended, result.skipped_duplicate, result.skipped_invalid) == (2, 0, 0)
    assert result.deduped_in_batch == 0
    written = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert written == rows


def test_append_benchmarks_rows_dedups_same_round_task_variant_seed(tmp_path):
    ledger = tmp_path / "lab_results.jsonl"
    ledger.write_text(json.dumps(_bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)) + "\n")
    rows = [
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0),  # already in ledger
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 1),
    ]
    result = gpu_check._append_benchmarks_rows(ledger, rows, _BENCH_ROUNDS)
    assert (result.appended, result.skipped_duplicate, result.skipped_invalid) == (1, 1, 0)
    lines = ledger.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[-1])["seed"] == 1


def test_append_benchmarks_rows_retry_is_idempotent(tmp_path):
    ledger = tmp_path / "lab_results.jsonl"
    rows = [_bench_row(_BENCH_ROUNDS[0], "s5", "log", s) for s in range(3)]
    first = gpu_check._append_benchmarks_rows(ledger, rows, _BENCH_ROUNDS)
    assert (first.appended, first.skipped_duplicate, first.skipped_invalid) == (3, 0, 0)
    second = gpu_check._append_benchmarks_rows(ledger, rows, _BENCH_ROUNDS)
    assert (second.appended, second.skipped_duplicate, second.skipped_invalid) == (0, 3, 0)
    assert len(ledger.read_text().splitlines()) == 3


def test_build_benchmarks_sidecar_keys_per_variant_seed_not_bare_seed():
    # Two arms sharing seed 0 must land under distinct variant sub-keys,
    # never collide onto a single seed-keyed map the way hetero36's
    # per_seed_wall_secs would (there, one round already implied one arm).
    rows = [
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0, secs=11.0),
        _bench_row(_BENCH_ROUNDS[0], "s5", "signed", 0, secs=22.0),
    ]
    result = gpu_check._append_benchmarks_rows(Path("/nonexistent-ledger"), [], _BENCH_ROUNDS)
    sidecar = gpu_check._build_benchmarks_sidecar({"torch": "2.8.0"}, "s5", rows, result)
    assert sidecar["per_variant_seed_wall_secs"] == {
        "log": {"0": 11.0},
        "signed": {"0": 22.0},
    }
    assert sidecar["task"] == "s5"
    assert sidecar["rows_extracted"] == 2


# ------------------------------------- sidecar merge (Task 6 hardening)
# `--shards` calls `_finish_benchmarks` once per shard; each shard's own
# sidecar write must MERGE onto a prior shard's already-written sidecar
# rather than clobber it, or a multi-shard round's on-disk sidecar would
# only ever reflect the last-finishing shard's rows/timings.


def test_read_benchmarks_sidecar_missing_file_returns_none(tmp_path):
    assert gpu_check._read_benchmarks_sidecar(tmp_path / "nope.json") is None


def test_read_benchmarks_sidecar_malformed_json_returns_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    assert gpu_check._read_benchmarks_sidecar(path) is None


def test_read_benchmarks_sidecar_non_object_json_returns_none(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]")
    assert gpu_check._read_benchmarks_sidecar(path) is None


def test_read_benchmarks_sidecar_well_formed_parses(tmp_path):
    path = tmp_path / "ok.json"
    path.write_text('{"task": "s5", "rows_extracted": 2}')
    assert gpu_check._read_benchmarks_sidecar(path) == {"task": "s5", "rows_extracted": 2}


def test_merge_benchmarks_sidecar_no_existing_returns_fresh_unchanged():
    fresh = gpu_check._build_benchmarks_sidecar(
        {"torch": "2.8.0"}, "s5", [_bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)], _APPEND_ONE
    )
    assert gpu_check._merge_benchmarks_sidecar(None, fresh) == fresh


def test_merge_benchmarks_sidecar_unions_disjoint_seeds_same_variant():
    existing = gpu_check._build_benchmarks_sidecar(
        {"torch": "2.8.0"},
        "s5",
        [_bench_row(_BENCH_ROUNDS[0], "s5", "log", 0, secs=10.0)],
        _APPEND_ONE,
    )
    fresh = gpu_check._build_benchmarks_sidecar(
        {"torch": "2.8.0"},
        "s5",
        [_bench_row(_BENCH_ROUNDS[0], "s5", "log", 1, secs=20.0)],
        _APPEND_ONE,
    )
    merged = gpu_check._merge_benchmarks_sidecar(existing, fresh)
    assert merged["per_variant_seed_wall_secs"] == {"log": {"0": 10.0, "1": 20.0}}
    # Reconciliation counts sum, not just the timing map -- an unsummed
    # count would silently under-report how many rows the merged sidecar
    # actually covers.
    assert merged["rows_extracted"] == 2
    assert merged["rows_appended"] == 2


def test_merge_benchmarks_sidecar_unions_across_different_variants():
    existing = gpu_check._build_benchmarks_sidecar(
        {"torch": "2.8.0"},
        "s5",
        [_bench_row(_BENCH_ROUNDS[0], "s5", "log", 0, secs=10.0)],
        _APPEND_ONE,
    )
    fresh = gpu_check._build_benchmarks_sidecar(
        {"torch": "2.8.0"},
        "s5",
        [_bench_row(_BENCH_ROUNDS[0], "s5", "signed", 0, secs=30.0)],
        _APPEND_ONE,
    )
    merged = gpu_check._merge_benchmarks_sidecar(existing, fresh)
    assert merged["per_variant_seed_wall_secs"] == {
        "log": {"0": 10.0},
        "signed": {"0": 30.0},
    }


def test_merge_benchmarks_sidecar_malformed_existing_count_treated_as_zero():
    fresh = gpu_check._build_benchmarks_sidecar(
        {"torch": "2.8.0"}, "s5", [_bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)], _APPEND_ONE
    )
    existing = {"per_variant_seed_wall_secs": {}, "rows_extracted": "not-an-int"}
    merged = gpu_check._merge_benchmarks_sidecar(existing, fresh)
    assert merged["rows_extracted"] == fresh["rows_extracted"]


def test_merge_benchmarks_sidecar_malformed_existing_timing_map_degrades_not_crashes():
    # `existing["per_variant_seed_wall_secs"]` isn't a dict (hand-mangled
    # prior sidecar, or a future schema change) -- must degrade to "nothing
    # to merge" rather than raising AttributeError on `.items()`.
    fresh = gpu_check._build_benchmarks_sidecar(
        {"torch": "2.8.0"},
        "s5",
        [_bench_row(_BENCH_ROUNDS[0], "s5", "log", 0, secs=10.0)],
        _APPEND_ONE,
    )
    existing = {"per_variant_seed_wall_secs": "not-a-dict", "rows_extracted": 5}
    merged = gpu_check._merge_benchmarks_sidecar(existing, fresh)
    assert merged["per_variant_seed_wall_secs"] == {"log": {"0": 10.0}}
    assert merged["rows_extracted"] == 5 + fresh["rows_extracted"]


class _FakeBenchJob:
    def __init__(self, logs: str) -> None:
        self.logs = logs


def _patch_benchmarks_paths(monkeypatch, tmp_path):
    ledger = tmp_path / "lab_results.jsonl"
    out_dir = tmp_path / "bench"
    monkeypatch.setattr(gpu_check, "_LEDGER_PATH", ledger)
    monkeypatch.setattr(gpu_check, "_DELTA_PROBE_OUT_DIR", out_dir)
    return ledger, out_dir


def test_finish_benchmarks_absent_rows_is_clear_error_and_writes_nothing(tmp_path, monkeypatch):
    ledger, out_dir = _patch_benchmarks_paths(monkeypatch, tmp_path)
    job = _FakeBenchJob(logs="no marked lines in this log at all\n")
    rc = gpu_check._finish_benchmarks(job, ok=True)
    assert rc == 1
    assert not ledger.exists()
    assert not out_dir.exists()


def test_finish_benchmarks_absent_env_line_is_clear_error_and_writes_nothing(tmp_path, monkeypatch):
    ledger, out_dir = _patch_benchmarks_paths(monkeypatch, tmp_path)
    row = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)
    logs = f"{_BENCH_ROW_PREFIX}{json.dumps(row)}\n"  # no MINGRU_LAB_ENV line
    job = _FakeBenchJob(logs=logs)
    rc = gpu_check._finish_benchmarks(job, ok=True)
    assert rc == 1
    assert not ledger.exists()
    assert not out_dir.exists()


def test_finish_benchmarks_writes_one_sidecar_per_task_and_appends_rows(tmp_path, monkeypatch):
    ledger, out_dir = _patch_benchmarks_paths(monkeypatch, tmp_path)
    rows = [
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0),
        _bench_row(_BENCH_ROUNDS[0], "s5", "signed", 0),
        _bench_row(_BENCH_ROUNDS[1], "mqar", "log", 0, secs=5.0),
    ]
    env = {"torch": "2.8.0", "cuda_device_name": "NVIDIA L4"}
    logs = "".join(f"{_BENCH_ROW_PREFIX}{json.dumps(r)}\n" for r in rows)
    logs += f"{_BENCH_ENV_PREFIX}{json.dumps(env)}\n"
    job = _FakeBenchJob(logs=logs)

    rc = gpu_check._finish_benchmarks(job, ok=True)

    assert rc == 0
    # Rows are appended per-task bucket (spec section 4: "merges are
    # order-independent" across tasks), so the ledger's cross-task row
    # order isn't the original log order -- only membership and each
    # task's own relative row order matter here.
    written = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert {json.dumps(r, sort_keys=True) for r in written} == {
        json.dumps(r, sort_keys=True) for r in rows
    }

    s5_sidecar = json.loads((out_dir / "bench_s5_env.json").read_text())
    assert s5_sidecar["env"] == env
    assert s5_sidecar["task"] == "s5"
    assert s5_sidecar["rows_extracted"] == 2
    assert s5_sidecar["rows_appended"] == 2
    assert s5_sidecar["per_variant_seed_wall_secs"]["log"]["0"] == rows[0]["secs"]
    assert s5_sidecar["per_variant_seed_wall_secs"]["signed"]["0"] == rows[1]["secs"]

    mqar_sidecar = json.loads((out_dir / "bench_mqar_env.json").read_text())
    assert mqar_sidecar["task"] == "mqar"
    assert mqar_sidecar["rows_extracted"] == 1
    assert mqar_sidecar["rows_appended"] == 1
    assert mqar_sidecar["per_variant_seed_wall_secs"]["log"]["0"] == 5.0

    assert not (out_dir / "bench_psmnist_env.json").exists()
    assert not (out_dir / "bench_pendulum_env.json").exists()


def test_finish_benchmarks_row_missing_task_is_warned_and_dropped_not_a_crash(
    tmp_path, monkeypatch, capsys
):
    ledger, out_dir = _patch_benchmarks_paths(monkeypatch, tmp_path)
    good = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)
    no_task = {"round": _BENCH_ROUNDS[0], "variant": "log", "seed": 1, "secs": 1.0}
    logs = (
        f"{_BENCH_ROW_PREFIX}{json.dumps(good)}\n"
        f"{_BENCH_ROW_PREFIX}{json.dumps(no_task)}\n"
        f"{_BENCH_ENV_PREFIX}{json.dumps({'torch': '2.8.0'})}\n"
    )
    job = _FakeBenchJob(logs=logs)

    rc = gpu_check._finish_benchmarks(job, ok=True)

    assert rc == 0
    written = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert written == [good]
    assert "no attributable" in capsys.readouterr().err
    assert (out_dir / "bench_s5_env.json").exists()


def test_finish_benchmarks_non_dict_row_is_warned_and_dropped_not_a_crash(
    tmp_path, monkeypatch, capsys
):
    # Quality review REQUIRED FIX 1/2: a non-dict MINGRU_LAB_ROW payload
    # (e.g. `[1, 2, 3]` -- exactly the case _extract_all's own docstring
    # calls out as expected, since that extractor only guards
    # PARSEABILITY, not SHAPE) must not crash the per-task bucketing loop
    # with AttributeError. It's warned about and dropped, exactly like the
    # missing-task-field case above (a different shape -- that one IS a
    # dict, just without a usable task field) -- valid rows still append
    # and their task's sidecar still writes.
    ledger, out_dir = _patch_benchmarks_paths(monkeypatch, tmp_path)
    good = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)
    logs = (
        f"{_BENCH_ROW_PREFIX}[1, 2, 3]\n"
        f"{_BENCH_ROW_PREFIX}{json.dumps(good)}\n"
        f"{_BENCH_ENV_PREFIX}{json.dumps({'torch': '2.8.0'})}\n"
    )
    job = _FakeBenchJob(logs=logs)

    rc = gpu_check._finish_benchmarks(job, ok=True)

    assert rc == 0
    written = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert written == [good]
    assert "no attributable" in capsys.readouterr().err
    sidecar = json.loads((out_dir / "bench_s5_env.json").read_text())
    assert sidecar["rows_extracted"] == 1
    assert sidecar["rows_appended"] == 1


def test_finish_benchmarks_retry_is_idempotent_end_to_end(tmp_path, monkeypatch):
    ledger, out_dir = _patch_benchmarks_paths(monkeypatch, tmp_path)
    rows = [
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0),
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 1),
    ]
    env = {"torch": "2.8.0"}
    logs = "".join(f"{_BENCH_ROW_PREFIX}{json.dumps(r)}\n" for r in rows)
    logs += f"{_BENCH_ENV_PREFIX}{json.dumps(env)}\n"
    job = _FakeBenchJob(logs=logs)

    first_rc = gpu_check._finish_benchmarks(job, ok=True)
    second_rc = gpu_check._finish_benchmarks(job, ok=True)

    assert (first_rc, second_rc) == (0, 0)
    assert len(ledger.read_text().splitlines()) == 2
    sidecar = json.loads((out_dir / "bench_s5_env.json").read_text())
    assert sidecar["rows_appended"] == 0
    assert sidecar["rows_extracted"] == 2


def test_finish_benchmarks_default_clobbers_sidecar_not_merges(tmp_path, monkeypatch):
    # Task 6 hardening's explicit invariant: the single-job path
    # (`merge_sidecar` defaults to `False`) must stay byte-identical to
    # before `--shards` existed -- a second call's sidecar wholly replaces
    # the first's, it does not accumulate onto it.
    ledger, out_dir = _patch_benchmarks_paths(monkeypatch, tmp_path)
    env = {"torch": "2.8.0"}

    first_row = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0, secs=10.0)
    logs_first = (
        f"{_BENCH_ROW_PREFIX}{json.dumps(first_row)}\n{_BENCH_ENV_PREFIX}{json.dumps(env)}\n"
    )
    gpu_check._finish_benchmarks(_FakeBenchJob(logs=logs_first), ok=True)

    second_row = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 1, secs=20.0)
    logs_second = (
        f"{_BENCH_ROW_PREFIX}{json.dumps(second_row)}\n{_BENCH_ENV_PREFIX}{json.dumps(env)}\n"
    )
    gpu_check._finish_benchmarks(_FakeBenchJob(logs=logs_second), ok=True)

    sidecar = json.loads((out_dir / "bench_s5_env.json").read_text())
    assert sidecar["per_variant_seed_wall_secs"]["log"] == {"1": 20.0}
    assert sidecar["rows_extracted"] == 1


def test_finish_benchmarks_merge_sidecar_unions_disjoint_shard_seed_timings(tmp_path, monkeypatch):
    # Two sequential finish calls (mirroring two shards of a --shards run,
    # each covering a disjoint seed slice) with merge_sidecar=True must
    # leave the union of both shards' timings on disk, not just the last
    # shard's.
    ledger, out_dir = _patch_benchmarks_paths(monkeypatch, tmp_path)
    env = {"torch": "2.8.0"}

    shard0_rows = [
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0, secs=10.0),
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 1, secs=11.0),
    ]
    logs0 = "".join(f"{_BENCH_ROW_PREFIX}{json.dumps(r)}\n" for r in shard0_rows)
    logs0 += f"{_BENCH_ENV_PREFIX}{json.dumps(env)}\n"
    rc0 = gpu_check._finish_benchmarks(_FakeBenchJob(logs=logs0), ok=True, merge_sidecar=True)

    shard1_rows = [
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 2, secs=12.0),
        _bench_row(_BENCH_ROUNDS[0], "s5", "log", 3, secs=13.0),
    ]
    logs1 = "".join(f"{_BENCH_ROW_PREFIX}{json.dumps(r)}\n" for r in shard1_rows)
    logs1 += f"{_BENCH_ENV_PREFIX}{json.dumps(env)}\n"
    rc1 = gpu_check._finish_benchmarks(_FakeBenchJob(logs=logs1), ok=True, merge_sidecar=True)

    assert (rc0, rc1) == (0, 0)
    sidecar = json.loads((out_dir / "bench_s5_env.json").read_text())
    assert sidecar["per_variant_seed_wall_secs"]["log"] == {
        "0": 10.0,
        "1": 11.0,
        "2": 12.0,
        "3": 13.0,
    }
    # Reconciliation counts sum across shards -- an unmerged count field
    # would silently understate the round's true row coverage.
    assert sidecar["rows_extracted"] == 4
    assert sidecar["rows_appended"] == 4
    # Both shards' rows land in the ledger; merging the sidecar never
    # touches ledger append semantics (already dedup-idempotent).
    assert len(ledger.read_text().splitlines()) == 4


def test_finish_benchmarks_nonzero_exit_when_job_not_ok_despite_valid_rows(tmp_path, monkeypatch):
    ledger, out_dir = _patch_benchmarks_paths(monkeypatch, tmp_path)
    row = _bench_row(_BENCH_ROUNDS[0], "s5", "log", 0)
    logs = (
        f"{_BENCH_ROW_PREFIX}{json.dumps(row)}\n"
        f"{_BENCH_ENV_PREFIX}{json.dumps({'torch': '2.8.0'})}\n"
    )
    job = _FakeBenchJob(logs=logs)

    rc = gpu_check._finish_benchmarks(job, ok=False)

    assert rc == 1
    assert ledger.exists()


# --------------------------------------- pandas preamble (Task 6, churn)
# The churn task's `RosbankLoader` lazy-imports pandas inside the job
# (mirroring the torchvision/psMNIST precedent) -- `build_benchmarks_command`
# must install it whenever churn is among the job's tasks, and must NOT
# install it for a job that never touches churn.


def test_build_benchmarks_command_installs_pandas_when_tasks_is_none():
    # tasks=None means "every task" (the campaign script's own default),
    # which includes churn -- pandas must be installed.
    command = build_benchmarks_command("git@example.com/repo.git", "deadbeef", None, None, None)
    assert "pip install --no-cache-dir pandas" in command


def test_build_benchmarks_command_installs_pandas_when_churn_named():
    command = build_benchmarks_command(
        "git@example.com/repo.git", "deadbeef", ["s5", "churn"], None, None
    )
    assert "pip install --no-cache-dir pandas" in command
    # Must precede the campaign invocation it's meant to cover.
    assert command.index("pip install --no-cache-dir pandas") < command.index(
        "python scripts/gpu_benchmark_campaign.py"
    )


def test_build_benchmarks_command_omits_pandas_when_churn_not_requested():
    command = build_benchmarks_command(
        "git@example.com/repo.git", "deadbeef", ["s5", "mqar"], None, None
    )
    assert "pandas" not in command


def test_build_benchmarks_command_still_installs_torchvision_alongside_pandas():
    # The pre-existing torchvision install (psMNIST) must survive unchanged
    # when pandas is also installed.
    command = build_benchmarks_command("git@example.com/repo.git", "deadbeef", None, None, None)
    assert "pip install --no-cache-dir torchvision" in command
    assert command.index("pip install --no-cache-dir torchvision") < command.index(
        "python scripts/gpu_benchmark_campaign.py"
    )


def test_build_benchmarks_command_preserves_mingru_scan_triton_with_pandas():
    command = build_benchmarks_command(
        "git@example.com/repo.git", "deadbeef", ["churn"], None, None
    )
    assert "export MINGRU_SCAN=triton" in command
    assert command.index("pip install --no-cache-dir pandas") < command.index(
        "export MINGRU_SCAN=triton"
    )


def test_build_benchmarks_command_with_pandas_still_single_foreground_chain():
    command = build_benchmarks_command(
        "git@example.com/repo.git", "deadbeef", ["churn"], None, None
    )
    assert "&" not in command.replace("&&", "")
    assert "set -eux" in command


# --------------------------------------------- churn round tags (Task 6)
# `_BENCHMARKS_ROUNDS` must accept both churn's pilot tag (`bench-churn-01`,
# hardcoded -- see `_BENCHMARKS_ROUNDS_PILOT`'s comment) and its current
# matrix tag (`bench-churn-02`, read live from `BENCH_ROUND_TAGS`).


def test_benchmarks_rounds_accepts_churn_pilot_tag():
    assert "bench-churn-01" in _BENCH_ROUNDS


def test_benchmarks_rounds_accepts_churn_matrix_tag():
    from experiments.benchmark_tasks import BENCH_ROUND_TAGS

    assert BENCH_ROUND_TAGS["churn"] == "bench-churn-02"
    assert BENCH_ROUND_TAGS["churn"] in _BENCH_ROUNDS


def test_valid_benchmarks_key_accepts_a_churn_pilot_row():
    row = _bench_row("bench-churn-01", "churn", "signed", 0)
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) == (
        "bench-churn-01",
        "churn",
        "signed",
        0,
    )


def test_valid_benchmarks_key_accepts_a_churn_matrix_row():
    row = _bench_row("bench-churn-02", "churn", "log", 5)
    assert gpu_check._valid_benchmarks_key(row, _BENCH_ROUNDS) == (
        "bench-churn-02",
        "churn",
        "log",
        5,
    )


# --------------------------------------------- --shards seed pool (Task 6)
# `_resolve_shard_seed_pool` mirrors `gpu_benchmark_campaign.py`'s own
# `_resolve_seeds` seed-source convention: explicit `--seeds` wins outright,
# otherwise every selected task's own `TaskSpec.seeds` count must agree.

_resolve_shard_seed_pool = gpu_check._resolve_shard_seed_pool
_shard_seed_lists = gpu_check._shard_seed_lists


def test_resolve_shard_seed_pool_explicit_seeds_returned_as_is():
    assert _resolve_shard_seed_pool(["s5"], [3, 7, 11]) == [3, 7, 11]


def test_resolve_shard_seed_pool_explicit_seeds_ignore_task_seed_counts():
    # An explicit --seeds list applies as given even if it doesn't match
    # any task's own seed-matrix size -- sharding partitions whatever seeds
    # were named, mirroring the campaign script's own override semantics.
    assert _resolve_shard_seed_pool(["psmnist"], [0, 1]) == [0, 1]


def test_resolve_shard_seed_pool_single_task_uses_its_seed_count():
    assert _resolve_shard_seed_pool(["churn"], None) == list(range(36))
    assert _resolve_shard_seed_pool(["psmnist"], None) == list(range(12))


def test_resolve_shard_seed_pool_multiple_tasks_same_count_agree():
    assert _resolve_shard_seed_pool(["s5", "mqar", "pendulum", "churn"], None) == list(range(36))


def test_resolve_shard_seed_pool_mismatched_task_counts_raises():
    # psmnist (12 seeds) mixed with churn (36 seeds), no explicit --seeds to
    # disambiguate -- must raise rather than silently sharding by one task's
    # count and running the wrong seeds for the other.
    with pytest.raises(ValueError):
        _resolve_shard_seed_pool(["psmnist", "churn"], None)


def test_resolve_shard_seed_pool_none_tasks_means_every_registered_task():
    # tasks=None mixes psmnist (12) with the four 36-seed tasks -- mismatch.
    with pytest.raises(ValueError):
        _resolve_shard_seed_pool(None, None)


def test_resolve_shard_seed_pool_unknown_task_raises_value_error_not_key_error():
    # `TASKS[name]` is unguarded internally -- an unrecognized --tasks name
    # must degrade to the module's standard ValueError + "error: ..." exit
    # 2 pattern (main() only catches ValueError around this call), never an
    # uncaught bare KeyError traceback.
    with pytest.raises(ValueError):
        _resolve_shard_seed_pool(["bogus-task"], None)


def test_resolve_shard_seed_pool_unknown_task_mixed_with_known_raises():
    with pytest.raises(ValueError):
        _resolve_shard_seed_pool(["s5", "bogus-task"], None)


# --------------------------------------------- --shards slicing (Task 6)


def test_shard_seed_lists_36_into_6_gives_six_of_six():
    shards = _shard_seed_lists(list(range(36)), 6)
    assert len(shards) == 6
    assert all(len(s) == 6 for s in shards)


def test_shard_seed_lists_36_into_6_is_disjoint_and_exhaustive():
    shards = _shard_seed_lists(list(range(36)), 6)
    flat = [seed for shard in shards for seed in shard]
    assert sorted(flat) == list(range(36))
    assert len(flat) == len(set(flat))  # disjoint


def test_shard_seed_lists_job_k_gets_seeds_6k_through_6k_plus_5():
    # Spec: "job k of 6 runs seeds 6k..6k+5".
    shards = _shard_seed_lists(list(range(36)), 6)
    for k, shard in enumerate(shards):
        assert shard == list(range(6 * k, 6 * k + 6))


def test_shard_seed_lists_single_shard_returns_the_whole_pool():
    assert _shard_seed_lists([0, 1, 2], 1) == [[0, 1, 2]]


def test_shard_seed_lists_uneven_split_rejected_loudly():
    with pytest.raises(ValueError):
        _shard_seed_lists(list(range(36)), 5)


def test_shard_seed_lists_zero_shards_rejected_loudly():
    with pytest.raises(ValueError):
        _shard_seed_lists(list(range(36)), 0)


def test_shard_seed_lists_negative_shards_rejected_loudly():
    with pytest.raises(ValueError):
        _shard_seed_lists(list(range(36)), -1)


def test_shard_seed_lists_more_shards_than_seeds_uneven_rejected_loudly():
    with pytest.raises(ValueError):
        _shard_seed_lists([0, 1, 2], 5)


# ------------------------------------- _run_sharded_benchmarks submission
# (Task 6 hardening) -- KeyboardInterrupt during the multi-submission
# window (before any job.wait()) must stop every already-submitted shard,
# not just leave them running, mirroring the wait-loop's existing
# "avoid orphaned billing" discipline. `_submit_job` is monkeypatched
# directly (no real lightning_sdk call) so this is a pure control-flow
# test of `_run_sharded_benchmarks` itself.


class _FakeSubmittedJob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_run_sharded_benchmarks_interrupt_during_submission_stops_already_submitted(
    monkeypatch,
):
    submitted: list[_FakeSubmittedJob] = []

    def fake_submit_job(args, machine, owner, ts_name, name, command):
        if name.endswith("shard2of4"):
            raise KeyboardInterrupt
        job = _FakeSubmittedJob(name)
        submitted.append(job)
        return job

    monkeypatch.setattr(gpu_check, "_submit_job", fake_submit_job)

    with pytest.raises(KeyboardInterrupt):
        gpu_check._run_sharded_benchmarks(
            args=None,
            machine=None,
            owner=None,
            ts_name="ts",
            commands=["c0", "c1", "c2", "c3"],
            job_name_prefix="prefix",
            ref="deadbeef",
        )

    # Shards 0 and 1 submitted successfully before shard 2's interrupt;
    # both must be stopped rather than left running/billing.
    assert len(submitted) == 2
    assert all(job.stopped for job in submitted)


class _FakeWaitableJob(_FakeSubmittedJob):
    status = "completed"

    def wait(self) -> None:
        pass


def test_run_sharded_benchmarks_no_interrupt_leaves_submitted_jobs_unstopped(monkeypatch):
    # Sanity converse of the interrupt test above: normal (non-interrupted)
    # submission must NOT call .stop() on anything during the submission
    # loop itself -- only an interrupt triggers the stop-everything path.
    submitted: list[_FakeWaitableJob] = []

    def fake_submit_job(args, machine, owner, ts_name, name, command):
        job = _FakeWaitableJob(name)
        submitted.append(job)
        return job

    monkeypatch.setattr(gpu_check, "_submit_job", fake_submit_job)
    monkeypatch.setattr(gpu_check, "_finish_benchmarks", lambda job, ok, merge_sidecar=False: 0)

    rc = gpu_check._run_sharded_benchmarks(
        args=None,
        machine=None,
        owner=None,
        ts_name="ts",
        commands=["c0", "c1"],
        job_name_prefix="prefix",
        ref="deadbeef",
    )

    assert rc == 0
    assert len(submitted) == 2
    assert not any(job.stopped for job in submitted)
