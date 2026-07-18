"""Subprocess-isolated mechanism x state scaling probe (Task 3).

Spec: ``.claude/output/specs/2026-07-18-matched-state-round-design.md``
section 4 ("A scaling-probe script measures, per (mechanism, state-size)
config in its own subprocess: median per-step forward+backward time ...
and peak RSS"), section 6 (probe artifact contract and grid), section 7
(invariants), and acceptance criterion 5. Intent ledger statement 4: a
mechanism x state probe at state in {64, 256, 1024, 4096} with per-step
fwd+bwd time and peak memory, INCLUDING where ``GivensMinGRU`` hits the
wall -- this script's artifact is the ONLY permitted source for the
round's verdict-table cost/memory columns. A timeout/OOM at givens@4096
is the finding, not a failure.

Grid (spec section 6, exact):

``delta`` (packaged ``DeltaMinGRU``, ``input_size=hidden_size=64``,
``n_heads=4``, ``nh=2``, ``chunk_size`` default): ``d_k=d_v`` in
``{4, 8, 16, 32}`` -> ``state_elements = n_heads * d_k * d_v`` in
``{64, 256, 1024, 4096}``; plus the two training-arm configs from the
round's new hetero-lab arms, each annotated ``training_arm: true`` in the
artifact -- ``n_heads=1, d_k=d_v=8`` (mirrors ``hetero-pd64``, state 64)
and ``n_heads=4, d_k=d_v=16`` (mirrors ``hetero-pd1024``, state 1024; this
duplicates the sweep's ``d_k=16`` point in model shape but is recorded as
its own grid entry per the spec's "plus the two training-arm configs"
wording, so a reader can find the exact training-arm row without having
to infer which sweep point it corresponds to).

``givens`` (packaged ``GivensMinGRU(64, d_h)``, ``block_size=8,
rounds=3``): ``d_h`` in ``{64, 256, 1024, 4096}`` -> ``state_elements =
d_h`` (state is a ``d_h``-dim vector, ``n_blocks = d_h / block_size``
blocks of size ``block_size``).

Ten configs total. Every config runs B=128, T=64, >=1 warmup step and
>=5 timed forward+backward steps (median of the timed steps reported,
plus the full list for transparency), in its OWN child subprocess (this
process never imports torch itself -- see the "no torch in the parent"
note below).

Parent/child protocol
----------------------
The parent (``run()``/``main()`` with no ``--child`` flag) never imports
torch: it only needs the config table below (pure Python/argparse), so a
crashed or hung child can never take the orchestrator down with it, and
the parent can enforce a wall-clock timeout on each child from outside
the process that might hang.

For each grid config (filtered by ``--only``, see below), the parent
spawns ``[sys.executable, __file__, "--child", <label>, ...]`` via
``subprocess.Popen`` and waits on ``communicate(timeout=...)``. Three
outcomes:

``ok``
    The child printed a ``MINGRU_PROBE_RESULT <json>`` line and exited 0.
    Its self-report becomes the artifact row verbatim.

``timeout``
    ``communicate()`` raised ``TimeoutExpired``. Per the documented
    ``Popen`` pattern, the parent kills the child and calls
    ``communicate()`` a second time (no timeout) to drain whatever
    output had already been buffered -- this recovers the child's
    ``MINGRU_PROBE_HEADER`` line (mechanism/config/state_elements/params,
    printed immediately after model construction, before the timed loop)
    even though the child never got to report a result. The row is
    written with ``status="timeout"`` and null timing/RSS fields (the
    child was killed before it could self-report ``ru_maxrss``). A
    ``SIGKILL`` landing mid-``print()`` can leave a truncated/malformed
    header line in the drained output (or no header line at all, if
    killed before the first byte of it flushed) -- exactly the
    givens@4096 wall case this probe exists to find. ``_extract_last``
    never raises on a malformed line (a parse failure degrades to "no
    match", never a crash -- see its own docstring), and this specific
    no-recoverable-header-under-timeout case is NOT treated as a
    probe/grid bug: it's an expected consequence of killing a hung
    child at an arbitrary instant, so the row falls back to the static
    grid table (``label``/``mechanism``/``config``/``state_elements``,
    ``params`` unknown) with ``status="timeout"`` and the grid loop
    continues -- one config timing out must never discard every other
    config's already-collected row or block the artifact write.

``oom``
    The child caught a ``RuntimeError``/``MemoryError`` during its
    warmup/timed loop (see ``_run_child``) and self-reported
    ``status="oom"`` before exiting 0 -- OR the child crashed some other
    way (nonzero exit, no ``MINGRU_PROBE_RESULT`` line) after having
    already printed its header, in a run that did NOT time out. This
    narrow probe only ever builds two well-tested packaged mixers and
    runs their existing forward/backward; at these deliberately
    memory-stressing shapes (state up to 4096, B=128, T=64) an
    unexpected crash after successful construction is overwhelmingly a
    resource wall, not a new bug in this script or the mixers -- so it
    is conservatively recorded as ``oom`` rather than a 4th status value
    outside the artifact's ``{ok, timeout, oom}`` contract. That
    conservatism is made auditable, not silent: the row also carries an
    ``oom_message_recognized`` bool (``True`` only when the caught
    exception's message matched a known allocator-failure pattern --
    see ``_OOM_MESSAGE_MARKERS``) and the exception/stderr text itself
    in ``detail``, and both surface in the rendered ``.md`` table so a
    reader auditing the round's sole cost/memory source can spot an
    ``oom`` row whose message did NOT look like a real allocation
    failure (e.g. a numerics/shape bug at an extreme state size) rather
    than trusting the status label blindly. If a config crashes before
    ever printing its header in a run that did NOT time out (i.e.
    construction itself failed) that is NOT a resource wall -- it means
    this script's own grid entry is broken -- and ``run()`` aborts the
    whole probe loudly (nonzero exit, no artifact written) rather than
    fabricate a misleading row. (A missing/malformed header IS possible
    under a timeout, but that case is handled entirely within
    ``timeout`` above and never reaches this abort path.)

Evidence-pin requirement: mirrors ``scripts/bench_delta.py`` exactly --
every measured number in this artifact must come from a child running
under ``torch==2.5.1`` (this repository's evidence-pin environment).
The child hard-asserts ``torch.__version__ == "2.5.1"`` before
constructing any model and refuses to run otherwise; the parent also
surfaces the child's version string in the env block. Run via, e.g.,
``uv run --no-project --with torch==2.5.1 python scripts/scaling_probe.py``.

Import-path note: like ``scripts/bench_delta.py``, this script must
import ``mingru`` even when the project isn't pip/uv-installed under the
evidence-pin invocation (whose ``torch==2.5.1`` pin conflicts with this
package's declared ``torch>=2.8`` floor). It inserts ``src/`` onto
``sys.path`` ahead of site-packages, derived from ``__file__`` -- but
only in the child process (``_run_child``), immediately before the
``mingru`` import, since the parent never imports torch or ``mingru`` at
all. Unlike ``scripts/bench_delta.py``'s module-level bootstrap, this
bootstrap and the ``torch``/``mingru`` imports it precedes are entirely
inside ``_run_child``'s function body -- function-scoped imports are
never flagged by ruff's import-order check (E402 only applies to
module-level statements), so this file needs no per-file lint exemption
for it.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "experiments" / "bench"

_EVIDENCE_PIN_TORCH_VERSION = "2.5.1"

_DEFAULT_BATCH_SIZE = 128
_DEFAULT_SEQ_LEN = 64
_DEFAULT_WARMUP = 1
_DEFAULT_STEPS = 5
_DEFAULT_TIMEOUT_SECS = 600.0
_SEED = 0

# DeltaMinGRU's own default (spec section 6: "chunk_size left at its
# default (64)") -- stated explicitly in each delta config's build_kwargs
# below rather than left implicit, so a reader of scaling_frontier.json
# doesn't need to know the packaged class's default to see the exact
# config that was measured.
_DELTA_CHUNK_SIZE = 64

_HEADER_PREFIX = "MINGRU_PROBE_HEADER "
_RESULT_PREFIX = "MINGRU_PROBE_RESULT "

# Substrings (checked case-insensitively) that identify a caught
# RuntimeError/MemoryError during the warmup/timed loop as a memory
# allocation failure rather than some other kind of exception. Matches
# both PyTorch's own CPU allocator message ("DefaultCPUAllocator: not
# enough memory") and the C++-level std::bad_alloc PyTorch sometimes
# surfaces as a bare RuntimeError.
_OOM_MESSAGE_MARKERS = ("memory", "alloc")


@dataclass(frozen=True)
class _ProbeConfig:
    """One grid entry: enough to build the model AND to label/filter it.

    ``state_elements`` is *not* independently measured here -- it is the
    value the config's ``build_kwargs`` are chosen to hit (n_heads * d_k *
    d_v for delta, hidden_size for givens), stated for labeling/filtering
    and as a fallback for a row the child never got to self-report (see
    the "timeout"/no-header case in the module docstring). Every ``ok``
    or ``oom`` row's ``state_elements`` in the artifact is instead read
    live off the constructed module by the child (``_run_child``'s
    ``live_state_elements``), so this field can never silently drift
    from what was actually built.
    """

    label: str
    mechanism: str  # "delta" | "givens"
    build_kwargs: dict[str, Any]
    state_elements: int
    training_arm: bool = False


def _delta_configs() -> tuple[_ProbeConfig, ...]:
    sweep = tuple(
        _ProbeConfig(
            label=f"delta_nh4_dk{d}",
            mechanism="delta",
            build_kwargs={
                "input_size": 64,
                "hidden_size": 64,
                "n_heads": 4,
                "nh": 2,
                "d_k": d,
                "d_v": d,
                "chunk_size": _DELTA_CHUNK_SIZE,
            },
            state_elements=4 * d * d,
        )
        for d in (4, 8, 16, 32)
    )
    training_arms = (
        _ProbeConfig(
            label="delta_train_pd64",
            mechanism="delta",
            build_kwargs={
                "input_size": 64,
                "hidden_size": 64,
                "n_heads": 1,
                "nh": 2,
                "d_k": 8,
                "d_v": 8,
                "chunk_size": _DELTA_CHUNK_SIZE,
            },
            state_elements=1 * 8 * 8,
            training_arm=True,
        ),
        _ProbeConfig(
            label="delta_train_pd1024",
            mechanism="delta",
            build_kwargs={
                "input_size": 64,
                "hidden_size": 64,
                "n_heads": 4,
                "nh": 2,
                "d_k": 16,
                "d_v": 16,
                "chunk_size": _DELTA_CHUNK_SIZE,
            },
            state_elements=4 * 16 * 16,
            training_arm=True,
        ),
    )
    return sweep + training_arms


def _givens_configs() -> tuple[_ProbeConfig, ...]:
    return tuple(
        _ProbeConfig(
            label=f"givens_dh{d_h}",
            mechanism="givens",
            build_kwargs={
                "input_size": 64,
                "hidden_size": d_h,
                "block_size": 8,
                "rounds": 3,
            },
            state_elements=d_h,
        )
        for d_h in (64, 256, 1024, 4096)
    )


_GRID: tuple[_ProbeConfig, ...] = _delta_configs() + _givens_configs()
_GRID_BY_LABEL: dict[str, _ProbeConfig] = {c.label: c for c in _GRID}


def _select_grid(only: list[str] | None) -> list[_ProbeConfig]:
    """Filter ``_GRID`` by ``--only`` tokens.

    Each token is either an exact ``label`` (e.g. ``delta_nh4_dk16``) or
    ``mechanism:state_elements`` (e.g. ``delta:64``). The latter form can
    match more than one grid entry -- two delta configs share
    ``state_elements=64`` (the ``d_k=4`` sweep point and the
    ``delta_train_pd64`` training arm) and likewise at 1024 -- which is
    intentional: a reduced smoke grid should not depend on being able to
    disambiguate them, and running both is still cheap.
    """
    if not only:
        return list(_GRID)
    selected: list[_ProbeConfig] = []
    for token in only:
        if token in _GRID_BY_LABEL:
            selected.append(_GRID_BY_LABEL[token])
            continue
        if ":" in token:
            mech, _, state_str = token.partition(":")
            try:
                state = int(state_str)
            except ValueError:
                raise SystemExit(
                    f"--only token {token!r}: expected 'mechanism:state_elements' "
                    "with an integer state, or an exact config label"
                ) from None
            matches = [c for c in _GRID if c.mechanism == mech and c.state_elements == state]
            if not matches:
                raise SystemExit(f"--only token {token!r} matched no grid config")
            selected.extend(matches)
            continue
        raise SystemExit(
            f"--only token {token!r} is not a known config label and has no ':' "
            "(expected 'mechanism:state_elements' or an exact label)"
        )
    # De-duplicate while preserving first-seen order (a token list could
    # overlap, e.g. an explicit label plus a mechanism:state token that
    # also matches it).
    seen: set[str] = set()
    deduped = []
    for cfg in selected:
        if cfg.label not in seen:
            seen.add(cfg.label)
            deduped.append(cfg)
    return deduped


# --- child process ------------------------------------------------------


def _run_child(label: str, batch_size: int, seq_len: int, warmup: int, steps: int) -> int:
    """Build the config's model, time it, and print its self-report.

    Runs entirely inside the child subprocess spawned by ``run()``. See
    the module docstring's "Parent/child protocol" for the two lines
    this prints (``MINGRU_PROBE_HEADER`` right after construction,
    ``MINGRU_PROBE_RESULT`` at the end) and how the parent uses each.
    """
    # See the module docstring's "Import-path note": insert `src/` onto
    # `sys.path` ahead of site-packages so `from mingru import ...` below
    # resolves from the source tree whether or not this project is
    # pip/uv-installed. Must happen before the `mingru` import; both are
    # function-scoped (unlike `scripts/bench_delta.py`'s module-level
    # bootstrap), so ruff's E402 never applies here.
    src_dir = _REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    import resource

    import torch

    from mingru import DeltaMinGRU, GivensMinGRU

    if torch.__version__ != _EVIDENCE_PIN_TORCH_VERSION:
        print(
            f"FAILED: scaling_probe child must run under torch=="
            f"{_EVIDENCE_PIN_TORCH_VERSION} (the repository's evidence-pin "
            f"environment) but found torch=={torch.__version__} -- refusing "
            "to measure. Re-run the parent via, e.g., `uv run --no-project "
            "--with torch==2.5.1 python scripts/scaling_probe.py`.",
            file=sys.stderr,
        )
        return 1

    cfg = _GRID_BY_LABEL.get(label)
    if cfg is None:
        print(f"FAILED: unknown config label {label!r}", file=sys.stderr)
        return 1

    torch.manual_seed(_SEED)
    if cfg.mechanism == "delta":
        layer = DeltaMinGRU(**cfg.build_kwargs)
        live_state_elements = layer.n_heads * layer.d_k * layer.d_v
    elif cfg.mechanism == "givens":
        layer = GivensMinGRU(**cfg.build_kwargs)
        live_state_elements = layer.hidden_size
    else:  # pragma: no cover - grid is fixed at module load time
        print(f"FAILED: unknown mechanism {cfg.mechanism!r}", file=sys.stderr)
        return 1

    params = sum(p.numel() for p in layer.parameters())
    input_size = cfg.build_kwargs["input_size"]
    x = torch.randn(batch_size, seq_len, input_size, requires_grad=True)

    header = {
        "label": cfg.label,
        "mechanism": cfg.mechanism,
        "config": cfg.build_kwargs,
        "training_arm": cfg.training_arm,
        "state_elements": live_state_elements,
        "params": params,
        "num_threads": torch.get_num_threads(),
    }
    print(_HEADER_PREFIX + json.dumps(header), flush=True)

    step_secs: list[float] = []
    status = "ok"
    detail: str | None = None
    # `oom_message_recognized` is the visible flag the module docstring's
    # "oom" outcome promises: `None` while no exception has been caught
    # (not applicable), then set below to whether the caught message
    # actually matched a known allocator-failure pattern. This is what
    # lets a reader of the rendered artifact tell an `oom` row backed by
    # a real allocation failure apart from one backed by some other
    # RuntimeError (e.g. a numerics/shape bug) that this narrow probe
    # still conservatively files under `status="oom"` rather than a 4th
    # status value outside the artifact's {ok, timeout, oom} contract.
    oom_message_recognized: bool | None = None
    try:
        for i in range(warmup + steps):
            layer.zero_grad(set_to_none=True)
            x.grad = None
            start = time.perf_counter()
            loss = layer(x).sum()
            loss.backward()
            elapsed = time.perf_counter() - start
            if i >= warmup:
                step_secs.append(elapsed)
    except (RuntimeError, MemoryError) as exc:
        message = str(exc)
        status = "oom"
        oom_message_recognized = any(marker in message.lower() for marker in _OOM_MESSAGE_MARKERS)
        detail = message

    # `status` here reflects only whether the loop raised (see the except
    # clause above); it does NOT re-derive "ok" from len(step_secs) >= 5.
    # The >=5-timed-steps guarantee for `status="ok"` rows is the caller's
    # responsibility (`run()` defaults `--steps` to 5 and only relaxes that
    # floor, with a printed warning, for an explicit smoke override -- see
    # the module docstring's grid description and the CLI help for
    # `--steps`) -- reclassifying a successful run's status here based on
    # the step count would let a genuine crash and a merely-short smoke
    # run collide into the same signal.
    median = None
    if step_secs:
        sorted_secs = sorted(step_secs)
        mid = len(sorted_secs) // 2
        median = (
            sorted_secs[mid]
            if len(sorted_secs) % 2
            else (sorted_secs[mid - 1] + sorted_secs[mid]) / 2
        )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        **header,
        "step_secs_median": median,
        "step_secs_all": step_secs,
        "peak_rss_bytes": peak_rss,
        "status": status,
        "oom_message_recognized": oom_message_recognized,
        "detail": detail,
    }
    print(_RESULT_PREFIX + json.dumps(result), flush=True)
    return 0


# --- parent orchestration ------------------------------------------------


@dataclass
class _RunOutcome:
    row: dict[str, Any]
    aborted: bool = False
    abort_reason: str | None = None


def _extract_last(prefix: str, text: str) -> dict[str, Any] | None:
    """Parse the JSON payload of the last well-formed ``prefix``-prefixed line.

    ``text`` is a child's captured stdout, which may be a partial capture
    -- most notably when the parent kills a hung child on timeout (see the
    module docstring's "timeout" outcome): a ``SIGKILL`` landing
    mid-``print()`` can leave a truncated, non-JSON tail on the last
    matching line. A malformed matching line is treated as no match (this
    function returns ``None`` if every matching line fails to parse, or
    keeps the last line that DID parse if an earlier one is malformed and
    a later one isn't) -- it must never raise, since a caller-level
    ``JSONDecodeError`` here would propagate out of ``_run_one_config``
    into ``run()``'s grid loop and abort the whole probe before the
    artifact is written, discarding every already-collected row over one
    truncated line. See ``tests/test_scaling_probe.py`` for the
    truncated/malformed-line regression this guards (loaded by path,
    since ``scripts/`` is not an importable package -- stdlib-only,
    no subprocess, no torch).
    """
    parsed: dict[str, Any] | None = None
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            parsed = json.loads(line[len(prefix) :])
        except (json.JSONDecodeError, ValueError):
            continue
    return parsed


def _timeout_fallback_row(cfg: _ProbeConfig) -> dict[str, Any]:
    """Row for a timed-out config whose header never arrived intact.

    Used when the post-kill ``communicate()`` drain in ``_run_one_config``
    recovers nothing (or a malformed ``MINGRU_PROBE_HEADER`` line --
    ``_extract_last`` treats those the same, see its docstring) -- most
    likely a ``SIGKILL`` landing mid-``print()``, exactly the givens@4096
    wall case this probe exists to find. This is an EXPECTED timeout
    outcome, not a probe/grid bug (contrast the no-header case on a
    non-killed exit, which stays fail-loud -- see ``_run_one_config``),
    so it must not abort the whole grid: ``config``/``state_elements``
    here come from the static grid table (the value ``build_kwargs`` is
    chosen to hit -- see ``_ProbeConfig``'s docstring), not a live
    measurement, and ``params`` is unknown since the child never got to
    report it.
    """
    return {
        "label": cfg.label,
        "mechanism": cfg.mechanism,
        "config": cfg.build_kwargs,
        "training_arm": cfg.training_arm,
        "state_elements": cfg.state_elements,
        "params": None,
        "num_threads": None,
        "step_secs_median": None,
        "step_secs_all": [],
        "peak_rss_bytes": None,
        "status": "timeout",
        "oom_message_recognized": None,
        "detail": "child header unavailable or malformed at kill time "
        "(no self-report recovered before the timeout kill)",
    }


def _run_one_config(
    cfg: _ProbeConfig,
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
    timeout_secs: float,
) -> _RunOutcome:
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        cfg.label,
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--warmup",
        str(warmup),
        "--steps",
        str(steps),
    ]
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=_REPO_ROOT
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        proc.kill()
        # Documented Popen pattern: a second, timeout-less communicate()
        # after kill() drains whatever the child had already buffered
        # (see the module docstring's "timeout" outcome) -- this is how
        # the header line survives the kill.
        stdout, stderr = proc.communicate()
        header = _extract_last(_HEADER_PREFIX, stdout)
        if header is None:
            # No recoverable header -- see `_timeout_fallback_row`'s
            # docstring for why this is an expected timeout outcome (not
            # a probe/grid bug) and must not abort the rest of the grid.
            return _RunOutcome(row=_timeout_fallback_row(cfg))
        row = {
            **header,
            "step_secs_median": None,
            "step_secs_all": [],
            "peak_rss_bytes": None,
            "status": "timeout",
            "oom_message_recognized": None,
            "detail": None,
        }
        return _RunOutcome(row=row)

    result = _extract_last(_RESULT_PREFIX, stdout)
    if result is not None:
        return _RunOutcome(row=result)

    header = _extract_last(_HEADER_PREFIX, stdout)
    if header is None:
        return _RunOutcome(
            row={},
            aborted=True,
            abort_reason=(
                f"{cfg.label}: child exited {proc.returncode} before printing a header "
                f"(construction failed, not a resource wall) -- stderr tail:\n"
                + "\n".join(stderr.splitlines()[-20:])
            ),
        )
    # Header printed, but no result and a nonzero/ungraceful exit: see
    # the module docstring's "oom" outcome for why this is recorded as
    # oom rather than aborting the whole probe. `oom_message_recognized`
    # is `False` here (not `None`, unlike the ok/timeout cases): this is
    # a parent-side inference from a crashed-without-self-report child,
    # never a classified allocator message, so it should read as
    # unrecognized in the rendered artifact, not "not applicable".
    row = {
        **header,
        "step_secs_median": None,
        "step_secs_all": [],
        "peak_rss_bytes": None,
        "status": "oom",
        "oom_message_recognized": False,
        "detail": "\n".join(stderr.splitlines()[-20:]) or f"child exited {proc.returncode}",
    }
    return _RunOutcome(row=row)


# --- reporting -------------------------------------------------------------


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _cpu_info() -> str:
    """Best-effort CPU model string; falls back to ``platform.processor()``.

    Mirrors ``scripts/bench_delta.py``'s ``_cpu_info`` (see that file's
    docstring for the same rationale); duplicated here rather than
    imported since ``scripts/bench_delta.py`` is outside this task's
    footprint -- flagged as DUPLICATION-PENDING for the orchestrator.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return out.stdout.strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _oom_detail_cell(row: dict[str, Any]) -> str:
    """The 'oom detail' table cell: empty unless ``status == "oom"``.

    Surfaces ``oom_message_recognized`` and the first line of ``detail``
    directly in the rendered artifact -- see the module docstring's
    "oom" outcome for why this visibility matters (this artifact is the
    round's sole cost/memory source, so a reader must be able to spot an
    ``oom`` row whose underlying message didn't actually look like a
    memory failure). Only the first line is shown and ``|`` is escaped:
    ``detail`` can be a multi-line stderr tail, which would otherwise
    break the Markdown table's row structure.
    """
    if row["status"] != "oom":
        return ""
    recognized = row.get("oom_message_recognized")
    tag = "recognized" if recognized else "UNRECOGNIZED"
    detail = row.get("detail") or ""
    first_line = detail.splitlines()[0] if detail else ""
    first_line = first_line.replace("|", "\\|")
    return f"{tag}: {first_line}" if first_line else tag


def _render_markdown(rows: list[dict[str, Any]], env: dict[str, Any]) -> str:
    lines = [
        "# Mechanism x state scaling probe (time + peak RSS)",
        "",
        "Subprocess-isolated, uncontended forward+backward timing and peak "
        "RSS per (mechanism, state) config, per spec section 6 / intent "
        "ledger statement 4. This artifact is the sole source for the "
        "round's verdict-table cost/memory columns.",
        "",
        f"Env: torch {env['torch_version']} (evidence pin), CPU: {env['cpu_info']} "
        f"({env['num_threads']} torch threads), platform {env['platform']}, "
        f"B={env['batch_size']}, T={env['seq_len']}, warmup={env['warmup']}, "
        f"timed steps={env['steps']}, per-config timeout={env['timeout_secs']}s, "
        f"commit {env['git_commit']}, generated {env['timestamp']}.",
        "",
        "RSS units depend on platform: bytes on macOS (Darwin), kilobytes on "
        f"Linux -- this run's platform is `{env['platform']}` (see the env "
        "block above); `peak_rss_bytes` below is `ru_maxrss` reported "
        "verbatim by the child, not normalized.",
        "",
        "An `oom` row's 'oom detail' column shows whether the underlying "
        "exception message matched a known memory-allocator pattern "
        "(`recognized`) or not (`UNRECOGNIZED`, worth auditing -- this "
        "probe conservatively files any crash after a successful timed-loop "
        "start as `oom`, so an unrecognized message could be a real "
        "resource wall with an unfamiliar message, or a genuine bug at "
        "that extreme state size; see the module docstring's `oom` "
        "outcome), followed by the first line of the captured message.",
        "",
        "| mechanism | config | state elements | params | training arm | "
        "step secs (median) | peak RSS | status | oom detail |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        cfg_str = ", ".join(f"{k}={v}" for k, v in row["config"].items() if k != "input_size")
        median = f"{row['step_secs_median']:.4f}" if row["step_secs_median"] is not None else "n/a"
        rss = row["peak_rss_bytes"] if row["peak_rss_bytes"] is not None else "n/a"
        params = row["params"] if row["params"] is not None else "n/a"
        lines.append(
            f"| {row['mechanism']} | {cfg_str} | {row['state_elements']} | {params} | "
            f"{'yes' if row['training_arm'] else ''} | {median} | {rss} | {row['status']} | "
            f"{_oom_detail_cell(row)} |"
        )
    lines += [
        "",
        "Timeout/OOM rows are the frontier finding, not a failure -- see the "
        "module docstring of `scripts/scaling_probe.py`.",
        "",
    ]
    return "\n".join(lines) + "\n"


# --- CLI ---------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--child",
        metavar="LABEL",
        default=None,
        help=argparse.SUPPRESS,  # internal: parent re-invokes itself with this flag
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Reduced grid: 'label' or 'mechanism:state_elements' tokens "
        "(e.g. 'delta:64 givens:64'); default runs the full ten-config grid.",
    )
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=_DEFAULT_SEQ_LEN)
    parser.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP)
    parser.add_argument("--steps", type=int, default=_DEFAULT_STEPS)
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_SECS,
        help=f"per-config wall-clock timeout in seconds (default {_DEFAULT_TIMEOUT_SECS:g})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_OUT_DIR,
        help="artifact directory (default experiments/bench); override for smoke runs "
        "so smoke output does not land in the tracked artifact path",
    )
    return parser.parse_args(argv)


def run(argv: list[str]) -> int:
    args = _parse_args(argv)

    if args.child is not None:
        return _run_child(args.child, args.batch_size, args.seq_len, args.warmup, args.steps)

    # Hard floors (distinct from the spec-recommended defaults below): a
    # negative warmup count or zero timed steps isn't a valid-but-weak
    # smoke override, it's a nonsensical request (a negative range()
    # length, or an "ok" row with an empty step_secs_all and a null
    # median) -- fail loudly rather than silently changing what counts as
    # timed.
    if args.warmup < 0:
        print(f"FAILED: --warmup must be >= 0 (got {args.warmup})", file=sys.stderr)
        return 1
    if args.steps < 1:
        print(f"FAILED: --steps must be >= 1 (got {args.steps})", file=sys.stderr)
        return 1

    # Spec section 4 requires >=1 warmup and >=5 timed steps for a config's
    # `status="ok"` row to count as evidence -- enforced here as the
    # defaults, not a hard floor, so an explicit override (e.g. `--warmup 1
    # --steps 2` for a fast smoke run per this script's own docstring/the
    # task brief) still runs; it just prints a loud warning rather than
    # silently producing an under-sampled artifact that looks spec-compliant.
    if args.warmup < 1 or args.steps < 5:
        print(
            f"WARNING: --warmup={args.warmup} --steps={args.steps} is below the "
            "spec section 4 floor (warmup>=1, steps>=5) for a config's 'ok' row "
            "to count as evidence -- proceeding anyway (smoke/debugging use "
            "only; do not treat this artifact as the round's cost/memory "
            "source).",
            file=sys.stderr,
        )

    grid = _select_grid(args.only)
    print(f"scaling_probe: {len(grid)} config(s) to run (sequential, uncontended)")

    rows: list[dict[str, Any]] = []
    for cfg in grid:
        print(f"  [{cfg.mechanism}] {cfg.label} (state_elements={cfg.state_elements})...", end=" ")
        outcome = _run_one_config(
            cfg, args.batch_size, args.seq_len, args.warmup, args.steps, args.timeout
        )
        if outcome.aborted:
            print("ABORT")
            print(
                f"\nFAILED: {outcome.abort_reason}\n"
                "This means the config never reached the timed loop -- a probe/grid "
                "bug, not a resource wall -- so no artifact is being written. Fix "
                "the config and re-run.",
                file=sys.stderr,
            )
            return 1
        print(outcome.row["status"])
        rows.append(outcome.row)

    # Children don't echo torch_version in their header/result -- every
    # child that reached its header already passed the pin assert, so the
    # evidence-pin constant itself is the correct value for the artifact's
    # env block. num_threads is read off the first row that reported one
    # (every child uses the process-default thread count, so this is
    # representative unless --only mixed configs run under different
    # environments, which this single invocation never does).
    torch_version = _EVIDENCE_PIN_TORCH_VERSION
    num_threads = next(
        (r.get("num_threads") for r in rows if r.get("num_threads") is not None), None
    )

    env = {
        "torch_version": torch_version,
        "machine": platform.node(),
        "cpu_info": _cpu_info(),
        "num_threads": num_threads,
        "platform": platform.platform(),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "steps": args.steps,
        "timeout_secs": args.timeout,
        "git_commit": _git_commit_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "scaling_frontier.json"
    out_md = args.out_dir / "scaling_frontier.md"
    out_json.write_text(json.dumps({"env": env, "configs": rows}, indent=2) + "\n")
    out_md.write_text(_render_markdown(rows, env))
    print(f"\nwrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
