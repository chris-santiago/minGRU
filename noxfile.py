"""Nox sessions for mingru-scans.

Modeled on ferrum's noxfile (uv-backed sessions for lint/test/doctests/build/
docs), minus the Rust sessions, plus this repo's evidence gates: the
frozen-AST check, the installed-wheel proof, and the torch==2.5.1 evidence
selftest.

Freeze discipline: ``src/mingru/min_gru.py`` and ``src/mingru/triton_scans.py``
are evidence-frozen (AST-identical to the pinned pre-packaging baseline;
enforced by ``scripts/check_frozen_ast.py``). The lint session therefore never
auto-fixes or reformats ``src/`` or the root evidence drivers -- fix/format is
scoped to ``tests/`` and ``scripts/`` -- and the ``freeze`` session runs the
gate after any lint.
"""

import nox

nox.options.default_venv_backend = "uv"


@nox.session(python=False)
def lint(session: nox.Session) -> None:
    """Ruff check --fix and format, repo-wide.

    Frozen files included by user decision: formatting is AST-invariant and
    the freeze session verifies identity afterward -- review the diffs.
    """
    session.run("uv", "run", "--group", "dev", "ruff", "check", ".", "--fix", external=True)
    session.run("uv", "run", "--group", "dev", "ruff", "format", ".", external=True)


@nox.session(python=False)
def freeze(session: nox.Session) -> None:
    """Frozen-AST gate: library + relocated selftests vs the pinned baseline."""
    session.run(
        "uv",
        "run",
        "--python",
        "3.12",
        "python",
        "scripts/check_frozen_ast.py",
        external=True,
    )


@nox.session
def test(session: nox.Session) -> None:
    """Run the CPU test suite in an isolated venv."""
    session.run("uv", "sync", "--group", "dev", external=True)
    session.run("uv", "run", "pytest", *session.posargs, external=True)


@nox.session
def doctests(session: nox.Session) -> None:
    """Run the min_gru docstring Examples as doctests.

    Scoped to ``src/mingru/min_gru.py`` (62 runnable shape-contract examples).
    ``triton_scans.py`` is deliberately not collected: its examples reference
    CUDA/Triton execution that cannot run on a CPU-only environment.
    """
    session.run("uv", "sync", "--group", "dev", external=True)
    session.run(
        "uv",
        "run",
        "pytest",
        "--doctest-modules",
        "src/mingru/min_gru.py",
        *session.posargs,
        external=True,
    )


@nox.session(python=False)
def evidence(session: nox.Session) -> None:
    """Evidence-pin selftest: the relocated suites under torch==2.5.1, no install."""
    session.run(
        "uv",
        "run",
        "--python",
        "3.12",
        "--with",
        "torch==2.5.1",
        "python",
        "min_gru.py",
        external=True,
    )


@nox.session(python=False)
def wheel(session: nox.Session) -> None:
    """Full installed-wheel proof: build, fresh venv, import smoke, suite."""
    session.run("bash", "scripts/check_wheel.sh", external=True)


@nox.session(python=False)
def gpu(session: nox.Session) -> None:
    """Run the GPU conformance suite on an ephemeral Lightning AI job.

    Requires LIGHTNING_USER_ID / LIGHTNING_API_KEY and a teamspace
    (MINGRU_LIGHTNING_TEAMSPACE or --teamspace). Consumes GPU credits;
    pass --dry-run to inspect the job spec without submitting.
    """
    session.run(
        "uv",
        "run",
        "--group",
        "dev",
        "python",
        "scripts/gpu_check.py",
        *session.posargs,
        external=True,
    )


@nox.session()
def build(session: nox.Session) -> None:
    """Build the wheel and sdist."""
    session.run("uv", "build", external=True)


@nox.session(python=False)
def docs(session: nox.Session) -> None:
    """Build the docs site with the pinned docs toolchain; strict (fails on warnings)."""
    session.run(
        "uv",
        "run",
        "--only-group",
        "docs",
        "zensical",
        "build",
        "--clean",
        "--strict",
        external=True,
    )
