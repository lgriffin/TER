"""Developer tasks: one command for each thing CI checks.

Run ``python dev.py`` to list the tasks, ``python dev.py check`` to run the
CI lint and test jobs (the package build job is CI's alone). Works the same on
Linux, macOS and Windows, needs only the standard library and the ``.[dev]``
install, and runs every tool from the interpreter that runs this file, so a
tool from another environment is never picked up. tests/docs/test_dev_tasks.py
fails if a task runs a command that no CI step runs.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACE = "req-trace.json"
GATES = ("L0", "L1", "L2", "L3")
#: The coverage CI measures; pyproject.toml sets the 90% floor.
COVERAGE = ("--cov=ter_calculator", "--cov=ter", "--cov-branch")
PY_PATHS = ("src", "tests", "dev.py")

Command = list[str]


def py(*args: str) -> Command:
    """A module run by this interpreter: ``python -m <args>``."""
    return [sys.executable, "-m", *args]


def tool(name: str, *args: str) -> Command:
    """A console script installed beside this interpreter (``ter-req``, ...).

    Never a bare name looked up on PATH, which could be another environment's
    copy; a missing script fails with a pointer to ``python dev.py setup``.
    """
    suffix = ".exe" if os.name == "nt" else ""
    path = Path(sysconfig.get_path("scripts")) / f"{name}{suffix}"
    return [str(path), *args]


@dataclass(frozen=True)
class Task:
    help: str
    steps: Callable[[list[str]], list[Command]]


def _setup(_: list[str]) -> list[Command]:
    return [
        py("pip", "install", "-c", "constraints/dev.txt", "-e", ".[dev]"),
        py("pre_commit", "install"),
    ]


def _fmt(_: list[str]) -> list[Command]:
    return [py("ruff", "format", *PY_PATHS), py("ruff", "check", "--fix", *PY_PATHS)]


def _lint(_: list[str]) -> list[Command]:
    return [
        py("ruff", "format", "--check", *PY_PATHS),
        py("ruff", "check", *PY_PATHS),
        py("mypy", "src/"),
        tool("lint-imports"),
        tool("ter-req", "lint", "--strict", "--tests", "tests"),
        tool("ter-req", "points", "--check"),
    ]


def _test(extra: list[str]) -> list[Command]:
    return [py("pytest", *extra)]


def _fast(extra: list[str]) -> list[Command]:
    return [py("pytest", "-q", "-x", "--ff", "-m", "not embeddings", *extra)]


def _l0(_: list[str]) -> list[Command]:
    return [py("pytest", "tests/golden", "tests/contract", "tests/architecture", "-q")]


def _gates(extra: list[str]) -> list[Command]:
    return [
        py("pytest", *COVERAGE, f"--req-trace={TRACE}", *extra),
        *(tool("ter-req", "trace", "--results", TRACE, "--gate", g) for g in GATES),
    ]


def _check(extra: list[str]) -> list[Command]:
    return [*_lint(extra), *_gates(extra)]


def _golden(extra: list[str]) -> list[Command]:
    return [py("pytest", "tests/golden", "-q", *extra)]


def _points(_: list[str]) -> list[Command]:
    return [tool("ter-req", "points")]


TASKS: dict[str, Task] = {
    "setup": Task("install TER with the dev extra and the pre-commit hooks", _setup),
    "fmt": Task("format and auto-fix lint (ruff)", _fmt),
    "lint": Task(
        "ruff, mypy, import contracts, EARS lint, points index (CI lint job)", _lint
    ),
    "test": Task("the whole suite; extra arguments go to pytest", _test),
    "fast": Task(
        "quick loop: stop at first failure, last failures first, no model", _fast
    ),
    "l0": Task("the L0 gate: golden parity, port contracts, architecture", _l0),
    "gates": Task(
        "full suite with coverage and traces, then the L0-L3 gates (CI test job)",
        _gates,
    ),
    "check": Task(
        "lint, then gates: the CI lint and test jobs (not the package build)", _check
    ),
    "golden": Task(
        "golden snapshots; TER_UPDATE_GOLDEN=1 rewrites them on purpose", _golden
    ),
    "points": Task(
        "regenerate docs/ter4/points.md from requirements/points.yaml", _points
    ),
}


def _embeddings_installed() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def run(name: str, extra: list[str]) -> int:
    # src first, so an editable install that points at another checkout is
    # never what gets tested.
    paths = [str(ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    for command in TASKS[name].steps(extra):
        shown = " ".join(
            Path(command[0]).name if i == 0 else w for i, w in enumerate(command)
        )
        print(f"\n==> {shown}", flush=True)
        started = time.monotonic()
        try:
            code = subprocess.call(command, cwd=ROOT, env=env)
        except FileNotFoundError:
            print(
                f"\nFAILED {name}: `{shown}` is not installed in this environment"
                f" ({Path(sys.executable).parent}); run `python dev.py setup`",
                file=sys.stderr,
            )
            return 127
        if code != 0:
            print(f"\nFAILED {name}: `{shown}` exited {code}", file=sys.stderr)
            return code
        print(f"    ok in {time.monotonic() - started:.1f}s", flush=True)
    print(f"\nPASSED {name}")
    if name in {"check", "gates", "test"} and not _embeddings_installed():
        print(
            "note: the embeddings extra is not installed, so the tests marked"
            " `embeddings` were skipped; CI runs them. Install it with"
            " `python -m pip install -c constraints/dev.txt -e .[dev,embeddings]`."
        )
    return 0


def usage() -> str:
    width = max(map(len, TASKS))
    lines = ["usage: python dev.py <task> [pytest arguments]", "", "tasks:"]
    lines += [f"  {name:<{width}}  {task.help}" for name, task in TASKS.items()]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(usage())
        return 0
    name, extra = argv[0], argv[1:]
    if name not in TASKS:
        print(f"unknown task `{name}`\n\n{usage()}", file=sys.stderr)
        return 2
    return run(name, extra)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
