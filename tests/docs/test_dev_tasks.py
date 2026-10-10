"""dev.py, the developer task runner, checks what CI checks.

``python dev.py check`` is the promise that a local run is a CI run. These
tests hold it: every command the lint, l0 and gates tasks run is also a CI
step, and every ``python dev.py <task>`` the docs show is a real task.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOC_PAGES = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    *sorted((ROOT / "docs" / "guides").glob("*.md")),
)
SHOWN_TASK = re.compile(r"python dev\.py ([a-z0-9-]+)")


def _load_dev() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ter_dev_tasks", ROOT / "dev.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


dev = _load_dev()


def _ci_runs() -> list[list[str]]:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text("utf-8"))
    runs = []
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                for line in str(step["run"]).replace("\\\n", " ").splitlines():
                    words = line.split()
                    if words[:3] == ["python", "-m", "pytest"]:
                        words = words[2:]
                    runs.append(words)
    return runs


def _as_ci_words(command: list[str]) -> list[str]:
    """A dev.py command as CI spells it: ``ruff ...``, not ``<python> -m ruff``."""
    if command[0] == sys.executable and command[1] == "-m":
        return command[2:]
    return [Path(command[0]).name.removesuffix(".exe"), *command[1:]]


@pytest.mark.parametrize("task", ["lint", "l0", "gates"])
def test_every_command_a_ci_task_runs_is_a_ci_step(task: str) -> None:
    runs = _ci_runs()
    for command in dev.TASKS[task].steps([]):
        words = _as_ci_words(command)
        assert any(run[:1] == words[:1] and set(words) <= set(run) for run in runs), (
            f"`python dev.py {task}` runs `{' '.join(words)}`, which CI does not"
        )


def test_check_is_lint_then_gates() -> None:
    assert dev.TASKS["check"].steps([]) == [
        *dev.TASKS["lint"].steps([]),
        *dev.TASKS["gates"].steps([]),
    ]


def test_extra_arguments_reach_pytest() -> None:
    (command,) = dev.TASKS["test"].steps(["-k", "golden"])
    assert command[-2:] == ["-k", "golden"]


@pytest.mark.parametrize("page", DOC_PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_task_the_docs_show_exists(page: Path) -> None:
    shown = set(SHOWN_TASK.findall(page.read_text(encoding="utf-8")))
    assert shown <= set(dev.TASKS), f"unknown tasks: {sorted(shown - set(dev.TASKS))}"


def test_usage_lists_every_task_and_unknown_tasks_fail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert dev.main([]) == 0
    listed = capsys.readouterr().out
    assert all(f"  {name} " in listed for name in dev.TASKS)
    assert dev.main(["nope"]) == 2
    assert "unknown task `nope`" in capsys.readouterr().err


def test_a_failing_step_stops_the_task(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []

    def call(command: list[str], **_: object) -> int:
        calls.append(command)
        return 3

    monkeypatch.setattr(dev.subprocess, "call", call)
    assert dev.main(["lint"]) == 3
    assert len(calls) == 1
    assert "FAILED lint" in capsys.readouterr().err


def test_tasks_test_this_checkout_first(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def call(command: list[str], *, cwd: Path, env: dict[str, str]) -> int:
        seen.update(env)
        return 0

    monkeypatch.setenv("PYTHONPATH", "elsewhere")
    monkeypatch.setattr(dev.subprocess, "call", call)
    assert dev.main(["points"]) == 0
    assert seen["PYTHONPATH"].split(dev.os.pathsep) == [str(ROOT / "src"), "elsewhere"]
