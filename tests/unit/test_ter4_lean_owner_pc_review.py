"""Rules tightened after running TER over the owner's own sessions on a
Windows PC (25 Nov 2026): 47 sessions, aggregates only, no transcript text.

- PowerShell, Claude Code's shell tool on Windows, is a shell (TER-DET-012).
- Running a script the agent edited checks that edit (TER-DET-013).
- Drift: an announcement must name the change (TER-ITN-009); a rule for the
  work is not a dropped goal (TER-ITN-010); refused edits and scratch files
  are not judged (TER-ITN-011); a compaction summary is not intent
  (TER-ITN-008).

Every case below is synthetic; each mirrors a pattern counted on the owner's
sessions.
"""

from __future__ import annotations

from typing import Any

import pytest
from ter4_lean_builder import PASS, Script

from ter.adapters.driven.claude_code import tool_kind
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import ToolKind
from ter.domain.lean import Finding, LeanAnalysis, explain, key_terms
from ter.domain.lean.facts import (
    injected_prompt,
    runs_check,
    scripts_run,
    shell_intent,
)
from ter.domain.lean.intent import distinctive
from ter.domain.lean.model import ShellIntent


def run(script: Script) -> LeanAnalysis:
    return explain(script.events, RegexTokenizer())


def found(script: Script, detector: str) -> list[Finding]:
    return [f for f in run(script).findings if f.detector == detector]


def powershell(s: Script, command: str, output: str | None = "ok") -> Any:
    return s.call("PowerShell", ToolKind.EXEC_SHELL, {"command": command}, output)


# --- TER-DET-012: PowerShell is a shell ---------------------------------------


@pytest.mark.req("TER-DET-012", "TER-SRC-003")
def test_the_powershell_tool_is_a_shell() -> None:
    assert tool_kind("PowerShell") is ToolKind.EXEC_SHELL


@pytest.mark.req("TER-DET-012")
@pytest.mark.parametrize(
    ("command", "intent"),
    [
        # Cmdlets and aliases, in any case, after a move to a folder.
        ("Get-Content src\\a.py -TotalCount 40", ShellIntent.EXPLORE),
        ("Set-Location D:\\code\\x; Get-ChildItem -Recurse", ShellIntent.EXPLORE),
        ("Get-Date; select-string -Path a.py -Pattern x", ShellIntent.EXPLORE),
        ("gci src -Filter *.py", ShellIntent.EXPLORE),
        ("Remove-Item -Recurse -Force build", ShellIntent.CHANGE),
        ("Set-Content out.txt 'x' -Encoding utf8", ShellIntent.CHANGE),
        # Relative Windows paths and the call operator reach the program.
        (".\\.venv\\Scripts\\python -m pytest -q", ShellIntent.VALIDATE),
        ('& "C:\\venv\\Scripts\\python.exe" -m mypy src', ShellIntent.VALIDATE),
        # $env: assignments and script blocks are not commands.
        (
            "$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python -m pytest",
            ShellIntent.VALIDATE,
        ),
        ("npm ci; if ($?) { npm test }", ShellIntent.VALIDATE),
        # A here-string body is data, like a heredoc's.
        ("$t = @'\npytest -q\n'@\nSet-Content run.txt $t", ShellIntent.CHANGE),
        # Unknown cmdlets stay unclassified.
        ("Get-Process java | Format-Table", ShellIntent.OTHER),
    ],
)
def test_powershell_lines_are_classified(command: str, intent: ShellIntent) -> None:
    assert shell_intent(command) is intent


@pytest.mark.req("TER-DET-012")
@pytest.mark.parametrize(
    ("command", "intent"),
    [
        # POSIX lines mean what they meant before.
        ("cd src && ls -la", ShellIntent.EXPLORE),
        ("cd src && pytest -q", ShellIntent.VALIDATE),
        ("grep -rn 'a\\.b' src", ShellIntent.EXPLORE),
        ("FOO=1 python -m pytest", ShellIntent.VALIDATE),
        ("cat <<'EOF' > x.sh\npytest\nEOF", ShellIntent.EXPLORE),
    ],
)
def test_posix_lines_keep_their_meaning(command: str, intent: ShellIntent) -> None:
    assert shell_intent(command) is intent


@pytest.mark.req("TER-DET-012")
def test_a_check_run_in_powershell_is_a_check() -> None:
    assert runs_check(".\\.venv\\Scripts\\python -m pytest -q tests\\unit")
    assert not runs_check("Get-Content pytest.ini")


@pytest.mark.req("TER-DET-012", "TER-DET-005")
def test_an_edit_checked_in_powershell_is_validated() -> None:
    s = Script()
    s.prompt("Fix the parser in src/a.py")
    s.read("src/a.py", "def parse(): pass")
    s.edit("src/a.py")
    powershell(s, ".\\.venv\\Scripts\\python -m pytest -q", PASS)
    s.say("Fixed src/a.py; the tests pass.")
    assert found(s, "unvalidated_implementation") == []


@pytest.mark.req("TER-DET-012", "TER-DET-005")
def test_an_edit_only_looked_at_in_powershell_is_still_unvalidated() -> None:
    s = Script()
    s.prompt("Fix the parser in src/a.py")
    s.read("src/a.py", "def parse(): pass")
    s.edit("src/a.py")
    powershell(s, "Get-Content src\\a.py", "def parse(): return 1")
    s.say("Fixed src/a.py.")
    [f] = found(s, "unvalidated_implementation")
    assert f.confidence == 0.85


# --- TER-DET-013: running an edited script checks it --------------------------


@pytest.mark.req("TER-DET-013")
@pytest.mark.parametrize(
    ("command", "ran"),
    [
        ("python tools/sweep.py --all", {"sweep.py"}),
        ("./build.sh && echo done", {"build.sh"}),
        ("bash -x scripts/build.sh", {"build.sh"}),
        (
            'powershell -NoProfile -ExecutionPolicy Bypass -File "C:\\t\\chain.ps1"',
            {"chain.ps1"},
        ),
        ("& 'D:\\s\\sdp.ps1' -Repo x", {"sdp.ps1"}),
        ("cd D:\\x; .\\queue.ps1 run pc-1", {"queue.ps1"}),
        ("node dist/cli.mjs", {"cli.mjs"}),
        # Only looked at, copied or searched: not run.
        ("cat build.sh", set()),
        ("Get-Content chain.ps1 -Raw", set()),
        ("cp build.sh build2.sh", set()),
        ("grep -n main build.sh", set()),
        ("pytest -q", set()),
    ],
)
def test_scripts_run_by_a_shell_line(command: str, ran: set[str]) -> None:
    assert scripts_run(command) == frozenset(ran)


@pytest.mark.req("TER-DET-013", "TER-DET-005")
def test_running_the_edited_script_validates_it() -> None:
    s = Script()
    s.prompt("Make the nightly script skip empty folders")
    s.read("scripts/nightly.sh", "for d in *; do run $d; done")
    s.edit("scripts/nightly.sh")
    s.bash("bash scripts/nightly.sh --dry-run", "skipped 2 empty folders")
    s.say("The nightly script now skips empty folders.")
    assert found(s, "unvalidated_implementation") == []


@pytest.mark.req("TER-DET-013", "TER-DET-005")
def test_running_another_script_does_not_validate_the_edit() -> None:
    s = Script()
    s.prompt("Make the nightly script skip empty folders")
    s.read("scripts/nightly.sh", "for d in *; do run $d; done")
    s.edit("scripts/nightly.sh")
    s.edit("src/a.py")
    s.bash("bash scripts/nightly.sh --dry-run", "skipped 2 empty folders")
    s.say("Done.")
    [f] = found(s, "unvalidated_implementation")
    assert "src/a.py" in f.subject and "nightly" not in f.subject


@pytest.mark.req("TER-DET-013", "TER-DET-005")
def test_reading_the_edited_script_does_not_validate_it() -> None:
    s = Script()
    s.prompt("Make the nightly script skip empty folders")
    s.read("scripts/nightly.sh", "for d in *; do run $d; done")
    s.edit("scripts/nightly.sh")
    s.bash("cat scripts/nightly.sh", "for d in *; do ...")
    s.say("Done.")
    assert len(found(s, "unvalidated_implementation")) == 1


@pytest.mark.req("TER-DET-015", "TER-DET-005")
def test_unchecked_scratch_scripts_are_uncertain() -> None:
    s = Script()
    s.prompt("Queue the overnight benchmark")
    s.write("/tmp/claude/chain.ps1", "Start-Sleep 60")
    s.say("The chain will start when the current run ends.")
    [f] = found(s, "unvalidated_implementation")
    assert f.uncertain and f.confidence == 0.5 and "scratch" in f.explanation


@pytest.mark.req("TER-DET-015", "TER-DET-005")
def test_a_project_file_beside_a_scratch_file_still_needs_a_check() -> None:
    s = Script()
    s.prompt("Fix the parser in src/a.py")
    s.read("src/a.py", "def parse(): pass")
    s.edit("src/a.py")
    s.write("/tmp/claude/notes.py", "x = 1")
    s.say("Fixed.")
    [f] = found(s, "unvalidated_implementation")
    assert not f.uncertain


# --- TER-DET-014: a failing ad hoc run is not a failing check -----------------


@pytest.mark.req("TER-DET-014", "TER-DET-005")
def test_a_failing_probe_is_not_a_failing_check() -> None:
    s = Script()
    s.prompt("Is duckdb installed?")
    s.bash(
        'python -c "import duckdb"',
        "ModuleNotFoundError: No module named 'duckdb'\nExit code 1",
    )
    s.say("duckdb is not installed.")
    assert found(s, "unvalidated_implementation") == []


@pytest.mark.req("TER-DET-014", "TER-DET-005")
def test_a_failing_named_check_is_still_reported() -> None:
    s = Script()
    s.prompt("Run the tests")
    s.bash("pytest -q", "1 failed, 20 passed in 4.11s")
    s.say("One test fails.")
    [f] = found(s, "unvalidated_implementation")
    assert f.title == "Responded after a failing check" and f.confidence == 0.85


@pytest.mark.req("TER-DET-014", "TER-DET-005")
def test_an_ad_hoc_run_still_validates_the_edits_before_it() -> None:
    s = Script()
    s.prompt("Fix the parser in src/a.py")
    s.read("src/a.py", "def parse(): pass")
    s.edit("src/a.py")
    s.bash("python -c 'import a; a.parse()'", "ok")
    s.say("Fixed.")
    assert found(s, "unvalidated_implementation") == []


# --- TER-ITN-009: an announcement must name the change ------------------------

GOAL = "Add a parse_rows function to src/rows.py that splits CSV text into rows."
ROWS_EDIT = "def parse_rows(text):\n    return text.splitlines()\n"
CACHE_EDIT = (
    "class RetryPolicy:\n    pass\n\n\n"
    "def backoff_delay(attempt):\n    return 2 ** attempt\n"
)


def _announced(thought: str, edit: str = CACHE_EDIT) -> list[Finding]:
    s = Script()
    s.prompt(GOAL)
    s.read("src/rows.py", "")
    s.edit("src/rows.py", "", ROWS_EDIT)
    s.think(thought)
    s.edit("src/rows.py", "", edit)
    return found(s, "intent_drift")


@pytest.mark.req("TER-ITN-009", "TER-ITN-003")
def test_a_sentence_naming_a_distinctive_name_announces_it() -> None:
    [f] = _announced("I will also add a RetryPolicy so failed reads back off.")
    assert f.confidence == 0.85 and "additional" in f.explanation


@pytest.mark.req("TER-ITN-009", "TER-ITN-003")
def test_an_also_elsewhere_in_the_block_announces_nothing() -> None:
    [f] = _announced(
        "Rows parse now and I also checked the tests. "
        "Next I add RetryPolicy and backoff_delay so failed reads back off."
    )
    assert f.uncertain and f.confidence == 0.55


@pytest.mark.req("TER-ITN-009", "TER-ITN-003")
def test_one_shared_plain_word_announces_nothing() -> None:
    # 'delay' is a word of the sentence and of the change, not its name.
    [f] = _announced("I also want the delay to grow between calls.")
    assert f.uncertain


@pytest.mark.req("TER-ITN-009", "TER-ITN-003")
def test_two_plain_names_in_the_sentence_announce_it() -> None:
    edit = "def mean(xs):\n    return 0\n\n\ndef variance(xs):\n    return 0\n"
    [f] = _announced("I will also add mean and variance helpers.", edit)
    assert f.confidence == 0.85


@pytest.mark.req("TER-ITN-009", "TER-ITN-003")
def test_two_plain_names_of_a_large_change_announce_nothing() -> None:
    # A new module defining many plain names shares two with most sentences.
    edit = "".join(f"def {n}(x):\n    return x\n\n\n" for n in PLAIN_NAMES)
    [f] = _announced("I will also add mean and variance helpers.", edit)
    assert f.uncertain


PLAIN_NAMES = ("mean", "variance", "spread", "skew", "kurtosis", "quartile")


@pytest.mark.req("TER-ITN-009")
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RowCache", True),
        ("parse_rows", True),
        ("MAX_ROWS", True),
        ("utf8_name", True),
        ("_private_helper", True),
        ("rows", False),
        ("fees", False),
        ("out", False),
        ("Plan", False),
        ("__init__", False),
        ("TODAY", True),
        ("Ab", False),
    ],
)
def test_distinctive_names(name: str, expected: bool) -> None:
    assert distinctive(name) is expected


# --- TER-ITN-010: a rule for the work is not a dropped goal -------------------


@pytest.mark.req("TER-ITN-010", "TER-ITN-003")
@pytest.mark.parametrize(
    "rule",
    [
        "Don't push to main and stop at the first failing test.",
        "Do not modify the tests directory.",
        "Stop after the tests pass and tell me.",
    ],
)
def test_a_rule_for_the_work_drops_no_goal(rule: str) -> None:
    s = Script()
    s.prompt(GOAL + " " + rule)
    s.read("src/rows.py", "")
    s.edit("src/rows.py", "", ROWS_EDIT)
    a = run(s)
    current = a.intent.record.current
    assert current is not None and not current.abandoned


@pytest.mark.req("TER-ITN-010", "TER-ITN-003")
@pytest.mark.parametrize(
    "giving_up",
    [
        "Forget about the cache.",
        "Never mind the cache.",
        "We no longer need the cache.",
        "Stop working on the cache.",
        "Don't bother with the cache.",
    ],
)
def test_giving_a_goal_up_still_drops_it(giving_up: str) -> None:
    s = Script()
    s.prompt("Add a row cache to src/rows.py and parse CSV text into rows.")
    s.prompt(giving_up + " Just parse CSV text into rows in src/rows.py.")
    a = run(s)
    current = a.intent.record.current
    assert current is not None and key_terms("cache") <= current.abandoned


# --- TER-ITN-011: refused edits and scratch files are not drift ---------------


@pytest.mark.req("TER-ITN-011", "TER-ITN-003")
def test_a_refused_edit_is_not_drift() -> None:
    s = Script()
    s.prompt(GOAL)
    s.read("src/rows.py", "")
    s.think("I will also add a RetryPolicy so failed reads back off.")
    s.edit(
        "src/rows.py",
        "",
        CACHE_EDIT,
        output="<tool_use_error>File has not been read yet.</tool_use_error>",
    )
    assert found(s, "intent_drift") == []


@pytest.mark.req("TER-ITN-011", "TER-ITN-003")
@pytest.mark.parametrize(
    "path",
    [
        "C:\\Users\\dev\\AppData\\Local\\Temp\\claude\\p\\s\\scratchpad\\chain.py",
        "/tmp/claude/helper.py",
        "/var/folders/xy/T/helper.py",
        "work/scratchpad/helper.py",
    ],
)
def test_a_scratch_file_is_not_drift(path: str) -> None:
    s = Script()
    s.prompt(GOAL)
    s.read(path, "")
    s.think("I will also add a RetryPolicy so failed reads back off.")
    s.edit(path, "", CACHE_EDIT)
    assert found(s, "intent_drift") == []


@pytest.mark.req("TER-ITN-011", "TER-ITN-003")
def test_a_project_file_named_like_temp_is_still_judged() -> None:
    s = Script()
    s.prompt(GOAL)
    s.read("src/template.py", "")
    s.think("I will also add a RetryPolicy so failed reads back off.")
    s.edit("src/template.py", "", CACHE_EDIT)
    [f] = found(s, "intent_drift")
    assert f.confidence == 0.85


# --- TER-ITN-008: a compaction summary is not intent --------------------------


@pytest.mark.req("TER-ITN-008")
def test_a_compaction_summary_is_written_by_the_harness() -> None:
    summary = (
        "This session is being continued from a previous conversation that ran "
        "out of context. The summary below covers the earlier portion."
    )
    assert injected_prompt(summary)
    assert not injected_prompt("This session is slow; continue the parser work.")


@pytest.mark.req("TER-ITN-008", "TER-ITN-003")
def test_a_compaction_summary_does_not_replace_the_intent() -> None:
    s = Script()
    s.prompt(GOAL)
    s.prompt(
        "This session is being continued from a previous conversation that ran "
        "out of context. Summary: the developer asked for a release checklist, "
        "a changelog and a version bump."
    )
    s.read("src/rows.py", "")
    s.edit("src/rows.py", "", ROWS_EDIT)
    a = run(s)
    assert len(a.intent.record.revisions) == 1
    assert found(s, "intent_drift") == []
