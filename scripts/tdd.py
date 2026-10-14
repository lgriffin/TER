"""Test-first steps: a new test must fail before the change and pass after it.

``red`` runs the tests you just wrote, before the code they test exists. It
succeeds only when they ran and at least one failed, because a test that
already passes would not notice the behaviour missing; a test that does not
even run (a collection or usage error, no test selected) is not red either::

    python dev.py red tests/unit/test_ter4_policy_set.py::test_new_rule

``green`` runs the same tests after the change. It succeeds only when they
all pass and each one cites the EARS requirement it verifies, with
``@pytest.mark.req`` or a scenario's ``@req:`` tag (TER-TST-019,
TER-TST-020)::

    python dev.py green tests/unit/test_ter4_policy_set.py::test_new_rule

Then ``python dev.py check`` runs everything CI runs. See the testing guide,
"Test-first workflow".
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: No cache, so a red or green run never changes what ``--ff`` reruns.
PYTEST = ("-m", "pytest", "-q", "-p", "no:cacheprovider")
TRACE_PLUGIN = ("-p", "ter.adapters.driving.pytest_req")
#: pytest's exit codes (``pytest.ExitCode``), spelled out so this runs alone.
PASSED, FAILED, NO_TESTS = 0, 1, 5


def _pytest(tests: Sequence[str], *options: str, cwd: Path) -> int:
    command = [sys.executable, *PYTEST, *options, *tests]
    return subprocess.call(command, cwd=cwd)


def red(tests: Sequence[str], cwd: Path = ROOT) -> int:
    """0 when ``tests`` ran and at least one failed; 1 otherwise."""
    code = _pytest(tests, cwd=cwd)
    if code == FAILED:
        print("\nRED: the new test fails without the change. Now make it pass.")
        return 0
    if code == PASSED:
        print(
            "\nNOT RED: the tests pass already, so they would not notice the"
            " behaviour missing. Write a test that fails first.",
            file=sys.stderr,
        )
    elif code == NO_TESTS:
        print("\nNOT RED: no test was selected.", file=sys.stderr)
    else:
        print(
            f"\nNOT RED: the tests did not run (pytest exited {code}). A"
            " collection or usage error is not a failing test.",
            file=sys.stderr,
        )
    return 1


def collected(tests: Sequence[str], cwd: Path) -> list[str]:
    """The node ids pytest selects for ``tests``."""
    done = subprocess.run(
        [sys.executable, *PYTEST, "--collect-only", *tests],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in done.stdout.splitlines() if "::" in line]


def uncited(nodeids: Sequence[str], trace: dict[str, object]) -> list[str]:
    """The node ids that no requirement in a ``--req-trace`` document cites."""
    requirements = trace.get("requirements", {})
    cited = {
        entry["nodeid"]
        for entries in (requirements.values() if isinstance(requirements, dict) else ())
        for entry in entries
    }
    return [n for n in nodeids if n not in cited]


def green(tests: Sequence[str], cwd: Path = ROOT) -> int:
    """0 when ``tests`` all pass and each cites a requirement; 1 otherwise."""
    with tempfile.TemporaryDirectory() as scratch:
        trace_path = Path(scratch) / "trace.json"
        code = _pytest(tests, *TRACE_PLUGIN, f"--req-trace={trace_path}", cwd=cwd)
        if code != PASSED:
            print(f"\nNOT GREEN: pytest exited {code}.", file=sys.stderr)
            return 1
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    missing = uncited(collected(tests, cwd), trace)
    if missing:
        print(
            "\nNOT GREEN: these tests cite no requirement; tag each with"
            ' @pytest.mark.req("TER-XXX-NNN") (a scenario: @req:TER-XXX-NNN):',
            file=sys.stderr,
        )
        for nodeid in missing:
            print(f"  {nodeid}", file=sys.stderr)
        return 1
    print(
        "\nGREEN: the tests pass and cite their requirements. Now run"
        " `python dev.py check`."
    )
    return 0


def main(argv: Sequence[str]) -> int:
    if len(argv) < 2 or argv[0] not in {"red", "green"}:
        print("usage: python scripts/tdd.py red|green TEST [TEST ...]", file=sys.stderr)
        return 2
    step, tests = argv[0], argv[1:]
    return red(tests) if step == "red" else green(tests)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
