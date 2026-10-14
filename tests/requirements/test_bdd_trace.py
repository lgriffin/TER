"""BDD scenarios cite EARS requirements and count in the trace (TER-TST-017,
TER-TST-018).

A scenario's ``@req:<id>`` tag becomes ``@pytest.mark.req(<id>)``
(``ter.adapters.driving.pytest_req``), so ``--req-trace`` records its outcome
like any other test, and ``ter-req lint --tests`` reads the tags from the
``.feature`` files, so a tag citing an unknown id fails the lint.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ter.adapters.driving import req_cli

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ("-p", "ter.adapters.driving.pytest_req")

FEATURE = """\
Feature: Tagged scenarios

  @req:TER-AAA-001 @slow
  Scenario: A passing scenario
    Given a step

  @req:TER-AAA-001 @req:TER-AAA-002
  Scenario: A failing scenario
    Given a failing step
"""

STEPS = """\
import pytest
from pytest_bdd import given, scenarios

scenarios("tagged.feature")


@given("a step")
def a_step():
    pass


@given("a failing step")
def a_failing_step():
    raise AssertionError("fails")


def test_other_tags_stay_markers(request):
    pass
"""


def _bdd(pytester: pytest.Pytester, feature: str = FEATURE) -> None:
    pytester.makefile(".feature", tagged=feature)
    pytester.makepyfile(test_tagged=STEPS)
    pytester.makeini("[pytest]\nmarkers =\n    slow: a slow scenario\n")


@pytest.mark.req("TER-TST-017")
def test_a_scenarios_req_tags_count_in_the_trace(pytester: pytest.Pytester) -> None:
    _bdd(pytester)
    target = pytester.path / "trace.json"
    result = pytester.runpytest(*PLUGIN, f"--req-trace={target}")
    result.assert_outcomes(passed=2, failed=1)
    trace = json.loads(target.read_text(encoding="utf-8"))["requirements"]
    outcomes = {
        rid: sorted((e["nodeid"].split("::")[-1], e["outcome"]) for e in entries)
        for rid, entries in trace.items()
    }
    assert outcomes == {
        "TER-AAA-001": [
            ("test_a_failing_scenario", "failed"),
            ("test_a_passing_scenario", "passed"),
        ],
        "TER-AAA-002": [("test_a_failing_scenario", "failed")],
    }


@pytest.mark.req("TER-TST-017")
def test_other_tags_are_still_markers(pytester: pytest.Pytester) -> None:
    _bdd(pytester)
    result = pytester.runpytest(*PLUGIN, "-m", "slow")
    result.assert_outcomes(passed=1)


@pytest.mark.req("TER-TST-017")
def test_a_req_tag_without_an_id_is_refused(pytester: pytest.Pytester) -> None:
    _bdd(pytester, FEATURE.replace("@req:TER-AAA-001 @slow", "@req: @slow"))
    result = pytester.runpytest(*PLUGIN)
    assert result.ret != pytest.ExitCode.OK
    result.stdout.fnmatch_lines(["*names no requirement id*"])


@pytest.mark.req("TER-TST-018")
def test_the_lint_reads_req_tags_from_feature_files(tmp_path: Path) -> None:
    (tmp_path / "x.feature").write_text(
        textwrap.dedent(
            """\
            Feature: x
              # @req:TER-OLD-001 in a comment is not a citation
              @req:TER-AAA-001 @wip
              Scenario: one
                Given a step
              @req:TER-ZZZ-009
              Scenario: two
                Given a step
            """
        ),
        encoding="utf-8",
    )
    where = f"{tmp_path.as_posix()}/x.feature"
    assert req_cli.scan_citations(tmp_path) == {
        "TER-AAA-001": [f"{where}:3"],
        "TER-ZZZ-009": [f"{where}:6"],
    }


@pytest.mark.req("TER-TST-018")
def test_only_tag_lines_cite_and_an_empty_tag_is_reported(tmp_path: Path) -> None:
    (tmp_path / "x.feature").write_text(
        textwrap.dedent(
            """\
            Feature: x
              A description naming @req:TER-XXX-997 cites nothing.

              @req:
              Scenario: one
                Given a step about @req:TER-XXX-998
                  \"\"\"
                  @req:TER-XXX-999 inside a docstring
                  \"\"\"
                And a fenced block
                  ```
                  @req:TER-XXX-996
                  ```
            """
        ),
        encoding="utf-8",
    )
    assert req_cli.scan_citations(tmp_path) == {
        req_cli.EMPTY_TAG: [f"{tmp_path.as_posix()}/x.feature:4"]
    }


@pytest.mark.req("TER-TST-018")
def test_every_ter4_scenario_cites_a_known_requirement() -> None:
    """TER 4 scenarios live in tests/features/ter4 and each one cites the
    requirement it verifies; the lint (CI) checks the ids are known."""
    features = sorted((ROOT / "tests" / "features" / "ter4").glob("*.feature"))
    assert features
    for path in features:
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines):
            if line.strip().startswith(("Scenario:", "Scenario Outline:")):
                tags = lines[number - 1].split()
                assert any(t.startswith("@req:TER-") for t in tags), (
                    f"{path.name}:{number + 1} cites no requirement"
                )
