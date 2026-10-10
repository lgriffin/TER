"""Outcome and acceptance: the verdict, the JUnit reader and the A3 beside it.

TER-OUT-*: judging evidence against an acceptance contract, and the JUnit
reference adapter. TER-SCR-004..006: the verdict sits beside the scorecard,
no behaviour measure reads it, and tokens per verified outcome.
"""

from __future__ import annotations

import dataclasses
import io
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import grimp
import pytest

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.in_memory import InMemoryOutcomeSource
from ter.adapters.driven.junit import JUnitOutcomeSource
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.adapters.driving.cli import main as cli_main
from ter.adapters.driving.reports import render_a3_html
from ter.application import ExplainedSession, ExplainSession
from ter.bootstrap import cli_services, make_outcome_source
from ter.domain.outcome import (
    AcceptanceContract,
    Check,
    CheckEvidence,
    CheckStatus,
    OutcomeEvidence,
    OutcomeFormatError,
    OutcomeVerdict,
    Verdict,
    combine_statuses,
    judge,
    per_verified_outcome,
)

ROOT = Path(__file__).resolve().parents[2]
PYTEST_JUNIT = ROOT / "tests" / "fixtures" / "outcome" / "pytest-junit.xml"
SESSION = ROOT / "tests" / "golden" / "sessions" / "rework_loop.jsonl"

P, F, E, S = (
    CheckStatus.PASSED,
    CheckStatus.FAILED,
    CheckStatus.ERROR,
    CheckStatus.SKIPPED,
)


def _ev(*checks: tuple[str, CheckStatus]) -> OutcomeEvidence:
    return OutcomeEvidence(
        "run-1",
        "test",
        tuple(CheckEvidence(c, s, f"r#{i}") for i, (c, s) in enumerate(checks)),
    )


# --- judging ------------------------------------------------------------------


@pytest.mark.req("TER-OUT-004")
def test_all_required_checks_passing_is_accepted_with_every_checks_evidence() -> None:
    v = judge(_ev(("a", P), ("b", P)))
    assert v.verdict is Verdict.ACCEPTED and v.accepted
    assert v.contract.name == "every recorded check passes"
    assert [(r.check.id, r.status, len(r.evidence)) for r in v.results] == [
        ("a", P, 1),
        ("b", P, 1),
    ]
    assert v.reasons == ("all 2 required check(s) passed",)


@pytest.mark.req("TER-OUT-004")
def test_the_verdict_carries_no_weighted_score() -> None:
    fields = {f.name: f.type for f in dataclasses.fields(OutcomeVerdict)}
    assert "float" not in " ".join(str(t) for t in fields.values())
    payload = judge(_ev(("a", P), ("b", F))).as_dict()
    assert set(payload) == {
        "verdict",
        "run",
        "source",
        "contract",
        "reasons",
        "required_checks",
        "checks",
        "unlisted",
    }
    assert payload["verdict"] == "rejected"


@pytest.mark.req("TER-OUT-005")
def test_a_failing_or_erroring_required_check_rejects() -> None:
    assert judge(_ev(("a", P), ("b", F))).verdict is Verdict.REJECTED
    v = judge(_ev(("a", E), ("b", S)))
    assert v.verdict is Verdict.REJECTED
    assert v.reasons == ("1 required check(s) failed: a",)


@pytest.mark.req("TER-OUT-005")
def test_a_rerun_that_failed_once_still_counts_as_failed() -> None:
    assert combine_statuses([P, F]) is F
    assert combine_statuses([F, E]) is E
    assert combine_statuses([S, P]) is P
    assert combine_statuses([]) is None
    v = judge(_ev(("a", F), ("a", P)))
    assert v.verdict is Verdict.REJECTED
    assert len(v.results) == 1 and len(v.results[0].evidence) == 2


@pytest.mark.req("TER-OUT-006")
def test_skipped_or_missing_required_checks_are_incomplete() -> None:
    assert judge(_ev(("a", P), ("b", S))).verdict is Verdict.INCOMPLETE
    contract = AcceptanceContract("spec", (Check("a"), Check("c")))
    v = judge(_ev(("a", P)), contract)
    assert v.verdict is Verdict.INCOMPLETE
    assert v.results[1].status is None and v.results[1].evidence == ()
    assert v.reasons == ("1 required check(s) not shown to pass: c",)
    empty = judge(_ev())
    assert empty.verdict is Verdict.INCOMPLETE
    assert empty.reasons == ("the acceptance contract requires no check",)


@pytest.mark.req("TER-OUT-004")
def test_optional_checks_are_shown_but_never_decide() -> None:
    contract = AcceptanceContract("spec", (Check("a"), Check("lint", required=False)))
    v = judge(_ev(("a", P), ("lint", F), ("extra", F)), contract)
    assert v.verdict is Verdict.ACCEPTED
    assert v.results[1].status is F
    assert [e.check_id for e in v.unlisted] == ["extra"]
    assert v.count(P) == 1 and v.count(F, required=False) == 1


def test_contract_and_evidence_reject_blank_or_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate check ids"):
        AcceptanceContract("x", (Check("a"), Check("a")))
    with pytest.raises(ValueError):
        Check(" ")
    with pytest.raises(ValueError):
        CheckEvidence("", P, "r")


@pytest.mark.req("TER-SCR-006")
def test_per_verified_outcome_divides_by_accepted_outcomes_only() -> None:
    ok, bad = judge(_ev(("a", P))), judge(_ev(("a", F)))
    assert per_verified_outcome(1200, [ok, ok, bad]) == 600
    assert per_verified_outcome(1200, [bad]) is None
    assert per_verified_outcome(1200, []) is None


# --- the JUnit reference adapter ---------------------------------------------


@pytest.mark.req("TER-OUT-007")
def test_reads_the_junit_xml_pytest_writes() -> None:
    evidence = JUnitOutcomeSource().outcome(PYTEST_JUNIT)
    assert evidence is not None
    assert evidence.source == "junit" and evidence.run_ref == str(PYTEST_JUNIT)
    assert [(c.check_id, c.status) for c in evidence.checks] == [
        ("test_sample::test_parses_header", P),
        ("test_sample::test_rejects_empty_input", F),
        ("test_sample::test_fetches_remote", S),
        ("test_sample::test_uses_fixture", E),
        ("test_sample.TestParser::test_round_trip", P),
    ]
    failed = evidence.checks[1]
    assert failed.detail == "AssertionError: empty input was accepted"
    assert failed.source == "pytest-junit.xml#testcase-2"
    assert failed.seconds == 0.01
    assert judge(evidence).verdict is Verdict.REJECTED


@pytest.mark.req("TER-OUT-007")
def test_a_single_testsuite_root_and_missing_class_names_are_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "r.xml"
    path.write_text(
        '<testsuite><testcase name="go_test_ok" time="n/a"/>'
        '<testcase name="t2"><failure>line one\nline two</failure></testcase></testsuite>',
        encoding="utf-8",
    )
    evidence = JUnitOutcomeSource().outcome(path)
    assert evidence is not None
    assert [(c.check_id, c.status, c.detail, c.seconds) for c in evidence.checks] == [
        ("go_test_ok", P, "", None),
        ("t2", F, "line one", None),
    ]


@pytest.mark.req("TER-OUT-003")
@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("<report><testcase name='a'/></report>", "root element <report>"),
        ("<testsuite><testcase classname='x'/></testsuite>", "testcase 1 has no name"),
        ("", "not well-formed"),
    ],
)
def test_unreadable_results_raise_a_format_error(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "r.xml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(OutcomeFormatError, match=message):
        JUnitOutcomeSource().outcome(path)
    with pytest.raises(OutcomeFormatError, match="cannot read"):
        JUnitOutcomeSource().outcome(tmp_path)


LAUGHS = (
    '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
    + "".join(
        f'<!ENTITY lol{i} "{("&lol" + (str(i - 1) if i > 1 else "") + ";") * 10}">'
        for i in range(1, 10)
    )
    + ']><testsuite><testcase name="&lol9;"/></testsuite>'
)
EXTERNAL = (
    '<?xml version="1.0"?><!DOCTYPE t [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
    '<testsuite><testcase name="&x;"/></testsuite>'
)


@pytest.mark.req("TER-OUT-008")
@pytest.mark.parametrize(
    "content",
    [
        LAUGHS,
        EXTERNAL,
        '<!DOCTYPE testsuite SYSTEM "http://example.invalid/x.dtd"><testsuite/>',
    ],
    ids=["billion-laughs", "external-entity", "external-dtd"],
)
def test_a_declared_doctype_is_rejected_before_any_expansion(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "evil.xml"
    path.write_text(content, encoding="utf-8")
    started = time.perf_counter()
    with pytest.raises(OutcomeFormatError, match="must not declare a DOCTYPE"):
        JUnitOutcomeSource().outcome(path)
    assert time.perf_counter() - started < 1.0


@pytest.mark.req("TER-OUT-008")
def test_a_utf16_doctype_is_rejected_too(tmp_path: Path) -> None:
    path = tmp_path / "evil16.xml"
    path.write_bytes(
        EXTERNAL.replace('"1.0"?', '"1.0" encoding="utf-16"?').encode("utf-16")
    )
    with pytest.raises(OutcomeFormatError, match="must not declare a DOCTYPE"):
        JUnitOutcomeSource().outcome(path)


@pytest.mark.req("TER-ARC-004")
def test_the_junit_adapter_is_the_outcome_source_capability() -> None:
    assert isinstance(make_outcome_source(), JUnitOutcomeSource)


# --- beside the scorecard, never inside it ------------------------------------


def _explain(
    outcome_ref: str | None, records: dict[str, OutcomeEvidence] | None = None
) -> ExplainedSession:
    use_case = ExplainSession(
        ClaudeCodeJsonlSource(),
        RegexTokenizer(),
        None,
        JUnitOutcomeSource() if records is None else InMemoryOutcomeSource(records),
    )
    return use_case(SESSION, outcome_ref)


def _behaviour_only(a3: dict[str, Any]) -> dict[str, Any]:
    analysis = dict(a3["analysis"])
    analysis["scorecard"] = {
        k: v
        for k, v in analysis["scorecard"].items()
        if k != "software_value_efficiency"
    }
    analysis["dimensions"] = [
        {
            **d,
            "measures": [
                m for m in d["measures"] if m["key"] != "software_value_efficiency"
            ],
        }
        for d in analysis["dimensions"]
        if d["dimension"] != "outcome"
    ]
    return {**a3, "analysis": analysis}


@pytest.mark.req("TER-SCR-004", "TER-SCR-006")
def test_the_verdict_sits_beside_an_unchanged_scorecard() -> None:
    plain = _explain(None)
    judged = _explain(str(PYTEST_JUNIT))
    assert plain.outcome is None and plain.a3.outcome is None
    assert judged.outcome is not None and judged.outcome.verdict is Verdict.REJECTED
    assert judged.analysis.as_dict() == plain.analysis.as_dict()
    a3 = judged.a3.as_dict()
    outcome = a3.pop("outcome")
    # Every behaviour measure is unchanged; only the figures that join value
    # to the verdict (Software Value Efficiency, the outcome dimension) differ.
    assert _behaviour_only(a3) == _behaviour_only(plain.a3.as_dict())
    assert isinstance(outcome, dict)
    assert outcome["verdict"] == "rejected"
    assert outcome["generated_tokens_per_verified_outcome"] is None
    page = render_a3_html(judged.a3)
    assert "Outcome</h2>" in page and "rejected" in page
    assert "no accepted outcome to divide by" in page
    assert "Outcome</h2>" not in render_a3_html(plain.a3)


@pytest.mark.req("TER-SCR-006")
def test_an_accepted_outcome_reports_tokens_per_verified_outcome() -> None:
    green = _ev(("a", P))
    judged = _explain("run-1", {"run-1": green})
    generated = judged.analysis.scorecard.generated_tokens
    assert judged.a3.tokens_per_verified_outcome == generated
    outcome = judged.a3.as_dict()["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["generated_tokens_per_verified_outcome"] == generated
    assert "generated tokens / accepted outcomes" in render_a3_html(judged.a3)


@pytest.mark.req("TER-OUT-002")
def test_no_recorded_outcome_shows_no_verdict() -> None:
    judged = _explain("missing", {})
    assert judged.outcome is None and "outcome" not in judged.a3.as_dict()


BEHAVIOUR_MODULES = {
    "ter.domain.events",
    "ter.domain.scoring",
    "ter.domain.stream",
    "ter.domain.lean.analysis",
    "ter.domain.lean.detectors",
    "ter.domain.lean.steps",
}


@pytest.mark.req("TER-SCR-005")
def test_behaviour_measures_are_blind_to_the_outcome_module() -> None:
    # The contract itself is enforced by lint-imports and
    # tests/architecture/test_import_contracts.py (TER-ARC-001).
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contract = next(
        c
        for c in config["tool"]["importlinter"]["contracts"]
        if c["id"] == "behaviour-blind-to-outcome"
    )
    assert contract["type"] == "forbidden"
    assert contract["forbidden_modules"] == ["ter.domain.outcome"]
    assert BEHAVIOUR_MODULES <= set(contract["source_modules"])
    graph = grimp.build_graph("ter")
    for module in contract["source_modules"]:
        chain = graph.find_shortest_chain(
            importer=module, imported="ter.domain.outcome"
        )
        assert chain is None, f"{module} reaches the outcome verdict via {chain}"
    # ...while the A3 view-model, which shows the verdict, does import it.
    assert graph.find_shortest_chain(
        importer="ter.domain.lean.a3", imported="ter.domain.outcome"
    )


# --- command line -------------------------------------------------------------


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli_main(argv, cli_services(), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.req("TER-SCR-004")
def test_cli_outcome_option_shows_the_verdict(tmp_path: Path) -> None:
    code, out, _ = _run(["explain", str(SESSION), "--outcome", str(PYTEST_JUNIT)])
    assert code == 0
    assert "outcome          rejected: 2 required check(s) failed" in out
    assert "- failed: test_sample::test_rejects_empty_input · AssertionError" in out
    code, out, _ = _run(
        ["a3", str(SESSION), "--ter", "off", "--json", "--outcome", str(PYTEST_JUNIT)]
    )
    assert code == 0 and json.loads(out)["outcome"]["verdict"] == "rejected"
    code, out, _ = _run(
        ["explain", str(SESSION), "--json", "--outcome", str(PYTEST_JUNIT)]
    )
    assert json.loads(out)["outcome"]["required_checks"] == 5
    code, out, _ = _run(["explain", str(SESSION)])
    assert "outcome          " not in out


def test_cli_outcome_option_reports_missing_and_unreadable_files(
    tmp_path: Path,
) -> None:
    code, _, err = _run(["a3", str(SESSION), "--outcome", str(tmp_path / "none.xml")])
    assert code == 2 and "No such outcome file" in err
    bad = tmp_path / "bad.xml"
    bad.write_text("<html/>", encoding="utf-8")
    code, _, err = _run(["a3", str(SESSION), "--ter", "off", "--outcome", str(bad)])
    assert code == 2 and "Cannot read outcome" in err and "root element <html>" in err


@pytest.mark.req("TER-OUT-007")
@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("0.25", 0.25),
        ("1,234.5", 1234.5),  # comma with a point: thousands separator
        ("0,010", 0.01),  # lone comma: decimal comma
        ("Infinity", None),
        ("-inf", None),
        ("nan", None),
        ("abc", None),
        ("-1", None),
    ],
)
def test_junit_durations_are_finite_and_read_decimal_commas(
    tmp_path: Path, raw: str, seconds: float | None
) -> None:
    path = tmp_path / "r.xml"
    path.write_text(
        f'<testsuite><testcase name="t" time="{raw}"/></testsuite>', encoding="utf-8"
    )
    evidence = JUnitOutcomeSource().outcome(path)
    assert evidence is not None
    assert evidence.checks[0].seconds == seconds
    assert json.loads(json.dumps(evidence.checks[0].as_dict(), allow_nan=False))


@pytest.mark.req("TER-SCR-004", "TER-OUT-004")
def test_an_accepted_verdict_shows_its_passing_checks_and_their_sources() -> None:
    from ter.adapters.driving.cli import format_outcome

    green = OutcomeEvidence(
        "run-1",
        "test",
        (CheckEvidence("a", P, "r.xml#testcase-1"), CheckEvidence("b", P, "r.xml#2")),
    )
    judged = _explain("run-1", {"run-1": green})
    assert judged.outcome is not None and judged.outcome.accepted
    text = format_outcome(judged.outcome, Path("r.xml"))
    assert "  - passed: a [r.xml#testcase-1]\n" in text
    assert "  - passed: b [r.xml#2]\n" in text
    page = render_a3_html(judged.a3)
    assert (
        '<td><span class="tag ok">passed</span></td><td><code>a</code></td>'
        "<td><code>r.xml#testcase-1</code>" in page
    )


@pytest.mark.req("TER-SCR-004", "TER-OUT-004")
def test_outcome_lists_open_checks_first_and_folds_the_rest() -> None:
    from ter.adapters.driving.cli import OUTCOME_ROWS, format_outcome

    checks = [(f"ok{i:02}", P) for i in range(20)] + [("broken", F)]
    judged = _explain("run-1", {"run-1": _ev(*checks)})
    assert judged.outcome is not None
    text = format_outcome(judged.outcome, Path("r.xml"))
    rows = [line for line in text.splitlines() if line.startswith("  - ")]
    assert rows[0] == "  - failed: broken [r#20]"
    assert len(rows) == OUTCOME_ROWS
    assert f"… {21 - OUTCOME_ROWS} more (--json lists every check)" in text
    page = render_a3_html(judged.a3)
    assert "Show the other 13 checks</summary>" in page
    assert page.index("<code>broken</code>") < page.index("<code>ok00</code>")
    assert "<code>ok19</code>" in page  # every check is on the page


@pytest.mark.req("TER-SCR-004")
def test_an_incomplete_outcome_and_skipped_check_are_not_shown_as_failures() -> None:
    judged = _explain("run-1", {"run-1": _ev(("a", P), ("b", S))})
    assert judged.outcome is not None
    assert judged.outcome.verdict is Verdict.INCOMPLETE
    page = render_a3_html(judged.a3)
    assert '<li class="verdict unsure">Outcome incomplete</li>' in page
    assert '<span class="tag warn">skipped</span>' in page
    box = page[page.index('id="s-outcome"') :]
    assert '<span class="tag waste">' not in box[: box.index("</section>")]
    rejected = render_a3_html(_explain(str(PYTEST_JUNIT)).a3)
    assert '<li class="verdict bad">Outcome rejected</li>' in rejected
    assert '<span class="tag waste">failed</span>' in rejected
