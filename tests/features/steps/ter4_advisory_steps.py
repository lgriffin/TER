"""Step definitions for TER 4's L4 Advisory policies (advisory_policies.feature).

The steps drive the real policy file parser, approval record and Intervention
Engine; the requirement each scenario verifies is its ``@req:`` tag.
"""

from __future__ import annotations

import copy
from datetime import date
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from ter.domain.intervention import Outcome, Signal, Verdict
from ter.domain.maturity import Maturity
from ter.domain.policy_set import POLICY_SCHEMA, PolicySet, UnapprovedPolicyError

scenarios("../ter4/advisory_policies.feature")

DAY = date(2026, 10, 14)


@pytest.fixture
def world() -> dict[str, Any]:
    return {"ceiling": Maturity.ADVISORY, "verdicts": []}


def _ids(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _signal(position: int, confidence: float, evidence: tuple[str, ...]) -> Signal:
    return Signal(
        id=f"repeated_exploration:{position}",
        detector="repeated_exploration",
        waste="motion",
        kind="waste",
        confidence=confidence,
        title="Re-read src/a.py",
        subject="src/a.py",
        evidence=evidence,
        rule="0.85 when the same read returned identical output",
        position=position,
    )


def _apply(world: dict[str, Any], *signals: Signal) -> None:
    engine = world["policies"].engine(world["ceiling"])
    for signal in signals:
        world["verdicts"].extend(engine.apply(signal))


@given(
    parsers.parse('a policy "{policy_id}" that warns "{message}" on {detector}'),
    target_fixture="document",
)
def a_policy(policy_id: str, message: str, detector: str) -> dict[str, Any]:
    return {
        "schema": POLICY_SCHEMA,
        "eligibility": {"detectors": [], "basis": "judged in the feature"},
        "policies": [
            {
                "id": policy_id,
                "detectors": [detector],
                "advice": {"kind": "warning", "message": message},
                "actions": ["advise"],
            }
        ],
    }


@given(
    parsers.parse("the policy stays silent for {events:d} tool events after it advises")
)
def cooldown(document: dict[str, Any], events: int) -> None:
    document["policies"][0]["cooldown_tool_events"] = events


@given(parsers.parse("calibration marks {detector} eligible"))
def eligible(document: dict[str, Any], detector: str) -> None:
    document["eligibility"]["detectors"] = [detector]


@given("calibration marks no detector eligible")
def none_eligible(document: dict[str, Any]) -> None:
    document["eligibility"]["detectors"] = []


@given("the policy file is not approved")
def not_approved(world: dict[str, Any], document: dict[str, Any]) -> None:
    world["policies"] = PolicySet.from_mapping(document)


@given(parsers.parse('the policy file is approved by "{person}"'))
def approved(world: dict[str, Any], document: dict[str, Any], person: str) -> None:
    world["policies"] = PolicySet.from_mapping(document).approve(person, DAY)


@given(parsers.parse("the maturity ceiling is L{level:d}"))
def ceiling(world: dict[str, Any], level: int) -> None:
    world["ceiling"] = Maturity(level)


@when("TER builds the intervention engine")
def build_engine(world: dict[str, Any]) -> None:
    try:
        world["policies"].engine(world["ceiling"])
    except UnapprovedPolicyError as refused:
        world["refused"] = str(refused)


@when(parsers.parse('someone edits the advice to "{message}"'))
def edit_advice(world: dict[str, Any], message: str) -> None:
    edited = copy.deepcopy(world["policies"].as_dict())
    edited["policies"][0]["advice"]["message"] = message
    world["policies"] = PolicySet.from_mapping(edited)


@when(
    parsers.parse(
        "a repeated_exploration signal with confidence {confidence:f} "
        'cites events "{events}"'
    )
)
def signal_with_events(world: dict[str, Any], confidence: float, events: str) -> None:
    _apply(world, _signal(1, confidence, _ids(events)))


@when(
    parsers.parse(
        "a repeated_exploration signal with confidence {confidence:f} cites no events"
    )
)
def signal_without_events(world: dict[str, Any], confidence: float) -> None:
    _apply(world, _signal(1, confidence, ()))


@when(
    parsers.parse(
        "repeated_exploration signals arrive at tool events {first:d}, {second:d} "
        "and {third:d}"
    )
)
def signals_at(world: dict[str, Any], first: int, second: int, third: int) -> None:
    _apply(world, *(_signal(n, 0.85, ("e1", "e2")) for n in (first, second, third)))


@then(parsers.parse('TER refuses with "{text}"'))
def refuses(world: dict[str, Any], text: str) -> None:
    assert text in world.get("refused", ""), world.get("refused")


@then(parsers.parse('the policy file\'s approval state is "{state}"'))
def approval_state(world: dict[str, Any], state: str) -> None:
    assert world["policies"].state.value == state


@then(parsers.parse('building the intervention engine is refused with "{text}"'))
def engine_refused(world: dict[str, Any], text: str) -> None:
    with pytest.raises(UnapprovedPolicyError, match=text):
        world["policies"].engine(world["ceiling"])


def _only(world: dict[str, Any]) -> Verdict:
    [verdict] = world["verdicts"]
    assert isinstance(verdict, Verdict)
    return verdict


@then(parsers.parse('TER advises "{message}"'))
def advises(world: dict[str, Any], message: str) -> None:
    verdict = _only(world)
    assert verdict.outcome is Outcome.ADVISED
    assert verdict.advice is not None and verdict.advice.message == message


@then(parsers.parse('the advice shows the {detector} signal and events "{events}"'))
def shows_evidence(world: dict[str, Any], detector: str, events: str) -> None:
    verdict = _only(world)
    assert verdict.signal.detector == detector
    assert verdict.signal.evidence == _ids(events)


@then(parsers.parse('the outcome is "{outcome}"'))
def outcome_is(world: dict[str, Any], outcome: str) -> None:
    assert _only(world).outcome is Outcome(outcome)


@then(parsers.parse('the outcomes are "{outcomes}"'))
def outcomes_are(world: dict[str, Any], outcomes: str) -> None:
    got = [v.outcome for v in world["verdicts"]]
    assert got == [Outcome(o) for o in _ids(outcomes)]
