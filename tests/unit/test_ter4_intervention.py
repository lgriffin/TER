"""L4 Advisory: Intervention Engine signals, policies and guards."""

from __future__ import annotations

from dataclasses import replace

import pytest
from ter4_lean_builder import Script

from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain.intervention import (
    ADVICE_CONFIDENCE,
    Advice,
    AdviceKind,
    Eligibility,
    InterventionEngine,
    Outcome,
    Policy,
    PolicyError,
    Signal,
    decide,
)
from ter.domain.lean import explain
from ter.domain.lean.signals import detector_signals
from ter.domain.maturity import Maturity

ADVICE = Advice(AdviceKind.WARNING, "Re-read only after the file changed.")
ELIGIBLE = Eligibility(frozenset({"repeated_exploration"}), "judged 10 Oct 2026")


def signal(
    n: int = 1,
    *,
    position: int | None = None,
    detector: str = "repeated_exploration",
    confidence: float = 0.85,
    evidence: tuple[str, ...] = ("e1", "e2"),
) -> Signal:
    return Signal(
        id=f"{detector}:{n}",
        detector=detector,
        waste="motion",
        kind="waste",
        confidence=confidence,
        title="Re-read src/a.py",
        subject="src/a.py",
        evidence=evidence,
        rule="0.85 when the same read returned identical output",
        position=n if position is None else position,
    )


def policy(**kw: object) -> Policy:
    base = Policy("no-rereads", frozenset({"repeated_exploration"}), ADVICE)
    return replace(base, **kw)  # type: ignore[arg-type]


def outcomes(*signals: Signal, **kw: object) -> list[Outcome]:
    engine = InterventionEngine((policy(**kw),), ELIGIBLE)
    return [v.outcome for s in signals for v in engine.apply(s)]


# --- TER-INT-003 / TER-INT-014: advice with its signal and evidence --------


@pytest.mark.req("TER-INT-003", "TER-INT-014")
def test_a_matching_signal_gets_the_policys_advice_with_its_evidence() -> None:
    [verdict] = InterventionEngine((policy(),), ELIGIBLE).apply(signal())
    assert verdict.advised and verdict.advice == ADVICE
    assert verdict.signal.evidence == ("e1", "e2") and verdict.policy == "no-rereads"
    assert "repeated_exploration" in verdict.reason


def test_a_signal_no_policy_answers_gets_no_verdict() -> None:
    engine = InterventionEngine(
        (policy(),), Eligibility(frozenset({"regeneration"}), "judged")
    )
    assert engine.apply(signal(detector="regeneration")) == ()


# --- TER-INT-002: the ceiling ------------------------------------------------


@pytest.mark.req("TER-INT-002")
@pytest.mark.parametrize("ceiling", [Maturity.MEASURED, Maturity.GROUNDED])
def test_below_l4_nothing_is_advised(ceiling: Maturity) -> None:
    [verdict] = InterventionEngine((policy(),), ELIGIBLE, ceiling).apply(signal())
    assert verdict.outcome is Outcome.BELOW_CEILING and verdict.advice is None


@pytest.mark.req("TER-INT-002")
def test_boundary_at_l4_advice_is_given() -> None:
    [verdict] = InterventionEngine((policy(),), ELIGIBLE, Maturity.ADVISORY).apply(
        signal()
    )
    assert verdict.advised


# --- TER-INT-007: evidence beyond token counts ------------------------------


@pytest.mark.req("TER-INT-007")
def test_a_signal_citing_no_event_is_never_advised() -> None:
    assert outcomes(signal(evidence=())) == [Outcome.NO_EVIDENCE]


@pytest.mark.req("TER-INT-007")
def test_boundary_a_policy_may_ask_for_more_evidence() -> None:
    assert outcomes(signal(evidence=("e1",)), min_evidence=2) == [Outcome.NO_EVIDENCE]
    assert outcomes(signal(evidence=("e1", "e2")), min_evidence=2) == [Outcome.ADVISED]


@pytest.mark.req("TER-INT-007")
def test_no_policy_can_fire_on_a_token_count_alone() -> None:
    with pytest.raises(PolicyError, match="token count"):
        policy(min_evidence=0)


@pytest.mark.parametrize(
    "bad",
    [
        {"id": ""},
        {"detectors": frozenset()},
        {"min_confidence": 1.5},
        {"min_signals": 0},
        {"cooldown_tool_events": -1},
        {"advice": Advice(AdviceKind.WARNING, " ")},
    ],
)
def test_malformed_policies_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(PolicyError):
        policy(**bad)


def test_policy_ids_are_unique() -> None:
    with pytest.raises(PolicyError, match="unique"):
        InterventionEngine((policy(), policy()), ELIGIBLE)


# --- TER-INT-010: calibration eligibility -----------------------------------


@pytest.mark.req("TER-INT-010")
def test_a_detector_not_marked_eligible_is_never_advised() -> None:
    engine = InterventionEngine((policy(),), Eligibility(frozenset(), ""))
    [verdict] = engine.apply(signal())
    assert verdict.outcome is Outcome.INELIGIBLE and "calibration" in verdict.reason


# --- uncertain signals count as waste but never intervene (ADR 0006) --------


@pytest.mark.req("TER-INT-021")
def test_an_uncertain_signal_is_never_advised_even_if_the_policy_allows() -> None:
    assert outcomes(signal(confidence=0.69), min_confidence=0.5) == [
        Outcome.LOW_CONFIDENCE
    ]


def test_boundary_confidence_at_the_floor_is_advised() -> None:
    assert ADVICE_CONFIDENCE == 0.70
    assert outcomes(signal(confidence=0.70)) == [Outcome.ADVISED]
    assert outcomes(signal(confidence=0.80), min_confidence=0.85) == [
        Outcome.LOW_CONFIDENCE
    ]


# --- TER-INT-016: cooldown and hysteresis -----------------------------------


@pytest.mark.req("TER-INT-016")
def test_a_policy_is_silent_during_its_cooldown() -> None:
    got = outcomes(
        signal(1, position=5), signal(2, position=14), cooldown_tool_events=10
    )
    assert got == [Outcome.ADVISED, Outcome.COOLDOWN]


@pytest.mark.req("TER-INT-016")
def test_boundary_the_cooldown_ends_after_its_tool_events() -> None:
    got = outcomes(
        signal(1, position=5), signal(2, position=15), cooldown_tool_events=10
    )
    assert got == [Outcome.ADVISED, Outcome.ADVISED]


@pytest.mark.req("TER-INT-016")
def test_zero_cooldown_advises_every_signal() -> None:
    got = outcomes(signal(1), signal(2), cooldown_tool_events=0)
    assert got == [Outcome.ADVISED, Outcome.ADVISED]


@pytest.mark.req("TER-INT-016")
def test_hysteresis_waits_for_enough_matching_signals() -> None:
    got = outcomes(
        signal(1, position=1),
        signal(2, position=2),
        signal(3, position=3),
        min_signals=3,
        cooldown_tool_events=0,
    )
    assert got == [Outcome.BELOW_HYSTERESIS, Outcome.BELOW_HYSTERESIS, Outcome.ADVISED]


@pytest.mark.req("TER-INT-016")
def test_suppressed_signals_do_not_count_toward_hysteresis() -> None:
    got = outcomes(
        signal(1, confidence=0.5),
        signal(2, evidence=()),
        signal(3),
        min_signals=2,
    )
    assert got == [
        Outcome.LOW_CONFIDENCE,
        Outcome.NO_EVIDENCE,
        Outcome.BELOW_HYSTERESIS,
    ]


def test_policies_keep_separate_cooldowns() -> None:
    other = Policy(
        "rereads-again",
        frozenset({"repeated_exploration"}),
        ADVICE,
        cooldown_tool_events=0,
    )
    engine = InterventionEngine((policy(cooldown_tool_events=10), other), ELIGIBLE)
    engine.apply(signal(1))
    second = engine.apply(signal(2))
    assert [v.outcome for v in second] == [Outcome.COOLDOWN, Outcome.ADVISED]


# --- TER-INT-006: the engine reads only signals; live equals batch ----------


@pytest.mark.req("TER-INT-006")
def test_applying_a_signal_twice_changes_nothing() -> None:
    engine = InterventionEngine((policy(cooldown_tool_events=0),), ELIGIBLE)
    assert len(engine.apply(signal())) == 1
    assert engine.apply(signal()) == ()


@pytest.mark.req("TER-INT-006")
def test_batch_equals_incremental_in_order_of_position() -> None:
    signals = [signal(n, position=p) for n, p in [(3, 30), (1, 2), (2, 9), (4, 31)]]
    batch = decide(signals, [policy()], ELIGIBLE)
    engine = InterventionEngine((policy(),), ELIGIBLE)
    live = tuple(
        v for s in sorted(signals, key=lambda s: s.position) for v in engine.apply(s)
    )
    assert batch == live
    assert [v.outcome for v in batch] == [
        Outcome.ADVISED,
        Outcome.COOLDOWN,
        Outcome.ADVISED,
        Outcome.COOLDOWN,
    ]


@pytest.mark.req("TER-INT-006")
def test_detector_signals_carry_evidence_rule_and_tool_event_position() -> None:
    s = Script()
    s.prompt("look at a.py")
    s.read("src/a.py", "def a(): pass")
    s.bash("ls", "src")
    s.read("src/a.py", "def a(): pass")
    s.say("done")
    analysis = explain(s.events, RegexTokenizer())
    [found] = [
        x for x in detector_signals(analysis) if x.detector == "repeated_exploration"
    ]
    finding = next(f for f in analysis.findings if f.id == found.id)
    assert found.evidence == finding.evidence and found.has_evidence
    assert found.confidence == finding.confidence and found.waste == "motion"
    assert "identical output" in found.rule
    assert found.position == 3  # the third tool call is the re-read
    [verdict] = decide([found], [policy()], ELIGIBLE)
    assert verdict.advised


@pytest.mark.req("TER-INT-010")
def test_an_eligibility_list_needs_its_judged_basis() -> None:
    with pytest.raises(PolicyError, match="basis"):
        Eligibility(frozenset({"repeated_exploration"}), " ")


# --- live re-derivation: a finding can come back stronger under its id -----


@pytest.mark.req("TER-INT-006")
def test_a_finding_confirmed_later_is_judged_again() -> None:
    # Live, a re-read is first seen with its result pending (0.55), then
    # confirmed (0.85) under the same id once the result arrives.
    engine = InterventionEngine((policy(),), ELIGIBLE)
    pending = signal(1, confidence=0.55, evidence=("e1", "e2"))
    confirmed = replace(pending, confidence=0.85, evidence=("e1", "e2", "e3"))
    assert [v.outcome for v in engine.apply(pending)] == [Outcome.LOW_CONFIDENCE]
    assert [v.outcome for v in engine.apply(confirmed)] == [Outcome.ADVISED]
    assert decide([confirmed], [policy()], ELIGIBLE)[0].advised


@pytest.mark.req("TER-INT-016")
def test_an_advised_finding_is_never_advised_again_when_it_changes() -> None:
    engine = InterventionEngine((policy(cooldown_tool_events=0),), ELIGIBLE)
    first = signal(1, confidence=0.80)
    assert engine.apply(first)[0].advised
    assert engine.apply(replace(first, confidence=0.90)) == ()


@pytest.mark.req("TER-INT-016")
def test_a_re_derived_finding_counts_once_toward_hysteresis() -> None:
    engine = InterventionEngine(
        (policy(min_signals=2, cooldown_tool_events=0),), ELIGIBLE
    )
    first = signal(1, confidence=0.80)
    assert engine.apply(first)[0].outcome is Outcome.BELOW_HYSTERESIS
    again = engine.apply(replace(first, confidence=0.90))
    assert again[0].outcome is Outcome.BELOW_HYSTERESIS
    assert engine.apply(signal(2))[0].advised
