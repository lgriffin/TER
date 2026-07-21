from __future__ import annotations

import json
from pathlib import Path

from ter_calculator.intervention_policy import (
    ComplianceResult,
    InterventionAction,
    MetricSnapshot,
    PolicyConfig,
    PolicyDecision,
    PolicyState,
    append_intervention_outcome,
    build_recovery_instruction,
    classify_effect,
    consume_pending_intervention,
    evaluate_policy,
    new_intervention_record,
    write_pending_intervention,
)


def snap(ts: float, ter: float, waste: float, repeated: int = 0) -> MetricSnapshot:
    return MetricSnapshot(ts, ter, waste, repeated_tool_calls=repeated)


def test_transient_dip_does_not_trigger() -> None:
    state = PolicyState()
    for item in [snap(1, 0.80, 0.10), snap(2, 0.81, 0.10), snap(3, 0.80, 0.11)]:
        evaluate_policy(item, state)
    decision = evaluate_policy(snap(4, 0.65, 0.30), state)
    assert decision.action is InterventionAction.NONE


def test_sustained_degradation_triggers_refresh() -> None:
    state = PolicyState()
    cfg = PolicyConfig(degraded_windows_required=2, severe_windows_required=9)
    for item in [snap(1, 0.80, 0.10), snap(2, 0.80, 0.10), snap(3, 0.80, 0.10)]:
        evaluate_policy(item, state, cfg)
    assert (
        evaluate_policy(snap(4, 0.65, 0.30), state, cfg).action
        is InterventionAction.NONE
    )
    decision = evaluate_policy(snap(5, 0.54, 0.31), state, cfg)
    assert decision.action is InterventionAction.REFRESH_CONTEXT


def test_severe_degradation_triggers_replan_and_cooldown() -> None:
    state = PolicyState()
    cfg = PolicyConfig(severe_windows_required=1, degraded_windows_required=1)
    for item in [snap(1, 0.90, 0.10), snap(2, 0.90, 0.10), snap(3, 0.90, 0.10)]:
        evaluate_policy(item, state, cfg)
    first = evaluate_policy(snap(10, 0.50, 0.50), state, cfg)
    assert first.action is InterventionAction.REPLAN
    second = evaluate_policy(snap(20, 0.40, 0.60), state, cfg)
    assert second.action is InterventionAction.NONE


def test_recovery_resets_counter() -> None:
    state = PolicyState()
    cfg = PolicyConfig(degraded_windows_required=2, severe_windows_required=9)
    for item in [
        snap(1, 0.80, 0.10),
        snap(2, 0.80, 0.10),
        snap(3, 0.80, 0.10),
        snap(4, 0.65, 0.30),
    ]:
        evaluate_policy(item, state, cfg)
    evaluate_policy(snap(5, 0.82, 0.10), state, cfg)
    assert state.consecutive_degraded_windows == 0


def test_pending_intervention_consumed_once(tmp_path: Path) -> None:
    snapshot = snap(100, 0.40, 0.50)
    decision = PolicyDecision(InterventionAction.REPLAN, "bad", cooldown_seconds=60)
    record = new_intervention_record("s1", decision, snapshot)
    path = tmp_path / "pending.json"
    write_pending_intervention(path, record, decision)
    assert consume_pending_intervention(path, now=110) is not None
    assert consume_pending_intervention(path, now=111) is None


def test_expired_pending_intervention_is_ignored(tmp_path: Path) -> None:
    snapshot = snap(100, 0.40, 0.50)
    decision = PolicyDecision(InterventionAction.REPLAN, "bad", cooldown_seconds=10)
    record = new_intervention_record("s1", decision, snapshot)
    path = tmp_path / "pending.json"
    write_pending_intervention(path, record, decision)
    assert consume_pending_intervention(path, now=111) is None


def test_effect_classification_and_raw_outcome(tmp_path: Path) -> None:
    before = snap(1, 0.40, 0.50)
    after = snap(2, 0.55, 0.30)
    compliance = ComplianceResult(True, True, ["plan"], 0.8)
    assert classify_effect(before, after, compliance) == "improved"
    decision = PolicyDecision(InterventionAction.REFRESH_CONTEXT, "degraded")
    record = new_intervention_record("s", decision, before)
    path = tmp_path / "outcomes.jsonl"
    row = append_intervention_outcome(
        path, record=record, post=after, compliance=compliance
    )
    assert row["effect"] == "improved"
    assert json.loads(path.read_text())["deltas"]["ter"] > 0


def test_instruction_differs_by_action() -> None:
    replan = build_recovery_instruction(
        PolicyDecision(
            InterventionAction.REPLAN,
            "x",
            metrics={"ter": 0.4, "baseline_ter": 0.7, "waste_ratio": 0.4},
        )
    )
    refresh = build_recovery_instruction(
        PolicyDecision(
            InterventionAction.REFRESH_CONTEXT,
            "x",
            metrics={"ter": 0.5, "baseline_ter": 0.7, "waste_ratio": 0.3},
        )
    )
    assert "Replan Required" in replan
    assert "Context Refresh" in refresh


def test_effect_classification_all_branches_and_boundaries() -> None:
    baseline = snap(1, 0.50, 0.40)
    followed = ComplianceResult(True, True)
    assert classify_effect(baseline, snap(2, 0.581, 0.319), followed) == "improved"
    assert classify_effect(baseline, snap(2, 0.42, 0.40), followed) == "regressed"
    assert classify_effect(baseline, snap(2, 0.5799, 0.3201), followed) == "neutral"
    acknowledged = ComplianceResult(True, False)
    assert (
        classify_effect(baseline, snap(2, 0.51, 0.39), acknowledged)
        == "acknowledged_not_followed"
    )
    ignored = ComplianceResult(False, False)
    assert classify_effect(baseline, snap(2, 0.51, 0.39), ignored) == "ignored"
