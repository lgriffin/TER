from pathlib import Path
import json
from ter_calculator.closed_loop import (
    analyze_trends,
    build_effectiveness_dashboard_html,
)
from ter_calculator.intervention_policy import (
    ComplianceResult,
    InterventionAction,
    MetricSnapshot,
    PolicyDecision,
    append_intervention_outcome,
    new_intervention_record,
    PolicyConfig,
)
from ter_calculator.threshold_tuning import (
    recommend_policy_config,
    save_tuned_policy_config,
    load_tuned_policy_config,
)


def test_cost_delta_and_trend_dashboard(tmp_path: Path):
    outcomes = tmp_path / "outcomes.jsonl"
    lessons = tmp_path / "lessons.jsonl"
    before = MetricSnapshot(1, 0.4, 0.5, context_tokens=10000, cost_per_1k_tokens=0.01)
    record = new_intervention_record(
        "s", PolicyDecision(InterventionAction.REPLAN, "x"), before
    )
    after = MetricSnapshot(2, 0.6, 0.2, context_tokens=8000, cost_per_1k_tokens=0.01)
    row = append_intervention_outcome(
        outcomes,
        record=record,
        post=after,
        compliance=ComplianceResult(True, True, confidence=0.9),
    )
    assert row["deltas"]["estimated_cost_waste_usd"] > 0
    trends = analyze_trends(lessons, outcome_path=outcomes)
    assert trends["total_estimated_cost_saved_usd"] > 0
    html = build_effectiveness_dashboard_html(trends)
    assert "estimated saved" in html and "replan" in html


def test_tuning_is_bounded_deterministic_and_persisted(tmp_path: Path):
    current = PolicyConfig()
    effectiveness = {
        "replan": {"issued": 10, "improvement_rate": 0.8, "override_rate": 0.1},
        "refresh_context": {
            "issued": 10,
            "improvement_rate": 0.2,
            "override_rate": 0.6,
        },
    }
    a = recommend_policy_config(effectiveness, current)
    b = recommend_policy_config(effectiveness, current)
    assert a == b and a != current
    assert a.ter_drop_replan >= 0.10 and a.waste_ratio_warning <= 0.70
    save_tuned_policy_config(tmp_path, a)
    assert load_tuned_policy_config(tmp_path) == a


def test_tuning_ignores_small_samples():
    current = PolicyConfig()
    assert (
        recommend_policy_config(
            {"replan": {"issued": 2, "improvement_rate": 1, "override_rate": 0}},
            current,
        )
        == current
    )


def test_empty_dashboard_is_friendly():
    html = build_effectiveness_dashboard_html(
        {"intervention_effectiveness": {}, "scenarios": []}
    )
    assert "No intervention outcome data yet" in html
