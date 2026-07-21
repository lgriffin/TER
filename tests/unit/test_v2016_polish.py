from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

from ter_calculator.closed_loop import (
    _build_cost_trend_chart,
    _build_improvement_bar_chart,
    _weekly_cost_buckets,
    build_effectiveness_dashboard_html,
)
from ter_calculator.intervention_policy import PolicyConfig
from ter_calculator.threshold_tuning import describe_config_changes


def test_improvement_chart_svg_and_empty_state() -> None:
    html = _build_improvement_bar_chart(
        {
            "refresh_context": {"improvement_rate": 0.75},
            "replan": {"improvement_rate": 0.25},
        }
    )
    assert html.startswith("<svg")
    assert html.count('class="bar ') == 2
    assert "75%" in html and "25%" in html
    assert "No intervention data yet" in _build_improvement_bar_chart({})


def test_cost_chart_weekly_buckets_and_single_bucket() -> None:
    rows = [
        {
            "evaluated_at": "2026-01-05T10:00:00+00:00",
            "effect": "improved",
            "deltas": {"estimated_cost_waste_usd": 1.25},
        },
        {
            "evaluated_at": "2026-01-06T10:00:00+00:00",
            "effect": "regressed",
            "deltas": {"estimated_cost_waste_usd": -0.5},
        },
        {
            "evaluated_at": "2026-01-13T10:00:00+00:00",
            "effect": "improved",
            "deltas": {"estimated_cost_waste_usd": 2.0},
        },
    ]
    buckets = _weekly_cost_buckets(rows)
    assert len(buckets) == 2
    assert buckets[0][1:] == (1.25, 0.5)
    assert "polyline" in _build_cost_trend_chart(rows)
    assert "<rect" in _build_cost_trend_chart(rows[:2])
    assert "No dated cost outcomes yet" in _build_cost_trend_chart([])


def test_dashboard_tuning_sections() -> None:
    trends = {
        "intervention_effectiveness": {
            "replan": {"issued": 10, "improvement_rate": 0.8}
        },
        "outcome_rows": [],
    }
    current = PolicyConfig()
    recommended = PolicyConfig(ter_drop_replan=0.18, waste_ratio_replan=0.36)
    changes = describe_config_changes(
        current,
        recommended,
        {"replan": {"issued": 10, "improvement_rate": 0.8, "override_rate": 0.1}},
    )
    html = build_effectiveness_dashboard_html(
        trends,
        tuning_preview={"applied_config": current.__dict__, "changes": changes},
    )
    assert "Applied" in html
    assert "Pending preview (not yet applied)" in html
    assert "10 outcomes" in html
    assert html.count("<svg") >= 1

    no_change = build_effectiveness_dashboard_html(
        trends, tuning_preview={"applied_config": None, "changes": []}
    )
    assert "No changes recommended" in no_change


def test_describe_config_changes_only_reports_changed_fields() -> None:
    current = PolicyConfig()
    recommended = PolicyConfig(ter_drop_warning=0.132)
    changes = describe_config_changes(
        current,
        recommended,
        {
            "refresh_context": {
                "issued": 12,
                "improvement_rate": 0.2,
                "override_rate": 0.6,
            }
        },
    )
    assert [change["field"] for change in changes] == ["ter_drop_warning"]
    assert changes[0]["sample_size"] == 12
    assert "20% improved" in str(changes[0]["reason"])


def test_explicit_default_cli_value_beats_tuned_config(
    monkeypatch, tmp_path: Path
) -> None:
    import ter_calculator.commands.hook as command
    import ter_calculator.hook_monitor as monitor
    import ter_calculator.threshold_tuning as tuning

    captured = {}

    class CapturingConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(monitor, "HookConfig", CapturingConfig)
    monkeypatch.setattr(
        monitor, "load_state", lambda *args: SimpleNamespace(intervention_count=0)
    )
    monkeypatch.setattr(monitor, "save_state", lambda *args: None)
    monkeypatch.setattr(
        tuning,
        "load_tuned_policy_config",
        lambda root: PolicyConfig(ter_drop_warning=0.33),
    )
    monkeypatch.setattr(
        "ter_calculator.intervention.process_intervention_event",
        lambda event, state, config: ([], state, {}),
    )
    monkeypatch.setattr(
        "ter_calculator.intervention.record_tool_result", lambda *args: None
    )
    monkeypatch.setattr(
        command.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": "s",
                    "cwd": str(tmp_path),
                    "hook_event_name": "UserPromptSubmit",
                }
            )
        ),
    )

    args = SimpleNamespace(
        min_repetitive_reads=3,
        min_edit_fragments=3,
        min_repeated_commands=3,
        min_duplicate_calls=2,
        min_denied_calls=2,
        min_reasoning_loops=3,
        reasoning_similarity_threshold=0.9,
        no_bash_antipatterns=False,
        no_project_memory=True,
        memory_index=None,
        memory_limit=4,
        memory_minimum_score=0.18,
        lesson_store=None,
        outcome_store=None,
        policy_mode="suggest",
        ter_drop_warning=0.12,
        ter_drop_replan=None,
        waste_ratio_warning=None,
        waste_ratio_replan=None,
        degraded_windows_required=None,
        refresh_cooldown_seconds=None,
        replan_cooldown_seconds=None,
        state_dir=None,
        cost_per_1k_tokens=0.003,
    )
    assert command._cmd_hook_monitor(args) == 0
    assert captured["ter_drop_warning"] == 0.12
    assert captured["ter_drop_replan"] == 0.20
