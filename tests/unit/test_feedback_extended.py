import json
from types import SimpleNamespace

import pytest

from ter_calculator.feedback import (
    TERHistory,
    TrendDirection,
    _compute_trend_direction,
    check_threshold,
    generate_prompt_hints,
    get_stats_by_tag,
    tag_session,
)


def result(**overrides):
    base = dict(
        session_id="s1",
        aggregate_ter=0.8,
        total_tokens=100,
        waste_tokens=20,
        phase_scores={"reasoning": 0.8, "tool_use": 0.8, "generation": 0.8},
        waste_patterns=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_generate_prompt_hints_deduplicates_and_orders():
    p = SimpleNamespace(pattern_type="reasoning_loop")
    hints = generate_prompt_hints(
        result(
            waste_patterns=[p, p, SimpleNamespace(pattern_type="context_restatement")],
            phase_scores={"reasoning": 0.4, "tool_use": 0.2, "generation": 0.3},
        )
    )
    assert hints[0].estimated_impact == "high"
    assert sum(h.related_pattern_type == "reasoning_loop" for h in hints) == 1
    assert {h.category for h in hints} >= {"reasoning", "tool_use", "generation"}


def test_history_round_trip_summary_filter_and_empty(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "history.json"
    h = TERHistory(path)
    assert h._load() == []
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    assert h._load() == []
    path.write_text("{}", encoding="utf-8")
    assert h._load() == []
    path.unlink()

    times = iter([1.0, 2.0, 3.0])
    monkeypatch.setattr("ter_calculator.feedback.time.time", lambda: next(times))
    h.record(result(session_id="a", aggregate_ter=0.4), "/p", ["x"])
    h.record(result(session_id="b", aggregate_ter=0.8), "/p", [])
    h.record(result(session_id="c", aggregate_ter=0.9), "/q", [])
    assert [e.session_id for e in h.get_trend("/p", last_n=1)] == ["b"]
    s = h.get_summary("/p")
    assert s.session_count == 2 and s.avg_ter == pytest.approx(0.6)
    assert s.trend_direction is TrendDirection.IMPROVING
    with pytest.raises(ValueError):
        h.get_summary("missing")


def test_tag_stats_and_failures(tmp_path):
    path = tmp_path / "h.json"
    path.write_text(
        json.dumps(
            [
                {
                    "session_id": "a",
                    "timestamp": 1,
                    "aggregate_ter": 0.5,
                    "total_tokens": 100,
                    "waste_tokens": 20,
                    "project_path": "/p",
                    "tags": ["old"],
                },
                {
                    "session_id": "b",
                    "timestamp": 2,
                    "aggregate_ter": 0.9,
                    "total_tokens": 0,
                    "waste_tokens": 0,
                    "project_path": "/p",
                    "tags": ["old"],
                },
            ]
        ),
        encoding="utf-8",
    )
    tag_session("a", ["new", "old"], path)
    data = json.loads(path.read_text())
    assert data[0]["tags"] == ["new", "old"]
    stats = get_stats_by_tag("old", path)
    assert stats.session_count == 2 and stats.avg_ter == pytest.approx(0.7)
    with pytest.raises(ValueError):
        tag_session("z", ["x"], path)
    with pytest.raises(ValueError):
        get_stats_by_tag("z", path)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0.5], TrendDirection.STABLE),
        ([0.4, 0.41], TrendDirection.STABLE),
        ([0.2, 0.8], TrendDirection.IMPROVING),
        ([0.8, 0.2], TrendDirection.DECLINING),
    ],
)
def test_trend_direction(values, expected):
    assert _compute_trend_direction(values) is expected


def test_check_threshold_pass_and_multiple_failures():
    ok = check_threshold(result(), 0.7)
    assert ok.passed and "passed" in ok.message
    bad = check_threshold(
        result(aggregate_ter=0.5, phase_scores={"reasoning": 0.4, "tool_use": 0.9}),
        0.6,
        0.5,
    )
    assert not bad.passed
    assert bad.phase_failures == ["reasoning"]
    assert "aggregate TER" in bad.message and "phase failures" in bad.message
