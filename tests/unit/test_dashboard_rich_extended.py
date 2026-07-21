import time

from ter_calculator.dashboard import (
    create_dashboard_renderable,
    format_dashboard,
    format_dashboard_with_history,
    format_tokens_per_minute,
)
from ter_calculator.real_time import DriftDirection, TERSignal, WarningLevel
from ter_calculator.rich_components import (
    create_context_section,
    create_phases_table,
    create_ter_header_panel,
    create_tokens_table,
    create_tools_section,
    create_warnings_section,
    create_waste_patterns_section,
    format_duration,
    format_sparkline,
    ter_color,
)


def signal(**kw):
    base = dict(
        session_id="123456789",
        timestamp=time.time(),
        aggregate_ter=0.75,
        raw_ratio=0.7,
        message_index=5,
        total_tokens=1000,
        aligned_tokens=700,
        waste_tokens=300,
        drift=DriftDirection.STABLE,
        drift_magnitude=0.01,
        warning_level=WarningLevel.INFO,
        phase_ter={"reasoning": 0.8, "tool_use": 0.7, "generation": 0.75},
        warnings=["careful"],
        waste_sources={"reasoning_loop": 50, "tool_use": 20},
        total_tool_calls=4,
        unique_tool_types=2,
        session_duration_seconds=120,
        estimated_cost_usd=1.2,
        estimated_waste_cost_usd=0.3,
        total_input_tokens=200,
        cache_read_tokens=100,
        cache_hit_rate=0.5,
        context_growth_rate=2.0,
        context_bloat_detected=True,
        is_live=True,
        has_thinking_blocks=True,
    )
    base.update(kw)
    return TERSignal(**base)


def test_small_formatters():
    assert format_tokens_per_minute(100, 5) == "calculating..."
    assert "tok/min" in format_tokens_per_minute(100, 60)
    assert ter_color(0.9) != ter_color(0.2)
    assert format_sparkline([], 10) == ""
    assert len(format_sparkline([0.1, 0.5, 0.9], 2)) == 2
    assert format_duration(5) and format_duration(65) and format_duration(3700)


def test_component_branches_render():
    assert create_ter_header_panel(0.8, 10, "s") is not None
    assert create_phases_table({"reasoning": 0.2}, show_bars=False) is not None
    assert create_tokens_table(0, 0, 0) is not None
    assert create_context_section(1.0, 1, False) is None
    assert create_context_section(2.0, 10, True) is not None
    assert create_warnings_section([]) is None
    assert create_warnings_section(["x"]) is not None
    assert create_tools_section(0, 0, 0) is None
    assert create_tools_section(2, 1, 1) is not None
    assert create_waste_patterns_section({}, 3) is None
    assert create_waste_patterns_section({"x": 10}, 3) is not None


def test_dashboard_full_and_minimal():
    s = signal()
    assert create_dashboard_renderable(s, [0.5, 0.6, 0.75]) is not None
    text = format_dashboard(s)
    assert "TER Live Monitor" in text and "careful" in text
    assert format_dashboard_with_history(s, [0.6, 0.7]) is not None
    minimal = signal(
        total_tokens=0,
        waste_tokens=0,
        warnings=[],
        waste_sources={},
        total_tool_calls=0,
        unique_tool_types=0,
        session_duration_seconds=1,
        estimated_cost_usd=0,
        estimated_waste_cost_usd=0,
        total_input_tokens=0,
        cache_read_tokens=0,
        cache_hit_rate=0,
        context_growth_rate=1,
        context_bloat_detected=False,
        is_live=False,
        drift=DriftDirection.IMPROVING,
    )
    assert "HISTORY" in format_dashboard(minimal)
