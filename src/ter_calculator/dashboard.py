"""Live dashboard for TER monitoring using Rich."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ter_calculator.real_time import TERSignal

from .rich_components import (
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


def format_tokens_per_minute(total_tokens: int, duration_seconds: float) -> str:
    """Calculate and format tokens per minute rate."""
    if duration_seconds < 10:
        return "calculating..."
    rate = (total_tokens / duration_seconds) * 60
    return f"{rate:,.0f} tok/min"


def create_dashboard_renderable(signal: TERSignal, recent_ter_values: list[float] | None = None):
    """Create a Rich renderable object for live dashboard display."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text
    from datetime import datetime, timezone

    # Extract data from signal
    waste_pct = (signal.waste_tokens / signal.total_tokens * 100) if signal.total_tokens > 0 else 0
    session_short = signal.session_id[:8] if len(signal.session_id) > 8 else signal.session_id
    live_indicator = "🟢 LIVE" if signal.is_live else "🔵 HISTORY"

    # Build subtitle line: Drift | Messages | Duration | Rate
    drift_value = signal.drift.value if hasattr(signal.drift, 'value') else str(signal.drift)
    drift_arrow = "↑" if drift_value == "improving" else "↓" if drift_value == "degrading" else "→"
    drift_color = "green" if drift_value == "improving" else "red" if drift_value == "degrading" else "yellow"

    duration_str = format_duration(signal.session_duration_seconds)
    rate_str = format_tokens_per_minute(signal.total_tokens, signal.session_duration_seconds)

    subtitle_parts = [
        ("Drift: ", "bold"),
        (f"{drift_value} {drift_arrow}", drift_color),
        ("  │  ", "dim"),
        ("Messages: ", "bold"),
        (f"{signal.message_index}", ""),
        ("  │  ", "dim"),
        ("Active: ", "bold"),
        (duration_str, ""),
        ("  │  ", "dim"),
        ("Rate: ", "bold"),
        (rate_str, "dim"),
    ]
    subtitle_text = Text.assemble(*subtitle_parts)

    # Create header using shared component
    title = f"TER Live Monitor — Session: {session_short} — {live_indicator}"
    header_panel = create_ter_header_panel(
        ter_score=signal.aggregate_ter,
        waste_pct=waste_pct,
        session_id="",  # Don't use session_id as title, we have custom title
        cost_usd=signal.estimated_cost_usd if signal.estimated_cost_usd > 0 else None,
        waste_cost_usd=signal.estimated_waste_cost_usd if signal.estimated_waste_cost_usd > 0 else None,
        subtitle_text=str(subtitle_text),  # Pass assembled subtitle
    )
    # Override title
    header_panel.title = title

    # Create phases table using shared component
    phases_table = create_phases_table(
        phase_scores=signal.phase_ter,
        show_bars=True,
    )

    # Create tokens table using shared component
    tokens_table = create_tokens_table(
        total_tokens=signal.total_tokens,
        aligned_tokens=signal.aligned_tokens,
        waste_tokens=signal.waste_tokens,
        input_tokens=signal.total_input_tokens if signal.total_input_tokens > 0 else None,
        cache_read_tokens=signal.cache_read_tokens if signal.cache_read_tokens > 0 else None,
        cache_hit_rate=signal.cache_hit_rate if signal.cache_hit_rate > 0 else None,
    )

    # Build element list
    elements = [header_panel, phases_table, tokens_table]

    # Add context section if applicable
    context_section = create_context_section(
        growth_rate=signal.context_growth_rate,
        message_count=signal.message_index,
        bloat_detected=signal.context_bloat_detected,
    )
    if context_section:
        elements.append(context_section)

    # Add waste patterns if available
    if signal.waste_sources:
        waste_section = create_waste_patterns_section(
            waste_sources=signal.waste_sources,
            top_n=3,
        )
        if waste_section:
            elements.append(waste_section)

    # Add warnings if present
    if signal.warnings:
        warnings_section = create_warnings_section(signal.warnings)
        if warnings_section:
            elements.append(warnings_section)

    # Add tools section if any tool calls were made
    tools_section = create_tools_section(
        total_tool_calls=signal.total_tool_calls,
        unique_tool_types=signal.unique_tool_types,
        waste_tool_tokens=signal.waste_sources.get("tool_use", 0),
    )
    if tools_section:
        elements.append(tools_section)

    # Add TER trend sparkline if we have history
    if recent_ter_values and len(recent_ter_values) > 1:
        sparkline = format_sparkline(recent_ter_values, width=20)
        color = ter_color(signal.aggregate_ter)

        trend_text = Text.assemble(
            ("Recent TER: ", "bold"),
            (sparkline, color),
            (f"  ({signal.aggregate_ter:.2f})", color),
        )
        elements.append(trend_text)

        if signal.has_thinking_blocks:
            elements.append(Text(
                "Output tokens excludes extended thinking",
                style="dim italic",
            ))

    # Add footer with timestamp
    update_time = datetime.fromtimestamp(signal.timestamp, tz=timezone.utc).astimezone().strftime("%H:%M:%S")
    footer_text = Text(f"Last update: {update_time}", style="dim")
    footer_panel = Panel(footer_text, border_style="dim", expand=False)
    elements.append(footer_panel)

    return Group(*elements)


# Legacy string-based function for backward compatibility (if needed elsewhere)
def format_dashboard(signal: TERSignal) -> str:
    """Format dashboard as string (for non-Live use). Use create_dashboard_renderable for Live."""
    from rich.console import Console
    import io

    renderable = create_dashboard_renderable(signal)
    buf = io.StringIO()
    console = Console(file=buf, width=88, legacy_windows=False)
    console.print(renderable)
    return buf.getvalue()


def format_dashboard_with_history(signal: TERSignal, recent_ter_values: list[float]):
    """Create dashboard renderable with TER history."""
    return create_dashboard_renderable(signal, recent_ter_values)
