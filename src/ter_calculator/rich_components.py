"""Shared Rich rendering components for TER displays.

This module provides reusable Rich components used by both:
- Post-hoc analysis (formatter.py with TERResult)
- Live monitoring (dashboard.py with TERSignal)

Components accept primitive types (not TERResult/TERSignal) so both can use them.
"""

from __future__ import annotations


def ter_color(value: float) -> str:
    """Return Rich color name for a TER score.

    Args:
        value: TER score (0.0 to 1.0)

    Returns:
        Rich color name: "green", "yellow", or "red"
    """
    if value >= 0.7:
        return "green"
    if value >= 0.4:
        return "yellow"
    return "red"


def format_sparkline(values: list[float], width: int = 10) -> str:
    """Create a text sparkline from TER values using Unicode blocks.

    Args:
        values: List of TER scores to visualize
        width: Maximum number of characters (takes last N values)

    Returns:
        String of Unicode block characters representing the trend
    """
    if not values:
        return ""

    # Take last N values
    recent = values[-width:]
    if len(recent) < 2:
        return "█" * len(recent)

    # Map to Unicode block characters
    blocks = " ▁▂▃▄▅▆▇█"
    min_val = min(recent)
    max_val = max(recent)

    if max_val == min_val:
        # All same value
        return "▄" * len(recent)

    # Normalize to 0-8 range
    normalized = [
        int((v - min_val) / (max_val - min_val) * 8)
        for v in recent
    ]

    return "".join(blocks[n] for n in normalized)


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "5m 30s" or "2h 15m"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m {secs}s"


def create_ter_header_panel(
    ter_score: float,
    waste_pct: float,
    session_id: str = "",
    cost_usd: float | None = None,
    waste_cost_usd: float | None = None,
    subtitle_text: str | None = None,
) -> "Panel":
    """Create header panel with TER score, waste, and cost.

    Args:
        ter_score: Overall TER score (0-1)
        waste_pct: Waste percentage (0-100)
        session_id: Session ID for title (will be truncated if long)
        cost_usd: Total cost in USD (optional)
        waste_cost_usd: Waste cost in USD (optional)
        subtitle_text: Additional line below main metrics (optional)

    Returns:
        Rich Panel object
    """
    from rich.panel import Panel
    from rich.text import Text

    # Build main line: TER | Waste | Cost | Waste $
    parts = [
        ("TER: ", "bold"),
        (f"{ter_score:.2f}", ter_color(ter_score)),
        ("  |  ", "dim"),
        ("Waste: ", "bold"),
        (f"{waste_pct:.1f}%", "red" if waste_pct > 10 else "yellow" if waste_pct > 5 else "green"),
    ]

    if cost_usd is not None:
        parts.extend([
            ("  |  ", "dim"),
            ("Cost: ", "bold"),
            (f"${cost_usd:.2f}", ""),
        ])

    if waste_cost_usd is not None and waste_cost_usd > 0:
        parts.extend([
            ("  |  ", "dim"),
            ("Waste $: ", "bold"),
            (f"${waste_cost_usd:.4f}", "red"),
        ])

    # Add subtitle if provided
    if subtitle_text:
        parts.extend([("\n", ""), (subtitle_text, "")])

    header_text = Text.assemble(*parts)

    # Truncate session ID if long
    title = session_id
    if len(session_id) > 20:
        title = session_id[:8] + "..."

    return Panel(header_text, title=title, expand=False, border_style="blue")


def create_phases_table(
    phase_scores: dict[str, float],
    show_bars: bool = True,
    width: int = 12,
) -> "Table":
    """Create standalone phases table with scores and optional bars.

    Args:
        phase_scores: Dict like {"reasoning": 0.95, "tool_use": 0.92, ...}
        show_bars: Whether to show visual bars below scores
        width: Column width

    Returns:
        Rich Table object
    """
    from rich.table import Table
    from rich.text import Text

    table = Table(show_header=True, show_edge=False, box=None, padding=(0, 2), expand=False)
    table.add_column("Phases", style="bold", width=width)
    table.add_column("Reasoning", justify="center", width=width)
    table.add_column("Tool Use", justify="center", width=width)
    table.add_column("Generation", justify="center", width=width)

    rea = phase_scores.get("reasoning", 1.0)
    too = phase_scores.get("tool_use", 1.0)
    gen = phase_scores.get("generation", 1.0)

    # Score row
    table.add_row(
        "Score",
        Text(f"{rea:.2f}", style=ter_color(rea)),
        Text(f"{too:.2f}", style=ter_color(too)),
        Text(f"{gen:.2f}", style=ter_color(gen)),
    )

    # Visual bars
    if show_bars:
        bar_len = 8
        rea_bar = "█" * int(rea * bar_len)
        too_bar = "█" * int(too * bar_len)
        gen_bar = "█" * int(gen * bar_len)

        table.add_row(
            "",
            Text(f"{rea_bar:<{bar_len}}", style=ter_color(rea)),
            Text(f"{too_bar:<{bar_len}}", style=ter_color(too)),
            Text(f"{gen_bar:<{bar_len}}", style=ter_color(gen)),
        )

    return table


def create_tokens_table(
    total_tokens: int,
    aligned_tokens: int,
    waste_tokens: int,
    input_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_hit_rate: float | None = None,
) -> "Table":
    """Create standalone tokens table with output and input metrics.

    Args:
        total_tokens: Total output tokens
        aligned_tokens: Aligned (non-waste) tokens
        waste_tokens: Waste tokens
        input_tokens: Input tokens (optional)
        cache_read_tokens: Cache read tokens (optional)
        cache_hit_rate: Cache hit rate 0-1 (optional)

    Returns:
        Rich Table object
    """
    from rich.table import Table
    from rich.text import Text

    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2), expand=False)
    table.add_column("Label", style="bold", width=12)
    table.add_column("Value", width=70)

    # Output row
    table.add_row(
        "Tokens",
        Text.assemble(
            ("Output: ", ""),
            (f"{total_tokens:,}", "cyan"),
            ("  │  ", "dim"),
            ("Aligned: ", ""),
            (f"{aligned_tokens:,}", "green"),
            ("  │  ", "dim"),
            ("Waste: ", ""),
            (f"{waste_tokens:,}", "red"),
        )
    )

    # Input/cache row (if data available)
    if input_tokens is not None or cache_read_tokens is not None:
        parts = []

        if input_tokens is not None:
            parts.extend([
                ("Input: ", ""),
                (f"{input_tokens:,}", "cyan"),
                ("  │  ", "dim"),
            ])

        if cache_read_tokens is not None:
            parts.extend([
                ("Cache: ", ""),
                (f"{cache_read_tokens:,}", "cyan"),
                ("  │  ", "dim"),
            ])

        if cache_hit_rate is not None:
            cache_pct = cache_hit_rate * 100
            cache_color = "green" if cache_pct > 90 else "yellow" if cache_pct > 50 else "red"
            parts.extend([
                ("Hit: ", ""),
                (f"{cache_pct:.1f}%", cache_color),
            ])

        if parts:
            table.add_row("", Text.assemble(*parts))

    return table


def create_context_section(
    growth_rate: float,
    message_count: int,
    bloat_detected: bool = False,
) -> "Table | None":
    """Create context growth section.

    Args:
        growth_rate: Context growth rate (e.g., 3.2 = 3.2x)
        message_count: Number of messages/turns
        bloat_detected: Whether context bloat was detected

    Returns:
        Rich Table object or None if no growth to display
    """
    from rich.table import Table
    from rich.text import Text

    if growth_rate <= 1.0:
        return None

    bloat_indicator = "  ⚠️  BLOAT" if bloat_detected else ""
    growth_color = "red" if bloat_detected else "yellow" if growth_rate > 2.0 else "green"

    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2), expand=False)
    table.add_column("Label", style="bold", width=12)
    table.add_column("Value", width=70)

    table.add_row(
        "Context",
        Text.assemble(
            ("Growth: ", ""),
            (f"{growth_rate:.1f}x", growth_color),
            (f" over {message_count} turns", ""),
            (bloat_indicator, "red" if bloat_detected else ""),
        )
    )

    return table


def create_warnings_section(warnings: list[str]) -> "Table | None":
    """Create warnings section.

    Args:
        warnings: List of warning messages

    Returns:
        Rich Table object or None if no warnings
    """
    from rich.table import Table
    from rich.text import Text

    if not warnings:
        return None

    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2), expand=False)
    table.add_column("Label", style="bold", width=12)
    table.add_column("Value", width=70)

    for i, warning in enumerate(warnings[:3]):  # Max 3 warnings
        label = "Warnings" if i == 0 else ""
        table.add_row(label, Text.assemble(("⚠️  ", "yellow"), (warning, "")))

    return table


def create_waste_patterns_section(
    waste_sources: dict[str, int],
    top_n: int = 3,
) -> "Table | None":
    """Create waste patterns section showing top sources.

    Args:
        waste_sources: Dict of {source_name: token_count}
        top_n: Number of top sources to show

    Returns:
        Rich Table object or None if no waste sources
    """
    from rich.table import Table
    from rich.text import Text

    if not waste_sources:
        return None

    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2), expand=False)
    table.add_column("Label", style="bold", width=12)
    table.add_column("Value", width=70)

    # Sort by token count and take top N
    sorted_sources = sorted(waste_sources.items(), key=lambda x: x[1], reverse=True)[:top_n]

    for i, (source, tokens) in enumerate(sorted_sources):
        label = "Top Waste" if i == 0 else ""
        table.add_row(
            label,
            Text.assemble(("• ", ""), (f"{source}: ", ""), (f"{tokens:,} tokens", "red"))
        )

    return table
