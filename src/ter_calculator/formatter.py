"""Output formatting for TER results."""

from __future__ import annotations

import json
import io

from .models import TERResult, WastePattern
from .waste import summarize_waste


def format_ter_result(
    result: TERResult, fmt: str = "text", use_rich: bool = True,
) -> str:
    """Format a TER result for output."""
    if fmt == "json":
        return _format_json(result)
    if use_rich:
        try:
            return _format_rich(result)
        except (ImportError, UnicodeEncodeError):
            pass
    return _format_text(result)


def format_comparison(
    results: list[TERResult], fmt: str = "text", use_rich: bool = True,
) -> str:
    """Format multiple TER results as a comparison."""
    if fmt == "json":
        return _format_comparison_json(results)
    if use_rich:
        try:
            return _format_comparison_rich(results)
        except (ImportError, UnicodeEncodeError):
            pass
    return _format_comparison_text(results)


# --- Rich formatting ---


def _ter_color(value: float) -> str:
    """Return Rich color for a TER score."""
    if value >= 0.7:
        return "green"
    if value >= 0.4:
        return "yellow"
    return "red"


def _format_rich(result: TERResult) -> str:
    """Format TER result using Rich library."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=72)

    # --- Header with score and cost ---
    ter_text = Text(f"{result.aggregate_ter:.2f}", style=_ter_color(result.aggregate_ter))
    waste_pct = (result.waste_tokens / result.total_tokens * 100) if result.total_tokens else 0
    header_parts = [("TER: ", "bold"), ter_text]
    header_parts.append(("  |  ", ""))
    header_parts.append((f"Waste: {waste_pct:.1f}%", "red" if waste_pct > 10 else ""))
    if result.economics:
        header_parts.append(("  |  ", ""))
        header_parts.append((f"Cost: ${result.economics.estimated_cost_usd:.2f}", ""))
        if result.economics.estimated_waste_cost_usd > 0:
            header_parts.append(("  |  ", ""))
            header_parts.append((f"Waste $: ${result.economics.estimated_waste_cost_usd:.2f}", "red"))
    header = Text.assemble(*header_parts)
    sid = result.session_id
    if len(sid) > 20:
        sid = sid[:8] + "..."
    console.print(Panel(header, title=sid, expand=False))

    # --- Combined scores table (phases + tokens in one) ---
    table = Table(show_header=True, show_edge=True)
    table.add_column("Phase", style="bold", width=12)
    table.add_column("TER", justify="right", width=6)
    table.add_column("", width=3)
    table.add_column("Metric", style="bold", width=14)
    table.add_column("Value", justify="right", width=12)

    phases = [
        ("Reasoning", result.phase_scores.get("reasoning", 0)),
        ("Tool Use", result.phase_scores.get("tool_use", 0)),
        ("Generation", result.phase_scores.get("generation", 0)),
    ]
    right_rows = [
        ("Output Tokens", f"{result.total_tokens:,}"),
        ("Aligned", f"{result.aligned_tokens:,}"),
        ("Waste", f"{result.waste_tokens:,}"),
    ]

    for i in range(3):
        p_name, p_score = phases[i]
        p_color = _ter_color(p_score)
        r_label, r_value = right_rows[i]
        table.add_row(
            p_name,
            f"[{p_color}]{p_score:.2f}[/{p_color}]",
            "",
            r_label,
            r_value,
        )
    console.print(table)

    # --- Session economics (compact) ---
    if result.economics is not None:
        econ = result.economics
        cache_pct = econ.cache_hit_rate * 100
        cache_color = "green" if cache_pct >= 50 else "yellow" if cache_pct >= 20 else "red"

        econ_table = Table(show_header=True, show_edge=True)
        econ_table.add_column("Economics", style="bold", width=18)
        econ_table.add_column("", justify="right", width=12)
        econ_table.add_column("", width=3)
        econ_table.add_column("Context", style="bold", width=14)
        econ_table.add_column("", justify="right", width=12)

        pos = econ.positional
        g = econ.input_growth
        bloat_str = "[red]YES[/red]" if g.context_bloat_detected else (
            "[yellow]WATCH[/yellow]" if g.is_superlinear else "[green]NO[/green]"
        )

        left_rows = [
            ("Input Tokens", f"{econ.total_input_tokens:,}"),
            ("Cache Read", f"{econ.total_cache_read_tokens:,}"),
            ("Cache Hit Rate", f"[{cache_color}]{cache_pct:.1f}%[/{cache_color}]"),
        ]
        right_rows_e = [
            ("Growth", f"{g.growth_rate:.1f}x ({len(g.turn_input_tokens)} turns)"),
            ("Bloat", bloat_str),
            ("Positional", f"{pos.early_ter:.2f} / {pos.mid_ter:.2f} / {pos.late_ter:.2f}"),
        ]

        for i in range(3):
            l_label, l_value = left_rows[i]
            r_label, r_value = right_rows_e[i]
            econ_table.add_row(l_label, l_value, "", r_label, r_value)
        console.print(econ_table)

    # --- Waste patterns (collapsed) ---
    if result.waste_patterns:
        summary = _collapse_waste_patterns(result.waste_patterns)
        console.print(f"\n[bold]Waste Patterns ({len(result.waste_patterns)})[/bold]")
        for line in summary:
            console.print(f"  {line}")
    elif result.waste_patterns is not None and result.classified_spans:
        waste_summary = summarize_waste(result.classified_spans, [])
        if waste_summary["total_waste_tokens"] > 0:
            console.print(f"\n[dim]No structural patterns, but {waste_summary['total_waste_tokens']:,} tokens classified as waste[/dim]")

    return buf.getvalue().rstrip()


def _collapse_waste_patterns(patterns: list[WastePattern]) -> list[str]:
    """Collapse waste patterns into a summary by type."""
    by_type: dict[str, list[WastePattern]] = {}
    for wp in patterns:
        by_type.setdefault(wp.pattern_type, []).append(wp)

    lines: list[str] = []
    for ptype, wps in by_type.items():
        label = ptype.replace("_", " ").title()
        total_tokens = sum(wp.tokens_wasted for wp in wps)
        count = len(wps)
        if count == 1:
            lines.append(f"{label}: {wps[0].description} ({total_tokens:,} tokens)")
        else:
            lines.append(f"{label}: {count} instances ({total_tokens:,} tokens)")
    return lines


def _format_comparison_rich(results: list[TERResult]) -> str:
    """Format comparison using Rich table."""
    from rich.console import Console
    from rich.table import Table

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=90)

    table = Table(title="TER Comparison", show_header=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Session", style="bold")
    table.add_column("TER", justify="right")
    table.add_column("Waste%", justify="right")
    table.add_column("Cache%", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Waste $", justify="right")
    table.add_column("Patterns", justify="right")

    for i, r in enumerate(results, 1):
        color = _ter_color(r.aggregate_ter)
        pattern_count = len(r.waste_patterns) if r.waste_patterns else 0
        waste_pct = (r.waste_tokens / r.total_tokens * 100) if r.total_tokens else 0
        cache_str = ""
        cost_str = ""
        waste_cost_str = ""
        if r.economics:
            cache_pct = r.economics.cache_hit_rate * 100
            cache_str = f"{cache_pct:.0f}%"
            cost_str = f"${r.economics.estimated_cost_usd:.2f}"
            waste_cost_str = f"[red]${r.economics.estimated_waste_cost_usd:.2f}[/red]"
        sid = r.session_id
        if len(sid) > 20:
            sid = sid[:8] + "..."
        table.add_row(
            str(i),
            sid,
            f"[{color}]{r.aggregate_ter:.2f}[/{color}]",
            f"{waste_pct:.1f}%",
            cache_str,
            cost_str,
            waste_cost_str,
            str(pattern_count),
        )

    console.print(table)

    if results:
        avg_ter = sum(r.aggregate_ter for r in results) / len(results)
        total_cost = sum(r.economics.estimated_cost_usd for r in results if r.economics)
        total_waste_cost = sum(r.economics.estimated_waste_cost_usd for r in results if r.economics)
        color = _ter_color(avg_ter)
        console.print(f"\nAverage TER: [{color}]{avg_ter:.2f}[/{color}]  |  Total Cost: ${total_cost:.2f}  |  Total Waste: [red]${total_waste_cost:.2f}[/red]")

    return buf.getvalue().rstrip()


# --- Plain text formatting ---


def _format_text(result: TERResult) -> str:
    """Format TER result as plain text."""
    waste_pct = (result.waste_tokens / result.total_tokens * 100) if result.total_tokens else 0
    sid = result.session_id
    if len(sid) > 20:
        sid = sid[:8] + "..."

    lines = [
        f"TER Report: {sid}",
        "\u2550" * 40,
        "",
    ]

    # Headline.
    cost_str = ""
    if result.economics:
        cost_str = f"  |  Cost: ${result.economics.estimated_cost_usd:.2f}"
        if result.economics.estimated_waste_cost_usd > 0:
            cost_str += f"  |  Waste $: ${result.economics.estimated_waste_cost_usd:.2f}"
    lines.append(f"TER: {result.aggregate_ter:.2f}  |  Waste: {waste_pct:.1f}%{cost_str}")
    lines.append("")

    # Phases.
    lines.append("Phases:     Reasoning  Tool Use  Generation")
    lines.append(
        f"            {result.phase_scores.get('reasoning', 0):.2f}"
        f"       {result.phase_scores.get('tool_use', 0):.2f}"
        f"      {result.phase_scores.get('generation', 0):.2f}"
    )
    lines.append("")

    # Tokens.
    lines.append(f"Output Tokens: {result.total_tokens:,}  (aligned: {result.aligned_tokens:,}  waste: {result.waste_tokens:,})")

    # Economics.
    if result.economics is not None:
        econ = result.economics
        cache_pct = econ.cache_hit_rate * 100
        pos = econ.positional
        g = econ.input_growth

        lines.extend([
            "",
            f"Input: {econ.total_input_tokens:,}  Cache Read: {econ.total_cache_read_tokens:,}  Cache Hit: {cache_pct:.1f}%",
            f"Context Growth: {g.growth_rate:.1f}x over {len(g.turn_input_tokens)} turns"
            + (" [BLOAT]" if g.context_bloat_detected else (" [WATCH]" if g.is_superlinear else "")),
            f"Positional TER: {pos.early_ter:.2f} (early) / {pos.mid_ter:.2f} (mid) / {pos.late_ter:.2f} (late)",
        ])

    # Waste patterns (collapsed).
    if result.waste_patterns:
        summary = _collapse_waste_patterns(result.waste_patterns)
        lines.extend(["", f"Waste Patterns ({len(result.waste_patterns)}):"])
        for line in summary:
            lines.append(f"  {line}")
    elif result.waste_patterns is not None and result.classified_spans:
        waste_summary = summarize_waste(result.classified_spans, [])
        if waste_summary["total_waste_tokens"] > 0:
            lines.extend(["", f"No structural patterns, but {waste_summary['total_waste_tokens']:,} tokens classified as waste"])

    return "\n".join(lines)


def _format_comparison_text(results: list[TERResult]) -> str:
    """Format comparison as a plain text table."""
    lines = [
        "TER Comparison",
        "\u2550" * 40,
        "",
        f"  {'#':<3} {'Session':<12} {'TER':<6} {'Waste%':<8} {'Cache%':<8} {'Cost':<10} {'Waste $':<10} {'Patterns':<8}",
    ]

    for i, r in enumerate(results, 1):
        sid = r.session_id[:12] if len(r.session_id) <= 12 else r.session_id[:8] + "..."
        pattern_count = len(r.waste_patterns) if r.waste_patterns else 0
        waste_pct = (r.waste_tokens / r.total_tokens * 100) if r.total_tokens else 0
        cache_str = ""
        cost_str = ""
        waste_cost_str = ""
        if r.economics:
            cache_pct = r.economics.cache_hit_rate * 100
            cache_str = f"{cache_pct:.0f}%"
            cost_str = f"${r.economics.estimated_cost_usd:.2f}"
            waste_cost_str = f"${r.economics.estimated_waste_cost_usd:.2f}"
        lines.append(
            f"  {i:<3} {sid:<12} {r.aggregate_ter:<6.2f} "
            f"{waste_pct:<8.1f} {cache_str:<8} {cost_str:<10} {waste_cost_str:<10} {pattern_count:<8}"
        )

    if results:
        avg_ter = sum(r.aggregate_ter for r in results) / len(results)
        total_cost = sum(r.economics.estimated_cost_usd for r in results if r.economics)
        total_waste_cost = sum(r.economics.estimated_waste_cost_usd for r in results if r.economics)
        lines.extend(["", f"Average TER: {avg_ter:.2f}  |  Total Cost: ${total_cost:.2f}  |  Total Waste: ${total_waste_cost:.2f}"])

    return "\n".join(lines)


# --- JSON formatting ---


def _format_json(result: TERResult) -> str:
    """Format TER result as JSON."""
    data = _ter_result_to_dict(result)
    return json.dumps(data, indent=2)


def _format_comparison_json(results: list[TERResult]) -> str:
    """Format comparison as JSON."""
    data = {
        "sessions": [_ter_result_to_dict(r) for r in results],
        "average_ter": round(
            sum(r.aggregate_ter for r in results) / len(results), 4
        ) if results else 0.0,
    }
    return json.dumps(data, indent=2)


def _ter_result_to_dict(result: TERResult) -> dict:
    """Convert TERResult to a JSON-serializable dict."""
    data: dict = {
        "session_id": result.session_id,
        "aggregate_ter": result.aggregate_ter,
        "raw_ratio": result.raw_ratio,
        "phase_scores": result.phase_scores,
        "total_tokens": result.total_tokens,
        "aligned_tokens": result.aligned_tokens,
        "waste_tokens": result.waste_tokens,
    }
    if result.intent:
        data["intent_confidence"] = result.intent.confidence
    if result.waste_patterns:
        data["waste_patterns"] = [
            {
                "type": wp.pattern_type,
                "start_position": wp.start_position,
                "end_position": wp.end_position,
                "spans_involved": wp.spans_involved,
                "tokens_wasted": wp.tokens_wasted,
                "description": wp.description,
            }
            for wp in result.waste_patterns
        ]
    if result.classified_spans:
        summary = summarize_waste(result.classified_spans, result.waste_patterns or [])
        data["waste_summary"] = {
            "total_waste_tokens": summary["total_waste_tokens"],
            "waste_by_category": summary["waste_by_category"],
            "waste_by_phase": summary["waste_by_phase"],
            "top_patterns": summary["top_patterns"],
            "explanation": summary["explanation"],
        }
    if result.economics is not None:
        econ = result.economics
        data["economics"] = {
            "total_input_tokens": econ.total_input_tokens,
            "total_output_tokens": econ.total_output_tokens,
            "total_cache_creation_tokens": econ.total_cache_creation_tokens,
            "total_cache_read_tokens": econ.total_cache_read_tokens,
            "input_output_ratio": econ.input_output_ratio,
            "cache_hit_rate": econ.cache_hit_rate,
            "estimated_cost_usd": econ.estimated_cost_usd,
            "estimated_waste_cost_usd": econ.estimated_waste_cost_usd,
            "cost_model": {
                "input_rate": econ.cost_model.input_rate,
                "output_rate": econ.cost_model.output_rate,
                "cache_read_rate": econ.cost_model.cache_read_rate,
                "cache_write_rate": econ.cost_model.cache_write_rate,
            },
            "positional": {
                "early_ter": econ.positional.early_ter,
                "mid_ter": econ.positional.mid_ter,
                "late_ter": econ.positional.late_ter,
                "early_span_count": econ.positional.early_span_count,
                "mid_span_count": econ.positional.mid_span_count,
                "late_span_count": econ.positional.late_span_count,
            },
            "input_growth": {
                "turn_input_tokens": econ.input_growth.turn_input_tokens,
                "growth_rate": econ.input_growth.growth_rate,
                "is_superlinear": econ.input_growth.is_superlinear,
                "context_bloat_detected": econ.input_growth.context_bloat_detected,
            },
        }
    return data


def _format_waste_pattern(wp: WastePattern) -> str:
    """Format a single waste pattern for text display."""
    pos = (
        f"spans {wp.start_position}-{wp.end_position}"
        if wp.start_position != wp.end_position
        else f"span {wp.start_position}"
    )
    label = wp.pattern_type.replace("_", " ").title()
    return f"{label} ({pos}): {wp.description}, {wp.tokens_wasted:,} tokens"
