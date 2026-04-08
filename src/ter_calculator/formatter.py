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
    console = Console(file=buf, force_terminal=True, width=60)

    # Header panel.
    ter_text = Text(f"{result.aggregate_ter:.4f}", style=_ter_color(result.aggregate_ter))
    header = Text.assemble("Aggregate TER: ", ter_text, "  |  Raw Ratio: ", f"{result.raw_ratio:.4f}")
    console.print(Panel(header, title=f"TER Report: {result.session_id}", expand=False))

    # Phase scores table.
    phase_table = Table(title="Phase Scores", show_header=True)
    phase_table.add_column("Phase", style="bold")
    phase_table.add_column("Score", justify="right")

    for phase_name in ["reasoning", "tool_use", "generation"]:
        score = result.phase_scores.get(phase_name, 0)
        color = _ter_color(score)
        phase_table.add_row(
            phase_name.replace("_", " ").title(),
            f"[{color}]{score:.4f}[/{color}]",
        )
    console.print(phase_table)

    # Token summary table.
    token_table = Table(title="Token Summary", show_header=True)
    token_table.add_column("Metric", style="bold")
    token_table.add_column("Count", justify="right")
    token_table.add_row("Total", f"{result.total_tokens:,}")
    token_table.add_row("Aligned", f"[green]{result.aligned_tokens:,}[/green]")
    token_table.add_row("Waste", f"[red]{result.waste_tokens:,}[/red]")
    console.print(token_table)

    # Intent confidence.
    if result.intent and result.intent.confidence < 1.0:
        console.print(f"\nIntent Confidence: {result.intent.confidence:.2f}")

    # Waste patterns.
    if result.waste_patterns:
        console.print(f"\n[bold]Waste Patterns Found: {len(result.waste_patterns)}[/bold]")
        for i, wp in enumerate(result.waste_patterns, 1):
            console.print(f"  {i}. {_format_waste_pattern(wp)}")
    elif result.waste_patterns is not None:
        console.print("\nWaste Patterns Found: 0")

    # Waste summary.
    if result.classified_spans:
        summary = summarize_waste(result.classified_spans, result.waste_patterns or [])
        if summary["total_waste_tokens"] > 0:
            console.print(f"\n[bold]Waste Summary[/bold]")
            console.print(f"  {summary['explanation']}")

    return buf.getvalue().rstrip()


def _format_comparison_rich(results: list[TERResult]) -> str:
    """Format comparison using Rich table."""
    from rich.console import Console
    from rich.table import Table

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=80)

    table = Table(title="TER Comparison", show_header=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Session", style="bold")
    table.add_column("TER", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Waste", justify="right")
    table.add_column("Patterns", justify="right")

    for i, r in enumerate(results, 1):
        color = _ter_color(r.aggregate_ter)
        pattern_count = len(r.waste_patterns) if r.waste_patterns else 0
        table.add_row(
            str(i),
            r.session_id[:25],
            f"[{color}]{r.aggregate_ter:.4f}[/{color}]",
            f"{r.total_tokens:,}",
            f"{r.waste_tokens:,}",
            str(pattern_count),
        )

    console.print(table)

    if results:
        avg_ter = sum(r.aggregate_ter for r in results) / len(results)
        color = _ter_color(avg_ter)
        console.print(f"\nAverage TER: [{color}]{avg_ter:.4f}[/{color}]")

    return buf.getvalue().rstrip()


# --- Plain text formatting ---


def _format_text(result: TERResult) -> str:
    """Format TER result as plain text."""
    lines = [
        f"TER Report: {result.session_id}",
        "\u2550" * 40,
        "",
        f"Aggregate TER:    {result.aggregate_ter:.4f}",
        f"Raw Ratio:        {result.raw_ratio:.4f}",
        "",
        "Phase Scores:",
        f"  Reasoning:      {result.phase_scores.get('reasoning', 0):.4f}",
        f"  Tool Use:       {result.phase_scores.get('tool_use', 0):.4f}",
        f"  Generation:     {result.phase_scores.get('generation', 0):.4f}",
        "",
        "Token Summary:",
        f"  Total:          {result.total_tokens:,}",
        f"  Aligned:        {result.aligned_tokens:,}",
        f"  Waste:          {result.waste_tokens:,}",
    ]

    if result.intent and result.intent.confidence < 1.0:
        lines.extend(["", f"Intent Confidence: {result.intent.confidence:.2f}"])

    if result.waste_patterns:
        lines.extend(["", f"Waste Patterns Found: {len(result.waste_patterns)}"])
        for i, wp in enumerate(result.waste_patterns, 1):
            lines.append(f"  {i}. {_format_waste_pattern(wp)}")
    elif result.waste_patterns is not None:
        lines.extend(["", "Waste Patterns Found: 0"])

    # Waste summary.
    if result.classified_spans:
        summary = summarize_waste(result.classified_spans, result.waste_patterns or [])
        if summary["total_waste_tokens"] > 0:
            lines.extend(["", "Waste Summary:", f"  {summary['explanation']}"])

    return "\n".join(lines)


def _format_comparison_text(results: list[TERResult]) -> str:
    """Format comparison as a plain text table."""
    lines = [
        "TER Comparison",
        "\u2550" * 40,
        "",
        f"  {'#':<3} {'Session':<20} {'TER':<8} {'Tokens':<10} {'Waste':<10} {'Patterns':<8}",
    ]

    for i, r in enumerate(results, 1):
        sid = r.session_id[:20]
        pattern_count = len(r.waste_patterns) if r.waste_patterns else 0
        lines.append(
            f"  {i:<3} {sid:<20} {r.aggregate_ter:<8.4f} "
            f"{r.total_tokens:<10,} {r.waste_tokens:<10,} {pattern_count:<8}"
        )

    if results:
        avg_ter = sum(r.aggregate_ter for r in results) / len(results)
        lines.extend(["", f"Average TER: {avg_ter:.4f}"])

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
