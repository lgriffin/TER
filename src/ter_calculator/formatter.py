"""Output formatting for TER results."""

from __future__ import annotations

import json
import io

from .models import InputAnalysis, TERResult, WastePattern
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

    # --- Header panel ---
    ter_text = Text(f"{result.aggregate_ter:.2f}", style=_ter_color(result.aggregate_ter))
    waste_pct = (result.waste_tokens / result.total_tokens * 100) if result.total_tokens else 0
    sid = result.session_id
    if len(sid) > 20:
        sid = sid[:8] + "..."

    # Line 1: TER | Waste | Cost
    line1_parts: list = [("TER: ", "bold"), ter_text]
    line1_parts.append(("  |  ", ""))
    line1_parts.append((f"Waste: {waste_pct:.1f}%", "red" if waste_pct > 10 else ""))
    if result.economics:
        line1_parts.append(("  |  ", ""))
        line1_parts.append((f"Cost: ${result.economics.estimated_cost_usd:.2f}", ""))
        waste_cost = _compute_waste_cost(result)
        if waste_cost > 0:
            line1_parts.append(("  |  ", ""))
            line1_parts.append((f"Waste $: ${waste_cost:.2f}", "red"))

    # Line 2: Input analysis headline (if available)
    ia = result.input_analysis
    if ia is not None:
        drift = ia.intent_drift
        pra = ia.prompt_response_alignment
        ps = ia.prompt_similarity
        bd = ia.token_breakdown

        drift_colors = {
            "convergent": "red", "divergent": "green",
            "stable": "green", "mixed": "yellow",
        }
        d_color = drift_colors.get(drift.overall_trajectory, "")
        a_color = "red" if pra.average_alignment < 0.3 else (
            "yellow" if pra.average_alignment < 0.5 else "green"
        )
        r_color = "red" if ps.prompt_redundancy_score > 0.5 else (
            "yellow" if ps.prompt_redundancy_score > 0 else "green"
        )

        line2_parts: list = [
            ("Drift: ", "bold"),
            (f"{drift.overall_trajectory}", d_color),
        ]
        if pra.pairs:
            line2_parts.append(("  |  ", ""))
            line2_parts.append(("Alignment: ", "bold"))
            line2_parts.append((f"{pra.average_alignment:.2f}", a_color))
        if ps.prompt_count >= 2:
            line2_parts.append(("  |  ", ""))
            line2_parts.append(("Redundancy: ", "bold"))
            line2_parts.append((f"{ps.prompt_redundancy_score:.0%}", r_color))
        line2_parts.append(("  |  ", ""))
        line2_parts.append((f"User: {bd.user_ratio:.0%}", "dim"))

        header = Text.assemble(
            *line1_parts, ("\n", ""), *line2_parts,
        )
    else:
        header = Text.assemble(*line1_parts)

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

    # --- Waste breakdown table ---
    _format_waste_breakdown_rich(console, result)

    # --- Input analysis ---
    if result.input_analysis is not None:
        _format_input_analysis_rich(console, result.input_analysis)

    return buf.getvalue().rstrip()


def _compute_waste_cost(result: TERResult) -> float:
    """Compute total waste cost from the waste breakdown (all sources)."""
    rows = _build_waste_breakdown(result)
    if not rows:
        return 0.0
    cost_rate = 15.0
    if result.economics:
        cost_rate = result.economics.cost_model.output_rate
    total_tokens = sum(t for _, t, _ in rows)
    return total_tokens * cost_rate / 1_000_000


def _build_waste_breakdown(result: TERResult) -> list[tuple[str, int, int]]:
    """Build waste breakdown rows: (label, tokens, instance_count).

    Combines classified span waste (by label) with structural pattern
    waste (repetitive reads, edit fragmentation) without double-counting.
    """
    from .models import ALIGNED_LABELS

    rows: list[tuple[str, int, int]] = []

    # Classified output waste by category.
    category_map = {
        "redundant_reasoning": "Redundant Reasoning",
        "unnecessary_tool_call": "Unnecessary Tool Calls",
        "over_explanation": "Over-Explanation",
    }
    cat_tokens: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    for cs in result.classified_spans:
        if cs.label in ALIGNED_LABELS:
            continue
        label = cs.label.value
        cat = category_map.get(label, label.replace("_", " ").title())
        cat_tokens[cat] = cat_tokens.get(cat, 0) + cs.span.token_count
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    for cat in category_map.values():
        if cat in cat_tokens:
            rows.append((cat, cat_tokens[cat], cat_counts[cat]))

    # Waste patterns grouped by type.
    # Skip types whose waste is already counted from classified spans.
    pattern_labels = {
        "reasoning_loop": "Reasoning Loops",
        "duplicate_tool_call": "Duplicate Tool Calls",
        "context_restatement": "Context Restatement",
        "repetitive_read": "Repetitive Reads",
        "edit_fragmentation": "Edit Fragmentation",
        "bash_antipattern": "Bash Anti-Patterns",
        "failed_tool_retry": "Failed Tool Retries",
        "repeated_command": "Repeated Commands",
    }
    # Map pattern types to the classified span category they overlap with.
    pattern_overlap = {
        "reasoning_loop": "Redundant Reasoning",
    }
    by_type: dict[str, list[WastePattern]] = {}
    for wp in (result.waste_patterns or []):
        by_type.setdefault(wp.pattern_type, []).append(wp)

    for ptype, wps in by_type.items():
        overlap_cat = pattern_overlap.get(ptype)
        if overlap_cat and overlap_cat in cat_tokens:
            continue  # Already counted from classified spans
        label = pattern_labels.get(ptype, ptype.replace("_", " ").title())
        tokens = sum(wp.tokens_wasted for wp in wps)
        rows.append((label, tokens, len(wps)))

    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _format_waste_breakdown_rich(console, result: TERResult) -> None:
    """Render waste breakdown as a Rich table."""
    from rich.table import Table

    rows = _build_waste_breakdown(result)
    if not rows:
        return

    cost_rate = 15.0  # default output rate $/MTok
    if result.economics:
        cost_rate = result.economics.cost_model.output_rate

    total_waste = sum(t for _, t, _ in rows)

    table = Table(show_header=True, show_edge=True, title="Waste Breakdown")
    table.add_column("Source", style="bold", width=22)
    table.add_column("Tokens", justify="right", width=10)
    table.add_column("%", justify="right", width=6)
    table.add_column("Cost", justify="right", width=8)
    table.add_column("Count", justify="right", width=6, style="dim")

    for label, tokens, count in rows:
        pct = (tokens / total_waste * 100) if total_waste > 0 else 0
        cost = tokens * cost_rate / 1_000_000
        table.add_row(
            label,
            f"{tokens:,}",
            f"{pct:.0f}%",
            f"${cost:.4f}",
            str(count),
        )

    table.add_section()
    total_cost = total_waste * cost_rate / 1_000_000
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_waste:,}[/bold]",
        "[bold]100%[/bold]",
        f"[bold]${total_cost:.4f}[/bold]",
        "",
    )
    console.print(table)


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
            wc = _compute_waste_cost(r)
            waste_cost_str = f"[red]${wc:.2f}[/red]"
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
        total_waste_cost = sum(_compute_waste_cost(r) for r in results)
        color = _ter_color(avg_ter)
        console.print(f"\nAverage TER: [{color}]{avg_ter:.2f}[/{color}]  |  Total Cost: ${total_cost:.2f}  |  Total Waste: [red]${total_waste_cost:.2f}[/red]")

    return buf.getvalue().rstrip()


def _format_input_analysis_rich(console, ia: InputAnalysis) -> None:
    """Render input analysis section using Rich."""
    from rich.table import Table

    bd = ia.token_breakdown
    ps = ia.prompt_similarity

    # Token breakdown table.
    console.print("\n[bold]Input Analysis[/bold]")
    tb = Table(show_header=True, show_edge=True)
    tb.add_column("Origin", style="bold", width=14)
    tb.add_column("Category", width=16)
    tb.add_column("Tokens", justify="right", width=10)

    tb.add_row("User", "Prompt Text", f"{bd.user_input_tokens:,}")
    tb.add_row("User", "Tool Results", f"{bd.user_result_tokens:,}")
    tb.add_row("Model", "Reasoning", f"{bd.model_reasoning_tokens:,}")
    tb.add_row("Model", "Tool Calls", f"{bd.model_tool_tokens:,}")
    tb.add_row("Model", "Generation", f"{bd.model_generation_tokens:,}")
    tb.add_section()
    tb.add_row("[bold]User Total[/bold]", "", f"[bold]{bd.total_user_tokens:,}[/bold]")
    tb.add_row("[bold]Model Total[/bold]", "", f"[bold]{bd.total_model_tokens:,}[/bold]")
    tb.add_row("User Ratio", "", f"{bd.user_ratio:.1%}")
    console.print(tb)

    # Prompt similarity.
    if ps.prompt_count >= 2:
        r_color = "red" if ps.prompt_redundancy_score > 0.5 else (
            "yellow" if ps.prompt_redundancy_score > 0 else "green"
        )
        console.print(
            f"\nPrompt Redundancy: [{r_color}]{ps.prompt_redundancy_score:.0%}[/{r_color}]"
            f"  ({ps.prompt_count} prompts, {len(ps.similar_pairs)} similar pair(s))"
        )
        for pair in ps.similar_pairs[:5]:
            a_text = pair.prompt_a_text[:40] + "..." if len(pair.prompt_a_text) > 40 else pair.prompt_a_text
            b_text = pair.prompt_b_text[:40] + "..." if len(pair.prompt_b_text) > 40 else pair.prompt_b_text
            console.print(
                f'  [dim]#{pair.prompt_a_index+1}[/dim] "{a_text}" '
                f'[dim]~[/dim] [dim]#{pair.prompt_b_index+1}[/dim] "{b_text}" '
                f'[yellow]({pair.similarity:.2f})[/yellow]'
            )

    # Intent drift.
    drift = ia.intent_drift
    if drift.steps:
        _drift_colors = {
            "convergent": "red", "divergent": "green",
            "stable": "green", "mixed": "yellow",
        }
        t_color = _drift_colors.get(drift.overall_trajectory, "")
        console.print(
            f"\nIntent Drift: [{t_color}]{drift.overall_trajectory}[/{t_color}]"
            f"  (avg similarity: {drift.average_drift:.2f})"
        )
        for step in drift.steps:
            s_color = "red" if step.drift_type == "convergent" else (
                "green" if step.drift_type == "divergent" else "yellow"
            )
            console.print(
                f"  #{step.from_index+1} -> #{step.to_index+1}: "
                f"[{s_color}]{step.drift_type}[/{s_color}] ({step.similarity:.2f})"
            )

    # Prompt-response alignment.
    pra = ia.prompt_response_alignment
    if pra.pairs:
        a_color = "red" if pra.average_alignment < 0.3 else (
            "yellow" if pra.average_alignment < 0.5 else "green"
        )
        console.print(
            f"\nPrompt-Response Alignment: [{a_color}]{pra.average_alignment:.2f}[/{a_color}]"
            f"  ({len(pra.pairs)} pair(s), {pra.low_alignment_count} low)"
        )
        for pair in pra.pairs:
            p_color = "red" if pair.alignment < 0.3 else (
                "yellow" if pair.alignment < 0.5 else "green"
            )
            prompt_short = pair.prompt_text[:50] + "..." if len(pair.prompt_text) > 50 else pair.prompt_text
            console.print(
                f'  [dim]#{pair.prompt_index+1}[/dim] "{prompt_short}" '
                f'-> [{p_color}]{pair.alignment:.2f}[/{p_color}]'
            )


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
        waste_cost = _compute_waste_cost(result)
        if waste_cost > 0:
            cost_str += f"  |  Waste $: ${waste_cost:.2f}"
    lines.append(f"TER: {result.aggregate_ter:.2f}  |  Waste: {waste_pct:.1f}%{cost_str}")

    # Input analysis headline.
    ia = result.input_analysis
    if ia is not None:
        drift = ia.intent_drift
        pra = ia.prompt_response_alignment
        ps = ia.prompt_similarity
        parts = [f"Drift: {drift.overall_trajectory}"]
        if pra.pairs:
            parts.append(f"Alignment: {pra.average_alignment:.2f}")
        if ps.prompt_count >= 2:
            parts.append(f"Redundancy: {ps.prompt_redundancy_score:.0%}")
        parts.append(f"User: {ia.token_breakdown.user_ratio:.0%}")
        lines.append("  |  ".join(parts))

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

    # Waste breakdown.
    rows = _build_waste_breakdown(result)
    if rows:
        cost_rate = 15.0
        if result.economics:
            cost_rate = result.economics.cost_model.output_rate
        total_waste = sum(t for _, t, _ in rows)
        lines.extend(["", "Waste Breakdown:"])
        lines.append(f"  {'Source':<24} {'Tokens':>10} {'%':>5} {'Cost':>10} {'Count':>6}")
        for label, tokens, count in rows:
            pct = (tokens / total_waste * 100) if total_waste > 0 else 0
            cost = tokens * cost_rate / 1_000_000
            lines.append(f"  {label:<24} {tokens:>10,} {pct:>4.0f}% ${cost:>8.4f} {count:>6}")
        total_cost = total_waste * cost_rate / 1_000_000
        lines.append(f"  {'Total':<24} {total_waste:>10,}  100% ${total_cost:>8.4f}")

    # Input analysis.
    if result.input_analysis is not None:
        lines.extend(_format_input_analysis_text(result.input_analysis))

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
            wc = _compute_waste_cost(r)
            waste_cost_str = f"${wc:.2f}"
        lines.append(
            f"  {i:<3} {sid:<12} {r.aggregate_ter:<6.2f} "
            f"{waste_pct:<8.1f} {cache_str:<8} {cost_str:<10} {waste_cost_str:<10} {pattern_count:<8}"
        )

    if results:
        avg_ter = sum(r.aggregate_ter for r in results) / len(results)
        total_cost = sum(r.economics.estimated_cost_usd for r in results if r.economics)
        total_waste_cost = sum(_compute_waste_cost(r) for r in results)
        lines.extend(["", f"Average TER: {avg_ter:.2f}  |  Total Cost: ${total_cost:.2f}  |  Total Waste: ${total_waste_cost:.2f}"])

    return "\n".join(lines)


def _format_input_analysis_text(ia: InputAnalysis) -> list[str]:
    """Format input analysis as plain text lines."""
    bd = ia.token_breakdown
    ps = ia.prompt_similarity

    lines = [
        "",
        "Input Analysis:",
        f"  User Tokens:   {bd.total_user_tokens:,} (prompt: {bd.user_input_tokens:,}, tool results: {bd.user_result_tokens:,})",
        f"  Model Tokens:  {bd.total_model_tokens:,} (reasoning: {bd.model_reasoning_tokens:,}, tool: {bd.model_tool_tokens:,}, generation: {bd.model_generation_tokens:,})",
        f"  User Ratio:    {bd.user_ratio:.1%}",
    ]

    if ps.prompt_count >= 2:
        lines.append(f"  Prompt Redundancy: {ps.prompt_redundancy_score:.0%} ({ps.prompt_count} prompts, {len(ps.similar_pairs)} similar pair(s))")
        for pair in ps.similar_pairs[:5]:
            a_text = pair.prompt_a_text[:40] + "..." if len(pair.prompt_a_text) > 40 else pair.prompt_a_text
            b_text = pair.prompt_b_text[:40] + "..." if len(pair.prompt_b_text) > 40 else pair.prompt_b_text
            lines.append(f'    #{pair.prompt_a_index+1} "{a_text}" ~ #{pair.prompt_b_index+1} "{b_text}" ({pair.similarity:.2f})')

    # Intent drift.
    drift = ia.intent_drift
    if drift.steps:
        lines.append(f"  Intent Drift: {drift.overall_trajectory} (avg similarity: {drift.average_drift:.2f})")
        for step in drift.steps:
            lines.append(f"    #{step.from_index+1} -> #{step.to_index+1}: {step.drift_type} ({step.similarity:.2f})")

    # Prompt-response alignment.
    pra = ia.prompt_response_alignment
    if pra.pairs:
        lines.append(f"  Prompt-Response Alignment: {pra.average_alignment:.2f} ({len(pra.pairs)} pair(s), {pra.low_alignment_count} low)")
        for pair in pra.pairs:
            prompt_short = pair.prompt_text[:50] + "..." if len(pair.prompt_text) > 50 else pair.prompt_text
            marker = " [LOW]" if pair.alignment < 0.3 else ""
            lines.append(f'    #{pair.prompt_index+1} "{prompt_short}" -> {pair.alignment:.2f}{marker}')

    return lines


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
    rows = _build_waste_breakdown(result)
    if rows:
        cost_rate = 15.0
        if result.economics:
            cost_rate = result.economics.cost_model.output_rate
        total_waste = sum(t for _, t, _ in rows)
        data["waste_breakdown"] = {
            "sources": [
                {
                    "source": label,
                    "tokens": tokens,
                    "percentage": round(tokens / total_waste * 100, 1) if total_waste > 0 else 0,
                    "cost_usd": round(tokens * cost_rate / 1_000_000, 6),
                    "count": count,
                }
                for label, tokens, count in rows
            ],
            "total_tokens": total_waste,
            "total_cost_usd": round(total_waste * cost_rate / 1_000_000, 6),
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
    if result.input_analysis is not None:
        ia = result.input_analysis
        bd = ia.token_breakdown
        ps = ia.prompt_similarity
        data["input_analysis"] = {
            "token_breakdown": {
                "user_input_tokens": bd.user_input_tokens,
                "user_result_tokens": bd.user_result_tokens,
                "model_reasoning_tokens": bd.model_reasoning_tokens,
                "model_tool_tokens": bd.model_tool_tokens,
                "model_generation_tokens": bd.model_generation_tokens,
                "total_user_tokens": bd.total_user_tokens,
                "total_model_tokens": bd.total_model_tokens,
                "user_ratio": bd.user_ratio,
            },
            "prompt_similarity": {
                "prompt_count": ps.prompt_count,
                "prompt_redundancy_score": ps.prompt_redundancy_score,
                "similar_pairs": [
                    {
                        "prompt_a_index": p.prompt_a_index,
                        "prompt_b_index": p.prompt_b_index,
                        "similarity": p.similarity,
                        "prompt_a_text": p.prompt_a_text,
                        "prompt_b_text": p.prompt_b_text,
                    }
                    for p in ps.similar_pairs
                ],
            },
            "intent_drift": {
                "overall_trajectory": ia.intent_drift.overall_trajectory,
                "average_drift": ia.intent_drift.average_drift,
                "steps": [
                    {
                        "from_index": s.from_index,
                        "to_index": s.to_index,
                        "similarity": s.similarity,
                        "drift_type": s.drift_type,
                    }
                    for s in ia.intent_drift.steps
                ],
            },
            "prompt_response_alignment": {
                "average_alignment": ia.prompt_response_alignment.average_alignment,
                "low_alignment_count": ia.prompt_response_alignment.low_alignment_count,
                "pairs": [
                    {
                        "prompt_index": p.prompt_index,
                        "prompt_text": p.prompt_text,
                        "response_text": p.response_text,
                        "alignment": p.alignment,
                    }
                    for p in ia.prompt_response_alignment.pairs
                ],
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
