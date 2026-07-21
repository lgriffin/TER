"""Rich terminal formatting for TER results."""

from __future__ import annotations

import io

from .models import CostModel, InputAnalysis, TERResult
from .rich_components import ter_color as _ter_color


def format_rich(result: TERResult) -> str:
    """Format TER result using Rich library."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from .formatter import _compute_waste_cost

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=72)

    ter_text = Text(
        f"{result.aggregate_ter:.2f}", style=_ter_color(result.aggregate_ter)
    )
    waste_pct = (
        (result.waste_tokens / result.total_tokens * 100) if result.total_tokens else 0
    )
    sid = result.session_id
    if len(sid) > 20:
        sid = sid[:8] + "..."

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

    ia = result.input_analysis
    if ia is not None:
        drift = ia.intent_drift
        pra = ia.prompt_response_alignment
        ps = ia.prompt_similarity
        bd = ia.token_breakdown

        drift_colors = {
            "convergent": "red",
            "divergent": "green",
            "stable": "green",
            "mixed": "yellow",
        }
        d_color = drift_colors.get(drift.overall_trajectory, "")
        a_color = (
            "red"
            if pra.average_alignment < 0.3
            else ("yellow" if pra.average_alignment < 0.5 else "green")
        )
        r_color = (
            "red"
            if ps.prompt_redundancy_score > 0.5
            else ("yellow" if ps.prompt_redundancy_score > 0 else "green")
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
            *line1_parts,
            ("\n", ""),
            *line2_parts,
        )
    else:
        header = Text.assemble(*line1_parts)

    console.print(Panel(header, title=sid, expand=False))

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

    if result.economics is not None:
        econ = result.economics
        cache_pct = econ.cache_hit_rate * 100
        cache_color = (
            "green" if cache_pct >= 50 else "yellow" if cache_pct >= 20 else "red"
        )

        econ_table = Table(show_header=True, show_edge=True)
        econ_table.add_column("Economics", style="bold", width=18)
        econ_table.add_column("", justify="right", width=12)
        econ_table.add_column("", width=3)
        econ_table.add_column("Context", style="bold", width=14)
        econ_table.add_column("", justify="right", width=12)

        pos = econ.positional
        g = econ.input_growth
        bloat_str = (
            "[red]YES[/red]"
            if g.context_bloat_detected
            else ("[yellow]WATCH[/yellow]" if g.is_superlinear else "[green]NO[/green]")
        )

        left_rows = [
            ("Input Tokens", f"{econ.total_input_tokens:,}"),
            ("Cache Read", f"{econ.total_cache_read_tokens:,}"),
            ("Cache Hit Rate", f"[{cache_color}]{cache_pct:.1f}%[/{cache_color}]"),
        ]
        right_rows_e = [
            ("Growth", f"{g.growth_rate:.1f}x ({len(g.turn_input_tokens)} turns)"),
            ("Bloat", bloat_str),
            (
                "Positional",
                f"{pos.early_ter:.2f} / {pos.mid_ter:.2f} / {pos.late_ter:.2f}",
            ),
        ]

        for i in range(3):
            l_label, l_value = left_rows[i]
            r_label, r_value = right_rows_e[i]
            econ_table.add_row(l_label, l_value, "", r_label, r_value)
        console.print(econ_table)

    _format_waste_breakdown_rich(console, result)

    if result.input_analysis is not None:
        _format_input_analysis_rich(console, result.input_analysis)

    if result.cost_report is not None:
        _format_cost_report_rich(console, result.cost_report)

    if result.overthinking_result is not None:
        _format_overthinking_rich(console, result.overthinking_result)

    return buf.getvalue().rstrip()


def _format_waste_breakdown_rich(console, result: TERResult) -> None:
    from rich.table import Table
    from .formatter import _build_waste_breakdown, _compute_waste_cost

    rows = _build_waste_breakdown(result)
    if not rows:
        return

    total_waste = sum(t for _, t, _, _ in rows)
    cm = result.economics.cost_model if result.economics else CostModel()

    table = Table(show_header=True, show_edge=True, title="Waste Breakdown")
    table.add_column("Source", style="bold", width=22)
    table.add_column("Tokens", justify="right", width=10)
    table.add_column("%", justify="right", width=6)
    table.add_column("Cost", justify="right", width=8)
    table.add_column("Count", justify="right", width=6, style="dim")

    for label, tokens, count, kind in rows:
        pct = (tokens / total_waste * 100) if total_waste > 0 else 0
        rate = cm.output_rate if kind == "output" else cm.input_rate
        row_cost = float(tokens) * rate / 1_000_000
        table.add_row(
            label,
            f"{tokens:,}",
            f"{pct:.0f}%",
            f"${row_cost:.4f}",
            str(count),
        )

    table.add_section()
    total_cost = _compute_waste_cost(result)
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_waste:,}[/bold]",
        "[bold]100%[/bold]",
        f"[bold]${total_cost:.4f}[/bold]",
        "",
    )
    console.print(table)


def format_comparison_rich(results: list[TERResult]) -> str:
    from rich.console import Console
    from rich.table import Table
    from .formatter import _compute_waste_cost

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
        console.print(
            f"\nAverage TER: [{color}]{avg_ter:.2f}[/{color}]  |  Total Cost: ${total_cost:.2f}  |  Total Waste: [red]${total_waste_cost:.2f}[/red]"
        )

    return buf.getvalue().rstrip()


def format_grouped_rich(
    parent_result: TERResult,
    subagent_results: list[TERResult],
) -> str:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from .formatter import _compute_group_aggregates, _compute_waste_cost

    all_results = [parent_result] + subagent_results
    agg = _compute_group_aggregates(all_results)

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=90)

    sid = parent_result.session_id
    if len(sid) > 20:
        sid = sid[:8] + "..."
    ter_text = Text(f"{agg['weighted_ter']:.2f}", style=_ter_color(agg["weighted_ter"]))

    header = Text.assemble(
        ("TER: ", "bold"),
        ter_text,
        ("  |  ", ""),
        (f"Waste: {agg['waste_pct']:.1f}%", "red" if agg["waste_pct"] > 10 else ""),
        ("  |  ", ""),
        (f"Cost: ${agg['total_cost_usd']:.2f}", ""),
        ("  |  ", ""),
        (f"Waste $: ${agg['total_waste_cost_usd']:.2f}", "red"),
        ("\n", ""),
        (f"Sessions: 1 parent + {len(subagent_results)} subagent(s)", "dim"),
        ("  |  ", ""),
        (f"Tokens: {agg['total_tokens']:,}", "dim"),
    )
    console.print(Panel(header, title=f"Group: {sid}", expand=False))

    table = Table(show_header=True, title="Session Breakdown")
    table.add_column("Role", width=10)
    table.add_column("Session", width=14)
    table.add_column("TER", justify="right", width=6)
    table.add_column("Waste%", justify="right", width=7)
    table.add_column("Tokens", justify="right", width=10)
    table.add_column("Cost", justify="right", width=8)
    table.add_column("Waste $", justify="right", width=8)
    table.add_column("Patterns", justify="right", width=8)

    def _add_session_row(r: TERResult, role: str):
        color = _ter_color(r.aggregate_ter)
        waste_pct = (r.waste_tokens / r.total_tokens * 100) if r.total_tokens else 0
        cost_str = f"${r.economics.estimated_cost_usd:.2f}" if r.economics else ""
        wc = _compute_waste_cost(r)
        waste_str = f"[red]${wc:.2f}[/red]" if wc > 0 else ""
        pattern_count = len(r.waste_patterns) if r.waste_patterns else 0
        rsid = r.session_id
        if len(rsid) > 14:
            rsid = rsid[:8] + "..."
        table.add_row(
            role,
            rsid,
            f"[{color}]{r.aggregate_ter:.2f}[/{color}]",
            f"{waste_pct:.1f}%",
            f"{r.total_tokens:,}",
            cost_str,
            waste_str,
            str(pattern_count),
        )

    _add_session_row(parent_result, "parent")
    for r in subagent_results:
        _add_session_row(r, "agent")

    table.add_section()
    color = _ter_color(agg["weighted_ter"])
    table.add_row(
        "[bold]Total[/bold]",
        "",
        f"[bold][{color}]{agg['weighted_ter']:.2f}[/{color}][/bold]",
        f"[bold]{agg['waste_pct']:.1f}%[/bold]",
        f"[bold]{agg['total_tokens']:,}[/bold]",
        f"[bold]${agg['total_cost_usd']:.2f}[/bold]",
        f"[bold][red]${agg['total_waste_cost_usd']:.2f}[/red][/bold]",
        "",
    )
    console.print(table)

    return buf.getvalue().rstrip()


def _format_input_analysis_rich(console, ia: InputAnalysis) -> None:
    from rich.table import Table

    bd = ia.token_breakdown
    ps = ia.prompt_similarity

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
    tb.add_row(
        "[bold]Model Total[/bold]", "", f"[bold]{bd.total_model_tokens:,}[/bold]"
    )
    tb.add_row("User Ratio", "", f"{bd.user_ratio:.1%}")
    console.print(tb)

    if ps.prompt_count >= 2:
        r_color = (
            "red"
            if ps.prompt_redundancy_score > 0.5
            else ("yellow" if ps.prompt_redundancy_score > 0 else "green")
        )
        console.print(
            f"\nPrompt Redundancy: [{r_color}]{ps.prompt_redundancy_score:.0%}[/{r_color}]"
            f"  ({ps.prompt_count} prompts, {len(ps.similar_pairs)} similar pair(s))"
        )
        for pair in ps.similar_pairs[:5]:
            a_text = (
                pair.prompt_a_text[:40] + "..."
                if len(pair.prompt_a_text) > 40
                else pair.prompt_a_text
            )
            b_text = (
                pair.prompt_b_text[:40] + "..."
                if len(pair.prompt_b_text) > 40
                else pair.prompt_b_text
            )
            console.print(
                f'  [dim]#{pair.prompt_a_index + 1}[/dim] "{a_text}" '
                f'[dim]~[/dim] [dim]#{pair.prompt_b_index + 1}[/dim] "{b_text}" '
                f"[yellow]({pair.similarity:.2f})[/yellow]"
            )

    drift = ia.intent_drift
    if drift.steps:
        _drift_colors = {
            "convergent": "red",
            "divergent": "green",
            "stable": "green",
            "mixed": "yellow",
        }
        t_color = _drift_colors.get(drift.overall_trajectory, "")
        console.print(
            f"\nIntent Drift: [{t_color}]{drift.overall_trajectory}[/{t_color}]"
            f"  (avg similarity: {drift.average_drift:.2f})"
        )
        for step in drift.steps:
            s_color = (
                "red"
                if step.drift_type == "convergent"
                else ("green" if step.drift_type == "divergent" else "yellow")
            )
            console.print(
                f"  #{step.from_index + 1} -> #{step.to_index + 1}: "
                f"[{s_color}]{step.drift_type}[/{s_color}] ({step.similarity:.2f})"
            )

    pra = ia.prompt_response_alignment
    if pra.pairs:
        a_color = (
            "red"
            if pra.average_alignment < 0.3
            else ("yellow" if pra.average_alignment < 0.5 else "green")
        )
        console.print(
            f"\nPrompt-Response Alignment: [{a_color}]{pra.average_alignment:.2f}[/{a_color}]"
            f"  ({len(pra.pairs)} pair(s), {pra.low_alignment_count} low)"
        )
        for alignment_pair in pra.pairs:
            p_color = (
                "red"
                if alignment_pair.alignment < 0.3
                else ("yellow" if alignment_pair.alignment < 0.5 else "green")
            )
            prompt_short = (
                alignment_pair.prompt_text[:50] + "..."
                if len(alignment_pair.prompt_text) > 50
                else alignment_pair.prompt_text
            )
            console.print(
                f'  [dim]#{alignment_pair.prompt_index + 1}[/dim] "{prompt_short}" '
                f"-> [{p_color}]{alignment_pair.alignment:.2f}[/{p_color}]"
            )


def _format_cost_report_rich(console, cost_report) -> None:
    from rich.table import Table

    console.print("\n[bold]Cost Analysis[/bold]")

    cost_table = Table(show_header=True, show_edge=True)
    cost_table.add_column("Metric", style="cyan", width=20)
    cost_table.add_column("Value", justify="right", width=16)

    cwter = cost_report.cost_ter
    cost_table.add_row("Cost-Weighted TER", f"{cwter.cost_weighted_ter:.4f}")
    cost_table.add_row("Raw TER", f"{cwter.raw_ter:.4f}")
    cost_table.add_row("Total Cost", f"${cwter.total_cost_usd:.4f}")
    cost_table.add_row("Waste Cost", f"${cwter.waste_cost_usd:.4f}")
    waste_pct = (
        (cwter.waste_cost_usd / cwter.total_cost_usd * 100)
        if cwter.total_cost_usd > 0
        else 0
    )
    cost_table.add_row("Waste %", f"{waste_pct:.1f}%")
    cost_table.add_row(
        "Semantic Density", f"{cost_report.session_density.density_score:.2%}"
    )
    cost_table.add_row(
        "Redundancy", f"{cost_report.session_density.redundancy_ratio:.2%}"
    )

    console.print(cost_table)

    if cost_report.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in cost_report.recommendations:
            console.print(f"  • {rec}")


def _format_overthinking_rich(console, ot) -> None:
    from rich.table import Table

    console.print("\n[bold]Overthinking Analysis[/bold]")

    status_color = "red" if ot.is_overthinking else "green"
    status_text = (
        "OVERTHINKING DETECTED" if ot.is_overthinking else "Efficient Reasoning"
    )
    console.print(f"Status: [{status_color}]{status_text}[/{status_color}]")

    ot_table = Table(show_header=True, show_edge=True)
    ot_table.add_column("Metric", style="cyan", width=20)
    ot_table.add_column("Value", justify="right", width=16)

    ot_table.add_row("Total Reasoning", f"{ot.total_reasoning_tokens:,} tokens")
    ot_table.add_row("Useful", f"{ot.useful_reasoning_tokens:,} tokens")
    ot_table.add_row("Efficiency", f"{ot.reasoning_efficiency:.0%}")
    ot_table.add_row("Wasted", f"{ot.wasted_reasoning_tokens:,} tokens")

    if ot.optimal_cutoff_index is not None:
        ot_table.add_row(
            "Optimal Cutoff", f"Span {ot.optimal_cutoff_index} (of {len(ot.segments)})"
        )

    ot_table.add_row("Recommended Budget", f"{ot.recommended_budget:,} tokens")

    console.print(ot_table)
    console.print(f"\n{ot.explanation}")
