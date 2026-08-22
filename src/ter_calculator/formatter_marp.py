"""Marp slide deck generation from TER analysis results.

Produces a Marp-compatible Markdown file with embedded SVG charts and
speaker notes summarizing session analysis findings.
"""

from __future__ import annotations

from .charts import (
    chart_composition,
    chart_economics,
    chart_key_metrics,
    chart_phase_scores,
    chart_positional_ter,
    chart_waste_breakdown,
    chart_waste_patterns,
)
from .formatter import _compute_waste_cost
from .models import TERResult


def _slide_separator() -> str:
    return "\n---\n\n"


def _sanitize_comment(text: str) -> str:
    """Escape sequences that could break out of an HTML comment."""
    return text.replace("--", "‐‐").replace(">", "›")


def _sanitize_md(text: str) -> str:
    """Neutralize Markdown/HTML metacharacters in session-derived strings."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _speaker_notes(text: str) -> str:
    lines = text.strip().split("\n")
    return "\n".join(
        f"<!-- {_sanitize_comment(line.strip())} -->"
        for line in lines
        if line.strip()
    )


def _slide_title(result: TERResult) -> str:
    lines = []
    lines.append("<!-- _class: lead -->")
    lines.append("")
    lines.append("# Token Efficiency Report")
    lines.append("")
    lines.append(f"**Session:** `{result.session_id}`")
    lines.append("")
    ter_score = f"{result.aggregate_ter:.2f}"
    waste_pct = (
        f"{result.waste_tokens / result.total_tokens * 100:.1f}%"
        if result.total_tokens
        else "0%"
    )
    summary = f"**TER Score: {ter_score}** | Waste: {waste_pct}"
    if result.economics:
        summary += f" | Est. Cost: ${result.economics.estimated_cost_usd:.4f}"
    lines.append(summary)
    lines.append("")
    lines.append(
        _speaker_notes(
            f"This report analyzes session {result.session_id}.\n"
            f"The overall Token Efficiency Ratio is {ter_score}, meaning "
            f"{waste_pct} of scored tokens were classified as waste.\n"
            f"Total tokens scored: {result.total_tokens:,}."
        )
    )
    return "\n".join(lines)


def _slide_key_metrics(result: TERResult) -> str:
    lines = []
    lines.append("## Key Metrics")
    lines.append("")

    svg = chart_key_metrics(result)
    if svg:
        lines.append(svg)
        lines.append("")

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| TER Score | {result.aggregate_ter:.3f} |")
    lines.append(f"| Total Tokens | {result.total_tokens:,} |")
    lines.append(f"| Aligned | {result.aligned_tokens:,} |")
    lines.append(f"| Waste | {result.waste_tokens:,} |")
    if result.economics:
        lines.append(
            f"| Est. Cost | ${result.economics.estimated_cost_usd:.4f} |"
        )
        lines.append(f"| Cache Hit Rate | {result.economics.cache_hit_rate * 100:.1f}% |")
    lines.append("")

    notes_lines = [
        f"The session scored {result.total_tokens:,} total tokens.",
        f"{result.aligned_tokens:,} were aligned to user intent,",
        f"{result.waste_tokens:,} were classified as waste.",
    ]
    if result.uncertainty:
        notes_lines.append(
            f"Reliability grade: {result.uncertainty.reliability}."
        )
    lines.append(_speaker_notes("\n".join(notes_lines)))
    return "\n".join(lines)


def _slide_waste_breakdown(result: TERResult) -> str:
    lines = []
    lines.append("## Token Allocation")
    lines.append("")

    svg = chart_waste_breakdown(result)
    if svg:
        lines.append(svg)
        lines.append("")

    if result.classified_spans:
        comp_svg = chart_composition(result)
        if comp_svg:
            lines.append(comp_svg)
            lines.append("")

    waste_pct = (
        result.waste_tokens / result.total_tokens * 100
        if result.total_tokens
        else 0.0
    )
    lines.append(
        _speaker_notes(
            f"{waste_pct:.1f}% of tokens were waste.\n"
            f"The composition chart breaks this down by classification label,\n"
            f"showing which phases contribute most to waste."
        )
    )
    return "\n".join(lines)


def _slide_phase_scores(result: TERResult) -> str:
    lines = []
    lines.append("## Phase Analysis")
    lines.append("")

    svg = chart_phase_scores(result)
    if svg:
        lines.append(svg)
        lines.append("")

    for phase, score in result.phase_scores.items():
        label = phase.replace("_", " ").title()
        bar = int(score * 20) * "█" + int((1 - score) * 20) * "░"
        lines.append(f"- **{label}**: {score:.3f} `{bar}`")
    lines.append("")

    lowest_phase = min(result.phase_scores, key=result.phase_scores.get)
    highest_phase = max(result.phase_scores, key=result.phase_scores.get)
    lines.append(
        _speaker_notes(
            f"Phase scores show per-phase token efficiency.\n"
            f"Strongest phase: {highest_phase.replace('_', ' ')} "
            f"at {result.phase_scores[highest_phase]:.3f}.\n"
            f"Weakest phase: {lowest_phase.replace('_', ' ')} "
            f"at {result.phase_scores[lowest_phase]:.3f}.\n"
            f"Focus optimization efforts on the weakest phase."
        )
    )
    return "\n".join(lines)


def _slide_waste_patterns(result: TERResult) -> str:
    if not result.waste_patterns:
        return ""

    lines = []
    lines.append("## Waste Patterns")
    lines.append("")

    svg = chart_waste_patterns(result)
    if svg:
        lines.append(svg)
        lines.append("")

    sorted_patterns = sorted(
        result.waste_patterns, key=lambda p: p.tokens_wasted, reverse=True
    )[:5]
    for p in sorted_patterns:
        label = p.pattern_type.replace("_", " ").title()
        desc = _sanitize_md(p.description[:100]) + ("..." if len(p.description) > 100 else "")
        lines.append(f"- **{label}** — {p.tokens_wasted:,} tokens: {desc}")
    lines.append("")

    total_waste_tokens = sum(p.tokens_wasted for p in result.waste_patterns)
    lines.append(
        _speaker_notes(
            f"Detected {len(result.waste_patterns)} waste patterns "
            f"totalling {total_waste_tokens:,} tokens.\n"
            f"The largest pattern is {sorted_patterns[0].pattern_type.replace('_', ' ')}."
        )
    )
    return "\n".join(lines)


def _slide_economics(result: TERResult) -> str:
    if not result.economics:
        return ""

    lines = []
    lines.append("## Token Economics")
    lines.append("")

    svg = chart_economics(result)
    if svg:
        lines.append(svg)
        lines.append("")

    e = result.economics
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Input Tokens | {e.total_input_tokens:,} |")
    lines.append(f"| Output Tokens | {e.total_output_tokens:,} |")
    lines.append(f"| Cache Hit Rate | {e.cache_hit_rate * 100:.1f}% |")
    lines.append(f"| Input/Output Ratio | {e.input_output_ratio:.2f} |")
    lines.append(f"| Est. Cost | ${e.estimated_cost_usd:.4f} |")
    waste_cost = _compute_waste_cost(result)
    lines.append(f"| Waste Cost | ${waste_cost:.4f} |")
    lines.append("")

    pos_svg = chart_positional_ter(result)
    if pos_svg:
        lines.append(pos_svg)
        lines.append("")

    notes = [
        f"Session cost: ${e.estimated_cost_usd:.4f}.",
        f"Waste cost: ${waste_cost:.4f}.",
        f"Cache hit rate: {e.cache_hit_rate * 100:.1f}%.",
    ]
    if e.input_growth.context_bloat_detected:
        notes.append("Context bloat detected — input tokens grew superlinearly.")
    lines.append(_speaker_notes("\n".join(notes)))
    return "\n".join(lines)


def _slide_recommendations(result: TERResult) -> str:
    lines = []
    lines.append("## Recommendations")
    lines.append("")

    recs: list[str] = []

    lowest_phase = min(result.phase_scores, key=result.phase_scores.get)
    recs.append(
        f"Tighten prompts or add CLAUDE.md rules targeting "
        f"**{lowest_phase.replace('_', ' ')}** (lowest phase TER)"
    )

    if result.economics and result.economics.cache_hit_rate < 0.5:
        recs.append(
            "Stabilize prompt prefixes to improve cache hit rate "
            f"(currently {result.economics.cache_hit_rate * 100:.0f}%)"
        )

    if result.waste_patterns:
        top = max(result.waste_patterns, key=lambda p: p.tokens_wasted)
        recs.append(
            f"Address **{top.pattern_type.replace('_', ' ')}** first "
            f"({top.tokens_wasted:,} tokens — largest waste bucket)"
        )

    if result.economics and result.economics.input_growth.context_bloat_detected:
        recs.append(
            "Investigate context bloat — input tokens are growing superlinearly"
        )

    ia = result.input_analysis
    if ia and ia.prompt_similarity.prompt_redundancy_score > 0.2:
        recs.append(
            "Consolidate prompts — redundancy score is "
            f"{ia.prompt_similarity.prompt_redundancy_score:.2f}"
        )

    if result.overthinking_result and result.overthinking_result.is_overthinking:
        recs.append(
            "Reduce reasoning budget — overthinking detected "
            f"(efficiency: {result.overthinking_result.reasoning_efficiency:.2f})"
        )

    for i, rec in enumerate(recs, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")

    lines.append(
        _speaker_notes(
            f"Key action items based on the analysis.\n"
            f"{len(recs)} recommendations identified.\n"
            f"Focus on the first item for maximum impact."
        )
    )
    return "\n".join(lines)


def _slide_closing() -> str:
    lines = []
    lines.append("<!-- _class: lead -->")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("*Generated by TER Calculator*")
    lines.append("")
    lines.append(
        "Token Efficiency Ratio analysis helps identify waste and optimize "
        "Claude Code session efficiency."
    )
    lines.append("")
    lines.append(
        _speaker_notes(
            "This concludes the TER analysis presentation.\n"
            "The TER score and waste patterns provide actionable guidance\n"
            "for improving token efficiency in future sessions."
        )
    )
    return "\n".join(lines)


def format_marp(result: TERResult) -> str:
    """Generate a complete Marp slide deck from a TERResult."""
    frontmatter = [
        "---",
        "marp: true",
        "theme: default",
        "paginate: true",
        "style: |",
        "  section { font-family: system-ui, -apple-system, sans-serif; }",
        "  h1 { color: #182033; }",
        "  h2 { color: #182033; border-bottom: 2px solid #dfe3ec; padding-bottom: 8px; }",
        "  table { font-size: 0.85em; }",
        "  svg { max-width: 100%; height: auto; }",
        "---",
        "",
    ]

    slides = [
        _slide_title(result),
        _slide_key_metrics(result),
        _slide_waste_breakdown(result),
        _slide_phase_scores(result),
    ]

    waste_slide = _slide_waste_patterns(result)
    if waste_slide:
        slides.append(waste_slide)

    econ_slide = _slide_economics(result)
    if econ_slide:
        slides.append(econ_slide)

    slides.append(_slide_recommendations(result))
    slides.append(_slide_closing())

    separator = _slide_separator()
    return "\n".join(frontmatter) + separator.join(slides) + "\n"
