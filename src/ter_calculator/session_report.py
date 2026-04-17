"""Human-readable Markdown reports for TER (closes the loop from JSON to action)."""

from __future__ import annotations

from .formatter import _compute_waste_cost
from .models import TERResult


def format_session_report_markdown(result: TERResult) -> str:
    """One-screen Markdown summary: headline metrics, calibration, top waste, next steps."""
    lines: list[str] = []
    sid = result.session_id
    lines.append(f"# TER report: `{sid}`")
    lines.append("")

    waste_pct = (
        (result.waste_tokens / result.total_tokens * 100) if result.total_tokens else 0.0
    )
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **TER** | {result.aggregate_ter:.3f} |")
    lines.append(f"| **Waste** | {result.waste_tokens:,} / {result.total_tokens:,} tokens ({waste_pct:.1f}%) |")
    if result.economics:
        e = result.economics
        lines.append(f"| **Est. session cost** | ${e.estimated_cost_usd:.4f} |")
        wc = _compute_waste_cost(result)
        lines.append(f"| **Waste $ (breakdown)** | ${wc:.4f} |")
        lines.append(
            f"| **Output calibration ratio** | {e.waste_output_calibration_ratio:.4f} "
            f"(≈ billed output ÷ heuristic assistant spans; near **1.0** is good) |"
        )
        lines.append(f"| **Cache hit rate** | {e.cache_hit_rate * 100:.1f}% |")
        pos = e.positional
        lines.append(
            f"| **Positional TER** | early {pos.early_ter:.2f} / mid {pos.mid_ter:.2f} / late {pos.late_ter:.2f} |"
        )
        if e.input_growth.context_bloat_detected:
            lines.append("| **Context growth** | **BLOAT** detected |")
        else:
            lines.append(f"| **Context growth** | rate {e.input_growth.growth_rate:.2f}× |")
    lines.append("")

    lines.append("## Phase scores")
    lines.append("")
    for phase, score in result.phase_scores.items():
        lines.append(f"- **{phase}**: {score:.3f}")
    lines.append("")

    patterns = sorted(
        result.waste_patterns,
        key=lambda p: p.tokens_wasted,
        reverse=True,
    )[:6]
    if patterns:
        lines.append("## Top structural waste patterns")
        lines.append("")
        for p in patterns:
            lines.append(
                f"- **{p.pattern_type}** — ~{p.tokens_wasted} tokens ({p.description[:120]}{'…' if len(p.description) > 120 else ''})"
            )
        lines.append("")

    lines.append("## Suggested next steps")
    lines.append("")
    lines.append(
        "- Tighten prompts or add `CLAUDE.md` rules for the **lowest phase TER** above."
    )
    if result.economics and result.economics.cache_hit_rate < 0.5:
        lines.append(
            "- **Cache hit rate** is low — stabilize prompt prefixes so Claude Code can cache."
        )
    if patterns:
        top = patterns[0].pattern_type
        lines.append(f"- Address **{top}** first (largest structural waste bucket).")
    ia = result.input_analysis
    if (
        ia
        and ia.prompt_similarity.prompt_redundancy_score > 0.2
    ):
        lines.append(
            "- **Prompt redundancy** flagged — consolidate user asks to reduce back-and-forth."
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*Interpretation limits: TER uses heuristics and `len(text)//4` spans; waste $ uses "
        "calibration to API `output_tokens` when present. See main README “Limits of interpretation”.*"
    )
    return "\n".join(lines)


def format_baseline_markdown(
    before: TERResult,
    after: TERResult,
    label_before: str = "Before",
    label_after: str = "After",
) -> str:
    """Compare two sessions (e.g. rules change) — narrative + delta table."""
    lines: list[str] = []
    lines.append("# TER baseline comparison")
    lines.append("")
    lines.append(f"- **{label_before}**: `{before.session_id}`")
    lines.append(f"- **{label_after}**: `{after.session_id}`")
    lines.append("")

    lines.append(f"| Metric | {label_before} | {label_after} | Δ |")
    lines.append("|--------|------------|-----------|---|")
    dt = after.aggregate_ter - before.aggregate_ter
    lines.append(
        f"| **TER** | {before.aggregate_ter:.4f} | {after.aggregate_ter:.4f} | {dt:+.4f} |"
    )
    wc_a = (before.waste_tokens / before.total_tokens) if before.total_tokens else 0.0
    wc_b = (after.waste_tokens / after.total_tokens) if after.total_tokens else 0.0
    lines.append(
        f"| **Waste ratio** | {wc_a:.4f} | {wc_b:.4f} | {wc_b - wc_a:+.4f} |"
    )
    if before.economics and after.economics:
        dc = after.economics.estimated_cost_usd - before.economics.estimated_cost_usd
        lines.append(
            f"| **Est. cost** | ${before.economics.estimated_cost_usd:.4f} | "
            f"${after.economics.estimated_cost_usd:.4f} | ${dc:+.4f} |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Positive **Δ TER** means the later session is more token-efficient on this heuristic. "
        "Compare similar tasks only; different tasks make raw TER incomparable."
    )
    lines.append("")
    return "\n".join(lines)
