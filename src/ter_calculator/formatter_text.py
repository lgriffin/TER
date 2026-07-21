"""Plain text formatting for TER results."""

from __future__ import annotations

from .models import CostModel, InputAnalysis, TERResult


def format_text(result: TERResult) -> str:
    from .formatter import _build_waste_breakdown, _compute_waste_cost

    waste_pct = (
        (result.waste_tokens / result.total_tokens * 100) if result.total_tokens else 0
    )
    sid = result.session_id
    if len(sid) > 20:
        sid = sid[:8] + "..."

    lines = [
        f"TER Report: {sid}",
        "═" * 40,
        "",
    ]

    cost_str = ""
    if result.economics:
        cost_str = f"  |  Cost: ${result.economics.estimated_cost_usd:.2f}"
        waste_cost = _compute_waste_cost(result)
        if waste_cost > 0:
            cost_str += f"  |  Waste $: ${waste_cost:.2f}"
    lines.append(
        f"TER: {result.aggregate_ter:.2f}  |  Waste: {waste_pct:.1f}%{cost_str}"
    )
    if result.uncertainty is not None:
        uncertainty = result.uncertainty
        lines.append(
            f"95% interval: {uncertainty.interval_lower:.2f}–{uncertainty.interval_upper:.2f}"
            f"  |  Confidence: {uncertainty.token_weighted_confidence:.0%}"
            f"  |  Low-confidence tokens: {uncertainty.low_confidence_share:.1%}"
            f"  |  Reliability: {uncertainty.reliability}"
        )

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

    lines.append("Phases:     Reasoning  Tool Use  Generation")
    lines.append(
        f"            {result.phase_scores.get('reasoning', 0):.2f}"
        f"       {result.phase_scores.get('tool_use', 0):.2f}"
        f"      {result.phase_scores.get('generation', 0):.2f}"
    )
    lines.append("")

    lines.append(
        f"Output Tokens: {result.total_tokens:,}  (aligned: {result.aligned_tokens:,}  waste: {result.waste_tokens:,})"
    )

    if result.economics is not None:
        econ = result.economics
        cache_pct = econ.cache_hit_rate * 100
        pos = econ.positional
        g = econ.input_growth

        lines.extend(
            [
                "",
                f"Input: {econ.total_input_tokens:,}  Cache Read: {econ.total_cache_read_tokens:,}  Cache Hit: {cache_pct:.1f}%",
                f"Context Growth: {g.growth_rate:.1f}x over {len(g.turn_input_tokens)} turns"
                + (
                    " [BLOAT]"
                    if g.context_bloat_detected
                    else (" [WATCH]" if g.is_superlinear else "")
                ),
                f"Positional TER: {pos.early_ter:.2f} (early) / {pos.mid_ter:.2f} (mid) / {pos.late_ter:.2f} (late)",
            ]
        )

    rows = _build_waste_breakdown(result)
    if rows:
        total_waste = sum(t for _, t, _, _ in rows)
        cm = result.economics.cost_model if result.economics else CostModel()
        lines.extend(["", "Waste Breakdown:"])
        lines.append(
            f"  {'Source':<24} {'Tokens':>10} {'%':>5} {'Cost':>10} {'Count':>6}"
        )
        for label, tokens, count, kind in rows:
            pct = (tokens / total_waste * 100) if total_waste > 0 else 0
            rate = cm.output_rate if kind == "output" else cm.input_rate
            row_cost = float(tokens) * rate / 1_000_000
            lines.append(
                f"  {label:<24} {tokens:>10,} {pct:>4.0f}% ${row_cost:>8.4f} {count:>6}"
            )
        total_cost = _compute_waste_cost(result)
        lines.append(f"  {'Total':<24} {total_waste:>10,}  100% ${total_cost:>8.4f}")

    if result.input_analysis is not None:
        lines.extend(_format_input_analysis_text(result.input_analysis))

    explained = [
        item for item in result.classified_spans if item.explanation is not None
    ]
    if explained:
        uncertain_or_waste = sorted(
            explained,
            key=lambda item: (item.label.value.startswith("aligned_"), item.confidence),
        )[:5]
        lines.extend(["", "Classification Evidence:"])
        for item in uncertain_or_waste:
            explanation = item.explanation
            assert explanation is not None
            prior = (
                f"; matched prior #{explanation.matched_prior_position}"
                if explanation.matched_prior_position is not None
                else ""
            )
            lines.append(
                f"  #{item.span.position} {item.label.value} ({item.confidence:.2f}): "
                f"{explanation.summary}{prior}"
            )

    return "\n".join(lines)


def format_comparison_text(results: list[TERResult]) -> str:
    from .formatter import _compute_waste_cost

    lines = [
        "TER Comparison",
        "═" * 40,
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
        lines.extend(
            [
                "",
                f"Average TER: {avg_ter:.2f}  |  Total Cost: ${total_cost:.2f}  |  Total Waste: ${total_waste_cost:.2f}",
            ]
        )

    return "\n".join(lines)


def format_grouped_text(
    parent_result: TERResult,
    subagent_results: list[TERResult],
) -> str:
    from .formatter import _compute_group_aggregates, _compute_waste_cost

    all_results = [parent_result] + subagent_results
    agg = _compute_group_aggregates(all_results)

    sid = parent_result.session_id
    if len(sid) > 20:
        sid = sid[:8] + "..."

    lines = [
        f"Group Analysis: {sid}",
        "═" * 50,
        "",
        f"TER: {agg['weighted_ter']:.2f}  |  Waste: {agg['waste_pct']:.1f}%"
        f"  |  Cost: ${agg['total_cost_usd']:.2f}"
        f"  |  Waste $: ${agg['total_waste_cost_usd']:.2f}",
        f"Sessions: 1 parent + {len(subagent_results)} subagent(s)  |  Tokens: {agg['total_tokens']:,}",
        "",
        f"  {'Role':<10} {'Session':<14} {'TER':<6} {'Waste%':<8} {'Tokens':<10} {'Cost':<10} {'Waste $':<10} {'Patterns':<8}",
    ]

    def _add_row(r: TERResult, role: str):
        rsid = (
            r.session_id[:14] if len(r.session_id) <= 14 else r.session_id[:8] + "..."
        )
        waste_pct = (r.waste_tokens / r.total_tokens * 100) if r.total_tokens else 0
        cost_str = f"${r.economics.estimated_cost_usd:.2f}" if r.economics else ""
        wc = _compute_waste_cost(r)
        waste_str = f"${wc:.2f}" if wc > 0 else ""
        pattern_count = len(r.waste_patterns) if r.waste_patterns else 0
        lines.append(
            f"  {role:<10} {rsid:<14} {r.aggregate_ter:<6.2f} "
            f"{waste_pct:<8.1f} {r.total_tokens:<10,} {cost_str:<10} {waste_str:<10} {pattern_count:<8}"
        )

    _add_row(parent_result, "[parent]")
    for r in subagent_results:
        _add_row(r, "[agent]")

    lines.extend(
        [
            "",
            f"  {'Total':<10} {'':<14} {agg['weighted_ter']:<6.2f} "
            f"{agg['waste_pct']:<8.1f} {agg['total_tokens']:<10,} "
            f"${agg['total_cost_usd']:<9.2f} ${agg['total_waste_cost_usd']:<9.2f}",
        ]
    )

    return "\n".join(lines)


def _format_input_analysis_text(ia: InputAnalysis) -> list[str]:
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
        lines.append(
            f"  Prompt Redundancy: {ps.prompt_redundancy_score:.0%} ({ps.prompt_count} prompts, {len(ps.similar_pairs)} similar pair(s))"
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
            lines.append(
                f'    #{pair.prompt_a_index + 1} "{a_text}" ~ #{pair.prompt_b_index + 1} "{b_text}" ({pair.similarity:.2f})'
            )

    drift = ia.intent_drift
    if drift.steps:
        lines.append(
            f"  Intent Drift: {drift.overall_trajectory} (avg similarity: {drift.average_drift:.2f})"
        )
        for step in drift.steps:
            lines.append(
                f"    #{step.from_index + 1} -> #{step.to_index + 1}: {step.drift_type} ({step.similarity:.2f})"
            )

    pra = ia.prompt_response_alignment
    if pra.pairs:
        lines.append(
            f"  Prompt-Response Alignment: {pra.average_alignment:.2f} ({len(pra.pairs)} pair(s), {pra.low_alignment_count} low)"
        )
        for alignment_pair in pra.pairs:
            prompt_short = (
                alignment_pair.prompt_text[:50] + "..."
                if len(alignment_pair.prompt_text) > 50
                else alignment_pair.prompt_text
            )
            marker = " [LOW]" if alignment_pair.alignment < 0.3 else ""
            lines.append(
                f'    #{alignment_pair.prompt_index + 1} "{prompt_short}" '
                f"-> {alignment_pair.alignment:.2f}{marker}"
            )

    return lines
