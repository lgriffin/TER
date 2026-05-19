"""Output formatting for TER results.

Public API: format_ter_result, format_comparison, format_grouped_analysis.
Format-specific rendering lives in formatter_rich, formatter_text, formatter_json.
"""

from __future__ import annotations

from .models import CostModel, TERResult, WastePattern


def format_ter_result(
    result: TERResult, fmt: str = "text", use_rich: bool = True,
) -> str:
    if fmt == "json":
        from .formatter_json import format_json
        return format_json(result)
    if use_rich:
        try:
            from .formatter_rich import format_rich
            return format_rich(result)
        except (ImportError, UnicodeEncodeError):
            pass
    from .formatter_text import format_text
    return format_text(result)


def format_comparison(
    results: list[TERResult], fmt: str = "text", use_rich: bool = True,
) -> str:
    if fmt == "json":
        from .formatter_json import format_comparison_json
        return format_comparison_json(results)
    if use_rich:
        try:
            from .formatter_rich import format_comparison_rich
            return format_comparison_rich(results)
        except (ImportError, UnicodeEncodeError):
            pass
    from .formatter_text import format_comparison_text
    return format_comparison_text(results)


def format_grouped_analysis(
    parent_result: TERResult,
    subagent_results: list[TERResult],
    fmt: str = "text",
    use_rich: bool = True,
) -> str:
    if fmt == "json":
        from .formatter_json import format_grouped_json
        return format_grouped_json(parent_result, subagent_results)
    if use_rich:
        try:
            from .formatter_rich import format_grouped_rich
            return format_grouped_rich(parent_result, subagent_results)
        except (ImportError, UnicodeEncodeError):
            pass
    from .formatter_text import format_grouped_text
    return format_grouped_text(parent_result, subagent_results)


# ---------------------------------------------------------------------------
# Shared helpers used by formatter_rich, formatter_text, formatter_json
# ---------------------------------------------------------------------------


def _compute_group_aggregates(all_results: list[TERResult]) -> dict:
    total_tokens = sum(r.total_tokens for r in all_results)
    total_waste = sum(r.waste_tokens for r in all_results)
    weighted_ter = (
        sum(r.aggregate_ter * r.total_tokens for r in all_results) / total_tokens
        if total_tokens > 0 else 0.0
    )
    total_cost = sum(
        r.economics.estimated_cost_usd for r in all_results if r.economics
    )
    total_waste_cost = sum(_compute_waste_cost(r) for r in all_results)
    waste_pct = (total_waste / total_tokens * 100) if total_tokens > 0 else 0.0

    return {
        "weighted_ter": round(weighted_ter, 4),
        "total_tokens": total_tokens,
        "total_waste_tokens": total_waste,
        "waste_pct": round(waste_pct, 1),
        "total_cost_usd": round(total_cost, 4),
        "total_waste_cost_usd": round(total_waste_cost, 4),
    }


def _compute_waste_cost(result: TERResult) -> float:
    rows = _build_waste_breakdown(result)
    if not rows:
        return 0.0
    cm = result.economics.cost_model if result.economics else CostModel()
    total = 0.0
    for _label, tokens, _count, kind in rows:
        rate = cm.output_rate if kind == "output" else cm.input_rate
        total += float(tokens) * rate / 1_000_000
    return total


def _pattern_pricing(pattern_type: str) -> str:
    if pattern_type in (
        "repetitive_read",
        "bash_antipattern",
        "failed_tool_retry",
        "repeated_command",
    ):
        return "input"
    return "output"


def _build_waste_breakdown(
    result: TERResult,
) -> list[tuple[str, int, int, str]]:
    """Build rows: (label, tokens, instance_count, pricing_kind)."""
    from .models import ALIGNED_LABELS

    rows: list[tuple[str, int, int, str]] = []

    category_map = {
        "redundant_reasoning": "Redundant Reasoning",
        "unnecessary_tool_call": "Unnecessary Tool Calls",
        "over_explanation": "Over-Explanation",
    }
    cat_tokens: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    user_waste_tokens = 0
    user_waste_count = 0
    for cs in result.classified_spans:
        if cs.label in ALIGNED_LABELS:
            continue
        if cs.span.source_role != "assistant":
            user_waste_tokens += cs.span.token_count
            user_waste_count += 1
            continue
        label = cs.label.value
        cat = category_map.get(label, label.replace("_", " ").title())
        cat_tokens[cat] = cat_tokens.get(cat, 0) + cs.span.token_count
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    for cat in category_map.values():
        if cat in cat_tokens:
            rows.append((cat, cat_tokens[cat], cat_counts[cat], "output"))

    if user_waste_tokens > 0:
        rows.append((
            "User-side context (waste)",
            user_waste_tokens,
            max(1, user_waste_count),
            "input",
        ))

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
    pattern_overlap = {
        "reasoning_loop": "Redundant Reasoning",
        "duplicate_tool_call": "Unnecessary Tool Calls",
        "context_restatement": "Over-Explanation",
    }
    by_type: dict[str, list[WastePattern]] = {}
    for wp in (result.waste_patterns or []):
        by_type.setdefault(wp.pattern_type, []).append(wp)

    for ptype, wps in by_type.items():
        overlap_cat = pattern_overlap.get(ptype)
        if overlap_cat and overlap_cat in cat_tokens:
            continue
        label = pattern_labels.get(ptype, ptype.replace("_", " ").title())
        tokens = sum(wp.tokens_wasted for wp in wps)
        kind = _pattern_pricing(ptype)
        rows.append((label, tokens, len(wps), kind))

    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _format_waste_pattern(wp: WastePattern) -> str:
    pos = (
        f"spans {wp.start_position}-{wp.end_position}"
        if wp.start_position != wp.end_position
        else f"span {wp.start_position}"
    )
    label = wp.pattern_type.replace("_", " ").title()
    return f"{label} ({pos}): {wp.description}, {wp.tokens_wasted:,} tokens"
