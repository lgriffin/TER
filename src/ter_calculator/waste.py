"""Waste pattern detection.

Detects three categories of waste:
1. Reasoning loops — the agent rehashes the same reasoning multiple times
2. Duplicate tool calls — identical tool invocations within a window
3. Context restatement — response text that repeats what was already said
"""

from __future__ import annotations

from .classifier import cosine_similarity
from .models import (
    ALIGNED_LABELS,
    ClassifiedSpan,
    SpanLabel,
    SpanPhase,
    WastePattern,
)


def detect_waste_patterns(
    classified_spans: list[ClassifiedSpan],
    restatement_threshold: float = 0.85,
) -> list[WastePattern]:
    """Detect all waste patterns in classified spans."""
    patterns: list[WastePattern] = []
    patterns.extend(detect_reasoning_loops(classified_spans))
    patterns.extend(detect_duplicate_tool_calls(classified_spans))
    patterns.extend(
        detect_context_restatement(classified_spans, restatement_threshold)
    )
    return patterns


def summarize_waste(
    classified_spans: list[ClassifiedSpan],
    waste_patterns: list[WastePattern],
) -> dict:
    """Produce a human-readable waste summary.

    Returns a dict with:
    - total_waste_tokens: total tokens classified as waste
    - waste_by_category: breakdown by waste type
    - waste_by_phase: breakdown by phase
    - top_patterns: the most impactful waste patterns
    - explanation: human-readable summary string
    """
    waste_spans = [
        cs for cs in classified_spans if cs.label not in ALIGNED_LABELS
    ]

    total_waste = sum(cs.span.token_count for cs in waste_spans)
    total_all = sum(cs.span.token_count for cs in classified_spans)

    # By category.
    by_category: dict[str, int] = {}
    for cs in waste_spans:
        cat = _label_to_category(cs.label)
        by_category[cat] = by_category.get(cat, 0) + cs.span.token_count

    # By phase.
    by_phase: dict[str, int] = {}
    for cs in waste_spans:
        phase = cs.span.phase.value
        by_phase[phase] = by_phase.get(phase, 0) + cs.span.token_count

    # Top patterns by tokens wasted.
    top_patterns = sorted(
        waste_patterns, key=lambda p: p.tokens_wasted, reverse=True
    )[:5]

    # Build explanation.
    explanation = _build_explanation(
        total_waste, total_all, by_category, by_phase, top_patterns
    )

    return {
        "total_waste_tokens": total_waste,
        "waste_by_category": by_category,
        "waste_by_phase": by_phase,
        "top_patterns": [
            {
                "type": p.pattern_type,
                "tokens_wasted": p.tokens_wasted,
                "description": p.description,
            }
            for p in top_patterns
        ],
        "explanation": explanation,
    }


def _label_to_category(label: SpanLabel) -> str:
    return {
        SpanLabel.REDUNDANT_REASONING: "Redundant Reasoning",
        SpanLabel.UNNECESSARY_TOOL_CALL: "Unnecessary Tool Calls",
        SpanLabel.OVER_EXPLANATION: "Over-Explanation",
    }.get(label, "Other")


def _build_explanation(
    total_waste: int,
    total_all: int,
    by_category: dict[str, int],
    by_phase: dict[str, int],
    top_patterns: list[WastePattern],
) -> str:
    if total_all == 0:
        return "No tokens to analyze."
    if total_waste == 0:
        return "No waste detected. All tokens contributed to the task."

    pct = total_waste / total_all * 100
    lines = [
        f"{total_waste:,} of {total_all:,} tokens ({pct:.1f}%) were "
        f"identified as waste.",
    ]

    if by_category:
        biggest = max(by_category, key=by_category.get)  # type: ignore[arg-type]
        lines.append(
            f"The largest waste category is {biggest} "
            f"({by_category[biggest]:,} tokens)."
        )

    if top_patterns:
        p = top_patterns[0]
        lines.append(
            f"The most impactful pattern: {p.description} "
            f"({p.tokens_wasted:,} tokens)."
        )

    return " ".join(lines)


# --- Pattern detectors ---


def detect_reasoning_loops(
    classified_spans: list[ClassifiedSpan],
    min_consecutive: int = 3,
) -> list[WastePattern]:
    """Detect 3+ consecutive redundant reasoning spans."""
    patterns: list[WastePattern] = []
    consecutive: list[ClassifiedSpan] = []

    for cs in classified_spans:
        if (cs.span.phase == SpanPhase.REASONING
                and cs.label == SpanLabel.REDUNDANT_REASONING):
            consecutive.append(cs)
        else:
            if len(consecutive) >= min_consecutive:
                patterns.append(_make_reasoning_loop_pattern(consecutive))
            consecutive = []

    if len(consecutive) >= min_consecutive:
        patterns.append(_make_reasoning_loop_pattern(consecutive))

    return patterns


def detect_duplicate_tool_calls(
    classified_spans: list[ClassifiedSpan],
    window_size: int = 5,
) -> list[WastePattern]:
    """Detect repeated tool calls with identical name+params within a window.

    Only considers actual tool_use blocks (not tool_result), since
    duplicate results just mean the system returned similar confirmations.
    """
    patterns: list[WastePattern] = []
    seen_sigs: set[str] = set()
    tool_spans = [
        cs for cs in classified_spans
        if cs.span.phase == SpanPhase.TOOL_USE
        and cs.span.block_type == "tool_use"
    ]

    for i, cs in enumerate(tool_spans):
        sig = _get_tool_signature(cs)
        if sig is None:
            continue

        window_start = max(0, i - window_size)
        for j in range(window_start, i):
            prev_sig = _get_tool_signature(tool_spans[j])
            if prev_sig == sig:
                # Deduplicate: only report once per unique signature.
                if sig not in seen_sigs:
                    patterns.append(WastePattern(
                        pattern_type="duplicate_tool_call",
                        description=f"Duplicate tool call: {sig[:60]}",
                        start_position=tool_spans[j].span.position,
                        end_position=cs.span.position,
                        spans_involved=2,
                        tokens_wasted=cs.span.token_count,
                        details={"signature": sig},
                    ))
                    seen_sigs.add(sig)
                break

    return patterns


def detect_context_restatement(
    classified_spans: list[ClassifiedSpan],
    similarity_threshold: float = 0.85,
) -> list[WastePattern]:
    """Detect generation spans that closely repeat prior generation spans."""
    patterns: list[WastePattern] = []
    prior_gen: list[ClassifiedSpan] = []

    for cs in classified_spans:
        if cs.span.phase != SpanPhase.GENERATION:
            continue
        if cs.span.embedding is None:
            prior_gen.append(cs)
            continue

        for prior in prior_gen:
            if prior.span.embedding is None:
                continue
            sim = cosine_similarity(cs.span.embedding, prior.span.embedding)
            if sim >= similarity_threshold:
                patterns.append(WastePattern(
                    pattern_type="context_restatement",
                    description=(
                        f"Response restates prior content "
                        f"(similarity: {sim:.2f})"
                    ),
                    start_position=cs.span.position,
                    end_position=cs.span.position,
                    spans_involved=1,
                    tokens_wasted=cs.span.token_count,
                    details={
                        "similarity": round(sim, 4),
                        "prior_position": prior.span.position,
                    },
                ))
                break

        prior_gen.append(cs)

    return patterns


def _make_reasoning_loop_pattern(spans: list[ClassifiedSpan]) -> WastePattern:
    total_tokens = sum(cs.span.token_count for cs in spans)
    return WastePattern(
        pattern_type="reasoning_loop",
        description=f"{len(spans)} consecutive redundant reasoning spans",
        start_position=spans[0].span.position,
        end_position=spans[-1].span.position,
        spans_involved=len(spans),
        tokens_wasted=total_tokens,
    )


def _get_tool_signature(cs: ClassifiedSpan) -> str | None:
    text = cs.span.text
    if not text:
        return None
    return text.strip()
