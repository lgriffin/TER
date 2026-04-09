"""Session economics: token usage, cost estimation, positional TER, and input growth."""

from __future__ import annotations

from .models import (
    ALIGNED_LABELS,
    ClassifiedSpan,
    CostModel,
    InputGrowth,
    PositionalBreakdown,
    Session,
    SessionEconomics,
    SpanLabel,
)


def compute_economics(
    session: Session,
    classified_spans: list[ClassifiedSpan],
    cost_model: CostModel | None = None,
) -> SessionEconomics:
    """Compute session economics from raw usage data and classified spans."""
    model = cost_model or CostModel()

    input_tok, output_tok, cache_create, cache_read = _aggregate_usage(session)
    cache_hit = _compute_cache_hit_rate(cache_read, input_tok)
    io_ratio = input_tok / output_tok if output_tok > 0 else 0.0
    cost = _estimate_cost(input_tok, output_tok, cache_read, cache_create, model)
    waste_cost = _estimate_waste_cost(classified_spans, model)
    positional = _compute_positional_breakdown(classified_spans)
    growth = _compute_input_growth(session)

    return SessionEconomics(
        total_input_tokens=input_tok,
        total_output_tokens=output_tok,
        total_cache_creation_tokens=cache_create,
        total_cache_read_tokens=cache_read,
        input_output_ratio=round(io_ratio, 2),
        cache_hit_rate=round(cache_hit, 4),
        estimated_cost_usd=round(cost, 4),
        estimated_waste_cost_usd=round(waste_cost, 4),
        cost_model=model,
        positional=positional,
        input_growth=growth,
    )


def _aggregate_usage(session: Session) -> tuple[int, int, int, int]:
    """Sum token usage fields across assistant messages."""
    input_tok = 0
    output_tok = 0
    cache_create = 0
    cache_read = 0

    for msg in session.messages:
        if msg.role != "assistant" or msg.usage is None:
            continue
        input_tok += msg.usage.input_tokens
        output_tok += msg.usage.output_tokens
        cache_create += msg.usage.cache_creation_input_tokens
        cache_read += msg.usage.cache_read_input_tokens

    return input_tok, output_tok, cache_create, cache_read


def _compute_cache_hit_rate(cache_read: int, input_tokens: int) -> float:
    """Cache hit rate: fraction of non-created input that came from cache.

    Anthropic API: input_tokens = non-cached input, cache_read = separate.
    Rate = cache_read / (cache_read + input_tokens).
    """
    total = cache_read + input_tokens
    if total == 0:
        return 0.0
    return cache_read / total


def _estimate_cost(
    input_tok: int,
    output_tok: int,
    cache_read_tok: int,
    cache_write_tok: int,
    cost_model: CostModel,
) -> float:
    """Estimate session cost in USD using per-million-token rates."""
    return (
        input_tok * cost_model.input_rate / 1_000_000
        + output_tok * cost_model.output_rate / 1_000_000
        + cache_read_tok * cost_model.cache_read_rate / 1_000_000
        + cache_write_tok * cost_model.cache_write_rate / 1_000_000
    )


def _estimate_waste_cost(
    classified_spans: list[ClassifiedSpan],
    cost_model: CostModel,
) -> float:
    """Estimate the dollar cost of waste output tokens."""
    waste_tokens = sum(
        cs.span.token_count for cs in classified_spans
        if cs.label not in ALIGNED_LABELS
    )
    return waste_tokens * cost_model.output_rate / 1_000_000


def _compute_positional_breakdown(
    classified_spans: list[ClassifiedSpan],
) -> PositionalBreakdown:
    """Split spans into thirds and compute TER per segment."""
    n = len(classified_spans)
    if n == 0:
        return PositionalBreakdown(
            early_ter=1.0, mid_ter=1.0, late_ter=1.0,
            early_span_count=0, mid_span_count=0, late_span_count=0,
        )

    third = n // 3
    # Ensure at least 1 span in early when n < 3
    if third == 0:
        third = 1

    early = classified_spans[:third]
    mid = classified_spans[third:2 * third]
    late = classified_spans[2 * third:]

    return PositionalBreakdown(
        early_ter=round(_segment_ter(early), 4),
        mid_ter=round(_segment_ter(mid), 4) if mid else 1.0,
        late_ter=round(_segment_ter(late), 4) if late else 1.0,
        early_span_count=len(early),
        mid_span_count=len(mid),
        late_span_count=len(late),
    )


def _segment_ter(spans: list[ClassifiedSpan]) -> float:
    """Compute aligned ratio for a segment of spans."""
    total = sum(cs.span.token_count for cs in spans)
    if total == 0:
        return 1.0
    aligned = sum(
        cs.span.token_count for cs in spans if cs.label in ALIGNED_LABELS
    )
    return aligned / total


def _compute_input_growth(session: Session) -> InputGrowth:
    """Track context size growth across assistant turns.

    Uses total context (input_tokens + cache_read_input_tokens) because
    input_tokens alone only counts non-cached tokens, which drops to
    near-zero once caching kicks in and doesn't reflect actual context size.
    """
    turn_contexts = [
        msg.usage.input_tokens + msg.usage.cache_read_input_tokens
        for msg in session.messages
        if msg.role == "assistant" and msg.usage is not None
    ]

    # Filter out tiny-context turns (initial handshake, setup).
    # The first few turns often have minimal context before caching kicks in.
    turn_contexts = [t for t in turn_contexts if t > 100]

    if len(turn_contexts) < 2:
        return InputGrowth(
            turn_input_tokens=turn_contexts,
            growth_rate=1.0,
            is_superlinear=False,
            context_bloat_detected=False,
        )

    first = turn_contexts[0]
    last = turn_contexts[-1]
    growth_rate = last / first if first > 0 else 0.0

    # Detect super-linear growth via second differences.
    is_superlinear = False
    if len(turn_contexts) >= 3:
        deltas = [
            turn_contexts[i + 1] - turn_contexts[i]
            for i in range(len(turn_contexts) - 1)
        ]
        second_deltas = [
            deltas[i + 1] - deltas[i]
            for i in range(len(deltas) - 1)
        ]
        avg_second = sum(second_deltas) / len(second_deltas)
        is_superlinear = avg_second > 0

    context_bloat = is_superlinear and growth_rate > 2.0

    return InputGrowth(
        turn_input_tokens=turn_contexts,
        growth_rate=round(growth_rate, 2),
        is_superlinear=is_superlinear,
        context_bloat_detected=context_bloat,
    )
