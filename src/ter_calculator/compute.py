"""TER score computation."""

from __future__ import annotations

from .models import (
    ALIGNED_LABELS,
    PHASE_WEIGHTS_DEFAULT,
    ClassifiedSpan,
    SpanPhase,
    TERResult,
    IntentVector,
)


def compute_ter(
    classified_spans: list[ClassifiedSpan],
    session_id: str,
    intent: IntentVector | None = None,
    phase_weights: dict[SpanPhase, float] | None = None,
) -> TERResult:
    """Compute the Token Efficiency Ratio from classified spans.

    Returns per-phase scores, weighted aggregate TER, raw ratio,
    and token counts.
    """
    weights = phase_weights or PHASE_WEIGHTS_DEFAULT

    phase_aligned: dict[SpanPhase, int] = {p: 0 for p in SpanPhase}
    phase_total: dict[SpanPhase, int] = {p: 0 for p in SpanPhase}

    for cs in classified_spans:
        phase = cs.span.phase
        phase_total[phase] += cs.span.token_count
        if cs.label in ALIGNED_LABELS:
            phase_aligned[phase] += cs.span.token_count

    # Per-phase scores.
    phase_scores: dict[str, float] = {}
    for phase in SpanPhase:
        total = phase_total[phase]
        if total > 0:
            phase_scores[phase.value] = round(
                phase_aligned[phase] / total, 4
            )
        else:
            phase_scores[phase.value] = 1.0  # No tokens → no waste.

    # Weighted aggregate TER.
    aggregate_ter = sum(
        weights[phase] * phase_scores[phase.value]
        for phase in SpanPhase
    )

    total_aligned = sum(phase_aligned.values())
    total_all = sum(phase_total.values())
    raw_ratio = total_aligned / total_all if total_all > 0 else 1.0

    return TERResult(
        session_id=session_id,
        aggregate_ter=round(aggregate_ter, 4),
        raw_ratio=round(raw_ratio, 4),
        phase_scores=phase_scores,
        total_tokens=total_all,
        aligned_tokens=total_aligned,
        waste_tokens=total_all - total_aligned,
        intent=intent,
        classified_spans=list(classified_spans),
    )
