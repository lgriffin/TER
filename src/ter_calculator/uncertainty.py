"""Deterministic confidence and uncertainty estimates for TER results."""

from __future__ import annotations

import random

from .models import (
    ALIGNED_LABELS,
    ClassifiedSpan,
    PHASE_WEIGHTS_DEFAULT,
    SpanPhase,
    UncertaintyReport,
)


def _weighted_ter(sample: list[ClassifiedSpan]) -> float:
    aligned = {phase: 0 for phase in SpanPhase}
    total = {phase: 0 for phase in SpanPhase}
    for classified in sample:
        phase = classified.span.phase
        tokens = classified.span.token_count
        total[phase] += tokens
        if classified.label in ALIGNED_LABELS:
            aligned[phase] += tokens
    return sum(
        PHASE_WEIGHTS_DEFAULT[phase]
        * (aligned[phase] / total[phase] if total[phase] else 1.0)
        for phase in SpanPhase
    )


def estimate_uncertainty(
    classified_spans: list[ClassifiedSpan],
    *,
    low_confidence_threshold: float = 0.65,
    bootstrap_samples: int = 400,
    seed: int = 17,
) -> UncertaintyReport:
    """Estimate confidence and a reproducible span-bootstrap TER interval."""
    if not classified_spans:
        return UncertaintyReport(
            1.0, 1.0, 0, 0.0, 1.0, 1.0, bootstrap_samples, 0, reliability="low"
        )

    total_tokens = sum(max(0, item.span.token_count) for item in classified_spans)
    mean_confidence = sum(item.confidence for item in classified_spans) / len(
        classified_spans
    )
    weighted_confidence = (
        sum(
            item.confidence * max(0, item.span.token_count) for item in classified_spans
        )
        / total_tokens
        if total_tokens
        else mean_confidence
    )
    low_tokens = sum(
        max(0, item.span.token_count)
        for item in classified_spans
        if item.confidence < low_confidence_threshold
    )

    rng = random.Random(seed)
    values: list[float] = []
    count = len(classified_spans)
    for _ in range(max(1, bootstrap_samples)):
        sample = [classified_spans[rng.randrange(count)] for _ in range(count)]
        values.append(_weighted_ter(sample))
    values.sort()
    lower_idx = int(0.025 * (len(values) - 1))
    upper_idx = int(0.975 * (len(values) - 1))
    reliability = "low" if count < 10 else ("moderate" if count < 50 else "high")

    return UncertaintyReport(
        mean_confidence=round(mean_confidence, 4),
        token_weighted_confidence=round(weighted_confidence, 4),
        low_confidence_tokens=low_tokens,
        low_confidence_share=round(low_tokens / total_tokens, 4)
        if total_tokens
        else 0.0,
        interval_lower=round(values[lower_idx], 4),
        interval_upper=round(values[upper_idx], 4),
        bootstrap_samples=max(1, bootstrap_samples),
        span_count=count,
        reliability=reliability,
    )
