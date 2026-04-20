"""Token span classification using contextual heuristics and cosine similarity.

Classification philosophy:
- Spans are ALIGNED BY DEFAULT. Most agent actions are purposeful.
- A span is only WASTE if we can identify a specific waste signal:
  1. It closely duplicates a prior span in the same phase (self-repetition)
  2. It's a reasoning span that rehashes without introducing new concepts
  3. It's a generation span that restates what was already said
- Cosine similarity to intent is used as a SIGNAL, not a binary gate.
  Low similarity doesn't mean waste — it means the span is indirect.
"""

from __future__ import annotations

import numpy as np

from .intent import embed_texts
from .models import (
    ClassifiedSpan,
    IntentVector,
    SpanLabel,
    SpanPhase,
    TokenSpan,
)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def classify_spans(
    spans: list[TokenSpan],
    intent: IntentVector,
    similarity_threshold: float = 0.40,
    confidence_threshold: float = 0.75,
) -> list[ClassifiedSpan]:
    """Classify token spans using contextual analysis.

    The approach:
    1. Compute intent similarity for each span (informational, not decisive)
    2. Compute self-similarity between consecutive same-phase spans
       to detect repetition
    3. Apply phase-specific heuristics to identify waste

    Spans are aligned by default. Only flagged as waste when a specific
    waste signal is detected.
    """
    if not spans:
        return []

    # Batch-embed all span texts.
    span_texts = [s.text for s in spans]
    embeddings = embed_texts(span_texts)

    for i, span in enumerate(spans):
        span.embedding = embeddings[i]

    # Compute intent similarity for all spans.
    intent_sims = [
        cosine_similarity(embeddings[i], intent.embedding)
        for i in range(len(spans))
    ]

    # Classify each span with full context.
    classified: list[ClassifiedSpan] = []
    prior_by_phase: dict[SpanPhase, list[int]] = {p: [] for p in SpanPhase}

    for i, span in enumerate(spans):
        sim = intent_sims[i]

        # Check for self-repetition against recent same-phase spans.
        is_repetition, rep_sim = _check_repetition(
            i, span.phase, embeddings, prior_by_phase
        )

        # Classify based on phase + signals.
        label, conf = _classify_span(
            sim=sim,
            phase=span.phase,
            is_repetition=is_repetition,
            repetition_similarity=rep_sim,
            similarity_threshold=similarity_threshold,
            confidence_threshold=confidence_threshold,
            span_text=span.text,
        )

        classified.append(ClassifiedSpan(
            span=span,
            label=label,
            confidence=conf,
            cosine_similarity=sim,
        ))

        prior_by_phase[span.phase].append(i)

    return classified


def _check_repetition(
    current_idx: int,
    phase: SpanPhase,
    embeddings: np.ndarray,
    prior_by_phase: dict[SpanPhase, list[int]],
    window: int = 10,
    repetition_threshold: float = 0.88,
) -> tuple[bool, float]:
    """Check if a span closely duplicates a recent same-phase span.

    Returns (is_repetition, highest_similarity_to_prior).
    """
    prior_indices = prior_by_phase[phase]
    if not prior_indices:
        return False, 0.0

    # Check against recent prior spans in the same phase.
    check_indices = prior_indices[-window:]
    max_sim = 0.0

    for idx in check_indices:
        sim = cosine_similarity(embeddings[current_idx], embeddings[idx])
        max_sim = max(max_sim, sim)

    return max_sim >= repetition_threshold, max_sim


def _classify_span(
    sim: float,
    phase: SpanPhase,
    is_repetition: bool,
    repetition_similarity: float,
    similarity_threshold: float,
    confidence_threshold: float,
    span_text: str,
) -> tuple[SpanLabel, float]:
    """Classify a single span using multiple signals.

    Default: aligned. Waste only if a specific signal fires.
    """
    # Signal 1: Self-repetition (strongest waste signal).
    if is_repetition:
        # Require strong agreement with a prior span; avoids borderline
        # embeddings being scored as duplicate work.
        if repetition_similarity < confidence_threshold:
            if phase == SpanPhase.REASONING:
                return SpanLabel.ALIGNED_REASONING, max(0.5, sim)
            if phase == SpanPhase.TOOL_USE:
                return SpanLabel.ALIGNED_TOOL_CALL, max(0.6, sim)
            return SpanLabel.ALIGNED_RESPONSE, max(0.5, sim)
        confidence = repetition_similarity
        if phase == SpanPhase.REASONING:
            return SpanLabel.REDUNDANT_REASONING, confidence
        if phase == SpanPhase.TOOL_USE:
            return SpanLabel.UNNECESSARY_TOOL_CALL, confidence
        return SpanLabel.OVER_EXPLANATION, confidence

    # Signal 2: Very low intent similarity + phase-specific checks.
    # Only for reasoning and generation — tool calls are actions,
    # not words, so low semantic similarity is expected and normal.
    # Bounds keep defaults close to legacy 0.10 / 0.08 when threshold≈0.40.
    filler_sim_max = max(0.06, min(0.14, similarity_threshold * 0.28))
    verbose_sim_max = max(0.05, min(0.12, similarity_threshold * 0.22))

    if phase == SpanPhase.REASONING:
        # Reasoning with very low relevance AND short text (filler).
        if sim < filler_sim_max and len(span_text.split()) < 15:
            return SpanLabel.REDUNDANT_REASONING, 0.5
        return SpanLabel.ALIGNED_REASONING, max(0.5, sim)

    if phase == SpanPhase.TOOL_USE:
        # Tool calls are almost always intentional. The agent chose
        # to invoke a tool — that's an action, not idle chatter.
        return SpanLabel.ALIGNED_TOOL_CALL, max(0.6, sim)

    if phase == SpanPhase.GENERATION:
        # Generation with extremely low relevance is suspicious,
        # but only if it's also substantial (short responses are fine).
        if sim < verbose_sim_max and len(span_text.split()) > 50:
            return SpanLabel.OVER_EXPLANATION, 0.4
        return SpanLabel.ALIGNED_RESPONSE, max(0.5, sim)

    # Fallback: aligned.
    return SpanLabel.ALIGNED_RESPONSE, max(0.5, sim)
