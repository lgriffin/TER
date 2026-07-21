"""Token span classification using contextual heuristics and cosine similarity.

Classification philosophy:
- Spans are ALIGNED BY DEFAULT. Most agent actions are purposeful.
- A span is only WASTE if we can identify a specific waste signal:
  1. It closely duplicates a prior span in the same phase (self-repetition)
  2. It's a reasoning span that rehashes without introducing new concepts
  3. It's a generation span that restates what was already said
- Cosine similarity to intent is used as a SIGNAL, not a binary gate.
  Low similarity doesn't mean waste — it means the span is indirect.

Gold set calibration (session 711bb9b1, 60 spans, May 2026):
- Waste and aligned reasoning spans have overlapping sim ranges (0.16–0.46).
  Similarity alone is therefore insufficient; token count is the stronger signal.
- Short reasoning spans (<25 words, sim<0.35) are almost universally action
  narrations ("Let me read X", "Good! Now I'll Y") with no analytical content.
- Repetition detection mis-fires on spans containing new specifics (line numbers,
  identifiers). A _has_specific_reference guard prevents false positives there.
- System artifacts ("[Request interrupted]") should be excluded before
  classification.
"""

from __future__ import annotations

import re

import numpy as np

from .intent import embed_texts
from .tool_fingerprints import (
    ToolCallFingerprint,
    build_tool_fingerprint,
    compare_tool_calls,
)
from .repetition_scoring import score_text_repetition, score_tool_repetition
from .models import (
    ClassifiedSpan,
    ClassificationExplanation,
    IntentVector,
    SpanLabel,
    SpanPhase,
    TokenSpan,
)

# Matches line-number references and long integers (e.g. "line 341", "lines 715, 722").
_SPECIFIC_REF_RE = re.compile(r"\blines?\s+\d+|\b\d{3,}\b")

# System artifact patterns that should never be classified as waste.
_SYSTEM_ARTIFACT_RE = re.compile(r"^\s*\[Request interrupted", re.IGNORECASE)

# Per-phase repetition thresholds.
#
# Tool calls share structural JSON (tool name, keys) regardless of topic, so
# two WebSearch calls for unrelated queries embed at ~0.85–0.92.  Using the
# same 0.88 threshold as reasoning causes false positives in heterogeneous
# sessions (session 94103fcd: dinosaur search + Man Utd kit search flagged as
# duplicate).  Raising the tool_use threshold to 0.93 means only near-identical
# calls (same query, same arguments) are flagged — genuine redundancy.
_REPETITION_THRESHOLDS: dict[SpanPhase, float] = {
    SpanPhase.REASONING: 0.88,
    SpanPhase.TOOL_USE: 0.93,
    SpanPhase.GENERATION: 0.88,
}


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def classify_spans(
    spans: list[TokenSpan],
    intent: IntentVector | list[IntentVector],
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

    # Normalise intent to a list so downstream code is uniform.
    # When a list of IntentVectors is provided (SlidingIntentExtractor output),
    # each span is scored against the nearest intent segment (max similarity).
    # This prevents blurred global intent from flagging spans from topic B as
    # waste simply because they don't match topic A's embedding.
    intent_list = intent if isinstance(intent, list) else [intent]

    # Compute intent similarity for all spans.
    intent_sims = [
        max(cosine_similarity(embeddings[i], iv.embedding) for iv in intent_list)
        for i in range(len(spans))
    ]

    # Classify each span with full context.
    classified: list[ClassifiedSpan] = []
    prior_by_phase: dict[SpanPhase, list[int]] = {p: [] for p in SpanPhase}
    prior_tool_fingerprints: list[tuple[ToolCallFingerprint, int]] = []

    for i, span in enumerate(spans):
        sim = intent_sims[i]

        # Check for self-repetition against recent same-phase spans.
        # Phase-specific threshold: tool calls need a higher bar (0.93) because
        # they are structurally similar across topics by design.
        matched_prior_idx: int | None = None
        repetition_evidence = None
        if span.phase == SpanPhase.TOOL_USE and span.tool_name:
            fingerprint = build_tool_fingerprint(span.tool_name, span.tool_input)
            rep_sim = 0.0
            for prior, prior_idx in prior_tool_fingerprints[-10:]:
                semantic = cosine_similarity(embeddings[i], embeddings[prior_idx])
                evidence = score_tool_repetition(
                    spans[prior_idx].text,
                    span.text,
                    semantic,
                    compare_tool_calls(prior, fingerprint),
                )
                if evidence.score > rep_sim:
                    rep_sim = evidence.score
                    repetition_evidence = evidence
                    matched_prior_idx = prior_idx
            is_repetition = rep_sim >= _REPETITION_THRESHOLDS[SpanPhase.TOOL_USE]
        else:
            fingerprint = None
            is_repetition, rep_sim, matched_prior_idx, repetition_evidence = (
                _find_blended_repetition(
                    i,
                    span.phase,
                    spans,
                    embeddings,
                    prior_by_phase,
                    repetition_threshold=_REPETITION_THRESHOLDS.get(span.phase, 0.88),
                )
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

        explanation = _build_explanation(
            span=span,
            label=label,
            confidence=conf,
            intent_similarity=sim,
            repetition_similarity=rep_sim,
            repetition_evidence=repetition_evidence,
            matched_prior=spans[matched_prior_idx]
            if matched_prior_idx is not None
            else None,
        )
        classified.append(
            ClassifiedSpan(
                span=span,
                label=label,
                confidence=conf,
                cosine_similarity=sim,
                explanation=explanation,
            )
        )

        prior_by_phase[span.phase].append(i)
        if fingerprint is not None:
            prior_tool_fingerprints.append((fingerprint, i))

    return classified


def _find_blended_repetition(
    current_idx: int,
    phase: SpanPhase,
    spans: list[TokenSpan],
    embeddings: np.ndarray,
    prior_by_phase: dict[SpanPhase, list[int]],
    window: int = 10,
    repetition_threshold: float = 0.88,
):
    """Return repetition decision, best score, prior index, and evidence."""
    prior_indices = prior_by_phase[phase][-window:]
    if not prior_indices:
        return False, 0.0, None, None

    max_score = 0.0
    matched_idx = None
    best_evidence = None
    for idx in prior_indices:
        semantic = cosine_similarity(embeddings[current_idx], embeddings[idx])
        evidence = score_text_repetition(
            spans[idx].text, spans[current_idx].text, semantic
        )
        if evidence.score > max_score:
            max_score = evidence.score
            matched_idx = idx
            best_evidence = evidence

    return max_score >= repetition_threshold, max_score, matched_idx, best_evidence


def _check_blended_repetition(
    current_idx: int,
    phase: SpanPhase,
    spans: list[TokenSpan],
    embeddings: np.ndarray,
    prior_by_phase: dict[SpanPhase, list[int]],
    window: int = 10,
    repetition_threshold: float = 0.88,
) -> tuple[bool, float]:
    """Compatibility wrapper returning only decision and score."""
    repeated, score, _, _ = _find_blended_repetition(
        current_idx,
        phase,
        spans,
        embeddings,
        prior_by_phase,
        window,
        repetition_threshold,
    )
    return repeated, score


def _build_explanation(
    *,
    span: TokenSpan,
    label: SpanLabel,
    confidence: float,
    intent_similarity: float,
    repetition_similarity: float,
    repetition_evidence,
    matched_prior: TokenSpan | None,
) -> ClassificationExplanation:
    """Construct stable, inspectable evidence for a classification."""
    signals = {
        "intent_similarity": round(intent_similarity, 4),
        "repetition_score": round(repetition_similarity, 4),
        "confidence": round(confidence, 4),
    }
    if repetition_evidence is not None:
        signals.update(
            {
                "semantic_similarity": round(repetition_evidence.semantic, 4),
                "lexical_similarity": round(repetition_evidence.lexical, 4),
                "entity_similarity": round(repetition_evidence.entity, 4),
                "action_similarity": round(repetition_evidence.action, 4),
                "parameter_novelty": round(repetition_evidence.parameter_novelty, 4),
            }
        )

    waste_labels = {
        SpanLabel.REDUNDANT_REASONING,
        SpanLabel.UNNECESSARY_TOOL_CALL,
        SpanLabel.OVER_EXPLANATION,
    }
    threshold = _REPETITION_THRESHOLDS.get(span.phase)
    if label in waste_labels and repetition_similarity >= (threshold or 1.0):
        reason = "repetition"
        summary = (
            "Strong repetition evidence matched an earlier span in the same phase."
        )
    elif label == SpanLabel.REDUNDANT_REASONING:
        reason = "low_information_reasoning"
        summary = (
            "The reasoning span is short and weakly aligned with the active intent."
        )
    elif label == SpanLabel.OVER_EXPLANATION:
        reason = "low_relevance_generation"
        summary = (
            "The generated response is substantial but has very low intent similarity."
        )
    elif repetition_similarity > 0.0 and threshold is not None:
        reason = "novel_or_below_threshold"
        summary = "Similar prior content exists, but novelty or insufficient evidence kept the span aligned."
    else:
        reason = "aligned_default"
        summary = "No specific waste signal exceeded the applicable threshold."

    return ClassificationExplanation(
        reason_code=reason,
        summary=summary,
        signals=signals,
        threshold=threshold,
        matched_prior_position=matched_prior.position if matched_prior else None,
        matched_prior_text=matched_prior.text[:240] if matched_prior else None,
    )


def _check_repetition(
    current_idx: int,
    phase: SpanPhase,
    embeddings: np.ndarray,
    prior_by_phase: dict[SpanPhase, list[int]],
    window: int = 10,
    repetition_threshold: float = 0.88,
) -> tuple[bool, float]:
    """Check if a span closely duplicates a recent same-phase span.

    The threshold is phase-specific — callers should pass the value from
    ``_REPETITION_THRESHOLDS`` rather than the default.

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


def _has_specific_reference(text: str) -> bool:
    """Return True if the span contains line numbers or long numeric identifiers.

    Used to protect spans like "both pass model=None on lines 341 and 358" from
    being flagged as redundant_reasoning purely on surface-form similarity — they
    contain actionable specifics even when they structurally echo earlier spans.
    """
    return bool(_SPECIFIC_REF_RE.search(text))


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
    # Guard: system artifacts are never waste — they are not agent reasoning.
    if _SYSTEM_ARTIFACT_RE.match(span_text):
        if phase == SpanPhase.REASONING:
            return SpanLabel.ALIGNED_REASONING, 0.5
        return SpanLabel.ALIGNED_RESPONSE, 0.5

    word_count = len(span_text.split())

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

        # Gold set finding: spans with specific line-number references introduce
        # new information even when surface form closely echoes a prior span.
        # Don't fire redundant_reasoning on spans with numeric specifics.
        if phase == SpanPhase.REASONING and _has_specific_reference(span_text):
            return SpanLabel.ALIGNED_REASONING, max(0.5, sim)

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
        # Tier 1 (existing): very low sim + very short → filler.
        if sim < filler_sim_max and word_count < 15:
            return SpanLabel.REDUNDANT_REASONING, 0.5

        # Tier 2 (gold set calibrated): short action-narration spans.
        # Gold set analysis of session 711bb9b1 (60 uncertain spans) showed
        # that reasoning spans with sim < 0.35 and fewer than 25 words are
        # almost exclusively action announcements ("Let me read X", "Good! Now
        # I'll add Y") with no analytical content. Aligned spans in this sim
        # range reliably have ≥ 25 words (they include analysis or rationale).
        short_narration_sim_max = 0.35
        short_narration_words_max = 25
        if (
            sim < short_narration_sim_max
            and word_count < short_narration_words_max
            and not _has_specific_reference(span_text)
        ):
            return SpanLabel.REDUNDANT_REASONING, 0.6

        return SpanLabel.ALIGNED_REASONING, max(0.5, sim)

    if phase == SpanPhase.TOOL_USE:
        # Tool calls are almost always intentional. The agent chose
        # to invoke a tool — that's an action, not idle chatter.
        return SpanLabel.ALIGNED_TOOL_CALL, max(0.6, sim)

    if phase == SpanPhase.GENERATION:
        # Generation with extremely low relevance is suspicious,
        # but only if it's also substantial (short responses are fine).
        if sim < verbose_sim_max and word_count > 50:
            return SpanLabel.OVER_EXPLANATION, 0.4
        return SpanLabel.ALIGNED_RESPONSE, max(0.5, sim)

    # Fallback: aligned.
    return SpanLabel.ALIGNED_RESPONSE, max(0.5, sim)
