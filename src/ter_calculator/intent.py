"""Intent extraction from user prompts."""

from __future__ import annotations

import numpy as np

from .embedding_cache import get_embedding_model
from .intent_construction import (
    compute_prompt_weights,
    detect_topic_shifts,
    intent_display_text,
    split_prompt_topics,
    weighted_centroid,
)
from .models import IntentVector, Session


def extract_intent(session: Session) -> IntentVector:
    """Extract a weighted intent vector from independently embedded prompts.

    Informative prompts receive explicit information, recency, and correction
    weights. Operational acknowledgements are strongly down-weighted. The
    returned embedding represents the latest detected topic so an unrelated
    earlier task cannot blur the active goal. ``source_prompts`` retains the
    complete audit trail for compatibility and explainability.
    """
    prompts = session.user_prompts
    if not prompts:
        return IntentVector(
            text="",
            embedding=np.zeros(384),
            confidence=0.0,
            source_prompts=[],
        )

    model = get_embedding_model()
    embeddings = np.asarray(model.encode(prompts, convert_to_numpy=True))
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    shifts = detect_topic_shifts(prompts, embeddings)
    topics = split_prompt_topics(prompts, shifts)
    active_prompts = topics[-1] if topics else prompts
    active_start = len(prompts) - len(active_prompts)
    active_embeddings = embeddings[active_start:]
    weights = compute_prompt_weights(active_prompts)
    embedding = weighted_centroid(active_embeddings, weights)

    confidence = _compute_confidence(active_prompts)
    if shifts:
        confidence = max(0.0, round(confidence - 0.05, 2))

    return IntentVector(
        text=intent_display_text(prompts),
        embedding=embedding,
        confidence=confidence,
        source_prompts=list(prompts),
    )


def extract_intent_topics(
    session: Session, *, split_threshold: float = 0.42
) -> list[IntentVector]:
    """Return one weighted intent vector per detected topic in the session."""
    prompts = session.user_prompts
    if not prompts:
        return [
            IntentVector(
                text="", embedding=np.zeros(384), confidence=0.0, source_prompts=[]
            )
        ]

    embeddings = np.asarray(
        get_embedding_model().encode(prompts, convert_to_numpy=True)
    )
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    shifts = detect_topic_shifts(prompts, embeddings, threshold=split_threshold)
    topics = split_prompt_topics(prompts, shifts)

    intents: list[IntentVector] = []
    offset = 0
    for topic in topics:
        topic_embeddings = embeddings[offset : offset + len(topic)]
        offset += len(topic)
        intents.append(
            IntentVector(
                text=intent_display_text(topic),
                embedding=weighted_centroid(
                    topic_embeddings, compute_prompt_weights(topic)
                ),
                confidence=_compute_confidence(topic),
                source_prompts=list(topic),
            )
        )
    return intents


def embed_text(text: str) -> np.ndarray:
    """Generate embedding for a single text string."""
    return np.asarray(get_embedding_model().encode(text, convert_to_numpy=True))


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for multiple texts (batched)."""
    if not texts:
        return np.zeros((0, 384))
    return np.asarray(get_embedding_model().encode(texts, convert_to_numpy=True))


def _combine_prompts_weighted(prompts: list[str]) -> str:
    """Legacy text-combination helper retained for external compatibility.

    New intent extraction does not use repeated text. Callers should migrate to
    ``compute_prompt_weights`` and ``weighted_centroid``.
    """
    if len(prompts) == 1:
        return prompts[0]
    parts: list[str] = [prompts[0]] if prompts else []
    for prompt in prompts[1:]:
        parts.extend([prompt, prompt])
    return " ".join(parts)


def _compute_confidence(prompts: list[str]) -> float:
    """Compute confidence from prompt information and refinement quality."""
    if not prompts:
        return 0.0

    informative = [
        item for item in compute_prompt_weights(prompts) if not item.is_operational
    ]
    selected = informative or compute_prompt_weights(prompts)
    word_count = sum(len(item.prompt.split()) for item in selected)

    if word_count <= 1:
        base = 0.2
    elif word_count <= 2:
        base = 0.3
    elif word_count <= 5:
        base = 0.5
    elif word_count <= 10:
        base = 0.7
    else:
        base = 0.85

    if len(informative) > 1:
        base = min(0.95, base + min(0.1, len(informative) * 0.03))
    if informative and any(item.is_correction for item in informative):
        base = min(0.95, base + 0.03)
    return round(base, 2)
