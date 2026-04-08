"""Intent extraction from user prompts."""

from __future__ import annotations

import numpy as np

from .models import IntentVector, Session

# Lazy-loaded model to avoid import-time download.
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def extract_intent(session: Session) -> IntentVector:
    """Extract user intent from session prompts.

    Combines user prompts with later messages weighted higher (refinements
    are more specific), generates a 384-dim embedding, computes confidence
    based on prompt quality, and returns an IntentVector.
    """
    prompts = session.user_prompts
    if not prompts:
        return IntentVector(
            text="",
            embedding=np.zeros(384),
            confidence=0.0,
            source_prompts=[],
        )

    combined_text = _combine_prompts_weighted(prompts)
    model = _get_model()
    embedding = model.encode(combined_text, convert_to_numpy=True)

    confidence = _compute_confidence(prompts)

    return IntentVector(
        text=combined_text,
        embedding=embedding,
        confidence=confidence,
        source_prompts=list(prompts),
    )


def embed_text(text: str) -> np.ndarray:
    """Generate embedding for a single text string."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for multiple texts (batched)."""
    if not texts:
        return np.zeros((0, 384))
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True)


def _combine_prompts_weighted(prompts: list[str]) -> str:
    """Combine prompts with later messages weighted higher.

    Later messages are refinements and tend to be more specific,
    so we repeat them to increase their influence on the embedding.
    """
    if len(prompts) == 1:
        return prompts[0]

    # Weight later prompts more heavily by repeating them.
    # First prompt appears once, subsequent prompts appear twice.
    parts: list[str] = [prompts[0]]
    for prompt in prompts[1:]:
        parts.extend([prompt, prompt])

    return " ".join(parts)


def _compute_confidence(prompts: list[str]) -> float:
    """Compute confidence in intent extraction based on prompt quality.

    Factors: total word count, number of prompts, and prompt specificity.
    """
    if not prompts:
        return 0.0

    combined = " ".join(prompts)
    word_count = len(combined.split())

    # Base confidence from word count.
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

    # Bonus for multiple prompts (user refined their request).
    if len(prompts) > 1:
        refinement_bonus = min(0.1, len(prompts) * 0.03)
        base = min(0.95, base + refinement_bonus)

    return round(base, 2)
