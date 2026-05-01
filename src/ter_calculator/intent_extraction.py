"""Advanced intent extraction strategies for multi-turn sessions.

The baseline intent module (intent.py) creates a single embedding for all
user prompts, which loses nuance in multi-turn sessions where user goals
evolve or branch. This module provides three improved strategies:

- SlidingIntentExtractor: groups prompts by proximity/topic into segments,
  producing one IntentVector per segment so spans are scored against the
  nearest (most relevant) intent rather than a blurred global one.

- HierarchicalIntentExtractor: extracts a high-level intent from the first
  prompt and sub-intents from follow-ups, scoring spans against the most
  specific applicable intent.

- LLMIntentExtractor: optionally uses Claude to summarise user intent as a
  structured goal statement before embedding, falling back to direct
  embedding when no API key is available.

All extractors implement the IntentStrategy protocol and can be swapped via
the create_intent_extractor factory function.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .models import IntentVector

__all__ = [
    "IntentStrategy",
    "SlidingIntentExtractor",
    "HierarchicalIntentExtractor",
    "LLMIntentExtractor",
    "StructuredGoal",
    "create_intent_extractor",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding helpers (lazy-loaded, mirrors intent.py pattern)
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    """Lazily load the sentence-transformers model."""
    global _model
    if _model is None:
        import warnings

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        for name in ("huggingface_hub", "transformers", "sentence_transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _embed(text: str) -> NDArray[np.float32]:
    """Embed a single text string into a 384-dim vector."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def _embed_batch(texts: list[str]) -> NDArray[np.float32]:
    """Embed multiple texts in a single batched call."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True)


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class IntentStrategy(Protocol):
    """Protocol that all intent extractors implement.

    The extract method takes a list of user prompt strings and returns a
    list of IntentVectors.  Downstream classification picks the nearest
    vector for each span.
    """

    def extract(self, prompts: list[str]) -> list[IntentVector]: ...


# ---------------------------------------------------------------------------
# Structured goal (used by LLMIntentExtractor)
# ---------------------------------------------------------------------------


@dataclass
class StructuredGoal:
    """Structured representation of user intent produced by LLM analysis."""

    primary_goal: str
    sub_goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Flatten the structured goal into a single text for embedding."""
        parts = [self.primary_goal]
        if self.sub_goals:
            parts.append("Sub-goals: " + "; ".join(self.sub_goals))
        if self.constraints:
            parts.append("Constraints: " + "; ".join(self.constraints))
        if self.expected_outputs:
            parts.append("Expected outputs: " + "; ".join(self.expected_outputs))
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# SlidingIntentExtractor
# ---------------------------------------------------------------------------


class SlidingIntentExtractor:
    """Create multiple intent vectors using a sliding window over prompts.

    Prompts are grouped into segments based on semantic similarity between
    consecutive prompts.  When adjacent prompts diverge beyond
    *split_threshold*, a new segment begins.  Each segment produces its own
    IntentVector so that spans can be scored against the closest intent.

    Parameters
    ----------
    window_size:
        Maximum number of prompts in a single segment.  If a segment grows
        beyond this size it is split even when prompts remain similar.
    split_threshold:
        Cosine similarity below which consecutive prompts are placed into
        separate segments (0-1, lower = more willing to split).
    """

    def __init__(
        self,
        window_size: int = 5,
        split_threshold: float = 0.45,
    ) -> None:
        self.window_size = max(1, window_size)
        self.split_threshold = split_threshold

    # -- public interface (IntentStrategy) ----------------------------------

    def extract(self, prompts: list[str]) -> list[IntentVector]:
        """Segment prompts by topic proximity and return per-segment intents."""
        if not prompts:
            return [_empty_intent()]

        if len(prompts) == 1:
            emb = _embed(prompts[0])
            confidence = _prompt_confidence(prompts[0])
            return [
                IntentVector(
                    text=prompts[0],
                    embedding=emb,
                    confidence=confidence,
                    source_prompts=[prompts[0]],
                )
            ]

        segments = self._segment_prompts(prompts)
        return [self._segment_to_intent(seg) for seg in segments]

    # -- internal -----------------------------------------------------------

    def _segment_prompts(self, prompts: list[str]) -> list[list[str]]:
        """Split prompts into topic-coherent segments."""
        embeddings = _embed_batch(prompts)

        segments: list[list[str]] = [[prompts[0]]]
        segment_start_idx = 0

        for i in range(1, len(prompts)):
            sim = _cosine_similarity(embeddings[i], embeddings[i - 1])
            segment_len = i - segment_start_idx

            if sim < self.split_threshold or segment_len >= self.window_size:
                # Start a new segment.
                segments.append([prompts[i]])
                segment_start_idx = i
            else:
                segments[-1].append(prompts[i])

        return segments

    @staticmethod
    def _segment_to_intent(segment: list[str]) -> IntentVector:
        """Build an IntentVector from a group of related prompts."""
        combined = " ".join(segment)
        emb = _embed(combined)
        confidence = _segment_confidence(segment)
        return IntentVector(
            text=combined,
            embedding=emb,
            confidence=confidence,
            source_prompts=list(segment),
        )


# ---------------------------------------------------------------------------
# HierarchicalIntentExtractor
# ---------------------------------------------------------------------------


class HierarchicalIntentExtractor:
    """Extract a hierarchy of intents: one high-level goal plus sub-intents.

    The first user prompt is treated as the overall goal.  Follow-up prompts
    are sub-intents (refinements/clarifications).  When scoring a span, use
    ``score_span`` to compare against the most specific applicable intent.

    Parameters
    ----------
    sub_intent_weight:
        Blending weight for sub-intents when computing the final similarity
        score.  A value of 0.7 means 70% sub-intent, 30% high-level intent.
    """

    def __init__(self, sub_intent_weight: float = 0.7) -> None:
        self.sub_intent_weight = max(0.0, min(1.0, sub_intent_weight))

    # -- public interface (IntentStrategy) ----------------------------------

    def extract(self, prompts: list[str]) -> list[IntentVector]:
        """Return [high_level_intent, *sub_intents].

        The first element is always the high-level intent derived from the
        first prompt.  Subsequent elements are sub-intents from follow-up
        prompts.  If there is only one prompt, the list has a single entry.
        """
        if not prompts:
            return [_empty_intent()]

        high_level = self._build_high_level(prompts[0])
        if len(prompts) == 1:
            return [high_level]

        sub_intents = self._build_sub_intents(prompts[1:])
        return [high_level, *sub_intents]

    # -- scoring helper (not part of protocol, but useful downstream) -------

    def score_span(
        self,
        span_embedding: NDArray[np.float32],
        intents: list[IntentVector],
    ) -> tuple[float, IntentVector]:
        """Score a span against the most specific applicable intent.

        Compares the span to every intent and returns the best blended score
        together with the matching intent.  The blended score mixes the
        high-level intent similarity with the best sub-intent similarity
        using *sub_intent_weight*.

        Returns
        -------
        (score, best_intent) where score is in [-1, 1].
        """
        if not intents:
            return 0.0, _empty_intent()

        high_level = intents[0]
        high_sim = _cosine_similarity(span_embedding, high_level.embedding)

        if len(intents) == 1:
            return high_sim, high_level

        # Find the best sub-intent match.
        best_sub_sim = -1.0
        best_sub = intents[1]
        for sub in intents[1:]:
            sim = _cosine_similarity(span_embedding, sub.embedding)
            if sim > best_sub_sim:
                best_sub_sim = sim
                best_sub = sub

        # Blend: if the best sub-intent is a strong match, weight it highly.
        w = self.sub_intent_weight
        blended = w * best_sub_sim + (1.0 - w) * high_sim

        # Return the intent that contributed more to the score.
        if best_sub_sim >= high_sim:
            return blended, best_sub
        return blended, high_level

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _build_high_level(prompt: str) -> IntentVector:
        """Build the high-level intent from the first prompt."""
        emb = _embed(prompt)
        confidence = _prompt_confidence(prompt)
        return IntentVector(
            text=prompt,
            embedding=emb,
            confidence=confidence,
            source_prompts=[prompt],
        )

    @staticmethod
    def _build_sub_intents(follow_ups: list[str]) -> list[IntentVector]:
        """Build sub-intents from follow-up prompts."""
        if not follow_ups:
            return []

        embeddings = _embed_batch(follow_ups)
        sub_intents: list[IntentVector] = []
        for i, prompt in enumerate(follow_ups):
            confidence = _prompt_confidence(prompt)
            sub_intents.append(
                IntentVector(
                    text=prompt,
                    embedding=embeddings[i],
                    confidence=confidence,
                    source_prompts=[prompt],
                )
            )
        return sub_intents


# ---------------------------------------------------------------------------
# LLMIntentExtractor
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an intent-extraction assistant. Given a sequence of user prompts \
from a coding session, produce a JSON object with exactly these keys:
  "primary_goal": a single sentence describing the user's main objective,
  "sub_goals": a list of secondary objectives or refinements,
  "constraints": a list of constraints the user specified (languages, \
frameworks, performance requirements, etc.),
  "expected_outputs": a list of concrete deliverables the user expects \
(files, functions, tests, etc.)
Respond ONLY with the JSON object, no markdown fencing or commentary.\
"""


class LLMIntentExtractor:
    """Use Claude to summarise user intent before embedding.

    If an Anthropic API key is available (via *api_key* or the
    ``ANTHROPIC_API_KEY`` environment variable), the user prompts are sent
    to Claude which returns a :class:`StructuredGoal`.  The goal text is
    then embedded to produce the IntentVector.

    When no API key is available, falls back to direct embedding of the
    concatenated prompts (identical to the baseline approach).

    Parameters
    ----------
    api_key:
        Anthropic API key.  If ``None``, falls back to ``ANTHROPIC_API_KEY``
        environment variable.
    model:
        Claude model to use for summarisation.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    # -- public interface (IntentStrategy) ----------------------------------

    def extract(self, prompts: list[str]) -> list[IntentVector]:
        """Extract intent, optionally via LLM summarisation.

        Returns a single-element list containing one IntentVector built from
        the structured goal (or direct embedding as fallback).
        """
        if not prompts:
            return [_empty_intent()]

        goal = self._summarise(prompts)
        if goal is not None:
            text = goal.to_embedding_text()
            emb = _embed(text)
            return [
                IntentVector(
                    text=text,
                    embedding=emb,
                    confidence=0.95,  # LLM-derived intents are high confidence
                    source_prompts=list(prompts),
                )
            ]

        # Fallback: direct embedding.
        combined = " ".join(prompts)
        emb = _embed(combined)
        confidence = _segment_confidence(prompts)
        return [
            IntentVector(
                text=combined,
                embedding=emb,
                confidence=confidence,
                source_prompts=list(prompts),
            )
        ]

    # -- LLM interaction ----------------------------------------------------

    @property
    def structured_goal(self) -> StructuredGoal | None:
        """The most recently extracted structured goal, if any."""
        return getattr(self, "_last_goal", None)

    def _summarise(self, prompts: list[str]) -> StructuredGoal | None:
        """Ask Claude to produce a StructuredGoal from user prompts."""
        if not self.api_key:
            logger.debug("No API key available; falling back to direct embedding.")
            return None

        try:
            client = self._get_client()
            user_content = "User prompts:\n" + "\n---\n".join(prompts)

            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )

            raw_text = response.content[0].text
            data = json.loads(raw_text)
            goal = StructuredGoal(
                primary_goal=data.get("primary_goal", ""),
                sub_goals=data.get("sub_goals", []),
                constraints=data.get("constraints", []),
                expected_outputs=data.get("expected_outputs", []),
            )
            self._last_goal = goal
            return goal

        except Exception:
            logger.warning(
                "LLM intent extraction failed; falling back to direct embedding.",
                exc_info=True,
            )
            return None

    def _get_client(self):
        """Lazily create the Anthropic client."""
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for LLM-assisted intent "
                    "extraction. Install it with: pip install anthropic"
                )
        return self._client


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_intent_extractor(strategy: str = "sliding", **kwargs) -> IntentStrategy:
    """Create an intent extractor for the given strategy name.

    Parameters
    ----------
    strategy:
        One of ``"sliding"``, ``"hierarchical"``, or ``"llm"``.
    **kwargs:
        Forwarded to the chosen extractor's constructor.

    Returns
    -------
    An object satisfying :class:`IntentStrategy`.

    Raises
    ------
    ValueError
        If *strategy* is not a recognised name.
    """
    strategies: dict[str, type] = {
        "sliding": SlidingIntentExtractor,
        "hierarchical": HierarchicalIntentExtractor,
        "llm": LLMIntentExtractor,
    }
    cls = strategies.get(strategy)
    if cls is None:
        valid = ", ".join(sorted(strategies))
        raise ValueError(
            f"Unknown intent extraction strategy {strategy!r}. "
            f"Valid strategies: {valid}"
        )
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _empty_intent() -> IntentVector:
    """Return an IntentVector representing no extractable intent."""
    return IntentVector(
        text="",
        embedding=np.zeros(384, dtype=np.float32),
        confidence=0.0,
        source_prompts=[],
    )


def _prompt_confidence(prompt: str) -> float:
    """Estimate confidence for a single prompt based on length/specificity."""
    word_count = len(prompt.split())
    if word_count <= 1:
        return 0.2
    if word_count <= 2:
        return 0.3
    if word_count <= 5:
        return 0.5
    if word_count <= 10:
        return 0.7
    return 0.85


def _segment_confidence(prompts: list[str]) -> float:
    """Confidence for a segment of prompts (aggregated)."""
    if not prompts:
        return 0.0

    # Average individual confidences, with a small bonus for multiple prompts.
    individual = [_prompt_confidence(p) for p in prompts]
    base = sum(individual) / len(individual)

    if len(prompts) > 1:
        refinement_bonus = min(0.1, len(prompts) * 0.03)
        base = min(0.95, base + refinement_bonus)

    return round(base, 2)
