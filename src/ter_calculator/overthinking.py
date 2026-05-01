"""Overthinking detector with information-theoretic analysis.

Detects when reasoning tokens stop contributing marginal value, enabling
real-time budget cutoff decisions.  Inspired by:

- Apple's "Illusion of Thinking" -- models often find the answer early
  but continue exploring, wasting reasoning tokens.
- Mutual information analysis in reasoning chains -- specific reflective
  tokens ("Wait", "Hmm", "Therefore") carry disproportionate value while
  filler tokens have near-zero information content.
- SelfBudgeter / TALE -- adaptive token budgets that scale with task
  complexity.

Key components:

- EntropyTracker: sliding-window entropy analysis of reasoning spans to
  detect when novelty plateaus.
- ReasoningPhaseClassifier: labels reasoning chunks as {exploring,
  confirming, ambiguous, near_answer} based on lexical and structural
  cues.
- OverthinkingResult: quantifies how many reasoning tokens could have
  been saved by earlier termination.
- find_optimal_cutoff: given a sequence of reasoning spans, identifies
  the point of diminishing returns.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ter_calculator.models import ClassifiedSpan, TokenSpan

__all__ = [
    "EntropyTracker",
    "OverthinkingResult",
    "ReasoningPhase",
    "ReasoningPhaseClassifier",
    "ReasoningSegment",
    "analyze_overthinking",
    "find_optimal_cutoff",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTROPY_WINDOW = 5
"""Number of reasoning spans in the sliding entropy window."""

NOVELTY_THRESHOLD = 0.15
"""Minimum normalised novelty score below which reasoning is considered stale."""

MIN_REASONING_SPANS = 3
"""Minimum spans before overthinking analysis is meaningful."""

HIGH_VALUE_TOKENS: frozenset[str] = frozenset(
    {
        "wait",
        "hmm",
        "actually",
        "but",
        "however",
        "therefore",
        "so",
        "instead",
        "alternatively",
        "correction",
        "no,",
        "let me reconsider",
        "on second thought",
    }
)
"""Tokens / phrases that carry disproportionate reasoning value per mutual
information analysis.  Their presence indicates genuine course-correction."""

FILLER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blet me think\b", re.IGNORECASE),
    re.compile(r"\bi need to\b", re.IGNORECASE),
    re.compile(r"\blet me re-read\b", re.IGNORECASE),
    re.compile(r"\blet me re-examine\b", re.IGNORECASE),
    re.compile(r"\blet me check\b", re.IGNORECASE),
    re.compile(r"\blet me look\b", re.IGNORECASE),
    re.compile(r"\bOK so\b", re.IGNORECASE),
    re.compile(r"\bso basically\b", re.IGNORECASE),
    re.compile(r"\bI should\b", re.IGNORECASE),
    re.compile(r"\bI'll need to\b", re.IGNORECASE),
)
"""Patterns that indicate reasoning filler with low information content."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReasoningPhase(Enum):
    """Coarse classification of a reasoning span's purpose."""

    EXPLORING = "exploring"
    CONFIRMING = "confirming"
    AMBIGUOUS = "ambiguous"
    NEAR_ANSWER = "near_answer"
    FILLER = "filler"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReasoningSegment:
    """Analysis of a single reasoning span."""

    index: int
    text: str
    token_count: int
    phase: ReasoningPhase
    novelty_score: float
    high_value_token_count: int
    filler_ratio: float
    cumulative_novelty: float
    marginal_value: float


@dataclass(frozen=True, slots=True)
class OverthinkingResult:
    """Summary of overthinking analysis for a session's reasoning."""

    is_overthinking: bool
    total_reasoning_tokens: int
    useful_reasoning_tokens: int
    wasted_reasoning_tokens: int
    optimal_cutoff_index: int | None
    reasoning_efficiency: float
    segments: list[ReasoningSegment]
    recommended_budget: int
    explanation: str


# ---------------------------------------------------------------------------
# EntropyTracker
# ---------------------------------------------------------------------------


class EntropyTracker:
    """Sliding-window entropy analysis over reasoning text.

    Uses character-trigram distributions to measure information novelty.
    When a new reasoning span introduces trigrams already well-represented
    in the window, its novelty score is low.
    """

    def __init__(self, window_size: int = ENTROPY_WINDOW) -> None:
        self.window_size = window_size
        self._window_trigrams: list[Counter[str]] = []
        self._cumulative: Counter[str] = Counter()

    def _extract_trigrams(self, text: str) -> Counter[str]:
        text_lower = text.lower()
        trigrams: Counter[str] = Counter()
        for i in range(len(text_lower) - 2):
            trigrams[text_lower[i : i + 3]] += 1
        return trigrams

    def _entropy(self, counter: Counter[str]) -> float:
        total = sum(counter.values())
        if total == 0:
            return 0.0
        ent = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    def add_span(self, text: str) -> float:
        """Add a reasoning span and return its novelty score [0, 1].

        Novelty is measured as the ratio of new trigrams introduced by
        this span vs. its total trigrams.  High novelty = genuinely new
        reasoning content.
        """
        trigrams = self._extract_trigrams(text)
        if not trigrams:
            return 0.0

        new_count = sum(1 for t in trigrams if t not in self._cumulative)
        total_unique = len(trigrams)
        novelty = new_count / total_unique if total_unique > 0 else 0.0

        self._window_trigrams.append(trigrams)
        self._cumulative.update(trigrams)

        if len(self._window_trigrams) > self.window_size:
            old = self._window_trigrams.pop(0)
            self._cumulative.subtract(old)
            self._cumulative = +self._cumulative

        return novelty

    @property
    def current_entropy(self) -> float:
        return self._entropy(self._cumulative)

    def reset(self) -> None:
        self._window_trigrams.clear()
        self._cumulative.clear()


# ---------------------------------------------------------------------------
# ReasoningPhaseClassifier
# ---------------------------------------------------------------------------


class ReasoningPhaseClassifier:
    """Classifies reasoning spans by their role in the reasoning chain.

    Uses lexical cues and structural patterns rather than ML, so it runs
    in <1ms per span.
    """

    _EXPLORING_CUES = (
        "let me", "could", "might", "perhaps", "what if",
        "one approach", "another way", "option", "possible",
    )
    _CONFIRMING_CUES = (
        "yes", "correct", "right", "confirmed", "verified",
        "this works", "that's right", "exactly",
    )
    _NEAR_ANSWER_CUES = (
        "therefore", "so the answer", "in conclusion", "final",
        "the solution is", "here's the plan", "to summarize",
        "putting it all together",
    )

    def classify(self, text: str) -> ReasoningPhase:
        lower = text.lower()

        filler_matches = sum(1 for p in FILLER_PATTERNS if p.search(lower))
        word_count = len(lower.split())
        if word_count > 0 and filler_matches / max(1, word_count / 20) > 0.5:
            return ReasoningPhase.FILLER

        near_answer = sum(1 for c in self._NEAR_ANSWER_CUES if c in lower)
        confirming = sum(1 for c in self._CONFIRMING_CUES if c in lower)
        exploring = sum(1 for c in self._EXPLORING_CUES if c in lower)

        scores = {
            ReasoningPhase.NEAR_ANSWER: near_answer * 2.0,
            ReasoningPhase.CONFIRMING: confirming * 1.5,
            ReasoningPhase.EXPLORING: exploring * 1.0,
        }

        best = max(scores, key=lambda k: scores[k])
        if scores[best] > 0:
            return best
        return ReasoningPhase.AMBIGUOUS


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _count_high_value_tokens(text: str) -> int:
    lower = text.lower()
    return sum(1 for token in HIGH_VALUE_TOKENS if token in lower)


def _filler_ratio(text: str) -> float:
    if not text:
        return 0.0
    matches = sum(1 for p in FILLER_PATTERNS if p.search(text))
    sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    return min(1.0, matches / sentences)


def find_optimal_cutoff(segments: list[ReasoningSegment]) -> int | None:
    """Find the span index after which reasoning value drops off.

    Uses the "elbow" method on cumulative novelty: the point where
    marginal novelty falls below NOVELTY_THRESHOLD for 2+ consecutive
    spans.
    """
    if len(segments) < MIN_REASONING_SPANS:
        return None

    consecutive_low = 0
    for seg in segments:
        if seg.novelty_score < NOVELTY_THRESHOLD:
            consecutive_low += 1
            if consecutive_low >= 2:
                return seg.index - 1
        else:
            consecutive_low = 0

    return None


def analyze_overthinking(
    reasoning_texts: Sequence[str],
    *,
    window_size: int = ENTROPY_WINDOW,
    novelty_threshold: float = NOVELTY_THRESHOLD,
) -> OverthinkingResult:
    """Analyze a sequence of reasoning spans for overthinking.

    Args:
        reasoning_texts: Ordered list of reasoning/thinking span texts
            from a session.
        window_size: Entropy tracker window size.
        novelty_threshold: Threshold below which novelty is considered
            stale.

    Returns:
        OverthinkingResult with analysis details and recommendations.
    """
    if not reasoning_texts:
        return OverthinkingResult(
            is_overthinking=False,
            total_reasoning_tokens=0,
            useful_reasoning_tokens=0,
            wasted_reasoning_tokens=0,
            optimal_cutoff_index=None,
            reasoning_efficiency=1.0,
            segments=[],
            recommended_budget=0,
            explanation="No reasoning spans to analyze.",
        )

    tracker = EntropyTracker(window_size=window_size)
    classifier = ReasoningPhaseClassifier()
    segments: list[ReasoningSegment] = []
    cumulative_novelty = 0.0
    total_tokens = 0

    for i, text in enumerate(reasoning_texts):
        tokens = _estimate_tokens(text)
        total_tokens += tokens
        novelty = tracker.add_span(text)
        cumulative_novelty += novelty
        phase = classifier.classify(text)
        hv_count = _count_high_value_tokens(text)
        filler = _filler_ratio(text)

        marginal = novelty * (1.0 + 0.2 * hv_count) * (1.0 - 0.5 * filler)

        segments.append(
            ReasoningSegment(
                index=i,
                text=text[:200],
                token_count=tokens,
                phase=phase,
                novelty_score=novelty,
                high_value_token_count=hv_count,
                filler_ratio=filler,
                cumulative_novelty=cumulative_novelty,
                marginal_value=marginal,
            )
        )

    cutoff = find_optimal_cutoff(segments)

    if cutoff is not None:
        useful_tokens = sum(s.token_count for s in segments[: cutoff + 1])
        wasted_tokens = total_tokens - useful_tokens
        is_overthinking = wasted_tokens > total_tokens * 0.2
    else:
        useful_tokens = total_tokens
        wasted_tokens = 0
        is_overthinking = False

    efficiency = useful_tokens / total_tokens if total_tokens > 0 else 1.0
    recommended = int(useful_tokens * 1.2) if cutoff is not None else total_tokens

    if is_overthinking:
        pct = round(wasted_tokens / total_tokens * 100)
        explanation = (
            f"Reasoning plateaued after span {cutoff}. "
            f"~{pct}% of reasoning tokens ({wasted_tokens:,}) added "
            f"diminishing value. Recommended budget: {recommended:,} tokens."
        )
    elif len(segments) >= MIN_REASONING_SPANS:
        explanation = (
            "Reasoning maintained novelty throughout. No overthinking detected."
        )
    else:
        explanation = "Too few reasoning spans for meaningful analysis."

    return OverthinkingResult(
        is_overthinking=is_overthinking,
        total_reasoning_tokens=total_tokens,
        useful_reasoning_tokens=useful_tokens,
        wasted_reasoning_tokens=wasted_tokens,
        optimal_cutoff_index=cutoff,
        reasoning_efficiency=round(efficiency, 4),
        segments=segments,
        recommended_budget=recommended,
        explanation=explanation,
    )
