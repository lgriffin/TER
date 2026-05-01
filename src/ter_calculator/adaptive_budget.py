"""Adaptive token budget recommender.

Analyzes historical TER data to recommend thinking token budgets and
model routing decisions based on task complexity.  Inspired by:

- SelfBudgeter: pre-estimate reasoning cost, then budget-guided RL.
- TALE: 81% accuracy at 32% of vanilla CoT token cost.
- Route-To-Reason: joint model + strategy routing — 60% token reduction.
- BACR: budget as a continuous conditioning signal.

Key components:

- ComplexityEstimator: classifies task complexity from intent text using
  lightweight features (length, vocabulary, domain signals).
- BudgetRecommendation: a concrete recommendation for thinking tokens,
  model tier, and confidence.
- HistoricalBudgetAnalyzer: learns budget-to-outcome mappings from prior
  TER results to refine future recommendations.
- ModelTier: Haiku/Sonnet/Opus routing based on complexity.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BudgetRecommendation",
    "ComplexityEstimator",
    "ComplexityTier",
    "HistoricalBudgetAnalyzer",
    "ModelTier",
    "estimate_complexity",
    "recommend_budget",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HISTORY_PATH = Path.home() / ".cache" / "ter" / "budget_history.json"

MAX_THINKING_TOKENS_SIMPLE = 2_048
MAX_THINKING_TOKENS_STANDARD = 8_192
MAX_THINKING_TOKENS_COMPLEX = 32_768

COST_PER_MTOK: dict[str, dict[str, float]] = {
    "haiku": {"input": 0.25, "output": 1.25},
    "sonnet": {"input": 3.00, "output": 15.00},
    "opus": {"input": 15.00, "output": 75.00},
}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ComplexityTier(Enum):
    """Task complexity classification."""

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


class ModelTier(Enum):
    """Anthropic model tier for routing."""

    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BudgetRecommendation:
    """Concrete recommendation for a session's token budget."""

    complexity: ComplexityTier
    model_tier: ModelTier
    max_thinking_tokens: int
    estimated_total_tokens: int
    estimated_cost_usd: float
    confidence: float
    reasoning: str
    features: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class HistoryEntry:
    """A single historical outcome for learning."""

    intent_text: str
    complexity: str
    actual_thinking_tokens: int
    actual_total_tokens: int
    actual_ter: float
    model_used: str
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# ComplexityEstimator
# ---------------------------------------------------------------------------


_MULTI_FILE_CUES = (
    "refactor", "rename across", "update all", "find and replace",
    "migration", "across the codebase", "every file",
)

_ARCHITECTURE_CUES = (
    "architect", "design", "system", "infrastructure",
    "microservice", "database schema", "api design",
    "scalab", "distributed",
)

_SIMPLE_CUES = (
    "fix typo", "rename", "add comment", "update readme",
    "change color", "update version", "bump", "lint",
    "format", "simple", "quick", "trivial",
)

_BUG_CUES = (
    "bug", "fix", "broken", "error", "crash", "fail",
    "doesn't work", "not working", "regression",
)

_FEATURE_CUES = (
    "implement", "add feature", "build", "create",
    "new endpoint", "new component", "integrate",
)


class ComplexityEstimator:
    """Estimates task complexity from intent text using lexical features."""

    def extract_features(self, intent_text: str) -> dict[str, float]:
        lower = intent_text.lower()
        words = lower.split()
        word_count = len(words)
        unique_words = len(set(words))

        multi_file = sum(1 for c in _MULTI_FILE_CUES if c in lower)
        architecture = sum(1 for c in _ARCHITECTURE_CUES if c in lower)
        simple = sum(1 for c in _SIMPLE_CUES if c in lower)
        bug = sum(1 for c in _BUG_CUES if c in lower)
        feature = sum(1 for c in _FEATURE_CUES if c in lower)

        question_marks = lower.count("?")
        sentence_count = max(1, lower.count(".") + lower.count("!") + question_marks)
        has_code = 1.0 if ("`" in lower or "```" in lower) else 0.0
        has_file_paths = 1.0 if re.search(r"[/\\]\w+\.\w+", lower) else 0.0

        return {
            "word_count": float(word_count),
            "unique_ratio": unique_words / max(1, word_count),
            "sentence_count": float(sentence_count),
            "multi_file_cues": float(multi_file),
            "architecture_cues": float(architecture),
            "simple_cues": float(simple),
            "bug_cues": float(bug),
            "feature_cues": float(feature),
            "has_code": has_code,
            "has_file_paths": has_file_paths,
            "question_count": float(question_marks),
        }

    def estimate(self, intent_text: str) -> tuple[ComplexityTier, float, dict[str, float]]:
        """Classify complexity and return (tier, confidence, features)."""
        features = self.extract_features(intent_text)

        simple_score = (
            features["simple_cues"] * 3.0
            + (1.0 if features["word_count"] < 15 else 0.0) * 2.0
            + (1.0 if features["sentence_count"] <= 2 else 0.0)
        )

        complex_score = (
            features["multi_file_cues"] * 3.0
            + features["architecture_cues"] * 3.0
            + (1.0 if features["word_count"] > 80 else 0.0) * 2.0
            + (1.0 if features["sentence_count"] > 5 else 0.0)
            + features["has_code"] * 1.5
        )

        standard_score = (
            features["bug_cues"] * 2.0
            + features["feature_cues"] * 2.0
            + (1.0 if 15 <= features["word_count"] <= 80 else 0.0) * 1.5
            + features["has_file_paths"] * 1.0
        )

        scores = {
            ComplexityTier.SIMPLE: simple_score,
            ComplexityTier.STANDARD: standard_score,
            ComplexityTier.COMPLEX: complex_score,
        }

        best = max(scores, key=lambda k: scores[k])
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.33

        if total == 0:
            best = ComplexityTier.STANDARD
            confidence = 0.33

        return best, min(1.0, confidence), features


# ---------------------------------------------------------------------------
# Budget recommendation
# ---------------------------------------------------------------------------

_TIER_TO_MODEL: dict[ComplexityTier, ModelTier] = {
    ComplexityTier.SIMPLE: ModelTier.HAIKU,
    ComplexityTier.STANDARD: ModelTier.SONNET,
    ComplexityTier.COMPLEX: ModelTier.OPUS,
}

_TIER_TO_THINKING: dict[ComplexityTier, int] = {
    ComplexityTier.SIMPLE: MAX_THINKING_TOKENS_SIMPLE,
    ComplexityTier.STANDARD: MAX_THINKING_TOKENS_STANDARD,
    ComplexityTier.COMPLEX: MAX_THINKING_TOKENS_COMPLEX,
}

_TIER_TO_TOTAL_ESTIMATE: dict[ComplexityTier, int] = {
    ComplexityTier.SIMPLE: 5_000,
    ComplexityTier.STANDARD: 25_000,
    ComplexityTier.COMPLEX: 100_000,
}


def estimate_complexity(intent_text: str) -> tuple[ComplexityTier, float, dict[str, float]]:
    """Convenience wrapper around ComplexityEstimator."""
    return ComplexityEstimator().estimate(intent_text)


def recommend_budget(
    intent_text: str,
    *,
    history: HistoricalBudgetAnalyzer | None = None,
) -> BudgetRecommendation:
    """Produce a budget recommendation for a given intent.

    If *history* is provided, recommendations are refined using historical
    TER outcomes for similar complexity tasks.
    """
    estimator = ComplexityEstimator()
    complexity, confidence, features = estimator.estimate(intent_text)

    model_tier = _TIER_TO_MODEL[complexity]
    thinking = _TIER_TO_THINKING[complexity]
    total_est = _TIER_TO_TOTAL_ESTIMATE[complexity]

    if history is not None:
        adj = history.get_adjustment(complexity)
        thinking = int(thinking * adj.thinking_multiplier)
        total_est = int(total_est * adj.total_multiplier)
        if adj.model_override is not None:
            model_tier = adj.model_override
        confidence = min(1.0, confidence * (0.5 + 0.5 * adj.data_confidence))

    model_name = model_tier.value
    cost_rates = COST_PER_MTOK.get(model_name, COST_PER_MTOK["sonnet"])
    input_tokens = total_est - thinking
    cost = (
        input_tokens / 1_000_000 * cost_rates["input"]
        + (thinking + total_est * 0.3) / 1_000_000 * cost_rates["output"]
    )

    reasons: list[str] = []
    reasons.append(f"Complexity: {complexity.value} (confidence {confidence:.0%})")
    reasons.append(f"Model: {model_tier.value}")
    if features.get("multi_file_cues", 0) > 0:
        reasons.append("Multi-file indicators detected — elevated budget")
    if features.get("simple_cues", 0) > 0:
        reasons.append("Simple task indicators — reduced budget")
    if history and history.entry_count > 10:
        reasons.append(f"Adjusted from {history.entry_count} historical outcomes")

    return BudgetRecommendation(
        complexity=complexity,
        model_tier=model_tier,
        max_thinking_tokens=thinking,
        estimated_total_tokens=total_est,
        estimated_cost_usd=round(cost, 4),
        confidence=round(confidence, 4),
        reasoning="; ".join(reasons),
        features=features,
    )


# ---------------------------------------------------------------------------
# HistoricalBudgetAnalyzer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BudgetAdjustment:
    """Adjustment factors derived from historical data."""

    thinking_multiplier: float
    total_multiplier: float
    model_override: ModelTier | None
    data_confidence: float


class HistoricalBudgetAnalyzer:
    """Learns from past TER outcomes to refine budget recommendations.

    Maintains a JSON file of {complexity, tokens_used, TER} tuples.
    When enough data accumulates, it adjusts default budgets:

    - If tasks at a complexity tier consistently use fewer thinking tokens
      than budgeted, it reduces the recommendation.
    - If TER is consistently low at a tier, it may recommend upgrading the
      model tier.
    """

    def __init__(self, history_path: Path | str | None = None) -> None:
        self.path = Path(history_path) if history_path else DEFAULT_HISTORY_PATH
        self._entries: list[HistoryEntry] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for item in data:
                self._entries.append(HistoryEntry(**item))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Could not load budget history: %s", exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for e in self._entries:
            data.append(
                {
                    "intent_text": e.intent_text[:200],
                    "complexity": e.complexity,
                    "actual_thinking_tokens": e.actual_thinking_tokens,
                    "actual_total_tokens": e.actual_total_tokens,
                    "actual_ter": e.actual_ter,
                    "model_used": e.model_used,
                    "timestamp": e.timestamp,
                }
            )
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def record(self, entry: HistoryEntry) -> None:
        self._entries.append(entry)
        self.save()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def get_adjustment(self, complexity: ComplexityTier) -> BudgetAdjustment:
        """Compute adjustment factors for a complexity tier."""
        tier_entries = [
            e for e in self._entries if e.complexity == complexity.value
        ]

        if len(tier_entries) < 5:
            return BudgetAdjustment(
                thinking_multiplier=1.0,
                total_multiplier=1.0,
                model_override=None,
                data_confidence=len(tier_entries) / 5.0,
            )

        default_thinking = _TIER_TO_THINKING[complexity]
        default_total = _TIER_TO_TOTAL_ESTIMATE[complexity]

        avg_thinking = sum(e.actual_thinking_tokens for e in tier_entries) / len(tier_entries)
        avg_total = sum(e.actual_total_tokens for e in tier_entries) / len(tier_entries)
        avg_ter = sum(e.actual_ter for e in tier_entries) / len(tier_entries)

        thinking_mult = max(0.3, min(2.0, avg_thinking / default_thinking * 1.2))
        total_mult = max(0.3, min(2.0, avg_total / default_total * 1.2))

        model_override: ModelTier | None = None
        if avg_ter < 0.45 and complexity != ComplexityTier.COMPLEX:
            model_override = ModelTier(
                {
                    ComplexityTier.SIMPLE: "sonnet",
                    ComplexityTier.STANDARD: "opus",
                }[complexity]
            )
            logger.info(
                "Historical TER for %s tasks is low (%.2f) — recommending model upgrade to %s",
                complexity.value,
                avg_ter,
                model_override.value,
            )

        if avg_ter > 0.85 and complexity != ComplexityTier.SIMPLE:
            model_override = ModelTier(
                {
                    ComplexityTier.STANDARD: "haiku",
                    ComplexityTier.COMPLEX: "sonnet",
                }[complexity]
            )
            logger.info(
                "Historical TER for %s tasks is high (%.2f) — recommending model downgrade to %s",
                complexity.value,
                avg_ter,
                model_override.value,
            )

        data_confidence = min(1.0, len(tier_entries) / 20.0)

        return BudgetAdjustment(
            thinking_multiplier=round(thinking_mult, 3),
            total_multiplier=round(total_mult, 3),
            model_override=model_override,
            data_confidence=round(data_confidence, 3),
        )

    def get_summary(self) -> dict[str, Any]:
        """Return a human-readable summary of historical budget data."""
        by_tier: dict[str, list[HistoryEntry]] = defaultdict(list)
        for e in self._entries:
            by_tier[e.complexity].append(e)

        summary: dict[str, Any] = {"total_entries": len(self._entries), "tiers": {}}
        for tier_name, entries in by_tier.items():
            avg_ter = sum(e.actual_ter for e in entries) / len(entries)
            avg_thinking = sum(e.actual_thinking_tokens for e in entries) / len(entries)
            avg_total = sum(e.actual_total_tokens for e in entries) / len(entries)
            summary["tiers"][tier_name] = {
                "count": len(entries),
                "avg_ter": round(avg_ter, 4),
                "avg_thinking_tokens": int(avg_thinking),
                "avg_total_tokens": int(avg_total),
            }
        return summary
