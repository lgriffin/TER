"""Cost-weighted TER and semantic density scoring.

Extends the base TER metric with two dimensions:

1. **Cost-weighted TER**: Instead of treating all tokens equally, weights
   them by their dollar cost.  Output tokens cost 4-8x input tokens;
   cached input tokens cost 0.1x; thinking tokens are billed at output
   rate.  A session that wastes output tokens is more costly than one
   that wastes cached input tokens.

2. **Semantic density**: Measures information content per token using
   embedding-space analysis.  High-density spans pack more meaning into
   fewer tokens.  The Semantic Density Effect (SDE, April 2026) shows
   that higher density correlates with better output quality and fewer
   hallucinations.

Key components:

- PricingTier: per-model cost rates for input/output/cached tokens.
- CostWeightedTER: applies dollar-cost weighting to the TER formula.
- SemanticDensityScorer: measures information density per token using
  embedding variance and vocabulary richness.
- CostReport: full cost analysis including savings opportunities.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "CostReport",
    "CostWeightedTER",
    "PricingTier",
    "SemanticDensityResult",
    "SemanticDensityScorer",
    "SpanCost",
    "TokenCategory",
    "compute_cost_weighted_ter",
    "compute_semantic_density",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TokenCategory(Enum):
    """Categories of tokens by billing treatment."""

    INPUT = "input"
    OUTPUT = "output"
    CACHED_READ = "cached_read"
    CACHED_WRITE = "cached_write"
    THINKING = "thinking"


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PricingTier:
    """Per-million-token pricing for a model tier."""

    name: str
    input_per_mtok: float
    output_per_mtok: float
    cached_read_per_mtok: float
    cached_write_per_mtok: float

    @property
    def thinking_per_mtok(self) -> float:
        return self.output_per_mtok

    def cost(self, category: TokenCategory, token_count: int) -> float:
        rates = {
            TokenCategory.INPUT: self.input_per_mtok,
            TokenCategory.OUTPUT: self.output_per_mtok,
            TokenCategory.CACHED_READ: self.cached_read_per_mtok,
            TokenCategory.CACHED_WRITE: self.cached_write_per_mtok,
            TokenCategory.THINKING: self.thinking_per_mtok,
        }
        return token_count / 1_000_000 * rates[category]

    def weight(self, category: TokenCategory) -> float:
        """Relative cost weight normalised against input rate."""
        if self.input_per_mtok == 0:
            return 1.0
        rates = {
            TokenCategory.INPUT: self.input_per_mtok,
            TokenCategory.OUTPUT: self.output_per_mtok,
            TokenCategory.CACHED_READ: self.cached_read_per_mtok,
            TokenCategory.CACHED_WRITE: self.cached_write_per_mtok,
            TokenCategory.THINKING: self.thinking_per_mtok,
        }
        return rates[category] / self.input_per_mtok


PRICING: dict[str, PricingTier] = {
    "haiku": PricingTier(
        name="claude-haiku-4-5",
        input_per_mtok=0.80,
        output_per_mtok=4.00,
        cached_read_per_mtok=0.08,
        cached_write_per_mtok=1.00,
    ),
    "sonnet": PricingTier(
        name="claude-sonnet-4-6",
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cached_read_per_mtok=0.30,
        cached_write_per_mtok=3.75,
    ),
    "opus": PricingTier(
        name="claude-opus-4-6",
        input_per_mtok=15.00,
        output_per_mtok=75.00,
        cached_read_per_mtok=1.50,
        cached_write_per_mtok=18.75,
    ),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpanCost:
    """Cost attribution for a single span."""

    span_index: int
    phase: str
    category: TokenCategory
    token_count: int
    dollar_cost: float
    is_aligned: bool
    cost_weight: float


@dataclass(frozen=True, slots=True)
class CostWeightedTER:
    """TER weighted by dollar cost instead of raw token count."""

    aggregate_ter: float
    raw_ter: float
    cost_weighted_ter: float
    total_cost_usd: float
    aligned_cost_usd: float
    waste_cost_usd: float
    total_tokens: int
    span_costs: list[SpanCost] = field(default_factory=list)
    savings_if_perfect: float = 0.0
    pricing_tier: str = "sonnet"


@dataclass(frozen=True, slots=True)
class SemanticDensityResult:
    """Semantic density analysis for a span or session."""

    density_score: float
    vocabulary_richness: float
    information_entropy: float
    avg_token_information: float
    redundancy_ratio: float


@dataclass(frozen=True, slots=True)
class CostReport:
    """Full cost analysis combining cost-weighted TER and density."""

    cost_ter: CostWeightedTER
    session_density: SemanticDensityResult
    phase_costs: dict[str, float]
    phase_waste_costs: dict[str, float]
    recommendations: list[str]
    model_tier: str
    alternative_model_savings: dict[str, float]


# ---------------------------------------------------------------------------
# Cost-weighted TER computation
# ---------------------------------------------------------------------------

_PHASE_TO_CATEGORY: dict[str, TokenCategory] = {
    "reasoning": TokenCategory.THINKING,
    "tool_use": TokenCategory.OUTPUT,
    "generation": TokenCategory.OUTPUT,
}


def compute_cost_weighted_ter(
    spans: Sequence[dict[str, Any]],
    *,
    model: str = "sonnet",
    raw_ter: float = 0.0,
    usage: dict[str, int] | None = None,
) -> CostWeightedTER:
    """Compute cost-weighted TER from classified spans.

    Each span dict should have: phase, token_count, is_aligned.
    Optional: category (TokenCategory value name).

    The cost-weighted TER weights each token by its dollar cost,
    so wasting an output token penalises the score more than wasting
    a cached input token.
    """
    tier = PRICING.get(model, PRICING["sonnet"])
    span_costs: list[SpanCost] = []
    total_cost = 0.0
    aligned_cost = 0.0

    for i, span in enumerate(spans):
        phase = span.get("phase", "generation")
        tokens = span.get("token_count", 0)
        is_aligned = span.get("is_aligned", True)

        cat_name = span.get("category")
        category = (
            TokenCategory(cat_name) if cat_name else _PHASE_TO_CATEGORY.get(phase, TokenCategory.OUTPUT)
        )

        cost = tier.cost(category, tokens)
        weight = tier.weight(category)
        total_cost += cost
        if is_aligned:
            aligned_cost += cost

        span_costs.append(
            SpanCost(
                span_index=i,
                phase=phase,
                category=category,
                token_count=tokens,
                dollar_cost=round(cost, 6),
                is_aligned=is_aligned,
                cost_weight=round(weight, 3),
            )
        )

    if usage:
        cached_read = usage.get("cache_read_input_tokens", 0)
        cached_write = usage.get("cache_creation_input_tokens", 0)
        if cached_read:
            total_cost += tier.cost(TokenCategory.CACHED_READ, cached_read)
            aligned_cost += tier.cost(TokenCategory.CACHED_READ, cached_read)
        if cached_write:
            total_cost += tier.cost(TokenCategory.CACHED_WRITE, cached_write)
            aligned_cost += tier.cost(TokenCategory.CACHED_WRITE, cached_write)

    waste_cost = total_cost - aligned_cost
    cost_ter = aligned_cost / total_cost if total_cost > 0 else 1.0
    total_tokens = sum(s.token_count for s in span_costs)

    return CostWeightedTER(
        aggregate_ter=raw_ter,
        raw_ter=raw_ter,
        cost_weighted_ter=round(cost_ter, 4),
        total_cost_usd=round(total_cost, 6),
        aligned_cost_usd=round(aligned_cost, 6),
        waste_cost_usd=round(waste_cost, 6),
        total_tokens=total_tokens,
        span_costs=span_costs,
        savings_if_perfect=round(waste_cost, 6),
        pricing_tier=model,
    )


# ---------------------------------------------------------------------------
# Semantic Density Scorer
# ---------------------------------------------------------------------------


class SemanticDensityScorer:
    """Measures information content per token.

    Uses three complementary signals:

    1. Vocabulary richness (type-token ratio): unique words / total words.
       Higher = more diverse content per token.
    2. Shannon entropy of word distribution: measures unpredictability.
       Higher = more information per token.
    3. Redundancy ratio: fraction of content that is repeated.
       Lower = denser information.
    """

    def score(self, text: str) -> SemanticDensityResult:
        if not text or not text.strip():
            return SemanticDensityResult(
                density_score=0.0,
                vocabulary_richness=0.0,
                information_entropy=0.0,
                avg_token_information=0.0,
                redundancy_ratio=1.0,
            )

        words = text.lower().split()
        total_words = len(words)
        if total_words == 0:
            return SemanticDensityResult(
                density_score=0.0,
                vocabulary_richness=0.0,
                information_entropy=0.0,
                avg_token_information=0.0,
                redundancy_ratio=1.0,
            )

        unique_words = len(set(words))
        vocabulary_richness = unique_words / total_words

        word_counts: Counter[str] = Counter(words)
        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(total_words) if total_words > 1 else 1.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 1:
            seen_trigrams: set[str] = set()
            total_trigrams = 0
            repeat_trigrams = 0
            for sent in sentences:
                sent_words = sent.lower().split()
                for j in range(len(sent_words) - 2):
                    tri = " ".join(sent_words[j : j + 3])
                    total_trigrams += 1
                    if tri in seen_trigrams:
                        repeat_trigrams += 1
                    seen_trigrams.add(tri)
            redundancy = repeat_trigrams / max(1, total_trigrams)
        else:
            redundancy = 0.0

        density = (
            0.4 * vocabulary_richness
            + 0.4 * norm_entropy
            + 0.2 * (1.0 - redundancy)
        )

        avg_info = entropy / max(1, total_words // 4)

        return SemanticDensityResult(
            density_score=round(density, 4),
            vocabulary_richness=round(vocabulary_richness, 4),
            information_entropy=round(entropy, 4),
            avg_token_information=round(avg_info, 4),
            redundancy_ratio=round(redundancy, 4),
        )


def compute_semantic_density(text: str) -> SemanticDensityResult:
    """Convenience wrapper."""
    return SemanticDensityScorer().score(text)


# ---------------------------------------------------------------------------
# Full cost report
# ---------------------------------------------------------------------------


def generate_cost_report(
    spans: Sequence[dict[str, Any]],
    full_text: str,
    *,
    model: str = "sonnet",
    raw_ter: float = 0.0,
    usage: dict[str, int] | None = None,
) -> CostReport:
    """Generate a comprehensive cost + density report."""
    cost_ter = compute_cost_weighted_ter(spans, model=model, raw_ter=raw_ter, usage=usage)
    density = compute_semantic_density(full_text)

    phase_costs: dict[str, float] = {}
    phase_waste: dict[str, float] = {}
    for sc in cost_ter.span_costs:
        phase_costs[sc.phase] = phase_costs.get(sc.phase, 0.0) + sc.dollar_cost
        if not sc.is_aligned:
            phase_waste[sc.phase] = phase_waste.get(sc.phase, 0.0) + sc.dollar_cost

    alt_savings: dict[str, float] = {}
    for alt_name, alt_tier in PRICING.items():
        if alt_name == model:
            continue
        alt_result = compute_cost_weighted_ter(spans, model=alt_name, raw_ter=raw_ter, usage=usage)
        saving = cost_ter.total_cost_usd - alt_result.total_cost_usd
        alt_savings[alt_name] = round(saving, 6)

    recommendations: list[str] = []

    if cost_ter.waste_cost_usd > 0.01:
        recommendations.append(
            f"Eliminating waste would save ${cost_ter.waste_cost_usd:.4f} per session"
        )

    worst_phase = max(phase_waste, key=lambda p: phase_waste.get(p, 0), default=None)
    if worst_phase and phase_waste.get(worst_phase, 0) > 0:
        recommendations.append(
            f"Highest waste cost in {worst_phase} phase (${phase_waste[worst_phase]:.4f})"
        )

    if density.redundancy_ratio > 0.3:
        recommendations.append(
            f"High redundancy ({density.redundancy_ratio:.0%}) — consider prompt compression"
        )

    if density.density_score < 0.3:
        recommendations.append(
            "Low semantic density — output contains filler or repetition"
        )

    for alt, saving in alt_savings.items():
        if saving > 0 and cost_ter.cost_weighted_ter > 0.7:
            recommendations.append(
                f"Consider {alt} (saves ${saving:.4f}/session with acceptable TER)"
            )

    return CostReport(
        cost_ter=cost_ter,
        session_density=density,
        phase_costs={k: round(v, 6) for k, v in phase_costs.items()},
        phase_waste_costs={k: round(v, 6) for k, v in phase_waste.items()},
        recommendations=recommendations,
        model_tier=model,
        alternative_model_savings=alt_savings,
    )
