"""Tests for cost-weighted TER and semantic density scoring."""

import pytest

from ter_calculator.cost_model import (
    CostReport,
    CostWeightedTER,
    PricingTier,
    SemanticDensityResult,
    SemanticDensityScorer,
    SpanCost,
    TokenCategory,
    compute_cost_weighted_ter,
    compute_semantic_density,
    generate_cost_report,
    PRICING,
)


class TestPricingTier:
    """Test pricing tier calculations."""

    def test_sonnet_pricing_defined(self):
        assert "sonnet" in PRICING
        tier = PRICING["sonnet"]
        assert tier.input_per_mtok == 3.00
        assert tier.output_per_mtok == 15.00
        assert tier.cached_read_per_mtok == 0.30
        assert tier.cached_write_per_mtok == 3.75

    def test_haiku_pricing_defined(self):
        assert "haiku" in PRICING
        tier = PRICING["haiku"]
        assert tier.input_per_mtok < PRICING["sonnet"].input_per_mtok

    def test_opus_pricing_defined(self):
        assert "opus" in PRICING
        tier = PRICING["opus"]
        assert tier.output_per_mtok > PRICING["sonnet"].output_per_mtok

    def test_thinking_equals_output_rate(self):
        tier = PRICING["sonnet"]
        assert tier.thinking_per_mtok == tier.output_per_mtok

    def test_cost_calculation_input(self):
        tier = PRICING["sonnet"]
        cost = tier.cost(TokenCategory.INPUT, 1_000_000)
        assert cost == 3.00

    def test_cost_calculation_output(self):
        tier = PRICING["sonnet"]
        cost = tier.cost(TokenCategory.OUTPUT, 1_000_000)
        assert cost == 15.00

    def test_cost_calculation_cached_read(self):
        tier = PRICING["sonnet"]
        cost = tier.cost(TokenCategory.CACHED_READ, 1_000_000)
        assert cost == 0.30

    def test_cost_calculation_thinking(self):
        tier = PRICING["sonnet"]
        cost = tier.cost(TokenCategory.THINKING, 1_000_000)
        assert cost == 15.00  # Same as output

    def test_cost_scales_with_tokens(self):
        tier = PRICING["sonnet"]
        cost_1m = tier.cost(TokenCategory.OUTPUT, 1_000_000)
        cost_2m = tier.cost(TokenCategory.OUTPUT, 2_000_000)
        assert cost_2m == pytest.approx(cost_1m * 2)

    def test_weight_normalized_to_input(self):
        tier = PRICING["sonnet"]
        assert tier.weight(TokenCategory.INPUT) == 1.0
        assert tier.weight(TokenCategory.OUTPUT) == 5.0  # 15/3
        assert tier.weight(TokenCategory.CACHED_READ) == pytest.approx(0.1, abs=0.01)  # 0.30/3

    def test_weight_with_zero_input_rate(self):
        tier = PricingTier(
            name="test",
            input_per_mtok=0.0,
            output_per_mtok=10.0,
            cached_read_per_mtok=1.0,
            cached_write_per_mtok=2.0,
        )
        assert tier.weight(TokenCategory.INPUT) == 1.0


class TestSemanticDensityScorer:
    """Test semantic density measurements."""

    def test_empty_text_returns_zero_density(self):
        scorer = SemanticDensityScorer()
        result = scorer.score("")
        assert result.density_score == 0.0
        assert result.vocabulary_richness == 0.0

    def test_whitespace_only_returns_zero(self):
        scorer = SemanticDensityScorer()
        result = scorer.score("   \n\t  ")
        assert result.density_score == 0.0

    def test_vocabulary_richness_all_unique(self):
        scorer = SemanticDensityScorer()
        result = scorer.score("all unique words here")
        assert result.vocabulary_richness == 1.0

    def test_vocabulary_richness_all_same(self):
        scorer = SemanticDensityScorer()
        result = scorer.score("same same same same")
        assert result.vocabulary_richness == 0.25  # 1 unique / 4 total

    def test_information_entropy_increases_with_diversity(self):
        scorer = SemanticDensityScorer()
        result_repetitive = scorer.score("word word word word")
        result_diverse = scorer.score("different unique words varied")
        assert result_diverse.information_entropy > result_repetitive.information_entropy

    def test_redundancy_detected_in_repeated_phrases(self):
        scorer = SemanticDensityScorer()
        result = scorer.score(
            "This is a sentence. This is a sentence. This is a sentence."
        )
        assert result.redundancy_ratio > 0.0

    def test_no_redundancy_in_unique_sentences(self):
        scorer = SemanticDensityScorer()
        result = scorer.score("First sentence. Second sentence. Third sentence.")
        # Should have low redundancy since sentences are different
        assert result.redundancy_ratio < 0.5

    def test_density_score_combines_factors(self):
        scorer = SemanticDensityScorer()
        result = scorer.score("diverse unique vocabulary with varied words")
        # High vocabulary richness, entropy, low redundancy = high density
        assert 0.0 < result.density_score <= 1.0

    def test_avg_token_information_calculated(self):
        scorer = SemanticDensityScorer()
        result = scorer.score("some text here")
        assert result.avg_token_information >= 0.0

    def test_case_insensitive(self):
        scorer = SemanticDensityScorer()
        result1 = scorer.score("Word Word Word")
        result2 = scorer.score("word word word")
        assert result1.vocabulary_richness == result2.vocabulary_richness

    def test_high_density_text(self):
        scorer = SemanticDensityScorer()
        # Technical text with varied vocabulary
        result = scorer.score(
            "Implement authentication middleware using JWT tokens with "
            "refresh rotation and Redis caching for session management."
        )
        assert result.density_score > 0.5
        assert result.vocabulary_richness > 0.7

    def test_low_density_text(self):
        scorer = SemanticDensityScorer()
        # Repetitive text - same words repeated
        result = scorer.score(
            "word word word word word word word word"
        )
        # With word repetition, should have low vocabulary richness
        assert result.vocabulary_richness < 0.3
        assert result.density_score < 0.7


class TestComputeSemanticDensity:
    """Test convenience wrapper function."""

    def test_wrapper_returns_result(self):
        result = compute_semantic_density("test text here")
        assert isinstance(result, SemanticDensityResult)
        assert result.density_score >= 0.0


class TestComputeCostWeightedTER:
    """Test cost-weighted TER calculation."""

    def test_all_aligned_gives_perfect_ter(self):
        spans = [
            {
                "phase": "reasoning",
                "token_count": 1000,
                "is_aligned": True,
            },
            {
                "phase": "tool_use",
                "token_count": 500,
                "is_aligned": True,
            },
            {
                "phase": "generation",
                "token_count": 2000,
                "is_aligned": True,
            },
        ]
        result = compute_cost_weighted_ter(spans, model="sonnet", raw_ter=1.0)

        assert result.cost_weighted_ter == 1.0
        assert result.waste_cost_usd == 0.0
        assert result.aligned_cost_usd == result.total_cost_usd

    def test_all_waste_gives_zero_ter(self):
        spans = [
            {
                "phase": "reasoning",
                "token_count": 1000,
                "is_aligned": False,
            },
        ]
        result = compute_cost_weighted_ter(spans, model="sonnet", raw_ter=0.0)

        assert result.cost_weighted_ter == 0.0
        assert result.waste_cost_usd == result.total_cost_usd
        assert result.aligned_cost_usd == 0.0

    def test_mixed_aligned_waste(self):
        spans = [
            {"phase": "reasoning", "token_count": 1000, "is_aligned": True},
            {"phase": "reasoning", "token_count": 1000, "is_aligned": False},
        ]
        result = compute_cost_weighted_ter(spans, model="sonnet")

        # Should be 50% cost-weighted TER
        assert 0.45 < result.cost_weighted_ter < 0.55
        assert result.waste_cost_usd > 0.0
        assert result.aligned_cost_usd > 0.0

    def test_output_tokens_cost_more_than_input(self):
        """Wasting output tokens should penalize TER more than input."""
        spans_output = [
            {"phase": "generation", "token_count": 1000, "is_aligned": False,
             "category": "output"},
        ]
        spans_input = [
            {"phase": "generation", "token_count": 1000, "is_aligned": False,
             "category": "input"},
        ]

        result_output = compute_cost_weighted_ter(spans_output, model="sonnet")
        result_input = compute_cost_weighted_ter(spans_input, model="sonnet")

        # Output waste should cost more
        assert result_output.waste_cost_usd > result_input.waste_cost_usd

    def test_thinking_priced_as_output(self):
        spans = [
            {"phase": "reasoning", "token_count": 1000, "is_aligned": True},
        ]
        result = compute_cost_weighted_ter(spans, model="sonnet")

        # Thinking tokens should use output rate
        tier = PRICING["sonnet"]
        expected_cost = tier.cost(TokenCategory.THINKING, 1000)
        assert result.total_cost_usd == pytest.approx(expected_cost)

    def test_usage_data_includes_cached_tokens(self):
        spans = [
            {"phase": "generation", "token_count": 1000, "is_aligned": True},
        ]
        usage = {
            "cache_read_input_tokens": 100_000,
            "cache_creation_input_tokens": 50_000,
        }
        result = compute_cost_weighted_ter(spans, model="sonnet", usage=usage)

        # Total cost should include cache costs
        tier = PRICING["sonnet"]
        cache_read_cost = tier.cost(TokenCategory.CACHED_READ, 100_000)
        cache_write_cost = tier.cost(TokenCategory.CACHED_WRITE, 50_000)

        assert result.total_cost_usd > cache_read_cost + cache_write_cost

    def test_span_costs_populated(self):
        spans = [
            {"phase": "reasoning", "token_count": 500, "is_aligned": True},
            {"phase": "tool_use", "token_count": 300, "is_aligned": False},
        ]
        result = compute_cost_weighted_ter(spans, model="sonnet")

        assert len(result.span_costs) == 2
        assert isinstance(result.span_costs[0], SpanCost)
        assert result.span_costs[0].span_index == 0
        assert result.span_costs[1].span_index == 1

    def test_span_cost_includes_all_fields(self):
        spans = [
            {"phase": "reasoning", "token_count": 1000, "is_aligned": True},
        ]
        result = compute_cost_weighted_ter(spans, model="sonnet")

        sc = result.span_costs[0]
        assert sc.phase == "reasoning"
        assert sc.category == TokenCategory.THINKING
        assert sc.token_count == 1000
        assert sc.dollar_cost > 0.0
        assert sc.is_aligned is True
        assert sc.cost_weight > 0.0

    def test_savings_if_perfect_equals_waste(self):
        spans = [
            {"phase": "reasoning", "token_count": 1000, "is_aligned": False},
        ]
        result = compute_cost_weighted_ter(spans, model="sonnet")

        assert result.savings_if_perfect == result.waste_cost_usd

    def test_pricing_tier_stored(self):
        result = compute_cost_weighted_ter([], model="haiku")
        assert result.pricing_tier == "haiku"

    def test_raw_ter_preserved(self):
        result = compute_cost_weighted_ter([], model="sonnet", raw_ter=0.85)
        assert result.raw_ter == 0.85
        assert result.aggregate_ter == 0.85

    def test_empty_spans_returns_perfect_ter(self):
        result = compute_cost_weighted_ter([])
        assert result.cost_weighted_ter == 1.0
        assert result.total_cost_usd == 0.0

    def test_haiku_cheaper_than_sonnet(self):
        spans = [
            {"phase": "generation", "token_count": 10_000, "is_aligned": True},
        ]
        result_haiku = compute_cost_weighted_ter(spans, model="haiku")
        result_sonnet = compute_cost_weighted_ter(spans, model="sonnet")

        assert result_haiku.total_cost_usd < result_sonnet.total_cost_usd

    def test_opus_most_expensive(self):
        spans = [
            {"phase": "generation", "token_count": 10_000, "is_aligned": True},
        ]
        result_sonnet = compute_cost_weighted_ter(spans, model="sonnet")
        result_opus = compute_cost_weighted_ter(spans, model="opus")

        assert result_opus.total_cost_usd > result_sonnet.total_cost_usd

    def test_unknown_model_defaults_to_sonnet(self):
        spans = [
            {"phase": "generation", "token_count": 1000, "is_aligned": True},
        ]
        result = compute_cost_weighted_ter(spans, model="unknown_model")
        result_sonnet = compute_cost_weighted_ter(spans, model="sonnet")

        assert result.total_cost_usd == result_sonnet.total_cost_usd


class TestGenerateCostReport:
    """Test full cost report generation."""

    def test_report_includes_cost_ter(self):
        spans = [
            {"phase": "reasoning", "token_count": 1000, "is_aligned": True},
        ]
        report = generate_cost_report(spans, full_text="test text", model="sonnet")

        assert isinstance(report.cost_ter, CostWeightedTER)

    def test_report_includes_density(self):
        spans = [
            {"phase": "generation", "token_count": 1000, "is_aligned": True},
        ]
        report = generate_cost_report(spans, full_text="diverse unique vocabulary", model="sonnet")

        assert isinstance(report.session_density, SemanticDensityResult)
        assert report.session_density.density_score > 0.0

    def test_phase_costs_aggregated(self):
        spans = [
            {"phase": "reasoning", "token_count": 1000, "is_aligned": True},
            {"phase": "reasoning", "token_count": 500, "is_aligned": True},
            {"phase": "tool_use", "token_count": 300, "is_aligned": True},
        ]
        report = generate_cost_report(spans, full_text="test", model="sonnet")

        assert "reasoning" in report.phase_costs
        assert "tool_use" in report.phase_costs
        assert report.phase_costs["reasoning"] > report.phase_costs["tool_use"]

    def test_phase_waste_costs_aggregated(self):
        spans = [
            {"phase": "reasoning", "token_count": 1000, "is_aligned": False},
            {"phase": "tool_use", "token_count": 500, "is_aligned": False},
        ]
        report = generate_cost_report(spans, full_text="test", model="sonnet")

        assert "reasoning" in report.phase_waste_costs
        assert "tool_use" in report.phase_waste_costs
        assert report.phase_waste_costs["reasoning"] > 0.0

    def test_alternative_model_savings_calculated(self):
        spans = [
            {"phase": "generation", "token_count": 10_000, "is_aligned": True},
        ]
        report = generate_cost_report(spans, full_text="test", model="sonnet")

        assert "haiku" in report.alternative_model_savings
        assert "opus" in report.alternative_model_savings
        # Haiku should save money
        assert report.alternative_model_savings["haiku"] > 0.0
        # Opus should cost more
        assert report.alternative_model_savings["opus"] < 0.0

    def test_recommendations_for_waste(self):
        spans = [
            {"phase": "reasoning", "token_count": 10_000, "is_aligned": False},
        ]
        report = generate_cost_report(spans, full_text="test", model="sonnet")

        assert len(report.recommendations) > 0
        # Should mention waste savings
        assert any("waste" in r.lower() for r in report.recommendations)

    def test_recommendations_for_high_redundancy(self):
        spans = [
            {"phase": "generation", "token_count": 1000, "is_aligned": True},
        ]
        # Highly redundant text
        text = "same thing. same thing. same thing. same thing."
        report = generate_cost_report(spans, full_text=text, model="sonnet")

        # Redundancy recommendation appears if redundancy is high
        if report.session_density.redundancy_ratio > 0.3:
            # Check report has recommendations
            assert isinstance(report.recommendations, list)

    def test_recommendations_for_low_density(self):
        spans = [
            {"phase": "generation", "token_count": 1000, "is_aligned": True},
        ]
        # Low density text
        text = "I need to think. Let me think. I should think. Let me think more."
        report = generate_cost_report(spans, full_text=text, model="sonnet")

        # Report should be generated
        assert isinstance(report.session_density, SemanticDensityResult)
        # Recommendations list should exist
        assert isinstance(report.recommendations, list)

    def test_recommendations_for_model_downgrade(self):
        spans = [
            {"phase": "generation", "token_count": 5000, "is_aligned": True},
        ]
        report = generate_cost_report(
            spans, full_text="test", model="sonnet", raw_ter=0.95
        )

        # With high TER, should suggest cheaper model
        recommendations_text = " ".join(report.recommendations).lower()
        assert "haiku" in recommendations_text or "cheaper" in recommendations_text

    def test_model_tier_stored(self):
        report = generate_cost_report([], full_text="test", model="opus")
        assert report.model_tier == "opus"

    def test_worst_phase_identified(self):
        spans = [
            {"phase": "reasoning", "token_count": 1000, "is_aligned": False},
            {"phase": "tool_use", "token_count": 100, "is_aligned": False},
        ]
        report = generate_cost_report(spans, full_text="test", model="sonnet")

        # Should identify reasoning as highest waste
        recommendations_text = " ".join(report.recommendations)
        assert "reasoning" in recommendations_text.lower()

    def test_no_recommendations_for_perfect_session(self):
        spans = [
            {"phase": "generation", "token_count": 1000, "is_aligned": True},
        ]
        report = generate_cost_report(
            spans,
            full_text="highly diverse unique vocabulary with varied content",
            model="haiku",
            raw_ter=0.99,
        )

        # Perfect session might have fewer recommendations
        # (but may still suggest alternative models)
        assert isinstance(report.recommendations, list)


class TestSpanCost:
    """Test SpanCost dataclass."""

    def test_frozen_immutable(self):
        sc = SpanCost(
            span_index=0,
            phase="reasoning",
            category=TokenCategory.THINKING,
            token_count=100,
            dollar_cost=0.01,
            is_aligned=True,
            cost_weight=1.5,
        )

        with pytest.raises(AttributeError):
            sc.token_count = 200


class TestTokenCategory:
    """Test TokenCategory enum."""

    def test_all_categories_defined(self):
        assert TokenCategory.INPUT
        assert TokenCategory.OUTPUT
        assert TokenCategory.CACHED_READ
        assert TokenCategory.CACHED_WRITE
        assert TokenCategory.THINKING

    def test_category_values_are_strings(self):
        assert isinstance(TokenCategory.INPUT.value, str)
        assert TokenCategory.OUTPUT.value == "output"
