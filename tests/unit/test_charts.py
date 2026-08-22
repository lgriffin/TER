"""Tests for SVG chart generation."""

import numpy as np

from ter_calculator.charts import (
    chart_composition,
    chart_economics,
    chart_key_metrics,
    chart_phase_scores,
    chart_positional_ter,
    chart_waste_breakdown,
    chart_waste_patterns,
    generate_all_charts,
)
from ter_calculator.models import (
    ClassificationExplanation,
    ClassifiedSpan,
    CostModel,
    InputGrowth,
    PositionalBreakdown,
    SessionEconomics,
    SpanLabel,
    SpanPhase,
    TERResult,
    TokenSpan,
    UncertaintyReport,
    WastePattern,
)


def _make_span(phase, label, tokens=100):
    return ClassifiedSpan(
        span=TokenSpan(
            text="x" * tokens,
            phase=phase,
            position=0,
            token_count=tokens,
            source_message_uuid="msg-1",
        ),
        label=label,
        confidence=0.9,
        cosine_similarity=0.8,
    )


def _make_economics():
    return SessionEconomics(
        total_input_tokens=5000,
        total_output_tokens=2000,
        total_cache_creation_tokens=500,
        total_cache_read_tokens=3000,
        input_output_ratio=2.5,
        cache_hit_rate=0.75,
        estimated_cost_usd=0.05,
        estimated_waste_cost_usd=0.01,
        cost_model=CostModel(),
        positional=PositionalBreakdown(
            early_ter=0.85,
            mid_ter=0.72,
            late_ter=0.78,
            early_span_count=10,
            mid_span_count=10,
            late_span_count=10,
        ),
        input_growth=InputGrowth(
            turn_input_tokens=[100, 200, 300],
            growth_rate=1.5,
            is_superlinear=False,
            context_bloat_detected=False,
        ),
    )


def _make_result(**overrides):
    defaults = dict(
        session_id="test-session",
        aggregate_ter=0.78,
        raw_ratio=0.75,
        phase_scores={"reasoning": 0.80, "tool_use": 0.72, "generation": 0.82},
        total_tokens=1000,
        aligned_tokens=780,
        waste_tokens=220,
        classified_spans=[
            _make_span(SpanPhase.REASONING, SpanLabel.ALIGNED_REASONING, 400),
            _make_span(SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL, 300),
            _make_span(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE, 80),
            _make_span(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, 120),
            _make_span(SpanPhase.TOOL_USE, SpanLabel.UNNECESSARY_TOOL_CALL, 60),
            _make_span(SpanPhase.GENERATION, SpanLabel.OVER_EXPLANATION, 40),
        ],
        waste_patterns=[
            WastePattern(
                pattern_type="reasoning_loop",
                description="Repeated analysis of auth module",
                start_position=3,
                end_position=5,
                spans_involved=3,
                tokens_wasted=120,
            ),
            WastePattern(
                pattern_type="duplicate_tool_call",
                description="Read same file twice",
                start_position=8,
                end_position=9,
                spans_involved=2,
                tokens_wasted=60,
            ),
        ],
        economics=_make_economics(),
    )
    defaults.update(overrides)
    return TERResult(**defaults)


class TestChartKeyMetrics:
    def test_produces_svg(self):
        svg = chart_key_metrics(_make_result())
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_contains_ter_score(self):
        svg = chart_key_metrics(_make_result(aggregate_ter=0.85))
        assert "0.85" in svg

    def test_contains_cost_when_economics_present(self):
        svg = chart_key_metrics(_make_result())
        assert "$" in svg

    def test_no_cost_without_economics(self):
        svg = chart_key_metrics(_make_result(economics=None))
        assert "$" not in svg

    def test_contains_reliability_when_uncertainty_present(self):
        result = _make_result()
        result.uncertainty = UncertaintyReport(
            mean_confidence=0.85,
            token_weighted_confidence=0.83,
            low_confidence_tokens=50,
            low_confidence_share=0.05,
            interval_lower=0.72,
            interval_upper=0.84,
            bootstrap_samples=1000,
            span_count=30,
            reliability="high",
        )
        svg = chart_key_metrics(result)
        assert "high" in svg


class TestChartComposition:
    def test_produces_svg(self):
        svg = chart_composition(_make_result())
        assert "<svg" in svg

    def test_empty_spans_returns_empty(self):
        svg = chart_composition(_make_result(classified_spans=[]))
        assert svg == ""


class TestChartPhaseScores:
    def test_produces_svg(self):
        svg = chart_phase_scores(_make_result())
        assert "<svg" in svg
        assert "Reasoning" in svg
        assert "Tool Use" in svg
        assert "Generation" in svg


class TestChartWastePatterns:
    def test_produces_svg(self):
        svg = chart_waste_patterns(_make_result())
        assert "<svg" in svg
        assert "Reasoning Loop" in svg

    def test_no_patterns_returns_empty(self):
        svg = chart_waste_patterns(_make_result(waste_patterns=[]))
        assert svg == ""


class TestChartPositionalTer:
    def test_produces_svg(self):
        svg = chart_positional_ter(_make_result())
        assert "<svg" in svg
        assert "Early" in svg
        assert "Late" in svg

    def test_no_economics_returns_empty(self):
        svg = chart_positional_ter(_make_result(economics=None))
        assert svg == ""


class TestChartEconomics:
    def test_produces_svg(self):
        svg = chart_economics(_make_result())
        assert "<svg" in svg
        assert "Output Tokens" in svg

    def test_no_economics_returns_empty(self):
        svg = chart_economics(_make_result(economics=None))
        assert svg == ""


class TestChartWasteBreakdown:
    def test_produces_svg(self):
        svg = chart_waste_breakdown(_make_result())
        assert "<svg" in svg
        assert "Aligned" in svg
        assert "Waste" in svg


class TestGenerateAllCharts:
    def test_returns_all_expected_charts(self):
        charts = generate_all_charts(_make_result())
        assert "key_metrics" in charts
        assert "waste_breakdown" in charts
        assert "composition" in charts
        assert "phase_scores" in charts
        assert "waste_patterns" in charts
        assert "positional_ter" in charts
        assert "economics" in charts

    def test_minimal_result_still_produces_charts(self):
        result = _make_result(
            classified_spans=[],
            waste_patterns=[],
            economics=None,
        )
        charts = generate_all_charts(result)
        assert "key_metrics" in charts
        assert "waste_breakdown" in charts
        assert "phase_scores" in charts
        assert "composition" not in charts
        assert "waste_patterns" not in charts

    def test_all_charts_are_valid_svg(self):
        charts = generate_all_charts(_make_result())
        for name, svg in charts.items():
            assert svg.startswith("<svg"), f"{name} doesn't start with <svg"
            assert "</svg>" in svg, f"{name} missing </svg>"
