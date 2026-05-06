"""Tests for session economics computation."""

import numpy as np
import pytest

from ter_calculator.economics import (
    _aggregate_usage,
    _compute_cache_hit_rate,
    _compute_input_growth,
    _compute_positional_breakdown,
    _estimate_cost,
    _estimate_waste_cost,
    compute_economics,
)
from ter_calculator.models import (
    ClassifiedSpan,
    CostModel,
    Message,
    Session,
    SpanLabel,
    SpanPhase,
    TokenSpan,
    TokenUsage,
)


def _make_session(messages: list[Message] | None = None) -> Session:
    return Session(
        session_id="test",
        file_path="test.jsonl",
        messages=messages or [],
    )


def _make_message(
    role: str = "assistant",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> Message:
    return Message(
        uuid="msg-1",
        role=role,
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )


def _make_classified(
    label: SpanLabel = SpanLabel.ALIGNED_RESPONSE,
    token_count: int = 10,
    position: int = 0,
) -> ClassifiedSpan:
    return ClassifiedSpan(
        span=TokenSpan(
            text="test",
            phase=SpanPhase.GENERATION,
            position=position,
            token_count=token_count,
            source_message_uuid="msg-1",
        ),
        label=label,
        confidence=0.9,
        cosine_similarity=0.8,
    )


class TestAggregateUsage:
    def test_sums_assistant_usage(self):
        session = _make_session([
            _make_message("assistant", input_tokens=100, output_tokens=50, cache_creation=10, cache_read=200),
            _make_message("assistant", input_tokens=150, output_tokens=80, cache_creation=20, cache_read=300),
        ])
        inp, out, create, read = _aggregate_usage(session)
        assert inp == 250
        assert out == 130
        assert create == 30
        assert read == 500

    def test_ignores_user_messages(self):
        session = _make_session([
            _make_message("user", input_tokens=999, output_tokens=999),
            _make_message("assistant", input_tokens=100, output_tokens=50),
        ])
        inp, out, create, read = _aggregate_usage(session)
        assert inp == 100
        assert out == 50

    def test_handles_none_usage(self):
        msg = Message(uuid="m1", role="assistant", usage=None)
        session = _make_session([msg])
        inp, out, create, read = _aggregate_usage(session)
        assert inp == 0
        assert out == 0

    def test_empty_session(self):
        session = _make_session([])
        inp, out, create, read = _aggregate_usage(session)
        assert inp == 0
        assert out == 0
        assert create == 0
        assert read == 0


class TestCacheHitRate:
    def test_basic_rate(self):
        assert _compute_cache_hit_rate(800, 200) == pytest.approx(0.8)

    def test_zero_tokens(self):
        assert _compute_cache_hit_rate(0, 0) == 0.0

    def test_no_cache(self):
        assert _compute_cache_hit_rate(0, 500) == pytest.approx(0.0)

    def test_all_cached(self):
        assert _compute_cache_hit_rate(1000, 0) == pytest.approx(1.0)


class TestEstimateCost:
    def test_sonnet_default_pricing(self):
        model = CostModel()
        cost = _estimate_cost(
            input_tok=1_000_000,
            output_tok=100_000,
            cache_read_tok=500_000,
            cache_write_tok=200_000,
            cost_model=model,
        )
        expected = (
            1_000_000 * 3.00 / 1_000_000
            + 100_000 * 15.00 / 1_000_000
            + 500_000 * 0.30 / 1_000_000
            + 200_000 * 3.75 / 1_000_000
        )
        assert cost == pytest.approx(expected)

    def test_custom_cost_model(self):
        model = CostModel(input_rate=6.0, output_rate=30.0, cache_read_rate=0.60, cache_write_rate=7.50)
        cost = _estimate_cost(1_000_000, 1_000_000, 0, 0, model)
        assert cost == pytest.approx(36.0)

    def test_zero_tokens(self):
        assert _estimate_cost(0, 0, 0, 0, CostModel()) == 0.0


class TestPositionalBreakdown:
    def test_all_aligned(self):
        spans = [_make_classified(SpanLabel.ALIGNED_RESPONSE, position=i) for i in range(9)]
        result = _compute_positional_breakdown(spans)
        assert result.early_ter == pytest.approx(1.0)
        assert result.mid_ter == pytest.approx(1.0)
        assert result.late_ter == pytest.approx(1.0)
        assert result.early_span_count == 3
        assert result.mid_span_count == 3
        assert result.late_span_count == 3

    def test_waste_concentrated_late(self):
        spans = [
            _make_classified(SpanLabel.ALIGNED_RESPONSE, position=i) for i in range(6)
        ] + [
            _make_classified(SpanLabel.OVER_EXPLANATION, position=i) for i in range(6, 9)
        ]
        result = _compute_positional_breakdown(spans)
        assert result.early_ter == pytest.approx(1.0)
        assert result.mid_ter == pytest.approx(1.0)
        assert result.late_ter == pytest.approx(0.0)

    def test_single_span(self):
        spans = [_make_classified(SpanLabel.ALIGNED_RESPONSE)]
        result = _compute_positional_breakdown(spans)
        assert result.early_ter == pytest.approx(1.0)
        assert result.early_span_count == 1
        assert result.mid_span_count == 0
        assert result.late_span_count == 0

    def test_empty_spans(self):
        result = _compute_positional_breakdown([])
        assert result.early_ter == pytest.approx(1.0)
        assert result.mid_ter == pytest.approx(1.0)
        assert result.late_ter == pytest.approx(1.0)

    def test_two_spans(self):
        spans = [
            _make_classified(SpanLabel.ALIGNED_RESPONSE, position=0),
            _make_classified(SpanLabel.OVER_EXPLANATION, position=1),
        ]
        result = _compute_positional_breakdown(spans)
        assert result.early_span_count == 1
        assert result.mid_span_count == 1
        assert result.late_span_count == 0


class TestInputGrowth:
    def test_linear_growth(self):
        # Uses input_tokens + cache_read as total context.
        # Turns with context <= 100 are filtered out.
        session = _make_session([
            _make_message("assistant", input_tokens=500, cache_read=500),
            _make_message("assistant", input_tokens=500, cache_read=1500),
            _make_message("assistant", input_tokens=500, cache_read=2500),
        ])
        result = _compute_input_growth(session)
        assert result.turn_input_tokens == [1000, 2000, 3000]
        assert result.growth_rate == pytest.approx(3.0)
        assert result.is_superlinear is False
        assert result.context_bloat_detected is False

    def test_superlinear_growth(self):
        session = _make_session([
            _make_message("assistant", input_tokens=100, cache_read=900),
            _make_message("assistant", input_tokens=100, cache_read=1900),
            _make_message("assistant", input_tokens=100, cache_read=3900),
            _make_message("assistant", input_tokens=100, cache_read=8900),
        ])
        result = _compute_input_growth(session)
        assert result.is_superlinear is True
        assert result.context_bloat_detected is True

    def test_context_bloat_requires_high_growth(self):
        # Superlinear but growth_rate <= 2.0: no bloat.
        session = _make_session([
            _make_message("assistant", input_tokens=500, cache_read=500),
            _make_message("assistant", input_tokens=510, cache_read=500),
            _make_message("assistant", input_tokens=530, cache_read=500),
            _make_message("assistant", input_tokens=560, cache_read=500),
        ])
        result = _compute_input_growth(session)
        assert result.is_superlinear is True
        assert result.context_bloat_detected is False

    def test_single_turn(self):
        session = _make_session([
            _make_message("assistant", input_tokens=200, cache_read=300),
        ])
        result = _compute_input_growth(session)
        assert result.growth_rate == pytest.approx(1.0)
        assert result.is_superlinear is False
        assert result.context_bloat_detected is False
        assert result.turn_input_tokens == [500]

    def test_constant_input(self):
        session = _make_session([
            _make_message("assistant", input_tokens=500, cache_read=500),
            _make_message("assistant", input_tokens=500, cache_read=500),
            _make_message("assistant", input_tokens=500, cache_read=500),
        ])
        result = _compute_input_growth(session)
        assert result.growth_rate == pytest.approx(1.0)
        assert result.is_superlinear is False

    def test_tiny_contexts_filtered(self):
        # Turns with context <= 100 are dropped (handshake/setup).
        session = _make_session([
            _make_message("assistant", input_tokens=3, cache_read=0),
            _make_message("assistant", input_tokens=5, cache_read=0),
            _make_message("assistant", input_tokens=500, cache_read=500),
            _make_message("assistant", input_tokens=500, cache_read=1500),
        ])
        result = _compute_input_growth(session)
        assert result.turn_input_tokens == [1000, 2000]
        assert result.growth_rate == pytest.approx(2.0)

    def test_empty_session(self):
        session = _make_session([])
        result = _compute_input_growth(session)
        assert result.growth_rate == pytest.approx(1.0)
        assert result.turn_input_tokens == []


class TestEstimateWasteCost:
    def test_all_aligned_zero_waste(self):
        spans = [_make_classified(SpanLabel.ALIGNED_RESPONSE, token_count=100)]
        cost, ratio = _estimate_waste_cost(spans, CostModel())
        assert cost == 0.0
        assert ratio == pytest.approx(1.0)

    def test_waste_tokens_costed_at_output_rate(self):
        spans = [_make_classified(SpanLabel.OVER_EXPLANATION, token_count=1_000_000)]
        cost, ratio = _estimate_waste_cost(spans, CostModel())
        # 1M waste tokens × $15/MTok = $15
        assert cost == pytest.approx(15.0)
        assert ratio == pytest.approx(1.0)

    def test_mixed_spans(self):
        spans = [
            _make_classified(SpanLabel.ALIGNED_RESPONSE, token_count=500_000),
            _make_classified(SpanLabel.OVER_EXPLANATION, token_count=200_000),
            _make_classified(SpanLabel.REDUNDANT_REASONING, token_count=300_000),
        ]
        cost, ratio = _estimate_waste_cost(spans, CostModel())
        # 500k waste tokens × $15/MTok = $7.50
        assert cost == pytest.approx(7.5)
        assert ratio == pytest.approx(1.0)

    def test_custom_output_rate(self):
        spans = [_make_classified(SpanLabel.OVER_EXPLANATION, token_count=1_000_000)]
        cost, ratio = _estimate_waste_cost(spans, CostModel(output_rate=30.0))
        assert cost == pytest.approx(30.0)
        assert ratio == pytest.approx(1.0)

    def test_calibration_scales_to_billed_output(self):
        spans = [_make_classified(SpanLabel.OVER_EXPLANATION, token_count=100)]
        cost, ratio = _estimate_waste_cost(
            spans, CostModel(), billed_output_tokens=400,
        )
        assert ratio == pytest.approx(4.0)
        assert cost == pytest.approx(400 * 15.0 / 1_000_000)

    def test_user_origin_waste_not_in_output_waste_cost(self):
        span = TokenSpan(
            text="x",
            phase=SpanPhase.GENERATION,
            position=0,
            token_count=500_000,
            source_message_uuid="u1",
            source_role="user",
        )
        spans = [
            ClassifiedSpan(
                span=span,
                label=SpanLabel.OVER_EXPLANATION,
                confidence=0.5,
                cosine_similarity=0.1,
            ),
        ]
        cost, ratio = _estimate_waste_cost(
            spans, CostModel(), billed_output_tokens=100,
        )
        assert cost == 0.0
        assert ratio == pytest.approx(1.0)


class TestComputeEconomics:
    def test_full_computation(self):
        session = _make_session([
            _make_message("assistant", input_tokens=200, output_tokens=100, cache_creation=50, cache_read=800),
            _make_message("assistant", input_tokens=300, output_tokens=150, cache_creation=30, cache_read=1200),
        ])
        spans = [_make_classified(SpanLabel.ALIGNED_RESPONSE, position=i) for i in range(6)]

        result = compute_economics(session, spans)

        assert result.total_input_tokens == 500
        assert result.total_output_tokens == 250
        assert result.total_cache_creation_tokens == 80
        assert result.total_cache_read_tokens == 2000
        assert result.cache_hit_rate == pytest.approx(2000 / (2000 + 500), abs=0.001)
        assert result.input_output_ratio == pytest.approx(2.0)
        assert result.estimated_cost_usd > 0
        assert result.estimated_waste_cost_usd == 0.0  # all aligned
        assert result.positional.early_span_count > 0
        # Total context = input_tokens + cache_read per turn
        assert result.input_growth.turn_input_tokens == [1000, 1500]

    def test_with_custom_cost_model(self):
        session = _make_session([
            _make_message("assistant", input_tokens=1_000_000, output_tokens=100_000),
        ])
        spans = [_make_classified()]
        default = compute_economics(session, spans)
        custom = compute_economics(session, spans, CostModel(input_rate=6.0, output_rate=30.0))
        assert custom.estimated_cost_usd > default.estimated_cost_usd
