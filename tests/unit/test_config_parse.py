"""Tests for config_parse module."""

import pytest

from ter_calculator.config_parse import parse_cost_model, parse_phase_weights
from ter_calculator.models import CostModel, SpanPhase


class TestParseCostModel:
    def test_sonnet_returns_default(self):
        model = parse_cost_model("sonnet")
        assert isinstance(model, CostModel)
        assert model.input_rate == 3.00
        assert model.output_rate == 15.00

    def test_sonnet_case_insensitive(self):
        model = parse_cost_model("Sonnet")
        assert isinstance(model, CostModel)

    def test_custom_four_rates(self):
        model = parse_cost_model("1.0,2.0,0.5,1.5")
        assert model.input_rate == 1.0
        assert model.output_rate == 2.0
        assert model.cache_read_rate == 0.5
        assert model.cache_write_rate == 1.5

    def test_three_rates_raises(self):
        with pytest.raises(ValueError, match="4 comma-separated"):
            parse_cost_model("1.0,2.0,3.0")

    def test_five_rates_raises(self):
        with pytest.raises(ValueError, match="4 comma-separated"):
            parse_cost_model("1.0,2.0,3.0,4.0,5.0")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="Invalid cost model"):
            parse_cost_model("a,b,c,d")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_cost_model("")


class TestParsePhaseWeights:
    def test_default_weights(self):
        weights = parse_phase_weights("0.3,0.4,0.3")
        assert weights[SpanPhase.REASONING] == pytest.approx(0.3)
        assert weights[SpanPhase.TOOL_USE] == pytest.approx(0.4)
        assert weights[SpanPhase.GENERATION] == pytest.approx(0.3)

    def test_equal_weights(self):
        weights = parse_phase_weights("0.333,0.334,0.333")
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_two_values_raises(self):
        with pytest.raises(ValueError, match="3 comma-separated"):
            parse_phase_weights("0.5,0.5")

    def test_four_values_raises(self):
        with pytest.raises(ValueError, match="3 comma-separated"):
            parse_phase_weights("0.25,0.25,0.25,0.25")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="Invalid phase weight"):
            parse_phase_weights("a,b,c")

    def test_weights_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            parse_phase_weights("0.3,0.3,0.3")

    def test_returns_span_phase_keys(self):
        weights = parse_phase_weights("0.3,0.4,0.3")
        assert set(weights.keys()) == {
            SpanPhase.REASONING,
            SpanPhase.TOOL_USE,
            SpanPhase.GENERATION,
        }
