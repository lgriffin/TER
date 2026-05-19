"""Unit tests for ter_calculator.token_counting module."""

from __future__ import annotations

import pytest

from ter_calculator.token_counting import (
    CountMethod,
    PhaseMultipliers,
    TokenCountResult,
    _code_density,
    calibrate_multiplier,
    count_tokens,
    estimate_tokens_heuristic,
    token_count_confidence,
)


# ── _code_density ──────────────────────────────────────────────────────────


class TestCodeDensity:
    def test_empty_text_returns_zero(self):
        assert _code_density("") == 0.0

    def test_natural_text_low_density(self):
        text = "This is a simple natural language sentence with no code."
        density = _code_density(text)
        assert density < 0.05

    def test_code_text_higher_density(self):
        text = "if (x > 0) { return arr[i]; }"
        density = _code_density(text)
        assert density > 0.1

    def test_pure_punctuation_capped_at_one(self):
        text = "{}[]();=<>"
        density = _code_density(text)
        assert density <= 1.0

    def test_json_like_text(self):
        text = '{"key": "value", "list": [1, 2, 3]}'
        density = _code_density(text)
        assert density > 0.05


# ── estimate_tokens_heuristic ──────────────────────────────────────────────


class TestEstimateTokensHeuristic:
    def test_empty_text_returns_zero(self):
        assert estimate_tokens_heuristic("") == 0

    def test_normal_text_default_ratio(self):
        text = "a" * 40  # 40 chars / 4.0 = 10 tokens
        result = estimate_tokens_heuristic(text)
        assert result == 10

    def test_with_reasoning_phase(self):
        text = "a" * 40  # ratio 4.0 => 10 tokens
        result = estimate_tokens_heuristic(text, phase="reasoning")
        assert result == 10

    def test_with_tool_use_phase(self):
        text = "a" * 32  # ratio 3.2 => 10 tokens
        result = estimate_tokens_heuristic(text, phase="tool_use")
        assert result == 10

    def test_with_custom_multipliers(self):
        custom = PhaseMultipliers(reasoning=2.0, generation=2.0, tool_use=2.0)
        text = "a" * 20  # 20 / 2.0 = 10
        result = estimate_tokens_heuristic(
            text, phase="reasoning", multipliers=custom
        )
        assert result == 10

    def test_unknown_phase_falls_back_to_default(self):
        text = "a" * 40  # default ratio 4.0 => 10
        result = estimate_tokens_heuristic(text, phase="unknown_phase")
        assert result == 10

    def test_no_phase_uses_default_ratio(self):
        text = "a" * 100  # 100 / 4.0 = 25
        result = estimate_tokens_heuristic(text)
        assert result == 25

    def test_result_is_non_negative(self):
        # Even for very short text, result should be >= 0
        result = estimate_tokens_heuristic("a")
        assert result >= 0

    def test_rounding(self):
        # 5 chars / 4.0 = 1.25, rounds to 1
        assert estimate_tokens_heuristic("a" * 5) == 1
        # 6 chars / 4.0 = 1.5, rounds to 2
        assert estimate_tokens_heuristic("a" * 6) == 2


# ── calibrate_multiplier ──────────────────────────────────────────────────


class TestCalibrateMultiplier:
    def test_normal_samples(self):
        # If text has 40 chars and known count is 10, multiplier = 40/10 = 4.0
        # OLS formula: m = sum(c*t) / sum(t*t) = (40*10)/(10*10) = 4.0
        samples = [("a" * 40, 10)]
        result = calibrate_multiplier(samples)
        assert result == pytest.approx(4.0)

    def test_multiple_samples(self):
        samples = [
            ("a" * 40, 10),  # c=40, t=10
            ("b" * 80, 20),  # c=80, t=20
        ]
        # sum_ct = 40*10 + 80*20 = 400 + 1600 = 2000
        # sum_tt = 10*10 + 20*20 = 100 + 400 = 500
        # m = 2000/500 = 4.0
        result = calibrate_multiplier(samples)
        assert result == pytest.approx(4.0)

    def test_different_ratios(self):
        samples = [
            ("a" * 30, 10),  # ratio 3.0
            ("b" * 50, 10),  # ratio 5.0
        ]
        # sum_ct = 30*10 + 50*10 = 300 + 500 = 800
        # sum_tt = 10*10 + 10*10 = 100 + 100 = 200
        # m = 800/200 = 4.0
        result = calibrate_multiplier(samples)
        assert result == pytest.approx(4.0)

    def test_empty_samples_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            calibrate_multiplier([])

    def test_all_zero_token_counts_raises_value_error(self):
        samples = [("some text", 0), ("more text", 0)]
        with pytest.raises(ValueError, match="zero or negative"):
            calibrate_multiplier(samples)

    def test_negative_token_counts_skipped(self):
        # Negative counts are skipped; if all are negative => error
        samples = [("text", -5)]
        with pytest.raises(ValueError, match="zero or negative"):
            calibrate_multiplier(samples)

    def test_mixed_valid_and_invalid(self):
        samples = [
            ("a" * 40, 10),  # valid
            ("ignored", 0),  # skipped (zero)
            ("also ignored", -1),  # skipped (negative)
        ]
        # Only first sample contributes: m = (40*10)/(10*10) = 4.0
        result = calibrate_multiplier(samples)
        assert result == pytest.approx(4.0)


# ── token_count_confidence ─────────────────────────────────────────────────


class TestTokenCountConfidence:
    def test_api_method_always_1_0(self):
        assert token_count_confidence("any text", CountMethod.API) == 1.0

    def test_api_method_with_code_still_1_0(self):
        code_text = "if (x) { return arr[i]; }"
        assert token_count_confidence(code_text, CountMethod.API) == 1.0

    def test_heuristic_normal_text(self):
        text = "This is a normal English sentence without code."
        conf = token_count_confidence(text, CountMethod.HEURISTIC)
        # Base 0.8, low density -> minimal penalty
        assert 0.75 <= conf <= 0.80

    def test_heuristic_code_heavy_lower_confidence(self):
        code = "{[()];=<>{[()];=<>}"
        conf = token_count_confidence(code, CountMethod.HEURISTIC)
        # Code density is high -> larger penalty from base 0.8
        assert conf < 0.80

    def test_calibrated_normal_text(self):
        text = "Normal text for calibrated counting."
        conf = token_count_confidence(text, CountMethod.CALIBRATED)
        # Base 0.9, low density -> near 0.9
        assert 0.85 <= conf <= 0.90

    def test_calibrated_code_text_penalized(self):
        code = "function() { return {}; }"
        conf = token_count_confidence(code, CountMethod.CALIBRATED)
        assert conf < 0.90

    def test_confidence_never_exceeds_one(self):
        conf = token_count_confidence("hello", CountMethod.API)
        assert conf <= 1.0

    def test_confidence_never_below_zero(self):
        # Even with maximum code density, confidence >= 0
        extreme = "{" * 1000
        conf = token_count_confidence(extreme, CountMethod.HEURISTIC)
        assert conf >= 0.0

    def test_empty_text_heuristic(self):
        conf = token_count_confidence("", CountMethod.HEURISTIC)
        # Empty text -> density 0.0 -> no penalty -> base 0.8
        assert conf == pytest.approx(0.8)


# ── count_tokens ───────────────────────────────────────────────────────────


class TestCountTokens:
    def test_empty_text(self):
        result = count_tokens("")
        assert result.estimated_tokens == 0
        assert result.confidence == 1.0
        assert result.method_used is CountMethod.HEURISTIC

    def test_default_heuristic(self):
        text = "a" * 40
        result = count_tokens(text)
        assert result.estimated_tokens == 10
        assert result.method_used is CountMethod.HEURISTIC
        assert 0.0 <= result.confidence <= 1.0

    def test_with_phase(self):
        text = "a" * 32
        result = count_tokens(text, phase="tool_use")
        assert result.estimated_tokens == 10
        assert result.method_used is CountMethod.HEURISTIC

    def test_with_calibrated_multiplier(self):
        text = "a" * 50
        result = count_tokens(text, calibrated_multiplier=5.0)
        assert result.estimated_tokens == 10  # 50 / 5.0
        assert result.method_used is CountMethod.CALIBRATED
        assert result.confidence <= 0.9  # calibrated base

    def test_calibrated_takes_precedence_over_heuristic(self):
        text = "a" * 40
        result = count_tokens(
            text, phase="reasoning", calibrated_multiplier=4.0
        )
        # Calibrated path should be chosen over heuristic
        assert result.method_used is CountMethod.CALIBRATED

    def test_zero_calibrated_multiplier_falls_to_heuristic(self):
        text = "a" * 40
        result = count_tokens(text, calibrated_multiplier=0.0)
        assert result.method_used is CountMethod.HEURISTIC

    def test_negative_calibrated_multiplier_falls_to_heuristic(self):
        text = "a" * 40
        result = count_tokens(text, calibrated_multiplier=-1.0)
        assert result.method_used is CountMethod.HEURISTIC

    def test_use_api_false_skips_api(self):
        text = "some text"
        result = count_tokens(text, use_api=False)
        assert result.method_used in (CountMethod.HEURISTIC, CountMethod.CALIBRATED)

    def test_custom_multipliers_passed_through(self):
        custom = PhaseMultipliers(reasoning=2.0, generation=2.0, tool_use=2.0)
        text = "a" * 20  # 20 / 2.0 = 10
        result = count_tokens(text, phase="reasoning", multipliers=custom)
        assert result.estimated_tokens == 10
        assert result.method_used is CountMethod.HEURISTIC

    def test_result_is_token_count_result(self):
        result = count_tokens("hello world")
        assert isinstance(result, TokenCountResult)

    def test_result_is_frozen(self):
        result = count_tokens("hello world")
        with pytest.raises(AttributeError):
            result.estimated_tokens = 999  # type: ignore[misc]


# ── PhaseMultipliers defaults ──────────────────────────────────────────────


class TestPhaseMultipliers:
    def test_default_values(self):
        pm = PhaseMultipliers()
        assert pm.reasoning == 4.0
        assert pm.generation == 4.0
        assert pm.tool_use == 3.2

    def test_custom_values(self):
        pm = PhaseMultipliers(reasoning=3.0, generation=5.0, tool_use=2.5)
        assert pm.reasoning == 3.0
        assert pm.generation == 5.0
        assert pm.tool_use == 2.5


# ── CountMethod enum ──────────────────────────────────────────────────────


class TestCountMethod:
    def test_values(self):
        assert CountMethod.API.value == "api"
        assert CountMethod.CALIBRATED.value == "calibrated"
        assert CountMethod.HEURISTIC.value == "heuristic"

    def test_members_count(self):
        assert len(CountMethod) == 3
