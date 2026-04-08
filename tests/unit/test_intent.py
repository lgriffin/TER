"""Tests for intent extraction."""

import pytest

from ter_calculator.intent import _compute_confidence, _combine_prompts_weighted


class TestComputeConfidence:
    def test_empty_prompts(self):
        assert _compute_confidence([]) == 0.0

    def test_single_word(self):
        conf = _compute_confidence(["fix"])
        assert conf <= 0.3

    def test_two_words(self):
        conf = _compute_confidence(["fix bug"])
        assert 0.2 <= conf <= 0.4

    def test_medium_prompt(self):
        conf = _compute_confidence(["Add a login page with email"])
        assert 0.5 <= conf <= 0.8

    def test_detailed_prompt(self):
        conf = _compute_confidence([
            "Add a login page with email and password validation "
            "that redirects to the dashboard after successful login"
        ])
        assert conf >= 0.8

    def test_multiple_prompts_boost(self):
        single = _compute_confidence(["Add login"])
        multi = _compute_confidence(["Add login", "Use email and password"])
        assert multi >= single

    def test_confidence_capped(self):
        conf = _compute_confidence(["a"] * 100)
        assert conf <= 1.0


class TestCombinePromptsWeighted:
    def test_single_prompt(self):
        result = _combine_prompts_weighted(["Hello world"])
        assert result == "Hello world"

    def test_multiple_prompts_weights_later(self):
        result = _combine_prompts_weighted(["first", "second"])
        # Later prompts should appear more times than earlier ones.
        assert result.count("second") > result.count("first")

    def test_preserves_all_prompts(self):
        result = _combine_prompts_weighted(["alpha", "beta", "gamma"])
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result
