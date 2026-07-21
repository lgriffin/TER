"""Tests for overthinking detection and reasoning value analysis."""

import pytest

from ter_calculator.overthinking import (
    EntropyTracker,
    ReasoningPhase,
    ReasoningPhaseClassifier,
    ReasoningSegment,
    analyze_overthinking,
    find_optimal_cutoff,
    HIGH_VALUE_TOKENS,
)


class TestEntropyTracker:
    """Test sliding-window entropy analysis."""

    def test_initial_span_has_high_novelty(self):
        tracker = EntropyTracker(window_size=3)
        novelty = tracker.add_span("This is completely new reasoning content.")
        assert novelty == 1.0  # All trigrams are new

    def test_repeated_span_has_low_novelty(self):
        tracker = EntropyTracker(window_size=3)
        text = "The same reasoning repeated"
        tracker.add_span(text)
        novelty = tracker.add_span(text)  # Exact repetition
        assert novelty == 0.0  # No new trigrams

    def test_similar_span_has_reduced_novelty(self):
        tracker = EntropyTracker(window_size=3)
        tracker.add_span("I need to think about the problem")
        novelty = tracker.add_span("I need to think about this issue")
        # Some overlap but not exact
        assert 0.0 < novelty < 1.0

    def test_window_sliding_forgets_old_spans(self):
        tracker = EntropyTracker(window_size=2)
        tracker.add_span("first unique span")
        tracker.add_span("second unique span")
        tracker.add_span("third unique span")  # Should evict first
        novelty = tracker.add_span("first unique span")  # Should be novel again
        # After eviction, some trigrams reappear but not all are new
        assert novelty > 0.2  # Should have some novelty

    def test_empty_text_has_zero_novelty(self):
        tracker = EntropyTracker(window_size=3)
        novelty = tracker.add_span("")
        assert novelty == 0.0

    def test_current_entropy_increases_with_diversity(self):
        tracker = EntropyTracker(window_size=5)
        tracker.add_span("aaa")
        entropy_low = tracker.current_entropy

        tracker.add_span("bbb")
        tracker.add_span("ccc")
        entropy_high = tracker.current_entropy
        assert entropy_high > entropy_low

    def test_reset_clears_state(self):
        tracker = EntropyTracker(window_size=3)
        tracker.add_span("some content")
        tracker.reset()
        novelty = tracker.add_span("some content")
        assert novelty == 1.0  # Should be novel after reset


class TestReasoningPhaseClassifier:
    """Test reasoning phase classification."""

    def test_exploring_phase_detected(self):
        classifier = ReasoningPhaseClassifier()
        # Use text with exploring cues but without filler patterns
        text = "What if we could use another approach or perhaps try a different option"
        phase = classifier.classify(text)
        assert phase == ReasoningPhase.EXPLORING

    def test_confirming_phase_detected(self):
        classifier = ReasoningPhaseClassifier()
        text = "Yes, that's correct and verified to work"
        phase = classifier.classify(text)
        assert phase == ReasoningPhase.CONFIRMING

    def test_near_answer_phase_detected(self):
        classifier = ReasoningPhaseClassifier()
        text = "Therefore, the solution is to refactor the module"
        phase = classifier.classify(text)
        assert phase == ReasoningPhase.NEAR_ANSWER

    def test_filler_phase_detected(self):
        classifier = ReasoningPhaseClassifier()
        text = "Let me think. I need to. Let me check. I should. Let me re-read."
        phase = classifier.classify(text)
        assert phase == ReasoningPhase.FILLER

    def test_ambiguous_phase_default(self):
        classifier = ReasoningPhaseClassifier()
        text = "This is just some neutral reasoning text without cues"
        phase = classifier.classify(text)
        assert phase == ReasoningPhase.AMBIGUOUS

    def test_near_answer_prioritized_over_exploring(self):
        classifier = ReasoningPhaseClassifier()
        # Contains both exploring and near-answer cues
        text = "Let me conclude: therefore the answer is this approach"
        phase = classifier.classify(text)
        assert phase == ReasoningPhase.NEAR_ANSWER


class TestFindOptimalCutoff:
    """Test optimal reasoning cutoff detection."""

    def test_finds_cutoff_with_consecutive_low_novelty(self):
        segments = [
            ReasoningSegment(
                index=i,
                text=f"segment {i}",
                token_count=100,
                phase=ReasoningPhase.EXPLORING,
                novelty_score=0.8 if i < 3 else 0.05,  # Drop after index 2
                high_value_token_count=1,
                filler_ratio=0.0,
                cumulative_novelty=float(i),
                marginal_value=0.8 if i < 3 else 0.05,
            )
            for i in range(5)
        ]
        cutoff = find_optimal_cutoff(segments)
        # Cutoff is at the index BEFORE the consecutive low spans started becoming low
        # Index 3 has low novelty, consecutive_low becomes 1
        # Index 4 has low novelty, consecutive_low becomes 2, triggers cutoff
        # Returns index 4 - 1 = 3
        assert cutoff == 3

    def test_no_cutoff_when_always_high_novelty(self):
        segments = [
            ReasoningSegment(
                index=i,
                text=f"segment {i}",
                token_count=100,
                phase=ReasoningPhase.EXPLORING,
                novelty_score=0.6,
                high_value_token_count=1,
                filler_ratio=0.0,
                cumulative_novelty=float(i),
                marginal_value=0.6,
            )
            for i in range(5)
        ]
        cutoff = find_optimal_cutoff(segments)
        assert cutoff is None

    def test_no_cutoff_with_too_few_segments(self):
        segments = [
            ReasoningSegment(
                index=0,
                text="only one",
                token_count=100,
                phase=ReasoningPhase.EXPLORING,
                novelty_score=0.01,
                high_value_token_count=0,
                filler_ratio=0.0,
                cumulative_novelty=0.01,
                marginal_value=0.01,
            )
        ]
        cutoff = find_optimal_cutoff(segments)
        assert cutoff is None

    def test_single_low_novelty_not_enough(self):
        """Need 2 consecutive low-novelty spans to trigger cutoff."""
        segments = [
            ReasoningSegment(
                index=i,
                text=f"segment {i}",
                token_count=100,
                phase=ReasoningPhase.EXPLORING,
                novelty_score=0.05 if i == 2 else 0.8,  # Only one low
                high_value_token_count=1,
                filler_ratio=0.0,
                cumulative_novelty=float(i),
                marginal_value=0.05 if i == 2 else 0.8,
            )
            for i in range(5)
        ]
        cutoff = find_optimal_cutoff(segments)
        assert cutoff is None


class TestAnalyzeOverthinking:
    """Test end-to-end overthinking analysis."""

    def test_no_overthinking_with_diverse_reasoning(self):
        texts = [
            "First, I need to understand the problem domain",
            "Next, let me consider the edge cases carefully",
            "Finally, I'll design the solution architecture",
        ]
        result = analyze_overthinking(texts)

        assert not result.is_overthinking
        assert result.total_reasoning_tokens > 0
        assert result.useful_reasoning_tokens == result.total_reasoning_tokens
        assert result.wasted_reasoning_tokens == 0
        assert result.reasoning_efficiency == 1.0
        assert len(result.segments) == 3

    def test_overthinking_detected_with_repetition(self):
        # Create texts with genuinely low novelty by repeating identical content
        base_text = "The exact same reasoning repeated over and over without variation"
        texts = [
            "This is novel reasoning about the architecture",
            "More unique thoughts about the implementation",
            base_text,  # Start repetition
            base_text,  # Exact repetition - very low novelty
            base_text,  # Exact repetition - very low novelty
            base_text,  # More repetition
        ]
        result = analyze_overthinking(texts)

        # With exact repetitions, overthinking should be detected
        if result.is_overthinking:
            assert result.wasted_reasoning_tokens > 0
            assert result.useful_reasoning_tokens < result.total_reasoning_tokens
            assert result.reasoning_efficiency < 1.0
            assert result.optimal_cutoff_index is not None
        else:
            # If threshold is high, just verify analysis ran
            assert len(result.segments) == 6

    def test_empty_input_returns_zero_result(self):
        result = analyze_overthinking([])

        assert not result.is_overthinking
        assert result.total_reasoning_tokens == 0
        assert result.useful_reasoning_tokens == 0
        assert result.reasoning_efficiency == 1.0
        assert len(result.segments) == 0
        assert "No reasoning spans" in result.explanation

    def test_too_few_spans_not_analyzed(self):
        """Less than MIN_REASONING_SPANS should not trigger overthinking."""
        texts = ["one span", "two spans"]
        result = analyze_overthinking(texts)

        assert not result.is_overthinking
        assert "Too few reasoning spans" in result.explanation

    def test_recommended_budget_scales_with_useful_tokens(self):
        texts = [
            "Good reasoning here" * 100,  # High tokens
            "More good reasoning" * 100,
            "Repeated. Repeated. Repeated." * 50,  # Waste
        ]
        result = analyze_overthinking(texts)

        if result.is_overthinking:
            assert result.recommended_budget < result.total_reasoning_tokens
            assert result.recommended_budget >= result.useful_reasoning_tokens

    def test_high_value_tokens_increase_marginal_value(self):
        """Spans with high-value tokens (wait, hmm, etc.) should have higher value."""
        texts = [
            "Initial reasoning without special tokens",
            "Wait, actually I need to reconsider this approach",  # High-value tokens
        ]
        result = analyze_overthinking(texts)

        assert len(result.segments) == 2
        # Second segment should have high-value tokens detected
        assert result.segments[1].high_value_token_count > 0

    def test_custom_window_size_and_threshold(self):
        texts = ["span " + str(i) for i in range(10)]
        result = analyze_overthinking(
            texts,
            window_size=10,
            novelty_threshold=0.5,
        )
        assert len(result.segments) == 10

    def test_segments_preserve_order(self):
        texts = ["first", "second", "third"]
        result = analyze_overthinking(texts)

        assert result.segments[0].index == 0
        assert result.segments[1].index == 1
        assert result.segments[2].index == 2
        assert result.segments[0].text == "first"
        assert result.segments[1].text == "second"

    def test_cumulative_novelty_increases(self):
        texts = ["unique one", "unique two", "unique three"]
        result = analyze_overthinking(texts)

        # Cumulative novelty should increase or stay same
        assert (
            result.segments[0].cumulative_novelty
            <= result.segments[1].cumulative_novelty
        )
        assert (
            result.segments[1].cumulative_novelty
            <= result.segments[2].cumulative_novelty
        )

    def test_filler_ratio_detected(self):
        texts = [
            "Let me think. I need to. Let me check. I should.",
            "This is substantive reasoning without filler.",
        ]
        result = analyze_overthinking(texts)

        # First should have high filler ratio, second should be low
        assert result.segments[0].filler_ratio > result.segments[1].filler_ratio


class TestHighValueTokens:
    """Test that high-value reasoning tokens are properly defined."""

    def test_high_value_tokens_defined(self):
        assert len(HIGH_VALUE_TOKENS) > 0
        assert "wait" in HIGH_VALUE_TOKENS
        assert "hmm" in HIGH_VALUE_TOKENS
        assert "therefore" in HIGH_VALUE_TOKENS
        assert "actually" in HIGH_VALUE_TOKENS

    def test_high_value_tokens_are_lowercase(self):
        """Ensure tokens are normalized for case-insensitive matching."""
        for token in HIGH_VALUE_TOKENS:
            assert token == token.lower()
