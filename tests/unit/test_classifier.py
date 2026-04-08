"""Tests for span classification."""

import numpy as np
import pytest

from ter_calculator.classifier import (
    cosine_similarity,
    _classify_span,
    _check_repetition,
)
from ter_calculator.models import SpanLabel, SpanPhase


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero(self):
        a = np.zeros(3)
        assert cosine_similarity(a, a) == 0.0


class TestClassifySpan:
    """Test the aligned-by-default classification logic."""

    def test_repetition_reasoning_is_waste(self):
        """Self-repetition in reasoning phase → redundant reasoning."""
        label, conf = _classify_span(
            sim=0.8, phase=SpanPhase.REASONING,
            is_repetition=True, repetition_similarity=0.92,
            similarity_threshold=0.40, span_text="some reasoning text",
        )
        assert label == SpanLabel.REDUNDANT_REASONING
        assert conf == 0.92

    def test_repetition_tool_use_is_waste(self):
        """Self-repetition in tool_use phase → unnecessary tool call."""
        label, conf = _classify_span(
            sim=0.5, phase=SpanPhase.TOOL_USE,
            is_repetition=True, repetition_similarity=0.90,
            similarity_threshold=0.40, span_text="Bash ls -la",
        )
        assert label == SpanLabel.UNNECESSARY_TOOL_CALL

    def test_repetition_generation_is_waste(self):
        """Self-repetition in generation phase → over-explanation."""
        label, conf = _classify_span(
            sim=0.6, phase=SpanPhase.GENERATION,
            is_repetition=True, repetition_similarity=0.95,
            similarity_threshold=0.40, span_text="here is the answer again",
        )
        assert label == SpanLabel.OVER_EXPLANATION

    def test_reasoning_aligned_by_default(self):
        """Non-repetitive reasoning is aligned even with moderate similarity."""
        label, conf = _classify_span(
            sim=0.3, phase=SpanPhase.REASONING,
            is_repetition=False, repetition_similarity=0.0,
            similarity_threshold=0.40, span_text="Let me think about how to approach this problem step by step",
        )
        assert label == SpanLabel.ALIGNED_REASONING

    def test_reasoning_very_low_sim_short_text_is_waste(self):
        """Very low relevance + short filler text → redundant reasoning."""
        label, conf = _classify_span(
            sim=0.05, phase=SpanPhase.REASONING,
            is_repetition=False, repetition_similarity=0.0,
            similarity_threshold=0.40, span_text="hmm okay let me see",
        )
        assert label == SpanLabel.REDUNDANT_REASONING

    def test_tool_use_always_aligned(self):
        """Tool calls are actions, almost always intentional."""
        label, conf = _classify_span(
            sim=0.05, phase=SpanPhase.TOOL_USE,
            is_repetition=False, repetition_similarity=0.0,
            similarity_threshold=0.40, span_text="Read some/file.py",
        )
        assert label == SpanLabel.ALIGNED_TOOL_CALL

    def test_generation_aligned_by_default(self):
        """Non-repetitive generation is aligned even with lower similarity."""
        label, conf = _classify_span(
            sim=0.2, phase=SpanPhase.GENERATION,
            is_repetition=False, repetition_similarity=0.0,
            similarity_threshold=0.40, span_text="Here is your answer.",
        )
        assert label == SpanLabel.ALIGNED_RESPONSE

    def test_generation_very_low_sim_long_text_is_waste(self):
        """Extremely low relevance + long text → over-explanation."""
        long_text = " ".join(["word"] * 60)
        label, conf = _classify_span(
            sim=0.03, phase=SpanPhase.GENERATION,
            is_repetition=False, repetition_similarity=0.0,
            similarity_threshold=0.40, span_text=long_text,
        )
        assert label == SpanLabel.OVER_EXPLANATION

    def test_high_similarity_reasoning(self):
        """High similarity reasoning is aligned with high confidence."""
        label, conf = _classify_span(
            sim=0.9, phase=SpanPhase.REASONING,
            is_repetition=False, repetition_similarity=0.0,
            similarity_threshold=0.40, span_text="analyzing the user's request for auth",
        )
        assert label == SpanLabel.ALIGNED_REASONING
        assert conf >= 0.5


class TestCheckRepetition:
    def test_no_prior_spans(self):
        """No prior spans → not a repetition."""
        embeddings = np.random.rand(1, 384).astype(np.float32)
        prior_by_phase = {p: [] for p in SpanPhase}
        is_rep, sim = _check_repetition(0, SpanPhase.REASONING, embeddings, prior_by_phase)
        assert is_rep is False
        assert sim == 0.0

    def test_identical_prior_is_repetition(self):
        """Identical embedding to a prior span → repetition."""
        emb = np.random.rand(384).astype(np.float32)
        embeddings = np.stack([emb, emb])  # Two identical embeddings
        prior_by_phase = {p: [] for p in SpanPhase}
        prior_by_phase[SpanPhase.REASONING] = [0]
        is_rep, sim = _check_repetition(1, SpanPhase.REASONING, embeddings, prior_by_phase)
        assert is_rep is True
        assert sim == pytest.approx(1.0)

    def test_different_prior_not_repetition(self):
        """Orthogonal embedding → not a repetition."""
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(384, dtype=np.float32)
        emb2[1] = 1.0
        embeddings = np.stack([emb1, emb2])
        prior_by_phase = {p: [] for p in SpanPhase}
        prior_by_phase[SpanPhase.REASONING] = [0]
        is_rep, sim = _check_repetition(1, SpanPhase.REASONING, embeddings, prior_by_phase)
        assert is_rep is False
