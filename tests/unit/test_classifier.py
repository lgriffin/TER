"""Tests for span classification."""

import numpy as np
import pytest

from ter_calculator.classifier import (
    cosine_similarity,
    _classify_span,
    _check_repetition,
    _has_specific_reference,
    _REPETITION_THRESHOLDS,
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
            sim=0.8,
            phase=SpanPhase.REASONING,
            is_repetition=True,
            repetition_similarity=0.92,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="some reasoning text",
        )
        assert label == SpanLabel.REDUNDANT_REASONING
        assert conf == 0.92

    def test_repetition_tool_use_is_waste(self):
        """Self-repetition in tool_use phase → unnecessary tool call."""
        label, conf = _classify_span(
            sim=0.5,
            phase=SpanPhase.TOOL_USE,
            is_repetition=True,
            repetition_similarity=0.90,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="Bash ls -la",
        )
        assert label == SpanLabel.UNNECESSARY_TOOL_CALL

    def test_weak_repetition_respects_confidence_threshold(self):
        """Near-threshold self-similarity stays aligned when confidence bar is high."""
        label, conf = _classify_span(
            sim=0.5,
            phase=SpanPhase.REASONING,
            is_repetition=True,
            repetition_similarity=0.76,
            similarity_threshold=0.40,
            confidence_threshold=0.80,
            span_text="thinking again",
        )
        assert label == SpanLabel.ALIGNED_REASONING

    def test_repetition_generation_is_waste(self):
        """Self-repetition in generation phase → over-explanation."""
        label, conf = _classify_span(
            sim=0.6,
            phase=SpanPhase.GENERATION,
            is_repetition=True,
            repetition_similarity=0.95,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="here is the answer again",
        )
        assert label == SpanLabel.OVER_EXPLANATION

    def test_reasoning_aligned_by_default(self):
        """Non-repetitive reasoning with substantive content is aligned even at moderate similarity.

        The text must be >= 25 words to clear the short-narration gate (gold set
        calibrated: spans < 25 words + sim < 0.35 are action narrations, not analysis).
        """
        text = (
            "The root cause is that tool_result tokens are counted twice: once as "
            "assistant tokens and again when we process the next user message, because "
            "the user-side tool_result block contains the same content as the assistant "
            "turn that generated it."
        )
        assert len(text.split()) >= 25
        label, conf = _classify_span(
            sim=0.3,
            phase=SpanPhase.REASONING,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text=text,
        )
        assert label == SpanLabel.ALIGNED_REASONING

    def test_reasoning_very_low_sim_short_text_is_waste(self):
        """Very low relevance + short filler text → redundant reasoning."""
        label, conf = _classify_span(
            sim=0.05,
            phase=SpanPhase.REASONING,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="hmm okay let me see",
        )
        assert label == SpanLabel.REDUNDANT_REASONING

    def test_tool_use_always_aligned(self):
        """Tool calls are actions, almost always intentional."""
        label, conf = _classify_span(
            sim=0.05,
            phase=SpanPhase.TOOL_USE,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="Read some/file.py",
        )
        assert label == SpanLabel.ALIGNED_TOOL_CALL

    def test_generation_aligned_by_default(self):
        """Non-repetitive generation is aligned even with lower similarity."""
        label, conf = _classify_span(
            sim=0.2,
            phase=SpanPhase.GENERATION,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="Here is your answer.",
        )
        assert label == SpanLabel.ALIGNED_RESPONSE

    def test_generation_very_low_sim_long_text_is_waste(self):
        """Extremely low relevance + long text → over-explanation."""
        long_text = " ".join(["word"] * 60)
        label, conf = _classify_span(
            sim=0.03,
            phase=SpanPhase.GENERATION,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text=long_text,
        )
        assert label == SpanLabel.OVER_EXPLANATION

    def test_high_similarity_reasoning(self):
        """High similarity reasoning is aligned with high confidence."""
        label, conf = _classify_span(
            sim=0.9,
            phase=SpanPhase.REASONING,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="analyzing the user's request for auth",
        )
        assert label == SpanLabel.ALIGNED_REASONING
        assert conf >= 0.5


class TestHasSpecificReference:
    def test_line_number_detected(self):
        assert _has_specific_reference("both pass model=None on lines 341 and 358")

    def test_single_line_detected(self):
        assert _has_specific_reference("add the --dashboard flag after line 89")

    def test_long_integer_detected(self):
        assert _has_specific_reference("the hash is 711bb9b1 which has 462 entries")

    def test_plain_text_no_match(self):
        assert not _has_specific_reference("Let me read that function.")

    def test_short_number_no_match(self):
        assert not _has_specific_reference("there are 2 items to fix")

    def test_line_without_number_no_match(self):
        assert not _has_specific_reference("I'll add a new line here")


class TestShortNarrationGate:
    """Gold set calibrated: short low-sim reasoning → waste."""

    def test_short_action_announcement_is_waste(self):
        """'Let me read X' style spans with sim < 0.35 and < 25 words → waste."""
        label, _ = _classify_span(
            sim=0.21,
            phase=SpanPhase.REASONING,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="Let me read the mapping definition.",
        )
        assert label == SpanLabel.REDUNDANT_REASONING

    def test_good_now_narration_is_waste(self):
        """'Good, I'll add X before Y' style — 20 token narration → waste."""
        label, _ = _classify_span(
            sim=0.18,
            phase=SpanPhase.REASONING,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="Good, I'll add the load_embedding_model function right before the Enums section.",
        )
        assert label == SpanLabel.REDUNDANT_REASONING

    def test_short_span_with_line_ref_stays_aligned(self):
        """Short span with specific line numbers is NOT narration — keep aligned."""
        label, _ = _classify_span(
            sim=0.22,
            phase=SpanPhase.REASONING,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="I need to fix lines 341 and 358 where model=None is passed.",
        )
        assert label == SpanLabel.ALIGNED_REASONING

    def test_long_reasoning_span_stays_aligned_despite_low_sim(self):
        """25+ word reasoning with low sim → still aligned (contains analysis)."""
        text = (
            "The rewrite needs: change the signature, update callers in "
            "classify_spans to accept a list, add repetition_threshold param, "
            "and remove the single-intent fallback path from the pipeline."
        )
        assert len(text.split()) >= 25
        label, _ = _classify_span(
            sim=0.25,
            phase=SpanPhase.REASONING,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text=text,
        )
        assert label == SpanLabel.ALIGNED_REASONING

    def test_sim_above_threshold_stays_aligned(self):
        """Short span with sim >= 0.35 is above the narration gate threshold."""
        label, _ = _classify_span(
            sim=0.38,
            phase=SpanPhase.REASONING,
            is_repetition=False,
            repetition_similarity=0.0,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="Let me check the current model configuration.",
        )
        assert label == SpanLabel.ALIGNED_REASONING


class TestSpecificReferenceRepetitionGuard:
    """Gold set finding: spans with line refs stay aligned even under repetition."""

    def test_repetition_with_line_ref_stays_aligned(self):
        """Span with line numbers fires is_repetition but should remain aligned."""
        label, _ = _classify_span(
            sim=0.44,
            phase=SpanPhase.REASONING,
            is_repetition=True,
            repetition_similarity=0.90,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="Now I need to update lines 341 and 358 where model=None is passed.",
        )
        assert label == SpanLabel.ALIGNED_REASONING

    def test_repetition_without_ref_is_waste(self):
        """Span without specific refs fires is_repetition → waste as before."""
        label, _ = _classify_span(
            sim=0.80,
            phase=SpanPhase.REASONING,
            is_repetition=True,
            repetition_similarity=0.92,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="Good! Now let me find and fix the next occurrence.",
        )
        assert label == SpanLabel.REDUNDANT_REASONING


class TestSystemArtifactGuard:
    """[Request interrupted] spans are system artifacts, never waste."""

    def test_request_interrupted_is_aligned(self):
        label, _ = _classify_span(
            sim=0.85,
            phase=SpanPhase.REASONING,
            is_repetition=True,
            repetition_similarity=0.95,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="[Request interrupted by user]",
        )
        assert label == SpanLabel.ALIGNED_REASONING

    def test_normal_text_not_affected(self):
        label, _ = _classify_span(
            sim=0.80,
            phase=SpanPhase.REASONING,
            is_repetition=True,
            repetition_similarity=0.92,
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            span_text="some repeated reasoning content without interruption",
        )
        assert label == SpanLabel.REDUNDANT_REASONING


class TestPhaseSpecificRepetitionThresholds:
    """Tool calls need a higher repetition bar than reasoning/generation."""

    def test_tool_use_threshold_is_higher_than_reasoning(self):
        assert (
            _REPETITION_THRESHOLDS[SpanPhase.TOOL_USE]
            > _REPETITION_THRESHOLDS[SpanPhase.REASONING]
        )

    def test_tool_use_threshold_value(self):
        assert _REPETITION_THRESHOLDS[SpanPhase.TOOL_USE] == 0.93

    def test_reasoning_threshold_value(self):
        assert _REPETITION_THRESHOLDS[SpanPhase.REASONING] == 0.88

    def test_heterogeneous_tool_calls_not_repetition(self):
        """_check_repetition returns False for tool calls at 0.90 sim (below 0.93 threshold)."""
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0
        # Slightly different — cosine similarity ~0.90 (well below exact 1.0)
        emb2 = np.zeros(384, dtype=np.float32)
        emb2[0] = 0.90
        emb2[1] = 0.436  # norm(emb2) ≈ 1.0 after normalisation
        norm = np.linalg.norm(emb2)
        emb2 /= norm
        embeddings = np.stack([emb1, emb2])
        prior_by_phase = {p: [] for p in SpanPhase}
        prior_by_phase[SpanPhase.TOOL_USE] = [0]

        is_rep, sim = _check_repetition(
            1,
            SpanPhase.TOOL_USE,
            embeddings,
            prior_by_phase,
            repetition_threshold=_REPETITION_THRESHOLDS[SpanPhase.TOOL_USE],
        )
        # 0.90 < 0.93 → not repetition
        assert not is_rep

    def test_identical_tool_calls_still_flagged(self):
        """_check_repetition returns True for near-identical tool calls (≥ 0.93)."""
        emb = np.random.rand(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        embeddings = np.stack([emb, emb])
        prior_by_phase = {p: [] for p in SpanPhase}
        prior_by_phase[SpanPhase.TOOL_USE] = [0]

        is_rep, sim = _check_repetition(
            1,
            SpanPhase.TOOL_USE,
            embeddings,
            prior_by_phase,
            repetition_threshold=_REPETITION_THRESHOLDS[SpanPhase.TOOL_USE],
        )
        assert is_rep
        assert sim == pytest.approx(1.0)

    def test_reasoning_still_uses_lower_threshold(self):
        """_check_repetition fires for reasoning at 0.90 sim (above 0.88 threshold)."""
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(384, dtype=np.float32)
        emb2[0] = 0.90
        emb2[1] = 0.436
        emb2 /= np.linalg.norm(emb2)
        embeddings = np.stack([emb1, emb2])
        prior_by_phase = {p: [] for p in SpanPhase}
        prior_by_phase[SpanPhase.REASONING] = [0]

        is_rep, sim = _check_repetition(
            1,
            SpanPhase.REASONING,
            embeddings,
            prior_by_phase,
            repetition_threshold=_REPETITION_THRESHOLDS[SpanPhase.REASONING],
        )
        assert is_rep


class TestCheckRepetition:
    def test_no_prior_spans(self):
        """No prior spans → not a repetition."""
        embeddings = np.random.rand(1, 384).astype(np.float32)
        prior_by_phase = {p: [] for p in SpanPhase}
        is_rep, sim = _check_repetition(
            0, SpanPhase.REASONING, embeddings, prior_by_phase
        )
        assert is_rep is False
        assert sim == 0.0

    def test_identical_prior_is_repetition(self):
        """Identical embedding to a prior span → repetition."""
        emb = np.random.rand(384).astype(np.float32)
        embeddings = np.stack([emb, emb])  # Two identical embeddings
        prior_by_phase = {p: [] for p in SpanPhase}
        prior_by_phase[SpanPhase.REASONING] = [0]
        is_rep, sim = _check_repetition(
            1, SpanPhase.REASONING, embeddings, prior_by_phase
        )
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
        is_rep, sim = _check_repetition(
            1, SpanPhase.REASONING, embeddings, prior_by_phase
        )
        assert is_rep is False
