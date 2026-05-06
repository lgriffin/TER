"""Tests for TER computation."""

import numpy as np
import pytest

from ter_calculator.compute import compute_ter
from ter_calculator.models import (
    ALIGNED_LABELS,
    ClassifiedSpan,
    SpanLabel,
    SpanPhase,
    TokenSpan,
)


def _make_span(phase: SpanPhase, token_count: int, position: int = 0) -> TokenSpan:
    return TokenSpan(
        text="test",
        phase=phase,
        position=position,
        token_count=token_count,
        source_message_uuid="msg-1",
    )


def _make_classified(
    phase: SpanPhase,
    label: SpanLabel,
    token_count: int,
    position: int = 0,
) -> ClassifiedSpan:
    return ClassifiedSpan(
        span=_make_span(phase, token_count, position),
        label=label,
        confidence=0.9,
        cosine_similarity=0.8,
    )


class TestComputeTer:
    def test_all_aligned(self):
        spans = [
            _make_classified(SpanPhase.REASONING, SpanLabel.ALIGNED_REASONING, 100),
            _make_classified(SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL, 100),
            _make_classified(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE, 100),
        ]
        result = compute_ter(spans, session_id="test")
        assert result.aggregate_ter == pytest.approx(1.0)
        assert result.raw_ratio == pytest.approx(1.0)
        assert result.total_tokens == 300
        assert result.aligned_tokens == 300
        assert result.waste_tokens == 0

    def test_all_waste(self):
        spans = [
            _make_classified(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, 100),
            _make_classified(SpanPhase.TOOL_USE, SpanLabel.UNNECESSARY_TOOL_CALL, 100),
            _make_classified(SpanPhase.GENERATION, SpanLabel.OVER_EXPLANATION, 100),
        ]
        result = compute_ter(spans, session_id="test")
        assert result.aggregate_ter == pytest.approx(0.0)
        assert result.raw_ratio == pytest.approx(0.0)
        assert result.waste_tokens == 300

    def test_mixed_spans(self):
        spans = [
            _make_classified(SpanPhase.REASONING, SpanLabel.ALIGNED_REASONING, 80),
            _make_classified(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, 20),
            _make_classified(SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL, 50),
            _make_classified(SpanPhase.TOOL_USE, SpanLabel.UNNECESSARY_TOOL_CALL, 50),
            _make_classified(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE, 60),
            _make_classified(SpanPhase.GENERATION, SpanLabel.OVER_EXPLANATION, 40),
        ]
        result = compute_ter(spans, session_id="test")

        assert result.phase_scores["reasoning"] == pytest.approx(0.8)
        assert result.phase_scores["tool_use"] == pytest.approx(0.5)
        assert result.phase_scores["generation"] == pytest.approx(0.6)

        # Weighted: 0.3*0.8 + 0.4*0.5 + 0.3*0.6 = 0.24 + 0.20 + 0.18 = 0.62
        assert result.aggregate_ter == pytest.approx(0.62, abs=0.001)

    def test_empty_spans(self):
        result = compute_ter([], session_id="test")
        assert result.aggregate_ter == pytest.approx(1.0)
        assert result.total_tokens == 0

    def test_single_phase_only(self):
        spans = [
            _make_classified(SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL, 100),
        ]
        result = compute_ter(spans, session_id="test")
        # Reasoning and generation have no tokens → 1.0 score.
        # Tool use is 100% aligned → 1.0 score.
        assert result.aggregate_ter == pytest.approx(1.0)

    def test_session_id_propagated(self):
        result = compute_ter([], session_id="my-session")
        assert result.session_id == "my-session"

    def test_custom_phase_weights(self):
        spans = [
            _make_classified(SpanPhase.REASONING, SpanLabel.ALIGNED_REASONING, 100),
            _make_classified(SpanPhase.TOOL_USE, SpanLabel.UNNECESSARY_TOOL_CALL, 100),
            _make_classified(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE, 100),
        ]
        weights = {
            SpanPhase.REASONING: 0.5,
            SpanPhase.TOOL_USE: 0.1,
            SpanPhase.GENERATION: 0.4,
        }
        result = compute_ter(spans, session_id="test", phase_weights=weights)
        # 0.5*1.0 + 0.1*0.0 + 0.4*1.0 = 0.9
        assert result.aggregate_ter == pytest.approx(0.9)
