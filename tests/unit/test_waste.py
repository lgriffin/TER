"""Tests for waste pattern detection."""

import numpy as np
import pytest

from ter_calculator.waste import (
    detect_reasoning_loops,
    detect_duplicate_tool_calls,
    detect_context_restatement,
    detect_waste_patterns,
)
from ter_calculator.models import (
    ClassifiedSpan,
    SpanLabel,
    SpanPhase,
    TokenSpan,
)


def _make_cs(
    phase: SpanPhase,
    label: SpanLabel,
    text: str = "test",
    position: int = 0,
    token_count: int = 50,
    embedding: np.ndarray | None = None,
    block_type: str = "",
) -> ClassifiedSpan:
    # Default block_type based on phase for convenience.
    if not block_type:
        if phase == SpanPhase.TOOL_USE:
            block_type = "tool_use"
        elif phase == SpanPhase.REASONING:
            block_type = "thinking"
        else:
            block_type = "text"
    span = TokenSpan(
        text=text,
        phase=phase,
        position=position,
        token_count=token_count,
        source_message_uuid="msg-1",
        block_type=block_type,
        embedding=embedding,
    )
    return ClassifiedSpan(
        span=span,
        label=label,
        confidence=0.9,
        cosine_similarity=0.5,
    )


class TestReasoningLoops:
    def test_detects_three_consecutive(self):
        spans = [
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=i)
            for i in range(3)
        ]
        patterns = detect_reasoning_loops(spans)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "reasoning_loop"
        assert patterns[0].spans_involved == 3

    def test_no_pattern_with_two(self):
        spans = [
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=i)
            for i in range(2)
        ]
        patterns = detect_reasoning_loops(spans)
        assert len(patterns) == 0

    def test_interrupted_sequence(self):
        spans = [
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=0),
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=1),
            _make_cs(SpanPhase.REASONING, SpanLabel.ALIGNED_REASONING, position=2),
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=3),
        ]
        patterns = detect_reasoning_loops(spans)
        assert len(patterns) == 0

    def test_tokens_wasted_sum(self):
        spans = [
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING,
                     position=i, token_count=100)
            for i in range(4)
        ]
        patterns = detect_reasoning_loops(spans)
        assert patterns[0].tokens_wasted == 400


class TestDuplicateToolCalls:
    def test_detects_duplicate(self):
        spans = [
            _make_cs(SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
                     text="Bash ls -la", position=0),
            _make_cs(SpanPhase.TOOL_USE, SpanLabel.UNNECESSARY_TOOL_CALL,
                     text="Bash ls -la", position=1),
        ]
        patterns = detect_duplicate_tool_calls(spans)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "duplicate_tool_call"

    def test_no_duplicate_different_calls(self):
        spans = [
            _make_cs(SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
                     text="Bash ls -la", position=0),
            _make_cs(SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
                     text="Read file.py", position=1),
        ]
        patterns = detect_duplicate_tool_calls(spans)
        assert len(patterns) == 0

    def test_outside_window(self):
        spans = []
        # First tool call at position 0.
        spans.append(_make_cs(
            SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
            text="Bash ls -la", position=0,
        ))
        # Fill window with different calls.
        for i in range(1, 7):
            spans.append(_make_cs(
                SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
                text=f"Bash cmd-{i}", position=i,
            ))
        # Duplicate beyond window.
        spans.append(_make_cs(
            SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
            text="Bash ls -la", position=7,
        ))
        patterns = detect_duplicate_tool_calls(spans, window_size=5)
        assert len(patterns) == 0


class TestContextRestatement:
    def test_detects_restatement(self):
        # Two spans with very similar embeddings.
        emb = np.random.rand(384).astype(np.float32)
        emb_similar = emb + np.random.rand(384).astype(np.float32) * 0.01

        spans = [
            _make_cs(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE,
                     text="first response", position=0, embedding=emb),
            _make_cs(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE,
                     text="similar response", position=1, embedding=emb_similar),
        ]
        patterns = detect_context_restatement(spans, similarity_threshold=0.85)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "context_restatement"

    def test_no_restatement_different_content(self):
        emb1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)

        spans = [
            _make_cs(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE,
                     text="first response", position=0, embedding=emb1),
            _make_cs(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE,
                     text="different response", position=1, embedding=emb2),
        ]
        patterns = detect_context_restatement(spans, similarity_threshold=0.85)
        assert len(patterns) == 0


class TestDetectWastePatterns:
    def test_combined_detection(self):
        spans = [
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=0),
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=1),
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=2),
        ]
        patterns = detect_waste_patterns(spans)
        assert len(patterns) >= 1

    def test_empty_spans(self):
        patterns = detect_waste_patterns([])
        assert patterns == []
