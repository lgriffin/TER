"""Tests for output formatting."""

import json

import numpy as np
import pytest

from ter_calculator.formatter import format_ter_result, format_comparison
from ter_calculator.models import IntentVector, TERResult, WastePattern


def _make_result(
    session_id: str = "test-session",
    aggregate_ter: float = 0.75,
    waste_patterns: list[WastePattern] | None = None,
) -> TERResult:
    return TERResult(
        session_id=session_id,
        aggregate_ter=aggregate_ter,
        raw_ratio=0.70,
        phase_scores={"reasoning": 0.80, "tool_use": 0.65, "generation": 0.72},
        total_tokens=1000,
        aligned_tokens=700,
        waste_tokens=300,
        waste_patterns=waste_patterns or [],
        intent=IntentVector(
            text="test intent",
            embedding=np.zeros(384),
            confidence=0.85,
            source_prompts=["test intent"],
        ),
    )


class TestFormatTerResult:
    def test_text_format_contains_key_fields(self):
        result = _make_result()
        output = format_ter_result(result, fmt="text", use_rich=False)
        assert "TER Report" in output
        assert "test-session" in output
        assert "0.75" in output
        assert "Phase Scores" in output
        assert "Token Summary" in output

    def test_json_format_valid(self):
        result = _make_result()
        output = format_ter_result(result, fmt="json")
        data = json.loads(output)
        assert data["session_id"] == "test-session"
        assert data["aggregate_ter"] == 0.75
        assert data["phase_scores"]["reasoning"] == 0.80
        assert data["total_tokens"] == 1000

    def test_json_includes_intent_confidence(self):
        result = _make_result()
        output = format_ter_result(result, fmt="json")
        data = json.loads(output)
        assert data["intent_confidence"] == 0.85

    def test_json_includes_waste_patterns(self):
        wp = WastePattern(
            pattern_type="reasoning_loop",
            description="3 redundant spans",
            start_position=5,
            end_position=7,
            spans_involved=3,
            tokens_wasted=150,
        )
        result = _make_result(waste_patterns=[wp])
        output = format_ter_result(result, fmt="json")
        data = json.loads(output)
        assert len(data["waste_patterns"]) == 1
        assert data["waste_patterns"][0]["type"] == "reasoning_loop"

    def test_text_format_shows_waste_patterns(self):
        wp = WastePattern(
            pattern_type="duplicate_tool_call",
            description="Duplicate Bash call",
            start_position=10,
            end_position=12,
            spans_involved=2,
            tokens_wasted=80,
        )
        result = _make_result(waste_patterns=[wp])
        output = format_ter_result(result, fmt="text", use_rich=False)
        assert "Waste Patterns Found: 1" in output
        assert "Duplicate Tool Call" in output

    def test_rich_format_produces_output(self):
        result = _make_result()
        output = format_ter_result(result, fmt="text", use_rich=True)
        assert len(output) > 0
        assert "TER Report" in output


class TestFormatComparison:
    def test_text_comparison(self):
        results = [
            _make_result(session_id="session-a", aggregate_ter=0.80),
            _make_result(session_id="session-b", aggregate_ter=0.60),
        ]
        output = format_comparison(results, fmt="text", use_rich=False)
        assert "TER Comparison" in output
        assert "session-a" in output
        assert "session-b" in output
        assert "Average TER" in output

    def test_json_comparison(self):
        results = [
            _make_result(session_id="s1", aggregate_ter=0.80),
            _make_result(session_id="s2", aggregate_ter=0.60),
        ]
        output = format_comparison(results, fmt="json")
        data = json.loads(output)
        assert len(data["sessions"]) == 2
        assert data["average_ter"] == pytest.approx(0.70, abs=0.001)

    def test_empty_comparison(self):
        output = format_comparison([], fmt="text", use_rich=False)
        assert "TER Comparison" in output
