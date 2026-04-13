"""Tests for output formatting."""

import json

import numpy as np
import pytest

from ter_calculator.formatter import format_ter_result, format_comparison
from ter_calculator.models import (
    CostModel,
    InputGrowth,
    IntentVector,
    PositionalBreakdown,
    SessionEconomics,
    TERResult,
    WastePattern,
)


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
        assert "Phases" in output

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
        assert "Waste Breakdown" in output
        assert "Duplicate Tool Calls" in output

    def test_rich_format_produces_output(self):
        result = _make_result()
        output = format_ter_result(result, fmt="text", use_rich=True)
        assert len(output) > 0
        assert "test-session" in output

    def test_long_session_id_truncated(self):
        result = _make_result(session_id="a3b73c37-ddfa-45bd-9b30-5229f26a99a3")
        output = format_ter_result(result, fmt="text", use_rich=False)
        assert "a3b73c37..." in output

    def test_waste_patterns_collapsed(self):
        wps = [
            WastePattern(
                pattern_type="context_restatement",
                description=f"Response restates prior content (similarity: 0.9{i})",
                start_position=i * 10,
                end_position=i * 10,
                spans_involved=1,
                tokens_wasted=100,
            )
            for i in range(5)
        ]
        result = _make_result(waste_patterns=wps)
        output = format_ter_result(result, fmt="text", use_rich=False)
        # Should show context restatement in waste breakdown with total 500 tokens
        assert "Waste Breakdown" in output
        assert "Context Restatement" in output
        assert "500" in output


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


def _make_economics() -> SessionEconomics:
    return SessionEconomics(
        total_input_tokens=5000,
        total_output_tokens=1500,
        total_cache_creation_tokens=200,
        total_cache_read_tokens=3000,
        input_output_ratio=3.33,
        cache_hit_rate=0.375,
        estimated_cost_usd=0.0384,
        estimated_waste_cost_usd=0.005,
        cost_model=CostModel(),
        positional=PositionalBreakdown(
            early_ter=0.90, mid_ter=0.75, late_ter=0.50,
            early_span_count=5, mid_span_count=5, late_span_count=5,
        ),
        input_growth=InputGrowth(
            turn_input_tokens=[100, 200, 400, 800],
            growth_rate=8.0,
            is_superlinear=True,
            context_bloat_detected=True,
        ),
    )


class TestFormatEconomics:
    def test_text_format_includes_economics(self):
        result = _make_result()
        result.economics = _make_economics()
        output = format_ter_result(result, fmt="text", use_rich=False)
        assert "5,000" in output
        assert "Cache Hit" in output
        assert "Positional TER" in output
        assert "Growth" in output

    def test_json_format_includes_economics(self):
        result = _make_result()
        result.economics = _make_economics()
        output = format_ter_result(result, fmt="json")
        data = json.loads(output)
        assert "economics" in data
        econ = data["economics"]
        assert econ["total_input_tokens"] == 5000
        assert econ["total_output_tokens"] == 1500
        assert econ["cache_hit_rate"] == 0.375
        assert econ["estimated_cost_usd"] == 0.0384
        assert econ["positional"]["early_ter"] == 0.90
        assert econ["input_growth"]["context_bloat_detected"] is True
        assert econ["cost_model"]["input_rate"] == 3.00

    def test_rich_format_includes_economics(self):
        result = _make_result()
        result.economics = _make_economics()
        output = format_ter_result(result, fmt="text", use_rich=True)
        assert "Economics" in output
        assert "Growth" in output

    def test_text_format_no_economics(self):
        result = _make_result()
        result.economics = None
        output = format_ter_result(result, fmt="text", use_rich=False)
        assert "Cache Hit" not in output

    def test_comparison_includes_economics(self):
        r1 = _make_result(session_id="s1", aggregate_ter=0.80)
        r1.economics = _make_economics()
        r2 = _make_result(session_id="s2", aggregate_ter=0.60)
        r2.economics = _make_economics()
        output = format_comparison([r1, r2], fmt="text", use_rich=False)
        assert "Cache%" in output
        assert "Cost" in output
