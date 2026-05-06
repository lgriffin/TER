"""Tests for Markdown session report and baseline formatting."""

from ter_calculator.models import (
    CostModel,
    InputAnalysis,
    InputGrowth,
    PositionalBreakdown,
    PromptSimilarityResult,
    SessionEconomics,
    TERResult,
)
from ter_calculator.session_report import (
    format_baseline_markdown,
    format_session_report_markdown,
)


def _minimal_result(session_id: str, ter: float, waste: int, total: int) -> TERResult:
    econ = SessionEconomics(
        total_input_tokens=1000,
        total_output_tokens=500,
        total_cache_creation_tokens=0,
        total_cache_read_tokens=2000,
        input_output_ratio=0.5,
        cache_hit_rate=0.8,
        estimated_cost_usd=0.05,
        estimated_waste_cost_usd=0.01,
        cost_model=CostModel(),
        positional=PositionalBreakdown(
            early_ter=0.8,
            mid_ter=0.7,
            late_ter=0.75,
            early_span_count=10,
            mid_span_count=10,
            late_span_count=10,
        ),
        input_growth=InputGrowth(
            turn_input_tokens=[100, 200],
            growth_rate=2.0,
            is_superlinear=False,
            context_bloat_detected=False,
        ),
        waste_output_calibration_ratio=0.95,
    )
    return TERResult(
        session_id=session_id,
        aggregate_ter=ter,
        raw_ratio=0.9,
        phase_scores={"reasoning": 0.8, "tool_use": 0.85, "generation": 0.82},
        total_tokens=total,
        aligned_tokens=total - waste,
        waste_tokens=waste,
        waste_patterns=[],
        economics=econ,
        input_analysis=InputAnalysis(prompt_similarity=PromptSimilarityResult()),
    )


def test_report_markdown_contains_session_and_ter():
    r = _minimal_result("abc", 0.85, 100, 1000)
    md = format_session_report_markdown(r)
    assert "abc" in md
    assert "0.850" in md or "0.85" in md
    assert "Calibration" in md or "calibration" in md


def test_baseline_markdown_delta():
    a = _minimal_result("before", 0.70, 200, 1000)
    b = _minimal_result("after", 0.80, 150, 1000)
    md = format_baseline_markdown(a, b)
    assert "before" in md
    assert "after" in md
    assert "TER" in md
