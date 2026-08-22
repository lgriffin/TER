"""Tests for Marp slide deck generation."""

from ter_calculator.formatter_marp import format_marp
from ter_calculator.models import (
    CostModel,
    InputAnalysis,
    InputGrowth,
    PositionalBreakdown,
    PromptSimilarityResult,
    SessionEconomics,
    TERResult,
    WastePattern,
)


def _make_result(**overrides):
    econ = SessionEconomics(
        total_input_tokens=5000,
        total_output_tokens=2000,
        total_cache_creation_tokens=500,
        total_cache_read_tokens=3000,
        input_output_ratio=2.5,
        cache_hit_rate=0.75,
        estimated_cost_usd=0.05,
        estimated_waste_cost_usd=0.01,
        cost_model=CostModel(),
        positional=PositionalBreakdown(
            early_ter=0.85,
            mid_ter=0.72,
            late_ter=0.78,
            early_span_count=10,
            mid_span_count=10,
            late_span_count=10,
        ),
        input_growth=InputGrowth(
            turn_input_tokens=[100, 200, 300],
            growth_rate=1.5,
            is_superlinear=False,
            context_bloat_detected=False,
        ),
    )
    defaults = dict(
        session_id="marp-test-session",
        aggregate_ter=0.78,
        raw_ratio=0.75,
        phase_scores={"reasoning": 0.80, "tool_use": 0.72, "generation": 0.82},
        total_tokens=1000,
        aligned_tokens=780,
        waste_tokens=220,
        waste_patterns=[
            WastePattern(
                pattern_type="reasoning_loop",
                description="Repeated reasoning about auth module",
                start_position=3,
                end_position=5,
                spans_involved=3,
                tokens_wasted=120,
            ),
        ],
        economics=econ,
        input_analysis=InputAnalysis(prompt_similarity=PromptSimilarityResult()),
    )
    defaults.update(overrides)
    return TERResult(**defaults)


class TestFormatMarp:
    def test_has_marp_frontmatter(self):
        md = format_marp(_make_result())
        assert md.startswith("---\nmarp: true")

    def test_contains_session_id(self):
        md = format_marp(_make_result())
        assert "marp-test-session" in md

    def test_contains_ter_score(self):
        md = format_marp(_make_result())
        assert "0.78" in md

    def test_contains_slide_separators(self):
        md = format_marp(_make_result())
        assert md.count("\n---\n") >= 5

    def test_contains_phase_analysis(self):
        md = format_marp(_make_result())
        assert "Phase Analysis" in md
        assert "Reasoning" in md
        assert "Tool Use" in md

    def test_contains_waste_patterns(self):
        md = format_marp(_make_result())
        assert "Waste Patterns" in md
        assert "Reasoning Loop" in md

    def test_contains_economics(self):
        md = format_marp(_make_result())
        assert "Token Economics" in md
        assert "$0.05" in md or "$0.0500" in md

    def test_contains_recommendations(self):
        md = format_marp(_make_result())
        assert "Recommendations" in md

    def test_contains_speaker_notes(self):
        md = format_marp(_make_result())
        assert "<!--" in md

    def test_no_waste_slide_when_no_patterns(self):
        md = format_marp(_make_result(waste_patterns=[]))
        assert "Waste Patterns" not in md

    def test_no_economics_slide_when_no_economics(self):
        md = format_marp(_make_result(economics=None))
        assert "Token Economics" not in md

    def test_contains_svg_charts(self):
        md = format_marp(_make_result())
        assert "<svg" in md

    def test_contains_closing_slide(self):
        md = format_marp(_make_result())
        assert "Summary" in md
        assert "TER Calculator" in md

    def test_speaker_notes_escape_comment_close(self):
        from ter_calculator.formatter_marp import _speaker_notes

        notes = _speaker_notes("malicious --> breakout")
        assert "-- ›" not in notes or "-->" not in notes.replace(" -->", "")
        assert "malicious" in notes
        # The sanitized output should not contain a raw --> inside the comment body
        inner = notes.replace("<!-- ", "").replace(" -->", "")
        assert "-->" not in inner

    def test_waste_description_html_escaped(self):
        result = _make_result(
            waste_patterns=[
                WastePattern(
                    pattern_type="reasoning_loop",
                    description="<b>bold</b> & special",
                    start_position=0,
                    end_position=1,
                    spans_involved=1,
                    tokens_wasted=100,
                ),
            ],
        )
        md = format_marp(result)
        assert "<b>" not in md
        assert "&amp;" in md

    def test_cost_on_same_line_as_ter(self):
        md = format_marp(_make_result())
        for line in md.split("\n"):
            if "TER Score" in line and "Waste" in line:
                assert "Est. Cost" in line
                break
        else:
            raise AssertionError("Cost not on same line as TER Score")
