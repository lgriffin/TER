from types import SimpleNamespace
import numpy as np
from ter_calculator.models import *
from ter_calculator.formatter_text import (
    format_text,
    format_comparison_text,
    format_grouped_text,
)
from ter_calculator.formatter_rich import (
    format_rich,
    format_comparison_rich,
    format_grouped_rich,
)


def result(ter=0.4):
    r = TERResult(
        "very-long-session-identifier-12345",
        ter,
        ter,
        {"reasoning": 0.2, "tool_use": 0.5, "generation": 0.8},
        100,
        60,
        40,
        intent=IntentVector("x", np.zeros(2), 0.8, ["x"]),
    )
    r.waste_patterns = [WastePattern("reasoning_loop", "loop", 0, 1, 2, 20)]
    r.economics = SessionEconomics(
        100,
        100,
        0,
        10,
        1,
        0.1,
        0.2,
        0.1,
        CostModel(),
        PositionalBreakdown(0.9, 0.5, 0.2, 1, 1, 1),
        InputGrowth([1, 3], 3, True, True),
    )
    ia = InputAnalysis(
        TokenBreakdown(10, 20, 30, 40, 50, 30, 120, 0.2),
        PromptSimilarityResult([], [PromptPair(0, 1, 0.9, "a" * 50, "b" * 50)], 0.7, 2),
        IntentDrift([IntentDriftStep(0, 1, 0.2, "divergent")], "mixed", 0.2),
        PromptResponseAlignment([PromptResponsePair(0, "p" * 60, "r", 0.2)], 0.2, 1),
    )
    r.input_analysis = ia
    r.cost_report = SimpleNamespace(
        cost_ter=SimpleNamespace(
            cost_weighted_ter=0.7, raw_ter=0.6, total_cost_usd=1, waste_cost_usd=0.2
        ),
        session_density=SimpleNamespace(density_score=0.5, redundancy_ratio=0.3),
        recommendations=["use less"],
    )
    r.overthinking_result = SimpleNamespace(
        is_overthinking=True,
        total_reasoning_tokens=100,
        useful_reasoning_tokens=40,
        reasoning_efficiency=0.4,
        wasted_reasoning_tokens=60,
        optimal_cutoff_index=1,
        segments=[1, 2],
        recommended_budget=50,
        explanation="stop",
    )
    return r


def test_full_text_and_rich_paths():
    r = result()
    t = format_text(r)
    rr = format_rich(r)
    for needle in [
        "Input Analysis",
        "Prompt Redundancy",
        "Intent Drift",
        "Prompt-Response Alignment",
        "Waste Breakdown",
    ]:
        assert needle in t
    for needle in [
        "Cost Analysis",
        "Overthinking Analysis",
        "Recommendations",
        "Input Analysis",
    ]:
        assert needle in rr


def test_comparison_and_grouped_all_formats():
    a, b = result(0.9), result(0.2)
    b.economics = None
    assert "Average TER" in format_comparison_text([a, b])
    assert "TER Comparison" in format_comparison_rich([a, b])
    assert "Group Analysis" in format_grouped_text(a, [b])
    assert "Session Breakdown" in format_grouped_rich(a, [b])
    assert "TER Comparison" in format_comparison_text([])
    assert "TER Comparison" in format_comparison_rich([])


def test_low_information_branches():
    r = result()
    r.total_tokens = 0
    r.economics.input_growth = InputGrowth([1], 1, False, False)
    r.input_analysis = InputAnalysis()
    assert "Waste: 0.0%" in format_text(r)
    assert "Output Tokens" in format_rich(r)
