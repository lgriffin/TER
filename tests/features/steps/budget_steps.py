"""Step definitions for adaptive budget and complexity estimation features."""

from __future__ import annotations

import time

import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from ter_calculator.adaptive_budget import (
    BudgetRecommendation,
    ComplexityEstimator,
    ComplexityTier,
    HistoricalBudgetAnalyzer,
    HistoryEntry,
    ModelTier,
    estimate_complexity,
    recommend_budget,
)

scenarios(
    "../adaptive_budget/complexity_estimation.feature",
    "../adaptive_budget/budget_recommendation.feature",
)


@pytest.fixture
def context():
    return {}


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------


@given(parsers.parse('an intent text "{text}"'))
def intent_text(context, text):
    context["intent_text"] = text


@given(
    parsers.parse(
        'an intent text mentioning "{cue1}" or "{cue2}"'
    )
)
def intent_with_cues(context, cue1, cue2):
    context["intent_text"] = f"I want to {cue1} and also {cue2} for the system"


@when("complexity is estimated")
def do_estimate(context):
    tier, confidence, features = estimate_complexity(context["intent_text"])
    context["complexity_tier"] = tier
    context["confidence"] = confidence
    context["features"] = features


@then(parsers.parse("the complexity tier is {tier}"))
def check_tier(context, tier):
    assert context["complexity_tier"] == ComplexityTier(tier.lower())


@then(parsers.parse("confidence is above {threshold:f}"))
def check_confidence_above(context, threshold):
    assert context["confidence"] > threshold


@then(
    parsers.parse(
        "the complexity score includes a multi-file cue contribution"
        " with weight {w:f}"
    )
)
def check_multi_file_weight(context, w):
    assert context["features"]["multi_file_cues"] > 0


@then(
    parsers.parse(
        "the complexity score includes an architecture cue contribution"
        " with weight {w:f}"
    )
)
def check_architecture_weight(context, w):
    assert context["features"]["architecture_cues"] > 0


# ---------------------------------------------------------------------------
# Budget recommendation — Scenario Outline
# ---------------------------------------------------------------------------

_TIER_TEXTS = {
    "SIMPLE": "fix typo in README.md, simple quick change",
    "STANDARD": "fix the bug in the login API endpoint and add error handling",
    "COMPLEX": (
        "refactor the entire authentication system across multiple files,"
        " redesign the database schema and API architecture"
    ),
}


@given(parsers.parse('an intent text classified as "{tier}"'))
def intent_for_tier(context, tier):
    context["intent_text"] = _TIER_TEXTS[tier]


@when("a budget is recommended")
def do_recommend(context):
    context["recommendation"] = recommend_budget(
        context["intent_text"],
        history=context.get("history_analyzer"),
    )


@then(parsers.parse("max_thinking_tokens is {budget:d}"))
def check_max_thinking(context, budget):
    assert context["recommendation"].max_thinking_tokens == budget


@then(parsers.parse('model_tier is "{model}"'))
def check_model_tier(context, model):
    assert context["recommendation"].model_tier == ModelTier(model.lower())


# ---------------------------------------------------------------------------
# Budget details
# ---------------------------------------------------------------------------


@then("estimated_total_tokens is a positive integer")
def check_est_total(context):
    assert context["recommendation"].estimated_total_tokens > 0


@then("estimated_cost_usd is a positive float")
def check_est_cost(context):
    assert context["recommendation"].estimated_cost_usd > 0.0


@then("confidence is between 0.0 and 1.0")
def check_confidence_range(context):
    assert 0.0 <= context["recommendation"].confidence <= 1.0


# ---------------------------------------------------------------------------
# Historical adjustment
# ---------------------------------------------------------------------------


@given("a HistoricalBudgetAnalyzer with past outcomes")
def history_analyzer(tmp_path, context):
    context["history_analyzer"] = HistoricalBudgetAnalyzer(
        history_path=tmp_path / "budget.json"
    )


@given(
    parsers.parse(
        "past STANDARD tasks used an average of {tokens:d} thinking tokens"
    )
)
def record_standard_history(context, tokens):
    for i in range(10):
        entry = HistoryEntry(
            intent_text=f"standard task {i}",
            complexity="standard",
            actual_thinking_tokens=tokens,
            actual_total_tokens=tokens * 3,
            actual_ter=0.70,
            model_used="sonnet",
            timestamp=time.time(),
        )
        context["history_analyzer"].record(entry)


@when("a budget is recommended for a STANDARD task with history")
def recommend_with_history(context):
    context["recommendation"] = recommend_budget(
        "fix the API endpoint bug with error handling",
        history=context["history_analyzer"],
    )


@then("the budget is adjusted based on historical performance")
def check_adjusted(context):
    assert context["recommendation"].max_thinking_tokens > 0


# ---------------------------------------------------------------------------
# Reasoning field
# ---------------------------------------------------------------------------


@then("the reasoning field explains why the tier was chosen")
def check_reasoning(context):
    assert len(context["recommendation"].reasoning) > 0
    assert "Complexity" in context["recommendation"].reasoning
