"""Step definitions for cost economics features."""

from __future__ import annotations

import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from ter_calculator.cost_model import (
    PRICING,
    PricingTier,
    TokenCategory,
    SemanticDensityScorer,
    compute_semantic_density,
    compute_cost_weighted_ter,
)

scenarios(
    "../cost_economics/cost_weighted_ter.feature",
    "../cost_economics/semantic_density.feature",
    "../cost_economics/pricing_tiers.feature",
)


@pytest.fixture
def context():
    return {}


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


@given(parsers.parse('the default pricing tier is "{tier}"'))
def default_pricing_tier(context, tier):
    context["default_tier"] = tier


# ---------------------------------------------------------------------------
# Pricing tiers  (pricing_tiers.feature)
# ---------------------------------------------------------------------------


@given(parsers.parse('the "{tier}" pricing tier'))
def pricing_tier(context, tier):
    context["tier"] = PRICING[tier]
    context["tier_name"] = tier


@then(parsers.parse("input_per_mtok is {rate:f}"))
def check_input_rate(context, rate):
    assert context["tier"].input_per_mtok == pytest.approx(rate)


@then(parsers.parse("output_per_mtok is {rate:f}"))
def check_output_rate(context, rate):
    assert context["tier"].output_per_mtok == pytest.approx(rate)


@then(parsers.parse("cached_read_per_mtok is {rate:f}"))
def check_cached_read(context, rate):
    assert context["tier"].cached_read_per_mtok == pytest.approx(rate)


@then(parsers.parse("cached_write_per_mtok is {rate:f}"))
def check_cached_write(context, rate):
    assert context["tier"].cached_write_per_mtok == pytest.approx(rate)


# --- Scenario Outline: Different tiers produce different output costs ---


@given(parsers.parse("a session with {count:d} output tokens"))
def session_with_output_tokens(context, count):
    context["spans"] = [
        {
            "phase": "generation",
            "token_count": count,
            "is_aligned": True,
            "category": "output",
        }
    ]
    context["output_token_count"] = count


@when(parsers.parse('cost is computed for tier "{tier}"'))
def compute_cost_for_tier(context, tier):
    result = compute_cost_weighted_ter(context["spans"], model=tier)
    context["cost_result"] = result
    context["tier_name"] = tier


@then(parsers.parse("the output cost uses rate {rate:f} per million tokens"))
def check_output_cost_rate(context, rate):
    expected = context["output_token_count"] / 1_000_000 * rate
    assert context["cost_result"].total_cost_usd == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Cost-weighted TER  (cost_weighted_ter.feature)
# ---------------------------------------------------------------------------


@given(
    parsers.parse(
        "a session with {out_count:d} waste output tokens and {in_count:d} waste input tokens"
    )
)
def session_waste_output_and_input(context, out_count, in_count):
    context["waste_output_count"] = out_count
    context["waste_input_count"] = in_count
    context["spans"] = [
        {
            "phase": "generation",
            "token_count": out_count,
            "is_aligned": False,
            "category": "output",
        },
        {
            "phase": "generation",
            "token_count": in_count,
            "is_aligned": False,
            "category": "input",
        },
    ]


@when(parsers.parse('cost-weighted TER is computed for the "{tier}" tier'))
def compute_cwter_for_tier(context, tier):
    result = compute_cost_weighted_ter(context["spans"], model=tier)
    context["cost_result"] = result


@then("the waste cost from output tokens exceeds the waste cost from input tokens")
def waste_output_exceeds_input(context):
    result = context["cost_result"]
    output_waste = 0.0
    input_waste = 0.0
    for sc in result.span_costs:
        if not sc.is_aligned:
            if sc.category == TokenCategory.OUTPUT:
                output_waste += sc.dollar_cost
            elif sc.category == TokenCategory.INPUT:
                input_waste += sc.dollar_cost
    assert output_waste > input_waste


# --- Compute total session cost in USD ---


@given(parsers.parse("a session with classified spans totalling {count:d} tokens"))
def session_classified_spans(context, count):
    # Split tokens across aligned output and waste input spans
    aligned_count = count * 7 // 10
    waste_count = count - aligned_count
    context["spans"] = [
        {
            "phase": "generation",
            "token_count": aligned_count,
            "is_aligned": True,
            "category": "output",
        },
        {
            "phase": "generation",
            "token_count": waste_count,
            "is_aligned": False,
            "category": "input",
        },
    ]


@when("cost-weighted TER is computed")
def compute_cwter_default(context):
    model = context.get("default_tier", "sonnet")
    result = compute_cost_weighted_ter(context["spans"], model=model)
    context["cost_result"] = result


@then("total_cost_usd is a positive dollar amount")
def total_cost_positive(context):
    assert context["cost_result"].total_cost_usd > 0


@then("aligned_cost_usd plus waste_cost_usd equals total_cost_usd")
def aligned_plus_waste_equals_total(context):
    r = context["cost_result"]
    assert r.aligned_cost_usd + r.waste_cost_usd == pytest.approx(r.total_cost_usd)


# --- Cost-weighted TER differs from raw TER ---


@given("a session where waste is concentrated in expensive output tokens")
def session_waste_in_expensive_output(context):
    # 50/50 split by token count, but all waste is output (expensive)
    # and all aligned is input (cheap).  raw_ter treats them equally,
    # cost_weighted_ter penalises the expensive waste more.
    context["spans"] = [
        {
            "phase": "generation",
            "token_count": 1000,
            "is_aligned": True,
            "category": "input",
        },
        {
            "phase": "generation",
            "token_count": 1000,
            "is_aligned": False,
            "category": "output",
        },
    ]
    # raw_ter: aligned_tokens / total_tokens = 1000/2000 = 0.5
    context["raw_ter_value"] = 0.5


@then("cost_weighted_ter is lower than raw_ter")
def cwter_lower_than_raw(context):
    r = context["cost_result"]
    # cost_weighted_ter is the aligned fraction by cost.
    # Since waste is in expensive output tokens and aligned is in cheap
    # input tokens, the cost-weighted aligned fraction is lower than the
    # raw token-count fraction.
    raw = context.get("raw_ter_value", r.raw_ter)
    assert r.cost_weighted_ter < raw


# --- Cached tokens are costed at reduced rate ---


@given(parsers.parse('a session with {count:d} cached read tokens on the "{tier}" tier'))
def session_cached_read_tokens(context, count, tier):
    context["default_tier"] = tier
    context["cached_read_count"] = count
    # Build a span that uses the cached_read category
    context["spans"] = [
        {
            "phase": "generation",
            "token_count": count,
            "is_aligned": True,
            "category": "cached_read",
        },
    ]


@then(
    parsers.parse(
        "cached read tokens are billed at ${cached_rate:f} per million not ${full_rate:f} per million"
    )
)
def cached_billed_at_reduced_rate(context, cached_rate, full_rate):
    count = context["cached_read_count"]
    expected_cached_cost = count / 1_000_000 * cached_rate
    full_cost = count / 1_000_000 * full_rate
    actual_cost = context["cost_result"].total_cost_usd
    assert actual_cost == pytest.approx(expected_cached_cost, rel=1e-4)
    assert actual_cost < full_cost


# --- Thinking tokens billed at output rate ---


@given("a session with reasoning phase spans")
def session_reasoning_phase(context):
    context["spans"] = [
        {
            "phase": "reasoning",
            "token_count": 2000,
            "is_aligned": True,
        },
        {
            "phase": "generation",
            "token_count": 1000,
            "is_aligned": True,
            "category": "output",
        },
    ]


@then("reasoning tokens are categorised as THINKING")
def reasoning_categorised_thinking(context):
    result = context["cost_result"]
    reasoning_spans = [sc for sc in result.span_costs if sc.phase == "reasoning"]
    assert len(reasoning_spans) > 0
    for sc in reasoning_spans:
        assert sc.category == TokenCategory.THINKING


@then("THINKING tokens are billed at the output rate")
def thinking_billed_at_output_rate(context):
    result = context["cost_result"]
    tier = PRICING[result.pricing_tier]
    reasoning_spans = [sc for sc in result.span_costs if sc.phase == "reasoning"]
    for sc in reasoning_spans:
        expected_cost = sc.token_count / 1_000_000 * tier.output_per_mtok
        assert sc.dollar_cost == pytest.approx(expected_cost, rel=1e-4)


# ---------------------------------------------------------------------------
# Semantic density  (semantic_density.feature)
# ---------------------------------------------------------------------------


@given("a text with diverse vocabulary and no repetition")
def diverse_text(context):
    context["text"] = (
        "The authentication module validates JWT tokens by parsing headers, "
        "extracting claims, verifying signatures against the public key, "
        "checking expiration timestamps, and returning decoded payloads."
    )


@given("a text repeating the same sentence 5 times")
def repetitive_text(context):
    sentence = "The function processes the input data and returns results. "
    context["text"] = sentence * 5


@given("an empty text string")
def empty_text(context):
    context["text"] = ""


@given("a text with known vocabulary_richness, information_entropy, and redundancy")
def text_with_known_metrics(context):
    # Use a specific text so we can verify the formula.
    # The scorer will compute real values; we just need to confirm that the
    # formula  density = 0.4*richness + 0.4*norm_entropy + 0.2*(1-redundancy)
    # holds for whatever text we supply.
    context["text"] = (
        "Alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oscar papa quebec romeo sierra tango."
    )


@when("semantic density is computed")
def compute_density(context):
    context["density"] = compute_semantic_density(context["text"])


@then(parsers.parse("density_score is above {threshold:f}"))
def check_density_above(context, threshold):
    assert context["density"].density_score > threshold


@then(parsers.parse("density_score is below {threshold:f}"))
def check_density_below(context, threshold):
    assert context["density"].density_score < threshold


@then(parsers.parse("density_score is {value:f}"))
def check_density_exact(context, value):
    assert context["density"].density_score == pytest.approx(value, abs=0.01)


@then(parsers.parse("vocabulary_richness is above {threshold:f}"))
def check_richness_above(context, threshold):
    assert context["density"].vocabulary_richness > threshold


@then(parsers.parse("redundancy_ratio is below {threshold:f}"))
def check_redundancy_below(context, threshold):
    assert context["density"].redundancy_ratio < threshold


@then(parsers.parse("redundancy_ratio is above {threshold:f}"))
def check_redundancy_above(context, threshold):
    assert context["density"].redundancy_ratio > threshold


@then(
    "density_score equals 0.4 times vocabulary_richness plus 0.4 times normalised_entropy plus 0.2 times one minus redundancy_ratio"
)
def check_density_formula(context):
    d = context["density"]
    import math
    # Recompute normalised entropy from the text to verify the formula
    text = context["text"]
    words = text.lower().split()
    total_words = len(words)
    from collections import Counter
    word_counts = Counter(words)
    entropy = 0.0
    for count in word_counts.values():
        p = count / total_words
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(total_words) if total_words > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    expected = (
        0.4 * d.vocabulary_richness
        + 0.4 * norm_entropy
        + 0.2 * (1.0 - d.redundancy_ratio)
    )
    assert d.density_score == pytest.approx(expected, abs=0.01)
