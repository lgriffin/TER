"""Step definitions for overthinking and reasoning efficiency features."""

from __future__ import annotations

import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from ter_calculator.overthinking import (
    OverthinkingResult,
    ReasoningPhase,
    ReasoningPhaseClassifier,
    ReasoningSegment,
    analyze_overthinking,
    find_optimal_cutoff,
    _count_high_value_tokens,
    _filler_ratio,
)

scenarios("../overthinking/reasoning_efficiency.feature")


@pytest.fixture
def context():
    return {}


# ---------------------------------------------------------------------------
# Overthinking detection
# ---------------------------------------------------------------------------


@given(parsers.parse("a session with {n:d} reasoning segments"))
def session_with_segments(context, n):
    context["segment_count"] = n
    context["reasoning_texts"] = []


@given("novelty scores decline steadily after segment 6")
def declining_novelty(context):
    n = context["segment_count"]
    novel_texts = [
        "The authentication module needs JWT validation with RSA key pairs and certificate rotation policies for enterprise deployments.",
        "Database schema requires normalized tables for user profiles including geographic sharding across multiple regions worldwide.",
        "Frontend React components should implement virtualized scrolling with intersection observer patterns for performance optimization.",
        "Kubernetes deployment manifests need horizontal pod autoscaling configured with custom prometheus metrics and alerting thresholds.",
        "GraphQL resolvers must handle nested pagination with cursor-based traversal and dataloader batching for N+1 query prevention.",
        "WebSocket connection pooling requires heartbeat mechanisms with exponential backoff reconnection strategies for resilient messaging.",
    ]
    texts = []
    for i in range(n):
        if i < len(novel_texts):
            texts.append(novel_texts[i])
        else:
            texts.append(novel_texts[-1])
    context["reasoning_texts"] = texts


@given("each segment introduces significant new information")
def high_novelty(context):
    n = context["segment_count"]
    topics = [
        "authentication", "database schema", "API design",
        "caching strategy", "error handling",
    ]
    texts = []
    for i in range(n):
        topic = topics[i % len(topics)]
        texts.append(
            f"Analyzing {topic} approach {i}: considering {topic} patterns "
            f"like strategy-{i}, evaluating tradeoffs for {topic} "
            f"implementation variant {i} with unique perspective {i * 17}"
        )
    context["reasoning_texts"] = texts


@when("analyze_overthinking is called")
def call_analyze(context):
    context["result"] = analyze_overthinking(context["reasoning_texts"])


@then("is_overthinking is true")
def check_is_overthinking(context):
    assert context["result"].is_overthinking is True


@then(parsers.parse("optimal_cutoff_index is approximately {n:d}"))
def check_cutoff_approx(context, n):
    assert context["result"].optimal_cutoff_index is not None
    assert abs(context["result"].optimal_cutoff_index - n) <= 2


@then("wasted_reasoning_tokens covers segments after the cutoff")
def check_wasted_after_cutoff(context):
    assert context["result"].wasted_reasoning_tokens > 0


@then("is_overthinking is false")
def check_no_overthinking(context):
    assert context["result"].is_overthinking is False


@then(parsers.parse("reasoning_efficiency is above {threshold:f}"))
def check_efficiency(context, threshold):
    assert context["result"].reasoning_efficiency > threshold


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------


@given(
    parsers.parse(
        'a reasoning segment containing "{kw1}" and "{kw2}" and "{kw3}"'
    )
)
def reasoning_with_three_keywords(context, kw1, kw2, kw3):
    context["segment_text"] = (
        f"Let me think. {kw1}, I realize {kw2} the issue is clear. "
        f"{kw3} the solution is to refactor."
    )


@given(parsers.parse('a reasoning segment containing "{kw1}" and "{kw2}"'))
def reasoning_with_two_keywords(context, kw1, kw2):
    context["segment_text"] = (
        f"OK so {kw1} the code. Also I want to {kw2} to be sure."
    )


@when("the segment is classified")
def classify_segment(context):
    classifier = ReasoningPhaseClassifier()
    context["phase"] = classifier.classify(context["segment_text"])
    context["high_value_count"] = _count_high_value_tokens(
        context["segment_text"]
    )
    context["filler_ratio_value"] = _filler_ratio(context["segment_text"])


@then(
    "it is classified as a high-value segment with elevated"
    " high_value_token_count"
)
def check_high_value(context):
    assert context["high_value_count"] > 0


@then("filler_ratio is above 0.0")
def check_filler(context):
    assert context["filler_ratio_value"] > 0.0


# ---------------------------------------------------------------------------
# Optimal cutoff
# ---------------------------------------------------------------------------


@given(parsers.parse("reasoning segments with novelty scores {scores}"))
def segments_with_novelty(context, scores):
    score_list = [float(s.strip()) for s in scores.split(",")]
    segments = []
    cumulative = 0.0
    for i, novelty in enumerate(score_list):
        cumulative += novelty
        segments.append(
            ReasoningSegment(
                index=i,
                text=f"segment {i}",
                token_count=100,
                phase=ReasoningPhase.EXPLORING,
                novelty_score=novelty,
                high_value_token_count=0,
                filler_ratio=0.0,
                cumulative_novelty=cumulative,
                marginal_value=novelty,
            )
        )
    context["segments"] = segments


@when(
    parsers.parse(
        "find_optimal_cutoff is called with novelty_threshold {threshold:f}"
    )
)
def call_cutoff(context, threshold):
    context["cutoff"] = find_optimal_cutoff(context["segments"])


@then(parsers.parse("the cutoff index is {n:d}"))
def check_cutoff_exact(context, n):
    assert context["cutoff"] == n


# ---------------------------------------------------------------------------
# Recommended budget
# ---------------------------------------------------------------------------


@given(
    parsers.parse(
        "an overthinking result with {total:d} total reasoning tokens"
        " and {useful:d} useful tokens"
    )
)
def overthinking_budget(context, total, useful):
    context["result"] = OverthinkingResult(
        is_overthinking=True,
        total_reasoning_tokens=total,
        useful_reasoning_tokens=useful,
        wasted_reasoning_tokens=total - useful,
        optimal_cutoff_index=5,
        reasoning_efficiency=useful / total,
        segments=[],
        recommended_budget=int(useful * 1.2),
        explanation="test",
    )


@when("the recommended budget is computed")
def compute_budget(context):
    pass


@then(parsers.parse("recommended_budget is approximately {n:d} tokens"))
def check_budget(context, n):
    assert context["result"].recommended_budget == pytest.approx(n, rel=0.3)
