"""Step definitions for intent alignment features."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from ter_calculator.intent_extraction import (
    HierarchicalIntentExtractor,
    LLMIntentExtractor,
    SlidingIntentExtractor,
    StructuredGoal,
    create_intent_extractor,
    _prompt_confidence,
    _embed,
    _cosine_similarity,
)

scenarios(
    "../intent_alignment/sliding_intent.feature",
    "../intent_alignment/hierarchical_intent.feature",
    "../intent_alignment/llm_intent.feature",
)


def _table_to_dicts(datatable: list[list[str]]) -> list[dict[str, str]]:
    """Convert pytest-bdd raw datatable (list of lists) to list of dicts."""
    headers = datatable[0]
    return [dict(zip(headers, row)) for row in datatable[1:]]


@pytest.fixture
def context():
    return {}


# ---------------------------------------------------------------------------
# Sliding intent -- Background & Given steps
# ---------------------------------------------------------------------------


@given(
    parsers.parse(
        "a SlidingIntentExtractor with window_size {ws:d} and split_threshold {st:f}"
    )
)
def sliding_extractor(context, ws, st):
    context["extractor"] = SlidingIntentExtractor(window_size=ws, split_threshold=st)


@given("user prompts:")
def user_prompts_table(datatable, context):
    rows = _table_to_dicts(datatable)
    context["prompts"] = [row["prompt"] for row in rows]


@given(parsers.parse('user prompts that shift topic from "{topic_a}" to "{topic_b}"'))
def user_prompts_shift_topic(context, topic_a, topic_b):
    context["prompts"] = [
        f"Add a {topic_a} page with email and password",
        f"Implement {topic_a} session management",
        f"Set up {topic_b} scripts for PostgreSQL",
        f"Create {topic_b} rollback procedures",
    ]


@given(parsers.parse("{n:d} user prompts all about authentication"))
def user_prompts_about_auth(context, n):
    # Prompts must be semantically close enough to stay above the 0.45
    # cosine-similarity threshold so the sliding extractor keeps them
    # in a single segment.
    auth_prompts = [
        "Add user authentication with email and password login",
        "Add user authentication with username and password login",
        "Add user authentication with email and passphrase login",
        "Add user authentication using email-based login",
        "Add user authentication via email password login",
    ]
    context["prompts"] = auth_prompts[:n]


@given(parsers.parse("{n:d} prompts all on the same topic"))
def n_prompts_same_topic(context, n):
    context["prompts"] = [
        f"Add authentication feature number {i + 1} for login"
        for i in range(n)
    ]


@given("no user prompts")
def no_prompts(context):
    context["prompts"] = []


@given("the cosine similarity between adjacent prompts drops below 0.45")
def cosine_drops_below():
    # Informational step -- the prompts constructed in the preceding Given
    # step are designed to produce this behaviour naturally.
    pass


@given("their pairwise cosine similarity is above 0.45")
def cosine_above():
    # Informational step -- the prompts constructed in the preceding Given
    # step are designed so they remain similar.
    pass


# ---------------------------------------------------------------------------
# Sliding intent -- When / Then
# ---------------------------------------------------------------------------


@when("sliding intent extraction runs")
def run_sliding(context):
    context["intents"] = context["extractor"].extract(context.get("prompts", []))


@then(parsers.parse("exactly {n:d} IntentVector is returned"))
def check_intent_count_singular(context, n):
    assert len(context["intents"]) == n


@then(parsers.parse("exactly {n:d} IntentVector objects are returned"))
def check_intent_count(context, n):
    assert len(context["intents"]) == n


@then("the IntentVector embedding has 384 dimensions")
def check_embedding_dim(context):
    assert context["intents"][0].embedding.shape == (384,)


@then(parsers.parse("{n:d} or more IntentVector objects are returned"))
def check_min_intents(context, n):
    assert len(context["intents"]) >= n


@then(parsers.parse("at least {n:d} segments are produced"))
def check_min_segments(context, n):
    assert len(context["intents"]) >= n


@then(parsers.parse("1 IntentVector is returned with empty text and confidence {c:f}"))
def check_empty_intent(context, c):
    assert len(context["intents"]) == 1
    assert context["intents"][0].text == ""
    assert context["intents"][0].confidence == pytest.approx(c)


# ---------------------------------------------------------------------------
# Hierarchical intent -- Background & Given steps
# ---------------------------------------------------------------------------


@given(parsers.parse("a HierarchicalIntentExtractor with sub_intent_weight {w:f}"))
def hierarchical_extractor(context, w):
    context["extractor"] = HierarchicalIntentExtractor(sub_intent_weight=w)


@given(parsers.parse('a high-level intent about "{topic}"'))
def high_level_intent_about(context, topic):
    emb = _embed(topic)
    from ter_calculator.models import IntentVector

    context["high_level_intent"] = IntentVector(
        text=topic, embedding=emb, confidence=0.8, source_prompts=[topic]
    )


@given(parsers.parse('a sub-intent about "{topic}"'))
def sub_intent_about(context, topic):
    emb = _embed(topic)
    from ter_calculator.models import IntentVector

    context["sub_intent"] = IntentVector(
        text=topic, embedding=emb, confidence=0.7, source_prompts=[topic]
    )


@given(parsers.parse('a span about "{text}"'))
def span_about(context, text):
    context["span_embedding"] = _embed(text)


# ---------------------------------------------------------------------------
# Hierarchical intent -- When / Then
# ---------------------------------------------------------------------------


@when("hierarchical intent extraction runs")
def run_hierarchical(context):
    context["intents"] = context["extractor"].extract(context.get("prompts", []))


@when("the span is scored against the intents")
def score_span(context):
    extractor = context.get("extractor")
    if extractor is None:
        extractor = HierarchicalIntentExtractor(sub_intent_weight=0.7)
    intents = [context["high_level_intent"], context["sub_intent"]]
    score, best = extractor.score_span(context["span_embedding"], intents)
    context["blended_score"] = score
    context["best_intent"] = best


@then("the first IntentVector represents the high-level intent")
def check_high_level(context):
    assert len(context["intents"]) > 0
    assert context["intents"][0].text != ""


@then(parsers.parse("{n:d} additional sub-intent IntentVectors are returned"))
def check_sub_intents(context, n):
    assert len(context["intents"]) == n + 1


@then(
    parsers.parse(
        "the blended score uses {sub:d} percent sub-intent and {hi:d} percent high-level similarity"
    )
)
def check_blended_score(context, sub, hi):
    # Verify the blended score matches w * sub_sim + (1-w) * high_sim
    w = sub / 100.0
    high_sim = _cosine_similarity(
        context["span_embedding"], context["high_level_intent"].embedding
    )
    sub_sim = _cosine_similarity(
        context["span_embedding"], context["sub_intent"].embedding
    )
    expected = w * sub_sim + (1.0 - w) * high_sim
    assert context["blended_score"] == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Confidence scaling
# ---------------------------------------------------------------------------


@given(parsers.parse("a prompt with {n:d} words"))
def prompt_words(context, n):
    context["prompt"] = " ".join(["word"] * n)


@when("intent confidence is computed")
def compute_confidence(context):
    context["confidence"] = _prompt_confidence(context["prompt"])


@then(parsers.parse("confidence is {c:f}"))
def check_confidence(context, c):
    assert context["confidence"] == pytest.approx(c, abs=0.01)


# ---------------------------------------------------------------------------
# LLM intent -- Given steps
# ---------------------------------------------------------------------------


@given("an LLM intent extractor with a valid API key")
def llm_valid_key(context):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Mock the LLM extractor so the test can run without a real key.
        extractor = LLMIntentExtractor(api_key="sk-test-fake-key")
        mock_goal = StructuredGoal(
            primary_goal="Build a REST API for user management",
            sub_goals=["Support pagination", "Support filtering"],
            constraints=["Use Python"],
            expected_outputs=["api.py"],
        )
        extractor._summarise = MagicMock(return_value=mock_goal)
        context["extractor"] = extractor
        context["_llm_mocked"] = True
    else:
        context["extractor"] = LLMIntentExtractor(api_key=api_key)
        context["_llm_mocked"] = False


@given("an LLM intent extractor with no API key")
def llm_no_key(context):
    context["extractor"] = LLMIntentExtractor(api_key=None)


@given(parsers.parse('a StructuredGoal with primary_goal "{goal}"'))
def structured_goal(context, goal):
    context["goal"] = StructuredGoal(primary_goal=goal)


@given(parsers.parse('sub_goals "{sg1}" and "{sg2}"'))
def sub_goals(context, sg1, sg2):
    context["goal"].sub_goals = [sg1, sg2]


@given(parsers.parse('constraints "{c}"'))
def constraints(context, c):
    context["goal"].constraints = [c]


@given(parsers.parse('expected_outputs "{eo}"'))
def expected_outputs(context, eo):
    context["goal"].expected_outputs = [eo]


# ---------------------------------------------------------------------------
# LLM intent -- When / Then
# ---------------------------------------------------------------------------


@when("LLM intent extraction runs")
def run_llm(context):
    context["intents"] = context["extractor"].extract(context.get("prompts", []))


@then(
    "a StructuredGoal is produced with primary_goal, sub_goals, constraints, and expected_outputs"
)
def check_structured_goal(context):
    extractor = context["extractor"]
    goal = extractor.structured_goal
    if context.get("_llm_mocked"):
        # When mocked, _summarise was called which sets _last_goal via the real
        # extract path. Verify the intent text contains structured content.
        intent = context["intents"][0]
        assert intent.text != ""
        assert "|" in intent.text or "Build" in intent.text
    else:
        assert goal is not None
        assert goal.primary_goal != ""
        assert isinstance(goal.sub_goals, list)
        assert isinstance(goal.constraints, list)
        assert isinstance(goal.expected_outputs, list)


@then(parsers.parse("the IntentVector confidence is {c:f}"))
def check_intent_confidence(context, c):
    assert context["intents"][0].confidence == pytest.approx(c, abs=0.01)


@then("the fallback produces an IntentVector by direct embedding")
def check_fallback(context):
    assert len(context["intents"]) == 1
    assert context["intents"][0].embedding.shape == (384,)


@then("no error is raised")
def no_error():
    pass


@when("to_embedding_text is called")
def call_to_text(context):
    context["embedding_text"] = context["goal"].to_embedding_text()


@then("the output combines all fields separated by pipes")
def check_combined(context):
    text = context["embedding_text"]
    assert "|" in text
    goal = context["goal"]
    assert goal.primary_goal in text
    if goal.sub_goals:
        for sg in goal.sub_goals:
            assert sg in text
    if goal.constraints:
        for c in goal.constraints:
            assert c in text
    if goal.expected_outputs:
        for eo in goal.expected_outputs:
            assert eo in text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@when(parsers.parse('create_intent_extractor is called with strategy "{strategy}"'))
def call_factory(context, strategy):
    try:
        context["created_extractor"] = create_intent_extractor(strategy)
    except ValueError as e:
        context["error"] = e


@then("a SlidingIntentExtractor is returned")
def check_sliding(context):
    assert isinstance(context["created_extractor"], SlidingIntentExtractor)


@then("a HierarchicalIntentExtractor is returned")
def check_hierarchical_type(context):
    assert isinstance(context["created_extractor"], HierarchicalIntentExtractor)


@then("a LLMIntentExtractor is returned")
def check_llm_type(context):
    assert isinstance(context["created_extractor"], LLMIntentExtractor)


@then("a ValueError is raised")
def check_value_error(context):
    assert isinstance(context.get("error"), ValueError)
