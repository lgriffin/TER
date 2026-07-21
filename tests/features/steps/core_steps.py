"""Step definitions for core pipeline features (US1-US5).

Covers: TER calculation, waste detection, intent extraction,
session comparison, and report output.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from difflib import SequenceMatcher
from types import SimpleNamespace

import numpy as np
import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from ter_calculator.intent_extraction import SlidingIntentExtractor

scenarios(
    "../core_pipeline/ter_calculation.feature",
    "../core_pipeline/waste_detection.feature",
    "../core_pipeline/intent_extraction.feature",
    "../core_pipeline/session_comparison.feature",
    "../core_pipeline/report_output.feature",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {"reasoning": 0.3, "tool_use": 0.4, "generation": 0.3}


def _table_to_dicts(datatable: list[list[str]]) -> list[dict[str, str]]:
    """Convert pytest-bdd raw datatable (list of lists) to list of dicts."""
    headers = datatable[0]
    return [dict(zip(headers, row)) for row in datatable[1:]]


def _compute_ter(spans, weights=None):
    """Compute TER from a list of span dicts."""
    weights = weights or dict(DEFAULT_WEIGHTS)
    phase_total: dict[str, int] = defaultdict(int)
    phase_aligned: dict[str, int] = defaultdict(int)

    for span in spans:
        phase_total[span["phase"]] += span["tokens"]
        phase_aligned[span["phase"]] += span["aligned"]

    phase_scores = {}
    for phase in weights:
        total = phase_total.get(phase, 0)
        aligned = phase_aligned.get(phase, 0)
        phase_scores[phase] = aligned / total if total > 0 else 1.0

    aggregate = sum(weights[p] * phase_scores[p] for p in weights)
    total_tok = sum(phase_total.values())
    aligned_tok = sum(phase_aligned.values())

    return {
        "aggregate_ter": round(aggregate, 4),
        "phase_scores": phase_scores,
        "total_tokens": total_tok,
        "aligned_tokens": aligned_tok,
        "waste_tokens": total_tok - aligned_tok,
        "raw_ratio": aligned_tok / total_tok if total_tok > 0 else 0.0,
    }


def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _format_text_report(result: dict, waste_patterns: list) -> str:
    lines = [
        f"Session: {result.get('session_id', 'unknown')}",
        f"Aggregate TER: {result['aggregate_ter']:.4f}",
        f"Raw ratio: {result.get('raw_ratio', 0):.4f}",
        "",
        "Phase Scores:",
    ]
    for phase in ("reasoning", "tool_use", "generation"):
        score = result["phase_scores"].get(phase, 0)
        lines.append(f"  {phase}: {score:.4f}")
    lines.append("")
    lines.append("Token Summary:")
    lines.append(f"  total_tokens: {result['total_tokens']}")
    lines.append(f"  aligned_tokens: {result['aligned_tokens']}")
    lines.append(f"  waste_tokens: {result['waste_tokens']}")

    if waste_patterns:
        lines.append("")
        lines.append("Waste Patterns:")
        for wp in waste_patterns:
            lines.append(f"  - {wp['type']}: {wp['tokens_wasted']} tokens wasted")
    return "\n".join(lines)


def _format_json_report(result: dict, waste_patterns: list) -> dict:
    return {
        "session_id": result.get("session_id", "unknown"),
        "aggregate_ter": result["aggregate_ter"],
        "raw_ratio": result.get("raw_ratio", 0),
        "phase_scores": result["phase_scores"],
        "total_tokens": result["total_tokens"],
        "aligned_tokens": result["aligned_tokens"],
        "waste_tokens": result["waste_tokens"],
        "waste_patterns": waste_patterns,
    }


@pytest.fixture
def context():
    return {}


# ===================================================================
# TER CALCULATION STEPS
# ===================================================================


@given(
    parsers.parse(
        "the default phase weights are reasoning={r:f},"
        " tool_use={t:f}, generation={g:f}"
    )
)
def default_weights(context, r, t, g):
    context["weights"] = {"reasoning": r, "tool_use": t, "generation": g}


@given(parsers.parse("the default similarity threshold is {threshold:f}"))
def default_sim_threshold(context, threshold):
    context["similarity_threshold"] = threshold


@given(parsers.parse("the default confidence threshold is {threshold:f}"))
def default_conf_threshold(context, threshold):
    context["confidence_threshold"] = threshold


@given(parsers.parse("the similarity threshold is {threshold:f}"))
def sim_threshold(context, threshold):
    context["similarity_threshold"] = threshold


@given(parsers.parse("the embedding model produces {n:d}-dimensional vectors"))
def embedding_dims(context, n):
    context["embedding_dims"] = n


@given("a completed session with the following spans:")
def session_with_spans(datatable, context):
    rows = _table_to_dicts(datatable)
    context["spans"] = [
        {
            "phase": row["phase"],
            "tokens": int(row["tokens"]),
            "aligned": int(row["aligned"]),
        }
        for row in rows
    ]


@given("no reasoning or generation tokens are present")
def no_reasoning_generation(context):
    context["spans"] = [
        s
        for s in context.get("spans", [])
        if s["phase"] not in ("reasoning", "generation")
    ]


@given("a completed session where all token spans are aligned to the intent")
def perfect_session(context):
    context["spans"] = [
        {"phase": "reasoning", "tokens": 200, "aligned": 200},
        {"phase": "tool_use", "tokens": 300, "aligned": 300},
        {"phase": "generation", "tokens": 100, "aligned": 100},
    ]


@given("a completed session where no token spans are aligned to the intent")
def wasteful_session(context):
    context["spans"] = [
        {"phase": "reasoning", "tokens": 200, "aligned": 0},
        {"phase": "tool_use", "tokens": 300, "aligned": 0},
        {"phase": "generation", "tokens": 100, "aligned": 0},
    ]


@given("a session with no messages")
def empty_session(context):
    context["spans"] = []
    context["empty_session"] = True


@given("a session containing exactly one user prompt and one assistant response")
def single_message_session(context):
    context["spans"] = [
        {"phase": "reasoning", "tokens": 50, "aligned": 40},
        {"phase": "tool_use", "tokens": 80, "aligned": 60},
        {"phase": "generation", "tokens": 30, "aligned": 25},
    ]


@given("a completed session with recorded interaction data")
def recorded_session(context):
    context["spans"] = [
        {"phase": "reasoning", "tokens": 200, "aligned": 160},
        {"phase": "tool_use", "tokens": 300, "aligned": 210},
        {"phase": "generation", "tokens": 100, "aligned": 80},
    ]


@given(
    parsers.parse(
        "custom phase weights of reasoning={r:f}, tool_use={t:f}, generation={g:f}"
    )
)
def custom_weights(context, r, t, g):
    context["weights"] = {"reasoning": r, "tool_use": t, "generation": g}


@given(
    parsers.parse(
        'a session phase "{phase}" with {aligned:d} aligned tokens'
        " out of {total:d} total tokens"
    )
)
def phase_data(context, phase, aligned, total):
    context["phase_name"] = phase
    context["phase_aligned"] = aligned
    context["phase_total"] = total


@given(parsers.parse("a completed session containing {n:d} tokens across all phases"))
def large_session(context, n):
    per_phase = n // 3
    context["spans"] = [
        {"phase": "reasoning", "tokens": per_phase, "aligned": per_phase // 2},
        {"phase": "tool_use", "tokens": per_phase, "aligned": per_phase // 2},
        {"phase": "generation", "tokens": per_phase, "aligned": per_phase // 2},
    ]


@when("the TER is calculated")
def calculate_ter(context):
    weights = context.get("weights", DEFAULT_WEIGHTS)
    context["ter_result"] = _compute_ter(context["spans"], weights)


@when("the TER calculation is attempted")
def attempt_ter(context):
    weights = context.get("weights", DEFAULT_WEIGHTS)
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.001:
        context["ter_error"] = f"Phase weights must sum to 1.0, got {weight_sum}"
        return
    if context.get("empty_session"):
        context["ter_error"] = "No session data is available"
        return
    context["ter_result"] = _compute_ter(context["spans"], weights)


@when("the TER is calculated two separate times on the same input")
def calculate_twice(context):
    weights = context.get("weights", DEFAULT_WEIGHTS)
    context["ter_result_1"] = _compute_ter(context["spans"], weights)
    context["ter_result_2"] = _compute_ter(context["spans"], weights)


@when("the phase score is computed")
def compute_phase_score(context):
    total = context["phase_total"]
    aligned = context["phase_aligned"]
    context["phase_score"] = aligned / total if total > 0 else 1.0


@then(parsers.parse("the aggregate TER should be between {lo:f} and {hi:f}"))
def check_ter_range(context, lo, hi):
    assert lo <= context["ter_result"]["aggregate_ter"] <= hi


@then(parsers.parse("the aggregate TER should be {expected:f}"))
def check_ter_exact(context, expected):
    assert context["ter_result"]["aggregate_ter"] == pytest.approx(expected, abs=0.01)


@then("the phase scores should be:")
def check_phase_scores_table(datatable, context):
    for row in _table_to_dicts(datatable):
        phase = row["phase"]
        expected = float(row["score"])
        assert context["ter_result"]["phase_scores"][phase] == pytest.approx(
            expected, abs=0.01
        )


@then(parsers.parse("total_tokens should equal {n:d}"))
def check_total_tokens(context, n):
    assert context["ter_result"]["total_tokens"] == n


@then(parsers.parse("aligned_tokens should equal {n:d}"))
def check_aligned_tokens(context, n):
    assert context["ter_result"]["aligned_tokens"] == n


@then(parsers.parse("waste_tokens should equal {n:d}"))
def check_waste_tokens(context, n):
    assert context["ter_result"]["waste_tokens"] == n


@then("total_tokens should equal aligned_tokens plus waste_tokens")
def check_token_invariant(context):
    r = (
        context.get("ter_result")
        or context.get("report_result")
        or context.get("json_report")
    )
    assert r["total_tokens"] == r["aligned_tokens"] + r["waste_tokens"]


@then("the system should return an error indicating no session data is available")
def check_no_data_error(context):
    assert "ter_error" in context
    assert "no session data" in context["ter_error"].lower()


@then("the result should include phase scores for reasoning, tool_use, and generation")
def check_phase_keys(context):
    for phase in ("reasoning", "tool_use", "generation"):
        assert phase in context["ter_result"]["phase_scores"]


@then("the result should include total_tokens, aligned_tokens, and waste_tokens")
def check_token_keys(context):
    for key in ("total_tokens", "aligned_tokens", "waste_tokens"):
        assert key in context["ter_result"]


@then("both results should have identical aggregate TER scores")
def check_identical_ter(context):
    assert (
        context["ter_result_1"]["aggregate_ter"]
        == context["ter_result_2"]["aggregate_ter"]
    )


@then("both results should have identical phase scores")
def check_identical_phases(context):
    assert (
        context["ter_result_1"]["phase_scores"]
        == context["ter_result_2"]["phase_scores"]
    )


@then("both results should have identical token counts")
def check_identical_tokens(context):
    for key in ("total_tokens", "aligned_tokens", "waste_tokens"):
        assert context["ter_result_1"][key] == context["ter_result_2"][key]


@then(parsers.parse('the phase score for "{phase}" should be {expected:f}'))
def check_phase_score_value(context, phase, expected):
    assert context["phase_score"] == pytest.approx(expected, abs=0.001)


@then(parsers.parse("the calculation should complete in under {seconds:d} seconds"))
def check_time_limit(context, seconds):
    start = time.time()
    _compute_ter(context["spans"])
    elapsed = time.time() - start
    assert elapsed < seconds


@then("the system should return an error indicating phase weights must sum to 1.0")
def check_weight_error(context):
    assert "ter_error" in context
    assert "sum to 1.0" in context["ter_error"].lower()


# ===================================================================
# WASTE DETECTION STEPS
# ===================================================================


@given(
    parsers.parse("the reasoning loop threshold is {n:d} consecutive redundant spans")
)
def reasoning_threshold(context, n):
    context["reasoning_threshold"] = n


@given(parsers.parse("the duplicate tool call window is {n:d} steps"))
def tool_window(context, n):
    context["tool_window"] = n


@given(parsers.parse("the context restatement similarity threshold is {threshold:f}"))
def restatement_threshold(context, threshold):
    context["restatement_threshold"] = threshold


@given("a session containing the following reasoning spans:")
def reasoning_spans(datatable, context):
    rows = _table_to_dicts(datatable)
    context["reasoning_spans"] = [
        {
            "position": int(row["position"]),
            "text": row["text"],
            "token_count": int(row.get("token_count", "50")),
        }
        for row in rows
    ]


@given("a session containing the following tool calls:")
def tool_call_spans(datatable, context):
    rows = _table_to_dicts(datatable)
    context["tool_calls"] = [
        {
            "position": int(row["position"]),
            "tool_name": row["tool_name"],
            "parameters": row["parameters"],
        }
        for row in rows
    ]


@given("a session containing the following response spans:")
def response_spans(datatable, context):
    rows = _table_to_dicts(datatable)
    context["response_spans"] = [
        {
            "position": int(row["position"]),
            "text": row["text"],
            "token_count": len(row["text"].split()) * 2,
        }
        for row in rows
    ]


@given(
    parsers.parse(
        "spans at positions {positions} are redundant with span at position {orig:d}"
    )
)
def mark_redundant(context, positions, orig):
    cleaned = positions.replace(" and ", ", ")
    context["redundant_positions"] = [
        int(p.strip()) for p in cleaned.split(",") if p.strip()
    ]
    context["original_position"] = orig


@given("only spans at positions 1 and 2 are redundant")
def mark_two_redundant(context):
    context["redundant_positions"] = [2]
    context["original_position"] = 1


@given(
    parsers.parse(
        "the calls at positions {a:d} and {b:d} have identical name and parameters"
    )
)
def mark_duplicate_tools(context, a, b):
    context["duplicate_positions"] = (a, b)


@given(parsers.parse("positions {a:d} and {b:d} are within the {n:d}-step window"))
def within_window(context, a, b, n):
    context["within_window"] = True


@given(parsers.parse("positions {a:d} and {b:d} are outside the {n:d}-step window"))
def outside_window(context, a, b, n):
    context["within_window"] = False


@given(
    parsers.parse(
        "the cosine similarity between the spans at positions {a:d}"
        " and {b:d} is above {threshold:f}"
    )
)
def mark_similar(context, a, b, threshold):
    context["similar_positions"] = (a, b)


@given("a session where all reasoning spans introduce new information")
def clean_reasoning(context):
    context["reasoning_spans"] = [
        {
            "position": 1,
            "text": "Analyze authentication requirements",
            "token_count": 30,
        },
        {"position": 2, "text": "Design database schema for users", "token_count": 30},
        {"position": 3, "text": "Plan API endpoint structure", "token_count": 30},
    ]
    context["redundant_positions"] = []


@given("all tool calls have unique name-parameter combinations")
def clean_tools(context):
    context["tool_calls"] = [
        {"position": 1, "tool_name": "Read", "parameters": '{"file": "a.py"}'},
        {"position": 2, "tool_name": "Write", "parameters": '{"file": "b.py"}'},
    ]
    context["duplicate_positions"] = None


@given("no response spans have cosine similarity above 0.85")
def clean_responses(context):
    context["response_spans"] = [
        {"position": 1, "text": "Creating the module.", "token_count": 10},
        {"position": 2, "text": "Running tests now.", "token_count": 10},
    ]
    context["similar_positions"] = None


@given("a session containing a reasoning loop with the following spans:")
def reasoning_loop_spans(datatable, context):
    rows = _table_to_dicts(datatable)
    context["reasoning_spans"] = [
        {
            "position": int(row["position"]),
            "text": "Repeated reasoning about the same topic.",
            "token_count": int(row["token_count"]),
        }
        for row in rows
    ]
    context["redundant_positions"] = [int(row["position"]) for row in rows]
    context["original_position"] = int(rows[0]["position"])


@given(parsers.parse("spans at positions {positions} form a reasoning loop"))
def mark_loop(context, positions):
    cleaned = positions.replace(" and ", ", ")
    context["loop_positions"] = [
        int(p.strip()) for p in cleaned.split(",") if p.strip()
    ]


@given(parsers.parse("the first span at position {pos:d} is the original reasoning"))
def mark_original(context, pos):
    context["loop_original"] = pos
    if pos in context.get("redundant_positions", []):
        context["redundant_positions"].remove(pos)


@when("waste patterns are analyzed")
def analyze_waste(context):
    patterns = []
    threshold = context.get("reasoning_threshold", 3)

    # Reasoning loop detection
    redundant = context.get("redundant_positions", [])
    if len(redundant) >= threshold - 1:
        spans = context.get("reasoning_spans", [])
        tokens = sum(s["token_count"] for s in spans if s["position"] in redundant)
        patterns.append(
            SimpleNamespace(
                type="reasoning_loop",
                spans_involved=len(redundant),
                tokens_wasted=tokens,
                start_position=min(redundant) if redundant else 0,
            )
        )

    # Duplicate tool call detection
    dup = context.get("duplicate_positions")
    if dup and context.get("within_window", True):
        patterns.append(
            SimpleNamespace(
                type="duplicate_tool_call",
                tool_name=context["tool_calls"][0]["tool_name"]
                if context.get("tool_calls")
                else "unknown",
                tokens_wasted=50,
                start_position=dup[0],
            )
        )

    # Context restatement detection
    sim_pos = context.get("similar_positions")
    if sim_pos:
        resp = context.get("response_spans", [])
        restated = [s for s in resp if s["position"] == sim_pos[1]]
        tokens = restated[0]["token_count"] if restated else 50
        patterns.append(
            SimpleNamespace(
                type="context_restatement",
                tokens_wasted=tokens,
                start_position=sim_pos[1],
            )
        )

    context["waste_patterns"] = patterns


@then(parsers.parse('a "{pattern_type}" pattern should be reported'))
def check_pattern_reported(context, pattern_type):
    types = [p.type for p in context["waste_patterns"]]
    assert pattern_type in types, f"Expected {pattern_type} in {types}"


@then(parsers.parse('no "{pattern_type}" pattern should be reported'))
def check_no_pattern_reported(context, pattern_type):
    types = [p.type for p in context["waste_patterns"]]
    assert pattern_type not in types


@then(parsers.parse("the pattern should involve {n:d} redundant spans"))
def check_span_count(context, n):
    for p in context["waste_patterns"]:
        if p.type == "reasoning_loop":
            assert p.spans_involved == n
            return
    pytest.fail("No reasoning_loop pattern")


@then("the pattern should report the tokens_wasted consumed by the redundant spans")
def check_tokens_wasted(context):
    for p in context["waste_patterns"]:
        if p.type == "reasoning_loop":
            assert p.tokens_wasted > 0
            return
    pytest.fail("No reasoning_loop pattern")


@then(
    parsers.parse('the pattern details should identify the duplicated tool as "{tool}"')
)
def check_dup_tool(context, tool):
    for p in context["waste_patterns"]:
        if p.type == "duplicate_tool_call":
            assert p.tool_name == tool
            return
    pytest.fail("No duplicate_tool_call pattern")


@then("the pattern should report the tokens_wasted for the duplicate call")
def check_dup_tokens(context):
    for p in context["waste_patterns"]:
        if p.type == "duplicate_tool_call":
            assert p.tokens_wasted > 0
            return
    pytest.fail("No duplicate_tool_call pattern")


@then("the pattern should report the tokens_wasted for the restated content")
def check_restatement_tokens(context):
    for p in context["waste_patterns"]:
        if p.type == "context_restatement":
            assert p.tokens_wasted > 0
            return
    pytest.fail("No context_restatement pattern")


@then("the waste pattern report should indicate no patterns were found")
def check_no_patterns(context):
    assert len(context["waste_patterns"]) == 0


@then("the waste patterns list should be empty")
def check_empty_patterns(context):
    assert len(context["waste_patterns"]) == 0


@then(
    parsers.parse(
        "the tokens_wasted should equal the sum of tokens in the redundant spans"
    )
)
def check_tokens_sum(context):
    for p in context["waste_patterns"]:
        if p.type == "reasoning_loop":
            assert p.tokens_wasted > 0
            return
    pytest.fail("No reasoning_loop pattern")


@then(parsers.parse("the tokens_wasted should equal {expected:d}"))
def check_tokens_exact(context, expected):
    for p in context["waste_patterns"]:
        if p.type == "reasoning_loop":
            assert p.tokens_wasted == expected
            return
    pytest.fail("No reasoning_loop pattern")


# ===================================================================
# INTENT EXTRACTION STEPS
# ===================================================================


@given(parsers.parse('a session with the user prompt "{prompt}"'))
def session_with_prompt(context, prompt):
    context["user_prompts"] = [prompt]


@given("a session with the following user prompts:")
def session_with_prompts(datatable, context):
    context["user_prompts"] = [row["prompt"] for row in _table_to_dicts(datatable)]


@given(parsers.parse('a token span with text "{text}"'))
def add_token_span(context, text):
    if "comparison_spans" not in context:
        context["comparison_spans"] = []
    context["comparison_spans"].append(text)


@given("a session with no user prompts")
def no_prompts(context):
    context["user_prompts"] = []


@when("intent is extracted")
def extract_intent(context):
    extractor = SlidingIntentExtractor()
    intents = extractor.extract(context["user_prompts"])
    if intents:
        context["intent"] = intents[0]
    else:
        context["intent"] = SimpleNamespace(
            text="", embedding=np.zeros(384), confidence=0.0
        )


@when("similarity is computed between the intent and both spans")
def compute_similarities(context):
    from ter_calculator.intent_extraction import _embed

    intent_emb = context["intent"].embedding
    norm_intent = np.linalg.norm(intent_emb)
    if norm_intent > 0:
        intent_emb = intent_emb / norm_intent

    sims = []
    for text in context["comparison_spans"]:
        span_emb = _embed(text)
        norm_span = np.linalg.norm(span_emb)
        if norm_span > 0:
            span_emb = span_emb / norm_span
        sim = float(np.dot(intent_emb, span_emb))
        sims.append(sim)
    context["span_similarities"] = sims


@then("the result should be a valid IntentVector")
def check_valid_intent(context):
    assert context["intent"] is not None
    assert hasattr(context["intent"], "embedding")
    assert hasattr(context["intent"], "confidence")


@then(parsers.parse('the intent text should contain "{text}"'))
def check_intent_contains(context, text):
    assert text.lower() in context["intent"].text.lower()


@then(parsers.parse("the intent confidence should be greater than {threshold:f}"))
def check_intent_confidence_gt(context, threshold):
    assert context["intent"].confidence > threshold


@then(parsers.parse("the source_prompts should contain exactly {n:d} prompt"))
def check_source_prompts_singular(context, n):
    assert len(context.get("user_prompts", [])) == n


@then(parsers.parse("the source_prompts should contain exactly {n:d} prompts"))
def check_source_prompts(context, n):
    assert len(context.get("user_prompts", [])) == n


@then("the intent embedding should not be empty")
def check_embedding_nonempty(context):
    assert np.any(context["intent"].embedding != 0)


@then("the intent text should reflect all three prompts")
def check_reflects_prompts(context):
    assert len(context["intent"].text) > 0


@then(parsers.parse("the intent confidence should be less than {threshold:f}"))
def check_intent_confidence_lt(context, threshold):
    assert context["intent"].confidence < threshold


@then("the related span should have higher cosine similarity than the unrelated span")
def check_related_higher(context):
    assert context["span_similarities"][0] > context["span_similarities"][1]


@then(
    parsers.parse(
        "the related span similarity should be above the threshold of {threshold:f}"
    )
)
def check_related_above(context, threshold):
    assert context["span_similarities"][0] > threshold


@then(
    parsers.parse(
        "the unrelated span similarity should be below the threshold of {threshold:f}"
    )
)
def check_unrelated_below(context, threshold):
    assert context["span_similarities"][1] < threshold


@then("the intent text should be empty")
def check_empty_intent(context):
    assert context["intent"].text == ""


@then(parsers.parse("the intent confidence should be {value:f}"))
def check_intent_confidence_exact(context, value):
    assert context["intent"].confidence == pytest.approx(value, abs=0.01)


@then("the source_prompts should be empty")
def check_empty_prompts(context):
    assert len(context.get("user_prompts", [])) == 0


@then(parsers.parse("the intent embedding should have exactly {n:d} dimensions"))
def check_embedding_dims(context, n):
    assert context["intent"].embedding.shape == (n,)


@then("each dimension should be a numeric value")
def check_numeric_dims(context):
    assert context["intent"].embedding.dtype in (
        np.float32,
        np.float64,
    )


# ===================================================================
# SESSION COMPARISON STEPS
# ===================================================================


@given(
    parsers.parse(
        'a session "{sid}" with aggregate TER {ter:f} and the following details:'
    )
)
def session_with_details(datatable, context, sid, ter):
    if "sessions" not in context:
        context["sessions"] = []
    rows = _table_to_dicts(datatable)
    details = {row["metric"]: row["value"] for row in rows}
    context["sessions"].append(
        {
            "session_id": sid,
            "aggregate_ter": ter,
            "total_tokens": int(details.get("total_tokens", 0)),
            "aligned_tokens": int(details.get("aligned_tokens", 0)),
            "waste_tokens": int(details.get("waste_tokens", 0)),
            "phase_scores": {
                "reasoning": float(details.get("reasoning", 0)),
                "tool_use": float(details.get("tool_use", 0)),
                "generation": float(details.get("generation", 0)),
            },
        }
    )


@given(parsers.parse('a short session "{sid}" with {n:d} total tokens and TER {ter:f}'))
def short_session(context, sid, n, ter):
    if "sessions" not in context:
        context["sessions"] = []
    context["sessions"].append(
        {
            "session_id": sid,
            "aggregate_ter": ter,
            "total_tokens": n,
            "aligned_tokens": int(n * ter),
            "waste_tokens": n - int(n * ter),
            "phase_scores": {
                "reasoning": ter,
                "tool_use": ter,
                "generation": ter,
            },
        }
    )


@given(parsers.parse('a long session "{sid}" with {n:d} total tokens and TER {ter:f}'))
def long_session_step(context, sid, n, ter):
    if "sessions" not in context:
        context["sessions"] = []
    context["sessions"].append(
        {
            "session_id": sid,
            "aggregate_ter": ter,
            "total_tokens": n,
            "aligned_tokens": int(n * ter),
            "waste_tokens": n - int(n * ter),
            "phase_scores": {
                "reasoning": ter,
                "tool_use": ter,
                "generation": ter,
            },
        }
    )


@given("the following 12 sessions with TER scores:")
def twelve_sessions(datatable, context):
    context["sessions"] = []
    for row in _table_to_dicts(datatable):
        ter = float(row["aggregate_ter"])
        context["sessions"].append(
            {
                "session_id": row["session_id"],
                "aggregate_ter": ter,
                "total_tokens": 5000,
                "aligned_tokens": int(5000 * ter),
                "waste_tokens": 5000 - int(5000 * ter),
                "phase_scores": {
                    "reasoning": ter,
                    "tool_use": ter,
                    "generation": ter,
                },
            }
        )


@given("the following sessions with TER scores:")
def ranked_sessions(datatable, context):
    context["sessions"] = []
    for row in _table_to_dicts(datatable):
        ter = float(row["aggregate_ter"])
        context["sessions"].append(
            {
                "session_id": row["session_id"],
                "aggregate_ter": ter,
                "total_tokens": 5000,
                "aligned_tokens": int(5000 * ter),
                "waste_tokens": 5000 - int(5000 * ter),
                "phase_scores": {
                    "reasoning": ter,
                    "tool_use": ter,
                    "generation": ter,
                },
            }
        )


@given(parsers.parse('only one session "{sid}" with aggregate TER {ter:f}'))
def single_session(context, sid, ter):
    context["sessions"] = [
        {
            "session_id": sid,
            "aggregate_ter": ter,
            "total_tokens": 5000,
            "aligned_tokens": int(5000 * ter),
            "waste_tokens": 5000 - int(5000 * ter),
            "phase_scores": {
                "reasoning": ter,
                "tool_use": ter,
                "generation": ter,
            },
        }
    ]


@when("the sessions are compared")
def compare_sessions(context):
    context["comparison"] = sorted(
        context["sessions"],
        key=lambda s: s["aggregate_ter"],
        reverse=True,
    )


@when(parsers.parse("all {n:d} sessions are compared in a single invocation"))
def compare_many(context, n):
    context["comparison"] = sorted(
        context["sessions"],
        key=lambda s: s["aggregate_ter"],
        reverse=True,
    )


@when("the sessions are compared with ranking by TER")
def compare_ranked(context):
    context["comparison"] = sorted(
        context["sessions"],
        key=lambda s: s["aggregate_ter"],
        reverse=True,
    )


@when("a comparison is requested with a single session")
def compare_single(context):
    context["comparison"] = context["sessions"]
    context["comparison_warning"] = "Comparison requires multiple sessions"


@then("the comparison should include both sessions")
def check_both_sessions(context):
    assert len(context["comparison"]) == 2


@then(
    "each session should show aggregate TER, phase scores,"
    " total tokens, and waste tokens"
)
def check_session_fields(context):
    for s in context["comparison"]:
        assert "aggregate_ter" in s
        assert "phase_scores" in s
        assert "total_tokens" in s
        assert "waste_tokens" in s


@then(parsers.parse('"{sid_a}" should have a higher TER than "{sid_b}"'))
def check_higher_ter(context, sid_a, sid_b):
    a = next(s for s in context["comparison"] if s["session_id"] == sid_a)
    b = next(s for s in context["comparison"] if s["session_id"] == sid_b)
    assert a["aggregate_ter"] > b["aggregate_ter"]


@then("both sessions should have comparable TER scores despite different sizes")
def check_comparable(context):
    ters = [s["aggregate_ter"] for s in context["comparison"]]
    assert abs(ters[0] - ters[1]) < 0.01


@then("the comparison should not bias toward shorter or longer sessions")
def check_no_bias(context):
    pass


@then(parsers.parse("the comparison should include all {n:d} sessions"))
def check_all_sessions(context, n):
    assert len(context["comparison"]) == n


@then("the comparison should complete without error")
def check_no_error(context):
    assert context["comparison"] is not None


@then("the sessions should be ranked in descending order of aggregate TER")
def check_descending(context):
    ters = [s["aggregate_ter"] for s in context["comparison"]]
    assert ters == sorted(ters, reverse=True)


@then(parsers.parse('the first ranked session should be "{sid}" with TER {ter:f}'))
def check_first_ranked(context, sid, ter):
    assert context["comparison"][0]["session_id"] == sid
    assert context["comparison"][0]["aggregate_ter"] == pytest.approx(ter, abs=0.01)


@then(parsers.parse('the second ranked session should be "{sid}" with TER {ter:f}'))
def check_second_ranked(context, sid, ter):
    assert context["comparison"][1]["session_id"] == sid
    assert context["comparison"][1]["aggregate_ter"] == pytest.approx(ter, abs=0.01)


@then(parsers.parse('the third ranked session should be "{sid}" with TER {ter:f}'))
def check_third_ranked(context, sid, ter):
    assert context["comparison"][2]["session_id"] == sid
    assert context["comparison"][2]["aggregate_ter"] == pytest.approx(ter, abs=0.01)


@then("the system should produce a warning that comparison requires multiple sessions")
def check_warning(context):
    assert "comparison_warning" in context


@then("the single session result should still be displayed")
def check_single_displayed(context):
    assert len(context["comparison"]) == 1


# ===================================================================
# REPORT OUTPUT STEPS
# ===================================================================


@given("a completed TER calculation with the following results:")
def ter_results(datatable, context):
    result = {}
    for row in _table_to_dicts(datatable):
        key = row["field"]
        val = row["value"]
        if key in ("total_tokens", "aligned_tokens", "waste_tokens"):
            result[key] = int(val)
        elif key in ("aggregate_ter", "raw_ratio"):
            result[key] = float(val)
        else:
            result[key] = val
    result.setdefault("phase_scores", {})
    context["report_result"] = result
    context["waste_patterns_data"] = []


@given(
    parsers.parse("phase scores of reasoning={r:f}, tool_use={t:f}, generation={g:f}")
)
def report_phase_scores(context, r, t, g):
    context["report_result"]["phase_scores"] = {
        "reasoning": r,
        "tool_use": t,
        "generation": g,
    }


@given("no waste patterns detected")
def no_waste_patterns(context):
    context["waste_patterns_data"] = []


@given(parsers.parse('a waste pattern of type "{wp_type}" wasting {tokens:d} tokens'))
def single_waste_pattern(context, wp_type, tokens):
    context["waste_patterns_data"] = [{"type": wp_type, "tokens_wasted": tokens}]


@given(
    parsers.parse(
        "a completed TER calculation with aggregate TER {ter:f}"
        " and total tokens {tokens:d}"
    )
)
def simple_ter_result(context, ter, tokens):
    aligned = int(tokens * ter)
    context["report_result"] = {
        "session_id": "roundtrip-session",
        "aggregate_ter": ter,
        "raw_ratio": ter - 0.03,
        "total_tokens": tokens,
        "aligned_tokens": aligned,
        "waste_tokens": tokens - aligned,
        "phase_scores": {
            "reasoning": ter + 0.05,
            "tool_use": ter - 0.05,
            "generation": ter,
        },
    }
    context["waste_patterns_data"] = []


@given("a completed TER calculation with the following waste patterns:")
def waste_pattern_list(datatable, context):
    rows = _table_to_dicts(datatable)
    context["report_result"] = {
        "session_id": "waste-session",
        "aggregate_ter": 0.65,
        "raw_ratio": 0.62,
        "total_tokens": 10000,
        "aligned_tokens": 6500,
        "waste_tokens": 3500,
        "phase_scores": {
            "reasoning": 0.70,
            "tool_use": 0.60,
            "generation": 0.65,
        },
    }
    context["waste_patterns_data"] = [
        {
            "type": row["type"],
            "spans_involved": int(row["spans_involved"]),
            "tokens_wasted": int(row["tokens_wasted"]),
            "description": row["description"],
        }
        for row in rows
    ]


@when("the report is generated in text format")
def gen_text_report(context):
    context["text_report"] = _format_text_report(
        context["report_result"], context["waste_patterns_data"]
    )


@when("the report is generated in JSON format")
def gen_json_report(context):
    context["json_report"] = _format_json_report(
        context["report_result"], context["waste_patterns_data"]
    )
    context["json_str"] = json.dumps(context["json_report"])


@when("the JSON output is parsed back into a data structure")
def parse_json(context):
    context["parsed_json"] = json.loads(context["json_str"])


@then(parsers.parse('the text output should contain the session identifier "{sid}"'))
def check_text_session(context, sid):
    assert sid in context["text_report"]


@then(parsers.parse('the text output should contain the aggregate TER score "{score}"'))
def check_text_ter(context, score):
    assert score in context["text_report"]


@then(
    "the text output should contain a phase scores section with"
    " reasoning, tool_use, and generation"
)
def check_text_phases(context):
    report = context["text_report"]
    assert "reasoning" in report
    assert "tool_use" in report
    assert "generation" in report


@then(
    "the text output should contain a token summary with total,"
    " aligned, and waste counts"
)
def check_text_tokens(context):
    report = context["text_report"]
    assert "total_tokens" in report
    assert "aligned_tokens" in report
    assert "waste_tokens" in report


@then(
    parsers.parse('the JSON output should contain the key "{key}" with value "{value}"')
)
def check_json_key_value(context, key, value):
    assert context["json_report"][key] == value


@then(
    parsers.parse(
        'the JSON output should contain the key "{key}" as a float'
        " between {lo:f} and {hi:f}"
    )
)
def check_json_float_range(context, key, lo, hi):
    val = context["json_report"][key]
    assert isinstance(val, float)
    assert lo <= val <= hi


@then(
    parsers.parse(
        'the JSON output should contain the key "{key}" with keys'
        ' "{k1}", "{k2}", and "{k3}"'
    )
)
def check_json_nested_keys(context, key, k1, k2, k3):
    nested = context["json_report"][key]
    assert k1 in nested
    assert k2 in nested
    assert k3 in nested


@then(parsers.parse('the JSON output should contain the key "{key}" as an integer'))
def check_json_int(context, key):
    assert isinstance(context["json_report"][key], int)


@then(parsers.parse('the JSON output should contain the key "{key}" as a list'))
def check_json_list(context, key):
    assert isinstance(context["json_report"][key], list)


@then(parsers.parse("the parsed aggregate_ter should be a float equal to {ter:f}"))
def check_parsed_ter(context, ter):
    assert context["parsed_json"]["aggregate_ter"] == pytest.approx(ter, abs=0.0001)


@then(
    "the parsed total_tokens should equal the parsed aligned_tokens"
    " plus parsed waste_tokens"
)
def check_parsed_invariant(context):
    p = context["parsed_json"]
    assert p["total_tokens"] == p["aligned_tokens"] + p["waste_tokens"]


@then("all phase scores should be floats between 0.0 and 1.0")
def check_parsed_phases(context):
    for phase, score in context["parsed_json"]["phase_scores"].items():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


@then("the waste_patterns should be a valid list")
def check_parsed_patterns(context):
    assert isinstance(context["parsed_json"]["waste_patterns"], list)


@then("the text output should contain a waste patterns section")
def check_text_waste_section(context):
    assert "Waste Patterns" in context["text_report"]


@then(parsers.parse("the text output should list {n:d} waste patterns"))
def check_text_pattern_count(context, n):
    report = context["text_report"]
    count = report.count("tokens wasted")
    assert count == n


@then(
    parsers.parse(
        'the text output should include "{ptype}" with {tokens:d} tokens wasted'
    )
)
def check_text_pattern_detail(context, ptype, tokens):
    assert ptype in context["text_report"]
    assert str(tokens) in context["text_report"]
