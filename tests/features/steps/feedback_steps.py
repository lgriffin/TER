"""Step definitions for feedback loop features."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from ter_calculator.feedback import (
    CheckResult,
    PromptHint,
    TERHistory,
    TERHistoryEntry,
    TrendDirection,
    TrendSummary,
    check_threshold,
    generate_prompt_hints,
    get_stats_by_tag,
    tag_session,
)

scenarios(
    "../feedback_ci/prompt_hints.feature",
    "../feedback_ci/history_trending.feature",
    "../feedback_ci/session_tagging.feature",
    "../feedback_ci/ci_threshold.feature",
)


def _make_ter_result(**kwargs):
    """Build a mock TERResult-compatible object."""
    defaults = {
        "session_id": "test-session",
        "aggregate_ter": 0.0,
        "total_tokens": 1000,
        "waste_tokens": 200,
        "aligned_tokens": 800,
        "phase_scores": {"reasoning": 0.8, "tool_use": 0.8, "generation": 0.8},
        "waste_patterns": [],
        "raw_ratio": 0.0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_waste_pattern(pattern_type, tokens_wasted=100):
    return SimpleNamespace(pattern_type=pattern_type, tokens_wasted=tokens_wasted)


@pytest.fixture
def context():
    return {}


# ---------------------------------------------------------------------------
# Prompt hints
# ---------------------------------------------------------------------------


@given("a TER result with reasoning phase score below 0.5")
def low_reasoning(context):
    context["result"] = _make_ter_result(
        phase_scores={"reasoning": 0.35, "tool_use": 0.8, "generation": 0.8},
        aggregate_ter=0.65,
    )


@given(parsers.parse('a TER result with a "{pattern_type}" waste pattern'))
def result_with_pattern(context, pattern_type):
    context["result"] = _make_ter_result(
        waste_patterns=[_make_waste_pattern(pattern_type)],
        aggregate_ter=0.60,
        phase_scores={"reasoning": 0.7, "tool_use": 0.7, "generation": 0.7},
    )


@given("a TER result with aggregate_ter above 0.9 and no waste patterns")
def high_ter_result(context):
    context["result"] = _make_ter_result(
        aggregate_ter=0.95,
        phase_scores={"reasoning": 0.95, "tool_use": 0.95, "generation": 0.95},
        waste_patterns=[],
    )


@given("a TER result with multiple waste patterns")
def multiple_patterns(context):
    context["result"] = _make_ter_result(
        aggregate_ter=0.50,
        phase_scores={"reasoning": 0.4, "tool_use": 0.5, "generation": 0.6},
        waste_patterns=[
            _make_waste_pattern("reasoning_loop"),
            _make_waste_pattern("duplicate_tool_call"),
        ],
    )


@when("prompt hints are generated")
def gen_hints(context):
    context["hints"] = generate_prompt_hints(context["result"])


@then("at least one hint has category related to reasoning")
def check_reasoning_hint(context):
    categories = [h.category for h in context["hints"]]
    assert "reasoning" in categories


@then("at least one hint references the reasoning loop pattern")
def check_loop_hint(context):
    pattern_types = [
        h.related_pattern_type for h in context["hints"] if h.related_pattern_type
    ]
    assert "reasoning_loop" in pattern_types


@then("the hint includes an estimated_impact")
def check_impact(context):
    for h in context["hints"]:
        if h.related_pattern_type == "reasoning_loop":
            assert h.estimated_impact in ("high", "medium", "low")
            return
    pytest.fail("No reasoning_loop hint found")


@then("an empty list of hints is returned")
def check_empty_hints(context):
    assert context["hints"] == []


@then("each hint has category, suggestion, estimated_impact, and related_pattern_type")
def check_hint_fields(context):
    assert len(context["hints"]) > 0
    for h in context["hints"]:
        assert h.category
        assert h.suggestion
        assert h.estimated_impact in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# History trending
# ---------------------------------------------------------------------------


@given("a temporary TER history file")
def temp_history(tmp_path, context):
    context["history_path"] = tmp_path / "history.json"
    context["history"] = TERHistory(path=context["history_path"])


@given(parsers.parse('a TER result with session_id "{sid}" and aggregate_ter {ter:f}'))
def result_for_history(context, sid, ter):
    context["result"] = _make_ter_result(session_id=sid, aggregate_ter=ter)


@when("the result is recorded to history")
def record_result(context):
    context["history"].record(context["result"], project_path="/app")


@then(parsers.parse('the history file contains an entry for "{sid}"'))
def check_history_entry(context, sid):
    entries = context["history"].get_trend()
    session_ids = [e.session_id for e in entries]
    assert sid in session_ids


@given(parsers.parse('{n:d} recorded TER results for project "{project}"'))
def record_n_results(context, n, project):
    for i in range(n):
        result = _make_ter_result(
            session_id=f"session-{i}",
            aggregate_ter=0.70 + i * 0.01,
        )
        context["history"].record(result, project_path=project)


@when(parsers.parse('get_trend is called for project "{project}"'))
def call_get_trend(context, project):
    context["trend"] = context["history"].get_trend(project_path=project)


@then(parsers.parse("a list of {n:d} TERHistoryEntry objects is returned"))
def check_trend_count(context, n):
    assert len(context["trend"]) == n


@given(
    parsers.parse(
        "10 recorded results where the first 5 average TER {first:f}"
        " and the last 5 average {last:f}"
    )
)
def record_split_results(context, first, last):
    for i in range(5):
        result = _make_ter_result(session_id=f"s-{i}", aggregate_ter=first)
        context["history"].record(result, project_path="/app")
    for i in range(5, 10):
        result = _make_ter_result(session_id=f"s-{i}", aggregate_ter=last)
        context["history"].record(result, project_path="/app")


@when("get_summary is called")
def call_get_summary(context):
    context["summary"] = context["history"].get_summary()


@then(parsers.parse('trend_direction is "{direction}"'))
def check_trend_direction(context, direction):
    assert context["summary"].trend_direction.value == direction


@given(parsers.parse("recorded results with TERs {ters}"))
def record_specific_ters(context, ters):
    for i, ter_str in enumerate(ters.split(", ")):
        result = _make_ter_result(session_id=f"s-{i}", aggregate_ter=float(ter_str))
        context["history"].record(result, project_path="/app")


@then(parsers.parse("best_ter is {best:f} and worst_ter is {worst:f}"))
def check_best_worst(context, best, worst):
    assert context["summary"].best_ter == pytest.approx(best, abs=0.01)
    assert context["summary"].worst_ter == pytest.approx(worst, abs=0.01)


# ---------------------------------------------------------------------------
# Session tagging
# ---------------------------------------------------------------------------


@given("a temporary TER history file with recorded sessions")
def history_with_sessions(tmp_path, context):
    context["history_path"] = tmp_path / "history.json"
    context["history"] = TERHistory(path=context["history_path"])
    for i in range(3):
        result = _make_ter_result(
            session_id=f"session-{i + 1}",
            aggregate_ter=0.70 + i * 0.05,
        )
        context["history"].record(result, project_path="/app")


@given(parsers.parse('a recorded session with session_id "{sid}"'))
def recorded_session(context, sid):
    context["tag_session_id"] = sid


@when(parsers.parse('tag_session is called with tags "{t1}" and "{t2}"'))
def call_tag_session(context, t1, t2):
    tag_session(
        context["tag_session_id"],
        [t1, t2],
        history_path=context["history_path"],
    )


@then(parsers.parse('session "{sid}" has tags "{t1}" and "{t2}"'))
def check_tags(context, sid, t1, t2):
    entries = context["history"].get_trend()
    for e in entries:
        if e.session_id == sid:
            assert t1 in e.tags
            assert t2 in e.tags
            return
    pytest.fail(f"Session {sid} not found")


@given(parsers.parse('a session already tagged with "{tag}"'))
def pre_tagged_session(context, tag):
    context["tag_session_id"] = "session-1"
    tag_session("session-1", [tag], history_path=context["history_path"])


@then(parsers.parse('the session has tags "{t1}" and "{t2}" without duplicates'))
def check_deduped_tags(context, t1, t2):
    entries = context["history"].get_trend()
    for e in entries:
        if e.session_id == context["tag_session_id"]:
            assert t1 in e.tags
            assert t2 in e.tags
            assert len(e.tags) == len(set(e.tags))
            return
    pytest.fail("Session not found")


@given(parsers.parse('{n:d} sessions tagged "{tag}" with TERs {ters}'))
def tagged_sessions(context, n, tag, ters):
    cleaned = ters.replace(" and ", ", ")
    ter_list = [float(t.strip()) for t in cleaned.split(",") if t.strip()]
    for i, ter in enumerate(ter_list):
        sid = f"tagged-{tag}-{i}"
        result = _make_ter_result(session_id=sid, aggregate_ter=ter)
        context["history"].record(result, project_path="/app")
        tag_session(sid, [tag], history_path=context["history_path"])


@when(parsers.parse('get_stats_by_tag is called for "{tag}"'))
def call_stats_by_tag(context, tag):
    context["tag_stats"] = get_stats_by_tag(tag, history_path=context["history_path"])


@then(parsers.parse("the result includes session_count of {n:d}"))
def check_tag_count(context, n):
    assert context["tag_stats"].session_count == n


@then(parsers.parse("the average TER is approximately {avg:f}"))
def check_avg_ter(context, avg):
    assert context["tag_stats"].avg_ter == pytest.approx(avg, abs=0.02)


# ---------------------------------------------------------------------------
# CI threshold
# ---------------------------------------------------------------------------


@given(parsers.parse("a TER result with aggregate_ter {ter:f}"))
def ci_result(context, ter):
    context["result"] = _make_ter_result(
        aggregate_ter=ter,
        phase_scores={
            "reasoning": min(1.0, ter + 0.05),
            "tool_use": ter,
            "generation": min(1.0, ter + 0.05),
        },
    )


@given(
    parsers.parse(
        "a TER result with reasoning score {r:f}, tool_use score {t:f},"
        " and generation score {g:f}"
    )
)
def ci_result_phases(context, r, t, g):
    agg = r * 0.3 + t * 0.4 + g * 0.3
    context["result"] = _make_ter_result(
        aggregate_ter=agg,
        phase_scores={"reasoning": r, "tool_use": t, "generation": g},
    )


@when(parsers.parse("check_threshold is called with threshold {threshold:f}"))
def call_check(context, threshold):
    context["check"] = check_threshold(context["result"], threshold)


@when(
    parsers.parse(
        "check_threshold is called with aggregate threshold {agg:f}"
        " and phase threshold {phase:f}"
    )
)
def call_check_phase(context, agg, phase):
    context["check"] = check_threshold(context["result"], agg, phase_threshold=phase)


@then("the check passes")
def check_passes(context):
    assert context["check"].passed is True


@then("the check fails")
def check_fails(context):
    assert context["check"].passed is False


@then("the result message indicates the TER exceeds the threshold")
def check_pass_message(context):
    assert "passed" in context["check"].message.lower()


@then("the result message indicates the TER is below the threshold")
def check_fail_message(context):
    assert "failed" in context["check"].message.lower()


@then(
    parsers.parse(
        "the check fails because tool_use score {score:f}"
        " is below the phase threshold {threshold:f}"
    )
)
def check_phase_fail(context, score, threshold):
    assert context["check"].passed is False


@then(parsers.parse('phase_failures includes "{phase}"'))
def check_phase_failures(context, phase):
    assert phase in context["check"].phase_failures
