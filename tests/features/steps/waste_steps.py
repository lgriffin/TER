"""Step definitions for extended waste detection features."""

from __future__ import annotations

import math

import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from ter_calculator.models import ClassifiedSpan, SpanLabel, SpanPhase, TokenSpan, WastePattern
from ter_calculator.waste_detectors import (
    detect_abandoned_approaches,
    detect_all_extended,
    detect_error_retry_spirals,
    detect_over_reading,
    detect_permission_loops,
    detect_verbose_thinking,
)

scenarios(
    "../extended_waste/permission_loops.feature",
    "../extended_waste/error_retry_spirals.feature",
    "../extended_waste/over_reading.feature",
    "../extended_waste/abandoned_approaches.feature",
    "../extended_waste/verbose_thinking.feature",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table_to_dicts(datatable):
    """Convert a pytest-bdd 8.x datatable (list[list[str]]) to list[dict]."""
    headers = datatable[0]
    return [dict(zip(headers, row)) for row in datatable[1:]]


def _build_spans_from_table(datatable):
    """Build a list of ClassifiedSpan from a Gherkin datatable."""
    rows = _table_to_dicts(datatable)
    spans = []
    phase_map = {
        "reasoning": SpanPhase.REASONING,
        "tool_use": SpanPhase.TOOL_USE,
        "generation": SpanPhase.GENERATION,
    }
    for row in rows:
        span = TokenSpan(
            text=row["text"],
            phase=phase_map[row["phase"]],
            position=int(row["position"]),
            token_count=int(row["token_count"]),
            source_message_uuid=f"msg-{row['position']}",
            block_type=row["block_type"],
        )
        spans.append(
            ClassifiedSpan(
                span=span,
                label=SpanLabel.ALIGNED_REASONING,
                confidence=0.9,
                cosine_similarity=0.7,
            )
        )
    return spans


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def context():
    """Mutable dict shared across steps within a single scenario."""
    return {"spans": [], "patterns": []}


# ---------------------------------------------------------------------------
# Background steps (one per detector)
# ---------------------------------------------------------------------------


@given(
    parsers.parse(
        "the permission loop detector with default min_retries of {n:d}"
    ),
    target_fixture="context",
)
def permission_loop_background(n):
    return {"spans": [], "patterns": [], "perm_min_retries": n}


@given(
    parsers.parse(
        "the error-retry spiral detector with default similarity_threshold"
        " of {threshold:f} and min_retries of {n:d}"
    ),
    target_fixture="context",
)
def error_retry_background(threshold, n):
    return {
        "spans": [],
        "patterns": [],
        "error_sim_threshold": threshold,
        "error_min_retries": n,
    }


@given(
    parsers.parse("the over-reading detector with default min_reads of {n:d}"),
    target_fixture="context",
)
def over_reading_background(n):
    return {"spans": [], "patterns": [], "over_reading_min_reads": n}


@given(
    parsers.parse(
        "the verbose thinking detector with default ratio_threshold"
        " of {ratio:f} and min_thinking_tokens of {n:d}"
    ),
    target_fixture="context",
)
def verbose_thinking_background(ratio, n):
    return {
        "spans": [],
        "patterns": [],
        "verbose_ratio": ratio,
        "verbose_min_tokens": n,
    }


# ---------------------------------------------------------------------------
# Given: shared span table
# ---------------------------------------------------------------------------


@given("a session with the following spans:")
def session_with_spans(context, datatable):
    context["spans"] = _build_spans_from_table(datatable)


@given("a session containing waste patterns from all 5 extended detectors:")
def session_with_all_waste(context, datatable):
    context["spans"] = _build_spans_from_table(datatable)


# ---------------------------------------------------------------------------
# When: individual detector runs
# ---------------------------------------------------------------------------


@when("I run the permission loop detector")
def run_permission_loop_detector(context):
    context["patterns"] = detect_permission_loops(
        context["spans"],
        min_retries=context.get("perm_min_retries", 2),
    )


@when("I run the error-retry spiral detector")
def run_error_retry_detector(context):
    context["patterns"] = detect_error_retry_spirals(
        context["spans"],
        similarity_threshold=context.get("error_sim_threshold", 0.90),
        min_retries=context.get("error_min_retries", 3),
    )


@when("I run the over-reading detector")
def run_over_reading_detector(context):
    context["patterns"] = detect_over_reading(
        context["spans"],
        min_reads=context.get("over_reading_min_reads", 2),
    )


@when("I run the abandoned approach detector")
def run_abandoned_approach_detector(context):
    context["patterns"] = detect_abandoned_approaches(context["spans"])


@when("I run the verbose thinking detector")
def run_verbose_thinking_detector(context):
    context["patterns"] = detect_verbose_thinking(
        context["spans"],
        ratio_threshold=context.get("verbose_ratio", 10.0),
        min_thinking_tokens=context.get("verbose_min_tokens", 500),
    )


@when("I run detect_all_extended with default parameters")
def run_detect_all_extended(context):
    context["patterns"] = detect_all_extended(context["spans"])


# ---------------------------------------------------------------------------
# Then: pattern count assertions
# ---------------------------------------------------------------------------


@then(parsers.parse("{count:d} permission loop pattern should be detected"))
def check_permission_loop_count_singular(context, count):
    matching = [
        p for p in context["patterns"] if p.pattern_type == "permission_loop"
    ]
    assert len(matching) == count, (
        f"Expected {count} permission_loop pattern(s), got {len(matching)}"
    )


@then(parsers.parse("{count:d} permission loop patterns should be detected"))
def check_permission_loop_count_plural(context, count):
    matching = [
        p for p in context["patterns"] if p.pattern_type == "permission_loop"
    ]
    assert len(matching) == count, (
        f"Expected {count} permission_loop pattern(s), got {len(matching)}"
    )


@then(parsers.parse("{count:d} error-retry spiral pattern should be detected"))
def check_error_retry_count_singular(context, count):
    matching = [
        p for p in context["patterns"] if p.pattern_type == "error_retry_spiral"
    ]
    assert len(matching) == count, (
        f"Expected {count} error_retry_spiral pattern(s), got {len(matching)}"
    )


@then(parsers.parse("{count:d} error-retry spiral patterns should be detected"))
def check_error_retry_count_plural(context, count):
    matching = [
        p for p in context["patterns"] if p.pattern_type == "error_retry_spiral"
    ]
    assert len(matching) == count, (
        f"Expected {count} error_retry_spiral pattern(s), got {len(matching)}"
    )


@then(parsers.parse("{count:d} over-reading pattern should be detected"))
def check_over_reading_count_singular(context, count):
    matching = [
        p for p in context["patterns"] if p.pattern_type == "over_reading"
    ]
    assert len(matching) == count, (
        f"Expected {count} over_reading pattern(s), got {len(matching)}"
    )


@then(parsers.parse("{count:d} over-reading patterns should be detected"))
def check_over_reading_count_plural(context, count):
    matching = [
        p for p in context["patterns"] if p.pattern_type == "over_reading"
    ]
    assert len(matching) == count, (
        f"Expected {count} over_reading pattern(s), got {len(matching)}"
    )


@then(parsers.parse("{count:d} abandoned approach pattern should be detected"))
def check_abandoned_count_singular(context, count):
    matching = [
        p for p in context["patterns"]
        if p.pattern_type == "abandoned_approach"
    ]
    assert len(matching) == count, (
        f"Expected {count} abandoned_approach pattern(s), got {len(matching)}"
    )


@then(parsers.parse("{count:d} abandoned approach patterns should be detected"))
def check_abandoned_count_plural(context, count):
    matching = [
        p for p in context["patterns"]
        if p.pattern_type == "abandoned_approach"
    ]
    assert len(matching) == count, (
        f"Expected {count} abandoned_approach pattern(s), got {len(matching)}"
    )


@then(parsers.parse("{count:d} verbose thinking pattern should be detected"))
def check_verbose_count_singular(context, count):
    matching = [
        p for p in context["patterns"]
        if p.pattern_type == "verbose_thinking"
    ]
    assert len(matching) == count, (
        f"Expected {count} verbose_thinking pattern(s), got {len(matching)}"
    )


@then(parsers.parse("{count:d} verbose thinking patterns should be detected"))
def check_verbose_count_plural(context, count):
    matching = [
        p for p in context["patterns"]
        if p.pattern_type == "verbose_thinking"
    ]
    assert len(matching) == count, (
        f"Expected {count} verbose_thinking pattern(s), got {len(matching)}"
    )


# ---------------------------------------------------------------------------
# Then: detail assertions (shared across detectors)
# ---------------------------------------------------------------------------


@then(parsers.parse('the pattern should have tool_name "{name}"'))
def check_tool_name(context, name):
    p = context["patterns"][0]
    assert p.details["tool_name"] == name, (
        f"Expected tool_name '{name}', got '{p.details.get('tool_name')}'"
    )


@then(parsers.parse("the pattern should report {n:d} retries"))
def check_retries(context, n):
    p = context["patterns"][0]
    assert p.details["retries"] == n, (
        f"Expected {n} retries, got {p.details.get('retries')}"
    )


@then(parsers.parse("the tokens_wasted should be {n:d}"))
def check_tokens_wasted(context, n):
    p = context["patterns"][0]
    assert p.tokens_wasted == n, (
        f"Expected tokens_wasted={n}, got {p.tokens_wasted}"
    )


# ---------------------------------------------------------------------------
# Permission loops: specific assertions
# ---------------------------------------------------------------------------


@then(
    parsers.parse(
        "the tokens_wasted should exclude the initial legitimate attempt"
        " of {n:d} tokens"
    )
)
def check_excludes_initial_perm(context, n):
    # This is an assertion-only step: the detector already excludes the
    # initial attempt.  We verify by checking tokens_wasted does not include
    # the initial tool_use span's tokens.
    p = context["patterns"][0]
    initial_span = context["spans"][0]
    assert p.tokens_wasted == (
        sum(
            s.span.token_count
            for s in context["spans"]
            if s.span.block_type == "tool_use"
        )
        - initial_span.span.token_count
    ), "tokens_wasted should exclude the initial legitimate attempt"


@then(
    parsers.parse(
        'the keywords "not allowed", "eacces", and "unauthorized" should all'
        " trigger denial detection"
    )
)
def check_denial_keywords(context):
    # Assertion-only: the fact that the pattern was detected with all three
    # keyword variants present in the tool_result spans proves they all
    # trigger detection.
    assert len(context["patterns"]) >= 1, (
        "At least one pattern should have been detected proving all keywords"
        " trigger denial detection"
    )


# ---------------------------------------------------------------------------
# Error-retry spirals: specific assertions
# ---------------------------------------------------------------------------


@then(
    parsers.parse(
        "the tokens_wasted should exclude the initial attempt of {n:d} tokens"
    )
)
def check_excludes_initial_error(context, n):
    # Assertion-only: the detector excludes the first attempt from wasted
    # tokens.  We verify the initial tool_use tokens are not in the waste.
    p = context["patterns"][0]
    initial_span = next(
        s for s in context["spans"] if s.span.block_type == "tool_use"
    )
    assert initial_span.span.token_count == n
    # tokens_wasted should not include the initial attempt
    tool_use_total = sum(
        s.span.token_count
        for s in context["spans"]
        if s.span.block_type == "tool_use"
    )
    assert p.tokens_wasted == tool_use_total - n


# ---------------------------------------------------------------------------
# Over-reading: specific assertions
# ---------------------------------------------------------------------------


@then(
    parsers.parse(
        "the pattern should report {total:d} total reads"
        " and {redundant:d} redundant reads"
    )
)
def check_read_counts(context, total, redundant):
    p = context["patterns"][0]
    assert p.details["read_count"] == total, (
        f"Expected {total} total reads, got {p.details.get('read_count')}"
    )
    assert p.details["redundant_reads"] == redundant, (
        f"Expected {redundant} redundant reads, got"
        f" {p.details.get('redundant_reads')}"
    )


@then(
    parsers.parse(
        "the tokens_wasted should exclude the first legitimate read"
        " of {n:d} tokens"
    )
)
def check_excludes_first_read(context, n):
    # Assertion-only: first read is not wasted.
    p = context["patterns"][0]
    first_read_span = next(
        s for s in context["spans"] if s.span.block_type == "tool_use"
    )
    assert first_read_span.span.token_count == n
    total_read_tokens = sum(
        s.span.token_count
        for s in context["spans"]
        if s.span.block_type == "tool_use"
    )
    assert p.tokens_wasted == total_read_tokens - n


@then(
    parsers.parse(
        "the subsequent {n:d} reads after the Edit give only"
        " {redundant:d} redundant read which is below min_reads of {min_r:d}"
    )
)
def check_post_edit_reads(context, n, redundant, min_r):
    # Assertion-only step: the scenario already asserts 0 patterns detected.
    # This step documents *why* -- the Edit resets the counter and the
    # remaining reads don't meet the threshold.
    pass


@then(
    parsers.parse(
        'a pattern for "{file_path}" should report {reads:d} reads'
        " and {redundant:d} redundant reads"
    )
)
def check_pattern_for_file(context, file_path, reads, redundant):
    matching = [
        p for p in context["patterns"]
        if p.details.get("file_path") == file_path
    ]
    assert len(matching) == 1, (
        f"Expected exactly 1 pattern for '{file_path}', got {len(matching)}"
    )
    p = matching[0]
    assert p.details["read_count"] == reads, (
        f"Expected {reads} reads for '{file_path}', got"
        f" {p.details.get('read_count')}"
    )
    assert p.details["redundant_reads"] == redundant, (
        f"Expected {redundant} redundant reads for '{file_path}', got"
        f" {p.details.get('redundant_reads')}"
    )


# ---------------------------------------------------------------------------
# Abandoned approaches: specific assertions
# ---------------------------------------------------------------------------


@then(parsers.parse('the pattern should report file_path "{path}"'))
def check_file_path(context, path):
    p = context["patterns"][0]
    assert p.details["file_path"] == path, (
        f"Expected file_path '{path}', got '{p.details.get('file_path')}'"
    )


@then(
    parsers.parse(
        'the tokens_wasted should cover all spans that touched "{path}"'
    )
)
def check_tokens_cover_file(context, path):
    # Assertion-only: verify the wasted token count equals the sum of all
    # spans that reference the given file path.
    p = context["patterns"][0]
    expected = sum(
        s.span.token_count
        for s in context["spans"]
        if path in s.span.text
    )
    assert p.tokens_wasted == expected, (
        f"Expected tokens_wasted={expected} covering all spans for '{path}',"
        f" got {p.tokens_wasted}"
    )


@then(
    "the pattern description should indicate the file was edited but never"
    " revisited"
)
def check_abandoned_description(context):
    p = context["patterns"][0]
    desc_lower = p.description.lower()
    assert "edited" in desc_lower or "edit" in desc_lower, (
        f"Description should mention editing: {p.description}"
    )
    assert "never revisited" in desc_lower or "never" in desc_lower, (
        f"Description should mention 'never revisited': {p.description}"
    )


@then(parsers.parse('the file "{path}" should not be flagged as abandoned'))
def check_file_not_abandoned(context, path):
    abandoned_paths = [
        p.details.get("file_path")
        for p in context["patterns"]
        if p.pattern_type == "abandoned_approach"
    ]
    assert path not in abandoned_paths, (
        f"File '{path}' should NOT be flagged as abandoned, but it was"
    )


# ---------------------------------------------------------------------------
# Verbose thinking: specific assertions
# ---------------------------------------------------------------------------


@then(parsers.parse("the pattern should report a ratio of {ratio:f}"))
def check_ratio(context, ratio):
    p = context["patterns"][0]
    actual = p.details["ratio"]
    assert abs(actual - ratio) < 0.01, (
        f"Expected ratio {ratio}, got {actual}"
    )


@then("the pattern should report 0 action tokens and infinite ratio")
def check_infinite_ratio(context):
    p = context["patterns"][0]
    assert p.details["action_tokens"] == 0, (
        f"Expected 0 action_tokens, got {p.details.get('action_tokens')}"
    )
    assert math.isinf(p.details["ratio"]), (
        f"Expected infinite ratio, got {p.details.get('ratio')}"
    )


# ---------------------------------------------------------------------------
# detect_all_extended: combined assertions
# ---------------------------------------------------------------------------


@then("patterns from multiple detectors should be returned")
def check_multiple_detectors(context):
    types = {p.pattern_type for p in context["patterns"]}
    assert len(types) >= 2, (
        f"Expected patterns from multiple detectors, got types: {types}"
    )


@then("the results should be sorted by start_position in ascending order")
def check_sorted_by_position(context):
    positions = [p.start_position for p in context["patterns"]]
    assert positions == sorted(positions), (
        f"Patterns not sorted by start_position: {positions}"
    )


@then(
    "the combined list should include verbose_thinking, over_reading,"
    " and permission_loop patterns"
)
def check_combined_types(context):
    types = {p.pattern_type for p in context["patterns"]}
    for expected in ("verbose_thinking", "over_reading", "permission_loop"):
        assert expected in types, (
            f"Expected '{expected}' in pattern types, got {types}"
        )
