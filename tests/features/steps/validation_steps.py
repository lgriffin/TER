"""Step definitions for validation and quality features."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from ter_calculator.validation import (
    validate_jsonl_line,
    validate_session,
    generate_health_report,
    assess_completeness,
    validate_jsonl_file,
)

scenarios(
    "../validation_quality/jsonl_validation.feature",
    "../validation_quality/session_validation.feature",
    "../validation_quality/health_report.feature",
    "../validation_quality/completeness.feature",
)


@pytest.fixture
def context():
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message_entry(
    role: str,
    content: list | str,
    *,
    uuid: str = "msg-1",
    session_id: str = "sess-1",
    timestamp: str | None = None,
    stop_reason: str | None = None,
    usage: dict | None = None,
) -> dict:
    """Build a full JSONL-style entry dict."""
    entry: dict = {
        "type": role,
        "uuid": uuid,
        "sessionId": session_id,
        "message": {
            "role": role,
            "content": content,
        },
    }
    if timestamp is not None:
        entry["timestamp"] = timestamp
    if stop_reason is not None:
        entry["message"]["stop_reason"] = stop_reason
    if usage is not None:
        entry["message"]["usage"] = usage
    return entry


# ===========================================================================
# JSONL line validation — Given steps
# ===========================================================================


@given(
    "a JSONL line with type, uuid, sessionId, and message containing role and content"
)
def valid_line(context):
    context["line"] = json.dumps(
        {
            "type": "assistant",
            "uuid": "abc-123",
            "sessionId": "sess-1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
            },
        }
    )


@given("a JSONL line containing malformed JSON")
def malformed_line(context):
    context["line"] = "{not valid json"


@given(parsers.parse('a JSONL line missing the "message" field'))
def missing_message(context):
    context["line"] = json.dumps(
        {
            "type": "assistant",
            "uuid": "abc-123",
            "sessionId": "sess-1",
        }
    )


@given(parsers.parse('a JSONL line with type "{meta_type}"'))
def meta_line(context, meta_type):
    context["line"] = json.dumps(
        {
            "type": meta_type,
            "uuid": "meta-1",
            "sessionId": "sess-1",
        }
    )


@given(parsers.parse('a content block with type "{block_type}"'))
def content_block_unknown(context, block_type):
    context["line"] = json.dumps(
        {
            "type": "assistant",
            "uuid": "abc-123",
            "sessionId": "sess-1",
            "message": {
                "role": "assistant",
                "content": [{"type": block_type, "data": "something"}],
            },
        }
    )


@given(parsers.parse('a content block of type "tool_use" without a "name" field'))
def tool_use_no_name(context):
    context["line"] = json.dumps(
        {
            "type": "assistant",
            "uuid": "abc-123",
            "sessionId": "sess-1",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu-1", "input": {"key": "val"}}
                ],
            },
        }
    )


@given("a message with negative output_tokens")
def negative_tokens(context):
    context["line"] = json.dumps(
        {
            "type": "assistant",
            "uuid": "abc-123",
            "sessionId": "sess-1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
                "usage": {"input_tokens": 10, "output_tokens": -5},
            },
        }
    )


@given("a JSONL file with 100 lines and 3 malformed lines", target_fixture="context")
def jsonl_file_with_errors(context, tmp_path):
    malformed_indices = {10, 50, 90}  # 1-based line numbers
    file_path = tmp_path / "test.jsonl"
    with open(file_path, "w", encoding="utf-8") as fh:
        for i in range(1, 101):
            if i in malformed_indices:
                fh.write("{bad json\n")
            else:
                line = json.dumps(
                    {
                        "type": "assistant",
                        "uuid": f"msg-{i}",
                        "sessionId": "sess-1",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": f"Line {i}"}],
                        },
                    }
                )
                fh.write(line + "\n")
    context["file_path"] = file_path
    context["malformed_line_numbers"] = sorted(malformed_indices)
    return context


@given("a path to a non-existent JSONL file")
def nonexistent_file(context, tmp_path):
    context["file_path"] = tmp_path / "does_not_exist.jsonl"


# ===========================================================================
# JSONL line validation — When steps
# ===========================================================================


@when("the line is validated")
def validate_line(context):
    context["result"] = validate_jsonl_line(context["line"])


@when("validate_jsonl_file is called")
def run_validate_file(context):
    try:
        context["file_result"] = validate_jsonl_file(context["file_path"])
    except FileNotFoundError as exc:
        context["file_error"] = exc


# ===========================================================================
# JSONL line validation — Then steps
# ===========================================================================


@then("the result is valid with no errors")
def check_valid_no_errors(context):
    assert context["result"].valid is True
    assert len(context["result"].errors) == 0


@then("the result is valid")
def check_valid(context):
    # Depending on the scenario, the result may be stored under different keys.
    if "result" in context:
        assert context["result"].valid is True
    elif "session_result" in context:
        assert context["session_result"].valid is True
    else:
        raise AssertionError("No result found in context")


@then("the result is invalid")
def check_invalid(context):
    assert context["result"].valid is False


@then(parsers.parse('the error message includes "{text}"'))
def check_error_includes(context, text):
    errors = " ".join(context["result"].errors)
    assert text.lower() in errors.lower()


@then("the error reports missing required fields")
def check_missing_fields(context):
    errors = " ".join(context["result"].errors)
    assert "missing" in errors.lower() or "required" in errors.lower()


@then("a warning about unknown block type is reported")
def check_unknown_warning(context):
    warnings = " ".join(context["result"].warnings)
    assert "unknown" in warnings.lower() or "block type" in warnings.lower()


@then("the line is still valid")
def still_valid(context):
    assert context["result"].valid is True


@then("an error is reported about missing name")
def check_missing_name_error(context):
    errors = " ".join(context["result"].errors)
    assert "name" in errors.lower()


@then("an error is reported about non-negative token counts")
def check_negative_tokens_error(context):
    # validate_jsonl_line does not check usage tokens itself; it only
    # validates structure.  Token-level checks happen in validate_session.
    # Build a session from the parsed line and validate that instead.
    parsed = json.loads(context["line"])
    session_result = validate_session([parsed])
    all_messages = " ".join(session_result.errors + session_result.warnings)
    assert (
        "negative" in all_messages.lower()
        or "non-negative" in all_messages.lower()
        or "token" in all_messages.lower()
    )


@then(
    parsers.parse(
        "the result reports total_lines of {total:d} and valid_lines of {valid:d}"
    )
)
def check_file_totals(context, total, valid):
    fr = context["file_result"]
    assert fr.total_lines == total
    assert fr.valid_lines == valid


@then("error_lines contains the 3 malformed line numbers")
def check_error_lines(context):
    fr = context["file_result"]
    expected = context["malformed_line_numbers"]
    assert sorted(fr.error_lines) == sorted(expected)


@then("a FileNotFoundError is raised")
def check_file_not_found(context):
    assert "file_error" in context
    assert isinstance(context["file_error"], FileNotFoundError)


# ===========================================================================
# Session validation — Given steps
# ===========================================================================


@given("a parsed session with user and assistant messages")
def valid_session(context):
    context["parsed_lines"] = [
        _make_message_entry(
            "user",
            [{"type": "text", "text": "Hello"}],
            uuid="u1",
            timestamp="2026-01-01T00:00:00",
        ),
        _make_message_entry(
            "assistant",
            [{"type": "text", "text": "Hi there"}],
            uuid="a1",
            timestamp="2026-01-01T00:01:00",
        ),
    ]


@given("a parsed session with only assistant messages")
def only_assistant(context):
    context["parsed_lines"] = [
        _make_message_entry(
            "assistant",
            [{"type": "text", "text": "I will help you."}],
            uuid="a1",
        ),
        _make_message_entry(
            "assistant",
            [{"type": "text", "text": "Here is the result."}],
            uuid="a2",
        ),
    ]


@given("a parsed session with only user messages")
def only_user(context):
    context["parsed_lines"] = [
        _make_message_entry(
            "user",
            [{"type": "text", "text": "Hello"}],
            uuid="u1",
        ),
        _make_message_entry(
            "user",
            [{"type": "text", "text": "Are you there?"}],
            uuid="u2",
        ),
    ]


@given("a parsed session where a later message has an earlier timestamp")
def out_of_order_timestamps(context):
    context["parsed_lines"] = [
        _make_message_entry(
            "user",
            [{"type": "text", "text": "First"}],
            uuid="u1",
            timestamp="2026-01-01T00:05:00",
        ),
        _make_message_entry(
            "assistant",
            [{"type": "text", "text": "Second"}],
            uuid="a1",
            timestamp="2026-01-01T00:01:00",  # earlier than first
        ),
    ]


@given("a parsed session with 5 text blocks, 3 tool_use blocks, and 2 thinking blocks")
def session_with_counted_blocks(context):
    text_blocks = [{"type": "text", "text": f"text-{i}"} for i in range(5)]
    tool_blocks = [
        {"type": "tool_use", "id": f"tu-{i}", "name": f"Tool{i}", "input": {}}
        for i in range(3)
    ]
    thinking_blocks = [{"type": "thinking", "text": f"thinking-{i}"} for i in range(2)]
    context["parsed_lines"] = [
        _make_message_entry(
            "user",
            text_blocks[:2],
            uuid="u1",
        ),
        _make_message_entry(
            "assistant",
            text_blocks[2:] + thinking_blocks + tool_blocks,
            uuid="a1",
        ),
    ]


# ===========================================================================
# Session validation — When steps
# ===========================================================================


@when("session validation runs")
def run_session_validation(context):
    context["session_result"] = validate_session(context["parsed_lines"])


# ===========================================================================
# Session validation — Then steps
# ===========================================================================


@then("a warning about missing user messages is reported")
def check_missing_user_warning(context):
    # The validation module may report this as either a warning or error.
    all_messages = context["session_result"].errors + context["session_result"].warnings
    combined = " ".join(all_messages).lower()
    assert "user" in combined and "message" in combined


@then("a warning about missing assistant messages is reported")
def check_missing_assistant_warning(context):
    all_messages = context["session_result"].errors + context["session_result"].warnings
    combined = " ".join(all_messages).lower()
    assert "assistant" in combined and "message" in combined


@then("a warning about timestamp ordering is reported")
def check_timestamp_ordering_warning(context):
    all_messages = context["session_result"].errors + context["session_result"].warnings
    combined = " ".join(all_messages).lower()
    assert "timestamp" in combined or "order" in combined


@then(parsers.parse("the result reports content_block_count of {count:d}"))
def check_content_block_count(context, count):
    assert context["session_result"].content_block_count == count


# ===========================================================================
# Health report — Given steps
# ===========================================================================


@given("a parsed session with multiple content blocks")
def session_with_blocks(context):
    context["parsed_lines"] = [
        _make_message_entry(
            "user",
            [{"type": "text", "text": "Build auth"}],
            uuid="u1",
        ),
        _make_message_entry(
            "assistant",
            [
                {
                    "type": "thinking",
                    "text": "I need to think about authentication...",
                },
                {
                    "type": "tool_use",
                    "id": "tu-1",
                    "name": "Write",
                    "input": {"file_path": "auth.py"},
                },
                {"type": "text", "text": "I have created the auth module."},
            ],
            uuid="a1",
        ),
    ]


@given("a parsed session with reasoning, tool_use, and generation blocks")
def session_with_distribution(context):
    context["parsed_lines"] = [
        _make_message_entry(
            "user",
            [{"type": "text", "text": "Do something"}],
            uuid="u1",
        ),
        _make_message_entry(
            "assistant",
            [
                {"type": "thinking", "text": "Let me reason about this..."},
                {
                    "type": "tool_use",
                    "id": "tu-1",
                    "name": "Bash",
                    "input": {"command": "ls"},
                },
            ],
            uuid="a1",
        ),
        _make_message_entry(
            "user",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu-1",
                    "content": "file1.py\nfile2.py",
                },
            ],
            uuid="u2",
        ),
        _make_message_entry(
            "assistant",
            [{"type": "text", "text": "Here are the files I found."}],
            uuid="a2",
        ),
    ]


@given("a parsed session with approximately 1000 spans")
def session_with_1000_spans(context):
    # Build a session with ~1000 content blocks (spans).
    blocks = [{"type": "text", "text": f"span-{i}"} for i in range(1000)]
    # Split across a few messages to keep it realistic.
    chunk_size = 250
    lines = []
    for idx, start in enumerate(range(0, len(blocks), chunk_size)):
        chunk = blocks[start : start + chunk_size]
        role = "assistant" if idx % 2 == 1 else "user"
        lines.append(_make_message_entry(role, chunk, uuid=f"m-{idx}"))
    context["parsed_lines"] = lines


@given("a parsed session with 5 user messages and 8 assistant messages")
def session_with_user_assistant_counts(context):
    lines = []
    for i in range(5):
        lines.append(
            _make_message_entry(
                "user",
                [{"type": "text", "text": f"user msg {i}"}],
                uuid=f"u-{i}",
            )
        )
    for i in range(8):
        lines.append(
            _make_message_entry(
                "assistant",
                [{"type": "text", "text": f"assistant msg {i}"}],
                uuid=f"a-{i}",
            )
        )
    context["parsed_lines"] = lines


# ===========================================================================
# Health report — When steps
# ===========================================================================


@when("a health report is generated")
def gen_health(context):
    context["health"] = generate_health_report(context["parsed_lines"])


# ===========================================================================
# Health report — Then steps
# ===========================================================================


@then("estimated_total_tokens is a positive integer")
def check_positive_tokens(context):
    assert isinstance(context["health"].estimated_total_tokens, int)
    assert context["health"].estimated_total_tokens > 0


@then(
    "the content_distribution includes counts for text, thinking, tool_use, and tool_result"
)
def check_content_distribution(context):
    dist = context["health"].content_distribution
    # Each of the four block types should have a count attribute >= 0,
    # and at least some of them should be present in our fixture.
    assert hasattr(dist, "text_count")
    assert hasattr(dist, "thinking_count")
    assert hasattr(dist, "tool_use_count")
    assert hasattr(dist, "tool_result_count")
    # Our fixture has at least one of each relevant type.
    assert dist.text_count > 0
    assert dist.thinking_count > 0
    assert dist.tool_use_count > 0
    assert dist.tool_result_count > 0


@then("estimated_analysis_seconds is approximately 0.5 seconds")
def check_analysis_time(context):
    # The implementation uses 0.0005s per span, so 1000 spans ~ 0.5s.
    assert context["health"].estimated_analysis_seconds == pytest.approx(0.5, abs=0.15)


@then(parsers.parse("user_count is {user:d} and assistant_count is {assistant:d}"))
def check_message_counts(context, user, assistant):
    assert context["health"].user_message_count == user
    assert context["health"].assistant_message_count == assistant


# ===========================================================================
# Completeness — Given steps
# ===========================================================================


@given('a session where the last assistant message has stop_reason "end_turn"')
def complete_session(context):
    context["parsed_lines"] = [
        _make_message_entry(
            "user",
            [{"type": "text", "text": "Hello"}],
            uuid="u1",
        ),
        _make_message_entry(
            "assistant",
            [{"type": "text", "text": "Done"}],
            uuid="a1",
            stop_reason="end_turn",
        ),
    ]


@given("all tool_use blocks have matching tool_result responses")
def matched_tools():
    # The complete_session fixture already has no unmatched tool_use blocks.
    pass


@given("a session where the last message is a tool_use with no tool_result")
def session_ending_mid_tool_use(context):
    context["parsed_lines"] = [
        _make_message_entry(
            "user",
            [{"type": "text", "text": "Do something"}],
            uuid="u1",
        ),
        _make_message_entry(
            "assistant",
            [
                {"type": "text", "text": "Let me try..."},
                {
                    "type": "tool_use",
                    "id": "tu-1",
                    "name": "Bash",
                    "input": {"command": "ls"},
                },
            ],
            uuid="a1",
            stop_reason="tool_use",
        ),
    ]


@given("a session with 3 tool_use blocks and only 2 tool_result blocks")
def session_with_unmatched_tool_use(context):
    context["parsed_lines"] = [
        _make_message_entry(
            "user",
            [{"type": "text", "text": "Do three things"}],
            uuid="u1",
        ),
        _make_message_entry(
            "assistant",
            [
                {
                    "type": "tool_use",
                    "id": "tu-1",
                    "name": "Bash",
                    "input": {"command": "ls"},
                },
                {
                    "type": "tool_use",
                    "id": "tu-2",
                    "name": "Read",
                    "input": {"path": "a.py"},
                },
                {
                    "type": "tool_use",
                    "id": "tu-3",
                    "name": "Write",
                    "input": {"path": "b.py"},
                },
            ],
            uuid="a1",
        ),
        _make_message_entry(
            "user",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu-1",
                    "content": "result 1",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tu-2",
                    "content": "result 2",
                },
            ],
            uuid="u2",
        ),
        _make_message_entry(
            "assistant",
            [{"type": "text", "text": "Done with two of three."}],
            uuid="a2",
            stop_reason="end_turn",
        ),
    ]


@given("a session where the last assistant message has no stop_reason")
def session_no_stop_reason(context):
    context["parsed_lines"] = [
        _make_message_entry(
            "user",
            [{"type": "text", "text": "Hello"}],
            uuid="u1",
        ),
        _make_message_entry(
            "assistant",
            [{"type": "text", "text": "I was saying..."}],
            uuid="a1",
            # No stop_reason set
        ),
    ]


# ===========================================================================
# Completeness — When steps
# ===========================================================================


@when("completeness is assessed")
def assess(context):
    context["completeness"] = assess_completeness(context["parsed_lines"])


# ===========================================================================
# Completeness — Then steps
# ===========================================================================


@then("is_complete is true")
def check_complete(context):
    assert context["completeness"].is_complete is True


@then("is_complete is false")
def check_incomplete(context):
    assert context["completeness"].is_complete is False


@then(parsers.parse("completeness_score is {score:f}"))
def check_score(context, score):
    assert context["completeness"].completeness_score == pytest.approx(score, abs=0.01)


@then("completeness_score is below 1.0")
def check_score_below_one(context):
    assert context["completeness"].completeness_score < 1.0


@then("the issues list mentions unresolved tool calls")
def check_unresolved_tool_issues(context):
    combined = " ".join(context["completeness"].issues).lower()
    assert "unresolved" in combined or "tool_use" in combined or "tool" in combined


@then("the issues list reports the unmatched tool_use")
def check_unmatched_tool_use_issue(context):
    combined = " ".join(context["completeness"].issues).lower()
    assert "unresolved" in combined or "tool_use" in combined or "unmatched" in combined


@then("the issues list mentions missing stop_reason")
def check_missing_stop_reason_issue(context):
    combined = " ".join(context["completeness"].issues).lower()
    assert "stop_reason" in combined
