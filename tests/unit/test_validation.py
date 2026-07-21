"""Tests for JSONL session validation."""

import json

import pytest

from ter_calculator.validation import (
    CompletenessAssessment,
    ContentDistribution,
    FileValidationResult,
    HealthReport,
    SessionValidationResult,
    ValidationResult,
    assess_completeness,
    generate_health_report,
    validate_jsonl_file,
    validate_jsonl_line,
    validate_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_line(
    *,
    line_type="user",
    role="user",
    content="Hello",
    uuid="u1",
    session_id="s1",
    extra_top=None,
    extra_msg=None,
    omit_top=None,
    omit_msg=None,
):
    """Build a valid JSONL dict, then serialise it."""
    top = {
        "type": line_type,
        "uuid": uuid,
        "sessionId": session_id,
        "message": {
            "role": role,
            "content": content,
        },
    }
    if extra_top:
        top.update(extra_top)
    if extra_msg:
        top["message"].update(extra_msg)
    if omit_top:
        for k in omit_top:
            top.pop(k, None)
    if omit_msg:
        for k in omit_msg:
            top["message"].pop(k, None)
    return json.dumps(top)


def _make_assistant_line(
    content=None,
    uuid="a1",
    stop_reason="end_turn",
    usage=None,
    timestamp=None,
):
    """Build an assistant JSONL dict."""
    if content is None:
        content = [{"type": "text", "text": "Hi there!"}]
    msg = {"role": "assistant", "content": content, "stop_reason": stop_reason}
    if usage:
        msg["usage"] = usage
    entry = {
        "type": "assistant",
        "uuid": uuid,
        "sessionId": "s1",
        "message": msg,
    }
    if timestamp:
        entry["timestamp"] = timestamp
    return entry


def _make_user_line(content="Hello", uuid="u1", timestamp=None):
    """Build a user JSONL dict."""
    entry = {
        "type": "user",
        "uuid": uuid,
        "sessionId": "s1",
        "message": {"role": "user", "content": content},
    }
    if timestamp:
        entry["timestamp"] = timestamp
    return entry


# ---------------------------------------------------------------------------
# 1. validate_jsonl_line
# ---------------------------------------------------------------------------


class TestValidateJsonlLine:
    """Tests for single-line JSONL validation."""

    def test_valid_user_line(self):
        line = _make_line(role="user", content="Hello")
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True
        assert result.errors == []

    def test_valid_assistant_line_with_text_block(self):
        content = [{"type": "text", "text": "Response"}]
        line = _make_line(line_type="assistant", role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True
        assert result.errors == []

    def test_invalid_json(self):
        result = validate_jsonl_line("{not valid json", line_number=5)
        assert result.valid is False
        assert len(result.errors) == 1
        assert "Invalid JSON" in result.errors[0]
        assert result.line_number == 5

    def test_non_object_json(self):
        result = validate_jsonl_line(json.dumps([1, 2, 3]), line_number=2)
        assert result.valid is False
        assert "Expected a JSON object" in result.errors[0]

    def test_empty_line(self):
        result = validate_jsonl_line("", line_number=1)
        assert result.valid is True
        assert any("Empty line" in w for w in result.warnings)

    def test_whitespace_only_line(self):
        result = validate_jsonl_line("   \t  ", line_number=1)
        assert result.valid is True
        assert any("Empty line" in w for w in result.warnings)

    def test_missing_required_top_level_fields(self):
        line = _make_line(omit_top=["type", "uuid"])
        result = validate_jsonl_line(line, line_number=3)
        assert result.valid is False
        assert any("Missing required top-level fields" in e for e in result.errors)
        assert "type" in result.errors[0]
        assert "uuid" in result.errors[0]

    def test_missing_message_field(self):
        line = _make_line(omit_top=["message"])
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is False
        assert any("Missing required top-level fields" in e for e in result.errors)

    def test_missing_required_message_fields(self):
        line = _make_line(omit_msg=["role", "content"])
        result = validate_jsonl_line(line, line_number=4)
        assert result.valid is False
        assert any("Missing required message fields" in e for e in result.errors)

    def test_message_not_dict(self):
        raw = json.dumps(
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "s1",
                "message": "not a dict",
            }
        )
        result = validate_jsonl_line(raw, line_number=1)
        assert result.valid is False
        assert any("'message' must be a dict" in e for e in result.errors)

    def test_unexpected_role_warning(self):
        line = _make_line(role="system")
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True
        assert any("Unexpected role" in w for w in result.warnings)

    def test_unknown_block_type_warning(self):
        content = [{"type": "image_url", "url": "http://example.com"}]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True
        assert any("unknown block type" in w for w in result.warnings)

    def test_content_block_missing_type(self):
        content = [{"text": "no type field here"}]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is False
        assert any("missing 'type' field" in e for e in result.errors)

    def test_text_block_missing_text_field(self):
        content = [{"type": "text"}]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is False
        assert any("missing 'text' field" in e for e in result.errors)

    def test_text_block_text_not_string(self):
        content = [{"type": "text", "text": 42}]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is False
        assert any("'text' must be a string" in e for e in result.errors)

    def test_thinking_block_valid(self):
        content = [{"type": "thinking", "thinking": "Let me think..."}]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True

    def test_thinking_block_with_text_key(self):
        content = [{"type": "thinking", "text": "Thinking via text key"}]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True
        assert result.warnings == []

    def test_thinking_block_missing_both_fields(self):
        content = [{"type": "thinking"}]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True  # warning, not an error
        assert any("missing" in w and "thinking" in w for w in result.warnings)

    def test_tool_use_block_valid(self):
        content = [
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}}
        ]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True

    def test_tool_use_block_missing_name(self):
        content = [{"type": "tool_use", "id": "t1", "input": {}}]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is False
        assert any("missing 'name'" in e for e in result.errors)

    def test_tool_use_block_missing_id(self):
        content = [{"type": "tool_use", "name": "bash", "input": {}}]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is False
        assert any("missing 'id'" in e for e in result.errors)

    def test_tool_result_block_valid(self):
        content = [{"type": "tool_result", "tool_use_id": "t1", "content": "OK"}]
        line = _make_line(role="user", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True

    def test_tool_result_block_missing_tool_use_id(self):
        content = [{"type": "tool_result", "content": "OK"}]
        line = _make_line(role="user", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is False
        assert any("missing 'tool_use_id'" in e for e in result.errors)

    def test_content_not_string_or_list(self):
        line = _make_line(content=42)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is False
        assert any("'content' must be a string or list" in e for e in result.errors)

    def test_content_block_not_dict_warning(self):
        content = ["just a string in the list"]
        line = _make_line(role="assistant", content=content)
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True
        assert any("is not a dict" in w for w in result.warnings)

    @pytest.mark.parametrize(
        "meta_type",
        [
            "attachment",
            "file-history-snapshot",
            "last-prompt",
            "permission-mode",
            "progress",
            "queue-operation",
            "summary",
            "system",
        ],
    )
    def test_meta_line_types_are_valid(self, meta_type):
        raw = json.dumps({"type": meta_type, "data": "something"})
        result = validate_jsonl_line(raw, line_number=1)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_line_number_is_preserved(self):
        result = validate_jsonl_line("{bad json", line_number=99)
        assert result.line_number == 99

    def test_string_content_is_valid(self):
        line = _make_line(role="user", content="plain text content")
        result = validate_jsonl_line(line, line_number=1)
        assert result.valid is True


# ---------------------------------------------------------------------------
# 2. validate_session
# ---------------------------------------------------------------------------


class TestValidateSession:
    """Tests for full session validation."""

    def test_empty_session(self):
        result = validate_session([])
        assert result.valid is False
        assert result.message_count == 0
        assert any("no user messages" in e for e in result.errors)
        assert any("no assistant messages" in e for e in result.errors)

    def test_single_user_message(self):
        entries = [_make_user_line()]
        result = validate_session(entries)
        assert result.valid is False
        assert result.message_count == 1
        assert any("no assistant messages" in e for e in result.errors)

    def test_single_assistant_message(self):
        entries = [_make_assistant_line()]
        result = validate_session(entries)
        assert result.valid is False
        assert result.message_count == 1
        assert any("no user messages" in e for e in result.errors)

    def test_valid_user_assistant_pair(self):
        entries = [
            _make_user_line(timestamp="2026-04-01T10:00:00.000Z"),
            _make_assistant_line(timestamp="2026-04-01T10:00:01.000Z"),
        ]
        result = validate_session(entries)
        assert result.valid is True
        assert result.message_count == 2
        assert result.errors == []

    def test_timestamp_out_of_order(self):
        entries = [
            _make_user_line(uuid="u1", timestamp="2026-04-01T10:00:05.000Z"),
            _make_assistant_line(uuid="a1", timestamp="2026-04-01T10:00:01.000Z"),
        ]
        result = validate_session(entries)
        assert result.valid is False
        assert any("Timestamp out of order" in e for e in result.errors)

    def test_negative_token_count(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(
                usage={
                    "input_tokens": -5,
                    "output_tokens": 10,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                }
            ),
        ]
        result = validate_session(entries)
        assert result.valid is False
        assert any("Negative token count" in e for e in result.errors)

    def test_tool_result_without_matching_tool_use(self):
        entries = [
            _make_user_line(
                content=[
                    {"type": "tool_result", "tool_use_id": "orphan_id", "content": "ok"}
                ]
            ),
            _make_assistant_line(),
        ]
        result = validate_session(entries)
        assert result.valid is False
        assert any("non-existent tool_use ids" in e for e in result.errors)

    def test_tool_use_without_matching_tool_result_warning(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(
                content=[
                    {
                        "type": "tool_use",
                        "id": "t_orphan",
                        "name": "bash",
                        "input": {},
                    },
                    {"type": "text", "text": "done"},
                ]
            ),
        ]
        result = validate_session(entries)
        assert any("without matching tool_result" in w for w in result.warnings)

    def test_matched_tool_use_and_result(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(
                content=[
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {"cmd": "ls"},
                    }
                ]
            ),
            _make_user_line(
                uuid="u2",
                content=[
                    {"type": "tool_result", "tool_use_id": "t1", "content": "file.txt"}
                ],
            ),
            _make_assistant_line(uuid="a2"),
        ]
        result = validate_session(entries)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_meta_lines_are_skipped(self):
        entries = [
            {"type": "system", "data": "init"},
            _make_user_line(),
            {"type": "summary", "text": "summary"},
            _make_assistant_line(),
        ]
        result = validate_session(entries)
        assert result.valid is True
        assert result.message_count == 2

    def test_string_content_counts_as_block(self):
        entries = [
            _make_user_line(content="simple string"),
            _make_assistant_line(),
        ]
        result = validate_session(entries)
        assert result.content_block_count >= 2

    def test_non_dict_entries_skipped(self):
        entries = [
            "not a dict",
            42,
            None,
            _make_user_line(),
            _make_assistant_line(),
        ]
        result = validate_session(entries)
        assert result.valid is True
        assert result.message_count == 2


# ---------------------------------------------------------------------------
# 3. validate_jsonl_file
# ---------------------------------------------------------------------------


class TestValidateJsonlFile:
    """Tests for whole-file JSONL validation."""

    def test_valid_file(self, tmp_path):
        f = tmp_path / "valid.jsonl"
        lines = [
            _make_line(role="user", content="Hi"),
            _make_line(
                line_type="assistant",
                role="assistant",
                content=[{"type": "text", "text": "Hello"}],
                uuid="a1",
            ),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")

        result = validate_jsonl_file(str(f))
        assert result.valid is True
        assert result.total_lines == 2
        assert result.valid_lines == 2
        assert result.error_lines == []
        assert result.errors == []

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="JSONL file not found"):
            validate_jsonl_file("/tmp/does_not_exist_abc123.jsonl")

    def test_file_with_invalid_lines(self, tmp_path):
        f = tmp_path / "mixed.jsonl"
        valid = _make_line(role="user", content="Hello")
        invalid = "{bad json"
        f.write_text(f"{valid}\n{invalid}\n", encoding="utf-8")

        result = validate_jsonl_file(str(f))
        assert result.valid is False
        assert result.total_lines == 2
        assert result.valid_lines == 1
        assert 2 in result.error_lines
        assert len(result.errors) == 1
        assert "Invalid JSON" in result.errors[0]

    def test_file_with_empty_lines(self, tmp_path):
        f = tmp_path / "empties.jsonl"
        valid = _make_line(role="user", content="Hello")
        f.write_text(f"{valid}\n\n\n", encoding="utf-8")

        result = validate_jsonl_file(str(f))
        assert result.valid is True
        assert any("Empty line" in w for w in result.warnings)

    def test_file_with_meta_lines(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        meta = json.dumps({"type": "permission-mode", "mode": "default"})
        valid = _make_line(role="user", content="Hello")
        f.write_text(f"{meta}\n{valid}\n", encoding="utf-8")

        result = validate_jsonl_file(str(f))
        assert result.valid is True
        assert result.total_lines == 2
        assert result.valid_lines == 2

    def test_file_all_invalid(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text("{bad\n{also bad\n", encoding="utf-8")

        result = validate_jsonl_file(str(f))
        assert result.valid is False
        assert result.total_lines == 2
        assert result.valid_lines == 0
        assert result.error_lines == [1, 2]
        assert len(result.errors) == 2

    def test_file_accepts_path_object(self, tmp_path):
        f = tmp_path / "path_obj.jsonl"
        f.write_text(_make_line(role="user", content="Hi"), encoding="utf-8")
        result = validate_jsonl_file(f)  # Pass Path object, not str
        assert result.valid is True


# ---------------------------------------------------------------------------
# 4. assess_completeness
# ---------------------------------------------------------------------------


class TestAssessCompleteness:
    """Tests for session completeness assessment."""

    def test_complete_session(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(stop_reason="end_turn"),
        ]
        result = assess_completeness(entries)
        assert result.is_complete is True
        assert result.completeness_score == 1.0
        assert result.issues == []

    def test_no_assistant_messages(self):
        entries = [_make_user_line()]
        result = assess_completeness(entries)
        assert result.is_complete is False
        assert result.completeness_score < 1.0
        assert any("No assistant messages" in i for i in result.issues)

    def test_wrong_stop_reason(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(stop_reason="max_tokens"),
        ]
        result = assess_completeness(entries)
        assert result.is_complete is False
        assert result.completeness_score < 1.0
        assert any("stop_reason" in i for i in result.issues)

    def test_none_stop_reason(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(stop_reason=None),
        ]
        result = assess_completeness(entries)
        assert result.is_complete is False

    def test_unresolved_tool_use(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(
                content=[
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {},
                    },
                    {"type": "text", "text": "done"},
                ],
                stop_reason="end_turn",
            ),
        ]
        result = assess_completeness(entries)
        assert result.is_complete is False
        assert any("unresolved tool_use" in i for i in result.issues)

    def test_session_ends_mid_tool_use(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(
                content=[
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {},
                    }
                ],
                stop_reason="tool_use",
            ),
        ]
        result = assess_completeness(entries)
        assert result.is_complete is False
        assert any("ends mid-tool-use" in i for i in result.issues)

    def test_completeness_score_clamped_to_zero(self):
        result = assess_completeness([])
        assert result.completeness_score >= 0.0

    def test_resolved_tool_use_is_complete(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(
                uuid="a1",
                content=[
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {},
                    }
                ],
                stop_reason="tool_use",
            ),
            _make_user_line(
                uuid="u2",
                content=[
                    {"type": "tool_result", "tool_use_id": "t1", "content": "done"}
                ],
            ),
            _make_assistant_line(
                uuid="a2",
                content=[{"type": "text", "text": "Finished."}],
                stop_reason="end_turn",
            ),
        ]
        result = assess_completeness(entries)
        assert result.is_complete is True
        assert result.completeness_score == 1.0

    def test_meta_lines_ignored(self):
        entries = [
            {"type": "system", "data": "init"},
            _make_user_line(),
            _make_assistant_line(stop_reason="end_turn"),
        ]
        result = assess_completeness(entries)
        assert result.is_complete is True


# ---------------------------------------------------------------------------
# 5. generate_health_report
# ---------------------------------------------------------------------------


class TestGenerateHealthReport:
    """Tests for the pre-analysis health report."""

    def test_basic_health_report(self):
        entries = [
            _make_user_line(content="Hello"),
            _make_assistant_line(content=[{"type": "text", "text": "Hi there!"}]),
        ]
        report = generate_health_report(entries)
        assert report.user_message_count == 1
        assert report.assistant_message_count == 1
        assert report.content_distribution.text_count >= 2
        assert report.generation_tokens > 0
        assert report.parsing_warnings == []

    def test_empty_session_report(self):
        report = generate_health_report([])
        assert report.user_message_count == 0
        assert report.assistant_message_count == 0
        assert report.estimated_total_tokens == 0
        assert report.content_distribution.total == 0

    def test_content_distribution_counts(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(
                content=[
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Here is the answer"},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {"cmd": "ls"},
                    },
                ]
            ),
            _make_user_line(
                uuid="u2",
                content=[
                    {"type": "tool_result", "tool_use_id": "t1", "content": "file.txt"}
                ],
            ),
        ]
        report = generate_health_report(entries)
        dist = report.content_distribution
        assert dist.thinking_count == 1
        assert dist.text_count >= 1
        assert dist.tool_use_count == 1
        assert dist.tool_result_count == 1
        assert dist.total >= 4

    def test_reasoning_tokens_from_thinking_blocks(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(
                content=[
                    {"type": "thinking", "thinking": "Deep reasoning here " * 20},
                    {"type": "text", "text": "Answer"},
                ]
            ),
        ]
        report = generate_health_report(entries)
        assert report.reasoning_tokens > 0
        assert report.generation_tokens > 0

    def test_tool_use_tokens(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(
                content=[
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {"cmd": "echo hello world"},
                    },
                    {"type": "text", "text": "Done"},
                ]
            ),
            _make_user_line(
                uuid="u2",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "hello world",
                    }
                ],
            ),
        ]
        report = generate_health_report(entries)
        assert report.tool_use_tokens > 0

    def test_api_tokens_used_when_available(self):
        entries = [
            _make_user_line(),
            _make_assistant_line(
                usage={
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 5,
                }
            ),
        ]
        report = generate_health_report(entries)
        assert report.estimated_total_tokens == 165  # 100 + 50 + 10 + 5

    def test_fallback_to_estimated_tokens(self):
        entries = [
            _make_user_line(content="Hello world"),
            _make_assistant_line(content=[{"type": "text", "text": "Goodbye world"}]),
        ]
        report = generate_health_report(entries)
        assert report.estimated_total_tokens > 0
        assert report.estimated_total_tokens == (
            report.reasoning_tokens + report.tool_use_tokens + report.generation_tokens
        )

    def test_non_dict_entry_produces_warning(self):
        entries = ["not a dict", _make_user_line(), _make_assistant_line()]
        report = generate_health_report(entries)
        assert len(report.parsing_warnings) == 1
        assert "Non-dict" in report.parsing_warnings[0]

    def test_meta_lines_skipped(self):
        entries = [
            {"type": "system", "data": "init"},
            {"type": "summary", "text": "blah"},
            _make_user_line(),
            _make_assistant_line(),
        ]
        report = generate_health_report(entries)
        assert report.user_message_count == 1
        assert report.assistant_message_count == 1

    def test_estimated_analysis_seconds(self):
        entries = [
            _make_user_line(content="Hello"),
            _make_assistant_line(
                content=[
                    {"type": "text", "text": "A"},
                    {"type": "text", "text": "B"},
                    {"type": "text", "text": "C"},
                ]
            ),
        ]
        report = generate_health_report(entries)
        # 1 span for user string + 3 spans for assistant blocks = 4 spans
        # 4 * 0.0005 = 0.002
        assert report.estimated_analysis_seconds == pytest.approx(0.002)

    def test_string_content_counted_as_text(self):
        entries = [
            _make_user_line(content="Plain text user message"),
            _make_assistant_line(),
        ]
        report = generate_health_report(entries)
        assert report.content_distribution.text_count >= 1
        assert report.generation_tokens > 0


# ---------------------------------------------------------------------------
# 6. ContentDistribution properties
# ---------------------------------------------------------------------------


class TestContentDistribution:
    """Tests for ContentDistribution percentage calculations."""

    def test_total(self):
        cd = ContentDistribution(
            text_count=5,
            tool_use_count=3,
            tool_result_count=3,
            thinking_count=2,
            other_count=1,
        )
        assert cd.total == 14

    def test_percentages(self):
        cd = ContentDistribution(
            text_count=50,
            tool_use_count=25,
            tool_result_count=15,
            thinking_count=10,
            other_count=0,
        )
        assert cd.text_pct == 50.0
        assert cd.tool_use_pct == 25.0
        assert cd.tool_result_pct == 15.0
        assert cd.thinking_pct == 10.0
        assert cd.other_pct == 0.0

    def test_zero_total_returns_zero_pct(self):
        cd = ContentDistribution()
        assert cd.total == 0
        assert cd.text_pct == 0.0
        assert cd.tool_use_pct == 0.0
        assert cd.tool_result_pct == 0.0
        assert cd.thinking_pct == 0.0
        assert cd.other_pct == 0.0


# ---------------------------------------------------------------------------
# 7. Dataclass defaults
# ---------------------------------------------------------------------------


class TestDataclassDefaults:
    """Tests for dataclass default values."""

    def test_validation_result_defaults(self):
        vr = ValidationResult(valid=True)
        assert vr.errors == []
        assert vr.warnings == []
        assert vr.line_number == 0

    def test_session_validation_result_defaults(self):
        svr = SessionValidationResult(valid=True)
        assert svr.errors == []
        assert svr.warnings == []
        assert svr.message_count == 0
        assert svr.content_block_count == 0

    def test_file_validation_result_defaults(self):
        fvr = FileValidationResult(valid=True, total_lines=0, valid_lines=0)
        assert fvr.error_lines == []
        assert fvr.errors == []
        assert fvr.warnings == []

    def test_health_report_defaults(self):
        hr = HealthReport(
            user_message_count=0,
            assistant_message_count=0,
            estimated_total_tokens=0,
            content_distribution=ContentDistribution(),
            reasoning_tokens=0,
            tool_use_tokens=0,
            generation_tokens=0,
        )
        assert hr.parsing_warnings == []
        assert hr.estimated_analysis_seconds == 0.0

    def test_completeness_assessment_defaults(self):
        ca = CompletenessAssessment(is_complete=True, completeness_score=1.0)
        assert ca.issues == []
