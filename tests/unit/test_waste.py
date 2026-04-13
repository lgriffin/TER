"""Tests for waste pattern detection."""

import numpy as np
import pytest

from ter_calculator.waste import (
    detect_reasoning_loops,
    detect_duplicate_tool_calls,
    detect_context_restatement,
    detect_repetitive_reads,
    detect_edit_fragmentation,
    detect_bash_antipatterns,
    detect_failed_tool_retries,
    detect_repeated_commands,
    detect_waste_patterns,
)
from ter_calculator.models import (
    ClassifiedSpan,
    ContentBlock,
    Message,
    Session,
    SpanLabel,
    SpanPhase,
    TokenSpan,
)


def _make_cs(
    phase: SpanPhase,
    label: SpanLabel,
    text: str = "test",
    position: int = 0,
    token_count: int = 50,
    embedding: np.ndarray | None = None,
    block_type: str = "",
) -> ClassifiedSpan:
    # Default block_type based on phase for convenience.
    if not block_type:
        if phase == SpanPhase.TOOL_USE:
            block_type = "tool_use"
        elif phase == SpanPhase.REASONING:
            block_type = "thinking"
        else:
            block_type = "text"
    span = TokenSpan(
        text=text,
        phase=phase,
        position=position,
        token_count=token_count,
        source_message_uuid="msg-1",
        block_type=block_type,
        embedding=embedding,
    )
    return ClassifiedSpan(
        span=span,
        label=label,
        confidence=0.9,
        cosine_similarity=0.5,
    )


class TestReasoningLoops:
    def test_detects_three_consecutive(self):
        spans = [
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=i)
            for i in range(3)
        ]
        patterns = detect_reasoning_loops(spans)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "reasoning_loop"
        assert patterns[0].spans_involved == 3

    def test_no_pattern_with_two(self):
        spans = [
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=i)
            for i in range(2)
        ]
        patterns = detect_reasoning_loops(spans)
        assert len(patterns) == 0

    def test_interrupted_sequence(self):
        spans = [
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=0),
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=1),
            _make_cs(SpanPhase.REASONING, SpanLabel.ALIGNED_REASONING, position=2),
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=3),
        ]
        patterns = detect_reasoning_loops(spans)
        assert len(patterns) == 0

    def test_tokens_wasted_sum(self):
        spans = [
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING,
                     position=i, token_count=100)
            for i in range(4)
        ]
        patterns = detect_reasoning_loops(spans)
        assert patterns[0].tokens_wasted == 400


class TestDuplicateToolCalls:
    def test_detects_duplicate(self):
        spans = [
            _make_cs(SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
                     text="Bash ls -la", position=0),
            _make_cs(SpanPhase.TOOL_USE, SpanLabel.UNNECESSARY_TOOL_CALL,
                     text="Bash ls -la", position=1),
        ]
        patterns = detect_duplicate_tool_calls(spans)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "duplicate_tool_call"

    def test_no_duplicate_different_calls(self):
        spans = [
            _make_cs(SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
                     text="Bash ls -la", position=0),
            _make_cs(SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
                     text="Read file.py", position=1),
        ]
        patterns = detect_duplicate_tool_calls(spans)
        assert len(patterns) == 0

    def test_outside_window(self):
        spans = []
        # First tool call at position 0.
        spans.append(_make_cs(
            SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
            text="Bash ls -la", position=0,
        ))
        # Fill window with different calls.
        for i in range(1, 7):
            spans.append(_make_cs(
                SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
                text=f"Bash cmd-{i}", position=i,
            ))
        # Duplicate beyond window.
        spans.append(_make_cs(
            SpanPhase.TOOL_USE, SpanLabel.ALIGNED_TOOL_CALL,
            text="Bash ls -la", position=7,
        ))
        patterns = detect_duplicate_tool_calls(spans, window_size=5)
        assert len(patterns) == 0


class TestContextRestatement:
    def test_detects_restatement(self):
        # Two spans with very similar embeddings.
        emb = np.random.rand(384).astype(np.float32)
        emb_similar = emb + np.random.rand(384).astype(np.float32) * 0.01

        spans = [
            _make_cs(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE,
                     text="first response", position=0, embedding=emb),
            _make_cs(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE,
                     text="similar response", position=1, embedding=emb_similar),
        ]
        patterns = detect_context_restatement(spans, similarity_threshold=0.85)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "context_restatement"

    def test_no_restatement_different_content(self):
        emb1 = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0] + [0.0] * 381, dtype=np.float32)

        spans = [
            _make_cs(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE,
                     text="first response", position=0, embedding=emb1),
            _make_cs(SpanPhase.GENERATION, SpanLabel.ALIGNED_RESPONSE,
                     text="different response", position=1, embedding=emb2),
        ]
        patterns = detect_context_restatement(spans, similarity_threshold=0.85)
        assert len(patterns) == 0


def _make_session(messages: list[Message]) -> Session:
    """Helper to create a Session with pre-built messages."""
    return Session(
        session_id="test",
        file_path="test.jsonl",
        messages=messages,
    )


def _read_tool_use(tool_use_id: str, file_path: str) -> ContentBlock:
    return ContentBlock(
        block_type="tool_use",
        tool_name="Read",
        tool_use_id=tool_use_id,
        tool_input={"file_path": file_path},
    )


def _tool_result(tool_use_id: str, text: str) -> ContentBlock:
    return ContentBlock(
        block_type="tool_result",
        tool_use_id=tool_use_id,
        text=text,
    )


def _edit_tool_use(file_path: str, content: str = "change") -> ContentBlock:
    return ContentBlock(
        block_type="tool_use",
        tool_name="Edit",
        tool_use_id=f"edit-{file_path}",
        tool_input={"file_path": file_path, "old_string": "x", "new_string": "y"},
        text=content,
    )


class TestRepetitiveReads:
    def test_detects_three_reads(self):
        file_content = "x" * 400  # ~100 tokens
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _read_tool_use("r1", "/src/app.py"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("r1", file_content),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _read_tool_use("r2", "/src/app.py"),
            ]),
            Message(uuid="u2", role="user", content_blocks=[
                _tool_result("r2", file_content),
            ]),
            Message(uuid="a3", role="assistant", content_blocks=[
                _read_tool_use("r3", "/src/app.py"),
            ]),
            Message(uuid="u3", role="user", content_blocks=[
                _tool_result("r3", file_content),
            ]),
        ])
        patterns = detect_repetitive_reads(session)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "repetitive_read"
        assert patterns[0].details["read_count"] == 3
        # First read is needed, 2nd and 3rd are waste (~100 tokens each)
        assert patterns[0].tokens_wasted == 200

    def test_two_reads_not_detected(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _read_tool_use("r1", "/src/app.py"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("r1", "content"),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _read_tool_use("r2", "/src/app.py"),
            ]),
            Message(uuid="u2", role="user", content_blocks=[
                _tool_result("r2", "content"),
            ]),
        ])
        patterns = detect_repetitive_reads(session)
        assert len(patterns) == 0

    def test_different_files_not_grouped(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _read_tool_use("r1", "/src/app.py"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("r1", "content a"),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _read_tool_use("r2", "/src/test.py"),
            ]),
            Message(uuid="u2", role="user", content_blocks=[
                _tool_result("r2", "content b"),
            ]),
            Message(uuid="a3", role="assistant", content_blocks=[
                _read_tool_use("r3", "/src/main.py"),
            ]),
            Message(uuid="u3", role="user", content_blocks=[
                _tool_result("r3", "content c"),
            ]),
        ])
        patterns = detect_repetitive_reads(session)
        assert len(patterns) == 0

    def test_token_cost_varies_by_result_size(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _read_tool_use("r1", "/big.py"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("r1", "x" * 4000),  # ~1000 tokens
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _read_tool_use("r2", "/big.py"),
            ]),
            Message(uuid="u2", role="user", content_blocks=[
                _tool_result("r2", "x" * 4000),
            ]),
            Message(uuid="a3", role="assistant", content_blocks=[
                _read_tool_use("r3", "/big.py"),
            ]),
            Message(uuid="u3", role="user", content_blocks=[
                _tool_result("r3", "x" * 4000),
            ]),
        ])
        patterns = detect_repetitive_reads(session)
        assert patterns[0].tokens_wasted == 2000  # 2 redundant reads * 1000 tokens


class TestEditFragmentation:
    def test_detects_three_consecutive(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _edit_tool_use("/src/app.py", "edit one content here"),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _edit_tool_use("/src/app.py", "edit two content here"),
            ]),
            Message(uuid="a3", role="assistant", content_blocks=[
                _edit_tool_use("/src/app.py", "edit three content"),
            ]),
        ])
        patterns = detect_edit_fragmentation(session)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "edit_fragmentation"
        assert patterns[0].details["edit_count"] == 3

    def test_two_consecutive_not_detected(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _edit_tool_use("/src/app.py", "edit one"),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _edit_tool_use("/src/app.py", "edit two"),
            ]),
        ])
        patterns = detect_edit_fragmentation(session)
        assert len(patterns) == 0

    def test_interleaved_files_not_detected(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _edit_tool_use("/src/app.py", "edit a"),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _edit_tool_use("/src/test.py", "edit b"),
            ]),
            Message(uuid="a3", role="assistant", content_blocks=[
                _edit_tool_use("/src/app.py", "edit c"),
            ]),
        ])
        patterns = detect_edit_fragmentation(session)
        assert len(patterns) == 0

    def test_mixed_edit_write(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _edit_tool_use("/src/app.py", "edit content"),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                ContentBlock(
                    block_type="tool_use", tool_name="Write",
                    tool_input={"file_path": "/src/app.py", "content": "full write"},
                    text="full file write content here",
                ),
            ]),
            Message(uuid="a3", role="assistant", content_blocks=[
                _edit_tool_use("/src/app.py", "another edit"),
            ]),
        ])
        patterns = detect_edit_fragmentation(session)
        assert len(patterns) == 1
        assert patterns[0].details["edit_count"] == 3

    def test_empty_session(self):
        session = _make_session([])
        patterns = detect_edit_fragmentation(session)
        assert len(patterns) == 0


class TestDetectWastePatterns:
    def test_combined_detection(self):
        spans = [
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=0),
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=1),
            _make_cs(SpanPhase.REASONING, SpanLabel.REDUNDANT_REASONING, position=2),
        ]
        patterns = detect_waste_patterns(spans)
        assert len(patterns) >= 1

    def test_empty_spans(self):
        patterns = detect_waste_patterns([])
        assert patterns == []

    def test_session_patterns_included(self):
        """When session is provided, session-level patterns are detected."""
        file_content = "x" * 400
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _read_tool_use("r1", "/src/app.py"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("r1", file_content),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _read_tool_use("r2", "/src/app.py"),
            ]),
            Message(uuid="u2", role="user", content_blocks=[
                _tool_result("r2", file_content),
            ]),
            Message(uuid="a3", role="assistant", content_blocks=[
                _read_tool_use("r3", "/src/app.py"),
            ]),
            Message(uuid="u3", role="user", content_blocks=[
                _tool_result("r3", file_content),
            ]),
        ])
        patterns = detect_waste_patterns([], session=session)
        rep_patterns = [p for p in patterns if p.pattern_type == "repetitive_read"]
        assert len(rep_patterns) == 1


def _bash_tool_use(tool_use_id: str, command: str) -> ContentBlock:
    return ContentBlock(
        block_type="tool_use",
        tool_name="Bash",
        tool_use_id=tool_use_id,
        tool_input={"command": command},
    )


class TestBashAntipatterns:
    def test_detects_cat(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", "cat /src/app.py"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "file contents " * 50),
            ]),
        ])
        patterns = detect_bash_antipatterns(session)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "bash_antipattern"
        assert patterns[0].details["by_tool"]["Read"] == 1

    def test_detects_grep(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", "grep -r 'pattern' /src/"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "match1\nmatch2\n"),
            ]),
        ])
        patterns = detect_bash_antipatterns(session)
        assert len(patterns) == 1
        assert patterns[0].details["by_tool"]["Grep"] == 1

    def test_detects_find(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", 'find . -name "*.py"'),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "./a.py\n./b.py\n"),
            ]),
        ])
        patterns = detect_bash_antipatterns(session)
        assert len(patterns) == 1
        assert patterns[0].details["by_tool"]["Glob"] == 1

    def test_detects_piped_grep(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", "git log | grep error"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "some output"),
            ]),
        ])
        patterns = detect_bash_antipatterns(session)
        assert len(patterns) == 1

    def test_no_antipattern_for_git(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", "git status"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "On branch main"),
            ]),
        ])
        patterns = detect_bash_antipatterns(session)
        assert len(patterns) == 0

    def test_multiple_antipatterns_combined(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", "cat config.yaml"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "key: value"),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _bash_tool_use("b2", "grep TODO /src/app.py"),
            ]),
            Message(uuid="u2", role="user", content_blocks=[
                _tool_result("b2", "# TODO: fix"),
            ]),
        ])
        patterns = detect_bash_antipatterns(session)
        assert len(patterns) == 1
        assert patterns[0].spans_involved == 2
        assert patterns[0].details["by_tool"]["Read"] == 1
        assert patterns[0].details["by_tool"]["Grep"] == 1

    def test_empty_session(self):
        session = _make_session([])
        patterns = detect_bash_antipatterns(session)
        assert len(patterns) == 0


class TestFailedToolRetries:
    def test_detects_error(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", "ruff check ."),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                ContentBlock(
                    block_type="tool_result",
                    tool_use_id="b1",
                    text="Exit code 127\n/usr/bin/bash: ruff: command not found",
                ),
            ]),
        ])
        patterns = detect_failed_tool_retries(session)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "failed_tool_retry"
        assert patterns[0].details["error_count"] == 1

    def test_detects_tool_use_error(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                ContentBlock(
                    block_type="tool_use",
                    tool_name="Write",
                    tool_use_id="w1",
                    tool_input={"file_path": "/src/new.py", "content": "x"},
                ),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                ContentBlock(
                    block_type="tool_result",
                    tool_use_id="w1",
                    text="<tool_use_error>File has not been read yet. Read it first before writing to it.</tool_use_error>",
                ),
            ]),
        ])
        patterns = detect_failed_tool_retries(session)
        assert len(patterns) == 1

    def test_no_errors_no_pattern(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", "git status"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "On branch main\nnothing to commit"),
            ]),
        ])
        patterns = detect_failed_tool_retries(session)
        assert len(patterns) == 0

    def test_file_not_found(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _read_tool_use("r1", "/nonexistent.py"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                ContentBlock(
                    block_type="tool_result",
                    tool_use_id="r1",
                    text="File does not exist. Note: your current working directory is /src.",
                ),
            ]),
        ])
        patterns = detect_failed_tool_retries(session)
        assert len(patterns) == 1


class TestRepeatedCommands:
    def test_detects_three_repeats(self):
        cmd = "cd /app && ./gradlew build -x test"
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", cmd),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "BUILD FAILED" + "x" * 400),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _bash_tool_use("b2", cmd),
            ]),
            Message(uuid="u2", role="user", content_blocks=[
                _tool_result("b2", "BUILD FAILED" + "x" * 400),
            ]),
            Message(uuid="a3", role="assistant", content_blocks=[
                _bash_tool_use("b3", cmd),
            ]),
            Message(uuid="u3", role="user", content_blocks=[
                _tool_result("b3", "BUILD SUCCESS"),
            ]),
        ])
        patterns = detect_repeated_commands(session)
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "repeated_command"
        assert patterns[0].details["run_count"] == 3
        # First run is needed, 2nd and 3rd are waste
        assert patterns[0].tokens_wasted > 0

    def test_normalizes_tail_variants(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", "./gradlew build | tail -30"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "output"),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _bash_tool_use("b2", "./gradlew build | tail -50"),
            ]),
            Message(uuid="u2", role="user", content_blocks=[
                _tool_result("b2", "output"),
            ]),
            Message(uuid="a3", role="assistant", content_blocks=[
                _bash_tool_use("b3", "./gradlew build | head -20"),
            ]),
            Message(uuid="u3", role="user", content_blocks=[
                _tool_result("b3", "output"),
            ]),
        ])
        patterns = detect_repeated_commands(session)
        assert len(patterns) == 1
        assert patterns[0].details["run_count"] == 3

    def test_two_repeats_not_detected(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", "pytest -v"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "1 passed"),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _bash_tool_use("b2", "pytest -v"),
            ]),
            Message(uuid="u2", role="user", content_blocks=[
                _tool_result("b2", "1 passed"),
            ]),
        ])
        patterns = detect_repeated_commands(session)
        assert len(patterns) == 0

    def test_different_commands_not_grouped(self):
        session = _make_session([
            Message(uuid="a1", role="assistant", content_blocks=[
                _bash_tool_use("b1", "git status"),
            ]),
            Message(uuid="u1", role="user", content_blocks=[
                _tool_result("b1", "clean"),
            ]),
            Message(uuid="a2", role="assistant", content_blocks=[
                _bash_tool_use("b2", "git diff"),
            ]),
            Message(uuid="u2", role="user", content_blocks=[
                _tool_result("b2", "no diff"),
            ]),
            Message(uuid="a3", role="assistant", content_blocks=[
                _bash_tool_use("b3", "git log"),
            ]),
            Message(uuid="u3", role="user", content_blocks=[
                _tool_result("b3", "commits"),
            ]),
        ])
        patterns = detect_repeated_commands(session)
        assert len(patterns) == 0

    def test_empty_session(self):
        session = _make_session([])
        patterns = detect_repeated_commands(session)
        assert len(patterns) == 0
