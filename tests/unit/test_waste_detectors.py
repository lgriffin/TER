"""Tests for extended waste pattern detectors (waste_detectors.py)."""

import pytest

from ter_calculator.models import (
    ClassifiedSpan,
    SpanLabel,
    SpanPhase,
    TokenSpan,
)
from ter_calculator.waste_detectors import (
    ExtendedWasteType,
    detect_abandoned_approaches,
    detect_all_extended,
    detect_error_retry_spirals,
    detect_over_reading,
    detect_permission_loops,
    detect_verbose_thinking,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_cs(
    phase: SpanPhase,
    text: str = "test",
    position: int = 0,
    token_count: int = 50,
    block_type: str = "",
    label: SpanLabel = SpanLabel.ALIGNED_TOOL_CALL,
) -> ClassifiedSpan:
    """Build a ClassifiedSpan with convenient defaults."""
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
    )
    return ClassifiedSpan(
        span=span,
        label=label,
        confidence=0.9,
        cosine_similarity=0.5,
    )


def _tool_use(text: str, position: int, token_count: int = 50) -> ClassifiedSpan:
    """Shorthand for a tool_use span."""
    return _make_cs(
        SpanPhase.TOOL_USE,
        text=text,
        position=position,
        token_count=token_count,
        block_type="tool_use",
    )


def _tool_result(text: str, position: int, token_count: int = 20) -> ClassifiedSpan:
    """Shorthand for a tool_result span."""
    return _make_cs(
        SpanPhase.TOOL_USE,
        text=text,
        position=position,
        token_count=token_count,
        block_type="tool_result",
    )


def _reasoning(text: str, position: int, token_count: int = 100) -> ClassifiedSpan:
    """Shorthand for a reasoning span."""
    return _make_cs(
        SpanPhase.REASONING,
        text=text,
        position=position,
        token_count=token_count,
        label=SpanLabel.ALIGNED_REASONING,
    )


def _generation(text: str, position: int, token_count: int = 50) -> ClassifiedSpan:
    """Shorthand for a generation span."""
    return _make_cs(
        SpanPhase.GENERATION,
        text=text,
        position=position,
        token_count=token_count,
        label=SpanLabel.ALIGNED_RESPONSE,
    )


# ===================================================================
# 1. detect_permission_loops
# ===================================================================


class TestDetectPermissionLoops:
    def test_empty_input(self):
        assert detect_permission_loops([]) == []

    def test_no_tool_spans_returns_empty(self):
        """Non-tool spans should produce no permission-loop patterns."""
        spans = [
            _reasoning("thinking about it", position=0),
            _generation("some output", position=1),
        ]
        assert detect_permission_loops(spans) == []

    def test_no_permission_issues(self):
        """Tool calls that succeed should produce no patterns."""
        spans = [
            _tool_use('Bash {"command":"ls"}', position=0),
            _tool_result("file1.py file2.py", position=1),
            _tool_use('Bash {"command":"cat file1.py"}', position=2),
            _tool_result("contents...", position=3),
        ]
        assert detect_permission_loops(spans) == []

    def test_detects_permission_loop_default_min_retries(self):
        """Three identical calls with denial results between them = 2 retries."""
        spans = [
            _tool_use('Bash {"command":"rm /etc/passwd"}', position=0),
            _tool_result("permission denied", position=1),
            _tool_use('Bash {"command":"rm /etc/passwd"}', position=2),
            _tool_result("permission denied", position=3),
            _tool_use('Bash {"command":"rm /etc/passwd"}', position=4),
        ]
        patterns = detect_permission_loops(spans)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.pattern_type == ExtendedWasteType.PERMISSION_LOOP.value
        assert p.details["tool_name"] == "Bash"
        assert p.details["retries"] == 2
        assert p.start_position == 0
        assert p.end_position == 4
        assert p.spans_involved == 3
        # Wasted tokens = token_count of the 2 retries (positions 2 and 4)
        assert p.tokens_wasted == 100

    def test_below_min_retries_threshold(self):
        """Only 1 retry (2 calls total) with default min_retries=2 -- not flagged."""
        spans = [
            _tool_use('Bash {"command":"rm /root/x"}', position=0),
            _tool_result("access denied", position=1),
            _tool_use('Bash {"command":"rm /root/x"}', position=2),
        ]
        assert detect_permission_loops(spans) == []

    def test_custom_min_retries_1(self):
        """Lowering min_retries=1 should flag a single retry."""
        spans = [
            _tool_use('Bash {"command":"rm /root/x"}', position=0),
            _tool_result("access denied", position=1),
            _tool_use('Bash {"command":"rm /root/x"}', position=2),
        ]
        patterns = detect_permission_loops(spans, min_retries=1)
        assert len(patterns) == 1
        assert patterns[0].details["retries"] == 1

    def test_high_min_retries_not_flagged(self):
        """Raising min_retries above actual retries prevents detection."""
        spans = [
            _tool_use('Write {"file_path":"/etc/secret"}', position=0),
            _tool_result("permission denied", position=1),
            _tool_use('Write {"file_path":"/etc/secret"}', position=2),
            _tool_result("permission denied", position=3),
            _tool_use('Write {"file_path":"/etc/secret"}', position=4),
        ]
        # 2 retries, but we require 3
        patterns = detect_permission_loops(spans, min_retries=3)
        assert patterns == []

    def test_different_tool_breaks_chain(self):
        """Switching to a different tool between denied calls breaks the chain."""
        spans = [
            _tool_use('Bash {"command":"rm /root/x"}', position=0),
            _tool_result("permission denied", position=1),
            _tool_use('Read {"file_path":"/root/x"}', position=2),
            _tool_result("permission denied", position=3),
            _tool_use('Bash {"command":"rm /root/x"}', position=4),
        ]
        # The chain for Bash is broken by the intervening Read tool_use
        assert detect_permission_loops(spans) == []

    def test_all_permission_keywords(self):
        """Each denial keyword should be recognised (case-insensitive)."""
        for keyword in [
            "permission denied",
            "not allowed",
            "access denied",
            "EACCES: operation not permitted",
            "unauthorized request",
        ]:
            spans = [
                _tool_use('Write {"file_path":"/etc/secret"}', position=0),
                _tool_result(keyword, position=1),
                _tool_use('Write {"file_path":"/etc/secret"}', position=2),
                _tool_result(keyword, position=3),
                _tool_use('Write {"file_path":"/etc/secret"}', position=4),
            ]
            patterns = detect_permission_loops(spans)
            assert len(patterns) == 1, f"Failed for keyword: {keyword}"

    def test_intervening_reasoning_does_not_break_chain(self):
        """Reasoning spans between tool_use spans should not affect detection."""
        spans = [
            _tool_use('Bash {"command":"sudo rm"}', position=0),
            _tool_result("permission denied", position=1),
            _reasoning("Let me try again", position=2),
            _tool_use('Bash {"command":"sudo rm"}', position=3),
            _tool_result("permission denied", position=4),
            _reasoning("Still denied, trying once more", position=5),
            _tool_use('Bash {"command":"sudo rm"}', position=6),
        ]
        patterns = detect_permission_loops(spans)
        assert len(patterns) == 1
        assert patterns[0].details["retries"] == 2

    def test_no_denial_result_between_calls(self):
        """If the result between two identical calls is not a denial, no pattern."""
        spans = [
            _tool_use('Bash {"command":"make"}', position=0),
            _tool_result("build succeeded", position=1),
            _tool_use('Bash {"command":"make"}', position=2),
            _tool_result("build succeeded", position=3),
            _tool_use('Bash {"command":"make"}', position=4),
        ]
        assert detect_permission_loops(spans) == []


# ===================================================================
# 2. detect_error_retry_spirals
# ===================================================================


class TestDetectErrorRetrySpirals:
    def test_empty_input(self):
        assert detect_error_retry_spirals([]) == []

    def test_no_errors(self):
        """Successful tool calls should produce no patterns."""
        spans = [
            _tool_use('Bash {"command":"ls"}', position=0),
            _tool_result("file1.py", position=1),
            _tool_use('Bash {"command":"cat file1.py"}', position=2),
            _tool_result("content", position=3),
        ]
        assert detect_error_retry_spirals(spans) == []

    def test_detects_error_spiral_default_min_3(self):
        """4 identical calls with error results between them = 3 retries."""
        spans = [
            _tool_use('Bash {"command":"python run.py --flag=val"}', position=0),
            _tool_result("error: ModuleNotFoundError", position=1),
            _tool_use('Bash {"command":"python run.py --flag=val"}', position=2),
            _tool_result("error: ModuleNotFoundError", position=3),
            _tool_use('Bash {"command":"python run.py --flag=val"}', position=4),
            _tool_result("error: ModuleNotFoundError", position=5),
            _tool_use('Bash {"command":"python run.py --flag=val"}', position=6),
        ]
        patterns = detect_error_retry_spirals(spans)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.pattern_type == ExtendedWasteType.ERROR_RETRY_SPIRAL.value
        assert p.details["tool_name"] == "Bash"
        assert p.details["retries"] == 3
        assert p.spans_involved == 4

    def test_below_min_retries_threshold(self):
        """2 retries with default min_retries=3 should not be flagged."""
        spans = [
            _tool_use('Bash {"command":"make build"}', position=0),
            _tool_result("error: compilation failed", position=1),
            _tool_use('Bash {"command":"make build"}', position=2),
            _tool_result("error: compilation failed", position=3),
            _tool_use('Bash {"command":"make build"}', position=4),
        ]
        assert detect_error_retry_spirals(spans) == []

    def test_custom_min_retries_lower(self):
        """With min_retries=2, two retries should be flagged."""
        spans = [
            _tool_use('Bash {"command":"make build"}', position=0),
            _tool_result("error: compilation failed", position=1),
            _tool_use('Bash {"command":"make build"}', position=2),
            _tool_result("error: compilation failed", position=3),
            _tool_use('Bash {"command":"make build"}', position=4),
        ]
        patterns = detect_error_retry_spirals(spans, min_retries=2)
        assert len(patterns) == 1
        assert patterns[0].details["retries"] == 2

    def test_significantly_different_params_break_chain(self):
        """Completely different params should have low similarity and break chain."""
        spans = [
            _tool_use(
                'Bash {"command":"python run.py --mode=fast --verbose"}',
                position=0,
            ),
            _tool_result("error: failed to parse", position=1),
            _tool_use(
                'Bash {"command":"node server.js --port=3000 --host=localhost"}',
                position=2,
            ),
            _tool_result("error: failed to start", position=3),
            _tool_use(
                'Bash {"command":"cargo build --release --target=x86_64"}',
                position=4,
            ),
            _tool_result("error: missing dependency", position=5),
            _tool_use(
                'Bash {"command":"go run main.go --config=/etc/app.yaml"}',
                position=6,
            ),
        ]
        # Same tool name but very different params -> low similarity -> no chain
        patterns = detect_error_retry_spirals(spans, min_retries=2)
        assert len(patterns) == 0

    def test_different_tool_breaks_chain(self):
        """Switching tool names should break the chain."""
        spans = [
            _tool_use('Bash {"command":"ls"}', position=0),
            _tool_result("error: no such file", position=1),
            _tool_use('Read {"file_path":"x.py"}', position=2),
            _tool_result("error: file not found", position=3),
            _tool_use('Bash {"command":"ls"}', position=4),
        ]
        patterns = detect_error_retry_spirals(spans, min_retries=1)
        assert len(patterns) == 0

    def test_error_keywords_case_insensitive(self):
        """Error keywords are matched case-insensitively."""
        for keyword in [
            "Error occurred",
            "FAILED to execute",
            "Exception raised",
            "Traceback (most recent call last)",
        ]:
            spans = [
                _tool_use('Bash {"command":"test"}', position=0),
                _tool_result(keyword, position=1),
                _tool_use('Bash {"command":"test"}', position=2),
                _tool_result(keyword, position=3),
                _tool_use('Bash {"command":"test"}', position=4),
                _tool_result(keyword, position=5),
                _tool_use('Bash {"command":"test"}', position=6),
            ]
            patterns = detect_error_retry_spirals(spans)
            assert len(patterns) == 1, f"Failed for keyword: {keyword}"

    def test_custom_similarity_threshold(self):
        """A lower similarity threshold allows more variation in params."""
        # Params vary slightly each time
        spans = [
            _tool_use('Bash {"command":"python test.py --flag=a"}', position=0),
            _tool_result("error: test failed", position=1),
            _tool_use('Bash {"command":"python test.py --flag=b"}', position=2),
            _tool_result("error: test failed", position=3),
            _tool_use('Bash {"command":"python test.py --flag=c"}', position=4),
            _tool_result("error: test failed", position=5),
            _tool_use('Bash {"command":"python test.py --flag=d"}', position=6),
        ]
        # With a very strict threshold these may not chain; with relaxed they will
        patterns_strict = detect_error_retry_spirals(spans, similarity_threshold=0.99)
        patterns_relaxed = detect_error_retry_spirals(spans, similarity_threshold=0.50)
        # Relaxed should find at least as many patterns as strict
        assert len(patterns_relaxed) >= len(patterns_strict)

    def test_wasted_tokens_excludes_first_call(self):
        """tokens_wasted should only count retry calls, not the original."""
        spans = [
            _tool_use('Bash {"command":"test"}', position=0, token_count=100),
            _tool_result("error: fail", position=1),
            _tool_use('Bash {"command":"test"}', position=2, token_count=100),
            _tool_result("error: fail", position=3),
            _tool_use('Bash {"command":"test"}', position=4, token_count=100),
            _tool_result("error: fail", position=5),
            _tool_use('Bash {"command":"test"}', position=6, token_count=100),
        ]
        patterns = detect_error_retry_spirals(spans)
        assert len(patterns) == 1
        # 3 retries x 100 tokens = 300 wasted (first call excluded)
        assert patterns[0].tokens_wasted == 300


# ===================================================================
# 3. detect_over_reading
# ===================================================================


class TestDetectOverReading:
    def test_empty_input(self):
        assert detect_over_reading([]) == []

    def test_single_read_no_pattern(self):
        """A single read should never produce a pattern."""
        spans = [
            _tool_use('Read {"file_path":"src/main.py"}', position=0),
            _tool_result("def main(): pass", position=1),
        ]
        assert detect_over_reading(spans) == []

    def test_two_reads_no_pattern_default_min(self):
        """Two reads total = 1 redundant read, below default min_reads=2."""
        spans = [
            _tool_use('Read {"file_path":"src/main.py"}', position=0),
            _tool_result("content", position=1),
            _tool_use('Read {"file_path":"src/main.py"}', position=2),
            _tool_result("content", position=3),
        ]
        assert detect_over_reading(spans) == []

    def test_three_reads_detected(self):
        """Three reads of the same file = 2 redundant, triggers default min_reads=2."""
        spans = [
            _tool_use('Read {"file_path":"src/main.py"}', position=0),
            _tool_result("content", position=1),
            _tool_use('Read {"file_path":"src/main.py"}', position=2),
            _tool_result("content", position=3),
            _tool_use('Read {"file_path":"src/main.py"}', position=4),
            _tool_result("content", position=5),
        ]
        patterns = detect_over_reading(spans)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.pattern_type == ExtendedWasteType.OVER_READING.value
        assert p.details["file_path"] == "src/main.py"
        assert p.details["read_count"] == 3
        assert p.details["redundant_reads"] == 2
        # Wasted tokens = token_count of the 2 redundant reads
        assert p.tokens_wasted == 100  # 2 x 50

    def test_edit_resets_read_count(self):
        """An intervening Edit to the same file resets the read chain."""
        spans = [
            _tool_use('Read {"file_path":"src/main.py"}', position=0),
            _tool_result("content", position=1),
            _tool_use('Read {"file_path":"src/main.py"}', position=2),
            _tool_result("content", position=3),
            _tool_use('Edit {"file_path":"src/main.py"}', position=4),
            _tool_result("ok", position=5),
            _tool_use('Read {"file_path":"src/main.py"}', position=6),
            _tool_result("content", position=7),
        ]
        # After Edit at position 4, tracker resets. Only 1 read post-edit.
        assert detect_over_reading(spans) == []

    def test_write_resets_read_count(self):
        """Write tool also resets the read chain."""
        spans = [
            _tool_use('Read {"file_path":"x.py"}', position=0),
            _tool_use('Read {"file_path":"x.py"}', position=1),
            _tool_use('Read {"file_path":"x.py"}', position=2),
            _tool_use('Write {"file_path":"x.py"}', position=3),
            _tool_use('Read {"file_path":"x.py"}', position=4),
        ]
        # Write at position 3 resets; only 1 read afterwards
        assert detect_over_reading(spans) == []

    def test_different_files_tracked_independently(self):
        """Reads of different files should be tracked separately."""
        spans = [
            _tool_use('Read {"file_path":"a.py"}', position=0),
            _tool_use('Read {"file_path":"b.py"}', position=1),
            _tool_use('Read {"file_path":"a.py"}', position=2),
            _tool_use('Read {"file_path":"b.py"}', position=3),
        ]
        # Each file read twice = 1 redundant read each, below min_reads=2
        assert detect_over_reading(spans) == []

    def test_custom_min_reads_1(self):
        """Lowering min_reads=1 flags files read just twice."""
        spans = [
            _tool_use('Read {"file_path":"src/main.py"}', position=0),
            _tool_result("content", position=1),
            _tool_use('Read {"file_path":"src/main.py"}', position=2),
            _tool_result("content", position=3),
        ]
        patterns = detect_over_reading(spans, min_reads=1)
        assert len(patterns) == 1
        assert patterns[0].details["redundant_reads"] == 1

    def test_cat_tool_recognised_as_read(self):
        """The 'cat' tool name should also be treated as a read."""
        spans = [
            _tool_use('cat {"file_path":"src/main.py"}', position=0),
            _tool_use('cat {"file_path":"src/main.py"}', position=1),
            _tool_use('cat {"file_path":"src/main.py"}', position=2),
        ]
        patterns = detect_over_reading(spans)
        assert len(patterns) == 1

    def test_results_sorted_by_wasted_tokens_descending(self):
        """Multiple over-read files should be sorted by tokens_wasted descending."""
        spans = [
            # a.py read 3 times at 50 tokens each => 100 wasted
            _tool_use('Read {"file_path":"a.py"}', position=0, token_count=50),
            _tool_use('Read {"file_path":"a.py"}', position=1, token_count=50),
            _tool_use('Read {"file_path":"a.py"}', position=2, token_count=50),
            # b.py read 3 times at 200 tokens each => 400 wasted
            _tool_use('Read {"file_path":"b.py"}', position=3, token_count=200),
            _tool_use('Read {"file_path":"b.py"}', position=4, token_count=200),
            _tool_use('Read {"file_path":"b.py"}', position=5, token_count=200),
        ]
        patterns = detect_over_reading(spans)
        assert len(patterns) == 2
        assert patterns[0].details["file_path"] == "b.py"
        assert patterns[1].details["file_path"] == "a.py"

    def test_path_key_fallback(self):
        """When 'file_path' is absent, 'path' key should be used."""
        spans = [
            _tool_use('Read {"path":"src/utils.py"}', position=0),
            _tool_use('Read {"path":"src/utils.py"}', position=1),
            _tool_use('Read {"path":"src/utils.py"}', position=2),
        ]
        patterns = detect_over_reading(spans)
        assert len(patterns) == 1
        assert patterns[0].details["file_path"] == "src/utils.py"

    def test_no_file_path_spans_ignored(self):
        """Tool calls without parseable file paths are skipped."""
        spans = [
            _tool_use("Bash {}", position=0),
            _tool_use("Bash {}", position=1),
            _tool_use("Bash {}", position=2),
        ]
        assert detect_over_reading(spans) == []

    def test_non_tool_use_spans_ignored(self):
        """Reasoning / generation spans should not affect over-reading detection."""
        spans = [
            _tool_use('Read {"file_path":"x.py"}', position=0),
            _reasoning("thinking", position=1),
            _tool_use('Read {"file_path":"x.py"}', position=2),
            _generation("output", position=3),
            _tool_use('Read {"file_path":"x.py"}', position=4),
        ]
        patterns = detect_over_reading(spans)
        assert len(patterns) == 1


# ===================================================================
# 4. detect_abandoned_approaches
# ===================================================================


class TestDetectAbandonedApproaches:
    def test_empty_input(self):
        assert detect_abandoned_approaches([]) == []

    def test_no_abandonment_when_file_revisited(self):
        """File edited and then touched again later -- not abandoned."""
        spans = [
            _tool_use('Edit {"file_path":"src/a.py"}', position=0),
            _tool_result("ok", position=1),
            _tool_use('Read {"file_path":"src/b.py"}', position=2),
            _tool_result("content", position=3),
            _tool_use('Read {"file_path":"src/a.py"}', position=4),
            _tool_result("content", position=5),
        ]
        assert detect_abandoned_approaches(spans) == []

    def test_detects_abandoned_file(self):
        """File edited, then agent moves to different file and never returns."""
        spans = [
            _tool_use(
                'Edit {"file_path":"src/attempt1.py"}',
                position=0,
                token_count=80,
            ),
            _tool_result("ok", position=1),
            _tool_use(
                'Edit {"file_path":"src/attempt2.py"}',
                position=2,
                token_count=60,
            ),
            _tool_result("ok", position=3),
        ]
        patterns = detect_abandoned_approaches(spans)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.pattern_type == ExtendedWasteType.ABANDONED_APPROACH.value
        assert p.details["file_path"] == "src/attempt1.py"
        assert "attempt1.py" in p.description

    def test_last_file_not_abandoned(self):
        """The last file touched should not be flagged (no subsequent work)."""
        spans = [
            _tool_use('Edit {"file_path":"src/only.py"}', position=0),
            _tool_result("ok", position=1),
        ]
        assert detect_abandoned_approaches(spans) == []

    def test_file_revisited_later_not_abandoned(self):
        """If file is touched again after other work, it is not abandoned."""
        spans = [
            _tool_use('Edit {"file_path":"src/a.py"}', position=0),
            _tool_use('Edit {"file_path":"src/b.py"}', position=2),
            _tool_use('Edit {"file_path":"src/a.py"}', position=4),
        ]
        # a.py revisited at position 4, so not abandoned.
        # b.py: last touch is 2, agent works on a.py at 4 -> b.py abandoned.
        patterns = detect_abandoned_approaches(spans)
        assert any(p.details["file_path"] == "src/b.py" for p in patterns)
        assert not any(p.details["file_path"] == "src/a.py" for p in patterns)

    def test_write_tool_also_counts(self):
        """Write tool should be recognised same as Edit for abandonment."""
        spans = [
            _tool_use('Write {"file_path":"src/temp.py"}', position=0),
            _tool_result("ok", position=1),
            _tool_use('Edit {"file_path":"src/main.py"}', position=2),
            _tool_result("ok", position=3),
        ]
        patterns = detect_abandoned_approaches(spans)
        assert len(patterns) == 1
        assert patterns[0].details["file_path"] == "src/temp.py"

    def test_only_reads_no_abandonment(self):
        """Reading files (not editing) should not produce abandoned-approach patterns."""
        spans = [
            _tool_use('Read {"file_path":"src/a.py"}', position=0),
            _tool_result("content", position=1),
            _tool_use('Read {"file_path":"src/b.py"}', position=2),
            _tool_result("content", position=3),
        ]
        assert detect_abandoned_approaches(spans) == []

    def test_no_file_path_spans_ignored(self):
        """Tool calls without parseable file paths should be skipped."""
        spans = [
            _tool_use("Bash {}", position=0),
            _tool_result("ok", position=1),
        ]
        assert detect_abandoned_approaches(spans) == []

    def test_multiple_abandoned_files(self):
        """Multiple files can be flagged as abandoned."""
        spans = [
            _tool_use('Edit {"file_path":"src/a.py"}', position=0, token_count=100),
            _tool_use('Edit {"file_path":"src/b.py"}', position=1, token_count=200),
            _tool_use('Edit {"file_path":"src/final.py"}', position=2, token_count=50),
        ]
        patterns = detect_abandoned_approaches(spans)
        abandoned_files = {p.details["file_path"] for p in patterns}
        assert "src/a.py" in abandoned_files
        assert "src/b.py" in abandoned_files
        # final.py is the last file -- not abandoned
        assert "src/final.py" not in abandoned_files

    def test_results_sorted_by_wasted_tokens_descending(self):
        """Patterns should be sorted by tokens_wasted descending."""
        spans = [
            _tool_use('Edit {"file_path":"src/small.py"}', position=0, token_count=50),
            _tool_use('Edit {"file_path":"src/large.py"}', position=1, token_count=500),
            _tool_use('Edit {"file_path":"src/final.py"}', position=2, token_count=10),
        ]
        patterns = detect_abandoned_approaches(spans)
        assert len(patterns) == 2
        assert patterns[0].tokens_wasted >= patterns[1].tokens_wasted

    def test_duplicate_file_not_reported_twice(self):
        """Same file edited multiple times then abandoned should only appear once."""
        spans = [
            _tool_use('Edit {"file_path":"src/dup.py"}', position=0, token_count=100),
            _tool_use('Edit {"file_path":"src/dup.py"}', position=1, token_count=100),
            _tool_use('Edit {"file_path":"src/other.py"}', position=2, token_count=50),
        ]
        patterns = detect_abandoned_approaches(spans)
        dup_patterns = [p for p in patterns if p.details["file_path"] == "src/dup.py"]
        assert len(dup_patterns) == 1


# ===================================================================
# 5. detect_verbose_thinking
# ===================================================================


class TestDetectVerboseThinking:
    def test_empty_input(self):
        assert detect_verbose_thinking([]) == []

    def test_no_thinking_spans(self):
        """Non-reasoning spans should produce no patterns."""
        spans = [
            _generation("output text", position=0),
            _tool_use('Bash {"command":"ls"}', position=1),
        ]
        assert detect_verbose_thinking(spans) == []

    def test_proportional_thinking_not_flagged(self):
        """A reasonable thinking-to-action ratio should not be flagged."""
        spans = [
            _reasoning("Let me think about this...", position=0, token_count=200),
            _tool_use('Bash {"command":"ls"}', position=1, token_count=50),
        ]
        # ratio = 200/50 = 4.0 < default 10.0
        assert detect_verbose_thinking(spans) == []

    def test_below_min_thinking_tokens_not_flagged(self):
        """High ratio should not flag when thinking tokens < min_thinking_tokens."""
        spans = [
            _reasoning("Short thought", position=0, token_count=100),
            _tool_use('Bash {"command":"ls"}', position=1, token_count=5),
        ]
        # ratio = 100/5 = 20.0 > 10.0 but 100 < 500 default min
        assert detect_verbose_thinking(spans) == []

    def test_detects_verbose_thinking(self):
        """Large thinking block with small action should be flagged."""
        spans = [
            _reasoning("Very long reasoning...", position=0, token_count=6000),
            _tool_use('Bash {"command":"ls"}', position=1, token_count=50),
        ]
        # ratio = 6000/50 = 120.0 > 10.0, and 6000 > 500
        patterns = detect_verbose_thinking(spans)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.pattern_type == ExtendedWasteType.VERBOSE_THINKING.value
        assert p.details["thinking_tokens"] == 6000
        assert p.details["action_tokens"] == 50
        assert p.details["ratio"] == 120.0
        assert p.start_position == 0
        assert p.end_position == 1
        # Excess = 6000 - (50 * 10) = 5500
        assert p.tokens_wasted == 5500

    def test_thinking_with_no_subsequent_action(self):
        """Thinking block at end of session with no action is flagged."""
        spans = [
            _reasoning("Final rumination...", position=0, token_count=1000),
        ]
        patterns = detect_verbose_thinking(spans)
        assert len(patterns) == 1
        p = patterns[0]
        assert p.details["action_tokens"] == 0
        assert p.tokens_wasted == 1000

    def test_thinking_followed_by_zero_token_action(self):
        """Action with 0 tokens should produce infinite ratio and be flagged."""
        spans = [
            _reasoning("Thinking...", position=0, token_count=600),
            _tool_use('Bash {"command":""}', position=1, token_count=0),
        ]
        patterns = detect_verbose_thinking(spans)
        assert len(patterns) == 1
        assert patterns[0].details["ratio"] == float("inf")

    def test_custom_ratio_threshold(self):
        """Custom ratio_threshold should change what gets flagged."""
        spans = [
            _reasoning("Moderate thinking", position=0, token_count=600),
            _tool_use('Bash {"command":"ls"}', position=1, token_count=50),
        ]
        # ratio = 12.0
        assert detect_verbose_thinking(spans, ratio_threshold=15.0) == []
        patterns = detect_verbose_thinking(spans, ratio_threshold=5.0)
        assert len(patterns) == 1

    def test_custom_min_thinking_tokens(self):
        """Custom min_thinking_tokens should change what gets flagged."""
        spans = [
            _reasoning("Some thinking", position=0, token_count=200),
            _tool_use('Bash {"command":"ls"}', position=1, token_count=10),
        ]
        # ratio = 20.0 > 10.0, but 200 < 500 default -> not flagged
        assert detect_verbose_thinking(spans) == []
        # Lower min to 100
        patterns = detect_verbose_thinking(spans, min_thinking_tokens=100)
        assert len(patterns) == 1

    def test_skips_reasoning_to_find_next_action(self):
        """The detector looks past consecutive reasoning spans for the action."""
        spans = [
            _reasoning("First thought", position=0, token_count=2000),
            _reasoning("Second thought", position=1, token_count=1000),
            _tool_use('Bash {"command":"ls"}', position=2, token_count=50),
        ]
        # First reasoning: next non-reasoning = tool_use at position 2
        #   ratio = 2000/50 = 40.0
        # Second reasoning: next non-reasoning = tool_use at position 2
        #   ratio = 1000/50 = 20.0
        patterns = detect_verbose_thinking(spans)
        assert len(patterns) == 2

    def test_generation_as_action(self):
        """A generation span is a valid action target."""
        spans = [
            _reasoning("Deep thinking...", position=0, token_count=600),
            _generation("Here is the answer", position=1, token_count=50),
        ]
        # ratio = 600/50 = 12.0 > 10.0
        patterns = detect_verbose_thinking(spans)
        assert len(patterns) == 1
        assert patterns[0].details["action_tokens"] == 50

    def test_excess_calculation(self):
        """tokens_wasted should be thinking_tokens - (action_tokens * threshold)."""
        spans = [
            _reasoning("Long thought", position=0, token_count=1000),
            _tool_use("Bash {}", position=1, token_count=50),
        ]
        # ratio = 1000/50 = 20.0, excess = 1000 - (50 * 10) = 500
        patterns = detect_verbose_thinking(spans)
        assert len(patterns) == 1
        assert patterns[0].tokens_wasted == 500


# ===================================================================
# 6. detect_all_extended
# ===================================================================


class TestDetectAllExtended:
    def test_empty_input(self):
        assert detect_all_extended([]) == []

    def test_returns_list(self):
        """Even with no detectable patterns the return type should be list."""
        spans = [
            _reasoning("thinking", position=0),
            _generation("output", position=1),
        ]
        result = detect_all_extended(spans)
        assert isinstance(result, list)

    def test_combines_multiple_detectors(self):
        """detect_all_extended should run all five detectors and combine results."""
        spans = [
            # Permission loop: 3 calls with denial
            _tool_use('Bash {"command":"rm /root"}', position=0, token_count=30),
            _tool_result("permission denied", position=1),
            _tool_use('Bash {"command":"rm /root"}', position=2, token_count=30),
            _tool_result("permission denied", position=3),
            _tool_use('Bash {"command":"rm /root"}', position=4, token_count=30),
            # Over-reading: same file read 3 times
            _tool_use('Read {"file_path":"config.yaml"}', position=10, token_count=40),
            _tool_use('Read {"file_path":"config.yaml"}', position=11, token_count=40),
            _tool_use('Read {"file_path":"config.yaml"}', position=12, token_count=40),
            # Verbose thinking
            _reasoning("Lots of thinking...", position=20, token_count=5000),
            _tool_use('Bash {"command":"echo hi"}', position=21, token_count=10),
        ]

        patterns = detect_all_extended(spans)
        types_found = {p.pattern_type for p in patterns}
        assert ExtendedWasteType.PERMISSION_LOOP.value in types_found
        assert ExtendedWasteType.OVER_READING.value in types_found
        assert ExtendedWasteType.VERBOSE_THINKING.value in types_found

    def test_sorted_by_start_position(self):
        """Results from detect_all_extended should be sorted by start_position."""
        spans = [
            # Verbose thinking at position 20
            _reasoning("Lots of thinking...", position=20, token_count=5000),
            _tool_use('Bash {"command":"echo hi"}', position=21, token_count=10),
            # Permission loop at position 0
            _tool_use('Bash {"command":"rm /root"}', position=0, token_count=30),
            _tool_result("permission denied", position=1),
            _tool_use('Bash {"command":"rm /root"}', position=2, token_count=30),
            _tool_result("permission denied", position=3),
            _tool_use('Bash {"command":"rm /root"}', position=4, token_count=30),
        ]
        patterns = detect_all_extended(spans)
        positions = [p.start_position for p in patterns]
        assert positions == sorted(positions)

    def test_forwards_permission_min_retries(self):
        """permission_min_retries parameter should be forwarded."""
        spans = [
            _tool_use('Bash {"command":"rm /root"}', position=0, token_count=30),
            _tool_result("permission denied", position=1),
            _tool_use('Bash {"command":"rm /root"}', position=2, token_count=30),
            _tool_result("permission denied", position=3),
            _tool_use('Bash {"command":"rm /root"}', position=4, token_count=30),
        ]
        # Default min_retries=2 triggers (2 retries)
        patterns_default = detect_all_extended(spans)
        assert any(
            p.pattern_type == ExtendedWasteType.PERMISSION_LOOP.value
            for p in patterns_default
        )

        # Raising to 3 prevents detection
        patterns_strict = detect_all_extended(spans, permission_min_retries=3)
        assert not any(
            p.pattern_type == ExtendedWasteType.PERMISSION_LOOP.value
            for p in patterns_strict
        )

    def test_forwards_verbose_thinking_params(self):
        """verbose_ratio_threshold and verbose_min_thinking_tokens forwarded."""
        spans = [
            _reasoning("Thinking...", position=0, token_count=300),
            _tool_use("Bash {}", position=1, token_count=10),
        ]
        # Default: 300 < 500 min_thinking_tokens -> not flagged
        patterns_default = detect_all_extended(spans)
        assert not any(
            p.pattern_type == ExtendedWasteType.VERBOSE_THINKING.value
            for p in patterns_default
        )

        # Lower min_thinking_tokens to 100
        patterns_low_min = detect_all_extended(spans, verbose_min_thinking_tokens=100)
        assert any(
            p.pattern_type == ExtendedWasteType.VERBOSE_THINKING.value
            for p in patterns_low_min
        )

    def test_forwards_over_reading_min_reads(self):
        """over_reading_min_reads parameter should be forwarded."""
        spans = [
            _tool_use('Read {"file_path":"x.py"}', position=0),
            _tool_use('Read {"file_path":"x.py"}', position=1),
        ]
        # Default min_reads=2 requires 3 reads total -> not flagged
        patterns_default = detect_all_extended(spans)
        assert not any(
            p.pattern_type == ExtendedWasteType.OVER_READING.value
            for p in patterns_default
        )

        # Lower to min_reads=1 -> 2 reads total qualifies
        patterns_low = detect_all_extended(spans, over_reading_min_reads=1)
        assert any(
            p.pattern_type == ExtendedWasteType.OVER_READING.value for p in patterns_low
        )
