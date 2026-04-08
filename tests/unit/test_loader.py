"""Tests for JSONL session loader and span segmentation."""

import json
import tempfile
from pathlib import Path

import pytest

from ter_calculator.loader import load_session, segment_spans
from ter_calculator.models import SpanPhase


@pytest.fixture
def sample_session_path():
    return str(Path(__file__).parent.parent / "fixtures" / "sample_session.jsonl")


@pytest.fixture
def minimal_session(tmp_path):
    """Create a minimal valid session file."""
    data = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "sessionId": "s1",
            "timestamp": "2026-04-01T10:00:00.000Z",
            "message": {"role": "user", "content": "Hello world"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": "s1",
            "timestamp": "2026-04-01T10:00:01.000Z",
            "requestId": "r1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5,
                          "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0},
            },
        },
    ]
    f = tmp_path / "test.jsonl"
    f.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")
    return str(f)


class TestLoadSession:
    def test_load_sample_session(self, sample_session_path):
        session = load_session(sample_session_path)
        assert session.session_id == "session-test-001"
        assert len(session.messages) > 0
        assert len(session.user_prompts) > 0

    def test_load_minimal_session(self, minimal_session):
        session = load_session(minimal_session)
        assert session.session_id == "s1"
        assert len(session.messages) == 2
        assert session.user_prompts == ["Hello world"]
        assert session.total_tokens == 5

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_session("/nonexistent/file.jsonl")

    def test_invalid_extension(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected .jsonl"):
            load_session(str(f))

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_session(str(f))

    def test_deduplication(self, tmp_path):
        """Entries with same requestId should be deduplicated."""
        data = [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "s1",
                "message": {"role": "user", "content": "test"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "sessionId": "s1",
                "requestId": "r1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "partial"}],
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                },
            },
            {
                "type": "assistant",
                "uuid": "a1-full",
                "sessionId": "s1",
                "requestId": "r1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "full response"}],
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            },
        ]
        f = tmp_path / "dedup.jsonl"
        f.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")

        session = load_session(str(f))
        assistant_msgs = [m for m in session.messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert session.total_tokens == 20  # Kept the higher one


class TestSegmentSpans:
    def test_basic_segmentation(self, sample_session_path):
        session = load_session(sample_session_path)
        spans = segment_spans(session)
        assert len(spans) > 0

        phases = {s.phase for s in spans}
        assert SpanPhase.REASONING in phases
        assert SpanPhase.TOOL_USE in phases
        assert SpanPhase.GENERATION in phases

    def test_token_count_estimation(self, minimal_session):
        session = load_session(minimal_session)
        spans = segment_spans(session)
        for span in spans:
            assert span.token_count >= 1
            # Heuristic: len/4
            expected = max(1, len(span.text) // 4)
            assert span.token_count == expected

    def test_position_ordering(self, sample_session_path):
        session = load_session(sample_session_path)
        spans = segment_spans(session)
        positions = [s.position for s in spans]
        assert positions == list(range(len(spans)))
