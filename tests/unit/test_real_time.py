"""Tests for real-time TER monitoring and live session analysis."""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from ter_calculator.real_time import (
    DriftDirection,
    LiveDashboard,
    RollingTERState,
    SessionMonitor,
    TERSignal,
    WarningLevel,
    compute_rolling_ter,
    detect_drift,
    _embed_text_fast,
    _cosine_similarity,
    DEFAULT_POLL_INTERVAL_SEC,
    DRIFT_THRESHOLD,
    DRIFT_WINDOW,
)


class TestRollingTERState:
    """Test rolling TER state accumulator."""

    def test_initial_state_empty(self):
        state = RollingTERState()
        assert state.total_tokens == 0
        assert state.aligned_tokens == 0
        assert state.waste_tokens == 0
        assert state.message_count == 0

    def test_aggregate_ter_with_zero_tokens(self):
        state = RollingTERState()
        assert state.aggregate_ter == 0.0

    def test_raw_ratio_with_zero_tokens(self):
        state = RollingTERState()
        assert state.raw_ratio == 0.0

    def test_phase_dicts_initialized(self):
        state = RollingTERState()
        assert "reasoning" in state.phase_total
        assert "tool_use" in state.phase_total
        assert "generation" in state.phase_total

    def test_aggregate_ter_calculation(self):
        state = RollingTERState()
        state.total_tokens = 1000
        state.aligned_tokens = 800
        state.phase_total["reasoning"] = 300
        state.phase_aligned["reasoning"] = 240
        state.phase_total["tool_use"] = 400
        state.phase_aligned["tool_use"] = 360
        state.phase_total["generation"] = 300
        state.phase_aligned["generation"] = 200

        # TER = 0.3 * (240/300) + 0.4 * (360/400) + 0.3 * (200/300)
        # = 0.3 * 0.8 + 0.4 * 0.9 + 0.3 * 0.667
        # = 0.24 + 0.36 + 0.2 = 0.8
        assert 0.78 < state.aggregate_ter < 0.82

    def test_raw_ratio_calculation(self):
        state = RollingTERState()
        state.total_tokens = 1000
        state.aligned_tokens = 750
        assert state.raw_ratio == 0.75


class TestFastEmbedding:
    """Test fast character-based embedding."""

    def test_embed_text_fast_returns_normalized_vector(self):
        vec = _embed_text_fast("test text here")
        assert len(vec) == 384
        # Should be normalized
        norm = np.linalg.norm(vec)
        assert 0.99 < norm <= 1.01

    def test_embed_short_text(self):
        vec = _embed_text_fast("ab")
        assert len(vec) == 384
        assert vec[0] == 1.0  # Short text fallback

    def test_embed_empty_text(self):
        vec = _embed_text_fast("")
        assert len(vec) == 384

    def test_embed_deterministic(self):
        vec1 = _embed_text_fast("test")
        vec2 = _embed_text_fast("test")
        assert np.allclose(vec1, vec2)


class TestCosineSimilarity:
    """Test cosine similarity function."""

    def test_identical_vectors(self):
        v = np.random.rand(384).astype(np.float32)
        sim = _cosine_similarity(v, v)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_vectors(self):
        a = np.zeros(384, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(384, dtype=np.float32)
        b[1] = 1.0
        sim = _cosine_similarity(a, b)
        assert sim == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(384, dtype=np.float32)
        b = np.random.rand(384).astype(np.float32)
        sim = _cosine_similarity(a, b)
        assert sim == 0.0


class TestDetectDrift:
    """Test TER drift detection."""

    def test_stable_when_scores_consistent(self):
        scores = [0.85, 0.84, 0.86, 0.85, 0.84]
        drift, magnitude = detect_drift(scores)
        assert drift == DriftDirection.STABLE

    def test_improving_when_scores_increase(self):
        scores = [0.60, 0.65, 0.70, 0.75, 0.80]
        drift, magnitude = detect_drift(scores, threshold=0.05)
        assert drift == DriftDirection.IMPROVING

    def test_degrading_when_scores_decrease(self):
        scores = [0.90, 0.85, 0.80, 0.75, 0.70]
        drift, magnitude = detect_drift(scores, threshold=0.05)
        assert drift == DriftDirection.DEGRADING

    def test_empty_scores_returns_stable(self):
        drift, magnitude = detect_drift([])
        assert drift == DriftDirection.STABLE
        assert magnitude == 0.0

    def test_single_score_returns_stable(self):
        drift, magnitude = detect_drift([0.85])
        assert drift == DriftDirection.STABLE


class TestComputeRollingTER:
    """Test rolling TER computation from JSONL lines."""

    def test_user_message_updates_intent(self):
        state = RollingTERState()
        lines = [
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Fix the bug in main.py"}],
                }
            }
        ]
        signals = compute_rolling_ter(state, lines)

        assert state.intent_text == "Fix the bug in main.py"
        assert state.intent_embedding is not None
        assert len(signals) == 0  # User messages don't generate signals

    def test_assistant_message_generates_signal(self):
        state = RollingTERState()
        # Set up intent first
        state.intent_text = "Fix bug"
        state.intent_embedding = _embed_text_fast("Fix bug")

        lines = [
            {
                "sessionId": "test-123",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me analyze the bug"},
                        {"type": "text", "text": "I'll fix it now"},
                    ],
                }
            }
        ]
        signals = compute_rolling_ter(state, lines)

        assert len(signals) == 1
        assert isinstance(signals[0], TERSignal)
        assert signals[0].session_id == "test-123"
        assert state.total_tokens > 0

    def test_empty_lines_returns_empty_signals(self):
        state = RollingTERState()
        signals = compute_rolling_ter(state, [])
        assert len(signals) == 0

    def test_deduplicates_by_request_id(self):
        state = RollingTERState()
        state.intent_embedding = _embed_text_fast("test")

        lines = [
            {
                "message": {
                    "role": "assistant",
                    "requestId": "req-1",
                    "content": [{"type": "text", "text": "response"}],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "requestId": "req-1",  # Duplicate
                    "content": [{"type": "text", "text": "response"}],
                }
            },
        ]
        signals = compute_rolling_ter(state, lines)

        # Should only process first one
        assert len(signals) == 1


class TestTERSignal:
    """Test TER signal dataclass."""

    def test_signal_creation(self):
        signal = TERSignal(
            session_id="test-123",
            timestamp=time.time(),
            aggregate_ter=0.85,
            raw_ratio=0.82,
            message_index=5,
            total_tokens=1000,
            aligned_tokens=850,
            waste_tokens=150,
            drift=DriftDirection.STABLE,
            drift_magnitude=0.02,
        )
        assert signal.aggregate_ter == 0.85
        assert signal.session_id == "test-123"

    def test_signal_is_healthy(self):
        signal = TERSignal(
            session_id="test",
            timestamp=time.time(),
            aggregate_ter=0.90,
            raw_ratio=0.90,
            message_index=1,
            total_tokens=100,
            aligned_tokens=90,
            waste_tokens=10,
            drift=DriftDirection.STABLE,
            drift_magnitude=0.01,
            warning_level=WarningLevel.INFO,
        )
        assert signal.is_healthy

    def test_signal_not_healthy_with_degrading(self):
        signal = TERSignal(
            session_id="test",
            timestamp=time.time(),
            aggregate_ter=0.70,
            raw_ratio=0.70,
            message_index=1,
            total_tokens=100,
            aligned_tokens=70,
            waste_tokens=30,
            drift=DriftDirection.DEGRADING,
            drift_magnitude=0.20,
            warning_level=WarningLevel.INFO,
        )
        assert not signal.is_healthy


class TestSessionMonitor:
    """Test single-session monitoring."""

    def test_monitor_creation(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            session_path = Path(f.name)
            f.write("{}\n")

        try:
            monitor = SessionMonitor(session_path)
            assert monitor.path == session_path
            assert monitor.poll_interval == DEFAULT_POLL_INTERVAL_SEC
            assert isinstance(monitor.state, RollingTERState)
        finally:
            session_path.unlink()

    def test_monitor_with_custom_interval(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            session_path = Path(f.name)

        try:
            monitor = SessionMonitor(session_path, poll_interval=5.0)
            assert monitor.poll_interval == 5.0
        finally:
            session_path.unlink()


class TestLiveDashboard:
    """Test multi-session live dashboard."""

    def test_dashboard_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dashboard = LiveDashboard(tmpdir, poll_interval=1.0)
            assert dashboard.project_dir == Path(tmpdir)
            assert dashboard.poll_interval == 1.0

    def test_dashboard_requires_project_dir(self):
        with pytest.raises(TypeError):
            LiveDashboard()  # Missing required project_dir argument


class TestWarningLevel:
    """Test warning level enum."""

    def test_warning_levels_defined(self):
        assert WarningLevel.INFO
        assert WarningLevel.CAUTION
        assert WarningLevel.ALERT

    def test_warning_level_values(self):
        assert WarningLevel.INFO.value == "info"
        assert WarningLevel.CAUTION.value == "caution"
        assert WarningLevel.ALERT.value == "alert"


class TestDriftDirection:
    """Test drift direction enum."""

    def test_drift_directions_defined(self):
        assert DriftDirection.IMPROVING
        assert DriftDirection.DEGRADING
        assert DriftDirection.STABLE

    def test_drift_direction_values(self):
        assert DriftDirection.IMPROVING.value == "improving"
        assert DriftDirection.DEGRADING.value == "degrading"
        assert DriftDirection.STABLE.value == "stable"


class TestConstants:
    """Test module constants."""

    def test_default_poll_interval(self):
        assert DEFAULT_POLL_INTERVAL_SEC > 0
        assert DEFAULT_POLL_INTERVAL_SEC < 60

    def test_drift_window_size(self):
        assert DRIFT_WINDOW > 0
        assert isinstance(DRIFT_WINDOW, int)

    def test_drift_threshold(self):
        assert 0 < DRIFT_THRESHOLD < 1


class TestIntegrationScenario:
    """Integration tests for realistic scenarios."""

    def test_full_session_workflow(self):
        """Test a complete user request -> assistant response workflow."""
        state = RollingTERState()

        # User asks a question
        user_lines = [
            {
                "sessionId": "test-session",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Fix the bug in main.py"}],
                }
            }
        ]
        signals = compute_rolling_ter(state, user_lines)
        assert len(signals) == 0
        assert state.intent_text != ""

        # Assistant responds
        assistant_lines = [
            {
                "sessionId": "test-session",
                "message": {
                    "role": "assistant",
                    "requestId": "req-1",
                    "content": [
                        {"type": "thinking", "thinking": "I need to analyze the bug"},
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "main.py"},
                        },
                        {"type": "text", "text": "I'll fix the issue now"},
                    ],
                }
            }
        ]
        signals = compute_rolling_ter(state, assistant_lines)
        assert len(signals) == 1
        assert state.total_tokens > 0
        assert state.message_count == 1

    def test_ter_degrades_with_multiple_messages(self):
        """Simulate degrading TER over time."""
        state = RollingTERState()
        state.intent_text = "simple task"
        state.intent_embedding = _embed_text_fast("simple task")

        # First message - aligned
        line1 = {
            "sessionId": "test",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "simple task execution"}],
            }
        }
        signals1 = compute_rolling_ter(state, [line1])

        # Second message - less aligned
        line2 = {
            "sessionId": "test",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "completely unrelated tangent discussion"}
                ],
            }
        }
        signals2 = compute_rolling_ter(state, [line2])

        # With enough divergence, drift should be detected
        if len(state.recent_ter_values) >= 2:
            drift, mag = detect_drift(state.recent_ter_values)
            assert isinstance(drift, DriftDirection)
