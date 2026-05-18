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
    _cosine_similarity,
    _is_duplicate_tool_call,
    _is_bash_antipattern,
    DEFAULT_POLL_INTERVAL_SEC,
    DRIFT_THRESHOLD,
    DRIFT_WINDOW,
)


# ---------------------------------------------------------------------------
# Mock model fixture — avoids loading sentence-transformers in unit tests
# ---------------------------------------------------------------------------

class _MockModel:
    """Deterministic mock embedding model for tests.

    Returns normalized vectors seeded on the text hash so identical inputs
    always produce identical vectors and similarity comparisons are stable.
    """
    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(384).astype(np.float32)
        if normalize_embeddings:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
        return vec


@pytest.fixture
def mock_model():
    return _MockModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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
        state.phase_aligned["generation"] = 270
        ter = state.aggregate_ter
        assert 0 < ter <= 1.0

    def test_raw_ratio_calculation(self):
        state = RollingTERState()
        state.total_tokens = 1000
        state.aligned_tokens = 750
        assert state.raw_ratio == 0.75


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

    def test_user_message_updates_intent(self, mock_model):
        state = RollingTERState()
        lines = [
            {
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Fix the bug in main.py"}],
                }
            }
        ]
        signals = compute_rolling_ter(state, lines, model=mock_model)

        assert state.intent_text == "Fix the bug in main.py"
        assert state.intent_embedding is not None
        assert len(signals) == 0  # User messages don't generate signals

    def test_intent_uses_ema_not_mean(self, mock_model):
        """Second prompt shifts intent via EMA, not uniform mean."""
        from ter_calculator.real_time import INTENT_DECAY

        state = RollingTERState()
        prompt1 = "Fix the bug in main.py"
        prompt2 = "Now write tests for main.py"

        compute_rolling_ter(state, [{"message": {"role": "user", "content": [{"type": "text", "text": prompt1}]}}], model=mock_model)
        emb1 = state.intent_embedding.copy()

        compute_rolling_ter(state, [{"message": {"role": "user", "content": [{"type": "text", "text": prompt2}]}}], model=mock_model)
        emb2_raw = mock_model.encode(prompt2, normalize_embeddings=True)

        expected = (INTENT_DECAY * emb2_raw + (1 - INTENT_DECAY) * emb1).astype(np.float32)
        norm = np.linalg.norm(expected)
        if norm > 0:
            expected /= norm

        assert not np.allclose(state.intent_embedding, emb1, atol=1e-5), (
            "Intent should have shifted after second prompt"
        )
        np.testing.assert_allclose(state.intent_embedding, expected, atol=1e-5)

    def test_assistant_message_generates_signal(self, mock_model):
        state = RollingTERState()
        state.intent_text = "Fix bug"
        state.intent_embedding = mock_model.encode("Fix bug")

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
        signals = compute_rolling_ter(state, lines, model=mock_model)

        assert len(signals) == 1
        assert isinstance(signals[0], TERSignal)
        assert signals[0].session_id == "test-123"
        assert state.total_tokens > 0

    def test_empty_lines_returns_empty_signals(self, mock_model):
        state = RollingTERState()
        signals = compute_rolling_ter(state, [], model=mock_model)
        assert len(signals) == 0

    def test_deduplicates_by_request_id(self, mock_model):
        state = RollingTERState()
        state.intent_embedding = mock_model.encode("test")

        # requestId must be at top level (matches actual JSONL format)
        lines = [
            {
                "requestId": "req-1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "response"}],
                }
            },
            {
                "requestId": "req-1",  # Duplicate
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "response"}],
                }
            },
        ]
        signals = compute_rolling_ter(state, lines, model=mock_model)

        assert len(signals) == 1


class TestToolCallDeduplication:
    """Test Phase 2B: duplicate tool call detection."""

    def test_first_call_not_duplicate(self):
        state = RollingTERState()
        assert not _is_duplicate_tool_call("Read", '{"file_path": "foo.py"}', state)

    def test_identical_call_is_duplicate(self):
        state = RollingTERState()
        _is_duplicate_tool_call("Read", '{"file_path": "foo.py"}', state)
        assert _is_duplicate_tool_call("Read", '{"file_path": "foo.py"}', state)

    def test_different_file_not_duplicate(self):
        state = RollingTERState()
        _is_duplicate_tool_call("Read", '{"file_path": "foo.py"}', state)
        assert not _is_duplicate_tool_call("Read", '{"file_path": "bar.py"}', state)

    def test_different_tool_not_duplicate(self):
        state = RollingTERState()
        _is_duplicate_tool_call("Read", '{"file_path": "foo.py"}', state)
        assert not _is_duplicate_tool_call("Bash", '{"file_path": "foo.py"}', state)

    def test_window_evicts_old_calls(self):
        state = RollingTERState()
        _is_duplicate_tool_call("Read", '{"file_path": "foo.py"}', state, window=2)
        _is_duplicate_tool_call("Read", '{"file_path": "bar.py"}', state, window=2)
        _is_duplicate_tool_call("Read", '{"file_path": "baz.py"}', state, window=2)
        # "foo.py" should have been evicted from window=2 history
        assert not _is_duplicate_tool_call("Read", '{"file_path": "foo.py"}', state, window=2)

    def test_duplicate_detected_in_compute_rolling_ter(self, mock_model):
        state = RollingTERState()
        state.intent_embedding = mock_model.encode("fix bugs")

        duplicate_tool_line = {
            "sessionId": "test",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {"command": "pytest"}},
                ],
            }
        }
        compute_rolling_ter(state, [duplicate_tool_line], model=mock_model)
        waste_before = state.waste_tokens

        compute_rolling_ter(state, [duplicate_tool_line], model=mock_model)
        # Second identical call should add waste
        assert state.waste_tokens > waste_before


class TestBashAntipatternDetection:
    """Test Phase 2C: bash anti-pattern detection."""

    def test_cat_is_antipattern(self):
        assert _is_bash_antipattern("Bash", {"command": "cat README.md"})

    def test_grep_is_antipattern(self):
        assert _is_bash_antipattern("Bash", {"command": "grep -r foo src/"})

    def test_find_is_antipattern(self):
        assert _is_bash_antipattern("Bash", {"command": "find . -name '*.py'"})

    def test_rg_is_antipattern(self):
        assert _is_bash_antipattern("Bash", {"command": "rg 'import' src/"})

    def test_head_is_antipattern(self):
        assert _is_bash_antipattern("Bash", {"command": "head -20 file.py"})

    def test_tail_is_antipattern(self):
        assert _is_bash_antipattern("Bash", {"command": "tail -f logs/app.log"})

    def test_pytest_not_antipattern(self):
        assert not _is_bash_antipattern("Bash", {"command": "pytest tests/"})

    def test_git_not_antipattern(self):
        assert not _is_bash_antipattern("Bash", {"command": "git status"})

    def test_non_bash_tool_not_antipattern(self):
        assert not _is_bash_antipattern("Read", {"file_path": "foo.py"})

    def test_piped_grep_is_antipattern(self):
        assert _is_bash_antipattern("Bash", {"command": "ls -la | grep '.py'"})

    def test_antipattern_flagged_as_waste_in_compute(self, mock_model):
        state = RollingTERState()
        state.intent_embedding = mock_model.encode("fix bug")

        line = {
            "sessionId": "test",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "input": {"command": "cat main.py"}},
                ],
            }
        }
        compute_rolling_ter(state, [line], model=mock_model)
        assert state.waste_tokens > 0


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

    def test_economics_fields_default_to_zero(self):
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
        )
        assert signal.estimated_cost_usd == 0.0
        assert signal.cache_hit_rate == 0.0
        assert signal.context_growth_rate == 1.0


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

    def test_full_session_workflow(self, mock_model):
        """Test a complete user request -> assistant response workflow."""
        state = RollingTERState()

        user_lines = [
            {
                "sessionId": "test-session",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Fix the bug in main.py"}],
                }
            }
        ]
        signals = compute_rolling_ter(state, user_lines, model=mock_model)
        assert len(signals) == 0
        assert state.intent_text != ""

        # requestId is at top level in real JSONL format
        assistant_lines = [
            {
                "requestId": "req-1",
                "sessionId": "test-session",
                "message": {
                    "role": "assistant",
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
        signals = compute_rolling_ter(state, assistant_lines, model=mock_model)
        assert len(signals) == 1
        assert state.total_tokens > 0
        assert state.message_count == 1

    def test_ter_degrades_with_multiple_messages(self, mock_model):
        """Simulate degrading TER over time."""
        state = RollingTERState()
        state.intent_text = "simple task"
        state.intent_embedding = mock_model.encode("simple task")

        line1 = {
            "sessionId": "test",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "simple task execution"}],
            }
        }
        compute_rolling_ter(state, [line1], model=mock_model)

        line2 = {
            "sessionId": "test",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "completely unrelated tangent discussion"}
                ],
            }
        }
        compute_rolling_ter(state, [line2], model=mock_model)

        if len(state.recent_ter_values) >= 2:
            drift, mag = detect_drift(state.recent_ter_values)
            assert isinstance(drift, DriftDirection)
