import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

import ter_calculator.real_time as rt


class FakeModel:
    def encode(self, text, **kwargs):
        if isinstance(text, list):
            return np.array([self.encode(t) for t in text], dtype=np.float32)
        # deterministic non-zero vectors
        h = sum(ord(c) for c in str(text))
        return np.array(
            [1.0, float((h % 7) + 1), float((h % 11) + 1)], dtype=np.float32
        )


def test_state_economics_growth_and_health():
    s = rt.RollingTERState(
        total_input_tokens=1000,
        total_output_tokens=200,
        total_cache_read_tokens=500,
        total_cache_creation_tokens=100,
        assistant_waste_tokens=20,
        user_waste_tokens=10,
        turn_context_sizes=[200, 500, 1200],
        session_start_time=10.0,
    )
    cm = SimpleNamespace(
        input_rate=2, output_rate=10, cache_read_rate=0.2, cache_write_rate=3
    )
    assert s.get_cache_hit_rate() == pytest.approx(1 / 3)
    assert s.get_estimated_cost(cm) > 0
    assert s.get_estimated_waste_cost(cm) > 0
    assert s.get_context_growth_rate() == 6
    assert s.is_context_bloat_detected()
    assert s.get_session_duration(25) == 15
    sig = rt.TERSignal("s", 0, 0.5, 0.5, 1, 1, 1, 0, rt.DriftDirection.STABLE, 0)
    assert sig.is_healthy
    bad = rt.TERSignal("s", 0, 0.5, 0.5, 1, 1, 1, 0, rt.DriftDirection.DEGRADING, 0.2)
    assert not bad.is_healthy


def test_helper_edge_paths(monkeypatch):
    monkeypatch.setattr(
        "ter_calculator.embedding_cache.get_embedding_model", lambda n: ("model", n)
    )
    assert rt.load_embedding_model("x") == ("model", "x")
    assert rt._cosine_similarity(np.zeros(2), np.ones(2)) == 0
    assert rt._is_aligned(0.1, "tool_use", "")
    assert not rt._is_aligned(0, "reasoning", "tiny")
    assert not rt._is_aligned(0, "generation", "word " * 60)
    assert rt._is_aligned(0.9, "unknown", "x")
    assert not rt._is_bash_antipattern("Read", {"command": "cat x"})
    assert not rt._is_bash_antipattern("Bash", {})
    assert rt._is_bash_antipattern("Bash", {"command": "grep x file"})
    for text in (
        "<tool_use_error>x",
        "Error: bad",
        "Exit code 1",
        "Wasted call",
        "Permission denied",
    ):
        assert rt._is_error_result_text(text)
    assert not rt._is_error_result_text("ok")
    recent = [np.array([1.0, 0.0], dtype=np.float32)]
    assert rt._is_repetition(np.array([1.0, 0.0], dtype=np.float32), recent)
    for i in range(15):
        rt._record_embedding(np.array([i], dtype=np.float32), recent)
    assert len(recent) == rt.REPETITION_WINDOW
    assert (
        rt._extract_blocks_from_line({"message": {"content": "hello"}})[0]["type"]
        == "text"
    )
    assert rt._extract_blocks_from_line({"message": {"content": None}}) == []


def test_drift_all_directions():
    assert rt.detect_drift([]) == (rt.DriftDirection.STABLE, 0.0)
    assert rt.detect_drift([0.5, 0.51], threshold=0.2)[0] is rt.DriftDirection.STABLE
    assert (
        rt.detect_drift([0.1, 0.3, 0.6], threshold=0.05)[0]
        is rt.DriftDirection.IMPROVING
    )
    assert (
        rt.detect_drift([0.9, 0.5, 0.1], threshold=0.05)[0]
        is rt.DriftDirection.DEGRADING
    )


def test_compute_rolling_usage_errors_duplicates_and_bloat(monkeypatch):
    monkeypatch.setattr(rt.time, "time", lambda: 1000.0)
    state = rt.RollingTERState(turn_context_sizes=[200, 500, 1200])
    model = FakeModel()
    lines = [
        {
            "uuid": "u1",
            "requestId": "r1",
            "sessionId": "s",
            "timestamp": "1970-01-01T00:16:39Z",
            "message": {
                "role": "user",
                "content": "fix parser",
                "usage": {"input_tokens": 100, "cache_read_input_tokens": 50},
            },
        },
        {
            "uuid": "u2",
            "requestId": "r2",
            "sessionId": "s",
            "timestamp": 999,
            "message": {
                "role": "assistant",
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 50,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 5,
                },
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"command": "cat file.py"},
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "Error: failed",
                    },
                    {"type": "thinking", "thinking": "fix parser carefully"},
                    {"type": "text", "text": "done"},
                ],
            },
        },
        {
            "uuid": "u2",
            "requestId": "r2",
            "sessionId": "s",
            "message": {"role": "assistant", "content": "duplicate"},
        },
    ]
    signals = rt.compute_rolling_ter(state, lines, model=model)
    assert signals
    sig = signals[-1]
    assert sig.total_input_tokens == 100
    assert sig.total_output_tokens == 20
    assert state.total_tool_calls == 1
    assert state.has_thinking_blocks
    assert sig.context_growth_rate > 0
    assert sig.is_live


def test_session_monitor_file_paths_and_callbacks(tmp_path, monkeypatch):
    p = tmp_path / "s.jsonl"
    seen = []
    mon = rt.SessionMonitor(
        p, model=FakeModel(), on_signal=seen.append, skip_history=False
    )
    assert mon._read_new_lines() == []
    p.write_text(
        "\nnot-json\n"
        + json.dumps({"message": {"role": "user", "content": "task"}})
        + "\n",
        encoding="utf8",
    )
    assert len(mon._read_new_lines()) == 1
    # append assistant line and emit callback
    with p.open("a") as f:
        f.write(
            json.dumps(
                {
                    "sessionId": "s",
                    "message": {"role": "assistant", "content": "answer"},
                }
            )
            + "\n"
        )
    signals = mon.poll_once()
    assert signals and seen
    assert mon.current_ter >= 0
    assert isinstance(mon.signal_history, list)
    # OSError path
    monkeypatch.setattr(
        "builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("x"))
    )
    assert mon._read_new_lines() == []


def test_session_monitor_skip_history_only_calls_last(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {"sessionId": "s", "message": {"role": "user", "content": "task"}}
                ),
                json.dumps(
                    {
                        "sessionId": "s",
                        "message": {"role": "assistant", "content": "one"},
                    }
                ),
                json.dumps(
                    {
                        "sessionId": "s",
                        "message": {"role": "assistant", "content": "two"},
                    }
                ),
            ]
        )
        + "\n"
    )
    seen = []
    mon = rt.SessionMonitor(
        p, model=FakeModel(), on_signal=seen.append, skip_history=True
    )
    signals = mon.poll_once()
    assert signals and len(seen) == 1 and seen[0] is signals[-1]


def test_monitor_and_dashboard_run_stop_and_summary(tmp_path, monkeypatch):
    p = tmp_path / "a.jsonl"
    p.write_text("")
    mon = rt.SessionMonitor(p, poll_interval=0)
    calls = []

    def poll():
        calls.append(1)
        mon.stop()
        return []

    monkeypatch.setattr(mon, "poll_once", poll)
    monkeypatch.setattr(rt.time, "sleep", lambda x: None)
    mon.run()
    assert calls

    dash = rt.LiveDashboard(tmp_path, poll_interval=0, skip_history=False)
    assert dash._discover_sessions() == [p]
    m = dash._ensure_monitor(p)
    m.state.total_tokens = 10
    m.state.waste_tokens = 2
    m.state.recent_ter_values = [0.5, 0.7]
    assert dash._ensure_monitor(p) is m
    assert str(p) in dash.active_sessions
    summary = dash.get_summary()
    assert summary["session_count"] == 1 and summary["total_tokens"] == 10
    calls = []

    def dpoll():
        calls.append(1)
        dash.stop()
        return []

    monkeypatch.setattr(dash, "poll_once", dpoll)
    dash.run()
    assert calls
    assert rt.LiveDashboard(tmp_path / "missing")._discover_sessions() == []
