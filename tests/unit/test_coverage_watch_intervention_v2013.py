from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json

from ter_calculator.commands import watch
from ter_calculator import intervention as iv
from ter_calculator.hook_monitor import HookSessionState


def sig(**overrides):
    base = dict(
        session_id="session123",
        timestamp=0,
        is_live=True,
        aggregate_ter=0.5,
        raw_ratio=0.6,
        message_index=2,
        drift=SimpleNamespace(value="degrading"),
        drift_magnitude=0.2,
        warnings=["w"],
        warning_level=SimpleNamespace(value="warning"),
        total_tokens=100,
        aligned_tokens=50,
        waste_tokens=50,
        phase_ter={"reasoning": 0.4},
        waste_sources={"loop": 2},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_signal_rendering_json_text_and_log(tmp_path, capsys):
    s = sig()
    data = watch._signal_to_dict(s)
    assert data["ter"] == 0.5 and data["tokens"]["waste"] == 50
    log = (tmp_path / "l").open("w")
    watch._print_signal(s, "json", log)
    log.close()
    assert json.loads((tmp_path / "l").read_text())["session_id"] == "session123"
    watch._last_was_live = False
    watch._print_signal(s, "text")
    watch._print_signal(
        sig(is_live=False, drift=SimpleNamespace(value="improving"), warnings=[]),
        "text",
    )
    assert "LIVE" in capsys.readouterr().out


def test_cmd_watch_stream_and_error(monkeypatch, tmp_path, capsys):
    session = tmp_path / "s.jsonl"
    session.write_text("{}\n")
    monkeypatch.setattr(
        "ter_calculator.real_time.load_embedding_model", lambda: object()
    )

    class Monitor:
        def __init__(self, *a, **k):
            self.cb = k["on_signal"]
            self.stopped = False

        def run(self):
            self.cb(sig())

        def stop(self):
            self.stopped = True

    monkeypatch.setattr("ter_calculator.real_time.SessionMonitor", Monitor)
    args = SimpleNamespace(
        output_format="json",
        stream=True,
        latest=False,
        project_path=str(session),
        quiet=True,
        log_file=str(tmp_path / "log"),
        poll_interval=0.1,
        verbose=False,
    )
    assert watch._cmd_watch(args) == 0
    assert (tmp_path / "log").read_text()
    bad = SimpleNamespace(
        output_format="json",
        stream=True,
        latest=False,
        project_path=None,
        quiet=True,
        log_file=None,
        poll_interval=0.1,
        verbose=False,
    )
    assert watch._cmd_watch(bad) == 1
    monkeypatch.setattr(
        "ter_calculator.real_time.load_embedding_model",
        lambda: (_ for _ in ()).throw(ImportError("missing")),
    )
    assert watch._cmd_watch(args) == 1
    assert "Error" in capsys.readouterr().err


def test_intervention_low_level_branches(monkeypatch):
    state = HookSessionState(session_id="s")
    assert iv._extract_text({"content": ["a", {"text": "b"}]}) == "a\nb"
    assert iv._extract_text({}) == ""
    assert iv._token_cosine("", "x") == 0
    text = "repeat these exact reasoning words many times for loop detection now"
    assert iv.check_reasoning_loop(text, state, min_consecutive=2) is None
    assert iv.check_reasoning_loop(text, state, min_consecutive=2) is not None
    assert iv.check_pre_tool_duplicate({}, state) is None
    event = {"tool_name": "Read", "tool_input": {"file_path": "x"}}
    from ter_calculator.hook_monitor import _compute_tool_signature

    signature = _compute_tool_signature("Read", {"file_path": "x"})
    state.tool_call_counts[signature] = 1
    state.tool_result_summaries[signature] = "ok"
    assert iv.check_pre_tool_duplicate(event, state) is not None
    iv.record_tool_result({**event, "tool_response": {"ok": True}}, state)
    assert "ok" in state.tool_result_summaries[signature]
    assert (
        iv.check_permission_loop({"tool_name": "Bash", "decision": "allow"}, state)
        is None
    )
    assert (
        iv.check_permission_loop({"tool_name": "Bash", "decision": "deny"}, state)
        is None
    )
    assert (
        iv.check_permission_loop({"tool_name": "Bash", "is_denied": True}, state)
        is not None
    )
    hint, meta = iv.build_budget_hint(
        "implement a complex distributed migration with tests"
    )
    assert "TER Budget Hint" in hint and meta["max_thinking_tokens"] > 0
    assert iv._metric_payload({"metrics": {"aggregate_ter": 0.5}}) == {
        "aggregate_ter": 0.5
    }
    assert iv._metric_payload({"metrics": {"x": 1}}) is None
