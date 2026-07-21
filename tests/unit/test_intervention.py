from __future__ import annotations

import io
import json

from ter_calculator.cli import main
from ter_calculator.hook_monitor import HookConfig, HookSessionState
from ter_calculator.intervention import (
    build_budget_hint,
    check_permission_loop,
    check_pre_tool_duplicate,
    check_reasoning_loop,
    process_intervention_event,
)


def test_reasoning_loop_breaker_triggers_on_restatement():
    state = HookSessionState(session_id="s")
    text = "We should inspect the parser and then update the parser tests before running pytest."
    assert check_reasoning_loop(text, state) is None
    alert = check_reasoning_loop(text + " again", state)
    assert alert is not None
    assert alert.pattern_type == "reasoning_loop"
    assert "Move to a concrete action" in alert.message


def test_reasoning_loop_ignores_different_content():
    state = HookSessionState(session_id="s")
    check_reasoning_loop(
        "Inspect the parser implementation and identify the failing branch logic.",
        state,
    )
    alert = check_reasoning_loop(
        "Run the focused unit tests and collect the traceback for the failure.", state
    )
    assert alert is None


def test_pre_tool_duplicate_is_blocked_after_recorded_call():
    state = HookSessionState(session_id="s")
    event = {"tool_name": "Read", "tool_input": {"file_path": "a.py"}}
    from ter_calculator.hook_monitor import check_duplicate_tool_call

    assert (
        check_duplicate_tool_call("Read", {"file_path": "a.py"}, state, threshold=2)
        is None
    )
    alert = check_pre_tool_duplicate(event, state, threshold=2)
    assert alert is not None
    assert alert.pattern_type == "duplicate_tool_call_prevented"


def test_permission_loop_circuit_breaker():
    state = HookSessionState(session_id="s")
    event = {"tool_name": "Bash", "decision": "denied"}
    assert check_permission_loop(event, state, threshold=2) is None
    alert = check_permission_loop(event, state, threshold=2)
    assert alert is not None
    assert "denied 2 times" in alert.message


def test_session_start_budget_hint():
    hint, metadata = build_budget_hint("Fix a typo in the README")
    assert "TER Budget Hint" in hint
    assert metadata["max_thinking_tokens"] > 0
    assert metadata["model_tier"] in {"haiku", "sonnet", "opus"}


def test_pretool_output_uses_permission_decision():
    state = HookSessionState(session_id="s")
    cfg = HookConfig(min_duplicate_calls=2)
    from ter_calculator.hook_monitor import check_duplicate_tool_call

    check_duplicate_tool_call("Read", {"file_path": "a.py"}, state, threshold=2)
    event = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Read",
        "tool_input": {"file_path": "a.py"},
    }
    alerts, _, output = process_intervention_event(event, state, cfg)
    assert alerts
    specific = output["hookSpecificOutput"]
    assert specific["hookEventName"] == "PreToolUse"
    assert specific["permissionDecision"] == "deny"


def test_cli_session_start(monkeypatch, capsys, tmp_path):
    event = {
        "session_id": "phase3",
        "hook_event_name": "SessionStart",
        "prompt": "Implement a multi-file migration and tests",
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert main(["hook", "monitor", "--state-dir", str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert "additionalContext" in output
    assert "thinking tokens" in output["additionalContext"]


def test_cli_permission_loop_persists(monkeypatch, capsys, tmp_path):
    event = {
        "session_id": "phase3-perm",
        "hook_event_name": "PermissionRequest",
        "tool_name": "Bash",
        "decision": "denied",
    }
    for expected_alert in (False, True):
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
        assert main(["hook", "monitor", "--state-dir", str(tmp_path)]) == 0
        output = json.loads(capsys.readouterr().out)
        assert ("additionalContext" in output) is expected_alert
