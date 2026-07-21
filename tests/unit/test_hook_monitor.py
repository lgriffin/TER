"""Tests for the Claude Code hook waste monitor."""

from __future__ import annotations

import json
import os

from ter_calculator.hook_monitor import (
    HookConfig,
    HookSessionState,
    WasteAlert,
    check_bash_antipattern,
    check_duplicate_tool_call,
    check_edit_fragmentation,
    check_repeated_command,
    check_repetitive_read,
    format_guidance,
    format_notification,
    load_state,
    process_tool_event,
    save_state,
)


# -----------------------------------------------------------------------
# check_bash_antipattern
# -----------------------------------------------------------------------


class TestCheckBashAntipattern:
    def test_detects_cat(self):
        alert = check_bash_antipattern("Bash", {"command": "cat foo.py"})
        assert alert is not None
        assert alert.pattern_type == "bash_antipattern"
        assert "Read" in alert.message

    def test_detects_grep(self):
        alert = check_bash_antipattern("Bash", {"command": "grep pattern src/"})
        assert alert is not None
        assert "Grep" in alert.message

    def test_detects_find(self):
        alert = check_bash_antipattern("Bash", {"command": "find . -name '*.py'"})
        assert alert is not None
        assert "Glob" in alert.message

    def test_detects_rg(self):
        alert = check_bash_antipattern("Bash", {"command": "rg TODO"})
        assert alert is not None
        assert "Grep" in alert.message

    def test_detects_piped_grep(self):
        alert = check_bash_antipattern("Bash", {"command": "git log | grep error"})
        assert alert is not None
        assert "Grep" in alert.message

    def test_detects_head(self):
        alert = check_bash_antipattern("Bash", {"command": "head -20 src/models.py"})
        assert alert is not None
        assert "Read" in alert.message

    def test_detects_tail(self):
        alert = check_bash_antipattern("Bash", {"command": "tail -50 output.log"})
        assert alert is not None
        assert "Read" in alert.message

    def test_ignores_safe_commands(self):
        assert check_bash_antipattern("Bash", {"command": "git status"}) is None
        assert check_bash_antipattern("Bash", {"command": "pytest tests/"}) is None
        assert check_bash_antipattern("Bash", {"command": "npm install"}) is None

    def test_ignores_non_bash(self):
        assert check_bash_antipattern("Read", {"file_path": "foo.py"}) is None
        assert check_bash_antipattern("Edit", {"file_path": "foo.py"}) is None

    def test_ignores_empty_command(self):
        assert check_bash_antipattern("Bash", {"command": ""}) is None
        assert check_bash_antipattern("Bash", {}) is None

    def test_truncates_long_commands(self):
        long_cmd = "cat " + "x" * 200
        alert = check_bash_antipattern("Bash", {"command": long_cmd})
        assert alert is not None
        assert "..." in alert.message

    def test_details_contain_recommended_tool(self):
        alert = check_bash_antipattern("Bash", {"command": "cat foo.py"})
        assert alert is not None
        assert alert.details["recommended_tool"] == "Read"


# -----------------------------------------------------------------------
# check_repetitive_read
# -----------------------------------------------------------------------


class TestCheckRepetitiveRead:
    def test_no_alert_below_threshold(self):
        state = HookSessionState(session_id="s1")
        assert check_repetitive_read("Read", {"file_path": "a.py"}, state) is None
        assert check_repetitive_read("Read", {"file_path": "a.py"}, state) is None

    def test_alerts_at_threshold(self):
        state = HookSessionState(session_id="s1")
        check_repetitive_read("Read", {"file_path": "a.py"}, state)
        check_repetitive_read("Read", {"file_path": "a.py"}, state)
        alert = check_repetitive_read("Read", {"file_path": "a.py"}, state)
        assert alert is not None
        assert alert.pattern_type == "repetitive_read"
        assert "3 times" in alert.message

    def test_different_files_independent(self):
        state = HookSessionState(session_id="s1")
        check_repetitive_read("Read", {"file_path": "a.py"}, state)
        check_repetitive_read("Read", {"file_path": "b.py"}, state)
        check_repetitive_read("Read", {"file_path": "c.py"}, state)
        assert state.file_read_counts.get(os.path.normpath("a.py")) == 1

    def test_state_updated(self):
        state = HookSessionState(session_id="s1")
        check_repetitive_read("Read", {"file_path": "a.py"}, state)
        normalized = os.path.normpath("a.py")
        assert state.file_read_counts[normalized] == 1

    def test_ignores_non_read(self):
        state = HookSessionState(session_id="s1")
        assert check_repetitive_read("Bash", {"command": "ls"}, state) is None

    def test_ignores_empty_path(self):
        state = HookSessionState(session_id="s1")
        assert check_repetitive_read("Read", {"file_path": ""}, state) is None
        assert check_repetitive_read("Read", {}, state) is None

    def test_custom_threshold(self):
        state = HookSessionState(session_id="s1")
        check_repetitive_read("Read", {"file_path": "a.py"}, state, threshold=2)
        alert = check_repetitive_read("Read", {"file_path": "a.py"}, state, threshold=2)
        assert alert is not None

    def test_severity_escalates(self):
        state = HookSessionState(session_id="s1")
        for _ in range(4):
            check_repetitive_read("Read", {"file_path": "a.py"}, state)
        alert = check_repetitive_read("Read", {"file_path": "a.py"}, state)
        assert alert is not None
        assert alert.severity == "warning"


# -----------------------------------------------------------------------
# check_edit_fragmentation
# -----------------------------------------------------------------------


class TestCheckEditFragmentation:
    def test_no_alert_for_single_edit(self):
        state = HookSessionState(session_id="s1")
        assert check_edit_fragmentation("Edit", {"file_path": "a.py"}, state) is None

    def test_alerts_on_three_consecutive(self):
        state = HookSessionState(session_id="s1")
        check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        alert = check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        assert alert is not None
        assert alert.pattern_type == "edit_fragmentation"
        assert "3 consecutive" in alert.message

    def test_broken_by_different_file(self):
        state = HookSessionState(session_id="s1")
        check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        check_edit_fragmentation("Edit", {"file_path": "b.py"}, state)
        alert = check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        assert alert is None

    def test_broken_by_non_edit_tool(self):
        state = HookSessionState(session_id="s1")
        check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        check_edit_fragmentation("Read", {"file_path": "a.py"}, state)
        alert = check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        assert alert is None

    def test_write_counts_as_edit(self):
        state = HookSessionState(session_id="s1")
        check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        check_edit_fragmentation("Write", {"file_path": "a.py"}, state)
        alert = check_edit_fragmentation("Edit", {"file_path": "a.py"}, state)
        assert alert is not None

    def test_recent_edits_capped(self):
        state = HookSessionState(session_id="s1")
        for i in range(25):
            check_edit_fragmentation("Edit", {"file_path": f"f{i}.py"}, state)
        assert len(state.recent_edits) <= 20


# -----------------------------------------------------------------------
# check_duplicate_tool_call
# -----------------------------------------------------------------------


class TestCheckDuplicateToolCall:
    def test_no_alert_on_first_call(self):
        state = HookSessionState(session_id="s1")
        alert = check_duplicate_tool_call("Read", {"file_path": "a.py"}, state)
        assert alert is None

    def test_alerts_on_second_identical_call(self):
        state = HookSessionState(session_id="s1")
        check_duplicate_tool_call("Read", {"file_path": "a.py"}, state)
        alert = check_duplicate_tool_call("Read", {"file_path": "a.py"}, state)
        assert alert is not None
        assert alert.pattern_type == "duplicate_tool_call"
        assert "invocation #2" in alert.message

    def test_different_inputs_not_duplicate(self):
        state = HookSessionState(session_id="s1")
        check_duplicate_tool_call("Read", {"file_path": "a.py"}, state)
        alert = check_duplicate_tool_call("Read", {"file_path": "b.py"}, state)
        assert alert is None

    def test_different_tools_not_duplicate(self):
        state = HookSessionState(session_id="s1")
        check_duplicate_tool_call("Read", {"file_path": "a.py"}, state)
        alert = check_duplicate_tool_call("Edit", {"file_path": "a.py"}, state)
        assert alert is None

    def test_higher_threshold(self):
        state = HookSessionState(session_id="s1")
        check_duplicate_tool_call("Read", {"file_path": "a.py"}, state, threshold=3)
        alert = check_duplicate_tool_call(
            "Read", {"file_path": "a.py"}, state, threshold=3
        )
        assert alert is None
        alert = check_duplicate_tool_call(
            "Read", {"file_path": "a.py"}, state, threshold=3
        )
        assert alert is not None

    def test_tool_call_counts_capped(self):
        state = HookSessionState(session_id="s1")
        for i in range(600):
            check_duplicate_tool_call("Bash", {"command": f"cmd{i}"}, state)
        assert len(state.tool_call_counts) <= 510


# -----------------------------------------------------------------------
# check_repeated_command
# -----------------------------------------------------------------------


class TestCheckRepeatedCommand:
    def test_alerts_at_threshold(self):
        state = HookSessionState(session_id="s1")
        check_repeated_command("Bash", {"command": "pytest tests/"}, state)
        check_repeated_command("Bash", {"command": "pytest tests/"}, state)
        alert = check_repeated_command("Bash", {"command": "pytest tests/"}, state)
        assert alert is not None
        assert alert.pattern_type == "repeated_command"
        assert "3 times" in alert.message

    def test_no_alert_below_threshold(self):
        state = HookSessionState(session_id="s1")
        check_repeated_command("Bash", {"command": "pytest tests/"}, state)
        alert = check_repeated_command("Bash", {"command": "pytest tests/"}, state)
        assert alert is None

    def test_normalizes_tail_variants(self):
        state = HookSessionState(session_id="s1")
        check_repeated_command("Bash", {"command": "git log | tail -30"}, state)
        check_repeated_command("Bash", {"command": "git log | tail -50"}, state)
        alert = check_repeated_command("Bash", {"command": "git log | tail -10"}, state)
        assert alert is not None

    def test_different_commands_independent(self):
        state = HookSessionState(session_id="s1")
        check_repeated_command("Bash", {"command": "pytest tests/"}, state)
        check_repeated_command("Bash", {"command": "ruff check src/"}, state)
        check_repeated_command("Bash", {"command": "git status"}, state)
        for key in state.bash_command_counts.values():
            assert key == 1

    def test_ignores_non_bash(self):
        state = HookSessionState(session_id="s1")
        assert check_repeated_command("Read", {"file_path": "a.py"}, state) is None

    def test_ignores_empty_command(self):
        state = HookSessionState(session_id="s1")
        assert check_repeated_command("Bash", {"command": ""}, state) is None
        assert check_repeated_command("Bash", {}, state) is None

    def test_truncates_long_commands(self):
        state = HookSessionState(session_id="s1")
        long_cmd = "echo " + "x" * 200
        for _ in range(3):
            check_repeated_command("Bash", {"command": long_cmd}, state)
        alert = check_repeated_command("Bash", {"command": long_cmd}, state)
        assert alert is not None
        assert "..." in alert.message


# -----------------------------------------------------------------------
# process_tool_event
# -----------------------------------------------------------------------


class TestProcessToolEvent:
    def test_bash_antipattern_detected(self):
        state = HookSessionState(session_id="s1")
        event = {"tool_name": "Bash", "tool_input": {"command": "cat foo.py"}}
        alerts, state = process_tool_event(event, state)
        assert len(alerts) >= 1
        assert any(a.pattern_type == "bash_antipattern" for a in alerts)

    def test_multiple_alerts_returned(self):
        state = HookSessionState(session_id="s1")
        # Prime repeated command state
        for _ in range(2):
            process_tool_event(
                {"tool_name": "Bash", "tool_input": {"command": "cat foo.py"}},
                state,
            )
        # Third call: bash_antipattern + repeated_command + duplicate_tool_call
        alerts, state = process_tool_event(
            {"tool_name": "Bash", "tool_input": {"command": "cat foo.py"}},
            state,
        )
        types = {a.pattern_type for a in alerts}
        assert "bash_antipattern" in types
        assert "repeated_command" in types

    def test_state_updated_across_calls(self):
        state = HookSessionState(session_id="s1")
        process_tool_event(
            {"tool_name": "Read", "tool_input": {"file_path": "a.py"}},
            state,
        )
        assert state.total_events == 1
        normalized = os.path.normpath("a.py")
        assert state.file_read_counts[normalized] == 1

    def test_unknown_tool_no_crash(self):
        state = HookSessionState(session_id="s1")
        alerts, state = process_tool_event(
            {"tool_name": "UnknownTool", "tool_input": {"x": 1}},
            state,
        )
        assert alerts == []
        assert state.total_events == 1

    def test_non_dict_tool_input(self):
        state = HookSessionState(session_id="s1")
        alerts, state = process_tool_event(
            {"tool_name": "Bash", "tool_input": "not a dict"},
            state,
        )
        assert isinstance(alerts, list)

    def test_missing_fields(self):
        state = HookSessionState(session_id="s1")
        alerts, state = process_tool_event({}, state)
        assert alerts == []

    def test_config_disables_bash_antipatterns(self):
        state = HookSessionState(session_id="s1")
        config = HookConfig(enable_bash_antipatterns=False)
        alerts, state = process_tool_event(
            {"tool_name": "Bash", "tool_input": {"command": "cat foo.py"}},
            state,
            config,
        )
        assert not any(a.pattern_type == "bash_antipattern" for a in alerts)


# -----------------------------------------------------------------------
# State persistence
# -----------------------------------------------------------------------


class TestStatePersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        config = HookConfig(state_dir=str(tmp_path))
        state = HookSessionState(
            session_id="test-session",
            file_read_counts={"a.py": 3, "b.py": 1},
            bash_command_counts={"pytest": 2},
            total_events=5,
        )
        save_state(state, config)
        loaded = load_state("test-session", config)
        assert loaded.session_id == "test-session"
        assert loaded.file_read_counts == {"a.py": 3, "b.py": 1}
        assert loaded.bash_command_counts == {"pytest": 2}
        assert loaded.total_events == 5

    def test_load_missing_creates_fresh(self, tmp_path):
        config = HookConfig(state_dir=str(tmp_path))
        state = load_state("nonexistent", config)
        assert state.session_id == "nonexistent"
        assert state.total_events == 0
        assert state.file_read_counts == {}

    def test_state_dir_created(self, tmp_path):
        subdir = tmp_path / "nested" / "dir"
        config = HookConfig(state_dir=str(subdir))
        state = HookSessionState(session_id="s1")
        save_state(state, config)
        assert subdir.exists()

    def test_corrupt_state_file_handled(self, tmp_path):
        config = HookConfig(state_dir=str(tmp_path))
        state_file = tmp_path / "corrupt.json"
        state_file.write_text("not valid json", encoding="utf-8")
        state = load_state("corrupt", config)
        assert state.session_id == "corrupt"
        assert state.total_events == 0


# -----------------------------------------------------------------------
# format_guidance
# -----------------------------------------------------------------------


class TestFormatGuidance:
    def test_single_alert(self):
        alerts = [WasteAlert("bash_antipattern", "info", "Use Read instead")]
        result = format_guidance(alerts)
        assert "[TER Waste Monitor]" in result
        assert "Use Read instead" in result

    def test_multiple_alerts(self):
        alerts = [
            WasteAlert("bash_antipattern", "info", "Alert one"),
            WasteAlert("repetitive_read", "warning", "Alert two"),
        ]
        result = format_guidance(alerts)
        assert "1." in result
        assert "2." in result
        assert "Alert one" in result
        assert "Alert two" in result

    def test_empty_alerts(self):
        assert format_guidance([]) == ""


# -----------------------------------------------------------------------
# format_notification
# -----------------------------------------------------------------------


class TestFormatNotification:
    def test_single_info_alert(self):
        alerts = [WasteAlert("bash_antipattern", "info", "msg")]
        result = format_notification(alerts)
        assert "TER:" in result
        assert "bash antipattern" in result
        assert "[~]" in result

    def test_single_warning_alert(self):
        alerts = [WasteAlert("repetitive_read", "warning", "msg")]
        result = format_notification(alerts)
        assert "[!]" in result

    def test_multiple_alerts(self):
        alerts = [
            WasteAlert("bash_antipattern", "info", "msg1"),
            WasteAlert("repetitive_read", "warning", "msg2"),
        ]
        result = format_notification(alerts)
        assert "bash antipattern" in result
        assert "repetitive read" in result

    def test_empty_alerts(self):
        assert format_notification([]) == ""


# -----------------------------------------------------------------------
# CLI integration (ter hook monitor)
# -----------------------------------------------------------------------


class TestHookMonitorCLI:
    def test_monitor_bash_antipattern(self, monkeypatch, capsys, tmp_path):
        import io
        from ter_calculator.cli import main

        event = json.dumps(
            {
                "session_id": "test-cli",
                "tool_name": "Bash",
                "tool_input": {"command": "cat foo.py"},
            }
        )
        monkeypatch.setattr("sys.stdin", io.StringIO(event))
        result = main(["hook", "monitor", "--state-dir", str(tmp_path)])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert "additionalContext" in output
        assert "Read" in output["additionalContext"]
        assert "systemMessage" in output
        assert "TER:" in output["systemMessage"]

    def test_monitor_no_alerts(self, monkeypatch, capsys, tmp_path):
        import io
        from ter_calculator.cli import main

        event = json.dumps(
            {
                "session_id": "test-cli",
                "tool_name": "Bash",
                "tool_input": {"command": "git status"},
            }
        )
        monkeypatch.setattr("sys.stdin", io.StringIO(event))
        result = main(["hook", "monitor", "--state-dir", str(tmp_path)])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output == {}

    def test_monitor_invalid_stdin(self, monkeypatch, capsys):
        import io
        from ter_calculator.cli import main

        monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
        result = main(["hook", "monitor"])
        assert result == 1

    def test_hook_no_subcommand(self, capsys):
        from ter_calculator.cli import main

        result = main(["hook"])
        assert result == 1
        assert "Usage" in capsys.readouterr().err

# -----------------------------------------------------------------------
# Phase 2.1 live intervention
# -----------------------------------------------------------------------


class TestLiveEfficiencyIntervention:
    def test_repeated_failed_action_requests_replan(self):
        state = HookSessionState(session_id="phase2")
        cfg = HookConfig(enable_live_efficiency=False, min_repeated_failures=2)
        event = {
            "tool_name": "Bash",
            "tool_input": {"command": "pytest tests/unit/test_missing.py"},
            "tool_response": "Error: file not found; exit code 1",
        }
        alerts, _ = process_tool_event(event, state, cfg)
        assert not any(a.pattern_type == "repeated_failure" for a in alerts)
        alerts, _ = process_tool_event(event, state, cfg)
        failure = next(a for a in alerts if a.pattern_type == "repeated_failure")
        assert failure.severity == "warning"
        assert "revised plan" in failure.message

    def test_degrading_rolling_score_triggers_refresh(self):
        state = HookSessionState(session_id="phase2")
        cfg = HookConfig(
            rolling_window=6,
            min_events_for_efficiency=4,
            efficiency_threshold=0.90,
            drift_threshold=0.05,
            acceleration_threshold=0.05,
            intervention_cooldown=1,
            min_duplicate_calls=2,
        )
        # Two clean events establish the early baseline; repeated duplicate
        # calls then lower the rolling proxy and accelerate waste.
        events = [
            {"tool_name": "Bash", "tool_input": {"command": "git status"}},
            {"tool_name": "Read", "tool_input": {"file_path": "a.py"}},
            {"tool_name": "Glob", "tool_input": {"pattern": "**/*.py"}},
            {"tool_name": "Glob", "tool_input": {"pattern": "**/*.py"}},
            {"tool_name": "Glob", "tool_input": {"pattern": "**/*.py"}},
            {"tool_name": "Glob", "tool_input": {"pattern": "**/*.py"}},
        ]
        all_alerts = []
        for event in events:
            alerts, state = process_tool_event(event, state, cfg)
            all_alerts.extend(alerts)
        intervention = next(
            a for a in all_alerts if a.pattern_type == "efficiency_degradation"
        )
        assert intervention.details["action"] == "mandatory_replan"
        assert "refresh" in intervention.message

    def test_live_efficiency_can_be_disabled(self):
        state = HookSessionState(session_id="phase2")
        cfg = HookConfig(enable_live_efficiency=False, min_duplicate_calls=1)
        alerts, state = process_tool_event(
            {"tool_name": "Read", "tool_input": {"file_path": "a.py"}},
            state,
            cfg,
        )
        assert not any(a.pattern_type == "efficiency_degradation" for a in alerts)
        assert state.recent_efficiency == []

    def test_old_state_ignores_unknown_fields(self, tmp_path):
        config = HookConfig(state_dir=str(tmp_path))
        (tmp_path / "legacy.json").write_text(
            json.dumps({"session_id": "legacy", "total_events": 3, "obsolete": 1}),
            encoding="utf-8",
        )
        state = load_state("legacy", config)
        assert state.total_events == 3
        assert state.recent_efficiency == []
