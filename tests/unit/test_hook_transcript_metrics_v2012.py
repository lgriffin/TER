from __future__ import annotations

import json
from pathlib import Path

import pytest
import time

from ter_calculator.hook_monitor import HookConfig, HookSessionState
from ter_calculator.intervention import process_intervention_event
from ter_calculator.intervention_policy import (
    InterventionAction,
    MetricSnapshot,
    PolicyDecision,
    new_intervention_record,
    pending_intervention_path,
    write_pending_intervention,
)
from ter_calculator.transcript_metrics import derive_transcript_metrics


def _entry(uuid: str, content: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "uuid": uuid,
            "sessionId": "s1",
            "type": "assistant",
            "message": {"role": "assistant", "content": content},
        }
    )


def test_transcript_metrics_are_incremental(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        _entry(
            "1", [{"type": "thinking", "thinking": "Inspect the code and run tests."}]
        )
        + "\n",
        encoding="utf-8",
    )
    state = HookSessionState(session_id="s1")
    first = derive_transcript_metrics({"transcript_path": str(transcript)}, state)
    assert first is not None
    first_offset = state.transcript_offset
    first_tokens = state.transcript_total_tokens

    second = derive_transcript_metrics({"transcript_path": str(transcript)}, state)
    assert second is not None
    assert state.transcript_offset == first_offset
    assert state.transcript_total_tokens == first_tokens

    duplicate = {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}}
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(_entry("2", [duplicate]) + "\n")
        handle.write(_entry("3", [duplicate]) + "\n")
    third = derive_transcript_metrics({"transcript_path": str(transcript)}, state)
    assert third is not None
    assert third["repeated_tool_calls"] == 1
    assert third["waste_ratio"] > 0
    assert state.transcript_total_tokens > first_tokens


def test_transcript_metric_failure_is_silent(tmp_path: Path) -> None:
    state = HookSessionState(session_id="s1")
    assert (
        derive_transcript_metrics({"transcript_path": str(tmp_path / "missing")}, state)
        is None
    )


@pytest.mark.parametrize("mode", ["suggest", "warn", "block"])
def test_pending_intervention_surfaces_in_active_modes(
    tmp_path: Path, mode: str
) -> None:
    state = HookSessionState(session_id="s1")
    config = HookConfig(
        policy_mode=mode, state_dir=str(tmp_path / "state"), enable_project_memory=False
    )
    event = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "s1",
        "cwd": str(tmp_path),
    }
    decision = PolicyDecision(
        InterventionAction.REFRESH_CONTEXT,
        "degraded",
        metrics={"ter": 0.4, "baseline_ter": 0.7, "waste_ratio": 0.4},
        cooldown_seconds=120,
    )
    record = new_intervention_record(
        "s1", decision, MetricSnapshot(time.time(), 0.4, 0.4)
    )
    write_pending_intervention(
        pending_intervention_path(tmp_path, "s1"), record, decision
    )
    _, state, output = process_intervention_event(event, state, config)
    assert output["additionalContext"]
    assert output["systemMessage"]
    assert len(state.active_interventions) == 1


def test_pending_intervention_is_silent_but_recorded_in_observe_mode(
    tmp_path: Path,
) -> None:
    state = HookSessionState(session_id="s1")
    config = HookConfig(
        policy_mode="observe",
        state_dir=str(tmp_path / "state"),
        enable_project_memory=False,
    )
    event = {
        "hook_event_name": "PreToolUse",
        "session_id": "s1",
        "cwd": str(tmp_path),
        "tool_name": "Read",
        "tool_input": {"file_path": "a.py"},
    }
    decision = PolicyDecision(
        InterventionAction.REFRESH_CONTEXT,
        "degraded",
        metrics={"ter": 0.4, "baseline_ter": 0.7, "waste_ratio": 0.4},
        cooldown_seconds=120,
    )
    record = new_intervention_record(
        "s1", decision, MetricSnapshot(time.time(), 0.4, 0.4)
    )
    path = pending_intervention_path(tmp_path, "s1")
    write_pending_intervention(path, record, decision)
    _, state, output = process_intervention_event(event, state, config)
    assert output == {}
    assert len(state.active_interventions) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["consumed"] is True
