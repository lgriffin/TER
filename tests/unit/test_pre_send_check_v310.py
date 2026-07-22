from __future__ import annotations

import io
import json
from pathlib import Path

from ter_calculator.cli import main
from ter_calculator.hook_monitor import HookConfig, HookSessionState
from ter_calculator.intervention import (
    check_pre_send_pattern,
    process_intervention_event,
)
from ter_calculator.repository_memory import build_index


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "client.py").write_text(
        "def retry_request(client, request):\n"
        "    for attempt in range(3):\n"
        "        try:\n"
        "            return client.send(request)\n"
        "        except TimeoutError:\n"
        "            continue\n",
        encoding="utf-8",
    )
    build_index(root)
    return root


def test_pre_send_check_match_and_no_match(tmp_path: Path) -> None:
    root = _project(tmp_path)
    config = HookConfig(
        pre_send_check_enabled=True,
        pre_send_similarity_threshold=0.20,
        pre_send_cooldown_seconds=0,
        policy_mode="block",
    )
    state = HookSessionState(session_id="s")
    alert, status = check_pre_send_pattern(
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(root),
            "prompt": "Implement retry request client send TimeoutError three attempts",
        },
        state,
        config,
    )
    assert status is None
    assert alert is not None
    assert alert.pattern_type == "pre_send_check"
    assert "client.py" in alert.message

    state = HookSessionState(session_id="none")
    alert, status = check_pre_send_pattern(
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(root),
            "prompt": "write a haiku about mountains",
        },
        state,
        HookConfig(
            pre_send_check_enabled=True,
            pre_send_similarity_threshold=0.99,
            pre_send_cooldown_seconds=0,
        ),
    )
    assert alert is None
    assert status == "no_match"


def test_pre_send_acknowledged_and_overridden(tmp_path: Path) -> None:
    root = _project(tmp_path)
    for marker, expected in (
        ("[TER ACK pre_send_check]", "acknowledged"),
        ("[TER OVERRIDE pre_send_check]", "overridden"),
    ):
        state = HookSessionState(
            session_id=expected,
            pre_send_pending={"check_id": "c1", "reference": "client.py:1"},
        )
        _, status = check_pre_send_pattern(
            {
                "hook_event_name": "UserPromptSubmit",
                "cwd": str(root),
                "prompt": f"{marker} proceed with the reviewed change",
            },
            state,
            HookConfig(pre_send_check_enabled=True),
        )
        assert status == expected


def test_pre_send_block_integration_and_outcomes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root = _project(tmp_path)
    event = {
        "session_id": "integration",
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(root),
        "prompt": "Implement retry request client send TimeoutError three attempts",
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert (
        main(
            [
                "hook",
                "monitor",
                "--pre-send-check-enabled",
                "--pre-send-similarity-threshold",
                "0.20",
                "--pre-send-cooldown-seconds",
                "0",
                "--policy-mode",
                "block",
                "--state-dir",
                str(tmp_path / "state"),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert output["hookSpecificOutput"]["decision"] == "block"
    rows = [
        json.loads(line)
        for line in (root / ".ter" / "intervention-outcomes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[-1]["intervention_type"] == "pre_send_check"
    assert rows[-1]["outcome"] == "fired"


def test_process_pre_send_ack_clears_pending(tmp_path: Path) -> None:
    root = _project(tmp_path)
    state = HookSessionState(
        session_id="ack", pre_send_pending={"check_id": "x", "reference": "a.py:1"}
    )
    _, state, output = process_intervention_event(
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(root),
            "prompt": "[TER ACK pre_send_check] reviewed",
        },
        state,
        HookConfig(pre_send_check_enabled=True, enable_project_memory=False),
    )
    assert state.pre_send_pending == {}
    assert "hookSpecificOutput" not in output
