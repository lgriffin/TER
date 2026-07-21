"""Native transcript metrics through the real ``ter hook monitor`` CLI path."""

from __future__ import annotations
import io, json
from pathlib import Path
from ter_calculator.cli import main


def _row(uid, blocks):
    return json.dumps(
        {
            "uuid": uid,
            "sessionId": "native",
            "type": "assistant",
            "message": {"role": "assistant", "content": blocks},
        }
    )


def _call(monkeypatch, capsys, payload, state_dir, mode):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    code = main(
        [
            "--quiet",
            "hook",
            "monitor",
            "--policy-mode",
            mode,
            "--state-dir",
            str(state_dir),
            "--no-project-memory",
            "--ter-drop-warning",
            "0.01",
            "--ter-drop-replan",
            "1.0",
            "--waste-ratio-warning",
            "0.01",
            "--waste-ratio-replan",
            "1.0",
            "--degraded-windows-required",
            "1",
            "--refresh-cooldown-seconds",
            "120",
        ]
    )
    assert code == 0
    return json.loads(capsys.readouterr().out)


def _exercise(tmp_path, monkeypatch, capsys, mode):
    root = tmp_path / mode
    root.mkdir()
    state_dir = root / "state"
    transcript = root / "session.jsonl"
    transcript.write_text(
        "\n".join(
            _row(str(i), [{"type": "text", "text": f"Verified step {i}."}])
            for i in range(3)
        )
        + "\n"
    )
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": f"native-{mode}",
        "cwd": str(root),
        "prompt": "Continue",
        "transcript_path": str(transcript),
    }
    for _ in range(3):
        _call(monkeypatch, capsys, payload, state_dir, mode)
    tool = {
        "type": "tool_use",
        "name": "Read",
        "input": {"file_path": "src/ter_calculator/intervention.py"},
    }
    thought = {
        "type": "thinking",
        "thinking": "Inspect the same implementation and repeat the same approach.",
    }
    with transcript.open("a") as h:
        for i in range(8):
            h.write(_row(f"w{i}", [tool, thought]) + "\n")
    triggered = _call(monkeypatch, capsys, payload, state_dir, mode)
    consumed = _call(monkeypatch, capsys, payload, state_dir, mode)
    state = json.loads((state_dir / f"native-{mode}.json").read_text())
    pending = json.loads(
        (
            root / ".ter/runtime" / f"native-{mode}" / "pending-intervention.json"
        ).read_text()
    )
    assert (
        state["transcript_offset"] == transcript.stat().st_size
        and state["transcript_total_tokens"] > 0
    )
    assert (
        state["transcript_last_ter"] < 1
        and state["transcript_repeated_tool_calls"] > 0
        and state["reasoning_loop_streak"] > 0
    )
    assert state["active_interventions"] and pending["consumed"] is True
    return triggered, consumed


def test_native_transcript_payload_warn_and_observe_modes(
    tmp_path, monkeypatch, capsys
):
    a, b = _exercise(tmp_path, monkeypatch, capsys, "warn")
    assert "additionalContext" in a or "additionalContext" in b
    assert "systemMessage" in a or "systemMessage" in b
    a, b = _exercise(tmp_path, monkeypatch, capsys, "observe")
    assert a == {} and b == {}
