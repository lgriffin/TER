"""Tests for ``python -m ter`` (driving CLI adapter and composition root)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ter import bootstrap
from ter.adapters.driven.event_log import JsonlEventLog
from ter.adapters.driven.tokenizers import RegexTokenizer, TiktokenTokenizer
from ter.adapters.driving.cli import format_timeline, main

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "sample_sessions" / "example_session.jsonl"
HOOKS = REPO / "tests" / "fixtures" / "hooks"


def run(argv: list[str], stdin: str = "") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(
        argv,
        bootstrap.cli_services(),
        stdin=io.StringIO(stdin),
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


def test_observe_prints_a_summary_and_timeline() -> None:
    code, out, _ = run(["observe", str(SAMPLE), "--timeline", "--limit", "3"])
    assert code == 0
    assert "TER observe · session sample-session-001" in out
    assert "repeated reads     1  src/app.py ×2" in out
    assert "Timeline" in out and "… 11 more" in out


def test_observe_json_is_the_report_dict() -> None:
    code, out, _ = run(["observe", str(SAMPLE), "--json"])
    data = json.loads(out)
    assert code == 0
    assert data["total_events"] == 14
    assert data["tokenizer"] == "regex-v1"


def test_observe_missing_file_and_no_target(tmp_path: Path) -> None:
    code, _, err = run(["observe", str(tmp_path / "nope.jsonl")])
    assert code == 2 and "No such session file" in err
    code, _, err = run(["observe"])
    assert code == 2 and "needs a session file" in err


@pytest.mark.req("TER-OBS-008")
def test_hook_then_observe_event_log(tmp_path: Path) -> None:
    for name in ("session_start", "user_prompt_submit", "post_tool_use_read"):
        payload = (HOOKS / f"{name}.json").read_text(encoding="utf-8")
        code, out, _ = run(["hook", "--event-log", str(tmp_path)], stdin=payload)
        assert code == 0 and out == "{}\n"
    session = "3f0c9a1e-hook-demo"
    assert JsonlEventLog(tmp_path).sessions() == (session,)

    code, out, _ = run(["observe", "--event-log", str(tmp_path), "--json"])
    data = json.loads(out)
    assert code == 0 and data["total_events"] == 3
    code, out, _ = run(
        ["observe", "--event-log", str(tmp_path), "--session", session, "--timeline"]
    )
    assert code == 0 and "tool.completed" in out


def test_observe_event_log_needs_a_session_when_ambiguous(tmp_path: Path) -> None:
    code, _, err = run(["observe", "--event-log", str(tmp_path)])
    assert code == 2 and "0 sessions" in err and "(none)" in err


@pytest.mark.req("TER-OBS-008")
def test_hook_with_garbage_still_succeeds(tmp_path: Path) -> None:
    code, out, err = run(["hook", "--event-log", str(tmp_path)], stdin="{oops")
    assert code == 0 and out == "{}\n"
    assert err.startswith("ter hook: event not recorded: invalid JSON")
    assert not tmp_path.joinpath("x").exists()


@pytest.mark.req("TER-OBS-008")
def test_hook_that_cannot_record_says_why_on_stderr(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding="utf-8")
    payload = (HOOKS / "user_prompt_submit.json").read_text(encoding="utf-8")
    code, out, err = run(["hook", "--event-log", str(blocked)], stdin=payload)
    assert code == 0 and out == "{}\n"
    assert err.startswith("ter hook: event not recorded: ")


def test_format_timeline_without_limit() -> None:
    from ter.application import AnalyseTrace
    from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource

    report = AnalyseTrace(ClaudeCodeJsonlSource(), RegexTokenizer())(SAMPLE)
    text = format_timeline(report)
    assert text.count("\n") == 2 + report.total_events
    assert "more" not in text


class TestBootstrap:
    def test_event_log_dir_follows_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(bootstrap.EVENT_LOG_ENV, str(tmp_path))
        assert bootstrap.default_event_log_dir() == tmp_path
        monkeypatch.delenv(bootstrap.EVENT_LOG_ENV)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        assert bootstrap.default_event_log_dir() == tmp_path / "cache/ter/events"
        monkeypatch.delenv("XDG_CACHE_HOME")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        assert bootstrap.default_event_log_dir() == tmp_path / "home/.cache/ter/events"

    def test_make_tokenizer(self) -> None:
        assert isinstance(bootstrap.make_tokenizer("regex"), RegexTokenizer)
        assert isinstance(bootstrap.make_tokenizer("tiktoken"), TiktokenTokenizer)
        with pytest.raises(ValueError, match="Unknown tokenizer"):
            bootstrap.make_tokenizer("nope")

    def test_main_entry(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert bootstrap.main(["observe", str(SAMPLE)]) == 0
        assert "TER observe" in capsys.readouterr().out


@pytest.mark.req("TER-REQ-013")
def test_version_names_the_package_and_event_schema(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from importlib import metadata

    from ter import EVENT_SCHEMA_VERSION

    with pytest.raises(SystemExit) as exit_:
        run(["--version"])
    assert exit_.value.code == 0
    expected = f"python -m ter {metadata.version('ter-calculator')} (events {EVENT_SCHEMA_VERSION})"
    assert capsys.readouterr().out.strip() == expected


@pytest.mark.req("TER-OBS-008")
def test_python_dash_m_ter(tmp_path: Path) -> None:
    payload = (HOOKS / "post_tool_use_bash.json").read_text(encoding="utf-8")
    env = {"PYTHONPATH": str(REPO / "src"), "TER_EVENT_LOG_DIR": str(tmp_path)}
    hook = subprocess.run(
        [sys.executable, "-m", "ter", "hook"],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert hook.returncode == 0 and hook.stdout == "{}\n", hook.stderr
    observed = subprocess.run(
        [sys.executable, "-m", "ter", "observe", "--event-log", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert observed.returncode == 0, observed.stderr
    assert "exec.shell 1" in observed.stdout
