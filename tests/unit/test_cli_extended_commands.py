import json
from types import SimpleNamespace

import pytest

import ter_calculator.cli as cli
from ter_calculator.models import TERResult


def ns(**kwargs):
    base = dict(verbose=False, quiet=False)
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_main_help_and_dispatch(monkeypatch, capsys):
    assert cli.main([]) == 1
    assert "usage:" in capsys.readouterr().out
    monkeypatch.setattr(cli, "_cmd_analyze", lambda args: 7)
    assert cli.main(["analyze", "x.jsonl"]) == 7
    monkeypatch.setattr(cli, "_cmd_report", lambda args: 8)
    assert cli.main(["report", "x.jsonl"]) == 8
    monkeypatch.setattr(cli, "_cmd_list", lambda args: 9)
    assert cli.main(["list", "."]) == 9
    monkeypatch.setattr(cli, "_cmd_budget", lambda args: 10)
    assert cli.main(["budget", "task"]) == 10


def test_main_exception_paths(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_cmd_list", lambda args: (_ for _ in ()).throw(FileNotFoundError("gone"))
    )
    assert cli.main(["list", "."]) == 1
    assert "gone" in capsys.readouterr().err
    monkeypatch.setattr(
        cli, "_cmd_list", lambda args: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert cli.main(["list", "."]) == 1
    monkeypatch.setattr(
        cli, "_cmd_list", lambda args: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert cli.main(["--verbose", "list", "."]) == 1
    assert "Traceback" in capsys.readouterr().err


def test_cmd_analyze_missing_group_and_normal(monkeypatch, capsys):
    assert cli._cmd_analyze(ns(latest=False, session_path=None, group=False)) == 1
    monkeypatch.setattr(cli, "_cmd_analyze_group", lambda args: 4)
    assert cli._cmd_analyze(ns(latest=False, session_path="x", group=True)) == 4
    import ter_calculator.analyze_pipeline as ap
    import ter_calculator.formatter as fmt

    monkeypatch.setattr(
        ap, "analyze_session", lambda args: TERResult("s", 0.5, 0.5, {}, 1, 1, 0)
    )
    monkeypatch.setattr(fmt, "format_ter_result", lambda result, fmt: "FORMATTED")
    args = ns(latest=False, session_path="x", group=False, output_format="text")
    assert cli._cmd_analyze(args) == 0
    assert "FORMATTED" in capsys.readouterr().out


def test_cmd_report_stdout_file_latest(monkeypatch, tmp_path, capsys):
    import ter_calculator.analyze_pipeline as ap
    import ter_calculator.session_report as sr
    import ter_calculator.loader as ld

    monkeypatch.setattr(ap, "analyze_session", lambda args: object())
    monkeypatch.setattr(sr, "format_session_report_markdown", lambda result: "# report")
    assert cli._cmd_report(ns(latest=False, session_path=None, report_output=None)) == 1
    assert cli._cmd_report(ns(latest=False, session_path="x", report_output=None)) == 0
    assert "# report" in capsys.readouterr().out
    out = tmp_path / "r.md"
    assert (
        cli._cmd_report(ns(latest=False, session_path="x", report_output=str(out))) == 0
    )
    assert out.read_text() == "# report"
    monkeypatch.setattr(ld, "find_latest_session", lambda p: tmp_path / "latest.jsonl")
    args = ns(latest=True, session_path=None, report_output=None)
    assert cli._cmd_report(args) == 0 and args.session_path.endswith("latest.jsonl")


def test_cmd_list_text_json_and_errors(tmp_path, monkeypatch, capsys):
    import ter_calculator.loader as ld

    monkeypatch.setattr(
        ld, "discover_subagents", lambda p: [1, 2] if p.stem == "a" else []
    )
    assert (
        cli._cmd_list(
            ns(project_path=str(tmp_path / "missing"), output_format="text", limit=2)
        )
        == 1
    )
    assert (
        cli._cmd_list(ns(project_path=str(tmp_path), output_format="text", limit=2))
        == 0
    )
    assert "No sessions" in capsys.readouterr().out
    (tmp_path / "a.jsonl").write_text("abc")
    (tmp_path / "b.jsonl").write_text("def")
    assert (
        cli._cmd_list(ns(project_path=str(tmp_path), output_format="text", limit=2))
        == 0
    )
    assert "subagents" in capsys.readouterr().out
    assert (
        cli._cmd_list(ns(project_path=str(tmp_path), output_format="json", limit=2))
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 2


def test_cmd_budget_all_paths(monkeypatch, capsys):
    import ter_calculator.adaptive_budget as ab

    assert (
        cli._cmd_budget(
            ns(
                intent_text=" ",
                use_history=False,
                history_path=None,
                output_format="text",
            )
        )
        == 1
    )

    class EnumVal:
        def __init__(self, value):
            self.value = value

    rec = SimpleNamespace(
        complexity=EnumVal("simple"),
        model_tier=EnumVal("haiku"),
        max_thinking_tokens=10,
        estimated_total_tokens=20,
        estimated_cost_usd=0.1,
        confidence=0.9,
        reasoning="ok",
    )
    monkeypatch.setattr(ab, "recommend_budget", lambda text, history=None: rec)
    assert (
        cli._cmd_budget(
            ns(
                intent_text="task",
                use_history=False,
                history_path=None,
                output_format="json",
            )
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["complexity"] == "simple"
    assert (
        cli._cmd_budget(
            ns(
                intent_text="task",
                use_history=False,
                history_path=None,
                output_format="text",
            )
        )
        == 0
    )
    assert "Budget Recommendation" in capsys.readouterr().out
    monkeypatch.setattr(
        ab,
        "recommend_budget",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")),
    )
    assert (
        cli._cmd_budget(
            ns(
                intent_text="task",
                use_history=False,
                history_path=None,
                output_format="text",
            )
        )
        == 1
    )


def test_signal_serialization_and_print(capsys):
    class V:
        def __init__(self, value):
            self.value = value

    sig = SimpleNamespace(
        session_id="abcdefghijk",
        timestamp=0,
        is_live=True,
        aggregate_ter=0.87654,
        raw_ratio=0.8,
        message_index=2,
        drift=V("improving"),
        drift_magnitude=0.12345,
        warnings=["warn"],
        warning_level=V("high"),
        total_tokens=100,
        aligned_tokens=80,
        waste_tokens=20,
        phase_ter={"reasoning": 0.9},
        waste_sources={"repeat": 2},
    )
    data = cli._signal_to_dict(sig)
    assert data["ter"] == 0.8765 and data["tokens"]["waste"] == 20
    cli._print_signal(sig, "json")
    assert json.loads(capsys.readouterr().out)["is_live"] is True
    cli._last_was_live = False
    cli._print_signal(sig, "text")
    out = capsys.readouterr().out
    assert "LIVE" in out and "Phases" in out and "warn" in out
