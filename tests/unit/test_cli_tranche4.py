from types import SimpleNamespace
from pathlib import Path
import json
import ter_calculator.cli as cli
from ter_calculator.models import TERResult


def ns(**kw):
    base = dict(
        quiet=True,
        verbose=False,
        latest=False,
        session_path="parent.jsonl",
        group=True,
        similarity_threshold=0.4,
        confidence_threshold=0.75,
        restatement_threshold=0.85,
        phase_weights="0.3,0.4,0.3",
        cost_model="sonnet",
        no_waste_patterns=False,
        output_format="text",
        sort="ter",
        baseline=False,
        session_paths=[],
        project_path=None,
        limit=10,
        intent_text="task",
        use_history=False,
        history_path=None,
        poll_interval=0.01,
        stream=True,
        log_file=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def result(s="s", ter=0.5, tokens=10, waste=2):
    return TERResult(s, ter, ter, {}, tokens, tokens - waste, waste)


def patch_analysis(monkeypatch):
    import ter_calculator.loader as ld, ter_calculator.intent as intent, ter_calculator.classifier as classifier
    import ter_calculator.compute as compute, ter_calculator.waste as waste, ter_calculator.economics as eco

    monkeypatch.setattr(
        ld, "load_session", lambda p: SimpleNamespace(session_id=Path(str(p)).stem)
    )
    monkeypatch.setattr(ld, "segment_spans", lambda s: ["span"])
    monkeypatch.setattr(intent, "extract_intent", lambda s: "intent")
    monkeypatch.setattr(classifier, "classify_spans", lambda *a, **k: ["classified"])
    monkeypatch.setattr(
        compute, "compute_ter", lambda c, session_id, intent, **k: result(session_id)
    )
    monkeypatch.setattr(waste, "detect_waste_patterns", lambda *a, **k: ["w"])
    monkeypatch.setattr(eco, "compute_economics", lambda *a, **k: SimpleNamespace())


def test_analyze_group_fallback_and_success(monkeypatch, capsys, tmp_path):
    import ter_calculator.loader as ld, ter_calculator.formatter as fmt

    monkeypatch.setattr(ld, "discover_subagents", lambda p: [])
    monkeypatch.setattr(cli, "_cmd_analyze", lambda a: 7)
    a = ns()
    assert cli._cmd_analyze_group(a) == 7 and a.group is False
    patch_analysis(monkeypatch)
    sub = tmp_path / "sub.jsonl"
    sub.write_text("")
    monkeypatch.setattr(ld, "discover_subagents", lambda p: [sub])
    monkeypatch.setattr(fmt, "format_grouped_analysis", lambda p, s, fmt: "GROUP")
    assert cli._cmd_analyze_group(ns()) == 0
    assert "GROUP" in capsys.readouterr().out


def test_compare_empty_baseline_and_regular(monkeypatch, tmp_path, capsys):
    import ter_calculator.formatter as fmt

    assert cli._cmd_compare(ns(session_paths=[str(tmp_path)], baseline=False)) == 1
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    f1.write_text("")
    f2.write_text("")
    assert cli._cmd_compare(ns(session_paths=[str(f1)], baseline=True)) == 1
    import ter_calculator.analyze_pipeline as ap, ter_calculator.session_report as sr

    monkeypatch.setattr(ap, "default_analyze_args", lambda p: p)
    monkeypatch.setattr(ap, "analyze_session", lambda p: result(str(p)))
    monkeypatch.setattr(sr, "format_baseline_markdown", lambda a, b: "BASE")
    assert cli._cmd_compare(ns(session_paths=[str(f1), str(f2)], baseline=True)) == 0
    assert "BASE" in capsys.readouterr().out
    patch_analysis(monkeypatch)
    monkeypatch.setattr(fmt, "format_comparison", lambda rs, fmt: "COMPARE")
    for sort in ("ter", "tokens", "waste"):
        assert (
            cli._cmd_compare(
                ns(session_paths=[str(tmp_path)], baseline=False, sort=sort)
            )
            == 0
        )
    assert "COMPARE" in capsys.readouterr().out


def test_list_default_home_limit_and_subagent_skip(monkeypatch, tmp_path, capsys):
    import ter_calculator.loader as ld

    home = tmp_path / "home"
    projects = home / ".claude/projects"
    projects.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert cli._cmd_list(ns(project_path=None, limit=1)) == 0
    p = projects / "p"
    p.mkdir()
    (p / "a.jsonl").write_text("a")
    sd = p / "subagents"
    sd.mkdir()
    (sd / "x.jsonl").write_text("x")
    monkeypatch.setattr(ld, "discover_subagents", lambda p: [1])
    assert cli._cmd_list(ns(project_path=None, limit=1)) == 0
    assert "1 session" in capsys.readouterr().out


def test_budget_history_warning_and_verbose_error(monkeypatch, capsys):
    import ter_calculator.adaptive_budget as ab

    monkeypatch.setattr(
        ab,
        "HistoricalBudgetAnalyzer",
        lambda **k: (_ for _ in ()).throw(RuntimeError("history")),
    )
    rec = SimpleNamespace(
        complexity=SimpleNamespace(value="simple"),
        model_tier=SimpleNamespace(value="haiku"),
        max_thinking_tokens=1,
        estimated_total_tokens=2,
        estimated_cost_usd=0.1,
        confidence=0.8,
        reasoning="r",
    )
    monkeypatch.setattr(ab, "recommend_budget", lambda *a, **k: rec)
    assert cli._cmd_budget(ns(use_history=True, history_path="x")) == 0
    assert "Could not load history" in capsys.readouterr().err
    monkeypatch.setattr(
        ab,
        "recommend_budget",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert cli._cmd_budget(ns(verbose=True)) == 1
    assert "Traceback" in capsys.readouterr().err


def test_print_signal_zero_history_and_log(tmp_path, capsys):
    sig = SimpleNamespace(
        session_id="abcdefgh",
        timestamp=0,
        is_live=False,
        aggregate_ter=0.5,
        raw_ratio=0.4,
        message_index=1,
        drift=SimpleNamespace(value="stable"),
        drift_magnitude=0,
        warnings=[],
        warning_level=SimpleNamespace(value="none"),
        total_tokens=0,
        aligned_tokens=0,
        waste_tokens=0,
        phase_ter={},
        waste_sources={"x": 0},
    )
    log = tmp_path / "s.log"
    with log.open("w") as fh:
        cli._print_signal(sig, "text", fh)
    assert (
        "HISTORY" in capsys.readouterr().out
        and json.loads(log.read_text())["tokens"]["total"] == 0
    )


def test_watch_latest_directory_log_and_runtime_error(monkeypatch, tmp_path, capsys):
    import ter_calculator.real_time as rt, ter_calculator.loader as ld

    f = tmp_path / "s.jsonl"
    f.write_text("")
    monkeypatch.setattr(rt, "load_embedding_model", lambda: object())
    monkeypatch.setattr(ld, "find_latest_session", lambda p: f)

    class Monitor:
        def __init__(self, *a, on_signal=None, **k):
            self.on_signal = on_signal

        def run(self):
            raise KeyboardInterrupt

        def stop(self):
            pass

    monkeypatch.setattr(rt, "SessionMonitor", Monitor)
    log = tmp_path / "signals.log"
    assert (
        cli._cmd_watch(
            ns(
                latest=True,
                project_path=str(tmp_path),
                log_file=str(log),
                stream=True,
                quiet=False,
            )
        )
        == 0
    )

    class Boom(Monitor):
        def run(self):
            raise RuntimeError("run")

    monkeypatch.setattr(rt, "SessionMonitor", Boom)
    assert cli._cmd_watch(ns(project_path=str(f), stream=True, verbose=True)) == 1
    assert "Traceback" in capsys.readouterr().err


def test_watch_directory_live_dashboard_interrupt(monkeypatch, tmp_path):
    import ter_calculator.real_time as rt

    monkeypatch.setattr(rt, "load_embedding_model", lambda: object())

    class Dash:
        def __init__(self, *a, **k):
            pass

        def run(self):
            raise KeyboardInterrupt

        def stop(self):
            pass

    monkeypatch.setattr(rt, "LiveDashboard", Dash)
    assert (
        cli._cmd_watch(
            ns(project_path=str(tmp_path), stream=True, output_format="json")
        )
        == 0
    )
