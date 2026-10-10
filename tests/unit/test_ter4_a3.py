"""Unit tests for the A3 renderer and the `a3` / `explain` commands."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from xml.etree import ElementTree

import pytest
from ter4_lean_builder import FAIL, Script

from ter.adapters.driven.in_memory import FixedTerScorer, InMemorySessionSource
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.adapters.driving.cli import CliServices, format_findings
from ter.adapters.driving.cli import main as cli_main
from ter.adapters.driving.reports import render_a3_html
from ter.adapters.driving.reports.a3 import (
    activity_bar,
    flow_bar,
    fmt_seconds,
    pareto,
    value_stream_map,
)
from ter.application import ExplainSession
from ter.bootstrap import cli_services, make_ter_scorer
from ter.bootstrap import main as ter4_main
from ter.domain import SessionTrace
from ter.domain.lean import A3Report, build_a3, explain

SESSIONS = Path(__file__).resolve().parents[1] / "golden" / "sessions"
CORPUS = {p.stem: p for p in SESSIONS.glob("*.jsonl")}

SVG = re.compile(r"<svg .*?</svg>", re.S)


def _report(
    script: Script, intents: tuple[str, ...] = ("Fix <b>it</b> & more",)
) -> A3Report:
    return build_a3(explain(script.events, RegexTokenizer()), intents)


def _messy() -> Script:
    s = Script()
    s.prompt("Fix <b>it</b>")
    s.read("src/a.py", "def parse(): pass")
    s.read("src/a.py", "def parse(): pass")
    s.read("src/unused.py", "def other(): pass")
    s.bash("pytest -q", FAIL)
    s.edit("src/a.py", "1")
    s.bash("pytest -q", FAIL)
    s.say("Done with a.py <script>alert(1)</script>")
    return s


@pytest.mark.req("TER-RPT-004")
def test_html_is_self_contained_and_escaped() -> None:
    page = render_a3_html(_report(_messy()))
    assert page.startswith("<!doctype html>")
    assert "<script" not in page.lower()
    assert "http://" not in page.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "https://" not in page
    assert "default-src 'none'" in page
    assert "&lt;b&gt;it&lt;/b&gt;" in page
    for number, name in enumerate(
        [
            "Background",
            "Current state",
            "Analysis",
            "Root causes",
            "Countermeasures",
            "Follow-up",
        ],
        1,
    ):
        assert f'<span class="n">{number}</span>{name}' in page
    assert "prefers-color-scheme: dark" in page
    assert "@page{size:A3 landscape" in page


@pytest.mark.req("TER-RPT-004")
def test_every_chart_is_accessible_well_formed_svg() -> None:
    page = render_a3_html(_report(_messy()))
    charts = SVG.findall(page)
    assert len(charts) >= 4
    for svg in charts:
        root = ElementTree.fromstring(svg)
        assert root.get("role") == "img"
        title_id, desc_id = (root.get("aria-labelledby") or "").split()
        ns = "{http://www.w3.org/2000/svg}"
        assert root.find(f"{ns}title").get("id") == title_id  # type: ignore[union-attr]
        assert root.find(f"{ns}desc").get("id") == desc_id  # type: ignore[union-attr]


@pytest.mark.req("TER-RPT-004")
def test_charts_and_empty_states() -> None:
    clean = Script()
    clean.prompt("x")
    clean.say("y")
    report = _report(clean, ())
    page = render_a3_html(report)
    assert "No findings" in page and "Nothing to change" in page
    assert "No confident waste" in page
    assert "No prompt was recorded" in page
    assert pareto(report) == ""
    assert value_stream_map(()) == ""
    assert "Value-adding" in activity_bar(report)
    assert "Progressing" in flow_bar(report)
    assert "Progressing" in flow_bar(report, time=True)
    vsm = value_stream_map(_report(_messy()).analysis.value_stream)
    assert "Explore" in vsm and "waste" in vsm


@pytest.mark.req("TER-RPT-004")
def test_many_prompts_and_findings_are_summarised() -> None:
    s = Script()
    for i in range(4):
        s.prompt(f"prompt {i}")
    for i in range(8):
        s.bash(f"ls dir{i}", "a")
        s.bash(f"ls dir{i}", "a")
    s.say("done")
    page = render_a3_html(_report(s, tuple(f"p{i}" for i in range(4))))
    assert "1 more prompt(s)" in page
    assert "more finding(s) in the JSON output" in page


def test_fmt_seconds() -> None:
    assert fmt_seconds(4.4) == "4s"
    assert fmt_seconds(200) == "3m 20s"
    assert fmt_seconds(3900) == "1h 05m"


def _services(trace: SessionTrace) -> CliServices:
    use_case = ExplainSession(
        InMemorySessionSource({"s.jsonl": trace}),
        RegexTokenizer(),
        FixedTerScorer({"s.jsonl": 0.42}, "fixed"),
    )
    base = cli_services()
    return CliServices(
        analyse_transcript=base.analyse_transcript,
        log_sessions=base.log_sessions,
        analyse_log=base.analyse_log,
        hook_ingest=base.hook_ingest,
        default_log_dir=base.default_log_dir,
        explain_transcript=lambda path, tok, ter, outcome=None: use_case(path.name),
    )


@pytest.mark.req("TER-RPT-003", "TER-GRF-003")
def test_a3_command_writes_html_json_and_graph(tmp_path: Path) -> None:
    session = tmp_path / "s.jsonl"
    session.write_text("{}", encoding="utf-8")
    trace = SessionTrace("s", "script", tuple(_messy().events))
    out, err = io.StringIO(), io.StringIO()
    html_path, json_path, graph = (
        tmp_path / "o" / "a3.html",
        tmp_path / "a3.json",
        tmp_path / "g.json",
    )
    code = cli_main(
        [
            "a3",
            str(session),
            "--html",
            str(html_path),
            "--json",
            str(json_path),
            "--graph",
            str(graph),
        ],
        _services(trace),
        stdout=out,
        stderr=err,
    )
    assert code == 0
    assert "<!doctype html>" in html_path.read_text(encoding="utf-8")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["analysis"]["scorecard"]["ter"] == {"value": 0.42, "method": "fixed"}
    assert json.loads(graph.read_text(encoding="utf-8"))["schema"] == "ter.evidence/0.1"
    assert "Wrote" in err.getvalue()

    out = io.StringIO()
    cli_main(["a3", str(session), "--json"], _services(trace), stdout=out, stderr=err)
    assert json.loads(out.getvalue())["schema"] == "ter.a3/0.1"
    out = io.StringIO()
    cli_main(["a3", str(session)], _services(trace), stdout=out, stderr=err)
    assert out.getvalue().startswith("TER explain")


@pytest.mark.req("TER-ANL-021")
def test_explain_command(tmp_path: Path) -> None:
    session = tmp_path / "s.jsonl"
    session.write_text("{}", encoding="utf-8")
    trace = SessionTrace("s", "script", tuple(_messy().events))
    out = io.StringIO()
    assert (
        cli_main(["explain", str(session), "--json"], _services(trace), stdout=out) == 0
    )
    data = json.loads(out.getvalue())
    assert data["schema"] == "ter.lean/0.1" and "evidence_graph" in data
    out = io.StringIO()
    cli_main(["explain", str(session)], _services(trace), stdout=out)
    assert "Re-read src/a.py" in out.getvalue()
    assert "uncertain, counted until verified" in out.getvalue()


def test_explain_errors(tmp_path: Path) -> None:
    err = io.StringIO()
    base = cli_services()
    assert cli_main(["explain", str(tmp_path / "missing.jsonl")], base, stderr=err) == 2
    assert "No such session file" in err.getvalue()
    bare = CliServices(
        base.analyse_transcript,
        base.log_sessions,
        base.analyse_log,
        base.hook_ingest,
        base.default_log_dir,
    )
    err = io.StringIO()
    assert cli_main(["a3", str(tmp_path / "x.jsonl")], bare, stderr=err) == 2
    assert "not available" in err.getvalue()


@pytest.mark.req("TER-RPT-003")
def test_real_wiring_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    html_path = tmp_path / "a3.html"
    assert (
        ter4_main(
            ["a3", str(CORPUS["rework_loop"]), "--html", str(html_path), "--ter", "off"]
        )
        == 0
    )
    assert "Fix attempt did not change the failure" in html_path.read_text(
        encoding="utf-8"
    )
    assert make_ter_scorer("off") is None
    assert make_ter_scorer("model") is not None
    with pytest.raises(ValueError, match="Unknown TER mode"):
        make_ter_scorer("bogus")


def test_ter3_cli_delegates_a3(capsys: pytest.CaptureFixture[str]) -> None:
    from ter_calculator.cli import main as ter3_main

    assert ter3_main(["explain", str(CORPUS["duplicate_exploration"])]) == 0
    assert "Re-read src/cli.py" in capsys.readouterr().out


def test_format_findings_without_ter() -> None:
    text = format_findings(explain(_messy().events, RegexTokenizer()))
    assert (
        "flow efficiency" in text
        and "TER " not in text.split("findings")[0].split("activity")[1]
    )


@pytest.mark.req("TER-RPT-003")
@pytest.mark.parametrize("command", ["a3", "explain"])
@pytest.mark.parametrize("flag", ["--quiet", "--verbose"])
def test_ter3_cli_rejects_global_flags_before_delegated_commands(
    command: str, flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from ter_calculator.cli import main as ter3_main

    with pytest.raises(SystemExit) as exited:
        ter3_main([flag, command, str(CORPUS["duplicate_exploration"])])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert f"{flag} cannot be used with {command}" in err


@pytest.mark.req("TER-RPT-004")
def test_value_stream_total_is_real_when_nothing_was_generated() -> None:
    s = Script()
    s.prompt("Only a prompt")
    stages = explain(s.events, RegexTokenizer()).value_stream
    assert sum(st.tokens for st in stages) == 0
    vsm = value_stream_map(stages)
    assert "· 0 generated tokens ·" in vsm
    assert "· 1 generated tokens" not in vsm
