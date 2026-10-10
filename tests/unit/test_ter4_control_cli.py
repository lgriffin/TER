"""``python -m ter control``: measure, limits and chart end to end (point 201)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ter import bootstrap
from ter.adapters.driving.cli import main
from ter.adapters.driving.control_cli import session_files

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "golden" / "sessions"
SAMPLE = REPO / "sample_sessions"


def run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(argv, bootstrap.cli_services(), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("control") / "measures.json"
    code, _, err = run(
        ["control", "measure", str(GOLDEN), str(SAMPLE), "--out", str(out)]
    )
    assert code == 0, err
    assert "7 session(s) measured" in err
    return out


@pytest.fixture(scope="module")
def limits(measured: Path) -> Path:
    # Seven sessions are too few for limits, so pad the baseline by repeating
    # the measured sessions under new ids (same detector set).
    doc = json.loads(measured.read_text(encoding="utf-8"))
    extra = [
        {**s, "session": s["session"] + "-again", "started_at": None}
        for s in doc["sessions"]
    ]
    doc["sessions"] += extra
    measured.write_text(json.dumps(doc), encoding="utf-8")
    out = measured.parent / "limits.json"
    code, _, err = run(
        ["control", "limits", str(measured), "--out", str(out), "--date", "2026-10-10"]
    )
    assert code == 0, err
    assert "provisional" in err
    return out


@pytest.mark.req("TER-SPC-009")
def test_measure_writes_content_free_measures(measured: Path) -> None:
    doc = json.loads(measured.read_text(encoding="utf-8"))
    assert doc["schema"] == "ter.control-measures/1"
    assert len(doc["sessions"]) >= 7
    text = measured.read_text(encoding="utf-8")
    for session in GOLDEN.glob("*.jsonl"):
        first_prompt = next(
            (
                json.loads(line)["message"]["content"]
                for line in session.read_text(encoding="utf-8").splitlines()
                if '"type": "user"' in line or '"type":"user"' in line
            ),
            None,
        )
        if isinstance(first_prompt, str) and len(first_prompt) > 12:
            assert first_prompt not in text


@pytest.mark.req("TER-SPC-009")
def test_pseudonymise_replaces_session_ids(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    code, _, err = run(
        [
            "control",
            "measure",
            str(GOLDEN / "rework_loop.jsonl"),
            "--out",
            str(out),
            "--pseudonymise",
        ]
    )
    assert code == 0, err
    [session] = json.loads(out.read_text(encoding="utf-8"))["sessions"]
    assert session["session"].startswith("s-")


@pytest.mark.req("TER-SPC-001", "TER-SPC-003")
def test_chart_prints_limits_and_signals(measured: Path, limits: Path) -> None:
    code, out, _ = run(["control", "chart", str(measured), "--limits", str(limits)])
    assert code == 0
    assert out.startswith("Control charts · 14 sessions")
    assert "avoidable_share" in out and "never fires" in out


@pytest.mark.req("TER-SPC-010")
def test_chart_writes_html_and_json(
    measured: Path, limits: Path, tmp_path: Path
) -> None:
    html, js = tmp_path / "c.html", tmp_path / "c.json"
    code, _, err = run(
        [
            "control",
            "chart",
            str(measured),
            "--limits",
            str(limits),
            "--html",
            str(html),
            "--json",
            str(js),
        ]
    )
    assert code == 0, err
    assert "Control charts · 14 sessions" in err
    page = html.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "<script" not in page
    assert "Individuals (X)" in page and "Moving range (mR)" in page
    report = json.loads(js.read_text(encoding="utf-8"))
    assert report["schema"] == "ter.control-report/1" and not report["stale"]


@pytest.mark.req("TER-SPC-008")
def test_limits_keep_carries_a_hand_edit(
    measured: Path, limits: Path, tmp_path: Path
) -> None:
    doc = json.loads(limits.read_text(encoding="utf-8"))
    doc["measures"]["avoidable_share"]["tuned"] = {
        "ucl": 0.2,
        "lcl": None,
        "reason": "team target",
    }
    doc["measures"]["agent_seconds"]["enabled"] = False
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(doc), encoding="utf-8")
    again = tmp_path / "again.json"
    code, _, err = run(
        [
            "control",
            "limits",
            str(measured),
            "--out",
            str(again),
            "--keep",
            str(edited),
            "--method",
            "median",
        ]
    )
    assert code == 0, err
    kept = json.loads(again.read_text(encoding="utf-8"))
    assert kept["method"] == "median_moving_range"
    assert kept["measures"]["avoidable_share"]["tuned"]["reason"] == "team target"
    assert kept["measures"]["agent_seconds"]["enabled"] is False


@pytest.mark.req("TER-SPC-005")
def test_bad_limits_file_exits_2(measured: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "ter.control-limits/1"}), encoding="utf-8")
    code, _, err = run(["control", "chart", str(measured), "--limits", str(bad)])
    assert code == 2 and "ter control:" in err


@pytest.mark.req("TER-SPC-007")
def test_chart_warns_about_stale_limits(
    measured: Path, limits: Path, tmp_path: Path
) -> None:
    doc = json.loads(limits.read_text(encoding="utf-8"))
    doc["detectors"] = "000000000000"
    old = tmp_path / "old.json"
    old.write_text(json.dumps(doc), encoding="utf-8")
    code, out, err = run(["control", "chart", str(measured), "--limits", str(old)])
    assert code == 0
    assert "another detector set" in err and "stale" in out


@pytest.mark.req("TER-SPC-002")
def test_too_few_sessions_compute_no_limits(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    assert (
        run(["control", "measure", str(GOLDEN / "lean_mix.jsonl"), "--out", str(out)])[
            0
        ]
        == 0
    )
    code, _, err = run(
        ["control", "limits", str(out), "--out", str(tmp_path / "l.json")]
    )
    assert code == 1 and "enough sessions" in err


def test_missing_sources_and_empty_folders(tmp_path: Path) -> None:
    assert (
        run(["control", "measure", str(tmp_path / "nope"), "--out", "x.json"])[0] == 2
    )
    assert run(["control", "measure", str(tmp_path), "--out", "x.json"])[0] == 2
    assert (
        run(["control", "chart", str(tmp_path / "m.json"), "--limits", "l.json"])[0]
        == 2
    )


def test_unreadable_session_is_reported_and_exits_1(tmp_path: Path) -> None:
    (tmp_path / "broken.jsonl").write_text("{not json\n", encoding="utf-8")
    (tmp_path / "ok.jsonl").write_text(
        (GOLDEN / "lean_mix.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    out = tmp_path / "m.json"
    code, _, err = run(["control", "measure", str(tmp_path), "--out", str(out)])
    assert code == 1 and "broken.jsonl" in err and "Invalid JSON" in err
    assert len(json.loads(out.read_text(encoding="utf-8"))["sessions"]) == 1


def test_session_files_reads_a_corpus_and_skips_subagents(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions" / "p-1"
    (sessions / "abc" / "subagents").mkdir(parents=True)
    (sessions / "abc.jsonl").write_text("", encoding="utf-8")
    (sessions / "abc" / "subagents" / "agent-1.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert [p.name for p in session_files([tmp_path])] == ["abc.jsonl"]


@pytest.mark.req("TER-SPC-005")
def test_unreadable_measures_file_exits_2(tmp_path: Path) -> None:
    code, _, err = run(["control", "limits", str(tmp_path), "--out", "x.json"])
    assert code == 2 and "cannot read" in err
    bad = tmp_path / "latin.json"
    bad.write_bytes(b"\xff\xfe\x00")
    code, _, err = run(["control", "limits", str(bad), "--out", "x.json"])
    assert code == 2 and "cannot read" in err
