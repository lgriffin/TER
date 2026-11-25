"""TER 4 analysis from a session file: explain, A3, route, context and
corpus import, as a developer runs them (TER-TST-006, TER-TST-007)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import FIXTURE, SAMPLE, Cli

pytestmark = pytest.mark.smoke


@pytest.mark.req("TER-TST-006")
def test_explain_json_names_its_format(cli: Cli) -> None:
    doc = json.loads(cli.ter4("explain", str(SAMPLE), "--json").ok().out)
    assert doc["schema"] == "ter.lean/0.1"
    assert doc["session_id"] == "sample-session-001" and doc["findings"]
    assert "finding" in cli.ter4("explain", str(SAMPLE)).ok().out.lower()


@pytest.mark.req("TER-TST-006")
def test_a3_writes_html_and_json(cli: Cli, workdir: Path) -> None:
    cli.ter4("a3", str(SAMPLE), "--html", "a3.html", "--json", "a3.json").ok()
    page = (workdir / "a3.html").read_text(encoding="utf-8")
    assert page.lstrip().lower().startswith("<!doctype html") and "</html>" in page
    a3 = json.loads((workdir / "a3.json").read_text(encoding="utf-8"))
    assert a3["schema"] == "ter.a3/0.1" and a3["countermeasures"]


@pytest.mark.req("TER-TST-006")
def test_route_json_names_its_format(cli: Cli) -> None:
    doc = json.loads(cli.ter4("route", str(SAMPLE), "--json").ok().out)
    assert doc["schema"] == "ter.route/0.1"
    assert cli.ter4("route", str(FIXTURE)).ok().out.strip()


@pytest.mark.req("TER-TST-006")
def test_context_bundle_and_report(cli: Cli, workdir: Path) -> None:
    repo = workdir / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def main():\n    return 1\n")
    bundle = cli.ter4(
        "context",
        "bundle",
        str(SAMPLE),
        "--repo",
        str(repo),
        "--prompt",
        "fix main",
        "--json",
        "--no-record",
    ).ok()
    assert json.loads(bundle.out)["schema"] == "ter.context-bundle/1"
    report = cli.ter4("context", "report", str(SAMPLE), "--repo", str(repo), "--json")
    assert json.loads(report.ok().out)["schema"] == "ter.context-measures/1"


@pytest.mark.req("TER-TST-006")
def test_corpus_import_redacts_into_a_manifest(cli: Cli, workdir: Path) -> None:
    cli.ter4("corpus", "import", str(SAMPLE), "--out", "corpus").ok()
    manifest = json.loads((workdir / "corpus" / "manifest.json").read_text())
    assert manifest["schema"] == "ter.corpus-manifest/1"
    [session] = (workdir / "corpus" / "sessions").rglob("*.jsonl")
    # The redacted copy is still a session TER reads.
    cli.ter4("explain", str(session), "--json").ok()


@pytest.mark.req("TER-TST-006", "TER-REC-001", "TER-REC-005")
def test_record_export_then_verify(cli: Cli, workdir: Path) -> None:
    cli.ter4("record", "export", str(SAMPLE), "--out", "s.record.json").ok()
    record = json.loads((workdir / "s.record.json").read_text(encoding="utf-8"))
    assert record["schema"] == "ter.record/0.1"
    assert (
        "Verified"
        in cli.ter4("record", "verify", "s.record.json", str(SAMPLE)).ok().out
    )
    other = cli.ter4("record", "verify", "s.record.json", str(FIXTURE))
    assert other.code == 1 and "Not the same session" in other.out


# --- failures are reported, never a traceback (TER-TST-007) ---------------


def _refused(result_code: int, err: str) -> None:
    assert result_code != 0
    assert err.strip(), "say why on stderr"
    assert "Traceback" not in err, err


@pytest.mark.req("TER-TST-007")
@pytest.mark.parametrize("command", ["observe", "explain", "a3", "route"])
def test_a_missing_session_is_refused(cli: Cli, command: str) -> None:
    result = cli.ter4(command, "no-such-session.jsonl")
    _refused(result.code, result.err)


@pytest.mark.req("TER-TST-007")
@pytest.mark.parametrize("command", ["observe", "explain", "a3", "route"])
@pytest.mark.parametrize(
    ("content", "why"),
    [
        (b"not json\n", "Invalid JSON on line 1"),
        (b'{"type": "user"\n', "Invalid JSON on line 1"),
        (b"\x00\x01binary", "Invalid JSON on line 1"),
        (b"\xff\xfe{}\n", "can't decode"),
        (b"", "Session file is empty"),
    ],
    ids=["text", "torn", "binary", "not-utf8", "empty"],
)
def test_a_malformed_session_is_refused(
    cli: Cli, workdir: Path, command: str, content: bytes, why: str
) -> None:
    path = workdir / f"bad-{content.hex()[:16] or 'empty'}.jsonl"
    path.write_bytes(content)
    result = cli.ter4(command, str(path))
    _refused(result.code, result.err)
    assert result.code == 2, result.err
    assert f"Cannot read session {path}" in result.err and why in result.err


@pytest.mark.req("TER-TST-007")
def test_a_folder_that_is_not_a_gare_run_is_refused(cli: Cli, workdir: Path) -> None:
    run = workdir / "not-a-run"
    run.mkdir()
    (run / "explain.json").write_text("{}")
    result = cli.ter4("explain", str(run))
    _refused(result.code, result.err)
    assert result.code == 2 and "is not `gare explain --json` output" in result.err
