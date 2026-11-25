"""``python -m ter record export|verify`` (TER-REC-001, TER-REC-005, ADR 0007)."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ter import bootstrap
from ter.adapters.driving.cli import main as cli_main
from ter.adapters.driving.record_cli import SHOWN, format_check
from ter.application.export_record import RecordCheck, ter_mode_of

from tests.golden.corpus import CORPUS


LEAN = CORPUS["lean_mix"]
OTHER = CORPUS["rework_loop"]


def _run(*argv: str, services: Any = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli_main(
        list(argv),
        services or bootstrap.cli_services(),
        stdout=out,
        stderr=err,
    )
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("record") / "lean.record.json"
    code, _, err = _run("record", "export", str(LEAN), "--out", str(path))
    assert code == 0, err
    return path


def _edited(tmp_path: Path, exported: Path, change: Any) -> Path:
    doc = json.loads(exported.read_text(encoding="utf-8"))
    change(doc)
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.mark.req("TER-REC-001")
class TestExport:
    def test_export_to_stdout_is_a_record(self) -> None:
        code, out, _ = _run("record", "export", str(LEAN), "--ter", "off")
        doc = json.loads(out)
        assert code == 0 and doc["schema"] == "ter.record/0.1"
        assert doc["produced_by"]["ter_method"] is None
        # The demo model has no price in the shipped book: counted, not guessed.
        assert doc["cost"]["unpriced_turns"] == doc["usage"]["model_turns"]

    def test_export_writes_the_file_it_names(self, exported: Path) -> None:
        assert json.loads(exported.read_text(encoding="utf-8"))["work"]["ter"]

    def test_export_writes_into_a_new_folder(self, tmp_path: Path) -> None:
        path = tmp_path / "new" / "r.json"
        code, _, err = _run("record", "export", str(LEAN), "--out", str(path))
        assert code == 0 and path.is_file() and "Wrote" in err

    def test_an_unknown_tokenizer_is_refused(self) -> None:
        code, _, err = _run("record", "export", str(LEAN), "--tokenizer", "nope")
        assert code == 2 and "Unknown tokenizer" in err

    def test_a_missing_session_is_refused(self) -> None:
        code, _, err = _run("record", "export", "no-such.jsonl")
        assert code == 2 and "No such session file" in err

    def test_a_missing_outcome_is_refused(self) -> None:
        code, _, err = _run("record", "export", str(LEAN), "--outcome", "nope.xml")
        assert code == 2 and "No such outcome file" in err

    def test_a_malformed_session_is_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("not json\n", encoding="utf-8")
        code, _, err = _run("record", "export", str(bad))
        assert code == 2 and "Cannot read session" in err

    def test_without_services_the_command_says_so(self) -> None:
        services = replace(bootstrap.cli_services(), records=None)
        code, _, err = _run("record", "export", str(LEAN), services=services)
        assert code == 2 and "not available" in err


@pytest.mark.req("TER-REC-005")
class TestVerify:
    def test_the_session_rebuilds_its_record(self, exported: Path) -> None:
        code, out, _ = _run("record", "verify", str(exported), str(LEAN))
        assert code == 0 and out.startswith("Verified")

    def test_a_changed_figure_is_reported_by_path(
        self, tmp_path: Path, exported: Path
    ) -> None:
        def change(doc: dict[str, Any]) -> None:
            doc["work"]["generated_tokens"] += 5

        code, out, _ = _run(
            "record", "verify", str(_edited(tmp_path, exported, change)), str(LEAN)
        )
        assert code == 1
        assert "1 figure(s) differ" in out
        assert "work.generated_tokens" in out
        assert "Not the same session" not in out

    def test_another_session_is_not_the_same_session(self, exported: Path) -> None:
        code, out, _ = _run("record", "verify", str(exported), str(OTHER))
        assert code == 1 and "Not the same session" in out

    def test_a_record_that_is_not_json_is_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "r.json"
        bad.write_text("{", encoding="utf-8")
        code, _, err = _run("record", "verify", str(bad), str(LEAN))
        assert code == 2 and "is not JSON" in err

    def test_a_missing_record_is_refused(self, tmp_path: Path) -> None:
        code, _, err = _run("record", "verify", str(tmp_path / "none.json"), str(LEAN))
        assert code == 2 and "Cannot read record" in err

    def test_another_format_is_refused(self, tmp_path: Path) -> None:
        other = tmp_path / "a3.json"
        other.write_text(json.dumps({"schema": "ter.a3/0.1"}), encoding="utf-8")
        code, _, err = _run("record", "verify", str(other), str(LEAN))
        assert code == 2 and "is not a ter.record/0.1 record" in err

    def test_a_tokenizer_not_installed_is_refused(
        self, tmp_path: Path, exported: Path
    ) -> None:
        def change(doc: dict[str, Any]) -> None:
            doc["produced_by"]["tokenizer"] = "nope"

        code, _, err = _run(
            "record", "verify", str(_edited(tmp_path, exported, change)), str(LEAN)
        )
        assert code == 2 and "not installed" in err


@pytest.mark.req("TER-REC-005")
class TestFormatting:
    def test_many_differences_are_cut_with_a_count(self) -> None:
        diffs = tuple((f"p{i}", i, i + 1) for i in range(SHOWN + 3))
        text = format_check(RecordCheck(True, diffs))
        assert f"{SHOWN + 3} figure(s) differ" in text
        assert "… and 3 more" in text
        assert "p0: 0 → 1" in text

    def test_a_clean_check_says_verified(self) -> None:
        assert format_check(RecordCheck(True, ())).startswith("Verified")
        assert not RecordCheck(False, ()).ok

    @pytest.mark.parametrize(
        ("method", "mode"),
        [
            (None, "off"),
            ("TER 3, offline (regex-v1 tokens)", "offline"),
            ("TER 3 (sentence-transformers)", "model"),
        ],
    )
    def test_ter_mode_follows_the_recorded_method(
        self, method: str | None, mode: str
    ) -> None:
        assert ter_mode_of({"produced_by": {"ter_method": method}}) == mode
        assert ter_mode_of({}) == "off"


@pytest.mark.req("TER-REC-005")
def test_a_record_exported_before_by_model_still_verifies_with_a_note(
    tmp_path: Path,
) -> None:
    """Qodo #115 finding 2: Records exported before usage.by_model existed
    verify on every other figure, and say what was not compared."""
    from ter import bootstrap
    from ter.adapters.driving.cli import main as cli_main

    services = bootstrap.cli_services()
    session = (
        Path(__file__).resolve().parents[2]
        / "sample_sessions"
        / "example_session.jsonl"
    )
    record = tmp_path / "r.json"
    out, err = io.StringIO(), io.StringIO()
    argv = ["record", "export", str(session), "--ter", "off", "--out", str(record)]
    assert cli_main(argv, services, stdout=out, stderr=err) == 0, err.getvalue()
    doc = json.loads(record.read_text(encoding="utf-8"))
    del doc["usage"]["by_model"]
    record.write_text(json.dumps(doc), encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = cli_main(
        ["record", "verify", str(record), str(session)],
        services,
        stdout=out,
        stderr=err,
    )
    assert code == 0, out.getvalue() + err.getvalue()
    assert out.getvalue() == (
        "Verified: the session rebuilds this record exactly.\n"
        "Note: usage.by_model not compared: the record was exported before TER "
        "split usage by model; export it again to join it to billing\n"
    )
    # A by_model that is present is still compared.
    doc["usage"]["by_model"] = []
    record.write_text(json.dumps(doc), encoding="utf-8")
    out = io.StringIO()
    assert (
        cli_main(
            ["record", "verify", str(record), str(session)],
            services,
            stdout=out,
            stderr=err,
        )
        == 1
    )
    assert "usage.by_model" in out.getvalue()


@pytest.mark.req("TER-ANL-045")
def test_a_price_failure_is_named_and_a_session_failure_is_not_a_price_failure(
    tmp_path: Path,
) -> None:
    """Qodo #115 finding 12: only the price file's own failures read
    'Cannot read prices'."""
    from ter import bootstrap
    from ter.adapters.driving.cli import main as cli_main

    services = bootstrap.cli_services()
    session = (
        Path(__file__).resolve().parents[2]
        / "sample_sessions"
        / "example_session.jsonl"
    )
    folder = tmp_path / "prices.csv"
    folder.write_text("SkuId\nx\n", encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = cli_main(
        ["record", "export", str(session), "--prices", str(folder)],
        services,
        stdout=out,
        stderr=err,
    )
    assert code == 2 and err.getvalue().startswith("Cannot read prices:")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    err = io.StringIO()
    code = cli_main(["record", "export", str(bad)], services, stdout=out, stderr=err)
    assert code == 2 and "Cannot read prices" not in err.getvalue()
