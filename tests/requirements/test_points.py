"""Vision points: model, two-way link lint, repository checks and the index."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ter.adapters.driven.requirements_yaml import (
    CatalogueError,
    RepositoryChecks,
    load_catalogue,
)
from ter.adapters.driving import req_cli
from ter.adapters.driving.req_report import (
    render_points_index,
    status_bar,
    status_grid,
)
from ter.domain import Maturity
from ter.domain.points import (
    PointError,
    PointKind,
    PointStatus,
    Verification,
    VerificationKind,
    VisionPoint,
    lint_points,
    summarise,
)
from ter.domain.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = load_catalogue(ROOT / "requirements")


def _req(rid: str, points: list[int], status: str = "planned") -> Requirement:
    return Requirement.from_mapping(
        {
            "id": rid,
            "pattern": "ubiquitous",
            "text": "TER shall count tokens.",
            "level": "L0",
            "status": status,
            "rationale": "Because.",
            "source_points": points,
        }
    )


BASE: dict[str, Any] = {
    "id": "P001",
    "text": "Measure  things.",
    "level": "L2",
    "kind": "capability",
    "status": "not-started",
    "definition_of_done": ["A test shows it."],
    "rules": ["TER-AAA-001"],
    "verification": ["planned: tests for TER-AAA-001"],
}


def _point(**change: Any) -> VisionPoint:
    return VisionPoint.from_mapping({**BASE, **change})


def _always(_: Verification) -> bool:
    return True


def _codes(issues: list[Any]) -> list[str]:
    return sorted(i.code for i in issues)


# -- model -----------------------------------------------------------------


def test_verification_parses_each_kind() -> None:
    entry = Verification.parse("test: tests/a.py::test_b")
    assert (entry.kind, entry.target) == (VerificationKind.TEST, "tests/a.py::test_b")
    assert entry.is_check and str(entry) == "test: tests/a.py::test_b"
    assert Verification.parse("ci: Ruff lint").is_check
    assert not Verification.parse("planned: later").is_check
    assert not Verification.parse("branch: work/x tests").is_check


@pytest.mark.parametrize("text", ["tests/a.py", "proof: x", "test:", "planned:  "])
def test_verification_rejects_malformed_entries(text: str) -> None:
    with pytest.raises(PointError):
        Verification.parse(text)


def test_point_from_mapping_normalises_fields() -> None:
    point = _point(level=3)
    assert point.id == "P001" and point.number == 1
    assert point.text == "Measure things."
    assert point.level is Maturity.GROUNDED
    assert point.kind is PointKind.CAPABILITY
    assert point.status is PointStatus.NOT_STARTED
    assert point.verification[0].kind is VerificationKind.PLANNED


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": "P1"}, "does not match"),
        ({"id": 1}, "does not match"),
        ({"id": "P000"}, "outside"),
        ({"id": "P0201"}, "outside"),
        ({"id": "P10000"}, "does not match"),
        ({"extra": 1}, "unknown fields extra"),
        ({"text": " "}, "text is required"),
        ({"level": None}, "level is required"),
        ({"level": "L8"}, "Unknown maturity"),
        ({"kind": "wish"}, "wish"),
        ({"status": "doing"}, "doing"),
        ({"rules": "TER-AAA-001"}, "rules must be a list"),
        ({"definition_of_done": [1]}, "definition_of_done must be a list"),
        ({"verification": ["maybe: x"]}, "must start with one of"),
        ({"issue": 0}, "positive issue number"),
        ({"issue": True}, "positive issue number"),
        ({"issue": "34"}, "positive issue number"),
        ({"real_data": "yes"}, "real_data must be true or false"),
        ({"real_data_verified": 1}, "real_data_verified must be true or false"),
    ],
)
def test_point_from_mapping_rejects_malformed_entries(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(PointError, match=message):
        _point(**change)


def test_summarise_counts_by_status_and_level() -> None:
    points = [
        _point(),
        _point(id="P002", status="done", level="L0"),
        _point(id="P003", status="partial"),
    ]
    summary = summarise(points)
    assert summary.by_status == {
        PointStatus.DONE: 1,
        PointStatus.PARTIAL: 1,
        PointStatus.NOT_STARTED: 1,
    }
    assert summary.by_level[Maturity.EXPLAINED][PointStatus.PARTIAL] == 1
    assert summary.by_level[Maturity.MEASURED][PointStatus.DONE] == 1


# -- lint ------------------------------------------------------------------


@pytest.mark.req("TER-REQ-004")
def test_clean_points_pass() -> None:
    points = [_point(), _point(id="P002", rules=["TER-AAA-002"])]
    reqs = [_req("TER-AAA-001", [1]), _req("TER-AAA-002", [2])]
    assert lint_points(points, reqs, _always, total=2) == []


@pytest.mark.req("TER-REQ-004")
def test_points_need_done_rules_and_verification() -> None:
    points = [
        _point(definition_of_done=[], rules=[], verification=[]),
        _point(
            id="P002", definition_of_done=["a", "b", "c", "d"], rules=["TER-AAA-001"]
        ),
        _point(id="P003", definition_of_done=["a", ""]),
    ]
    issues = lint_points(points, [_req("TER-AAA-001", [2, 3])], _always, total=3)
    assert _codes(issues) == [
        "POINT-DOD",
        "POINT-DOD",
        "POINT-DOD",
        "POINT-RULES",
        "POINT-VERIFY",
    ]


@pytest.mark.req("TER-REQ-004")
def test_points_and_requirements_link_both_ways() -> None:
    points = [
        _point(rules=["TER-AAA-001", "TER-ZZZ-999"]),
        _point(id="P002", rules=["TER-AAA-001"]),
    ]
    reqs = [_req("TER-AAA-001", [1, 3]), _req("TER-AAA-002", [])]
    issues = lint_points(points, reqs, _always, total=2)
    by_code = {(i.requirement_id, i.code) for i in issues}
    assert by_code == {
        ("P001", "POINT-RULE-UNKNOWN"),
        ("P002", "POINT-LINK"),
        ("TER-AAA-001", "POINT-UNKNOWN"),
        ("TER-AAA-002", "REQ-ORPHAN"),
    }


def test_points_must_cover_every_number_once() -> None:
    points = [_point(), _point()]
    issues = lint_points(points, [_req("TER-AAA-001", [1])], _always, total=2)
    assert _codes(issues) == ["POINT-DUPLICATE", "POINT-MISSING"]


@pytest.mark.req("TER-REQ-004")
def test_done_points_need_verified_rules_or_an_existing_check() -> None:
    reqs = [_req("TER-AAA-001", [1, 2, 3, 4]), _req("TER-AAA-002", [5], "verified")]
    missing = Verification.parse("test: tests/missing.py")
    points = [
        _point(status="done"),
        _point(id="P002", status="done", verification=["test: tests/ok.py"]),
        _point(id="P003", status="done", verification=["branch: work/x tests"]),
        _point(id="P004", status="done", verification=[str(missing)]),
        _point(id="P005", status="done", rules=["TER-AAA-002"]),
    ]
    issues = lint_points(points, reqs, lambda v: v != missing, total=5)
    found = {(i.requirement_id, i.code, i.warning) for i in issues}
    assert found == {
        ("P001", "POINT-DONE", False),
        ("P003", "POINT-PENDING", True),
        ("P004", "POINT-CHECK-MISSING", False),
        ("P004", "POINT-DONE", False),
    }
    assert str(next(i for i in issues if i.warning)).startswith("warning: P003")


@pytest.mark.req("TER-REQ-006")
def test_real_data_points_are_not_done_on_synthetic_tests_alone() -> None:
    verified = _req("TER-AAA-001", [1, 2, 3, 4], "verified")
    points = [
        _point(status="done", issue=34, real_data=True),
        _point(
            id="P002", status="done", issue=34, real_data=True, real_data_verified=True
        ),
        _point(id="P003", issue=35),
        _point(id="P004", real_data_verified=True),
    ]
    issues = lint_points(points, [verified], _always, total=4)
    found = {(i.requirement_id, i.code) for i in issues}
    assert found == {
        ("P001", "POINT-REAL-DATA"),
        ("P003", "POINT-REAL-DATA-FLAG"),
        ("P004", "POINT-REAL-DATA-FLAG"),
    }
    assert "issue #34" in next(str(i) for i in issues if i.code == "POINT-REAL-DATA")


def test_real_data_point_without_issue_is_still_blocked() -> None:
    point = _point(status="done", real_data=True)
    issues = lint_points(
        [point], [_req("TER-AAA-001", [1], "verified")], _always, total=1
    )
    assert [i.code for i in issues] == ["POINT-REAL-DATA"]
    assert "issue #" not in str(issues[0])


def test_index_links_issues() -> None:
    points = [
        _point(issue=34, real_data=True),
        _point(id="P002", issue=35, real_data=True, real_data_verified=True),
    ]
    reqs = [_req("TER-AAA-001", [1, 2])]
    text = render_points_index(points, reqs)
    assert "[#34](https://github.com/lgriffin/TER/issues/34) |" in text
    assert "[#35](https://github.com/lgriffin/TER/issues/35) ✓ |" in text
    assert "**2 points** need it (1 verified)" in text
    assert "no issues" in render_points_index([_point()], reqs)


def test_shipped_real_data_points_match_their_issues() -> None:
    real = {p.id: p.issue for p in CATALOGUE.points if p.real_data}
    assert len(real) == 47
    # #34 to #46 collect Claude Code sessions; #55 a real recorded GARE run.
    assert all(
        issue is not None and (34 <= issue <= 46 or issue == 55)
        for issue in real.values()
    )
    assert real["P115"] == 35 and real["P200"] == 45 and real["P103"] == 55


# -- repository checks --------------------------------------------------------


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text(
        "class TestA:\n    def test_b(self): pass\n\nasync def test_c(): pass\n",
        encoding="utf-8",
    )
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(
        "jobs:\n  lint:\n    steps:\n      - uses: x\n      - name: Ruff lint\n"
        "  odd: 3\n",
        encoding="utf-8",
    )
    (workflows / "other.yaml").write_text("- not a mapping\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("entry", "exists"),
    [
        ("test: tests/test_a.py", True),
        ("test: tests/test_a.py::TestA::test_b", True),
        ("test: tests/test_a.py::TestA::test_b[param-1]", True),
        ("test: tests/test_a.py::test_c", True),
        ("test: tests/test_a.py::test_missing", False),
        ("test: tests/nope.py", False),
        ("ci: Ruff lint", True),
        ("ci: Deploy", False),
        ("planned: anything", True),
    ],
)
def test_repository_checks(tmp_path: Path, entry: str, exists: bool) -> None:
    checks = RepositoryChecks(_repo(tmp_path))
    assert checks(Verification.parse(entry)) is exists


def test_repository_checks_find_this_repositorys_ci_steps() -> None:
    steps = RepositoryChecks(ROOT).ci_steps()
    assert {
        "Architecture import contracts",
        "Vision points index is up to date",
    } <= steps


def test_load_points_rejects_bad_files(tmp_path: Path) -> None:
    (tmp_path / "points.yaml").write_text("points: 3\n", encoding="utf-8")
    with pytest.raises(CatalogueError, match="'points' list"):
        load_catalogue(tmp_path)
    (tmp_path / "points.yaml").write_text("points: [1]\n", encoding="utf-8")
    with pytest.raises(CatalogueError, match="point 0 is not a mapping"):
        load_catalogue(tmp_path)
    (tmp_path / "points.yaml").write_text("points:\n  - id: X\n", encoding="utf-8")
    with pytest.raises(CatalogueError, match="does not match"):
        load_catalogue(tmp_path)


# -- the shipped catalogue and index --------------------------------------------


@pytest.mark.req("TER-REQ-004")
def test_shipped_points_lint_without_errors() -> None:
    # Leigh's 200 plus the contributed points (P201 control charts).
    assert len(CATALOGUE.points) == 201
    issues = lint_points(
        CATALOGUE.points, CATALOGUE.requirements, RepositoryChecks(ROOT)
    )
    errors = [str(i) for i in issues if not i.warning]
    assert not errors, "\n".join(errors)
    assert all(i.code == "POINT-PENDING" for i in issues)


@pytest.mark.req("TER-REQ-005")
def test_committed_points_index_is_current() -> None:
    committed = (ROOT / "docs" / "ter4" / "points.md").read_text(encoding="utf-8")
    assert committed == render_points_index(CATALOGUE.points, CATALOGUE.requirements)


def test_index_visuals() -> None:
    counts = {PointStatus.DONE: 1, PointStatus.PARTIAL: 1, PointStatus.NOT_STARTED: 2}
    assert status_bar(counts, width=8) == "██▓▓░░░░"
    assert status_bar(dict.fromkeys(PointStatus, 0), width=3) == "░░░"
    points = [
        _point(id=f"P{n:03d}", status=s) for n, s in [(2, "done"), (1, "partial")]
    ]
    assert status_grid(points, per_row=1) == "P001 ◐\nP002 ●"


def test_index_shortens_long_points_and_marks_rules() -> None:
    point = _point(text="word " * 40, rules=["TER-AAA-001", "TER-ZZZ-001"])
    text = render_points_index([point], [_req("TER-AAA-001", [1], "verified")])
    assert "…" in text
    assert "`TER-AAA-001` ✓<br>`TER-ZZZ-001` ·" in text
    assert "| L2 Explained |" in text and "| L0 Measured |" not in text


# -- CLI -----------------------------------------------------------------------


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = req_cli.main(list(argv))
    return code, capsys.readouterr().out


def test_cli_lint_checks_points(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--catalogue", str(ROOT / "requirements"), "--root", str(ROOT), "lint"]
    code, out = _run(capsys, *base, "--strict")
    assert code == 0 and "201 points, 0 errors, 0 warnings" in out

    # A done point whose only proof is on another branch passes plain lint
    # with a warning, and fails --strict (what CI runs).
    copy = tmp_path / "requirements"
    shutil.copytree(ROOT / "requirements", copy)
    points = copy / "points.yaml"
    text = points.read_text(encoding="utf-8")
    start = text.index("id: P102")
    stop = text.index("- id:", start)
    block = text[start:stop]
    while "'test: " in block:
        proof = block.index("'test: ")
        end = block.index("'", proof + 1) + 1
        block = block[:proof] + "'branch: work/other pending proof'" + block[end:]
    text = text[:start] + block + text[stop:]
    # ... and a rule still planned here, so its proof really is elsewhere.
    rules = text.index("rules: [", start)
    text = text[: rules + len("rules: [")] + "TER-OBS-099, " + text[rules + 8 :]
    points.write_text(text, encoding="utf-8")
    l1 = copy / "l1_observed.yaml"
    l1.write_text(
        l1.read_text(encoding="utf-8").rstrip("\n")
        + "\n\n  - id: TER-OBS-099\n    pattern: ubiquitous\n"
        + "    text: >-\n      TER shall observe a pending behaviour.\n"
        + "    level: L1\n    source_points: [102]\n"
        + "    rationale: >-\n      Test fixture.\n    status: planned\n",
        encoding="utf-8",
    )
    pending = ["--catalogue", str(copy), "--root", str(ROOT), "lint"]
    code, out = _run(capsys, *pending)
    assert code == 0 and "warning: P102: [POINT-PENDING]" in out
    code, out = _run(capsys, *pending, "--strict")
    assert code == 1 and "warning: P102: [POINT-PENDING]" in out


def test_cli_points_writes_and_checks_the_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "docs" / "points.md"
    catalogue = ["--catalogue", str(ROOT / "requirements")]
    code, out = _run(capsys, *catalogue, "points", "--check", "--out", str(target))
    assert code == 1 and "is stale" in out
    code, out = _run(capsys, *catalogue, "points", "--out", str(target))
    assert code == 0 and f"wrote {target}" in out
    code, out = _run(capsys, *catalogue, "points", "--check", "--out", str(target))
    assert code == 0 and "is up to date" in out
