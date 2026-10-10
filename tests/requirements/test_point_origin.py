"""Points with a recorded origin: P001..P200 are Leigh's, past P200 contributed.

Uses temporary catalogues, whose example contributed point is P299 so it
never collides with a shipped contributed point (ADR 0005).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ter.adapters.driven.requirements_yaml import CatalogueError, load_catalogue
from ter.adapters.driving import req_cli
from ter.adapters.driving.req_report import render_points_index
from ter.domain.points import (
    PointError,
    PointOrigin,
    Verification,
    VisionPoint,
    lint_points,
)
from ter.domain.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE_POINTS = load_catalogue(ROOT / "requirements").points

GARE = {"source": "GARE", "ref": "docs/plan.md#seam", "author": "Leigh Griffin"}

BASE: dict[str, Any] = {
    "id": "P001",
    "text": "Measure things.",
    "level": "L2",
    "kind": "capability",
    "status": "not-started",
    "definition_of_done": ["A test shows it."],
    "rules": ["TER-AAA-001"],
    "verification": ["planned: tests for TER-AAA-001"],
}


def _point(**change: Any) -> VisionPoint:
    return VisionPoint.from_mapping({**BASE, **change})


def _req(rid: str, points: list[int]) -> Requirement:
    return Requirement.from_mapping(
        {
            "id": rid,
            "pattern": "ubiquitous",
            "text": "TER shall count tokens.",
            "level": "L0",
            "status": "planned",
            "rationale": "Because.",
            "source_points": points,
        }
    )


def _always(_: Verification) -> bool:
    return True


def _found(issues: list[Any]) -> set[tuple[str, str]]:
    return {(i.requirement_id, i.code) for i in issues}


# -- model -----------------------------------------------------------------


def test_origin_parses_and_normalises() -> None:
    point = _point(id="P201", origin={**GARE, "ref": "  docs/plan.md#seam "})
    assert point.number == 201 and point.id == "P201" and point.contributed
    assert point.origin == PointOrigin("GARE", "docs/plan.md#seam", "Leigh Griffin")
    assert str(point.origin) == "GARE: docs/plan.md#seam (Leigh Griffin)"
    assert _point(id="P1234", origin=GARE).id == "P1234"
    vision = _point()
    assert vision.origin is None and not vision.contributed


@pytest.mark.parametrize(
    ("origin", "message"),
    [
        ("GARE", "origin must be a mapping"),
        ({**GARE, "licence": "x"}, "unknown origin fields licence"),
        ({"source": "GARE", "ref": "x"}, "origin author is required"),
        ({**GARE, "source": " "}, "origin source is required"),
        ({**GARE, "ref": 3}, "origin ref is required"),
    ],
)
def test_origin_rejects_malformed_entries(origin: object, message: str) -> None:
    with pytest.raises(PointError, match=message):
        _point(id="P201", origin=origin)


# -- lint ------------------------------------------------------------------


@pytest.mark.req("TER-REQ-009")
def test_contributed_point_without_origin_is_rejected() -> None:
    points = [_point(), _point(id="P003", rules=["TER-AAA-003"])]
    reqs = [_req("TER-AAA-001", [1]), _req("TER-AAA-003", [3])]
    issues = lint_points(points, reqs, _always, total=1)
    assert _found(issues) == {("P003", "POINT-ORIGIN")}
    assert "records an origin (source, ref, author)" in str(issues[0])


@pytest.mark.req("TER-REQ-009")
def test_contributed_point_with_origin_passes() -> None:
    points = [_point(), _point(id="P002", rules=["TER-AAA-002"], origin=GARE)]
    reqs = [_req("TER-AAA-001", [1]), _req("TER-AAA-002", [2])]
    assert lint_points(points, reqs, _always, total=1) == []


@pytest.mark.req("TER-REQ-009")
def test_duplicate_contributed_point_is_rejected() -> None:
    contributed = _point(id="P201", rules=["TER-AAA-201"], origin=GARE)
    points = [_point(), contributed, contributed]
    reqs = [_req("TER-AAA-001", [1]), _req("TER-AAA-201", [201])]
    issues = lint_points(points, reqs, _always, total=1)
    assert ("P201", "POINT-DUPLICATE") in _found(issues)


@pytest.mark.req("TER-REQ-010")
def test_vision_point_with_origin_is_rejected() -> None:
    points = [_point(origin=GARE)]
    issues = lint_points(points, [_req("TER-AAA-001", [1])], _always, total=1)
    assert _found(issues) == {("P001", "POINT-ORIGIN")}
    assert "owner's vision list and record no origin" in str(issues[0])


@pytest.mark.req("TER-REQ-012")
def test_requirement_citing_an_uncatalogued_point_is_rejected() -> None:
    reqs = [_req("TER-AAA-001", [1, 205])]
    issues = lint_points([_point()], reqs, _always, total=1)
    assert _found(issues) == {("TER-AAA-001", "POINT-UNKNOWN")}
    assert "cites P205, which is not in the points catalogue" in str(issues[0])


# -- index -------------------------------------------------------------------


@pytest.mark.req("TER-REQ-011")
def test_index_shows_every_points_origin() -> None:
    points = [
        _point(),
        _point(id="P002", rules=["TER-AAA-002"], origin=GARE),
        _point(id="P003", rules=["TER-AAA-002"], origin={**GARE, "ref": "a|b"}),
    ]
    reqs = [_req("TER-AAA-001", [1]), _req("TER-AAA-002", [2, 3])]
    text = render_points_index(points, reqs)
    assert "| Id | Point | Origin | Level |" in text
    assert "| P001 | Measure things. | vision | L2 |" in text
    assert (
        "| P002 | Measure things. | **GARE** docs/plan.md#seam (Leigh Griffin) |"
        in text
    )
    assert "**GARE** a\\|b (Leigh Griffin)" in text
    assert "Origins: Leigh's vision 1 · GARE 2" in text


@pytest.mark.req("TER-REQ-011")
def test_shipped_index_shows_origins() -> None:
    committed = (ROOT / "docs" / "ter4" / "points.md").read_text(encoding="utf-8")
    assert "Origins: Leigh's vision 200" in committed
    assert "| P001 | " in committed and " | vision | " in committed


# -- the YAML catalogue and the CLI -------------------------------------------------


def _catalogue_with(tmp_path: Path, extra: str, cite: str = "") -> Path:
    """A copy of the shipped catalogue with one more point (and a citing rule)."""
    copy = tmp_path / "requirements"
    shutil.copytree(ROOT / "requirements", copy)
    with (copy / "points.yaml").open("a", encoding="utf-8") as handle:
        handle.write(extra)
    if cite:
        (copy / "contributed.yaml").write_text(cite, encoding="utf-8")
    return copy


POINT_201 = """
  - id: P299
    text: Read the usage rows an external capability records, by schema name.
    level: L1
    kind: capability
    status: not-started
{origin}    definition_of_done:
      - A contract suite pins the schema the adapter reads.
    rules: [TER-GAR-001]
    verification:
      - 'planned: contract suite for TER-GAR-001'
"""

RULE_201 = """requirements:
  - id: TER-GAR-001
    pattern: ubiquitous
    text: >-
      The usage adapter shall read usage rows by their schema name.
    level: L1
    source_points: [299]
    rationale: Example contributed rule.
    status: planned
"""

ORIGIN_YAML = (
    '    origin: {source: GARE, ref: "docs/plan.md#seam", author: "Leigh Griffin"}\n'
)


def _lint(capsys: pytest.CaptureFixture[str], catalogue: Path) -> tuple[int, str]:
    argv = ["--catalogue", str(catalogue), "--root", str(ROOT), "lint", "--strict"]
    code = req_cli.main(argv)
    return code, capsys.readouterr().out


@pytest.mark.req("TER-REQ-009")
def test_cli_lint_accepts_a_contributed_point_with_origin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalogue = _catalogue_with(
        tmp_path, POINT_201.format(origin=ORIGIN_YAML), RULE_201
    )
    code, out = _lint(capsys, catalogue)
    shipped = len(CATALOGUE_POINTS)
    assert code == 0 and f"{shipped + 1} points, 0 errors" in out
    loaded = load_catalogue(catalogue)
    contributed = [p for p in loaded.points if p.contributed]
    assert ("P299", "GARE: docs/plan.md#seam (Leigh Griffin)") in [
        (p.id, str(p.origin)) for p in contributed
    ]
    index = render_points_index(loaded.points, loaded.requirements)
    assert "Origins: Leigh's vision 200 · " in index and "GARE 1" in index
    assert "| P299 |" in index


@pytest.mark.req("TER-REQ-009")
def test_cli_lint_rejects_a_contributed_point_without_origin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalogue = _catalogue_with(tmp_path, POINT_201.format(origin=""), RULE_201)
    code, out = _lint(capsys, catalogue)
    assert code == 1 and "P299: [POINT-ORIGIN]" in out


@pytest.mark.req("TER-REQ-012")
def test_cli_lint_rejects_a_rule_citing_a_missing_point(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalogue = _catalogue_with(tmp_path, "", RULE_201)
    code, out = _lint(capsys, catalogue)
    assert code == 1 and "TER-GAR-001: [POINT-UNKNOWN]" in out


def test_yaml_loader_reports_a_malformed_origin(tmp_path: Path) -> None:
    bad = POINT_201.format(origin="    origin: GARE\n")
    with pytest.raises(CatalogueError, match="P299: origin must be a mapping"):
        load_catalogue(_catalogue_with(tmp_path, bad))
