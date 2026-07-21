import json
from pathlib import Path

from ter_calculator.cli import main
from ter_calculator.release_validation import (
    ReleaseGate,
    build_release_snapshot,
    evaluate_release,
)


def _result(session: str, ter: float, total: int, waste: int):
    return {
        "session_id": session,
        "aggregate_ter": ter,
        "total_tokens": total,
        "aligned_tokens": total - waste,
        "waste_tokens": waste,
        "phase_scores": {},
    }


def test_snapshot_is_deterministic():
    a = _result("a", 0.9, 100, 10)
    b = _result("b", 1.0, 100, 0)
    first = build_release_snapshot([a, b], version="2.0.8", source="results")
    second = build_release_snapshot([b, a], version="2.0.8", source="results")
    assert first["results_sha256"] == second["results_sha256"]
    assert first["weighted_ter"] == 0.95


def test_release_regression_gate():
    snapshot = {"sessions": 10, "weighted_ter": 0.80, "waste_ratio": 0.20}
    baseline = {"weighted_ter": 0.90, "waste_ratio": 0.10}
    assessment = evaluate_release(
        snapshot,
        ReleaseGate(
            minimum_sessions=5,
            maximum_weighted_ter_drop=0.05,
            maximum_waste_ratio_increase=0.05,
        ),
        baseline,
    )
    assert not assessment.passed
    assert len(assessment.violations) == 2


def test_release_cli_writes_manifest(tmp_path: Path):
    (tmp_path / "a.ter.json").write_text(
        json.dumps(_result("a", 1.0, 100, 0)), encoding="utf-8"
    )
    assert main(["--quiet", "release-check", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "ter-release-manifest.json").read_text())
    assert manifest["snapshot"]["ter_version"] == "2.0.14"
    assert manifest["assessment"]["passed"] is True
    assert manifest["files"][0]["sha256"]


def test_release_gate_validation():
    try:
        ReleaseGate(minimum_sessions=0)
    except ValueError as exc:
        assert "minimum_sessions" in str(exc)
    else:
        raise AssertionError("expected ValueError")
