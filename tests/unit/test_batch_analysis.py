from __future__ import annotations

import json
from pathlib import Path

from ter_calculator.batch_analysis import (
    aggregate_results,
    build_dashboard_html,
    discover_sessions,
    validate_result,
    write_combined_jsonl,
)


def _result(session_id: str, ter: float, total: int, waste: int) -> dict:
    return {
        "session_id": session_id,
        "aggregate_ter": ter,
        "phase_scores": {"reasoning": ter, "tool_use": ter, "generation": ter},
        "total_tokens": total,
        "aligned_tokens": total - waste,
        "waste_tokens": waste,
        "waste_summary": {
            "waste_by_category": {"repetition": waste} if waste else {},
            "waste_by_phase": {"reasoning": waste} if waste else {},
            "explanation": "test",
        },
    }


def test_discover_sessions_recursive(tmp_path: Path):
    (tmp_path / "a.jsonl").write_text("{}\n")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.jsonl").write_text("{}\n")
    assert [p.name for p in discover_sessions(tmp_path)] == ["a.jsonl", "b.jsonl"]
    assert [p.name for p in discover_sessions(tmp_path, recursive=False)] == ["a.jsonl"]


def test_validate_and_aggregate_results():
    results = [_result("a", 1.0, 100, 0), _result("b", 0.8, 100, 20)]
    assert validate_result(results[0]) == []
    summary = aggregate_results(results)
    assert summary["sessions"] == 2
    assert summary["weighted_ter"] == 0.9
    assert summary["sessions_with_waste"] == 1
    assert summary["waste_by_category"] == {"repetition": 20.0}


def test_validate_detects_inconsistent_tokens():
    payload = _result("bad", 0.5, 100, 10)
    payload["aligned_tokens"] = 95
    assert "aligned_tokens + waste_tokens != total_tokens" in validate_result(payload)


def test_combined_jsonl_and_dashboard(tmp_path: Path):
    results = [_result("a", 1.0, 100, 0), _result("b", 0.8, 100, 20)]
    output = tmp_path / "all-results.jsonl"
    write_combined_jsonl(results, output)
    assert len(output.read_text().splitlines()) == 2
    assert json.loads(output.read_text().splitlines()[0])["session_id"] == "a"
    dashboard = build_dashboard_html(results, aggregate_results(results), bucket_count=10)
    assert "TER portfolio dashboard" in dashboard
    assert "0.90–1.00" in dashboard
    assert "Session results" in dashboard


def test_dashboard_cli_from_existing_results(tmp_path: Path):
    from ter_calculator.cli import main

    result_path = tmp_path / "a.ter.json"
    result_path.write_text(json.dumps(_result("a", 1.0, 100, 0)))
    assert main(["--quiet", "dashboard", str(tmp_path), "--ter-buckets", "10"]) == 0
    assert (tmp_path / "ter-dashboard.html").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "all-results.jsonl").exists()
