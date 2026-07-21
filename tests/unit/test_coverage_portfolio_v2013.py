from __future__ import annotations
import json
from pathlib import Path
import pytest
from ter_calculator import portfolio_dashboard as pd


def _rows():
    return [
        {
            "session_id": "a",
            "aggregate_ter": 0.9,
            "total_tokens": 100,
            "aligned_tokens": 90,
            "waste_tokens": 10,
            "phase_scores": {"reasoning": 0.8, "tool_use": 0.9, "generation": 1},
            "waste_summary": {
                "waste_by_category": {"duplicate": 10},
                "waste_by_phase": {"reasoning": 5, "tool_use": 5},
            },
            "phase2_analysis": {
                "finding_count": 1,
                "signal_counts": {"loop": 1},
                "severity_counts": {"high": 1},
            },
        },
        {
            "session_id": "b",
            "aggregate_ter": 0.4,
            "total_tokens": 200,
            "aligned_tokens": 80,
            "waste_tokens": 120,
            "phase_scores": {"reasoning": 0.3, "tool_use": 0.4, "generation": 0.5},
            "waste_summary": {
                "waste_by_category": {"loop": 120},
                "waste_by_phase": {"reasoning": 100, "generation": 20},
            },
            "phase2_analysis": {
                "finding_count": 2,
                "signal_counts": {"loop": 2},
                "severity_counts": {"medium": 2},
            },
        },
    ]


def test_load_helpers_and_dashboard(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in _rows()) + "\n", encoding="utf-8")
    assert len(pd.load_jsonl(path)) == 2
    assert pd.number(True, 7) == 7 and pd.number("1", 2) == 2 and pd.number(3) == 3
    assert pd.percentile([], 0.5) == 0 and pd.percentile([0, 1], 0.5) == 0.5
    html = pd.make_dashboard(_rows(), ter_bucket_count=4)
    assert "TER portfolio dashboard" in html
    assert "plotly" in html.lower()
    assert "Top waste categories" in html
    with pytest.raises(ValueError):
        pd.make_dashboard([], 0)


def test_load_jsonl_errors_and_empty_dashboard(tmp_path: Path) -> None:
    empty = tmp_path / "e"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pd.load_jsonl(empty)
    bad = tmp_path / "b"
    bad.write_text("{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        pd.load_jsonl(bad)
    html = pd.make_dashboard([{"session_id": "x"}], 2)
    assert "No waste categories" in html
