from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ter_calculator import batch_analysis as ba
from ter_calculator.phase2_signals import (
    _canonical,
    _severity,
    _short,
    analyze_session_signals,
)


def _result(session_id: str, ter: float, total: int, waste: int) -> dict:
    return {
        "session_id": session_id,
        "aggregate_ter": ter,
        "phase_scores": {"reasoning": ter, "tool_use": ter / 2, "generation": 1.0},
        "total_tokens": total,
        "aligned_tokens": total - waste,
        "waste_tokens": waste,
        "waste_summary": {
            "waste_by_category": {"duplicate": waste, "loop": waste / 2},
            "waste_by_phase": {"reasoning": waste / 2, "tool_use": waste / 2},
        },
        "phase2_analysis": {
            "finding_count": 2,
            "signal_counts": {"repeated_tool_call": 1, "repeated_failure": 1},
            "severity_counts": {"high": 1, "medium": 1},
        },
    }


def test_batch_helpers_and_aggregate(tmp_path: Path) -> None:
    nested = tmp_path / "in" / "nested"
    nested.mkdir(parents=True)
    (tmp_path / "in" / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (nested / "b.jsonl").write_text("{}\n", encoding="utf-8")
    assert len(ba.discover_sessions(tmp_path / "in")) == 2
    assert len(ba.discover_sessions(tmp_path / "in", recursive=False)) == 1

    good = _result("a", 0.8, 100, 20)
    assert ba.validate_result(good) == []
    assert ba.validate_result([]) == ["result is not a JSON object"]
    bad = dict(good, aggregate_ter=2, aligned_tokens=70, waste_tokens=-1)
    errors = ba.validate_result(bad)
    assert "token counts must be non-negative" in errors
    assert "aligned_tokens + waste_tokens != total_tokens" in errors
    assert "aggregate_ter outside [0, 1]" in errors
    missing = ba.validate_result({})
    assert "missing field: session_id" in missing

    out = tmp_path / "results"
    out.mkdir()
    (out / "good.ter.json").write_text(json.dumps(good), encoding="utf-8")
    (out / "bad.ter.json").write_text("not-json", encoding="utf-8")
    (out / "invalid.ter.json").write_text(
        json.dumps({"session_id": 3}), encoding="utf-8"
    )
    results, invalid = ba.load_results(out)
    assert len(results) == 1 and len(invalid) == 2

    combined = tmp_path / "all.jsonl"
    ba.write_combined_jsonl(results, combined)
    assert "_result_path" not in combined.read_text(encoding="utf-8")

    summary = ba.aggregate_results([good, _result("b", 0.4, 200, 100)])
    assert summary["sessions"] == 2
    assert summary["weighted_ter"] == pytest.approx(0.6)
    assert summary["median_ter"] == pytest.approx(0.6)
    assert summary["phase2"]["total_findings"] == 4
    assert ba.aggregate_results([])["weighted_ter"] == 0

    svg = ba._bar_chart_svg(["<x>", "b"], [10, 0], "A & B")
    assert "&lt;x&gt;" in svg and "A &amp; B" in svg


def test_analyze_one_skip_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "session.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    target = tmp_path / "out.ter.json"
    target.write_text("{}", encoding="utf-8")
    assert ba._analyze_one(str(source), str(target), False).status == "skipped"

    target.write_text("bad", encoding="utf-8")
    monkeypatch.setattr(
        ba, "default_analyze_args", lambda path: SimpleNamespace(path=path)
    )
    monkeypatch.setattr(ba, "analyze_session", lambda args: object())
    monkeypatch.setattr(
        ba, "ter_result_to_dict", lambda result: _result("ok", 1.0, 10, 0)
    )
    monkeypatch.setattr(
        ba, "analyze_session_signals", lambda path: {"finding_count": 0}
    )
    assert ba._analyze_one(str(source), str(target), False).status == "completed"
    monkeypatch.setattr(
        ba, "analyze_session", lambda args: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    failed = ba._analyze_one(str(source), str(target), True)
    assert failed.status == "failed" and failed.error == "boom"


def _write_signal_session(path: Path) -> None:
    long_text = "This is a repeated reasoning block " * 8
    rows = [
        {
            "type": "user",
            "uuid": "u",
            "sessionId": "s",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "inspect"}],
            },
        }
    ]
    for i in range(12):
        rows.append(
            {
                "type": "assistant",
                "uuid": f"a{i}",
                "sessionId": "s",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "text": long_text},
                        {
                            "type": "tool_use",
                            "id": f"t{i}",
                            "name": "Read",
                            "input": {"file_path": "src/a.py"},
                        },
                    ],
                    "usage": {"input_tokens": 10, "output_tokens": 10},
                },
            }
        )
        if i < 3:
            rows.append(
                {
                    "type": "user",
                    "uuid": f"r{i}",
                    "sessionId": "s",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"t{i}",
                                "content": "ERROR permission denied",
                            }
                        ],
                    },
                }
            )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_phase2_all_detector_families(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write_signal_session(path)
    result = analyze_session_signals(path)
    kinds = set(result["signal_counts"])
    assert {
        "repeated_tool_call",
        "repeated_file_read",
        "repeated_failure",
        "repeated_generated_content",
        "high_activity_low_novelty",
    } <= kinds
    assert result["finding_count"] >= 5
    assert _canonical({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert _canonical({1, 2}).startswith("{")
    assert _short("a   b") == "a b"
    assert _short("x" * 300, 10).endswith("…")
    assert [_severity(n) for n in (1, 3, 6)] == ["low", "medium", "high"]
