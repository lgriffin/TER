import json
from pathlib import Path

from ter_calculator.phase2_signals import analyze_session_signals
from ter_calculator.batch_analysis import aggregate_results, build_dashboard_html


def _write_session(path: Path) -> None:
    rows = [
        {"type":"user","sessionId":"s1","uuid":"u1","timestamp":"2026-01-01T00:00:00Z","message":{"role":"user","content":"inspect file"}},
    ]
    for i in range(3):
        rows.append({"type":"assistant","sessionId":"s1","uuid":f"a{i}","timestamp":"2026-01-01T00:00:01Z","message":{"role":"assistant","content":[{"type":"tool_use","id":f"t{i}","name":"Read","input":{"file_path":"a.py"}}],"usage":{"output_tokens":10}}})
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_repeated_tool_call_is_explainable(tmp_path):
    source = tmp_path / "s.jsonl"
    _write_session(source)
    analysis = analyze_session_signals(source)
    finding = next(f for f in analysis["findings"] if f["signal_type"] == "repeated_tool_call")
    assert finding["occurrences"] == 3
    assert finding["evidence"]
    assert finding["recommendation"]


def test_phase2_aggregation_and_dashboard():
    result = {
        "session_id":"s1","aggregate_ter":0.8,"total_tokens":100,"aligned_tokens":80,"waste_tokens":20,
        "phase_scores":{"reasoning":0.8,"tool_use":0.7,"generation":0.9},
        "waste_summary":{"waste_by_category":{},"waste_by_phase":{}},
        "phase2_analysis":{"finding_count":1,"signal_counts":{"repeated_tool_call":1},"severity_counts":{"medium":1},"findings":[{"signal_type":"repeated_tool_call","severity":"medium","confidence":0.8,"occurrences":3,"summary":"Repeated","recommendation":"Replan","evidence":[{"source_lines":[3]}]}]},
    }
    summary = aggregate_results([result])
    assert summary["phase2"]["total_findings"] == 1
    html = build_dashboard_html([result], summary, bucket_count=10)
    assert "Phase 2 findings" in html
    assert "repeated_tool_call" in html
