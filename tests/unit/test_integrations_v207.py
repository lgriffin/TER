from pathlib import Path

from ter_calculator.integrations import (
    IntegrationGate,
    build_github_annotations,
    build_sarif,
    build_step_summary,
    evaluate_gate,
)


def _result(session: str, ter: float, total: int, waste: int):
    return {
        "session_id": session,
        "aggregate_ter": ter,
        "total_tokens": total,
        "aligned_tokens": total - waste,
        "waste_tokens": waste,
    }


def test_gate_passes_and_summary_is_portable():
    result = evaluate_gate(
        [_result("a", 0.9, 100, 10)],
        IntegrationGate(minimum_ter=0.8, maximum_waste_ratio=0.2),
    )
    assert result.passed
    assert "TER integration report" in build_step_summary(result, "results")
    assert "::notice" in build_github_annotations(result)


def test_gate_failure_and_sarif():
    records = [_result("bad", 0.4, 100, 60)]
    result = evaluate_gate(
        records, IntegrationGate(minimum_ter=0.8, maximum_waste_ratio=0.2)
    )
    assert not result.passed
    assert len(result.violations) == 2
    sarif = build_sarif(records, result)
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["results"]


def test_gate_validation():
    try:
        IntegrationGate(minimum_ter=1.1)
    except ValueError as exc:
        assert "minimum_ter" in str(exc)
    else:
        raise AssertionError("expected ValueError")
