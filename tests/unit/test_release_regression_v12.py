from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ter_calculator.commands.benchmark_compare import _cmd_benchmark_compare
from ter_calculator.evaluation import BenchmarkRecord, evaluate_benchmark
from ter_calculator.regression import (
    compare_benchmark_reports,
    dump_regression_json,
    format_regression_report,
)


def _records(*, candidate: bool = False) -> list[BenchmarkRecord]:
    predicted_last = "aligned_response" if candidate else "over_explanation"
    return [
        BenchmarkRecord(
            "1", "s1", "reasoning", "aligned_reasoning", "aligned_reasoning", 0.1, 10
        ),
        BenchmarkRecord(
            "2",
            "s1",
            "reasoning",
            "redundant_reasoning",
            "redundant_reasoning",
            0.9,
            20,
        ),
        BenchmarkRecord(
            "3", "s2", "tool_use", "unnecessary_tool_call", "aligned_tool_call", 0.7, 30
        ),
        BenchmarkRecord(
            "4", "s2", "generation", "aligned_response", predicted_last, 0.8, 40
        ),
    ]


def _write(path: Path, records: list[BenchmarkRecord]) -> None:
    lines = []
    for record in records:
        lines.append(
            json.dumps(
                {
                    "id": record.record_id,
                    "session_id": record.session_id,
                    "phase": record.phase,
                    "gold_label": record.gold_label,
                    "predicted_label": record.predicted_label,
                    "score": record.score,
                    "tokens": record.tokens,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compare_reports_passes_for_precision_improvement():
    baseline = evaluate_benchmark(_records(), bootstrap_samples=0)
    candidate = evaluate_benchmark(_records(candidate=True), bootstrap_samples=0)
    report = compare_benchmark_reports(
        baseline,
        candidate,
        minimum_precision=0.9,
        maximum_f0_5_drop=0.0,
    )
    assert report.passed
    assert report.precision.delta == pytest.approx(0.5)
    assert report.false_positives.delta == -1
    assert "Result: PASS" in format_regression_report(report)
    assert json.loads(dump_regression_json(report))["passed"] is True


def test_compare_reports_fails_precision_and_false_positive_gates():
    baseline = evaluate_benchmark(_records(candidate=True), bootstrap_samples=0)
    candidate = evaluate_benchmark(_records(), bootstrap_samples=0)
    report = compare_benchmark_reports(
        baseline,
        candidate,
        minimum_precision=0.9,
        maximum_precision_drop=0.0,
        maximum_false_positive_increase=0,
    )
    assert not report.passed
    failed = {gate.name for gate in report.gates if not gate.passed}
    assert {
        "minimum_precision",
        "precision_drop",
        "f0_5_drop",
        "false_positive_increase",
    } <= failed


def test_compare_requires_matching_dataset_shape():
    baseline = evaluate_benchmark(_records(), bootstrap_samples=0)
    candidate = evaluate_benchmark(_records()[:-1], bootstrap_samples=0)
    with pytest.raises(ValueError, match="same number of records"):
        compare_benchmark_reports(baseline, candidate)


def test_benchmark_compare_command_exit_codes_and_json(tmp_path: Path, capsys):
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    _write(baseline_path, _records())
    _write(candidate_path, _records(candidate=True))
    args = SimpleNamespace(
        baseline_path=str(baseline_path),
        candidate_path=str(candidate_path),
        baseline_threshold=None,
        candidate_threshold=None,
        minimum_precision=0.9,
        maximum_precision_drop=0.0,
        maximum_recall_drop=1.0,
        maximum_f0_5_drop=0.0,
        maximum_accuracy_drop=1.0,
        maximum_false_positive_increase=0,
        seed=17,
        output_format="text",
        output=None,
    )
    assert _cmd_benchmark_compare(args) == 0
    assert "TER Release Regression" in capsys.readouterr().out

    args.baseline_path, args.candidate_path = args.candidate_path, args.baseline_path
    output = tmp_path / "regression.json"
    args.output_format = "json"
    args.output = str(output)
    assert _cmd_benchmark_compare(args) == 2
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False
