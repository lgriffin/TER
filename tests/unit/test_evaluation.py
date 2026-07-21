from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ter_calculator.commands.benchmark import _cmd_benchmark
from ter_calculator.evaluation import (
    BenchmarkRecord,
    calibrate_threshold,
    evaluate_benchmark,
    format_benchmark_report,
    load_benchmark,
)


def records() -> list[BenchmarkRecord]:
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
            "4", "s2", "generation", "aligned_response", "over_explanation", 0.8, 40
        ),
    ]


def test_load_benchmark_and_metadata(tmp_path: Path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        "# comment\n"
        '{"id":"1","session_id":"s","phase":"reasoning",'
        '"gold_label":"aligned_reasoning","predicted_label":"aligned_reasoning",'
        '"tokens":2,"source_line":8}\n',
        encoding="utf-8",
    )
    loaded = load_benchmark(path)
    assert loaded[0].tokens == 2
    assert loaded[0].metadata == {"source_line": 8}


@pytest.mark.parametrize(
    "line, message",
    [
        ("not-json", "Invalid JSON"),
        ('{"id":"1"}', "missing required fields"),
        (
            '{"id":"1","session_id":"s","phase":"reasoning","gold_label":"bad"}',
            "unknown gold_label",
        ),
        (
            '{"id":"1","session_id":"s","phase":"reasoning","gold_label":"aligned_reasoning","score":2}',
            "score must",
        ),
        (
            '{"id":"1","session_id":"s","phase":"reasoning","gold_label":"aligned_reasoning","tokens":0}',
            "tokens must",
        ),
    ],
)
def test_load_benchmark_validation(tmp_path: Path, line: str, message: str):
    path = tmp_path / "bad.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_benchmark(path)


def test_duplicate_and_empty_benchmark_rejected(tmp_path: Path):
    path = tmp_path / "duplicate.jsonl"
    row = '{"id":"1","session_id":"s","phase":"reasoning","gold_label":"aligned_reasoning"}\n'
    path.write_text(row + row, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate record"):
        load_benchmark(path)
    path.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no records"):
        load_benchmark(path)


def test_evaluate_binary_multiclass_token_and_bootstrap():
    report = evaluate_benchmark(records(), bootstrap_samples=50, seed=4)
    assert report.record_count == 4
    assert report.session_count == 2
    assert report.binary.true_positive == 1
    assert report.binary.false_positive == 1
    assert report.binary.false_negative == 1
    assert report.binary.true_negative == 1
    assert report.binary.precision == pytest.approx(0.5)
    assert report.token_weighted_binary.false_positive == 40
    assert report.confusion_matrix["unnecessary_tool_call"]["aligned_tool_call"] == 1
    assert report.bootstrap_intervals["precision"].lower >= 0
    assert report.recommended_threshold is not None


def test_threshold_override_and_calibration_precision_constraint():
    report = evaluate_benchmark(records(), threshold=0.85, bootstrap_samples=0)
    assert report.binary.false_positive == 0
    assert report.binary.true_positive == 1
    recommendation = calibrate_threshold(records(), beta=0.5, minimum_precision=1.0)
    assert recommendation is not None
    assert recommendation.metrics.precision == 1.0


def test_missing_predictions_and_no_scores_fail():
    incomplete = [BenchmarkRecord("1", "s", "reasoning", "aligned_reasoning")]
    with pytest.raises(ValueError, match="neither predicted_label"):
        evaluate_benchmark(incomplete)
    assert calibrate_threshold(incomplete) is None


def test_text_report_contains_advisory():
    rendered = format_benchmark_report(
        evaluate_benchmark(records(), bootstrap_samples=5)
    )
    assert "Binary waste detection" in rendered
    assert "Token-weighted" in rendered
    assert "production defaults are unchanged" in rendered


def test_benchmark_command_text_and_json(tmp_path: Path, capsys):
    source = Path("benchmarks/example_annotations.jsonl")
    args = SimpleNamespace(
        benchmark_path=str(source),
        threshold=None,
        bootstrap_samples=10,
        seed=17,
        beta=0.5,
        minimum_precision=0.0,
        output_format="text",
        output=None,
    )
    assert _cmd_benchmark(args) == 0
    assert "TER Benchmark Evaluation" in capsys.readouterr().out

    output = tmp_path / "report.json"
    args.output_format = "json"
    args.output = str(output)
    assert _cmd_benchmark(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["record_count"] == 6
