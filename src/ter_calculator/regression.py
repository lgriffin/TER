"""Release-to-release benchmark comparison and quality gates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .evaluation import BenchmarkReport


@dataclass(frozen=True)
class MetricDelta:
    baseline: float
    candidate: float
    delta: float


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    actual: float
    limit: float
    message: str


@dataclass(frozen=True)
class RegressionReport:
    baseline_records: int
    candidate_records: int
    baseline_sessions: int
    candidate_sessions: int
    precision: MetricDelta
    recall: MetricDelta
    f0_5: MetricDelta
    accuracy: MetricDelta
    token_precision: MetricDelta
    token_recall: MetricDelta
    token_f0_5: MetricDelta
    false_positives: MetricDelta
    false_negatives: MetricDelta
    gates: list[GateResult]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def _delta(baseline: float, candidate: float) -> MetricDelta:
    return MetricDelta(
        baseline=baseline, candidate=candidate, delta=candidate - baseline
    )


def compare_benchmark_reports(
    baseline: BenchmarkReport,
    candidate: BenchmarkReport,
    *,
    minimum_precision: float = 0.0,
    maximum_precision_drop: float = 0.0,
    maximum_recall_drop: float = 1.0,
    maximum_f0_5_drop: float = 0.0,
    maximum_accuracy_drop: float = 1.0,
    maximum_false_positive_increase: int = 0,
) -> RegressionReport:
    """Compare benchmark reports and evaluate conservative release gates."""
    if baseline.record_count != candidate.record_count:
        raise ValueError(
            "Baseline and candidate must contain the same number of records"
        )
    if baseline.session_count != candidate.session_count:
        raise ValueError(
            "Baseline and candidate must contain the same number of sessions"
        )

    precision = _delta(baseline.binary.precision, candidate.binary.precision)
    recall = _delta(baseline.binary.recall, candidate.binary.recall)
    f0_5 = _delta(baseline.binary.f0_5, candidate.binary.f0_5)
    accuracy = _delta(baseline.binary.accuracy, candidate.binary.accuracy)
    token_precision = _delta(
        baseline.token_weighted_binary.precision,
        candidate.token_weighted_binary.precision,
    )
    token_recall = _delta(
        baseline.token_weighted_binary.recall,
        candidate.token_weighted_binary.recall,
    )
    token_f0_5 = _delta(
        baseline.token_weighted_binary.f0_5,
        candidate.token_weighted_binary.f0_5,
    )
    false_positives = _delta(
        float(baseline.binary.false_positive),
        float(candidate.binary.false_positive),
    )
    false_negatives = _delta(
        float(baseline.binary.false_negative),
        float(candidate.binary.false_negative),
    )

    gates = [
        GateResult(
            "minimum_precision",
            candidate.binary.precision >= minimum_precision,
            candidate.binary.precision,
            minimum_precision,
            "candidate precision must meet the configured floor",
        ),
        GateResult(
            "precision_drop",
            precision.delta >= -maximum_precision_drop,
            precision.delta,
            -maximum_precision_drop,
            "precision delta must not fall below the allowed regression",
        ),
        GateResult(
            "recall_drop",
            recall.delta >= -maximum_recall_drop,
            recall.delta,
            -maximum_recall_drop,
            "recall delta must not fall below the allowed regression",
        ),
        GateResult(
            "f0_5_drop",
            f0_5.delta >= -maximum_f0_5_drop,
            f0_5.delta,
            -maximum_f0_5_drop,
            "F0.5 delta must not fall below the allowed regression",
        ),
        GateResult(
            "accuracy_drop",
            accuracy.delta >= -maximum_accuracy_drop,
            accuracy.delta,
            -maximum_accuracy_drop,
            "accuracy delta must not fall below the allowed regression",
        ),
        GateResult(
            "false_positive_increase",
            false_positives.delta <= maximum_false_positive_increase,
            false_positives.delta,
            float(maximum_false_positive_increase),
            "false-positive count must not increase beyond the configured allowance",
        ),
    ]

    return RegressionReport(
        baseline_records=baseline.record_count,
        candidate_records=candidate.record_count,
        baseline_sessions=baseline.session_count,
        candidate_sessions=candidate.session_count,
        precision=precision,
        recall=recall,
        f0_5=f0_5,
        accuracy=accuracy,
        token_precision=token_precision,
        token_recall=token_recall,
        token_f0_5=token_f0_5,
        false_positives=false_positives,
        false_negatives=false_negatives,
        gates=gates,
    )


def format_regression_report(report: RegressionReport) -> str:
    """Render a concise CI-friendly comparison report."""

    def metric(name: str, value: MetricDelta) -> str:
        return (
            f"  {name:<18} {value.baseline:.3f} -> {value.candidate:.3f} "
            f"({value.delta:+.3f})"
        )

    lines = [
        "TER Release Regression",
        "======================",
        f"Records: {report.candidate_records} across {report.candidate_sessions} sessions",
        "",
        "Record-level metrics",
        metric("Precision", report.precision),
        metric("Recall", report.recall),
        metric("F0.5", report.f0_5),
        metric("Accuracy", report.accuracy),
        "",
        "Token-weighted metrics",
        metric("Precision", report.token_precision),
        metric("Recall", report.token_recall),
        metric("F0.5", report.token_f0_5),
        "",
        "Error counts",
        metric("False positives", report.false_positives),
        metric("False negatives", report.false_negatives),
        "",
        "Quality gates",
    ]
    for gate in report.gates:
        status = "PASS" if gate.passed else "FAIL"
        lines.append(
            f"  [{status}] {gate.name}: actual={gate.actual:.3f}, limit={gate.limit:.3f}"
        )
    lines.extend(["", f"Result: {'PASS' if report.passed else 'FAIL'}"])
    return "\n".join(lines)


def dump_regression_json(report: RegressionReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)
