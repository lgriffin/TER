"""Empirical evaluation and threshold calibration for TER classifiers.

The benchmark format is JSON Lines. Each record must contain ``gold_label`` and
may contain ``predicted_label`` and/or a numeric ``score``. A higher score means
stronger evidence that the record is waste. Records may optionally include a
positive integer ``tokens`` value for token-weighted metrics.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .models import ALIGNED_LABELS, SpanLabel

_LABEL_VALUES = {label.value for label in SpanLabel}
_ALIGNED_VALUES = {label.value for label in ALIGNED_LABELS}


@dataclass(frozen=True)
class BenchmarkRecord:
    """One independently annotated benchmark unit."""

    record_id: str
    session_id: str
    phase: str
    gold_label: str
    predicted_label: str | None = None
    score: float | None = None
    tokens: int = 1
    category: str | None = None
    text: str | None = None
    annotator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gold_is_waste(self) -> bool:
        return self.gold_label not in _ALIGNED_VALUES

    @property
    def predicted_is_waste(self) -> bool | None:
        if self.predicted_label is None:
            return None
        return self.predicted_label not in _ALIGNED_VALUES


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    f0_5: float
    accuracy: float
    support: int


@dataclass(frozen=True)
class LabelMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence: float = 0.95


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    metrics: BinaryMetrics


@dataclass(frozen=True)
class BenchmarkReport:
    record_count: int
    session_count: int
    binary: BinaryMetrics
    token_weighted_binary: BinaryMetrics
    per_label: list[LabelMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    bootstrap_intervals: dict[str, ConfidenceInterval]
    recommended_threshold: ThresholdResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_benchmark(path: str | Path) -> list[BenchmarkRecord]:
    """Load and validate benchmark records from a JSONL file."""
    benchmark_path = Path(path)
    records: list[BenchmarkRecord] = []
    seen_ids: set[str] = set()

    with benchmark_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {benchmark_path}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            record = _record_from_mapping(payload, line_number)
            if record.record_id in seen_ids:
                raise ValueError(
                    f"Duplicate record id {record.record_id!r} on line {line_number}"
                )
            seen_ids.add(record.record_id)
            records.append(record)

    if not records:
        raise ValueError(f"Benchmark file contains no records: {benchmark_path}")
    return records


def _record_from_mapping(payload: dict[str, Any], line_number: int) -> BenchmarkRecord:
    required = ("id", "session_id", "phase", "gold_label")
    missing = [name for name in required if not payload.get(name)]
    if missing:
        raise ValueError(
            f"Line {line_number} is missing required fields: {', '.join(missing)}"
        )

    gold_label = str(payload["gold_label"])
    predicted = payload.get("predicted_label")
    predicted_label = str(predicted) if predicted is not None else None
    for field_name, label in (
        ("gold_label", gold_label),
        ("predicted_label", predicted_label),
    ):
        if label is not None and label not in _LABEL_VALUES:
            raise ValueError(f"Line {line_number} has unknown {field_name}: {label!r}")

    score_raw = payload.get("score")
    score = float(score_raw) if score_raw is not None else None
    if score is not None and (not math.isfinite(score) or score < 0.0 or score > 1.0):
        raise ValueError(f"Line {line_number} score must be between 0 and 1")

    tokens = int(payload.get("tokens", 1))
    if tokens <= 0:
        raise ValueError(f"Line {line_number} tokens must be a positive integer")

    known = {
        "id",
        "session_id",
        "phase",
        "gold_label",
        "predicted_label",
        "score",
        "tokens",
        "category",
        "text",
        "annotator",
    }
    metadata = {key: value for key, value in payload.items() if key not in known}
    return BenchmarkRecord(
        record_id=str(payload["id"]),
        session_id=str(payload["session_id"]),
        phase=str(payload["phase"]),
        gold_label=gold_label,
        predicted_label=predicted_label,
        score=score,
        tokens=tokens,
        category=str(payload["category"])
        if payload.get("category") is not None
        else None,
        text=str(payload["text"]) if payload.get("text") is not None else None,
        annotator=str(payload["annotator"])
        if payload.get("annotator") is not None
        else None,
        metadata=metadata,
    )


def evaluate_benchmark(
    records: Sequence[BenchmarkRecord],
    *,
    threshold: float | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 17,
    beta: float = 0.5,
    minimum_precision: float = 0.0,
) -> BenchmarkReport:
    """Evaluate predictions and optionally calibrate a waste-score threshold."""
    if not records:
        raise ValueError("At least one benchmark record is required")
    evaluated = _materialize_predictions(records, threshold)
    binary = _binary_metrics(evaluated)
    token_binary = _binary_metrics(evaluated, token_weighted=True)
    labels = sorted(_LABEL_VALUES | {r.gold_label for r in evaluated})
    matrix = _confusion_matrix(evaluated, labels)
    per_label = [
        _label_metrics(matrix, label)
        for label in labels
        if _label_support(matrix, label)
    ]
    intervals = _bootstrap_intervals(evaluated, bootstrap_samples, seed)
    recommended = calibrate_threshold(
        records, beta=beta, minimum_precision=minimum_precision
    )
    return BenchmarkReport(
        record_count=len(evaluated),
        session_count=len({record.session_id for record in evaluated}),
        binary=binary,
        token_weighted_binary=token_binary,
        per_label=per_label,
        confusion_matrix=matrix,
        bootstrap_intervals=intervals,
        recommended_threshold=recommended,
    )


def _materialize_predictions(
    records: Sequence[BenchmarkRecord], threshold: float | None
) -> list[BenchmarkRecord]:
    output: list[BenchmarkRecord] = []
    for record in records:
        predicted = record.predicted_label
        if threshold is not None and record.score is not None:
            predicted = (
                SpanLabel.REDUNDANT_REASONING.value
                if record.score >= threshold
                else SpanLabel.ALIGNED_REASONING.value
            )
        if predicted is None:
            raise ValueError(
                f"Record {record.record_id!r} has neither predicted_label nor a usable score"
            )
        output.append(
            BenchmarkRecord(
                record_id=record.record_id,
                session_id=record.session_id,
                phase=record.phase,
                gold_label=record.gold_label,
                predicted_label=predicted,
                score=record.score,
                tokens=record.tokens,
                category=record.category,
                text=record.text,
                annotator=record.annotator,
                metadata=record.metadata,
            )
        )
    return output


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _fbeta(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    return _safe_div((1 + beta_sq) * precision * recall, denominator)


def _binary_metrics(
    records: Sequence[BenchmarkRecord], *, token_weighted: bool = False
) -> BinaryMetrics:
    tp = fp = tn = fn = 0
    for record in records:
        weight = record.tokens if token_weighted else 1
        predicted = record.predicted_is_waste
        if predicted is None:
            raise ValueError(f"Record {record.record_id!r} has no prediction")
        if record.gold_is_waste and predicted:
            tp += weight
        elif not record.gold_is_waste and predicted:
            fp += weight
        elif not record.gold_is_waste and not predicted:
            tn += weight
        else:
            fn += weight
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return BinaryMetrics(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=_fbeta(precision, recall, 1.0),
        f0_5=_fbeta(precision, recall, 0.5),
        accuracy=_safe_div(tp + tn, tp + fp + tn + fn),
        support=tp + fp + tn + fn,
    )


def _confusion_matrix(
    records: Sequence[BenchmarkRecord], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    matrix = {gold: {predicted: 0 for predicted in labels} for gold in labels}
    for record in records:
        if record.predicted_label is None:
            raise ValueError(f"Record {record.record_id!r} has no prediction")
        matrix.setdefault(record.gold_label, {}).setdefault(record.predicted_label, 0)
        matrix[record.gold_label][record.predicted_label] += 1
    return matrix


def _label_support(matrix: dict[str, dict[str, int]], label: str) -> int:
    return sum(matrix.get(label, {}).values())


def _label_metrics(matrix: dict[str, dict[str, int]], label: str) -> LabelMetrics:
    tp = matrix.get(label, {}).get(label, 0)
    fp = sum(row.get(label, 0) for gold, row in matrix.items() if gold != label)
    fn = sum(
        count
        for predicted, count in matrix.get(label, {}).items()
        if predicted != label
    )
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return LabelMetrics(
        label=label,
        precision=precision,
        recall=recall,
        f1=_fbeta(precision, recall, 1.0),
        support=tp + fn,
    )


def calibrate_threshold(
    records: Sequence[BenchmarkRecord],
    *,
    beta: float = 0.5,
    minimum_precision: float = 0.0,
) -> ThresholdResult | None:
    """Select a score threshold, prioritising F-beta then precision.

    Only records with numeric scores participate. Higher scores must represent
    stronger waste evidence. Ties prefer higher precision and then the higher
    threshold, which is conservative when false positives are costly.
    """
    scored = [record for record in records if record.score is not None]
    if not scored:
        return None
    score_values = {0.0, 1.0}
    for record in scored:
        assert record.score is not None
        score_values.add(record.score)
    candidates = sorted(score_values)
    best: tuple[float, float, float, ThresholdResult] | None = None
    for threshold in candidates:
        evaluated = _materialize_predictions(scored, threshold)
        metrics = _binary_metrics(evaluated)
        if metrics.precision < minimum_precision:
            continue
        objective = _fbeta(metrics.precision, metrics.recall, beta)
        result = ThresholdResult(threshold=threshold, metrics=metrics)
        rank = (objective, metrics.precision, threshold, result)
        if best is None or rank[:3] > best[:3]:
            best = rank
    return best[3] if best else None


def _bootstrap_intervals(
    records: Sequence[BenchmarkRecord], samples: int, seed: int
) -> dict[str, ConfidenceInterval]:
    if samples <= 0:
        return {}
    rng = random.Random(seed)
    values: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "f0_5": [],
        "accuracy": [],
    }
    for _ in range(samples):
        resample = [records[rng.randrange(len(records))] for _ in records]
        metrics = _binary_metrics(resample)
        values["precision"].append(metrics.precision)
        values["recall"].append(metrics.recall)
        values["f0_5"].append(metrics.f0_5)
        values["accuracy"].append(metrics.accuracy)
    return {
        name: ConfidenceInterval(_percentile(series, 0.025), _percentile(series, 0.975))
        for name, series in values.items()
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def format_benchmark_report(report: BenchmarkReport) -> str:
    """Format a concise human-readable benchmark report."""
    binary = report.binary
    lines = [
        "TER Benchmark Evaluation",
        "========================",
        f"Records: {report.record_count} across {report.session_count} sessions",
        "",
        "Binary waste detection",
        f"  Precision: {binary.precision:.3f}",
        f"  Recall:    {binary.recall:.3f}",
        f"  F1:        {binary.f1:.3f}",
        f"  F0.5:      {binary.f0_5:.3f}",
        f"  Accuracy:  {binary.accuracy:.3f}",
        f"  Confusion: TP={binary.true_positive} FP={binary.false_positive} "
        f"TN={binary.true_negative} FN={binary.false_negative}",
        "",
        "Token-weighted waste detection",
        f"  Precision: {report.token_weighted_binary.precision:.3f}",
        f"  Recall:    {report.token_weighted_binary.recall:.3f}",
        f"  F0.5:      {report.token_weighted_binary.f0_5:.3f}",
    ]
    if report.bootstrap_intervals:
        lines.extend(["", "Bootstrap 95% intervals"])
        for name, interval in report.bootstrap_intervals.items():
            lines.append(f"  {name}: {interval.lower:.3f}–{interval.upper:.3f}")
    if report.recommended_threshold is not None:
        rec = report.recommended_threshold
        lines.extend(
            [
                "",
                "Recommended score threshold",
                f"  Threshold: {rec.threshold:.3f}",
                f"  Precision: {rec.metrics.precision:.3f}",
                f"  Recall:    {rec.metrics.recall:.3f}",
                f"  F0.5:      {rec.metrics.f0_5:.3f}",
                "  Note: recommendation is advisory; production defaults are unchanged.",
            ]
        )
    return "\n".join(lines)


def dump_report_json(report: BenchmarkReport) -> str:
    """Serialize a benchmark report as stable, indented JSON."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)
