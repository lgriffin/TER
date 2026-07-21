"""History-driven, privacy-preserving TER policy optimization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .history_store import HistoryRecord


@dataclass(frozen=True)
class AdaptivePolicy:
    """Project-specific recommendations derived only from aggregate history."""

    version: str
    project: str
    generated_at: str
    sample_size: int
    confidence: str
    thresholds: dict[str, float]
    phase_weights: dict[str, float]
    token_budget: dict[str, int]
    intervention: dict[str, int | float]
    evidence: dict[str, float | int | str | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def learn_policy(
    records: Iterable[HistoryRecord],
    project: str,
    *,
    minimum_samples: int = 5,
) -> AdaptivePolicy:
    """Learn a bounded policy from project history without retaining raw prompts."""
    rows = list(records)
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be at least 1")
    if not rows:
        raise ValueError(f"No history records found for project {project!r}")

    sample_size = len(rows)
    total_tokens = sum(max(0, row.token_count) for row in rows)
    waste_tokens = sum(max(0, row.waste_tokens) for row in rows)
    waste_ratio = waste_tokens / total_tokens if total_tokens else 0.0
    average_ter = sum(row.aggregate_ter for row in rows) / sample_size

    token_values = sorted(max(1, row.token_count) for row in rows)
    phase_efficiency = _phase_efficiencies(rows)
    phase_weights = _inverse_efficiency_weights(phase_efficiency)
    waste_totals = _waste_totals(rows)
    repetition_tokens = sum(
        value
        for name, value in waste_totals.items()
        if any(
            term in name.lower() for term in ("repeat", "duplicate", "retry", "loop")
        )
    )
    repetition_share = repetition_tokens / waste_tokens if waste_tokens else 0.0

    thresholds = {
        "similarity": round(_clamp(0.36 + (0.82 - average_ter) * 0.20, 0.30, 0.55), 3),
        "confidence": round(_clamp(0.72 + waste_ratio * 0.35, 0.70, 0.90), 3),
        "restatement": round(_clamp(0.88 - repetition_share * 0.16, 0.70, 0.90), 3),
    }
    intervention = {
        "min_repetitive_reads": _bounded_count(4 - repetition_share * 2, 2, 5),
        "min_edit_fragments": _bounded_count(4 - waste_ratio * 4, 2, 5),
        "min_repeated_commands": _bounded_count(4 - repetition_share * 2, 2, 5),
        "min_duplicate_calls": _bounded_count(3 - repetition_share, 2, 4),
        "reasoning_similarity": round(
            _clamp(0.92 - repetition_share * 0.12, 0.78, 0.94), 3
        ),
    }
    token_budget = {
        "soft_limit": int(round(median(token_values))),
        "recommended": int(round(_quantile(token_values, 0.75))),
        "hard_limit": int(round(_quantile(token_values, 0.90) * 1.10)),
    }
    confidence = _confidence(sample_size, minimum_samples)
    main_waste = (
        max(waste_totals, key=lambda key: waste_totals[key]) if waste_totals else None
    )

    return AdaptivePolicy(
        version="1",
        project=project,
        generated_at=datetime.now(timezone.utc).isoformat(),
        sample_size=sample_size,
        confidence=confidence,
        thresholds=thresholds,
        phase_weights=phase_weights,
        token_budget=token_budget,
        intervention=intervention,
        evidence={
            "average_ter": round(average_ter, 4),
            "waste_ratio": round(waste_ratio, 4),
            "repetition_share": round(repetition_share, 4),
            "main_waste_source": main_waste,
            "minimum_samples": minimum_samples,
        },
    )


def personalize_policy(
    policy: AdaptivePolicy, prediction: dict[str, object]
) -> AdaptivePolicy:
    """Adjust token limits conservatively using a prompt-level TER prediction."""
    if not prediction.get("available"):
        return policy
    predicted_value = prediction.get("predicted_ter")
    if not isinstance(predicted_value, (int, float)):
        return policy
    predicted = float(predicted_value)
    multiplier = _clamp(0.85 + predicted * 0.30, 0.90, 1.15)
    budget = {
        name: max(1, int(round(value * multiplier)))
        for name, value in policy.token_budget.items()
    }
    evidence = dict(policy.evidence)
    evidence["predicted_ter"] = round(predicted, 4)
    neighbors = prediction.get("neighbors", 0)
    evidence["prompt_neighbors"] = int(neighbors) if isinstance(neighbors, int) else 0
    return AdaptivePolicy(
        version=policy.version,
        project=policy.project,
        generated_at=policy.generated_at,
        sample_size=policy.sample_size,
        confidence=policy.confidence,
        thresholds=policy.thresholds,
        phase_weights=policy.phase_weights,
        token_budget=budget,
        intervention=policy.intervention,
        evidence=evidence,
    )


def save_policy(policy: AdaptivePolicy, destination: str | Path) -> Path:
    """Atomically write a policy as portable JSON."""
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _phase_efficiencies(records: list[HistoryRecord]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for record in records:
        for phase, score in record.phase_ter.items():
            if math.isfinite(score):
                values.setdefault(phase, []).append(_clamp(score, 0.0, 1.0))
    return {
        phase: sum(scores) / len(scores) for phase, scores in values.items() if scores
    }


def _inverse_efficiency_weights(efficiencies: dict[str, float]) -> dict[str, float]:
    phases = ("reasoning", "tool_use", "generation")
    raw = {phase: 1.05 - efficiencies.get(phase, 0.75) for phase in phases}
    total = sum(raw.values()) or 1.0
    return {phase: round(value / total, 4) for phase, value in raw.items()}


def _waste_totals(records: list[HistoryRecord]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        for name, value in record.waste_breakdown.items():
            totals[name] = totals.get(name, 0) + max(0, int(value))
    return totals


def _quantile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    position = fraction * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _confidence(sample_size: int, minimum_samples: int) -> str:
    if sample_size < minimum_samples:
        return "insufficient"
    if sample_size < 20:
        return "experimental"
    if sample_size < 50:
        return "stable"
    return "mature"


def _bounded_count(value: float, minimum: int, maximum: int) -> int:
    return int(_clamp(round(value), minimum, maximum))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
