"""Phase 8 reproducible release manifests and regression gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ReleaseGate:
    minimum_sessions: int = 1
    minimum_weighted_ter: float = 0.0
    maximum_waste_ratio: float = 1.0
    maximum_weighted_ter_drop: float = 1.0
    maximum_waste_ratio_increase: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_sessions < 1:
            raise ValueError("minimum_sessions must be at least 1")
        for name, value in (
            ("minimum_weighted_ter", self.minimum_weighted_ter),
            ("maximum_waste_ratio", self.maximum_waste_ratio),
            ("maximum_weighted_ter_drop", self.maximum_weighted_ter_drop),
            ("maximum_waste_ratio_increase", self.maximum_waste_ratio_increase),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class ReleaseAssessment:
    passed: bool
    violations: tuple[str, ...]
    weighted_ter_delta: float | None
    waste_ratio_delta: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "violations": list(self.violations),
            "weighted_ter_delta": self.weighted_ter_delta,
            "waste_ratio_delta": self.waste_ratio_delta,
        }


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    return float(value) if isinstance(value, (int, float)) else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_release_snapshot(
    results: Iterable[dict[str, Any]], *, version: str, source: str
) -> dict[str, Any]:
    records = list(results)
    total_tokens = sum(_number(item.get("total_tokens")) for item in records)
    aligned_tokens = sum(_number(item.get("aligned_tokens")) for item in records)
    waste_tokens = sum(_number(item.get("waste_tokens")) for item in records)
    ters = [_number(item.get("aggregate_ter")) for item in records]
    normalized = [
        {
            "session_id": str(item.get("session_id", "unknown")),
            "aggregate_ter": _number(item.get("aggregate_ter")),
            "total_tokens": int(_number(item.get("total_tokens"))),
            "aligned_tokens": int(_number(item.get("aligned_tokens"))),
            "waste_tokens": int(_number(item.get("waste_tokens"))),
        }
        for item in records
    ]
    normalized.sort(key=lambda item: str(item["session_id"]))
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "ter_version": version,
        "source": source,
        "sessions": len(records),
        "total_tokens": int(total_tokens),
        "aligned_tokens": int(aligned_tokens),
        "waste_tokens": int(waste_tokens),
        "average_ter": sum(ters) / len(ters) if ters else 0.0,
        "weighted_ter": aligned_tokens / total_tokens if total_tokens else 0.0,
        "waste_ratio": waste_tokens / total_tokens if total_tokens else 0.0,
        "p10_ter": _percentile(ters, 0.10),
        "median_ter": _percentile(ters, 0.50),
        "p90_ter": _percentile(ters, 0.90),
        "results_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def build_file_checksums(result_dir: Path) -> list[dict[str, object]]:
    checksums: list[dict[str, object]] = []
    for path in sorted(result_dir.glob("**/*.ter.json")):
        data = path.read_bytes()
        checksums.append(
            {
                "path": path.relative_to(result_dir).as_posix(),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return checksums


def evaluate_release(
    snapshot: dict[str, Any],
    gate: ReleaseGate,
    baseline: dict[str, Any] | None = None,
) -> ReleaseAssessment:
    sessions = int(_number(snapshot.get("sessions")))
    weighted_ter = _number(snapshot.get("weighted_ter"))
    waste_ratio = _number(snapshot.get("waste_ratio"))
    violations: list[str] = []
    if sessions < gate.minimum_sessions:
        violations.append(
            f"sessions {sessions} is below required {gate.minimum_sessions}"
        )
    if weighted_ter < gate.minimum_weighted_ter:
        violations.append(
            f"weighted TER {weighted_ter:.4f} is below {gate.minimum_weighted_ter:.4f}"
        )
    if waste_ratio > gate.maximum_waste_ratio:
        violations.append(
            f"waste ratio {waste_ratio:.4f} exceeds {gate.maximum_waste_ratio:.4f}"
        )

    weighted_delta: float | None = None
    waste_delta: float | None = None
    if baseline is not None:
        weighted_delta = weighted_ter - _number(baseline.get("weighted_ter"))
        waste_delta = waste_ratio - _number(baseline.get("waste_ratio"))
        if weighted_delta < -gate.maximum_weighted_ter_drop:
            violations.append(
                f"weighted TER dropped {-weighted_delta:.4f}, exceeding "
                f"{gate.maximum_weighted_ter_drop:.4f}"
            )
        if waste_delta > gate.maximum_waste_ratio_increase:
            violations.append(
                f"waste ratio increased {waste_delta:.4f}, exceeding "
                f"{gate.maximum_waste_ratio_increase:.4f}"
            )
    return ReleaseAssessment(
        passed=not violations,
        violations=tuple(violations),
        weighted_ter_delta=weighted_delta,
        waste_ratio_delta=waste_delta,
    )


def build_release_summary(
    snapshot: dict[str, Any], assessment: ReleaseAssessment
) -> str:
    status = "PASS" if assessment.passed else "FAIL"
    violations = "\n".join(f"- {item}" for item in assessment.violations) or "- None"
    return f"""# TER release validation

**Status:** {status}

| Metric | Value |
|---|---:|
| Sessions | {snapshot["sessions"]} |
| Weighted TER | {snapshot["weighted_ter"]:.4f} |
| Average TER | {snapshot["average_ter"]:.4f} |
| Waste ratio | {snapshot["waste_ratio"]:.2%} |
| P10 / Median / P90 TER | {snapshot["p10_ter"]:.4f} / {snapshot["median_ter"]:.4f} / {snapshot["p90_ter"]:.4f} |
| Results fingerprint | `{snapshot["results_sha256"]}` |

## Violations
{violations}
"""
