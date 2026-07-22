"""Phase 7 ecosystem integration artifacts and CI release gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class IntegrationGate:
    minimum_ter: float = 0.0
    maximum_waste_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_ter <= 1.0:
            raise ValueError("minimum_ter must be between 0 and 1")
        if not 0.0 <= self.maximum_waste_ratio <= 1.0:
            raise ValueError("maximum_waste_ratio must be between 0 and 1")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    sessions: int
    average_ter: float
    weighted_ter: float
    waste_ratio: float
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "sessions": self.sessions,
            "average_ter": self.average_ter,
            "weighted_ter": self.weighted_ter,
            "waste_ratio": self.waste_ratio,
            "violations": list(self.violations),
        }


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    return float(value) if isinstance(value, (int, float)) else 0.0


def evaluate_gate(
    results: Iterable[dict[str, Any]], gate: IntegrationGate
) -> GateResult:
    records = list(results)
    total_tokens = sum(_number(item.get("total_tokens")) for item in records)
    aligned_tokens = sum(_number(item.get("aligned_tokens")) for item in records)
    waste_tokens = sum(_number(item.get("waste_tokens")) for item in records)
    ters = [_number(item.get("aggregate_ter")) for item in records]
    average_ter = sum(ters) / len(ters) if ters else 0.0
    weighted_ter = aligned_tokens / total_tokens if total_tokens else 0.0
    waste_ratio = waste_tokens / total_tokens if total_tokens else 0.0
    violations: list[str] = []
    if weighted_ter < gate.minimum_ter:
        violations.append(
            f"weighted TER {weighted_ter:.4f} is below {gate.minimum_ter:.4f}"
        )
    if waste_ratio > gate.maximum_waste_ratio:
        violations.append(
            f"waste ratio {waste_ratio:.4f} exceeds {gate.maximum_waste_ratio:.4f}"
        )
    return GateResult(
        passed=not violations,
        sessions=len(records),
        average_ter=average_ter,
        weighted_ter=weighted_ter,
        waste_ratio=waste_ratio,
        violations=tuple(violations),
    )


def build_step_summary(result: GateResult, source: str) -> str:
    status = "✅ Passed" if result.passed else "❌ Failed"
    violations = "\n".join(f"- {item}" for item in result.violations) or "- None"
    return f"""# TER integration report

**Status:** {status}
**Source:** `{source}`

| Metric | Value |
|---|---:|
| Sessions | {result.sessions} |
| Average TER | {result.average_ter:.4f} |
| Weighted TER | {result.weighted_ter:.4f} |
| Waste ratio | {result.waste_ratio:.2%} |

## Gate violations
{violations}
"""


def build_github_annotations(result: GateResult) -> str:
    if result.passed:
        return (
            "::notice title=TER quality gate::"
            f"Passed with weighted TER {result.weighted_ter:.4f} and "
            f"waste ratio {result.waste_ratio:.2%}"
        )
    return "\n".join(
        f"::error title=TER quality gate::{violation}"
        for violation in result.violations
    )


def build_sarif(
    results: Iterable[dict[str, Any]], gate_result: GateResult
) -> dict[str, Any]:
    sarif_results: list[dict[str, Any]] = []
    for item in results:
        session_id = str(item.get("session_id", "unknown"))
        ter = _number(item.get("aggregate_ter"))
        waste = int(_number(item.get("waste_tokens")))
        if waste <= 0:
            continue
        sarif_results.append(
            {
                "ruleId": "TER001",
                "level": "warning" if ter >= 0.75 else "error",
                "message": {
                    "text": f"Session {session_id} has TER {ter:.4f} and {waste} waste tokens."
                },
                "properties": {
                    "sessionId": session_id,
                    "aggregateTer": ter,
                    "wasteTokens": waste,
                },
            }
        )
    for violation in gate_result.violations:
        sarif_results.append(
            {
                "ruleId": "TER-GATE",
                "level": "error",
                "message": {"text": violation},
            }
        )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "TER",
                        "version": "2.0.8",
                        "informationUri": "https://github.com/cplegendre/TER",
                        "rules": [
                            {
                                "id": "TER001",
                                "name": "TokenWasteDetected",
                                "shortDescription": {"text": "Token waste detected"},
                            },
                            {
                                "id": "TER-GATE",
                                "name": "QualityGateFailed",
                                "shortDescription": {"text": "TER quality gate failed"},
                            },
                        ],
                    }
                },
                "results": sarif_results,
            }
        ],
    }


def atomic_write_text(path: str | Path, content: str) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
    return target


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
