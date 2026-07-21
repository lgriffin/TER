"""Metric-driven intervention policy and outcome evaluation.

This module is dependency-light so it can be used by both the live monitor and
Claude Code hooks. It deliberately separates detection, policy, persistence,
and outcome classification.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class InterventionAction(str, Enum):
    NONE = "none"
    WARN = "warn"
    REFRESH_CONTEXT = "refresh_context"
    REPLAN = "replan"
    BLOCK = "block"


@dataclass(frozen=True)
class MetricSnapshot:
    timestamp: float
    ter: float
    waste_ratio: float
    context_tokens: int = 0
    context_growth_rate: float = 1.0
    drift_score: float = 0.0
    repeated_tool_calls: int = 0
    reasoning_loop_streak: int = 0
    cost_per_1k_tokens: float = 0.003

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "MetricSnapshot":
        return cls(
            timestamp=float(value.get("timestamp", time.time())),
            ter=float(value.get("ter", value.get("aggregate_ter", 0.0))),
            waste_ratio=float(value.get("waste_ratio", 0.0)),
            context_tokens=int(
                value.get("context_tokens", value.get("total_tokens", 0))
            ),
            context_growth_rate=float(value.get("context_growth_rate", 1.0)),
            drift_score=float(
                value.get("drift_score", value.get("drift_magnitude", 0.0))
            ),
            repeated_tool_calls=int(value.get("repeated_tool_calls", 0)),
            reasoning_loop_streak=int(value.get("reasoning_loop_streak", 0)),
            cost_per_1k_tokens=float(value.get("cost_per_1k_tokens", 0.003)),
        )


@dataclass(frozen=True)
class PolicyDecision:
    action: InterventionAction
    reason: str
    severity: str = "info"
    confidence: float = 0.0
    metrics: dict[str, float | int] = field(default_factory=dict)
    cooldown_seconds: int = 0


@dataclass
class PolicyState:
    recent_snapshots: list[MetricSnapshot] = field(default_factory=list)
    last_action_at: dict[str, float] = field(default_factory=dict)
    consecutive_degraded_windows: int = 0


@dataclass(frozen=True)
class PolicyConfig:
    ter_drop_warning: float = 0.12
    ter_drop_replan: float = 0.20
    waste_ratio_warning: float = 0.25
    waste_ratio_replan: float = 0.40
    degraded_windows_required: int = 3
    severe_windows_required: int = 2
    refresh_cooldown_seconds: int = 120
    replan_cooldown_seconds: int = 180
    history_size: int = 10


@dataclass(frozen=True)
class InterventionRecord:
    intervention_id: str
    session_id: str
    action: str
    issued_at: float
    baseline: MetricSnapshot
    reason: str
    related_memory_ids: list[str] = field(default_factory=list)
    evaluation_due_after_events: int = 5


@dataclass(frozen=True)
class ComplianceResult:
    acknowledged: bool
    followed: bool
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0


def no_action() -> PolicyDecision:
    return PolicyDecision(action=InterventionAction.NONE, reason="Metrics are stable.")


def evaluate_policy(
    snapshot: MetricSnapshot,
    state: PolicyState,
    config: PolicyConfig | None = None,
) -> PolicyDecision:
    """Evaluate sustained TER degradation, waste, and cooldowns."""
    cfg = config or PolicyConfig()
    state.recent_snapshots.append(snapshot)
    state.recent_snapshots = state.recent_snapshots[-cfg.history_size :]
    if len(state.recent_snapshots) < 4:
        return no_action()

    baseline_rows = state.recent_snapshots[-4:-1]
    baseline_ter = sum(row.ter for row in baseline_rows) / len(baseline_rows)
    ter_drop = baseline_ter - snapshot.ter
    degraded = (
        ter_drop >= cfg.ter_drop_warning
        and snapshot.waste_ratio >= cfg.waste_ratio_warning
    )
    severe = (
        ter_drop >= cfg.ter_drop_replan
        and snapshot.waste_ratio >= cfg.waste_ratio_replan
    )

    state.consecutive_degraded_windows = (
        state.consecutive_degraded_windows + 1 if degraded else 0
    )
    metrics: dict[str, float | int] = {
        "ter": snapshot.ter,
        "baseline_ter": baseline_ter,
        "ter_drop": ter_drop,
        "waste_ratio": snapshot.waste_ratio,
        "context_growth_rate": snapshot.context_growth_rate,
        "degraded_windows": state.consecutive_degraded_windows,
    }

    if severe and state.consecutive_degraded_windows >= cfg.severe_windows_required:
        decision = PolicyDecision(
            action=InterventionAction.REPLAN,
            reason="TER dropped materially while waste remained elevated.",
            severity="critical",
            confidence=0.90,
            metrics=metrics,
            cooldown_seconds=cfg.replan_cooldown_seconds,
        )
    elif (
        degraded and state.consecutive_degraded_windows >= cfg.degraded_windows_required
    ):
        decision = PolicyDecision(
            action=InterventionAction.REFRESH_CONTEXT,
            reason="Sustained TER degradation and elevated waste detected.",
            severity="warning",
            confidence=0.80,
            metrics=metrics,
            cooldown_seconds=cfg.refresh_cooldown_seconds,
        )
    else:
        return no_action()

    last = state.last_action_at.get(decision.action.value)
    if last is not None and snapshot.timestamp - last < decision.cooldown_seconds:
        return no_action()
    state.last_action_at[decision.action.value] = snapshot.timestamp
    return decision


def build_recovery_instruction(
    decision: PolicyDecision, memory_guidance: str = ""
) -> str:
    metrics = decision.metrics
    summary = (
        f"TER {float(metrics.get('ter', 0.0)):.2f}, "
        f"baseline {float(metrics.get('baseline_ter', 0.0)):.2f}, "
        f"waste {float(metrics.get('waste_ratio', 0.0)):.0%}."
    )
    if decision.action == InterventionAction.REPLAN:
        body = (
            "[TER Replan Required]\n"
            f"{summary}\n\n"
            "Stop the current approach before invoking another tool. Return: objective; "
            "known facts; failed approaches; remaining tasks; the smallest next verifiable "
            "action; and the test that proves it worked. Do not repeat a failed or duplicate action."
        )
    else:
        body = (
            "[TER Context Refresh]\n"
            f"{summary}\n\n"
            "Before continuing: restate the objective, summarize completed verified work, "
            "identify the immediate blocker, discard obsolete hypotheses and duplicate context, "
            "then continue with one concrete next action."
        )
    return body + ("\n\n" + memory_guidance if memory_guidance else "")


def pending_intervention_path(root: str | Path, session_id: str) -> Path:
    return Path(root) / ".ter" / "runtime" / session_id / "pending-intervention.json"


def write_pending_intervention(
    path: str | Path, record: InterventionRecord, decision: PolicyDecision
) -> None:
    payload = {
        "record": _record_to_dict(record),
        "decision": _decision_to_dict(decision),
        "consumed": False,
    }
    _atomic_json(Path(path), payload)


def consume_pending_intervention(
    path: str | Path, *, now: float | None = None
) -> tuple[InterventionRecord, PolicyDecision] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if payload.get("consumed"):
        return None
    record_data = payload.get("record", {})
    decision_data = payload.get("decision", {})
    issued = float(record_data.get("issued_at", 0.0))
    cooldown = int(decision_data.get("cooldown_seconds", 0))
    current = time.time() if now is None else now
    if cooldown and current > issued + cooldown:
        return None
    payload["consumed"] = True
    _atomic_json(source, payload)
    return _record_from_dict(record_data), _decision_from_dict(decision_data)


def new_intervention_record(
    session_id: str,
    decision: PolicyDecision,
    baseline: MetricSnapshot,
    related_memory_ids: list[str] | None = None,
) -> InterventionRecord:
    return InterventionRecord(
        intervention_id=f"int-{uuid.uuid4().hex[:10]}",
        session_id=session_id,
        action=decision.action.value,
        issued_at=baseline.timestamp,
        baseline=baseline,
        reason=decision.reason,
        related_memory_ids=related_memory_ids or [],
    )


def classify_effect(
    baseline: MetricSnapshot,
    post: MetricSnapshot,
    compliance: ComplianceResult,
    *,
    meaningful_delta: float = 0.08,
) -> str:
    ter_gain = post.ter - baseline.ter
    waste_reduction = baseline.waste_ratio - post.waste_ratio
    if ter_gain >= meaningful_delta and waste_reduction >= meaningful_delta:
        return "improved"
    if ter_gain <= -meaningful_delta or post.waste_ratio >= baseline.waste_ratio + 0.10:
        return "regressed"
    if compliance.followed:
        return "neutral"
    if compliance.acknowledged:
        return "acknowledged_not_followed"
    return "ignored"


def append_intervention_outcome(
    path: str | Path,
    *,
    record: InterventionRecord,
    post: MetricSnapshot,
    compliance: ComplianceResult,
) -> dict[str, Any]:
    effect = classify_effect(record.baseline, post, compliance)
    before_waste_cost = (
        record.baseline.context_tokens * record.baseline.waste_ratio / 1000.0
    ) * record.baseline.cost_per_1k_tokens
    after_waste_cost = (
        post.context_tokens * post.waste_ratio / 1000.0
    ) * post.cost_per_1k_tokens
    estimated_cost_waste_usd = before_waste_cost - after_waste_cost
    row = {
        "intervention_id": record.intervention_id,
        "session_id": record.session_id,
        "intervention_type": record.action,
        "issued_at": record.issued_at,
        "evaluated_at": post.timestamp,
        "acknowledged": compliance.acknowledged,
        "followed": compliance.followed,
        "effect": effect,
        "confidence": compliance.confidence,
        "evidence": compliance.evidence,
        "before": asdict(record.baseline),
        "after": asdict(post),
        "deltas": {
            "ter": post.ter - record.baseline.ter,
            "waste_ratio": post.waste_ratio - record.baseline.waste_ratio,
            "estimated_cost_waste_usd": estimated_cost_waste_usd,
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def _record_to_dict(record: InterventionRecord) -> dict[str, Any]:
    value = asdict(record)
    return value


def _record_from_dict(value: dict[str, Any]) -> InterventionRecord:
    return InterventionRecord(
        intervention_id=str(value["intervention_id"]),
        session_id=str(value["session_id"]),
        action=str(value["action"]),
        issued_at=float(value["issued_at"]),
        baseline=MetricSnapshot.from_mapping(value["baseline"]),
        reason=str(value["reason"]),
        related_memory_ids=[str(item) for item in value.get("related_memory_ids", [])],
        evaluation_due_after_events=int(value.get("evaluation_due_after_events", 5)),
    )


def _decision_to_dict(decision: PolicyDecision) -> dict[str, Any]:
    value = asdict(decision)
    value["action"] = decision.action.value
    return value


def _decision_from_dict(value: dict[str, Any]) -> PolicyDecision:
    return PolicyDecision(
        action=InterventionAction(str(value["action"])),
        reason=str(value["reason"]),
        severity=str(value.get("severity", "info")),
        confidence=float(value.get("confidence", 0.0)),
        metrics=dict(value.get("metrics", {})),
        cooldown_seconds=int(value.get("cooldown_seconds", 0)),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
