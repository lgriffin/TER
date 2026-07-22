"""Phase 3 real-time intervention mechanisms for Claude Code hooks.

The module is deliberately dependency-light so hook execution remains fast.  It
supports four event classes:

* ``SessionStart``: inject a task-complexity based thinking-budget hint.
* ``PreToolUse``: block exact duplicate tool calls before execution.
* ``PermissionRequest`` / denied tool results: break repeated permission loops.
* ``Stop`` / assistant-message payloads: detect highly repetitive reasoning and
  inject a move-to-action instruction.

All functions are pure apart from updates to :class:`HookSessionState`, making
interventions deterministic and easy to test.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from pathlib import Path
from collections import Counter
from typing import Any

from .adaptive_budget import recommend_budget
from .closed_loop import build_memory_guidance, record_outcome, resolve_project_root
from .intervention_policy import (
    ComplianceResult,
    InterventionAction,
    InterventionRecord,
    MetricSnapshot,
    PolicyConfig,
    PolicyState,
    append_intervention_outcome,
    build_recovery_instruction,
    evaluate_policy,
    new_intervention_record,
    pending_intervention_path,
    consume_pending_intervention,
    write_pending_intervention,
)
from .hook_monitor import (
    HookConfig,
    HookSessionState,
    WasteAlert,
    _compute_tool_signature,
)

_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _extract_text(event_data: dict[str, Any]) -> str:
    """Extract assistant or prompt text from common Claude Code hook payloads."""
    for key in ("assistant_message", "message", "text", "prompt"):
        value = event_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    content = event_data.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks).strip()
    return ""


def _token_cosine(left: str, right: str) -> float:
    """Return cosine similarity over normalized word-frequency vectors."""
    a = Counter(_WORD_RE.findall(left.lower()))
    b = Counter(_WORD_RE.findall(right.lower()))
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(token, 0) for token, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def check_reasoning_loop(
    text: str,
    state: HookSessionState,
    *,
    threshold: float = 0.88,
    min_consecutive: int = 2,
) -> WasteAlert | None:
    """Detect consecutive assistant messages that substantially restate each other."""
    if not text or len(_WORD_RE.findall(text)) < 8:
        return None
    previous = state.recent_reasoning[-1] if state.recent_reasoning else ""
    similarity = _token_cosine(previous, text) if previous else 0.0
    state.recent_reasoning.append(text[-4000:])
    state.recent_reasoning = state.recent_reasoning[-4:]

    if similarity >= threshold:
        state.reasoning_loop_streak += 1
    else:
        state.reasoning_loop_streak = 0

    if state.reasoning_loop_streak >= max(1, min_consecutive - 1):
        return WasteAlert(
            pattern_type="reasoning_loop",
            severity="warning",
            message=(
                "You appear to be restating prior reasoning "
                f"(similarity {similarity:.0%}). Move to a concrete action, "
                "test, decision, or final answer instead of repeating the analysis."
            ),
            details={"similarity": round(similarity, 4)},
        )
    return None


def check_pre_tool_duplicate(
    event_data: dict[str, Any],
    state: HookSessionState,
    *,
    threshold: int = 2,
) -> WasteAlert | None:
    """Detect an exact duplicate before tool execution without double-counting it."""
    tool_name = str(event_data.get("tool_name", ""))
    tool_input = event_data.get("tool_input", {})
    if not tool_name or not isinstance(tool_input, dict):
        return None
    signature = _compute_tool_signature(tool_name, tool_input)
    prior_count = state.tool_call_counts.get(signature, 0)
    if prior_count >= max(1, threshold - 1):
        previous = state.tool_result_summaries.get(signature, "result already exists")
        return WasteAlert(
            pattern_type="duplicate_tool_call_prevented",
            severity="warning",
            message=(
                f"This exact {tool_name} call already ran {prior_count} time(s). "
                f"Previous outcome: {previous}. Reuse that result or change the parameters."
            ),
            details={
                "tool_name": tool_name,
                "signature": signature,
                "prior_count": prior_count,
            },
        )
    return None


def record_tool_result(event_data: dict[str, Any], state: HookSessionState) -> None:
    """Store a short result summary for future duplicate-call guidance."""
    tool_name = str(event_data.get("tool_name", ""))
    tool_input = event_data.get("tool_input", {})
    if not tool_name or not isinstance(tool_input, dict):
        return
    signature = _compute_tool_signature(tool_name, tool_input)
    result = event_data.get("tool_response", event_data.get("tool_result", "completed"))
    if isinstance(result, (dict, list)):
        summary = json.dumps(result, sort_keys=True)
    else:
        summary = str(result)
    summary = re.sub(r"\s+", " ", summary).strip()[:180] or "completed"
    state.tool_result_summaries[signature] = summary


def check_permission_loop(
    event_data: dict[str, Any],
    state: HookSessionState,
    *,
    threshold: int = 2,
) -> WasteAlert | None:
    """Track repeated denied permission requests for the same tool."""
    tool_name = str(event_data.get("tool_name", "unknown"))
    decision = str(
        event_data.get(
            "permission_decision",
            event_data.get("decision", event_data.get("status", "")),
        )
    ).lower()
    denied = decision in {"deny", "denied", "reject", "rejected"} or bool(
        event_data.get("is_denied")
    )
    if not denied:
        return None
    state.denied_tool_counts[tool_name] = state.denied_tool_counts.get(tool_name, 0) + 1
    count = state.denied_tool_counts[tool_name]
    if count >= threshold:
        return WasteAlert(
            pattern_type="permission_loop",
            severity="warning",
            message=(
                f"{tool_name} has been denied {count} times. Stop requesting the same "
                "permission and use a different, already-authorized approach."
            ),
            details={"tool_name": tool_name, "denial_count": count},
        )
    return None


def _pre_send_acknowledgement(text: str) -> str | None:
    lowered = text.lower()
    if "[ter ack pre_send_check]" in lowered:
        return "acknowledged"
    if "[ter override pre_send_check]" in lowered:
        return "overridden"
    return None


def _lesson_matches(
    path: Path, query: str, threshold: float, limit: int = 3
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    matches: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-1000:]
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        summary = str(row.get("summary", ""))
        score = _token_cosine(query, summary)
        if summary and score >= threshold:
            matches.append(
                {
                    "score": score,
                    "lesson_id": str(
                        row.get("lesson_id", row.get("timestamp", "unknown"))
                    ),
                    "path": "session-lesson",
                    "excerpt": summary,
                    "source_type": "lesson",
                }
            )
    matches.sort(key=lambda item: float(item["score"]), reverse=True)
    return matches[:limit]


def check_pre_send_pattern(
    event_data: dict[str, Any], state: HookSessionState, config: HookConfig
) -> tuple[WasteAlert | None, str | None]:
    """Check a user prompt against repository memory and durable lessons."""
    if not config.pre_send_check_enabled:
        return None, None
    prompt = _extract_text(event_data)
    if not prompt:
        return None, None
    acknowledgement = _pre_send_acknowledgement(prompt)
    if acknowledgement and state.pre_send_pending:
        return None, acknowledgement
    now = time.time()
    if now - state.pre_send_last_check_at < max(0, config.pre_send_cooldown_seconds):
        return None, None
    root = resolve_project_root(event_data)
    guidance, matches = build_memory_guidance(
        event_data,
        index_path=config.memory_index,
        limit=config.memory_limit,
        minimum_score=config.pre_send_similarity_threshold,
    )
    lessons = _lesson_matches(
        Path(config.lesson_store or root / ".ter" / "session-lessons.jsonl"),
        prompt,
        config.pre_send_similarity_threshold,
    )
    combined = [*matches, *lessons]
    if not combined:
        state.pre_send_last_check_at = now
        return None, "no_match"
    best = max(combined, key=lambda item: float(item.get("score", 0.0)))
    reference = (
        f"lesson {best.get('lesson_id')}"
        if best.get("source_type") == "lesson"
        else f"{best.get('path')}:{best.get('start_line', 0)}"
    )
    check_id = uuid.uuid4().hex[:12]
    state.pre_send_pending = {
        "check_id": check_id,
        "reference": reference,
        "score": float(best.get("score", 0.0)),
        "issued_at": now,
    }
    state.pre_send_last_check_at = now
    message = (
        f"A similar prior pattern was found at {reference} "
        f"(similarity {float(best.get('score', 0.0)):.0%}). Review it before sending. "
        "Resubmit with [TER ACK pre_send_check] to acknowledge and proceed, or "
        "[TER OVERRIDE pre_send_check] to proceed despite the warning."
    )
    if guidance:
        message += "\n\n" + guidance
    return WasteAlert(
        pattern_type="pre_send_check",
        severity="warning",
        message=message,
        details={
            "check_id": check_id,
            "reference": reference,
            "similarity": round(float(best.get("score", 0.0)), 6),
        },
    ), None


def build_budget_hint(prompt: str) -> tuple[str, dict[str, Any]]:
    """Build a session-start budget recommendation from the initial prompt."""
    recommendation = recommend_budget(prompt or "general coding task")
    message = (
        "[TER Budget Hint] "
        f"Task complexity is {recommendation.complexity.value}; target at most "
        f"{recommendation.max_thinking_tokens:,} thinking tokens and prefer the "
        f"{recommendation.model_tier.value} tier. Treat this as guidance, not a hard limit."
    )
    metadata = {
        "complexity": recommendation.complexity.value,
        "model_tier": recommendation.model_tier.value,
        "max_thinking_tokens": recommendation.max_thinking_tokens,
        "confidence": recommendation.confidence,
    }
    return message, metadata


def _policy_from_state(state: HookSessionState) -> PolicyState:
    return PolicyState(
        recent_snapshots=[
            MetricSnapshot.from_mapping(item) for item in state.policy_snapshots
        ],
        last_action_at=dict(state.policy_last_action_at),
        consecutive_degraded_windows=state.policy_degraded_windows,
    )


def _save_policy_state(policy: PolicyState, state: HookSessionState) -> None:
    from dataclasses import asdict

    state.policy_snapshots = [asdict(item) for item in policy.recent_snapshots]
    state.policy_last_action_at = dict(policy.last_action_at)
    state.policy_degraded_windows = policy.consecutive_degraded_windows


def _metric_payload(event_data: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("ter_metrics", "metrics", "ter_signal"):
        value = event_data.get(key)
        if isinstance(value, dict) and ("ter" in value or "aggregate_ter" in value):
            return value
    return None


def _evaluate_active_interventions(
    event_data: dict[str, Any], state: HookSessionState, config: HookConfig
) -> None:
    metrics = _metric_payload(event_data)
    if not metrics or not state.active_interventions:
        return
    post = MetricSnapshot.from_mapping(metrics)
    root = resolve_project_root(event_data)
    outcome_path = config.outcome_store or str(
        root / ".ter" / "intervention-outcomes.jsonl"
    )
    remaining: list[dict[str, Any]] = []
    for item in state.active_interventions:
        item["events_since_issue"] = int(item.get("events_since_issue", 0)) + 1
        record = InterventionRecord(
            intervention_id=str(item["intervention_id"]),
            session_id=str(item["session_id"]),
            action=str(item["action"]),
            issued_at=float(item["issued_at"]),
            baseline=MetricSnapshot.from_mapping(item["baseline"]),
            reason=str(item["reason"]),
            related_memory_ids=[str(v) for v in item.get("related_memory_ids", [])],
            evaluation_due_after_events=int(item.get("evaluation_due_after_events", 5)),
        )
        if item["events_since_issue"] < record.evaluation_due_after_events:
            remaining.append(item)
            continue
        text = _extract_text(event_data).lower()
        acknowledged = any(
            token in text
            for token in (
                "objective",
                "known facts",
                "next action",
                "blocker",
                "replan",
            )
        )
        followed = (
            acknowledged
            or post.repeated_tool_calls < record.baseline.repeated_tool_calls
        )
        compliance = ComplianceResult(
            acknowledged=acknowledged,
            followed=followed,
            evidence=["structured recovery response"] if acknowledged else [],
            confidence=0.75 if acknowledged else 0.55,
        )
        append_intervention_outcome(
            outcome_path, record=record, post=post, compliance=compliance
        )
    state.active_interventions = remaining


def process_intervention_event(
    event_data: dict[str, Any],
    state: HookSessionState,
    config: HookConfig,
) -> tuple[list[WasteAlert], HookSessionState, dict[str, Any]]:
    """Process a Phase 3 hook event and return alerts plus Claude Code output."""
    event_name = str(
        event_data.get("hook_event_name", event_data.get("event", "PostToolUse"))
    )
    alerts: list[WasteAlert] = []
    output: dict[str, Any] = {}
    _evaluate_active_interventions(event_data, state, config)

    root = resolve_project_root(event_data)
    if event_name == "UserPromptSubmit" and config.pre_send_check_enabled:
        alert, status = check_pre_send_pattern(event_data, state, config)
        outcome_path = config.outcome_store or str(
            root / ".ter" / "intervention-outcomes.jsonl"
        )
        if status in {"acknowledged", "overridden"}:
            pending_check = dict(state.pre_send_pending)
            record_outcome(
                outcome_path,
                session_id=state.session_id,
                intervention_type="pre_send_check",
                outcome=status,
                details=pending_check,
            )
            state.pre_send_pending = {}
        elif status == "no_match":
            record_outcome(
                outcome_path,
                session_id=state.session_id,
                intervention_type="pre_send_check",
                outcome="no_match",
            )
        elif alert is not None:
            record_outcome(
                outcome_path,
                session_id=state.session_id,
                intervention_type="pre_send_check",
                outcome="fired",
                details=alert.details,
            )
            if config.policy_mode in {"suggest", "warn", "block"}:
                alerts.append(alert)
                output["additionalContext"] = alert.message
                output["systemMessage"] = "TER: pre-send duplicate/pattern check"
            if config.policy_mode == "block":
                output["hookSpecificOutput"] = {
                    "hookEventName": "UserPromptSubmit",
                    "decision": "block",
                    "reason": alert.message,
                }
                return alerts, state, output

    pending = consume_pending_intervention(
        pending_intervention_path(root, state.session_id)
    )
    if pending and event_name in {"UserPromptSubmit", "PreToolUse", "Stop"}:
        record, decision = pending
        memory, _ = (
            build_memory_guidance(
                event_data,
                index_path=config.memory_index,
                limit=config.memory_limit,
                minimum_score=config.memory_minimum_score,
            )
            if config.enable_project_memory
            else ("", [])
        )
        instruction = build_recovery_instruction(decision, memory)
        if config.policy_mode in {"suggest", "warn", "block"}:
            output["additionalContext"] = instruction
            output["systemMessage"] = f"TER: {decision.action.value.replace('_', ' ')}"
        state.active_interventions.append(
            {
                **record.__dict__,
                "baseline": record.baseline.__dict__,
                "events_since_issue": 0,
            }
        )
        if (
            config.policy_mode == "block"
            and decision.action == InterventionAction.REPLAN
            and event_name == "PreToolUse"
        ):
            output["hookSpecificOutput"] = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": instruction,
            }

    metrics = _metric_payload(event_data)
    if metrics:
        snapshot = MetricSnapshot.from_mapping(metrics)
        policy = _policy_from_state(state)
        decision = evaluate_policy(
            snapshot,
            policy,
            PolicyConfig(
                ter_drop_warning=config.ter_drop_warning,
                ter_drop_replan=config.ter_drop_replan,
                waste_ratio_warning=config.waste_ratio_warning,
                waste_ratio_replan=config.waste_ratio_replan,
                degraded_windows_required=config.degraded_windows_required,
                refresh_cooldown_seconds=config.refresh_cooldown_seconds,
                replan_cooldown_seconds=config.replan_cooldown_seconds,
            ),
        )
        _save_policy_state(policy, state)
        if decision.action != InterventionAction.NONE:
            record = new_intervention_record(
                state.session_id, decision, snapshot, state.retrieved_memory_ids
            )
            write_pending_intervention(
                pending_intervention_path(root, state.session_id), record, decision
            )
            if config.policy_mode in {"suggest", "warn", "block"}:
                instruction = build_recovery_instruction(decision)
                output["additionalContext"] = "\n\n".join(
                    filter(None, [output.get("additionalContext", ""), instruction])
                )
                output["systemMessage"] = (
                    f"TER: {decision.action.value.replace('_', ' ')} triggered"
                )

    if event_name in {"SessionStart", "UserPromptSubmit"}:
        contexts: list[str] = []
        if event_name == "SessionStart":
            hint, metadata = build_budget_hint(_extract_text(event_data))
            state.budget_hints_issued += 1
            contexts.append(hint)
            state.last_budget_hint = metadata
        if config.enable_project_memory:
            guidance, matches = build_memory_guidance(
                event_data,
                index_path=config.memory_index,
                limit=config.memory_limit,
                minimum_score=config.memory_minimum_score,
            )
            if guidance:
                contexts.append(guidance)
                state.memory_guidance_count += 1
                state.retrieved_memory_ids = [
                    f"{m['path']}:{m.get('start_line', 0)}" for m in matches
                ][-20:]
        if contexts:
            output["additionalContext"] = "\n\n".join(
                filter(None, [output.get("additionalContext", ""), *contexts])
            )
            output.setdefault("systemMessage", "TER: adaptive guidance active")
    elif event_name == "PreToolUse":
        alert = check_pre_tool_duplicate(
            event_data, state, threshold=config.min_duplicate_calls
        )
        if alert:
            alerts.append(alert)
            output["hookSpecificOutput"] = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": alert.message,
            }
            output["systemMessage"] = "TER: duplicate tool call prevented"
    elif event_name in {"PermissionRequest", "PostToolUseFailure"}:
        alert = check_permission_loop(
            event_data, state, threshold=config.min_denied_calls
        )
        if alert:
            alerts.append(alert)
    elif event_name in {"Stop", "AssistantMessage", "assistant_message"}:
        alert = check_reasoning_loop(
            _extract_text(event_data),
            state,
            threshold=config.reasoning_similarity_threshold,
            min_consecutive=config.min_reasoning_loops,
        )
        if alert:
            alerts.append(alert)
    return alerts, state, output
