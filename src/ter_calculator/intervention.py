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
from collections import Counter
from typing import Any

from .adaptive_budget import recommend_budget
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

    if event_name == "SessionStart":
        hint, metadata = build_budget_hint(_extract_text(event_data))
        state.budget_hints_issued += 1
        output = {
            "additionalContext": hint,
            "systemMessage": "TER: adaptive budget hint active",
        }
        state.last_budget_hint = metadata
    elif event_name == "PreToolUse":
        alert = check_pre_tool_duplicate(
            event_data, state, threshold=config.min_duplicate_calls
        )
        if alert:
            alerts.append(alert)
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": alert.message,
                },
                "systemMessage": "TER: duplicate tool call prevented",
            }
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
