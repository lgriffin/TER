"""Incremental, dependency-light TER metrics derived from Claude Code transcripts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .hook_monitor import HookSessionState

_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_BLOCK_PHASE = {
    "thinking": "reasoning",
    "tool_use": "tool_use",
    "tool_result": "tool_use",
    "text": "generation",
}


def _tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _cosine(left: str, right: str) -> float:
    a = Counter(_WORD_RE.findall(left.lower()))
    b = Counter(_WORD_RE.findall(right.lower()))
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(token, 0) for token, value in a.items())
    na = math.sqrt(sum(value * value for value in a.values()))
    nb = math.sqrt(sum(value * value for value in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _iter_blocks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    message = entry.get("message", {})
    if not isinstance(message, dict):
        return []
    content = message.get("content", [])
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict)]


def _block_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type", "text"))
    if block_type in {"text", "thinking"}:
        return str(block.get("text", block.get("thinking", "")))
    if block_type == "tool_use":
        return json.dumps(
            {"name": block.get("name", ""), "input": block.get("input", {})},
            sort_keys=True,
        )
    content = block.get("content", "")
    return (
        json.dumps(content, sort_keys=True)
        if isinstance(content, (dict, list))
        else str(content)
    )


def _tool_signature(block: dict[str, Any]) -> str:
    raw = json.dumps(
        {"name": block.get("name", ""), "input": block.get("input", {})}, sort_keys=True
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def derive_transcript_metrics(
    event_data: dict[str, Any], state: HookSessionState
) -> dict[str, float | int] | None:
    """Read only appended transcript bytes and update rolling hook metrics.

    Transcript failures intentionally return no metrics: hook availability is more
    important than policy evaluation for any single event.
    """
    raw_path = event_data.get("transcript_path", event_data.get("transcript"))
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    try:
        size = path.stat().st_size
        if size < state.transcript_offset:
            state.transcript_offset = 0
        start_offset = state.transcript_offset
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(start_offset)
            rows: list[tuple[int, int, str]] = []
            while True:
                row_start = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                rows.append((row_start, handle.tell(), raw))
            end_offset = handle.tell()
    except OSError:
        return None

    previous_tokens = state.transcript_total_tokens
    previous_ter = state.transcript_last_ter
    processed = 0
    safe_offset = start_offset
    for row_start, row_end, raw in rows:
        stripped = raw.strip()
        if not stripped:
            safe_offset = row_end
            continue
        try:
            entry = json.loads(stripped)
        except (json.JSONDecodeError, TypeError, ValueError):
            break
        safe_offset = row_end
        if not isinstance(entry, dict):
            continue
        processed += 1
        message = entry.get("message", {})
        role = str(
            entry.get(
                "type", message.get("role", "") if isinstance(message, dict) else ""
            )
        )
        if role == "user":
            continue
        for block in _iter_blocks(entry):
            block_type = str(block.get("type", "text"))
            phase = _BLOCK_PHASE.get(block_type, "generation")
            text = _block_text(block)
            tokens = _tokens(text)
            if not tokens:
                continue
            waste = False
            if block_type == "tool_use":
                signature = _tool_signature(block)
                prior = state.transcript_tool_signatures.get(signature, 0)
                state.transcript_tool_signatures[signature] = prior + 1
                if prior:
                    waste = True
                    state.transcript_repeated_tool_calls += 1
            elif phase in {"reasoning", "generation"}:
                previous = (
                    state.transcript_recent_text[-1]
                    if state.transcript_recent_text
                    else ""
                )
                similarity = _cosine(previous, text) if previous else 0.0
                if phase == "reasoning" and similarity >= 0.88:
                    waste = True
                    state.reasoning_loop_streak += 1
                elif phase == "reasoning":
                    state.reasoning_loop_streak = 0
                state.transcript_recent_text.append(text[-4000:])
                state.transcript_recent_text = state.transcript_recent_text[-6:]
            state.transcript_total_tokens += tokens
            if waste:
                state.transcript_waste_tokens += tokens
            else:
                state.transcript_aligned_tokens += tokens
    state.transcript_offset = safe_offset if rows else end_offset

    if processed == 0 and state.transcript_total_tokens == 0:
        return None
    total = state.transcript_total_tokens
    ter = state.transcript_aligned_tokens / total if total else 1.0
    waste_ratio = state.transcript_waste_tokens / total if total else 0.0
    growth = total / previous_tokens if previous_tokens else 1.0
    drift = max(0.0, previous_ter - ter)
    state.transcript_last_ter = ter
    return {
        "timestamp": time.time(),
        "ter": round(ter, 4),
        "waste_ratio": round(waste_ratio, 4),
        "context_tokens": total,
        "context_growth_rate": round(growth, 4),
        "drift_score": round(drift, 4),
        "repeated_tool_calls": state.transcript_repeated_tool_calls,
        "reasoning_loop_streak": state.reasoning_loop_streak,
    }
