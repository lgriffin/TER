"""Explainable Phase 2 static detectors for Claude Code sessions."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .loader import load_session


@dataclass(frozen=True)
class SignalFinding:
    signal_type: str
    title: str
    severity: str
    confidence: float
    summary: str
    recommendation: str
    evidence: list[dict[str, Any]]
    occurrences: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError:
        return repr(value)


def _short(value: str, limit: int = 220) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _severity(count: int, medium: int = 3, high: int = 6) -> str:
    return "high" if count >= high else "medium" if count >= medium else "low"


def analyze_session_signals(path: str | Path) -> dict[str, Any]:
    session = load_session(path)
    tool_calls: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    reads: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for message_index, message in enumerate(session.messages):
        for block_index, block in enumerate(message.content_blocks):
            evidence = {
                "message_index": message_index,
                "message_uuid": message.uuid,
                "block_index": block_index,
                "source_lines": block.source_lines or ([block.source_line] if block.source_line else []),
            }
            if block.block_type == "tool_use":
                tool_name = block.tool_name or "unknown"
                tool_input = block.tool_input or {}
                fingerprint = hashlib.sha256(
                    f"{tool_name}:{_canonical(tool_input)}".encode("utf-8")
                ).hexdigest()[:16]
                record = {**evidence, "tool_name": tool_name, "tool_input": tool_input, "fingerprint": fingerprint}
                tool_calls.append(record)
                lowered = tool_name.lower()
                if any(term in lowered for term in ("read", "cat", "open", "view")):
                    target = str(tool_input.get("file_path") or tool_input.get("path") or tool_input.get("filename") or "")
                    if target:
                        reads[target].append(record)
            elif block.block_type == "tool_result":
                text = block.text or ""
                lowered = text.lower()
                if any(term in lowered for term in ("error", "failed", "exception", "traceback", "permission denied", "not found")):
                    signature = _short(lowered, 160)
                    failures[signature].append({**evidence, "excerpt": _short(text)})
            elif block.block_type in {"text", "thinking"} and block.text:
                normalized = " ".join(block.text.lower().split())
                if len(normalized) >= 120:
                    texts.append({**evidence, "text": block.text, "fingerprint": hashlib.sha256(normalized.encode()).hexdigest()[:16]})

    findings: list[SignalFinding] = []

    by_call: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in tool_calls:
        by_call[call["fingerprint"]].append(call)
    for occurrences in by_call.values():
        if len(occurrences) >= 3:
            sample = occurrences[0]
            findings.append(SignalFinding(
                "repeated_tool_call", "Repeated identical tool call", _severity(len(occurrences)), min(0.99, 0.72 + len(occurrences) * 0.04),
                f"The same {sample['tool_name']} call was issued {len(occurrences)} times.",
                "Review the first result before repeating the call; cache or summarize stable outputs.",
                occurrences[:8], len(occurrences),
            ))

    for target, occurrences in reads.items():
        if len(occurrences) >= 4:
            findings.append(SignalFinding(
                "repeated_file_read", "Repeated file reads", _severity(len(occurrences), 4, 7), min(0.98, 0.7 + len(occurrences) * 0.035),
                f"{target} was read {len(occurrences)} times in one session.",
                "Keep a concise working summary of the file and reread only after it changes.",
                [{**e, "target": target} for e in occurrences[:8]], len(occurrences),
            ))

    for signature, occurrences in failures.items():
        if len(occurrences) >= 2:
            findings.append(SignalFinding(
                "repeated_failure", "Repeated failure pattern", _severity(len(occurrences), 2, 4), min(0.99, 0.78 + len(occurrences) * 0.05),
                f"A similar failure appeared {len(occurrences)} times: {_short(signature, 110)}",
                "Stop retrying mechanically; summarize the failure, inspect prerequisites, and produce a revised plan.",
                occurrences[:8], len(occurrences),
            ))

    by_text: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for text in texts:
        by_text[text["fingerprint"]].append(text)
    for occurrences in by_text.values():
        if len(occurrences) >= 2:
            findings.append(SignalFinding(
                "repeated_generated_content", "Repeated generated content", _severity(len(occurrences), 2, 4), min(0.97, 0.74 + len(occurrences) * 0.05),
                f"Substantially identical long-form content appeared {len(occurrences)} times.",
                "Reference the prior conclusion and continue from it instead of regenerating it.",
                [{**e, "excerpt": _short(e["text"])} for e in occurrences[:6]], len(occurrences),
            ))

    total_actions = len(tool_calls)
    unique_actions = len(by_call)
    repetition_ratio = 1 - (unique_actions / total_actions) if total_actions else 0.0
    if total_actions >= 12 and repetition_ratio >= 0.45:
        findings.append(SignalFinding(
            "high_activity_low_novelty", "High activity with low tool-call novelty", "high" if repetition_ratio >= 0.65 else "medium", min(0.95, 0.65 + repetition_ratio * 0.4),
            f"{total_actions} tool calls contained only {unique_actions} unique call signatures ({repetition_ratio:.0%} repetition).",
            "Pause and re-plan; identify the missing information or decision before issuing more tools.",
            tool_calls[:8], total_actions - unique_actions,
        ))

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (order[f.severity], -f.confidence, -f.occurrences, f.signal_type))
    counts = Counter(f.severity for f in findings)
    return {
        "version": "2.0.2",
        "session_id": session.session_id,
        "source_file": str(path),
        "finding_count": len(findings),
        "severity_counts": dict(counts),
        "signal_counts": dict(Counter(f.signal_type for f in findings)),
        "findings": [f.to_dict() for f in findings],
    }
