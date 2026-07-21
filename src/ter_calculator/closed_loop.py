"""Closed-loop project memory, session lessons, and trend analysis.

The module stays stdlib-only so it can run inside latency-sensitive Claude Code
hooks.  It connects live events to repository memory, records durable lessons,
and aggregates recurring patterns across sessions.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .repository_memory import DEFAULT_INDEX, search_index


@dataclass(frozen=True)
class SessionLesson:
    timestamp: str
    session_id: str
    repository: str
    pattern_type: str
    severity: str
    summary: str
    details: dict[str, Any]
    outcome: str = "observed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_root(event_data: dict[str, Any]) -> Path:
    for key in ("cwd", "project_dir", "repository", "repo_root"):
        value = event_data.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).expanduser().resolve()
    return Path.cwd().resolve()


def build_memory_guidance(
    event_data: dict[str, Any],
    *,
    index_path: str | Path | None = None,
    limit: int = 4,
    minimum_score: float = 0.18,
) -> tuple[str, list[dict[str, Any]]]:
    """Retrieve project-specific context for a prompt or impending tool call."""
    query = _event_query(event_data)
    if not query:
        return "", []
    root = resolve_project_root(event_data)
    path = Path(index_path) if index_path else root / DEFAULT_INDEX
    if not path.exists():
        return "", []
    try:
        result = search_index(path, query, limit=limit, minimum_score=minimum_score)
    except (OSError, ValueError, json.JSONDecodeError):
        return "", []
    matches = result.get("matches", [])
    if not matches:
        return "", []
    lines = ["[TER Project Memory] Review before acting:"]
    for match in matches:
        location = str(match["path"])
        if match.get("start_line"):
            location += f":{match['start_line']}-{match['end_line']}"
        first_line = re.sub(r"\s+", " ", match.get("excerpt", "")).strip()[:180]
        lines.append(f"- {location} ({match['score']:.2f}): {first_line}")
    for flag in result.get("risk_flags", [])[:3]:
        if flag["type"] in {"duplicate_pattern", "semantic_duplicate_pattern"}:
            lines.append(
                f"- Risk: similar implementation already appears in {flag['path']}; reuse or consolidate it."
            )
        elif flag["type"] == "prior_defect_or_fix":
            lines.append(
                f"- Risk: {flag['path']} contains a related defect/fix history; inspect it before changing behavior."
            )
    return "\n".join(lines), matches


def _event_query(event_data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("prompt", "message", "text", "assistant_message"):
        value = event_data.get(key)
        if isinstance(value, str):
            chunks.append(value)
    tool_input = event_data.get("tool_input")
    if isinstance(tool_input, dict):
        for key in (
            "command",
            "file_path",
            "query",
            "pattern",
            "content",
            "new_string",
        ):
            value = tool_input.get(key)
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks).strip()[:8000]


def append_lessons(
    path: str | Path,
    *,
    session_id: str,
    repository: str,
    alerts: list[Any],
    outcome: str = "observed",
) -> int:
    """Append deduplicated alert-derived lessons to a JSONL store."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = _recent_keys(destination)
    rows: list[str] = []
    for alert in alerts:
        key = (session_id, alert.pattern_type, alert.message)
        if key in existing:
            continue
        lesson = SessionLesson(
            timestamp=_utc_now(),
            session_id=session_id,
            repository=repository,
            pattern_type=alert.pattern_type,
            severity=alert.severity,
            summary=alert.message,
            details=dict(alert.details),
            outcome=outcome,
        )
        rows.append(json.dumps(asdict(lesson), sort_keys=True))
        existing.add(key)
    if rows:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
    return len(rows)


def _recent_keys(path: Path, limit: int = 1000) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return set()
    keys: set[tuple[str, str, str]] = set()
    for line in lines:
        try:
            row = json.loads(line)
            keys.add(
                (str(row["session_id"]), str(row["pattern_type"]), str(row["summary"]))
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return keys


def record_outcome(
    path: str | Path,
    *,
    session_id: str,
    intervention_type: str,
    outcome: str,
    details: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _utc_now(),
        "session_id": session_id,
        "intervention_type": intervention_type,
        "outcome": outcome,
        "details": details or {},
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def analyze_trends(
    path: str | Path,
    *,
    minimum_occurrences: int = 2,
    outcome_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate recurring alert patterns and repository scenarios."""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            except json.JSONDecodeError:
                continue
    pattern_counts = Counter(str(row.get("pattern_type", "unknown")) for row in rows)
    repository_counts = Counter(str(row.get("repository", "unknown")) for row in rows)
    scenarios = [
        {
            "pattern_type": pattern,
            "occurrences": count,
            "message": f"Watch out for {pattern.replace('_', ' ')}; observed {count} times across recorded sessions.",
        }
        for pattern, count in pattern_counts.most_common()
        if count >= minimum_occurrences
    ]
    outcomes: list[dict[str, Any]] = []
    if outcome_path and Path(outcome_path).exists():
        for line in Path(outcome_path).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict) and row.get("effect"):
                    outcomes.append(row)
            except json.JSONDecodeError:
                continue
    effectiveness: dict[str, dict[str, Any]] = {}
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in outcomes:
        by_type.setdefault(str(row.get("intervention_type", "unknown")), []).append(row)
    for kind, group in by_type.items():
        issued = len(group)
        followed = sum(bool(row.get("followed")) for row in group)
        improved = sum(row.get("effect") == "improved" for row in group)
        overrides = sum(
            row.get("effect") in {"ignored", "acknowledged_not_followed"}
            for row in group
        )
        ter_deltas = [float(row.get("deltas", {}).get("ter", 0.0)) for row in group]
        waste_deltas = [
            float(row.get("deltas", {}).get("waste_ratio", 0.0)) for row in group
        ]
        effectiveness[kind] = {
            "issued": issued,
            "compliance_rate": followed / issued if issued else 0.0,
            "improvement_rate": improved / issued if issued else 0.0,
            "override_rate": overrides / issued if issued else 0.0,
            "mean_ter_delta": sum(ter_deltas) / issued if issued else 0.0,
            "mean_waste_delta": sum(waste_deltas) / issued if issued else 0.0,
        }
    return {
        "lesson_count": len(rows),
        "pattern_counts": dict(pattern_counts),
        "repository_counts": dict(repository_counts),
        "scenarios": scenarios,
        "outcome_count": len(outcomes),
        "intervention_effectiveness": effectiveness,
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
