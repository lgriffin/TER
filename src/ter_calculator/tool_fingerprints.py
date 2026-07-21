"""Structured fingerprints for tool-call repetition analysis.

The fingerprint intentionally represents the requested action rather than the
rendered JSON text. Exact fingerprints are strong duplicate evidence; changed
paths, ranges, queries, commands, or arguments are treated as parameter novelty.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any, Mapping

_VOLATILE_KEYS = {
    "request_id",
    "requestId",
    "tool_use_id",
    "toolUseId",
    "id",
    "timestamp",
    "timeout",
    "description",
}
_PATH_KEYS = {"path", "file", "file_path", "filepath", "directory", "cwd"}
_RANGE_KEYS = {
    "start",
    "end",
    "offset",
    "limit",
    "line",
    "line_start",
    "line_end",
    "start_line",
    "end_line",
}
_QUERY_KEYS = {"query", "pattern", "search", "q"}
_COMMAND_KEYS = {"command", "cmd"}
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ToolCallFingerprint:
    """Canonical representation of one tool action."""

    tool_name: str
    normalized_arguments: str
    digest: str
    path: str | None = None
    line_range: tuple[int | None, int | None] | None = None
    query: str | None = None
    command: str | None = None


@dataclass(frozen=True)
class ToolCallComparison:
    """Structured similarity result between two tool calls."""

    exact_duplicate: bool
    same_tool: bool
    parameter_novelty: float
    matching_fields: tuple[str, ...]
    changed_fields: tuple[str, ...]


def _normalize_text(value: str) -> str:
    return _WS_RE.sub(" ", value.strip())


def _normalize_path(value: str) -> str:
    value = value.replace("\\", "/")
    # PurePath avoids filesystem access; posixpath normalizes ./ and ../.
    normalized = posixpath.normpath(str(PurePath(value)).replace("\\", "/"))
    return normalized.rstrip("/") or "."


def _normalize_value(key: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _normalize_value(str(k), v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if str(k) not in _VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_value(key, item) for item in value]
    if isinstance(value, str):
        if key in _PATH_KEYS:
            return _normalize_path(value)
        return _normalize_text(value)
    return value


def normalize_tool_arguments(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return deterministic, JSON-serializable tool arguments."""
    if not arguments:
        return {}
    return {
        str(key): _normalize_value(str(key), value)
        for key, value in sorted(arguments.items(), key=lambda item: str(item[0]))
        if str(key) not in _VOLATILE_KEYS
    }


def _first_value(arguments: Mapping[str, Any], keys: set[str]) -> Any | None:
    for key in keys:
        if key in arguments:
            return arguments[key]
    return None


def build_tool_fingerprint(
    tool_name: str | None,
    arguments: Mapping[str, Any] | None,
) -> ToolCallFingerprint:
    """Build a stable SHA-256 fingerprint for a structured tool call."""
    name = _normalize_text(tool_name or "").casefold()
    normalized = normalize_tool_arguments(arguments)
    serialized = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(f"{name}\0{serialized}".encode()).hexdigest()

    raw_path = _first_value(normalized, _PATH_KEYS)
    path = raw_path if isinstance(raw_path, str) else None

    start = _first_value(
        normalized, {"start", "offset", "line_start", "start_line", "line"}
    )
    end = _first_value(normalized, {"end", "line_end", "end_line"})
    limit = normalized.get("limit")
    start_int = start if isinstance(start, int) else None
    end_int = end if isinstance(end, int) else None
    if end_int is None and start_int is not None and isinstance(limit, int):
        end_int = start_int + max(0, limit - 1)
    line_range = (
        (start_int, end_int) if start_int is not None or end_int is not None else None
    )

    raw_query = _first_value(normalized, _QUERY_KEYS)
    query = raw_query if isinstance(raw_query, str) else None
    raw_command = _first_value(normalized, _COMMAND_KEYS)
    command = raw_command if isinstance(raw_command, str) else None

    return ToolCallFingerprint(
        name, serialized, digest, path, line_range, query, command
    )


def compare_tool_calls(
    first: ToolCallFingerprint,
    second: ToolCallFingerprint,
) -> ToolCallComparison:
    """Compare two fingerprints and quantify changed action parameters."""
    same_tool = first.tool_name == second.tool_name
    fields = ("path", "line_range", "query", "command", "normalized_arguments")
    matching = tuple(
        field for field in fields if getattr(first, field) == getattr(second, field)
    )
    changed = tuple(
        field for field in fields if getattr(first, field) != getattr(second, field)
    )
    relevant_changed = [field for field in changed if field != "normalized_arguments"]
    novelty = 1.0 if not same_tool else min(1.0, len(relevant_changed) / 4.0)
    if (
        same_tool
        and first.normalized_arguments != second.normalized_arguments
        and not relevant_changed
    ):
        novelty = 0.25
    return ToolCallComparison(
        exact_duplicate=same_tool and first.digest == second.digest,
        same_tool=same_tool,
        parameter_novelty=novelty,
        matching_fields=matching,
        changed_fields=changed,
    )
