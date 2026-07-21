"""Stable identity and provenance helpers for JSONL session blocks."""

from __future__ import annotations

import hashlib
import json
from typing import Any

_VOLATILE_KEYS = {
    "requestId",
    "request_id",
    "timestamp",
    "uuid",
    "parentUuid",
    "parent_uuid",
}


def normalize_json(value: Any) -> Any:
    """Return a deterministic JSON-compatible value with volatile metadata removed."""
    if isinstance(value, dict):
        return {
            str(key): normalize_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_json(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def content_block_fingerprint(role: str, block: Any) -> str:
    """Create a stable SHA-256 fingerprint for one content block."""
    payload = {
        "role": role.strip().lower(),
        "block": normalize_json(block),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def entry_identity(entry: dict[str, Any], source_line: int) -> str:
    """Return a deterministic identity for an entry, including a line fallback."""
    request_id = entry.get("requestId")
    role = entry.get("message", {}).get("role", entry.get("type", ""))
    if request_id:
        return f"request:{request_id}:{role}"
    for key in ("messageId", "message_id"):
        value = entry.get(key)
        if value:
            return f"{key}:{value}:{role}"
    # UUID-only records historically represented independent messages. Keep
    # them distinct unless a stronger sibling identifier is available.
    return f"line:{source_line}:{content_block_fingerprint(role, entry)}"
