"""OpenTelemetry GenAI adapter: OTLP JSON span exports to ``ter.event`` traces
(Stage U2, ADR 0008)."""

from __future__ import annotations

from .session_source import (
    CONVENTION_UNSTATED,
    INTENTS_FROM_TRACES,
    KNOWN_CONVENTIONS,
    NO_RESPONSE_TEXT,
    SCHEMA_URL_PREFIX,
    OtelGenAiSource,
)

__all__ = [
    "CONVENTION_UNSTATED",
    "INTENTS_FROM_TRACES",
    "KNOWN_CONVENTIONS",
    "NO_RESPONSE_TEXT",
    "SCHEMA_URL_PREFIX",
    "OtelGenAiSource",
]
