"""The golden corpus written as OpenTelemetry GenAI span exports (ADR 0008).

A test helper, not a TER feature: it turns a session TER read from a Claude
Code transcript into the OTLP JSON an instrumented agent would have exported
for the same work, so the OpenTelemetry source can be checked against the
transcript reader on the same sessions (the document's harness-neutral test:
one session read both ways gives the same Record, up to coverage).

Each model turn (an event carrying usage, with the generated events after it
that carry none) becomes one ``chat`` span; each tool result one
``execute_tool`` span. Cached tokens go inside ``gen_ai.usage.input_tokens``,
as the conventions count them. ``content=False`` leaves out every opt-in
content attribute, as an agent with content capture off exports.

Regenerate the committed fixtures with ``python tests/golden/otel_export.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ter.domain.events import Event, EventKind, SessionTrace

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "otel"
CONVENTION = "1.45.0"
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _id(*parts: object, size: int = 16) -> str:
    raw = "\0".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:size]


def _attr(key: str, value: object) -> dict[str, Any]:
    if isinstance(value, bool):
        wrapped: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        wrapped = {"intValue": str(value)}
    else:
        wrapped = {"stringValue": str(value)}
    return {"key": key, "value": wrapped}


def _nanos(at: datetime) -> str:
    return str(int(at.timestamp() * 1_000_000) * 1000)


def _times(events: tuple[Event, ...]) -> list[datetime]:
    """Each event's time; an undated event takes the one before it plus 1 ms."""
    out: list[datetime] = []
    last = _EPOCH
    for event in events:
        at = event.timestamp
        if at is None:
            at = last + timedelta(milliseconds=1)
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        if at <= last and out:
            at = last + timedelta(microseconds=1)
        out.append(at)
        last = at
    return out


def otlp_export(
    trace: SessionTrace, *, convention: str | None = CONVENTION, content: bool = True
) -> dict[str, Any]:
    """The OTLP JSON ``ExportTraceServiceRequest`` for one session."""
    events = trace.events
    times = _times(events)
    trace_id = _id("trace", trace.session_id, size=32)
    conversation = trace.session_id
    spans: list[dict[str, Any]] = []
    prompt: str | None = None
    previous = times[0] if times else _EPOCH
    turn: dict[str, Any] | None = None

    def close() -> None:
        nonlocal turn
        if turn is None:
            return
        attrs = turn["attrs"]
        if content:
            if turn["prompt"] is not None:
                messages = [
                    {
                        "role": "user",
                        "parts": [{"type": "text", "content": turn["prompt"]}],
                    }
                ]
            else:
                messages = [{"role": "tool", "parts": [{"type": "tool_call_response"}]}]
            attrs.append(_attr("gen_ai.input.messages", json.dumps(messages)))
            attrs.append(
                _attr(
                    "gen_ai.output.messages",
                    json.dumps([{"role": "assistant", "parts": turn["parts"]}]),
                )
            )
        spans.append(
            {
                "traceId": trace_id,
                "spanId": turn["id"],
                "name": f"chat {turn['model']}",
                "kind": 3,
                "startTimeUnixNano": _nanos(turn["start"]),
                "endTimeUnixNano": _nanos(turn["end"]),
                "attributes": attrs,
            }
        )
        turn = None

    for event, at in zip(events, times):
        if event.kind is EventKind.PROMPT:
            close()
            prompt = event.text
        elif event.usage is not None and event.kind.is_generated:
            close()
            usage = event.usage
            model = usage.model or "unknown"
            attrs = [
                _attr("gen_ai.operation.name", "chat"),
                _attr("gen_ai.provider.name", "anthropic"),
                _attr("gen_ai.request.model", model),
                _attr("gen_ai.response.model", model),
                _attr("gen_ai.conversation.id", conversation),
                _attr(
                    "gen_ai.usage.input_tokens",
                    usage.input_tokens
                    + usage.cache_read_tokens
                    + usage.cache_creation_tokens,
                ),
                _attr("gen_ai.usage.output_tokens", usage.output_tokens),
            ]
            if usage.cache_reported:
                attrs.append(
                    _attr(
                        "gen_ai.usage.cache_read.input_tokens", usage.cache_read_tokens
                    )
                )
                attrs.append(
                    _attr(
                        "gen_ai.usage.cache_creation.input_tokens",
                        usage.cache_creation_tokens,
                    )
                )
            turn = {
                "id": _id("span", event.id),
                "model": model,
                "start": previous,
                "end": at,
                "attrs": attrs,
                "parts": [],
                "prompt": prompt,
            }
            prompt = None
        if turn is not None and event.kind.is_generated:
            turn["end"] = at
            if event.kind is EventKind.REASONING:
                turn["parts"].append({"type": "reasoning", "content": event.text})
            elif event.kind is EventKind.RESPONSE:
                turn["parts"].append({"type": "text", "content": event.text})
            elif event.kind is EventKind.TOOL_REQUESTED and event.tool is not None:
                turn["parts"].append(
                    {
                        "type": "tool_call",
                        "id": event.tool.call_id,
                        "name": event.tool.native_name,
                        "arguments": dict(event.tool.arguments),
                    }
                )
        if event.kind is EventKind.TOOL_COMPLETED and event.tool is not None:
            close()
            attrs = [
                _attr("gen_ai.operation.name", "execute_tool"),
                _attr("gen_ai.tool.name", event.tool.native_name),
                _attr("gen_ai.conversation.id", conversation),
            ]
            if event.tool.call_id:
                attrs.append(_attr("gen_ai.tool.call.id", event.tool.call_id))
            if content:
                attrs.append(_attr("gen_ai.tool.call.result", event.text))
            spans.append(
                {
                    "traceId": trace_id,
                    "spanId": _id("span", event.id),
                    "name": f"execute_tool {event.tool.native_name}",
                    "kind": 1,
                    "startTimeUnixNano": _nanos(previous),
                    "endTimeUnixNano": _nanos(at),
                    "attributes": attrs,
                }
            )
        previous = at
    close()
    scope: dict[str, Any] = {"scope": {"name": "ter.tests.otel_export"}, "spans": spans}
    if convention is not None:
        scope["schemaUrl"] = f"https://opentelemetry.io/schemas/{convention}"
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attr("service.name", "golden-agent")]},
                "scopeSpans": [scope],
            }
        ]
    }


def write_fixtures() -> list[Path]:
    """Write the committed fixtures: every golden session with content
    captured, and lean_mix with content capture off."""
    from tests.golden.corpus import CORPUS

    from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource

    FIXTURES.mkdir(parents=True, exist_ok=True)
    reader = ClaudeCodeJsonlSource()
    written: list[Path] = []
    for name, path in CORPUS.items():
        trace = reader.read(path)
        target = FIXTURES / f"{name}.json"
        target.write_text(
            json.dumps(otlp_export(trace), indent=1) + "\n", encoding="utf-8"
        )
        written.append(target)
    lean = reader.read(CORPUS["lean_mix"])
    target = FIXTURES / "lean_mix.no-content.json"
    target.write_text(
        json.dumps(otlp_export(lean, content=False), indent=1) + "\n", encoding="utf-8"
    )
    written.append(target)
    return written


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    for written in write_fixtures():
        print(written)
