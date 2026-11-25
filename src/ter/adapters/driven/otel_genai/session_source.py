"""OpenTelemetry GenAI spans as a :class:`~ter.ports.driven.SessionSource`
(Stage U2, ADR 0008, TER-SRC-025 to TER-SRC-027).

Any agent instrumented with the OpenTelemetry GenAI semantic conventions can
export its spans as OTLP JSON: one ``ExportTraceServiceRequest`` object per
file, or one per line as the Collector's file exporter writes them. TER reads
those files by the convention version they name, never through an
OpenTelemetry SDK (ADR 0005).

A file is one session. Its id is the ``gen_ai.conversation.id`` its spans
carry; a file whose spans name none is one trace, whose id stands in. A file
holding several conversations (or, without conversation ids, several traces)
is refused with their ids, as a GARE export of several runs is.

=================================  ============================  =========
Span (``gen_ai.operation.name``)   ter.event kind                actor
=================================  ============================  =========
inference (``chat``,               ``intent.stated`` when its    user
``text_completion``,               input ends with a new user
``generate_content``)              message (content captured)
                                   ``reasoning``, ``response``   assistant
                                   (the turn's usage on the
                                   first), ``tool.requested``
                                   per tool call it output
``execute_tool``                   ``tool.requested`` when no    assistant
                                   inference output named the
                                   call, then ``tool.completed``  tool
``invoke_agent`` under another     ``tool.requested``            assistant
agent span                         (``agent.handoff``), then
                                   ``subagent.completed``        system
``invoke_agent`` at the top,       documented metadata           (none)
``create_agent``
any other span                     counted as unrecognised       (none)
=================================  ============================  =========

Usage. The conventions count cached tokens inside
``gen_ai.usage.input_tokens``; TER's ``input_tokens`` are the uncached ones,
so the cache read and cache write counts are taken out of it. A turn with no
cache attribute is read with ``cache_reported`` false: the conventions record
cache counts only "when available and non-zero", so absence cannot be read
as zero and its cost is an estimate (TER-ANL-041).

Content. Message content is opt-in (``gen_ai.input.messages``,
``gen_ai.output.messages``, ``gen_ai.tool.call.arguments`` and ``.result``).
Without output messages a trace carries the ``no-response-text`` limit; without
input messages the developer's prompts cannot be told from tool results, so
each trace (one request of the application) counts as one intent and the
trace carries the ``intents-from-traces`` limit.

Versions. A span names its convention through the ``schemaUrl`` of its scope
or resource (``https://opentelemetry.io/schemas/<version>``). The adapter
reads the versions in :data:`KNOWN_CONVENTIONS` and refuses any other,
naming it (TER-SRC-026). The GenAI conventions moved to their own repository
in 1.42.0, which publishes no schema URL yet; a file that names no version is
read under the latest known one and carries the ``convention-unstated`` limit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ....domain.events import (
    Actor,
    Event,
    EventKind,
    MetadataRecord,
    Provenance,
    SessionFormatError,
    SessionTrace,
    TokenUsage,
    ToolCall,
    ToolKind,
    UnrecognisedRecord,
    make_event_id,
)
from ...claude_code_tools import tool_kind

__all__ = [
    "CONVENTION_UNSTATED",
    "INTENTS_FROM_TRACES",
    "KNOWN_CONVENTIONS",
    "NO_RESPONSE_TEXT",
    "SCHEMA_URL_PREFIX",
    "OtelGenAiSource",
]

SCHEMA_URL_PREFIX = "https://opentelemetry.io/schemas/"
#: Convention versions this adapter reads: those whose GenAI attributes name
#: the provider ``gen_ai.provider.name`` and the token counts
#: ``gen_ai.usage.input_tokens`` / ``output_tokens``, up to the newest
#: checked (25 Nov 2026). An older version (``gen_ai.system``,
#: ``prompt_tokens``) or a newer one is refused until it is added here with
#: fixtures (TER-SRC-026).
KNOWN_CONVENTIONS: tuple[str, ...] = (
    "1.37.0",
    "1.38.0",
    "1.39.0",
    "1.40.0",
    "1.41.0",
    "1.42.0",
    "1.43.0",
    "1.44.0",
    "1.45.0",
)
NO_RESPONSE_TEXT = "no-response-text"
INTENTS_FROM_TRACES = "intents-from-traces"
CONVENTION_UNSTATED = "convention-unstated"

_INFERENCE = frozenset({"chat", "text_completion", "generate_content"})
_TOOL = "execute_tool"
_AGENT = "invoke_agent"
_METADATA = frozenset({"create_agent"})
#: Cache counts: the registry names the write ``cache_creation``; the GenAI
#: repository's main branch names it ``cache_write``. Both are read.
_CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
_CACHE_WRITE = (
    "gen_ai.usage.cache_creation.input_tokens",
    "gen_ai.usage.cache_write.input_tokens",
)


@dataclass(frozen=True)
class _Span:
    """One span as read: where it is in the file and its decoded attributes."""

    order: int
    trace_id: str
    span_id: str
    parent_id: str
    name: str
    start: datetime | None
    end: datetime | None
    attributes: Mapping[str, Any]
    fingerprint: str

    @property
    def record_id(self) -> str:
        return f"{self.trace_id}:{self.span_id}"

    @property
    def operation(self) -> str:
        value = self.attributes.get("gen_ai.operation.name")
        return value if isinstance(value, str) else ""


@dataclass(frozen=True)
class _Draft:
    at: datetime | None
    order: tuple[int, int]
    kind: EventKind
    actor: Actor
    text: str
    span: _Span
    sub: int
    usage: TokenUsage | None = None
    tool: ToolCall | None = None


@dataclass
class _Read:
    spans: list[_Span] = field(default_factory=list)
    unrecognised: list[UnrecognisedRecord] = field(default_factory=list)
    conventions: set[str] = field(default_factory=set)
    unstated: bool = False


class OtelGenAiSource:
    """Reads one OTLP JSON export of OpenTelemetry GenAI spans."""

    format_name = "otel-genai"

    @staticmethod
    def accepts(ref: str | Path) -> bool:
        """True when ``ref`` is an OTLP JSON span export, reading only its start."""
        path = Path(ref)
        if not path.is_file() or path.suffix not in (".json", ".jsonl"):
            return False
        try:
            with path.open(encoding="utf-8") as handle:
                head = handle.read(4096)
        except (OSError, UnicodeDecodeError):
            return False
        return '"resourceSpans"' in head

    def read(self, ref: str | Path) -> SessionTrace:
        path = Path(ref)
        if not path.exists():
            raise FileNotFoundError(path)
        state = _parse(path)
        unknown = sorted(state.conventions - set(KNOWN_CONVENTIONS))
        if unknown:
            raise SessionFormatError(
                f"{path} names OpenTelemetry convention version(s) "
                f"{', '.join(unknown)}; this adapter reads "
                f"{KNOWN_CONVENTIONS[0]} to {KNOWN_CONVENTIONS[-1]}"
            )
        if not state.spans and not state.unrecognised:
            raise SessionFormatError(f"{path} holds no spans")
        session_id = _session_id(state.spans, path)
        by_id = {span.span_id: span for span in state.spans}
        content = any(
            "gen_ai.input.messages" in s.attributes
            for s in state.spans
            if s.operation in _INFERENCE
        )
        output_text = any(
            "gen_ai.output.messages" in s.attributes
            for s in state.spans
            if s.operation in _INFERENCE
        )
        drafts: list[_Draft] = []
        metadata: list[MetadataRecord] = []
        requested: set[str] = set()
        seen_traces: set[str] = set()
        for span in sorted(state.spans, key=_when):
            op = span.operation
            if not content and op in _INFERENCE and span.trace_id not in seen_traces:
                # Without input messages, each request of the application
                # (a trace) is one intent (INTENTS_FROM_TRACES).
                seen_traces.add(span.trace_id)
                drafts.append(
                    _draft(span, EventKind.PROMPT, Actor.USER, "", 0, start=True)
                )
            if op in _INFERENCE:
                drafts.extend(_inference(span, content, requested))
            elif op == _TOOL:
                drafts.extend(_tool(span, requested))
            elif op == _AGENT and _under_agent(span, by_id):
                drafts.extend(_handoff(span))
            elif op == _AGENT or op in _METADATA:
                metadata.append(MetadataRecord(span.order, f"span:{op}"))
            else:
                state.unrecognised.append(
                    UnrecognisedRecord(span.order, f"span:{op or 'not-gen_ai'}")
                )

        events: list[Event] = []
        for draft in sorted(drafts, key=_draft_key):
            events.append(
                Event(
                    id=make_event_id(
                        "otel",
                        session_id,
                        draft.span.record_id,
                        draft.kind.value,
                        draft.sub,
                    ),
                    session_id=session_id,
                    sequence=len(events),
                    kind=draft.kind,
                    actor=draft.actor,
                    text=draft.text,
                    provenance=Provenance(
                        source=path.name,
                        record_id=draft.span.record_id,
                        lines=(draft.span.order,),
                        fingerprint=draft.span.fingerprint,
                    ),
                    timestamp=draft.at,
                    tool=draft.tool,
                    usage=draft.usage,
                    parent_id=events[-1].id if events else None,
                )
            )
        limits: list[str] = []
        if not output_text:
            limits.append(NO_RESPONSE_TEXT)
        if not content:
            limits.append(INTENTS_FROM_TRACES)
        if state.unstated:
            limits.append(CONVENTION_UNSTATED)
        return SessionTrace(
            session_id=session_id,
            source_format=self.format_name,
            events=tuple(events),
            unrecognised=tuple(state.unrecognised),
            metadata=tuple(metadata),
            usage_limits=tuple(limits),
        )


# -- the file -----------------------------------------------------------------


def _parse(path: Path) -> _Read:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SessionFormatError(f"{path} is not UTF-8") from exc
    if not text.strip():
        raise SessionFormatError(f"{path} is empty")
    try:
        documents: list[tuple[int, Any]] = [(1, json.loads(text))]
    except ValueError:
        documents = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                documents.append((number, json.loads(line)))
            except ValueError:
                documents.append((number, None))
    state = _Read()
    if not any(
        isinstance(doc, Mapping) and isinstance(doc.get("resourceSpans"), list)
        for _, doc in documents
    ):
        raise SessionFormatError(f"{path} is not an OTLP JSON span export")
    order = 0
    for number, doc in documents:
        if not (
            isinstance(doc, Mapping) and isinstance(doc.get("resourceSpans"), list)
        ):
            state.unrecognised.append(UnrecognisedRecord(number, "otlp:invalid"))
            continue
        for resource in doc["resourceSpans"]:
            if not isinstance(resource, Mapping):
                state.unrecognised.append(UnrecognisedRecord(number, "otlp:invalid"))
                continue
            outer = _version(resource.get("schemaUrl"))
            for scope in _list(resource.get("scopeSpans")):
                if not isinstance(scope, Mapping):
                    state.unrecognised.append(
                        UnrecognisedRecord(number, "otlp:invalid")
                    )
                    continue
                version = _version(scope.get("schemaUrl")) or outer
                for raw in _list(scope.get("spans")):
                    order += 1
                    span = _span(raw, order)
                    if span is None:
                        state.unrecognised.append(
                            UnrecognisedRecord(order, "span:invalid")
                        )
                        continue
                    if span.operation:
                        if version is None:
                            state.unstated = True
                        else:
                            state.conventions.add(version)
                    state.spans.append(span)
    return state


def _version(url: object) -> str | None:
    """The convention version a schema URL names; an unknown URL is its own
    version, so it is refused by name."""
    if not isinstance(url, str) or not url:
        return None
    if url.startswith(SCHEMA_URL_PREFIX):
        return url[len(SCHEMA_URL_PREFIX) :].strip("/") or url
    return url


def _span(raw: object, order: int) -> _Span | None:
    if not isinstance(raw, Mapping):
        return None
    trace_id, span_id = raw.get("traceId"), raw.get("spanId")
    if not (
        isinstance(trace_id, str) and trace_id and isinstance(span_id, str) and span_id
    ):
        return None
    parent = raw.get("parentSpanId")
    name = raw.get("name")
    return _Span(
        order=order,
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent if isinstance(parent, str) else "",
        name=name if isinstance(name, str) else "",
        start=_nanos(raw.get("startTimeUnixNano")),
        end=_nanos(raw.get("endTimeUnixNano")),
        attributes=_attributes(raw.get("attributes")),
        fingerprint=_fingerprint(raw),
    )


def _attributes(raw: object) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in _list(raw):
        if isinstance(item, Mapping) and isinstance(item.get("key"), str):
            out[item["key"]] = _value(item.get("value"))
    return out


def _value(raw: object) -> Any:
    """An OTLP ``AnyValue`` as plain Python (ints arrive as strings in JSON)."""
    if not isinstance(raw, Mapping):
        return None
    if "stringValue" in raw:
        return raw["stringValue"]
    if "intValue" in raw:
        try:
            return int(raw["intValue"])
        except (TypeError, ValueError):
            return None
    if "doubleValue" in raw:
        value = raw["doubleValue"]
        return (
            value
            if isinstance(value, int | float) and not isinstance(value, bool)
            else None
        )
    if "boolValue" in raw:
        return raw["boolValue"] if isinstance(raw["boolValue"], bool) else None
    if "arrayValue" in raw:
        inner = raw["arrayValue"]
        values = inner.get("values") if isinstance(inner, Mapping) else None
        return [_value(v) for v in _list(values)]
    if "kvlistValue" in raw:
        inner = raw["kvlistValue"]
        values = inner.get("values") if isinstance(inner, Mapping) else None
        return _attributes(values)
    return None


def _session_id(spans: list[_Span], path: Path) -> str:
    conversations = sorted(
        {
            str(s.attributes["gen_ai.conversation.id"])
            for s in spans
            if isinstance(s.attributes.get("gen_ai.conversation.id"), str)
            and s.attributes["gen_ai.conversation.id"]
        }
    )
    if len(conversations) == 1:
        return conversations[0]
    if len(conversations) > 1:
        raise SessionFormatError(
            f"{path} holds {len(conversations)} conversations "
            f"({', '.join(conversations)}); export one conversation per file"
        )
    traces = sorted({s.trace_id for s in spans if s.operation})
    if len(traces) == 1:
        return traces[0]
    if not traces:
        raise SessionFormatError(f"{path} holds no GenAI spans")
    raise SessionFormatError(
        f"{path} holds {len(traces)} traces and no gen_ai.conversation.id to "
        "join them; export one trace per file"
    )


# -- spans to events ------------------------------------------------------------


def _draft(
    span: _Span,
    kind: EventKind,
    actor: Actor,
    text: str,
    sub: int,
    *,
    start: bool = False,
    usage: TokenUsage | None = None,
    tool: ToolCall | None = None,
) -> _Draft:
    at = span.start if start else (span.end or span.start)
    return _Draft(
        at=at,
        order=(span.order, sub),
        kind=kind,
        actor=actor,
        text=text,
        span=span,
        sub=sub,
        usage=usage,
        tool=tool,
    )


def _inference(span: _Span, content: bool, requested: set[str]) -> Iterator[_Draft]:
    attrs = span.attributes
    if content:
        prompt = _new_user_text(attrs.get("gen_ai.input.messages"))
        if prompt is not None:
            yield _draft(span, EventKind.PROMPT, Actor.USER, prompt, 0, start=True)
    reasoning, text, calls = _output(attrs.get("gen_ai.output.messages"))
    usage = _usage(attrs)
    generated: list[tuple[EventKind, str, ToolCall | None]] = [
        (EventKind.REASONING, r, None) for r in reasoning
    ]
    if text or not (calls or reasoning):
        # A turn that output only tool calls or reasoning has no response; one
        # that output nothing captured keeps an empty response to carry its usage.
        generated.append((EventKind.RESPONSE, "\n".join(text), None))
    for call in calls:
        if call.call_id:
            requested.add(call.call_id)
        generated.append((EventKind.TOOL_REQUESTED, _arguments(call.arguments), call))
    for index, (kind, body, tool) in enumerate(generated, 1):
        yield _draft(
            span,
            kind,
            Actor.ASSISTANT,
            body,
            index,
            usage=usage if index == 1 else None,
            tool=tool,
        )


def _tool(span: _Span, requested: set[str]) -> Iterator[_Draft]:
    attrs = span.attributes
    name = attrs.get("gen_ai.tool.name")
    native = name if isinstance(name, str) and name else span.name
    call_id = attrs.get("gen_ai.tool.call.id")
    call_id = call_id if isinstance(call_id, str) and call_id else None
    kind = (
        ToolKind.AGENT_HANDOFF if "gen_ai.transfer.mode" in attrs else tool_kind(native)
    )
    arguments = attrs.get("gen_ai.tool.call.arguments")
    args = _mapping(arguments)
    call = ToolCall(native_name=native, kind=kind, call_id=call_id, arguments=args)
    if call_id is None or call_id not in requested:
        # No inference output named the call (content not captured): the
        # span is the only record that the agent asked for it.
        yield _draft(
            span,
            EventKind.TOOL_REQUESTED,
            Actor.ASSISTANT,
            _arguments(args),
            0,
            start=True,
            tool=call,
        )
    result = attrs.get("gen_ai.tool.call.result")
    yield _draft(
        span,
        EventKind.TOOL_COMPLETED,
        Actor.TOOL,
        _as_text(result),
        1,
        tool=ToolCall(native_name=native, kind=kind, call_id=call_id),
    )


def _handoff(span: _Span) -> Iterator[_Draft]:
    name = span.attributes.get("gen_ai.agent.name")
    agent = name if isinstance(name, str) and name else span.name
    call = ToolCall(native_name="invoke_agent", kind=ToolKind.AGENT_HANDOFF)
    yield _draft(
        span, EventKind.TOOL_REQUESTED, Actor.ASSISTANT, agent, 0, start=True, tool=call
    )
    yield _draft(span, EventKind.SUBAGENT_COMPLETED, Actor.SYSTEM, agent, 1)


def _under_agent(span: _Span, by_id: Mapping[str, _Span]) -> bool:
    """True when an ancestor of ``span`` is an agent span (a nested agent)."""
    seen: set[str] = set()
    parent = by_id.get(span.parent_id)
    while parent is not None and parent.span_id not in seen:
        if parent.operation == _AGENT:
            return True
        seen.add(parent.span_id)
        parent = by_id.get(parent.parent_id)
    return False


def _usage(attrs: Mapping[str, Any]) -> TokenUsage | None:
    total_in = _int(attrs.get("gen_ai.usage.input_tokens"))
    output = _int(attrs.get("gen_ai.usage.output_tokens"))
    read = _int(attrs.get(_CACHE_READ))
    write_key = next((k for k in _CACHE_WRITE if k in attrs), None)
    write = _int(attrs.get(write_key)) if write_key else None
    if total_in is None and output is None:
        return None
    cached = (read or 0) + (write or 0)
    model = attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model")
    return TokenUsage(
        input_tokens=max(0, (total_in or 0) - cached),
        output_tokens=output or 0,
        cache_creation_tokens=write or 0,
        cache_read_tokens=read or 0,
        model=model if isinstance(model, str) and model else None,
        cache_reported=read is not None or write is not None,
    )


def _messages(value: object) -> list[Mapping[str, Any]]:
    """Messages as the conventions record them: a JSON string or a list."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return [m for m in _list(value) if isinstance(m, Mapping)]


def _parts(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [p for p in _list(message.get("parts")) if isinstance(p, Mapping)]


def _new_user_text(value: object) -> str | None:
    """The text of the last input message when it is the user's: a prompt the
    developer added, not a tool result returned to the model."""
    messages = _messages(value)
    if not messages or messages[-1].get("role") != "user":
        return None
    texts = [
        str(p.get("content"))
        for p in _parts(messages[-1])
        if p.get("type") == "text" and isinstance(p.get("content"), str)
    ]
    return "\n".join(texts) if texts else None


def _output(value: object) -> tuple[list[str], list[str], list[ToolCall]]:
    reasoning: list[str] = []
    text: list[str] = []
    calls: list[ToolCall] = []
    for message in _messages(value):
        for part in _parts(message):
            kind = part.get("type")
            if kind == "text" and isinstance(part.get("content"), str):
                text.append(part["content"])
            elif kind == "reasoning" and isinstance(part.get("content"), str):
                reasoning.append(part["content"])
            elif kind == "tool_call" and isinstance(part.get("name"), str):
                call_id = part.get("id")
                calls.append(
                    ToolCall(
                        native_name=part["name"],
                        kind=tool_kind(part["name"]),
                        call_id=call_id
                        if isinstance(call_id, str) and call_id
                        else None,
                        arguments=_mapping(part.get("arguments")),
                    )
                )
    return reasoning, text, calls


# -- values ---------------------------------------------------------------------


def _when(span: _Span) -> tuple[float, int]:
    at = span.start or span.end
    return (at.timestamp() if at else 0.0, span.order)


def _draft_key(draft: _Draft) -> tuple[float, tuple[int, int]]:
    return (draft.at.timestamp() if draft.at else 0.0, draft.order)


def _nanos(value: object) -> datetime | None:
    try:
        nanos = (
            int(value)
            if isinstance(value, str | int) and not isinstance(value, bool)
            else None
        )
    except ValueError:
        return None
    if nanos is None or nanos <= 0:
        return None
    try:
        return datetime.fromtimestamp(nanos / 1e9, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _arguments(arguments: Mapping[str, object]) -> str:
    return json.dumps(
        dict(arguments), sort_keys=True, separators=(",", ":"), default=str
    )


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _int(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _fingerprint(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
