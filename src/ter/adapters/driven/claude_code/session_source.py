"""Claude Code JSONL transcripts as a :class:`~ter.ports.driven.SessionSource`.

Reconstruction reuses the TER 3 loader (``ter_calculator.loader``), so sibling
merging, request-id de-duplication and content fingerprints behave exactly as
they do in TER 3. This adapter adds the translation into neutral events and
an honest count of records it could not map.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ter_calculator.loader import load_session
from ter_calculator.models import ContentBlock, Message
from ter_calculator.models import TokenUsage as LegacyUsage

from ....domain.events import (
    Actor,
    Event,
    EventId,
    EventKind,
    MetadataRecord,
    Provenance,
    SessionFormatError,
    SessionTrace,
    TokenUsage,
    ToolCall,
    UnrecognisedRecord,
)
from ...claude_code_ids import (
    TOOL_KINDS,
    prompt_event_id,
    queued_prompt_text,
    record_event_id,
    stop_event_id,
    subagent_event_id,
    tool_event_id,
)
from ...claude_code_subagents import (
    SUBAGENTS_DIR,
    FinishMarker,
    final_turn,
    parent_finishes,
    subagent_transcripts,
)
from ...claude_code_tools import tool_kind
from ...claude_code_turns import read_records, record_time, stop_records

#: Record types that carry conversation content and are mapped to events.
_CONTENT_TYPES = frozenset({"user", "assistant"})

#: Record types Claude Code writes that carry no agent activity: titles,
#: bookmarks, queue and permission bookkeeping, harness notices (``system``)
#: and context the harness injects (``attachment``). They are recognised as
#: metadata, never as events (TER-SRC-005). Observed in Claude Code 2.1.x on
#: the reference corpus; a type not listed here stays unrecognised.
METADATA_TYPES = frozenset(
    {
        "agent-name",
        "ai-title",
        "artifact-autoreact-ledger",
        "artifact-comment-monitor",
        "atis-latch",
        "attachment",
        "bridge-session",
        "cost-state",
        "custom-title",
        "file-history-delta",
        "file-history-snapshot",
        "fork-context-ref",
        "frame-link",
        "last-prompt",
        "mode",
        "permission-mode",
        "pr-link",
        "queue-operation",
        "summary",
        "system",
    }
)


def record_class(record_type: str) -> str:
    """How the session source treats a record type: ``content`` (mapped to
    events), ``metadata`` (documented, no agent activity) or ``unrecognised``."""
    if record_type in _CONTENT_TYPES:
        return "content"
    if record_type in METADATA_TYPES:
        return "metadata"
    return "unrecognised"


class ClaudeCodeJsonlSource:
    """Reads Claude Code ``.jsonl`` session transcripts.

    Event ids follow the shared rules in :mod:`ter.adapters.claude_code_ids`,
    so the hook adapter derives the same id for the same record (TER-OBS-007):
    tool requests and completions by ``tool_use_id``, everything else by
    record uuid and block index; a finished subagent by its agent id.
    """

    format_name = "claude-code-jsonl"

    def read(self, ref: str | Path) -> SessionTrace:
        path = Path(ref)
        try:
            session = load_session(path)
        except ValueError as exc:  # also UnicodeDecodeError
            raise SessionFormatError(str(exc)) from exc
        scan = _scan(path)
        usage_meta = scan.usage
        queued = list(scan.prompts)
        source = path.name

        events: list[Event] = []
        requests: dict[str, ToolCall] = {}
        keyed: set[EventId] = set()
        # Records by uuid (TER-TST-009): an exact copy of a record already read
        # adds nothing and is skipped; a different record reusing a uuid keeps
        # its events under ids of their own.
        copies: dict[str, list[object]] = {}
        previous: EventId | None = None

        def occurrence(uuid: str, content: object) -> str | None:
            """The uuid to derive ids from, or ``None`` for an exact copy."""
            seen = copies.setdefault(uuid, [])
            if content in seen:
                return None
            seen.append(content)
            return uuid if len(seen) == 1 else f"{uuid}~{len(seen) - 1}"

        def flush_queued(before: int | None) -> None:
            # Queued prompts take their place in file order among the messages.
            nonlocal previous
            while queued and (before is None or queued[0].line < before):
                prompt = queued.pop(0)
                uuid = occurrence(prompt.uuid, ("queued", prompt.text))
                if uuid is None:
                    continue
                event_id = prompt_event_id(session.session_id, uuid)
                events.append(
                    Event(
                        id=event_id,
                        session_id=session.session_id,
                        sequence=len(events),
                        kind=EventKind.PROMPT,
                        actor=Actor.USER,
                        text=prompt.text,
                        provenance=Provenance(
                            source=source,
                            record_id=prompt.uuid,
                            lines=(prompt.line,),
                            block_index=0,
                        ),
                        timestamp=prompt.timestamp,
                        parent_id=previous,
                    )
                )
                previous = event_id

        for message in session.messages:
            flush_queued(min(message.source_lines, default=None))
            record_uuid = occurrence(
                message.uuid,
                (
                    message.role,
                    *(b.content_fingerprint for b in message.content_blocks),
                ),
            )
            if record_uuid is None:
                continue
            model, cache_reported, harness = _meta(message, usage_meta)
            # A message the harness wrote itself (TER-ANL-044) is no model
            # turn: its text is the system's, and it carries no usage.
            usage = None if harness else _usage(message.usage, model, cache_reported)
            for index, block in enumerate(message.content_blocks):
                kind, actor = _classify(message.role, block.block_type)
                if kind is None or actor is None:
                    continue
                if harness and actor is Actor.ASSISTANT:
                    actor = Actor.SYSTEM
                block_index = (
                    block.block_index if block.block_index is not None else index
                )
                tool = _tool(block, requests)
                if (
                    tool is not None
                    and kind is EventKind.TOOL_REQUESTED
                    and tool.call_id
                ):
                    requests[tool.call_id] = tool
                # Sibling lines of one response merge into one message, but
                # each block keeps the uuid of its own line, so two blocks at
                # index 0 of different lines never share an id.
                line_uuid = block.source_uuid or message.uuid
                block_uuid = record_uuid if line_uuid == message.uuid else line_uuid
                event_id = _event_id(
                    session.session_id, block_uuid, block_index, kind, tool, keyed
                )
                events.append(
                    Event(
                        id=event_id,
                        session_id=session.session_id,
                        sequence=len(events),
                        kind=kind,
                        actor=actor,
                        text=_text(block),
                        provenance=Provenance(
                            source=source,
                            record_id=line_uuid,
                            lines=_lines(message, block),
                            block_index=block_index,
                            fingerprint=block.content_fingerprint,
                        ),
                        timestamp=message.timestamp,
                        tool=tool,
                        # Usage belongs to the model turn, so it is attached once
                        # (to the turn's first event) and never double counted.
                        usage=usage,
                        parent_id=previous,
                    )
                )
                usage = None
                previous = event_id
        flush_queued(None)

        return SessionTrace(
            session_id=session.session_id,
            source_format=self.format_name,
            events=_with_subagents(
                path,
                session.session_id,
                source,
                _with_stops(path, session.session_id, source, events),
            ),
            unrecognised=scan.unrecognised,
            metadata=scan.metadata,
        )


def _event_id(
    session_id: str,
    record_uuid: str,
    block_index: int,
    kind: EventKind,
    tool: ToolCall | None,
    keyed: set[EventId],
) -> EventId:
    """The shared id rule for one block (:mod:`ter.adapters.claude_code_ids`).

    A tool block is keyed by its ``tool_use_id``, as the hooks key it. A
    block without one, or one repeating an id already given in this session
    (a copied record), falls back to the record rule so ids stay unique.
    """
    if kind in TOOL_KINDS and tool is not None and tool.call_id:
        event_id = tool_event_id(session_id, tool.call_id, kind)
        if event_id not in keyed:
            keyed.add(event_id)
            return event_id
    event_id = record_event_id(session_id, record_uuid, block_index, kind)
    repeat = 0
    while event_id in keyed:
        # Two lines that reuse a uuid can merge into one message with the
        # same block index twice (TER-TST-009); the later block gets its own.
        repeat += 1
        event_id = record_event_id(
            session_id, f"{record_uuid}~{repeat}", block_index, kind
        )
    keyed.add(event_id)
    return event_id


#: Usage facts the TER 3 loader drops, by source line: the model named on the
#: record and whether its usage block carried any cache field.
_UsageMeta = dict[int, tuple[str | None, bool]]

_CACHE_KEYS = ("cache_creation_input_tokens", "cache_read_input_tokens")


@dataclass(frozen=True)
class _QueuedPrompt:
    line: int
    uuid: str
    text: str
    timestamp: datetime | None


@dataclass(frozen=True)
class _Scan:
    unrecognised: tuple[UnrecognisedRecord, ...]
    metadata: tuple[MetadataRecord, ...]
    usage: _UsageMeta
    prompts: tuple[_QueuedPrompt, ...]


def _scan(path: Path) -> _Scan:
    """Sort every record into content, documented metadata or unrecognised,
    read the usage facts (model, cache fields present) of content records, and
    collect the prompts the developer queued while the agent worked."""
    found: list[UnrecognisedRecord] = []
    metadata: list[MetadataRecord] = []
    meta: _UsageMeta = {}
    prompts: list[_QueuedPrompt] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                found.append(UnrecognisedRecord(line_number, type(record).__name__))
                continue
            record_type = str(record.get("type", ""))
            if record_type in METADATA_TYPES:
                prompt = _queued_prompt(line_number, record)
                if prompt is not None:
                    prompts.append(prompt)
                else:
                    metadata.append(MetadataRecord(line_number, record_type))
                continue
            if record_type not in _CONTENT_TYPES:
                found.append(
                    UnrecognisedRecord(line_number, record_type or "<missing>")
                )
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            usage = message.get("usage")
            model = message.get("model")
            meta[line_number] = (
                model if isinstance(model, str) and model else None,
                isinstance(usage, dict) and any(k in usage for k in _CACHE_KEYS),
            )
    return _Scan(tuple(found), tuple(metadata), meta, tuple(prompts))


def _queued_prompt(line: int, record: dict[str, object]) -> _QueuedPrompt | None:
    """A ``queued_command`` attachment the developer typed, else ``None``.

    The rule (:func:`ter.adapters.claude_code_ids.queued_prompt_text`) is the
    one the hook adapter uses to find a queued prompt's record.
    """
    text = queued_prompt_text(record)
    uuid = record.get("uuid")
    if text is None or not isinstance(uuid, str):
        return None
    return _QueuedPrompt(line, uuid, text, _timestamp(record.get("timestamp")))


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _meta(message: Message, meta: _UsageMeta) -> tuple[str | None, bool, bool]:
    """The model, cache-field presence and harness origin of a message.

    Read over its merged records. A message is the harness's own only when
    every record that names a model names :data:`HARNESS_MODEL`, so a real
    reply merged with a harness line keeps its usage.
    """
    models: list[str] = []
    cache = False
    for line in message.source_lines:
        line_model, line_cache = meta.get(line, (None, False))
        if line_model is not None:
            models.append(line_model)
        cache = cache or line_cache
    real = [m for m in models if m != HARNESS_MODEL]
    model = real[0] if real else (models[0] if models else None)
    return model, cache, bool(models) and not real


def _classify(role: str, block_type: str) -> tuple[EventKind | None, Actor | None]:
    if role == "user":
        if block_type == "text":
            return EventKind.PROMPT, Actor.USER
        if block_type == "tool_result":
            return EventKind.TOOL_COMPLETED, Actor.TOOL
        return None, None
    if role == "assistant":
        if block_type == "thinking":
            return EventKind.REASONING, Actor.ASSISTANT
        if block_type == "text":
            return EventKind.RESPONSE, Actor.ASSISTANT
        if block_type == "tool_use":
            return EventKind.TOOL_REQUESTED, Actor.ASSISTANT
    return None, None


def _text(block: ContentBlock) -> str:
    if block.block_type == "tool_use":
        return json.dumps(block.tool_input or {}, sort_keys=True, separators=(",", ":"))
    return block.text or ""


def _tool(block: ContentBlock, requests: dict[str, ToolCall]) -> ToolCall | None:
    if block.block_type == "tool_use":
        return ToolCall(
            native_name=block.tool_name or "",
            kind=tool_kind(block.tool_name),
            call_id=block.tool_use_id,
            arguments=dict(block.tool_input or {}),
        )
    if block.block_type == "tool_result":
        # A result carries its request's name and kind, so detectors can pair
        # them without re-deriving the mapping. An orphan result stays ``other``.
        request = requests.get(block.tool_use_id or "")
        return ToolCall(
            native_name=request.native_name if request else "",
            kind=request.kind if request else tool_kind(None),
            call_id=block.tool_use_id,
        )
    return None


#: The model Claude Code names on messages it writes itself (an interrupted
#: request, an API error). They carry usage blocks, but no model ran.
HARNESS_MODEL = "<synthetic>"


def _usage(
    usage: LegacyUsage | None, model: str | None, cache_reported: bool
) -> TokenUsage | None:
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_tokens=usage.cache_creation_input_tokens,
        cache_read_tokens=usage.cache_read_input_tokens,
        model=model,
        cache_reported=cache_reported,
    )


def _lines(message: Message, block: ContentBlock) -> tuple[int, ...]:
    lines = block.source_lines or ([block.source_line] if block.source_line else [])
    return tuple(lines or message.source_lines)


def _with_stops(
    path: Path, session_id: str, source: str, events: list[Event]
) -> tuple[Event, ...]:
    """``events`` with a ``task.completed`` for each stop the transcript records.

    A stop is a ``stop_hook_summary`` record; its id comes from the turn it
    closes, by the rule the Stop hook uses too
    (:mod:`ter.adapters.claude_code_turns`, TER-OBS-005). Each stop goes after
    the events read from lines before it, and the sequence and ``parent_id``
    chain follow the merged order. A transcript without stops is unchanged.
    """
    with open(path, encoding="utf-8") as handle:
        numbered = (
            (line_number, record)
            for line_number, line in enumerate(handle, 1)
            for record in read_records((line,))
        )
        stops = [
            Event(
                id=stop_event_id(session_id, turn),
                session_id=session_id,
                sequence=0,
                kind=EventKind.TASK_COMPLETED,
                actor=Actor.SYSTEM,
                text="",
                provenance=Provenance(
                    source=source,
                    record_id=str(record.get("uuid") or f"stop:turn:{turn}"),
                    lines=(line_number,),
                ),
                timestamp=record_time(record),
            )
            for line_number, record, turn in stop_records(numbered)
        ]
    if not stops:
        return tuple(events)
    merged: list[Event] = []
    pending = iter(stops)
    stop = next(pending, None)
    for event in events:
        while stop is not None and stop.provenance.lines[0] < min(
            event.provenance.lines or (stop.provenance.lines[0] + 1,)
        ):
            merged.append(stop)
            stop = next(pending, None)
        merged.append(event)
    while stop is not None:
        merged.append(stop)
        stop = next(pending, None)
    return _chained(merged)


def _chained(events: list[Event]) -> tuple[Event, ...]:
    """``events`` renumbered in order, each pointing at the one before."""
    chained: list[Event] = []
    previous: EventId | None = None
    for sequence, event in enumerate(events):
        chained.append(replace(event, sequence=sequence, parent_id=previous))
        previous = event.id
    return tuple(chained)


def _numbered(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    with open(path, encoding="utf-8") as handle:
        return [
            (line_number, record)
            for line_number, line in enumerate(handle, 1)
            for record in read_records((line,))
        ]


def _with_subagents(
    path: Path, session_id: str, source: str, events: tuple[Event, ...]
) -> tuple[Event, ...]:
    """``events`` with a ``subagent.completed`` for each finished subagent.

    A subagent is one ``<session>/subagents/agent-<id>.jsonl`` file; it is
    finished when the parent or its own transcript holds a finish marker
    (:mod:`ter.adapters.claude_code_subagents`). Its id is the one the
    SubagentStop hook gives the same agent (TER-OBS-007), one per agent
    however often it was resumed. The event takes the latest marker's time
    and goes after every event stamped no later than that; it cites the
    marker's record. A transcript without a subagents folder is unchanged.
    """
    agents = subagent_transcripts(path)
    if not agents:
        return events
    in_parent = parent_finishes(_numbered(path))
    found: list[tuple[datetime | None, str, Event]] = []
    for agent_id, agent_path in agents.items():
        markers: list[tuple[FinishMarker, str]] = [
            (m, source) for m in in_parent.get(agent_id, [])
        ]
        own = final_turn(_numbered(agent_path))
        if own is not None:
            markers.append((own, f"{SUBAGENTS_DIR}/{agent_path.name}"))
        if not markers:
            continue  # still running, or stopped without a trace: no event
        marker, cited = max(markers, key=lambda pair: _sort_time(pair[0].timestamp))
        found.append(
            (
                marker.timestamp,
                agent_id,
                Event(
                    id=subagent_event_id(session_id, agent_id),
                    session_id=session_id,
                    sequence=0,
                    kind=EventKind.SUBAGENT_COMPLETED,
                    actor=Actor.SYSTEM,
                    text=_agent_type(agent_path),
                    provenance=Provenance(
                        source=cited,
                        record_id=marker.record_id or f"subagent:{agent_id}",
                        lines=(marker.line,),
                    ),
                    timestamp=marker.timestamp,
                ),
            )
        )
    if not found:
        return events
    found.sort(key=lambda item: (_sort_time(item[0]), item[1]))
    merged: list[Event] = []
    pending = [event for _, _, event in found]
    for event in events:
        # An event without a time keeps its place; untimed subagents go last.
        while (
            pending
            and pending[0].timestamp is not None
            and event.timestamp is not None
            and _sort_time(pending[0].timestamp) < _sort_time(event.timestamp)
        ):
            merged.append(pending.pop(0))
        merged.append(event)
    merged.extend(pending)
    return _chained(merged)


def _sort_time(moment: datetime | None) -> tuple[int, datetime]:
    # Untimed sorts after timed; datetime.max keeps the tuple comparable.
    if moment is None:
        return (1, datetime.max.replace(tzinfo=UTC))
    return (0, moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC))


def _agent_type(agent_path: Path) -> str:
    """The subagent's type from its ``.meta.json`` (as SubagentStop's
    ``agent_type``), or ``""``. The corpus importer copies no meta files."""
    meta = agent_path.with_name(agent_path.stem + ".meta.json")
    try:
        value = json.loads(meta.read_text(encoding="utf-8")).get("agentType")
    except (OSError, ValueError, AttributeError):
        return ""
    return value if isinstance(value, str) else ""
