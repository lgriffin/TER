"""Provider-neutral event model: the ``ter.event`` contract.

An agent session is a sequence of events. Every harness adapter translates
its native records into these types, and every analysis in TER reads only
these types. Detectors reason about *tool kinds* (``fs.read``) rather than
native tool names (``Read``), so a second harness needs a new adapter and no
change here.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import NewType

#: Version of the normalised ``ter.event`` contract. Bumped on any change to
#: the fields below that consumers can observe.
#:
#: 0.2 added the lifecycle kinds ``task.completed`` and ``subagent.completed``;
#: 0.3 the routing kinds ``route.selected``, ``route.failover``,
#: ``attempt.started``, ``verification.completed`` and ``outcome.recorded``;
#: 0.4 the usage fields ``model`` and ``cache_reported``, so a session can be
#: priced from its events alone (TER-ANL-040, TER-ANL-041, TER-EXP-001);
#: 0.5 the kind ``context.supplied`` (TER-EVD-004).
EVENT_SCHEMA_VERSION = "ter.event/0.5"

#: Every contract version this build reads. Each is a subset of the current
#: one, so a record written under any of them decodes unchanged.
READABLE_SCHEMA_VERSIONS = frozenset(
    {
        "ter.event/0.1",
        "ter.event/0.2",
        "ter.event/0.3",
        "ter.event/0.4",
        EVENT_SCHEMA_VERSION,
    }
)


class SessionFormatError(ValueError):
    """A session source cannot read the file it was given as a session: not
    JSONL, not UTF-8, empty, or not a run of the format it reads."""


#: Stable, content-derived identity of an event. The same source record always
#: yields the same id, which makes replay idempotent and findings citable.
EventId = NewType("EventId", str)


class Actor(StrEnum):
    """Who produced an event."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class EventKind(StrEnum):
    """What happened. Values are stable identifiers in the ``ter.event`` contract."""

    PROMPT = "intent.stated"
    REASONING = "reasoning"
    RESPONSE = "response"
    TOOL_REQUESTED = "tool.requested"
    TOOL_COMPLETED = "tool.completed"
    TASK_COMPLETED = "task.completed"
    SUBAGENT_COMPLETED = "subagent.completed"
    # Routing harnesses (GARE, issue #52): how a run chose, retried and
    # judged its routes.
    ROUTE_SELECTED = "route.selected"
    ROUTE_FAILOVER = "route.failover"
    ATTEMPT_STARTED = "attempt.started"
    VERIFICATION_COMPLETED = "verification.completed"
    OUTCOME_RECORDED = "outcome.recorded"
    # Context bundles (TER-EVD-004, TER-CTX-001): TER handed one fragment
    # of a context bundle to the agent.
    CONTEXT_SUPPLIED = "context.supplied"
    # Model routing (L3, TER-RTE-002, TER-DET-011): a router moved a task to
    # a higher role of its routing profile. The text holds the triggering
    # signal, both roles, the latency and the token usage
    # (``ter.domain.routing.RouteEscalation``).
    ROUTE_ESCALATED = "route.escalated"
    # TER-EXP-002: measures TER records about a session (its TER 3 ratio and
    # its outcome verdict), so every reported metric is recomputed from the
    # event log alone. Records, not activity: the fold counts none of them.
    METRIC_RECORDED = "metric.recorded"
    VERDICT_RECORDED = "verdict.recorded"

    @property
    def is_record(self) -> bool:
        """True for a measure TER recorded about the session (TER-EXP-002).

        A record is not something the developer, the agent or a harness did:
        no step, no count, no token figure includes it.
        """
        return self in (EventKind.METRIC_RECORDED, EventKind.VERDICT_RECORDED)

    @property
    def is_lifecycle(self) -> bool:
        """True for markers of the session's shape that carry no conversation.

        Task and subagent ends, and a routing harness's route, attempt,
        verification and outcome markers: none is a step of the value stream
        and none is scored.
        """
        return self in _LIFECYCLE

    @property
    def is_generated(self) -> bool:
        """True for events the agent generated, which are the only ones TER scores."""
        return self in {
            EventKind.REASONING,
            EventKind.RESPONSE,
            EventKind.TOOL_REQUESTED,
        }


_LIFECYCLE = frozenset(
    {
        EventKind.TASK_COMPLETED,
        EventKind.SUBAGENT_COMPLETED,
        EventKind.ROUTE_SELECTED,
        EventKind.ROUTE_FAILOVER,
        EventKind.ATTEMPT_STARTED,
        EventKind.VERIFICATION_COMPLETED,
        EventKind.OUTCOME_RECORDED,
        EventKind.CONTEXT_SUPPLIED,
        EventKind.ROUTE_ESCALATED,
    }
)


class ToolKind(StrEnum):
    """Harness-independent category of a tool call."""

    FS_READ = "fs.read"
    FS_SEARCH = "fs.search"
    FS_EDIT = "fs.edit"
    FS_WRITE = "fs.write"
    EXEC_SHELL = "exec.shell"
    NET_FETCH = "net.fetch"
    AGENT_HANDOFF = "agent.handoff"
    PLAN = "plan.todo"
    OTHER = "other"


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported token usage for one model turn.

    ``model`` is the model the provider says served the turn (None when the
    source does not say). ``cache_reported`` is False when the source's usage
    record had no cache fields, so the cache figures are 0 because nothing
    was reported, not because nothing was cached; costs built on such a turn
    are estimates (TER-ANL-041).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    model: str | None = None
    cache_reported: bool = True

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation, keeping the native name alongside the neutral kind."""

    native_name: str
    kind: ToolKind
    call_id: str | None = None
    arguments: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Provenance:
    """Where an event came from, precisely enough to cite it in a report."""

    source: str
    record_id: str
    lines: tuple[int, ...] = ()
    block_index: int | None = None
    fingerprint: str | None = None


@dataclass(frozen=True)
class Event:
    """One normalised event in an agent session."""

    id: EventId
    session_id: str
    sequence: int
    kind: EventKind
    actor: Actor
    text: str
    provenance: Provenance
    timestamp: datetime | None = None
    tool: ToolCall | None = None
    usage: TokenUsage | None = None
    parent_id: EventId | None = None


#: How each known usage limit reads beside the figures it qualifies
#: (TER-SRC-014, TER-SRC-016).
USAGE_LIMIT_TEXT: Mapping[str, str] = {
    "no-cache-tokens": "the source reports no cache tokens; cache figures are 0",
    "no-response-text": (
        "the source exports no response text; text tokens count route labels, "
        "not what the model wrote"
    ),
    "intents-from-traces": (
        "the source captured no prompt text for some or all traces; each such "
        "trace counts as one intent"
    ),
    "convention-unstated": (
        "the source names no telemetry convention version; read under the "
        "latest one TER knows"
    ),
}
#: Limits that qualify token counts taken from event text, not usage figures.
TEXT_LIMITS = frozenset({"no-response-text"})


#: The text of a tool result that returned only images: the digests of
#: their bytes, so equal text means the same pixels (TER-SRC-025).
IMAGE_RESULT = re.compile(r"^\[image sha256:[0-9a-f]+(?:,[0-9a-f]+)*\]$")


def image_result_text(digests: Iterable[str]) -> str:
    """The text a source gives an image-only tool result."""
    return "[image sha256:" + ",".join(digests) + "]"


def describe_limit(limit: str) -> str:
    """The reader-facing text for a usage limit (the id when unknown)."""
    return USAGE_LIMIT_TEXT.get(limit, limit)


@dataclass(frozen=True)
class UnrecognisedRecord:
    """A source record the adapter could not map, kept so coverage is honest."""

    line: int
    record_type: str


@dataclass(frozen=True)
class MetadataRecord:
    """A source record of a documented type that carries no agent activity.

    Harness bookkeeping (titles, queue operations, permission modes, injected
    reminders) is recognised rather than mapped: it describes the session, it
    is not something the developer or the agent did (TER-SRC-005).
    """

    line: int
    record_type: str


@dataclass(frozen=True)
class SessionTrace:
    """A whole session as normalised events, plus what could not be mapped."""

    session_id: str
    source_format: str
    events: tuple[Event, ...]
    unrecognised: tuple[UnrecognisedRecord, ...] = ()
    #: Records of documented types that carry no agent activity.
    metadata: tuple[MetadataRecord, ...] = ()
    schema_version: str = EVENT_SCHEMA_VERSION
    #: What the source cannot report, e.g. ``no-cache-tokens``; reports state
    #: each limit beside their token figures.
    usage_limits: tuple[str, ...] = ()

    @property
    def unrecognised_by_type(self) -> Mapping[str, int]:
        return dict(Counter(r.record_type for r in self.unrecognised))

    @property
    def metadata_by_type(self) -> Mapping[str, int]:
        return dict(Counter(r.record_type for r in self.metadata))

    @property
    def coverage(self) -> float:
        """Share of source records accounted for (1.0 when empty).

        A record is accounted for when it produced at least one event or is a
        documented metadata record; unrecognised records are the rest.
        """
        mapped = len({e.provenance.record_id for e in self.events})
        known = mapped + len(self.metadata)
        total = known + len(self.unrecognised)
        return 1.0 if total == 0 else known / total

    def generated(self) -> tuple[Event, ...]:
        """Events the agent generated; user input and tool output are excluded."""
        return tuple(e for e in self.events if e.kind.is_generated)


def make_event_id(*parts: object) -> EventId:
    """Derive a stable event id from the parts that identify a source record.

    The id is the first 16 hex characters of a SHA-256 over the NUL-joined
    parts, so it is stable across runs, platforms and Python versions.
    """
    raw = "\0".join("" if p is None else str(p) for p in parts)
    return EventId(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16])
