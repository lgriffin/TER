"""From events to :class:`~.model.Step` values: the Lean view of a session.

:class:`StepLog` is a fold. Each :meth:`StepLog.add` reads one event's facts
in O(1) amortised time (plus the size of the event itself) and never revisits
earlier events, so it can ride inside the L1 ``AnalysisEngine``. The few facts
that depend on later events (a step's wall time, which is the gap to the next
event) are resolved by :meth:`StepLog.steps` when a report is asked for.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime

from ..events import Actor, Event, EventId, EventKind, ToolKind
from .facts import (
    content_words,
    defined_identifiers,
    failure_signature,
    normalise_command,
    output_fingerprint,
    question_words,
    runs_check,
    shell_intent,
    source_lines,
    tool_call_failed,
    tool_paths,
    validation_outcome,
    write_created,
)
from .model import Outcome, ShellIntent, Stage, Step

__all__ = ["StepLog", "stage_of_tool"]

_TOOL_STAGE: dict[ToolKind, Stage] = {
    ToolKind.FS_READ: Stage.EXPLORE,
    ToolKind.FS_SEARCH: Stage.EXPLORE,
    ToolKind.NET_FETCH: Stage.EXPLORE,
    ToolKind.AGENT_HANDOFF: Stage.EXPLORE,
    ToolKind.PLAN: Stage.PLAN,
    ToolKind.FS_EDIT: Stage.IMPLEMENT,
    ToolKind.FS_WRITE: Stage.IMPLEMENT,
}

_SHELL_STAGE: dict[ShellIntent, Stage] = {
    ShellIntent.VALIDATE: Stage.VALIDATE,
    ShellIntent.EXPLORE: Stage.EXPLORE,
    ShellIntent.CHANGE: Stage.IMPLEMENT,
    ShellIntent.OTHER: Stage.IMPLEMENT,
}

_SUBJECT_KEYS = (
    "file_path",
    "notebook_path",
    "pattern",
    "path",
    "url",
    "description",
    "query",
)


#: Routing markers that are steps of the value stream: the session waited on
#: them.
_STEP_MARKERS = frozenset({EventKind.ROUTE_FAILOVER, EventKind.ROUTE_ESCALATED})


def stage_of_tool(kind: ToolKind, shell: ShellIntent | None) -> Stage:
    """The value-stream stage a tool request belongs to."""
    if kind is ToolKind.EXEC_SHELL:
        return _SHELL_STAGE[shell or ShellIntent.OTHER]
    return _TOOL_STAGE.get(kind, Stage.EXPLORE)


def _call_key(kind: ToolKind, arguments: Mapping[str, object]) -> str:
    raw = (
        kind.value
        + "\0"
        + json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _subject(
    kind: ToolKind, arguments: Mapping[str, object], command: str | None
) -> str:
    if command is not None:
        return command
    for key in _SUBJECT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return kind.value


def _argument_text(arguments: Mapping[str, object]) -> str:
    return " ".join(
        str(v) for v in arguments.values() if isinstance(v, (str, int, float))
    )


@dataclass
class _Open:
    index: int
    stage: Stage
    shell: ShellIntent | None
    tool_kind: ToolKind
    native_name: str


class StepLog:
    """Accumulates steps, one event at a time. Idempotent by event id."""

    def __init__(self) -> None:
        self._steps: list[Step] = []
        self._seen: set[EventId] = set()
        self._requests: dict[str, _Open] = {}
        # Requests that carry no call id and have no result yet, oldest first.
        self._unkeyed: list[str] = []
        # An ``attempt.started`` seen since the last step: the next step opens
        # a new attempt (TER-DET-011).
        self._attempt = False

    def __len__(self) -> int:
        return len(self._steps)

    def add(self, event: Event, tokens: int) -> bool:
        """Read one event. Returns False (and changes nothing) for a redelivery."""
        if event.id in self._seen:
            return False
        self._seen.add(event.id)
        if event.kind is EventKind.ATTEMPT_STARTED:
            self._attempt = True
        if event.kind.is_lifecycle and event.kind not in _STEP_MARKERS:
            # A task or subagent finishing, a route chosen or an attempt
            # started is not a step of the value stream. A failed model route
            # is: the session waited on it (TER-DET-008); so is an escalated
            # one (TER-DET-011).
            return True
        step = self._read(event, tokens)
        if self._attempt:
            step = replace(step, opens_attempt=True)
            self._attempt = False
        self._steps.append(step)
        return True

    @property
    def last(self) -> Step | None:
        """The most recent step (lifecycle events add none)."""
        return self._steps[-1] if self._steps else None

    def _read(self, event: Event, tokens: int) -> Step:
        index = len(self._steps)
        tool = event.tool
        stage = Stage.PLAN
        tool_kind = tool.kind if tool else None
        native = tool.native_name if tool else None
        call_id = tool.call_id if tool else None
        request_index: int | None = None
        call_key: str | None = None
        paths: tuple[str, ...] = ()
        shell: ShellIntent | None = None
        command: str | None = None
        outcome: Outcome | None = None
        output_hash: str | None = None
        signature: str | None = None
        tool_failed = False
        created: bool | None = None
        generated = tokens if event.kind.is_generated else 0
        context = tokens if event.kind is EventKind.TOOL_COMPLETED else 0
        words = (
            content_words(event.text)
            if event.kind is not EventKind.TOOL_COMPLETED
            else frozenset()
        )
        identifiers: frozenset[str] = frozenset()
        lines: frozenset[str] = frozenset()
        subject = ""
        checks = False

        if event.kind is EventKind.PROMPT:
            stage = Stage.INTENT
        elif event.kind is EventKind.RESPONSE:
            stage = Stage.RESPOND
        elif event.kind in _STEP_MARKERS:
            # A model call that failed on its way to a response, or a
            # recorded escalation to another model.
            stage = Stage.RESPOND
            subject = event.text
        elif event.kind is EventKind.REASONING:
            stage = Stage.PLAN
        elif event.kind is EventKind.TOOL_REQUESTED and tool is not None:
            arguments = tool.arguments
            paths = tool_paths(arguments)
            call_key = _call_key(tool.kind, arguments)
            if tool.kind is ToolKind.EXEC_SHELL:
                raw = arguments.get("command")
                raw = raw if isinstance(raw, str) else ""
                command = normalise_command(raw)
                # Classified before normalising: newlines separate commands.
                shell = shell_intent(raw)
                checks = runs_check(raw)
            stage = stage_of_tool(tool.kind, shell)
            subject = _subject(tool.kind, arguments, command)
            words = content_words(_argument_text(arguments))
            content = arguments.get("content")
            if tool.kind is ToolKind.FS_WRITE and isinstance(content, str):
                lines = source_lines(content)
            key = call_id or f"\0{event.id}"
            self._requests[key] = _Open(
                index, stage, shell, tool.kind, tool.native_name
            )
            if not call_id:
                self._unkeyed.append(key)
        elif event.kind is EventKind.TOOL_COMPLETED:
            opened = self._requests.get(call_id) if call_id else self._unkeyed_request()
            output_hash = output_fingerprint(event.text)
            identifiers = defined_identifiers(event.text)
            tool_failed = tool_call_failed(event.text)
            if tool_kind is ToolKind.FS_WRITE:
                created = write_created(event.text)
            if opened is None:
                stage = Stage.EXPLORE
            else:
                request_index = opened.index
                stage = opened.stage
                shell = opened.shell
                tool_kind = opened.tool_kind
                native = opened.native_name
                request = self._steps[opened.index]
                paths = request.paths
                command = request.command
                subject = request.subject
                # A check chained beside a change (``sed -i … && pytest``)
                # reports its outcome too: the detector that clears edits on
                # it must also see when it failed.
                if shell is ShellIntent.VALIDATE or request.runs_check:
                    outcome = validation_outcome(event.text)
                    if outcome is Outcome.FAILED:
                        signature = failure_signature(event.text)
                if tool_kind is ToolKind.FS_READ:
                    lines = source_lines(event.text)
                elif (
                    tool_kind is not ToolKind.FS_EDIT
                    and tool_kind is not ToolKind.FS_WRITE
                ):
                    words = content_words(event.text)

        return Step(
            index=index,
            event_id=event.id,
            kind=event.kind,
            actor=event.actor,
            stage=stage,
            tool_kind=tool_kind,
            native_name=native,
            call_id=call_id,
            request_index=request_index,
            call_key=call_key,
            paths=paths,
            shell=shell,
            command=command,
            outcome=outcome,
            output_hash=output_hash,
            tool_failed=tool_failed,
            write_created=created,
            failure_signature=signature,
            tokens=generated,
            context_tokens=context,
            words=words,
            identifiers=identifiers,
            lines=lines,
            timestamp=event.timestamp,
            seconds=0.0,
            subject=subject or (event.text[:80] if event.actor is Actor.USER else ""),
            usage=event.usage,
            questions=(
                question_words(event.text)
                if event.kind
                in (EventKind.PROMPT, EventKind.REASONING, EventKind.RESPONSE)
                else frozenset()
            ),
            runs_check=checks,
        )

    def _unkeyed_request(self) -> _Open | None:
        """The request a result without a call id answers, when that is certain.

        Only one id-less request waiting for its result is unambiguous; with
        none or several the result stays an orphan rather than a guess.
        """
        if len(self._unkeyed) != 1:
            return None
        return self._requests[self._unkeyed.pop()]

    def steps(self) -> tuple[Step, ...]:
        """The steps so far, with wall time attributed.

        The gap before a tool result is the tool's running time and belongs to
        its request; every other gap belongs to the event it precedes (the
        model generating it, or the developer writing a prompt).
        """
        seconds = [0.0] * len(self._steps)
        previous: datetime | None = None
        for step in self._steps:
            current = step.timestamp
            if previous is not None and current is not None:
                gap = max((current - previous).total_seconds(), 0.0)
                owner = step.index
                if step.is_completion and step.request_index is not None:
                    owner = step.request_index
                seconds[owner] += gap
            if current is not None:
                previous = current
        return tuple(
            replace(step, seconds=seconds[step.index]) if seconds[step.index] else step
            for step in self._steps
        )
