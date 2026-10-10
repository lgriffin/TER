"""The formal Lean vocabulary of an agentic software session (maturity level L2).

Lean manufacturing asks of every activity: does it add value the customer
asked for, is it necessary but not itself valuable, or could it be avoided?
TER applies the same question to what a coding agent does, with the
developer's requested software outcome as the customer's value (point 6).

Everything here is a value type. Detectors (:mod:`.detectors`) produce
:class:`Finding` values that cite the events they rest on, so every
classification can be traced back to something observable (point 40).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ..events import Actor, EventId, EventKind, TokenUsage, ToolKind

__all__ = [
    "STAGE_ORDER",
    "UNCERTAIN_BELOW",
    "ActivityClass",
    "CycleVerdict",
    "ExplorationDriver",
    "ExplorationLabel",
    "Finding",
    "FindingKind",
    "FlowState",
    "LeanWaste",
    "Outcome",
    "ShellIntent",
    "Stage",
    "Step",
    "ValidationCycle",
]

#: Findings whose confidence is below this are reported as *uncertain*: shown,
#: labelled, and counted as waste until verified (points 85, 86; ADR 0006).
#: They never trigger an intervention such as model escalation (point 91).
UNCERTAIN_BELOW = 0.7


class ActivityClass(StrEnum):
    """The three Lean activity classes (point 15)."""

    VALUE_ADDING = "value_adding"
    NECESSARY_NON_VALUE_ADDING = "necessary_non_value_adding"
    AVOIDABLE = "avoidable"

    @property
    def label(self) -> str:
        return {
            ActivityClass.VALUE_ADDING: "Value-adding",
            ActivityClass.NECESSARY_NON_VALUE_ADDING: "Necessary, non-value-adding",
            ActivityClass.AVOIDABLE: "Avoidable",
        }[self]


class LeanWaste(StrEnum):
    """Classic Lean wastes, mapped onto agent behaviour (points 7, 12)."""

    REWORK = "rework"
    MOTION = "motion"
    OVER_PROCESSING = "over_processing"
    WAITING = "waiting"
    INVENTORY = "inventory"
    DEFECTS = "defects"
    OVERPRODUCTION = "overproduction"
    HANDOFFS = "handoffs"

    @property
    def label(self) -> str:
        return _WASTE_LABELS[self][0]

    @property
    def agentic_meaning(self) -> str:
        """What this waste looks like when the worker is a coding agent."""
        return _WASTE_LABELS[self][1]

    @property
    def flow(self) -> FlowState:
        """The flow state that tokens and time attributed to this waste are in."""
        return _WASTE_FLOW[self]


_WASTE_LABELS: dict[LeanWaste, tuple[str, str]] = {
    LeanWaste.REWORK: (
        "Rework",
        "Changing work again because an attempt did not move a failing check.",
    ),
    LeanWaste.MOTION: (
        "Motion",
        "Moving through the repository again: re-reading files, re-running searches.",
    ),
    LeanWaste.OVER_PROCESSING: (
        "Over-processing",
        "More handling than the outcome needs: repeated calls, reasoning or planning.",
    ),
    LeanWaste.WAITING: (
        "Waiting",
        "The session blocked on another model, agent or external call.",
    ),
    LeanWaste.INVENTORY: (
        "Inventory",
        "Context acquired and carried but never used by a later action.",
    ),
    LeanWaste.DEFECTS: (
        "Defects",
        "Work likely to hide defects: unvalidated or uninformed changes.",
    ),
    LeanWaste.OVERPRODUCTION: (
        "Overproduction",
        "Regenerating or rewriting work that was already satisfactory.",
    ),
    LeanWaste.HANDOFFS: (
        "Handoffs",
        "Delegating to a subagent or model and then doing the work anyway.",
    ),
}


class FlowState(StrEnum):
    """Where tokens and time went, for Agentic Flow Efficiency (points 79, 80)."""

    PROGRESSING = "progressing"
    RECOVERING = "recovering"
    REPEATING = "repeating"
    REWORKING = "reworking"
    WAITING = "waiting"
    INVENTORY = "inventory"

    @property
    def label(self) -> str:
        return self.value.capitalize()


_WASTE_FLOW: dict[LeanWaste, FlowState] = {
    LeanWaste.REWORK: FlowState.REWORKING,
    LeanWaste.OVERPRODUCTION: FlowState.REWORKING,
    LeanWaste.DEFECTS: FlowState.REWORKING,
    LeanWaste.MOTION: FlowState.REPEATING,
    LeanWaste.OVER_PROCESSING: FlowState.REPEATING,
    LeanWaste.WAITING: FlowState.WAITING,
    LeanWaste.HANDOFFS: FlowState.WAITING,
    LeanWaste.INVENTORY: FlowState.INVENTORY,
}


class Stage(StrEnum):
    """Stages of the Agentic Value Stream, from intent to response (point 13)."""

    INTENT = "intent"
    EXPLORE = "explore"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VALIDATE = "validate"
    RESPOND = "respond"

    @property
    def label(self) -> str:
        return self.value.capitalize()


#: Stages in value-stream order.
STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)


class ShellIntent(StrEnum):
    """What a shell command was for, read from the command line alone."""

    VALIDATE = "validate"
    EXPLORE = "explore"
    CHANGE = "change"
    OTHER = "other"


class Outcome(StrEnum):
    """Result of a validation run, read from its output."""

    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class FindingKind(StrEnum):
    """A *waste* finding costs tokens or time; a *risk* finding flags work at risk."""

    WASTE = "waste"
    RISK = "risk"


class CycleVerdict(StrEnum):
    """Iteration converges on a passing check; rework does not move it (point 37)."""

    ITERATION = "iteration"
    REWORK = "rework"


@dataclass(frozen=True)
class Step:
    """One event, with the facts detectors reason about.

    ``tokens`` are text-token estimates of generated events; ``context_tokens``
    are those of tool output returned to the agent. ``seconds`` is the wall
    time attributed to this step (the model producing a generated event, the
    tool running for a request, the developer for a prompt).
    """

    index: int
    event_id: EventId
    kind: EventKind
    actor: Actor
    stage: Stage
    tool_kind: ToolKind | None
    native_name: str | None
    call_id: str | None
    request_index: int | None
    call_key: str | None
    paths: tuple[str, ...]
    shell: ShellIntent | None
    command: str | None
    outcome: Outcome | None
    output_hash: str | None
    failure_signature: str | None
    tokens: int
    context_tokens: int
    words: frozenset[str]
    identifiers: frozenset[str]
    lines: frozenset[str]
    timestamp: datetime | None
    seconds: float
    subject: str
    #: The provider's usage for the model turn this event opened, if any; what
    #: a session is priced from (TER-ANL-040).
    usage: TokenUsage | None = None
    #: Content words of the questions the event's text asks (sentences ending
    #: in ``?``): the open questions a prompt, reasoning or response records.
    questions: frozenset[str] = frozenset()
    #: A shell request whose line runs a named check tool in any of its
    #: commands, even when ``shell`` reads it as a change (``sed -i … &&
    #: pytest``); see :func:`~.facts.runs_check`.
    runs_check: bool = False
    #: The first step after an ``attempt.started`` marker: the first event of
    #: a new attempt at the task (TER-DET-011 reads a re-attempt served by
    #: another model as an escalation).
    opens_attempt: bool = False
    #: A tool completion whose harness reported that the call itself failed
    #: (a refused write, an edit whose text was not found): it changed nothing.
    tool_failed: bool = False

    @property
    def is_generated(self) -> bool:
        return self.kind.is_generated

    @property
    def is_request(self) -> bool:
        return self.kind is EventKind.TOOL_REQUESTED

    @property
    def is_completion(self) -> bool:
        return self.kind is EventKind.TOOL_COMPLETED

    @property
    def is_edit(self) -> bool:
        return self.is_request and self.tool_kind in (
            ToolKind.FS_EDIT,
            ToolKind.FS_WRITE,
        )

    @property
    def is_validation(self) -> bool:
        return self.is_request and self.shell is ShellIntent.VALIDATE

    @property
    def checks(self) -> bool:
        """A request that runs a check: a validation, or a change line that
        also runs a named check tool."""
        return self.is_validation or (self.is_request and self.runs_check)

    @property
    def is_failover(self) -> bool:
        """A model route that failed and returned nothing (``route.failover``)."""
        return self.kind is EventKind.ROUTE_FAILOVER

    @property
    def is_escalation(self) -> bool:
        """A recorded model escalation (``route.escalated``, TER-RTE-002)."""
        return self.kind is EventKind.ROUTE_ESCALATED


@dataclass(frozen=True)
class Finding:
    """One detected waste or risk, with the evidence it rests on.

    ``evidence`` lists every event the finding cites, in session order.
    ``waste_events`` is the subset whose cost the finding claims, scaled by
    ``share`` (1.0 unless only part of an event is waste). A risk finding
    claims no cost.
    """

    id: str
    detector: str
    waste: LeanWaste
    kind: FindingKind
    activity_class: ActivityClass
    confidence: float
    title: str
    explanation: str
    evidence: tuple[EventId, ...]
    waste_events: tuple[EventId, ...]
    share: float
    subject: str
    tokens: int
    context_tokens: int
    seconds: float

    @property
    def uncertain(self) -> bool:
        return self.confidence < UNCERTAIN_BELOW

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "detector": self.detector,
            "waste": self.waste.value,
            "kind": self.kind.value,
            "activity_class": self.activity_class.value,
            "confidence": round(self.confidence, 4),
            "uncertain": self.uncertain,
            "title": self.title,
            "explanation": self.explanation,
            "evidence": list(self.evidence),
            "waste_events": list(self.waste_events),
            "share": round(self.share, 4),
            "subject": self.subject,
            "tokens": self.tokens,
            "context_tokens": self.context_tokens,
            "seconds": round(self.seconds, 3),
        }


@dataclass(frozen=True)
class ValidationCycle:
    """A failed validation, the edits that followed it, and the next run.

    ``verdict`` is :attr:`CycleVerdict.ITERATION` when the next run passed or
    failed differently (the edits moved the failure), and
    :attr:`CycleVerdict.REWORK` when it failed with the same signature.
    """

    failed_run: EventId
    failure: EventId
    fixes: tuple[EventId, ...]
    next_run: EventId
    next_result: EventId | None
    next_outcome: Outcome
    verdict: CycleVerdict
    command: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "failed_run": self.failed_run,
            "failure": self.failure,
            "fixes": list(self.fixes),
            "next_run": self.next_run,
            "next_result": self.next_result,
            "next_outcome": self.next_outcome.value,
            "verdict": self.verdict.value,
            "command": self.command,
            "reason": self.reason,
        }


class ExplorationDriver(StrEnum):
    """Why an exploration step happened, as far as the session shows (point 38).

    *Uncertainty-driven* exploration addresses an open question recorded
    earlier in the same task; *intent-directed* exploration names the task's
    own subject (a word or file of the prompt in force); *aimless* exploration
    is linked to neither. A label is evidence for a reader, never a waste
    classification by itself.
    """

    UNCERTAINTY_DRIVEN = "uncertainty_driven"
    INTENT_DIRECTED = "intent_directed"
    AIMLESS = "aimless"


@dataclass(frozen=True)
class ExplorationLabel:
    """The driver of one exploration request and the event that motivates it.

    ``motive`` is the event recording the open question it addresses (for
    :attr:`ExplorationDriver.UNCERTAINTY_DRIVEN`) or the prompt in force (for
    :attr:`ExplorationDriver.INTENT_DIRECTED`); ``shared`` are the words that
    link the two.
    """

    event_id: EventId
    driver: ExplorationDriver
    motive: EventId | None
    shared: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "driver": self.driver.value,
            "motive": self.motive,
            "shared": list(self.shared),
        }
