"""L2 analysis: classify every step, map the value stream, keep the scorecard.

:func:`analyse_steps` turns steps into a :class:`LeanAnalysis`:

1. Detectors (:mod:`.detectors`) produce findings and validation cycles.
2. Every generated step gets an activity class from its stage (the *basis*
   says which rule), and the share of it that a confident waste finding
   claims becomes avoidable; the share an uncertain finding claims is kept
   apart as *uncertain* (points 15, 16, 40, 85, 86). Uncertain waste is
   still waste until verified: it counts in the waste totals and flow
   efficiency, labelled so it is never mistaken for a confirmed finding.
3. Tokens and time are split into flow states for Agentic Flow Efficiency
   (points 79, 80): the share progressing or recovering (productive
   iteration), against repeating, reworking, waiting and inventory. The
   scorecard keeps every dimension separate (points 82 to 84).

:class:`LeanAnalyser` is the incremental form; batch analysis is the same
fold (:func:`explain`), so live and batch agree by construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ..events import Event, EventId, EventKind, ToolKind
from ..stack import FileTouch, SessionProfile, languages_of
from .detectors import (
    DEFAULT_REGISTRY,
    DetectorRegistry,
    SessionView,
    exploration_labels,
    validation_cycles,
)
from .drift import EVIDENCE_DETECTORS
from .evidence_graph import GroundedEvidenceGraph, build_grounded_graph
from .graph import EvidenceGraph, build_graph
from .grounding import RepositoryGrounding
from .intent import (
    DEFAULT_INTENT_CONFIG,
    LEXICAL_ALIGNMENT,
    AlignmentScorer,
    IntentConfig,
    IntentLog,
    IntentTimeline,
    build_intent,
)
from .model import (
    STAGE_ORDER,
    ActivityClass,
    CycleVerdict,
    ExplorationLabel,
    Finding,
    FindingKind,
    FlowState,
    Outcome,
    ShellIntent,
    Stage,
    Step,
    ValidationCycle,
)
from .steps import StepLog
from .surface import GROUNDED_DETECTORS, ChangeSurface, change_surfaces
from .usage import EvidenceUsage, FileNames, evidence_usage
from .value import OutcomeValue, outcome_value
from .wip import WipReport, WipTracker

__all__ = [
    "Classification",
    "Composite",
    "LeanAnalyser",
    "LeanAnalysis",
    "Scorecard",
    "StageSummary",
    "TerMeasure",
    "analyse_steps",
    "explain",
    "session_profile",
]

#: Key for tokens a finding below the confidence threshold may claim.
UNCERTAIN = "uncertain"


class _Counter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class TerMeasure:
    """The TER 3 ratio, carried into the scorecard with how it was computed."""

    value: float
    method: str


@dataclass(frozen=True)
class Classification:
    """How one event was classified, and on what basis (point 40).

    ``avoidable_share`` is the share a confident waste finding claims;
    ``uncertain_share`` the further share an uncertain one claims, charged
    to ``uncertain_basis``. ``basis`` is a finding id or ``stage:<rule>``.
    ``flow`` is the waste's flow state when either share is non-zero.
    """

    event_id: EventId
    stage: Stage
    activity_class: ActivityClass | None
    avoidable_share: float
    uncertain_share: float
    flow: FlowState
    basis: str
    uncertain_basis: str | None = None


@dataclass(frozen=True)
class StageSummary:
    """One box of the value stream map."""

    stage: Stage
    steps: int
    tokens: int
    context_tokens: int
    seconds: float
    avoidable_tokens: int
    uncertain_tokens: int
    findings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "steps": self.steps,
            "tokens": self.tokens,
            "context_tokens": self.context_tokens,
            "seconds": round(self.seconds, 3),
            "avoidable_tokens": self.avoidable_tokens,
            "uncertain_tokens": self.uncertain_tokens,
            "findings": list(self.findings),
        }


@dataclass(frozen=True)
class Composite:
    """An aggregate shown only with its composition (point 84)."""

    value: float
    components: tuple[tuple[str, float, float], ...]
    formula: str

    def as_dict(self) -> dict[str, object]:
        return {
            "value": round(self.value, 4),
            "components": [
                {"name": n, "value": round(v, 4), "weight": round(w, 4)}
                for n, v, w in self.components
            ],
            "formula": self.formula,
        }


@dataclass(frozen=True)
class Scorecard:
    """Every dimension on its own; no single opaque score (points 79 to 84).

    ``waste_tokens`` and ``findings`` include uncertain waste, which counts
    until verified (ADR 0006); ``uncertain_waste_tokens`` and
    ``uncertain_findings`` say how much of it is still unverified.
    """

    generated_tokens: int
    context_tokens: int
    agent_seconds: float
    developer_seconds: float
    flow_tokens: tuple[tuple[FlowState, int], ...]
    flow_seconds: tuple[tuple[FlowState, float], ...]
    flow_efficiency_tokens: float | None
    flow_efficiency_time: float | None
    activity_tokens: tuple[tuple[str, int], ...]
    activity_seconds: tuple[tuple[str, float], ...]
    waste_tokens: int
    waste_context_tokens: int
    waste_seconds: float
    uncertain_waste_tokens: int
    findings: int
    uncertain_findings: int
    risks: int
    iterations: int
    rework_cycles: int
    ter: TerMeasure | None
    composite: Composite | None
    # Quality: validation runs by what their output said (TER-SCR-001).
    validation_runs: int = 0
    validations_passed: int = 0
    validations_failed: int = 0

    def activity_share(self, key: ActivityClass | str) -> float:
        total = sum(n for _, n in self.activity_tokens)
        value = dict(self.activity_tokens).get(str(key), 0)
        return value / total if total else 0.0

    def as_dict(self) -> dict[str, object]:
        def eff(v: float | None) -> float | None:
            return None if v is None else round(v, 4)

        return {
            "generated_tokens": self.generated_tokens,
            "context_tokens": self.context_tokens,
            "agent_seconds": round(self.agent_seconds, 3),
            "developer_seconds": round(self.developer_seconds, 3),
            "flow_tokens": {k.value: n for k, n in self.flow_tokens},
            "flow_seconds": {k.value: round(n, 3) for k, n in self.flow_seconds},
            "flow_efficiency_tokens": eff(self.flow_efficiency_tokens),
            "flow_efficiency_time": eff(self.flow_efficiency_time),
            "activity_tokens": dict(self.activity_tokens),
            "activity_seconds": {k: round(v, 3) for k, v in self.activity_seconds},
            "waste_tokens": self.waste_tokens,
            "waste_context_tokens": self.waste_context_tokens,
            "waste_seconds": round(self.waste_seconds, 3),
            "uncertain_waste_tokens": self.uncertain_waste_tokens,
            "findings": self.findings,
            "uncertain_findings": self.uncertain_findings,
            "risks": self.risks,
            "iterations": self.iterations,
            "rework_cycles": self.rework_cycles,
            "ter": None
            if self.ter is None
            else {"value": round(self.ter.value, 4), "method": self.ter.method},
            "composite": None if self.composite is None else self.composite.as_dict(),
            "validation_runs": self.validation_runs,
            "validations_passed": self.validations_passed,
            "validations_failed": self.validations_failed,
        }


@dataclass(frozen=True)
class LeanAnalysis:
    """Everything L2 knows about a session, all traceable to event ids."""

    session_id: str | None
    events: int
    steps: tuple[Step, ...]
    findings: tuple[Finding, ...]
    cycles: tuple[ValidationCycle, ...]
    classifications: tuple[Classification, ...]
    value_stream: tuple[StageSummary, ...]
    scorecard: Scorecard
    graph: EvidenceGraph
    detectors: tuple[tuple[str, str, str, str], ...]
    #: Unresolved hypotheses, tasks, edits and failures after every event.
    wip: WipReport
    intent: IntentTimeline = field(default_factory=IntentTimeline)

    @property
    def drift_findings(self) -> tuple[Finding, ...]:
        """Findings of the ``intent_drift`` detector (TER-ITN-003)."""
        return tuple(f for f in self.findings if f.detector == "intent_drift")

    #: Why each exploration request happened (TER-DET-009, point 38).
    exploration: tuple[ExplorationLabel, ...] = ()
    #: The repository evidence the analysis was grounded on (L3), if any.
    repository: RepositoryGrounding | None = None
    #: The expected change surface of each task that edits (TER-EVD-006).
    surfaces: tuple[ChangeSurface, ...] = ()
    #: The languages the session read and edited and, with repository
    #: evidence, its repository's stack (TER-STK-001, TER-STK-002).
    profile: SessionProfile = field(default_factory=SessionProfile)
    #: Whether a later event used each repository read (TER-EVD-008, L3).
    usage: EvidenceUsage | None = None
    #: Exploration, reasoning and validation judged against the outcome
    #: (TER-LEN-009, L3).
    value: OutcomeValue | None = None
    #: The role-typed evidence graph with repository usage edges
    #: (TER-GRF-001, L3).
    grounded_graph: GroundedEvidenceGraph | None = None

    @property
    def waste_findings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.kind is FindingKind.WASTE)

    @property
    def risk_findings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.kind is FindingKind.RISK)

    def finding(self, finding_id: str) -> Finding:
        return next(f for f in self.findings if f.id == finding_id)

    def allocated_waste_tokens(self) -> dict[str, float]:
        """Generated tokens each waste finding is charged with.

        This is the scorecard's allocation: every event's avoidable share goes
        to the confident finding that claims it (its classification basis) and
        its uncertain share to the uncertain one (``uncertain_basis``), so the
        values add up to the scorecard's waste tokens, never counting an event
        twice and never counting context tokens.
        """
        out: dict[str, float] = {}
        for step, c in zip(self.steps, self.classifications, strict=True):
            if c.activity_class is None:
                continue
            if c.avoidable_share > 0:
                out[c.basis] = out.get(c.basis, 0.0) + step.tokens * c.avoidable_share
            if c.uncertain_basis is not None and c.uncertain_share > 0:
                out[c.uncertain_basis] = (
                    out.get(c.uncertain_basis, 0.0) + step.tokens * c.uncertain_share
                )
        return out

    def as_dict(self, *, graph: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "schema": "ter.lean/0.1",
            "session_id": self.session_id,
            "events": self.events,
            "findings": [f.as_dict() for f in self.findings],
            "cycles": [c.as_dict() for c in self.cycles],
            "value_stream": [s.as_dict() for s in self.value_stream],
            "scorecard": self.scorecard.as_dict(),
            "wip": self.wip.as_dict(),
            "classifications": [
                [
                    c.event_id,
                    c.stage.value,
                    c.activity_class.value if c.activity_class else None,
                    round(c.avoidable_share, 4),
                    round(c.uncertain_share, 4),
                    c.flow.value,
                    c.basis,
                ]
                for c in self.classifications
            ],
            "detectors": [
                {"id": i, "waste": w, "kind": k, "confidence_rule": r}
                for i, w, k, r in self.detectors
            ],
            "intent": self.intent.as_dict([f.id for f in self.drift_findings]),
            "exploration": [e.as_dict() for e in self.exploration],
            "profile": self.profile.as_dict(),
        }
        if self.repository is not None:
            # Only a grounded (L3) analysis has these; an L2 one is unchanged.
            out["repository"] = self.repository.as_dict()
            out["change_surfaces"] = [s.as_dict() for s in self.surfaces]
            if self.usage is not None:
                out["evidence_usage"] = self.usage.as_dict()
            if self.value is not None:
                out["outcome_value"] = self.value.as_dict()
            if graph and self.grounded_graph is not None:
                out["grounded_evidence_graph"] = self.grounded_graph.as_dict()
        if graph:
            out["evidence_graph"] = self.graph.as_dict()
        return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _final_responses(steps: tuple[Step, ...]) -> set[int]:
    """Indexes of the last response before each prompt and at the end."""
    finals: set[int] = set()
    last: int | None = None
    for step in steps:
        if step.kind is EventKind.PROMPT and last is not None:
            finals.add(last)
            last = None
        elif step.kind is EventKind.RESPONSE:
            last = step.index
    if last is not None:
        finals.add(last)
    return finals


def _base_class(
    step: Step, steps: tuple[Step, ...], finals: set[int]
) -> tuple[ActivityClass | None, str]:
    if step.kind is EventKind.PROMPT:
        return None, "stage:developer input is not scored"
    if step.is_completion:
        if step.request_index is None:
            return (
                ActivityClass.NECESSARY_NON_VALUE_ADDING,
                "stage:tool result without a request",
            )
        request = steps[step.request_index]
        cls, basis = _base_class(request, steps, finals)
        return cls, basis
    if step.stage is Stage.IMPLEMENT:
        if step.tool_kind in (ToolKind.FS_EDIT, ToolKind.FS_WRITE):
            return (
                ActivityClass.VALUE_ADDING,
                "stage:implement edits change the software",
            )
        return (
            ActivityClass.NECESSARY_NON_VALUE_ADDING,
            "stage:implement set-up command",
        )
    if step.is_failover:
        return (
            ActivityClass.NECESSARY_NON_VALUE_ADDING,
            "stage:respond failed model route returned nothing",
        )
    if step.is_escalation:
        return (
            ActivityClass.NECESSARY_NON_VALUE_ADDING,
            "stage:respond route escalated to another model",
        )
    if step.stage is Stage.RESPOND:
        if step.index in finals:
            return (
                ActivityClass.VALUE_ADDING,
                "stage:final response delivers the outcome",
            )
        return ActivityClass.NECESSARY_NON_VALUE_ADDING, "stage:intermediate narration"
    return (
        ActivityClass.NECESSARY_NON_VALUE_ADDING,
        f"stage:{step.stage.value} is necessary but adds no value itself",
    )


def _classify(
    steps: tuple[Step, ...],
    findings: tuple[Finding, ...],
    cycles: tuple[ValidationCycle, ...],
) -> tuple[Classification, ...]:
    confident: dict[EventId, Finding] = {}
    uncertain: dict[EventId, Finding] = {}
    for finding in findings:
        if finding.kind is not FindingKind.WASTE:
            continue
        target = uncertain if finding.uncertain else confident
        for event_id in finding.waste_events:
            held = target.get(event_id)
            if held is None or (finding.share, finding.confidence) > (
                held.share,
                held.confidence,
            ):
                target[event_id] = finding
    recovering = {
        e
        for c in cycles
        if c.verdict is CycleVerdict.ITERATION
        for e in (*c.fixes, c.next_run, *((c.next_result,) if c.next_result else ()))
    }
    finals = _final_responses(steps)
    out: list[Classification] = []
    for step in steps:
        cls, basis = _base_class(step, steps, finals)
        sure = confident.get(step.event_id)
        maybe = uncertain.get(step.event_id)
        avoid = sure.share if sure else 0.0
        unsure = min(maybe.share, 1.0 - avoid) if maybe else 0.0
        if sure is not None:
            flow = sure.waste.flow
            basis = sure.id
            if avoid >= 1.0:
                cls = ActivityClass.AVOIDABLE
        elif maybe is not None and unsure > 0:
            # Uncertain waste counts until verified (ADR 0006).
            flow = maybe.waste.flow
        elif step.event_id in recovering:
            flow = FlowState.RECOVERING
            basis = f"{basis}; productive iteration"
        else:
            flow = FlowState.PROGRESSING
        if maybe is not None and sure is None:
            basis = f"{basis}; uncertain {maybe.id}"
        out.append(
            Classification(
                step.event_id,
                step.stage,
                cls,
                avoid,
                unsure,
                flow,
                basis,
                maybe.id if maybe is not None and unsure > 0 else None,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Scorecard and value stream
# ---------------------------------------------------------------------------


def apportion(parts: Mapping[str, float], total: int) -> dict[str, int]:
    """Round ``parts`` to integers that sum to ``total`` (largest remainder)."""
    floors = {k: int(v) for k, v in parts.items()}
    remainder = total - sum(floors.values())
    order = sorted(parts, key=lambda k: (-(parts[k] - floors[k]), k))
    for k in order[: max(remainder, 0)]:
        floors[k] += 1
    return floors


def _scorecard(
    steps: tuple[Step, ...],
    classes: tuple[Classification, ...],
    findings: tuple[Finding, ...],
    cycles: tuple[ValidationCycle, ...],
    ter: TerMeasure | None,
) -> Scorecard:
    flow_tok: dict[str, float] = {f.value: 0.0 for f in FlowState}
    flow_sec: dict[str, float] = {f.value: 0.0 for f in FlowState}
    act_tok: dict[str, float] = {a.value: 0.0 for a in ActivityClass} | {UNCERTAIN: 0.0}
    act_sec: dict[str, float] = dict.fromkeys(act_tok, 0.0)
    generated = 0
    agent_seconds = developer_seconds = 0.0
    for step, c in zip(steps, classes, strict=True):
        if step.kind is EventKind.PROMPT:
            developer_seconds += step.seconds
            continue
        agent_seconds += step.seconds
        generated += step.tokens
        # Uncertain waste is waste until verified (ADR 0006).
        wasted = c.avoidable_share + c.uncertain_share
        waste_flow = c.flow.value
        base_flow = FlowState.PROGRESSING.value if wasted else c.flow.value
        for amount, flow, act in (
            (step.tokens, flow_tok, act_tok),
            (step.seconds, flow_sec, act_sec),
        ):
            flow[waste_flow] += amount * wasted
            flow[base_flow] += amount * (1 - wasted)
            if c.activity_class is None:
                continue
            base = (
                ActivityClass.NECESSARY_NON_VALUE_ADDING
                if c.activity_class is ActivityClass.AVOIDABLE
                else c.activity_class
            )
            act[ActivityClass.AVOIDABLE.value] += amount * c.avoidable_share
            act[UNCERTAIN] += amount * c.uncertain_share
            act[base.value] += amount * (1 - c.avoidable_share - c.uncertain_share)
    pairs = list(zip(steps, classes, strict=True))
    context = sum(s.context_tokens for s in steps)
    runs = [s for s in steps if s.is_completion and s.shell is ShellIntent.VALIDATE]
    flow_tokens = apportion(flow_tok, generated)
    activity_tokens = apportion(act_tok, generated)
    waste = [f for f in findings if f.kind is FindingKind.WASTE]
    unsure = sum(f.uncertain for f in waste)

    def eff(values: Mapping[str, float], total: float) -> float | None:
        # Productive iteration is work, not waste: it counts toward flow.
        moving = (
            values[FlowState.PROGRESSING.value] + values[FlowState.RECOVERING.value]
        )
        return moving / total if total > 0 else None

    efficiency_tokens = eff({k: float(v) for k, v in flow_tokens.items()}, generated)
    efficiency_time = eff(flow_sec, agent_seconds)
    composite: Composite | None = None
    if efficiency_tokens is not None:
        components: list[tuple[str, float]] = [
            ("Flow efficiency (tokens)", efficiency_tokens)
        ]
        if efficiency_time is not None:
            components.append(("Flow efficiency (time)", efficiency_time))
        if ter is not None:
            components.append(("TER", ter.value))
        weight = 1.0 / len(components)
        composite = Composite(
            value=sum(v for _, v in components) * weight,
            components=tuple((n, v, weight) for n, v in components),
            formula="Unweighted mean of " + ", ".join(n for n, _ in components) + ".",
        )
    return Scorecard(
        generated_tokens=generated,
        context_tokens=context,
        agent_seconds=agent_seconds,
        developer_seconds=developer_seconds,
        flow_tokens=tuple((f, flow_tokens[f.value]) for f in FlowState),
        flow_seconds=tuple((f, flow_sec[f.value]) for f in FlowState),
        flow_efficiency_tokens=efficiency_tokens,
        flow_efficiency_time=efficiency_time,
        activity_tokens=tuple((k, activity_tokens[k]) for k in act_tok),
        activity_seconds=tuple((k, act_sec[k]) for k in act_sec),
        waste_tokens=activity_tokens[ActivityClass.AVOIDABLE.value]
        + activity_tokens[UNCERTAIN],
        waste_context_tokens=round(
            sum(
                s.context_tokens * (c.avoidable_share + c.uncertain_share)
                for s, c in pairs
            )
        ),
        waste_seconds=sum(
            s.seconds * (c.avoidable_share + c.uncertain_share) for s, c in pairs
        ),
        uncertain_waste_tokens=activity_tokens[UNCERTAIN],
        findings=len(waste),
        uncertain_findings=unsure,
        risks=len(findings) - len(waste),
        iterations=sum(c.verdict is CycleVerdict.ITERATION for c in cycles),
        rework_cycles=sum(c.verdict is CycleVerdict.REWORK for c in cycles),
        ter=ter,
        composite=composite,
        validation_runs=len(runs),
        validations_passed=sum(s.outcome is Outcome.PASSED for s in runs),
        validations_failed=sum(s.outcome is Outcome.FAILED for s in runs),
    )


def _value_stream(
    steps: tuple[Step, ...],
    classes: tuple[Classification, ...],
    findings: tuple[Finding, ...],
) -> tuple[StageSummary, ...]:
    stage_of = {s.event_id: s.stage for s in steps}
    touched: dict[Stage, list[str]] = {s: [] for s in STAGE_ORDER}
    for finding in findings:
        cited = finding.waste_events or finding.evidence
        for stage in dict.fromkeys(
            stage_of[e] for e in cited if stage_of[e] is not Stage.INTENT
        ):
            touched[stage].append(finding.id)
    out: list[StageSummary] = []
    for stage in STAGE_ORDER:
        members = [
            (s, c) for s, c in zip(steps, classes, strict=True) if s.stage is stage
        ]
        out.append(
            StageSummary(
                stage=stage,
                steps=sum(1 for s, _ in members if not s.is_completion),
                tokens=sum(s.tokens for s, _ in members),
                context_tokens=sum(s.context_tokens for s, _ in members),
                seconds=sum(s.seconds for s, _ in members),
                avoidable_tokens=round(
                    sum(s.tokens * c.avoidable_share for s, c in members)
                ),
                uncertain_tokens=round(
                    sum(s.tokens * c.uncertain_share for s, c in members)
                ),
                findings=tuple(touched[stage]),
            )
        )
    return tuple(out)


def analyse_steps(
    session_id: str | None,
    steps: tuple[Step, ...],
    *,
    ter: TerMeasure | None = None,
    registry: DetectorRegistry = DEFAULT_REGISTRY,
    intent: IntentTimeline | None = None,
    wip: WipReport | None = None,
    repository: RepositoryGrounding | None = None,
) -> LeanAnalysis:
    """Run every detector over ``steps`` and build the analysis.

    ``intent`` is the session's intent timeline (:func:`.intent.build_intent`);
    without one, the intent detectors have nothing to judge against. ``wip`` is
    the WIP the incremental fold counted; without it, WIP is recounted from the
    steps alone (:meth:`WipTracker.of_steps`). With ``repository`` (L3), the
    grounded detectors (:data:`.surface.GROUNDED_DETECTORS`) join the
    registry and every task's change surface is reported.
    """
    timeline = intent if intent is not None else IntentTimeline()
    if repository is not None:
        registry = registry.extended((*GROUNDED_DETECTORS, *EVIDENCE_DETECTORS))
    view = SessionView.of(steps, timeline, repository)
    findings = registry.run(view)
    cycles = validation_cycles(view)
    classes = _classify(steps, findings, cycles)
    graph = build_graph(
        session_id, steps, findings, {c.event_id: c.activity_class for c in classes}
    )
    surfaces = change_surfaces(view)
    usage: EvidenceUsage | None = None
    value: OutcomeValue | None = None
    grounded: GroundedEvidenceGraph | None = None
    if repository is not None:
        names = FileNames(repository)
        usage = evidence_usage(steps, repository, view.completion_of, names)
        value = outcome_value(
            steps, repository, surfaces, usage, view.completion_of, names
        )
        grounded = build_grounded_graph(session_id, steps, usage)
    return LeanAnalysis(
        session_id=session_id,
        events=len(steps),
        steps=steps,
        findings=findings,
        cycles=cycles,
        classifications=classes,
        value_stream=_value_stream(steps, classes, findings),
        scorecard=_scorecard(steps, classes, findings, cycles, ter),
        graph=graph,
        detectors=tuple(
            (d.id, d.waste.value, d.kind.value, d.confidence_rule) for d in registry
        ),
        intent=timeline,
        wip=WipTracker.of_steps(steps) if wip is None else wip,
        exploration=exploration_labels(view),
        repository=repository,
        surfaces=surfaces,
        profile=session_profile(steps),
        usage=usage,
        value=value,
        grounded_graph=grounded,
    )


_FILE_TOOLS = frozenset({ToolKind.FS_READ, ToolKind.FS_EDIT, ToolKind.FS_WRITE})


def session_profile(steps: Iterable[Step]) -> SessionProfile:
    """The languages of the files a session's file tool requests name
    (TER-STK-001): a summary of the folded steps, so live and batch agree."""
    return languages_of(
        FileTouch(step.event_id, step.tool_kind is not ToolKind.FS_READ, path)
        for step in steps
        if step.kind is EventKind.TOOL_REQUESTED and step.tool_kind in _FILE_TOOLS
        for path in step.paths
    )


class LeanAnalyser:
    """Incremental L2 analysis: add events as they arrive, ask for the analysis."""

    def __init__(self) -> None:
        self._log = StepLog()
        self._intent = IntentLog()
        self._wip = WipTracker()
        self._session_id: str | None = None

    def __len__(self) -> int:
        return len(self._log)

    def add(self, event: Event, tokens: int) -> bool:
        before = len(self._log)
        accepted = self._log.add(event, tokens)
        if len(self._log) > before:
            self._intent.add(event)
        if not accepted:
            return False
        if self._session_id is None:
            self._session_id = event.session_id
        self._wip.add(event, None if event.kind.is_lifecycle else self._log.last)
        return True

    def analysis(
        self,
        *,
        ter: TerMeasure | None = None,
        registry: DetectorRegistry = DEFAULT_REGISTRY,
        alignment: AlignmentScorer = LEXICAL_ALIGNMENT,
        intent_config: IntentConfig = DEFAULT_INTENT_CONFIG,
        repository: RepositoryGrounding | None = None,
    ) -> LeanAnalysis:
        steps = self._log.steps()
        timeline = build_intent(
            steps, self._intent.reads(), scorer=alignment, config=intent_config
        )
        return analyse_steps(
            self._session_id,
            steps,
            ter=ter,
            registry=registry,
            intent=timeline,
            wip=self._wip.report(),
            repository=repository,
        )


def explain(
    events: Iterable[Event],
    tokenizer: _Counter,
    *,
    ter: TerMeasure | None = None,
    registry: DetectorRegistry = DEFAULT_REGISTRY,
    alignment: AlignmentScorer = LEXICAL_ALIGNMENT,
    intent_config: IntentConfig = DEFAULT_INTENT_CONFIG,
    repository: RepositoryGrounding | None = None,
) -> LeanAnalysis:
    """Batch L2 analysis of a whole stream: the fold of :meth:`LeanAnalyser.add`."""
    analyser = LeanAnalyser()
    for event in events:
        analyser.add(event, tokenizer.count(event.text))
    return analyser.analysis(
        ter=ter,
        registry=registry,
        alignment=alignment,
        intent_config=intent_config,
        repository=repository,
    )
