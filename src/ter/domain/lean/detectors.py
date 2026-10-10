"""Waste detectors: small plugins behind one protocol.

A detector reads a :class:`SessionView` and yields :class:`~.model.Finding`
values. Each one is conservative on purpose (point 91): it reports only what
the event stream shows, cites every event it relies on (point 40), and gives
a confidence from an explicit rule, published as ``confidence_rule``. Below
:data:`~.model.UNCERTAIN_BELOW` a finding is reported as uncertain rather
than dropped or asserted (points 85, 86).

To add a detector, implement :class:`WasteDetector` and register it in a
:class:`DetectorRegistry` (point 197). ``DEFAULT_REGISTRY`` holds the L2 set.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
import posixpath
from pathlib import PurePosixPath
from typing import Protocol

from ..events import EventId, EventKind, ToolKind
from .grounding import RepositoryGrounding
from .intent import IntentRelation, IntentTimeline, SubjectBasis
from .model import (
    ActivityClass,
    CycleVerdict,
    ExplorationDriver,
    ExplorationLabel,
    Finding,
    FindingKind,
    LeanWaste,
    Outcome,
    Stage,
    Step,
    ValidationCycle,
)

__all__ = [
    "DECISION_NOVELTY",
    "DOC_EDITS_WORTH_A_CHECK",
    "REGENERATED_SHARE",
    "JUDGED_CONFIDENCE",
    "RESTATED_NOVELTY",
    "DEFAULT_REGISTRY",
    "ContextBand",
    "DetectorRegistry",
    "ExcessiveContext",
    "ExcessivePlanning",
    "FailedRoute",
    "FragmentedEdits",
    "IntentDrift",
    "InsufficientContext",
    "PrematureImplementation",
    "Regeneration",
    "RepeatedExploration",
    "RepeatedReasoning",
    "RepeatedToolCall",
    "ReworkCycle",
    "SessionView",
    "UnnecessaryHandoff",
    "UnearnedEscalation",
    "UnusedContext",
    "UnusedTraversal",
    "UnvalidatedImplementation",
    "WasteDetector",
    "exploration_labels",
    "validation_cycles",
]

_EXPLORE_KINDS = frozenset({ToolKind.FS_READ, ToolKind.FS_SEARCH})
_CHANGE_KINDS = frozenset({ToolKind.FS_EDIT, ToolKind.FS_WRITE})
_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt", ".adoc"})

#: A reasoning span adds a decision when more than this share of its content
#: words are new: in neither the prompt nor the earlier reasoning it is
#: compared with (TER-LEN-004). Repeated reasoning uses the tighter
#: RESTATED_NOVELTY, so a block with 20% to 25% new words neither restates
#: nor adds a decision.
DECISION_NOVELTY = 0.25
#: Repeated reasoning: at most this share of a block's key words may be new
#: (judged on real sessions, 10 Oct 2026: blocks repeating 80% or more of
#: earlier key words were waste, 76% to 79% were not).
RESTATED_NOVELTY = 0.20
#: Documentation-only edits reported without a check raise a finding only
#: from this many edits on (fewer were never waste on real sessions).
DOC_EDITS_WORTH_A_CHECK = 3
#: A whole-file rewrite is regeneration only when at least this share of the
#: new file repeats existing content (less is mostly new work).
REGENERATED_SHARE = 0.30
#: Confidence of a detector whose judged sample was 10 of 10 waste: the 95%
#: Wilson lower bound of that sample (first judged sample, 10 Oct 2026).
JUDGED_CONFIDENCE = 0.72


@dataclass(frozen=True)
class SessionView:
    """Steps plus the indexes detectors need, built once per analysis."""

    steps: tuple[Step, ...]
    completion_of: dict[int, Step] = field(default_factory=dict)
    by_id: dict[EventId, Step] = field(default_factory=dict)
    intent: IntentTimeline = field(default_factory=IntentTimeline)
    #: Repository evidence for the session (L3); ``None`` at L2.
    repository: RepositoryGrounding | None = None

    @classmethod
    def of(
        cls,
        steps: Sequence[Step],
        intent: IntentTimeline | None = None,
        repository: RepositoryGrounding | None = None,
    ) -> SessionView:
        completion_of = {
            s.request_index: s
            for s in steps
            if s.is_completion and s.request_index is not None
        }
        return cls(
            tuple(steps),
            completion_of,
            {s.event_id: s for s in steps},
            intent if intent is not None else IntentTimeline(),
            repository,
        )

    def requests(self) -> Iterator[Step]:
        return (s for s in self.steps if s.is_request)

    def pair(self, request: Step) -> tuple[Step, ...]:
        """A request and, when observed, its completion."""
        done = self.completion_of.get(request.index)
        return (request,) if done is None else (request, done)

    def segment_end(self, index: int) -> int:
        """Index of the last step before the next prompt (or of the last step)."""
        for step in self.steps[index + 1 :]:
            if step.kind is EventKind.PROMPT:
                return step.index - 1
        return len(self.steps) - 1


class WasteDetector(Protocol):
    """A plugin that finds one kind of waste or risk in a session."""

    @property
    def id(self) -> str: ...

    @property
    def waste(self) -> LeanWaste: ...

    @property
    def kind(self) -> FindingKind: ...

    @property
    def summary(self) -> str: ...

    @property
    def confidence_rule(self) -> str: ...

    def detect(self, view: SessionView) -> Iterable[Finding]: ...


def _finding(
    detector: WasteDetector,
    view: SessionView,
    *,
    confidence: float,
    title: str,
    explanation: str,
    evidence: Iterable[Step],
    waste: Iterable[Step] = (),
    share: float = 1.0,
    subject: str = "",
    activity_class: ActivityClass = ActivityClass.AVOIDABLE,
    anchor: Step | None = None,
    key: str = "",
) -> Finding:
    cited = sorted({s.index: s for s in evidence}.values(), key=lambda s: s.index)
    wasted = (
        sorted({s.index: s for s in waste}.values(), key=lambda s: s.index)
        if detector.kind is FindingKind.WASTE
        else []
    )
    # The id names the step the finding is about: the first wasted step, an
    # explicit anchor, or else the first cited step. ``key`` tells apart
    # findings that share an anchor (one step touching several files).
    if anchor is None:
        anchor = wasted[0] if wasted else cited[0]
    share = max(0.0, min(1.0, share))
    return Finding(
        id=f"{detector.id}:{anchor.event_id}" + (f":{key}" if key else ""),
        detector=detector.id,
        waste=detector.waste,
        kind=detector.kind,
        activity_class=activity_class,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        title=title,
        explanation=explanation,
        evidence=tuple(s.event_id for s in cited),
        waste_events=tuple(s.event_id for s in wasted),
        share=share if wasted else 0.0,
        subject=subject,
        tokens=round(sum(s.tokens for s in wasted) * share),
        context_tokens=round(sum(s.context_tokens for s in wasted) * share),
        seconds=round(sum(s.seconds for s in wasted) * share, 3),
    )


def _edits_between(view: SessionView, start: int, end: int) -> list[Step]:
    return [s for s in view.steps[start + 1 : end] if s.is_edit]


def _basename(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).name


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepeatedToolCall:
    """Point 19 (and 39): the same call again, with the same result."""

    id: str = "repeated_tool_call"
    waste: LeanWaste = LeanWaste.OVER_PROCESSING
    kind: FindingKind = FindingKind.WASTE
    summary: str = "A tool call repeated with unchanged input and unchanged output."
    confidence_rule: str = (
        "Within one prompt's turn: 0.90 when input and output are identical "
        "and nothing was edited in between; 0.85 for a validation re-run with "
        "no edit in between; 0.75 when edits happened in between but the "
        "output is still identical. 0.50 (uncertain) when an output was not "
        "observed. No finding when a new prompt arrived between the two calls "
        "(new information: 17 of 17 such repeats judged on real sessions were "
        "not waste) or when the output differs. File reads and "
        "searches are left to repeated_exploration; a validation re-run after "
        "edits is left to rework_cycle."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        last: dict[str, Step] = {}
        last_prompt = -1
        for step in view.steps:
            if step.kind is EventKind.PROMPT:
                last_prompt = step.index
            if not step.is_request:
                continue
            key = step.call_key
            if key is None or step.tool_kind in _EXPLORE_KINDS:
                continue
            earlier = last.get(key)
            last[key] = step
            if earlier is None:
                continue
            edited = bool(_edits_between(view, earlier.index, step.index))
            if step.is_validation and edited:
                continue
            if last_prompt > earlier.index:
                # A new prompt is new information. Calibrated on real
                # sessions: 13 such repeats on this project's transcripts and
                # 4 more judged on 10 Oct 2026 (judge-l2), none of them waste.
                continue
            before = view.completion_of.get(earlier.index)
            after = view.completion_of.get(step.index)
            if before is None or after is None:
                confidence = 0.5
                note = "One of the two results was not observed, so the output may have differed."
            elif before.output_hash != after.output_hash:
                continue
            elif step.is_validation:
                confidence = 0.85
                note = "No file was edited between the two runs, so the second could not tell the agent anything new."
            elif edited:
                confidence = 0.75
                note = "Files were edited in between, yet the result was identical."
            else:
                confidence = 0.9
                note = "Nothing changed in between and the result was identical."
            what = (
                "validation run" if step.is_validation else f"{step.native_name} call"
            )
            yield _finding(
                self,
                view,
                confidence=confidence,
                title=f"Repeated {what}",
                explanation=(
                    f"{step.native_name} was called again with the same input "
                    f"({_short(step.subject)}). {note}"
                ),
                evidence=(*view.pair(earlier), *view.pair(step)),
                waste=view.pair(step),
                subject=step.subject,
            )


@dataclass(frozen=True)
class RepeatedExploration:
    """Points 18, 28, 36: re-reading a file or re-running a search."""

    id: str = "repeated_exploration"
    waste: LeanWaste = LeanWaste.MOTION
    kind: FindingKind = FindingKind.WASTE
    summary: str = (
        "A file re-read, or a search re-run, that returned what the agent already had."
    )
    confidence_rule: str = (
        "0.85 when the same read or search (same arguments) returned identical "
        "output and the file was not edited in between; 0.55 (uncertain) when "
        "either output was not observed. A read after an edit of that file, a "
        "read of a different range, or a changed output is not a finding."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        last: dict[str, Step] = {}
        for step in view.requests():
            if step.tool_kind not in _EXPLORE_KINDS or step.call_key is None:
                continue
            earlier = last.get(step.call_key)
            last[step.call_key] = step
            if earlier is None:
                continue
            edits = _edits_between(view, earlier.index, step.index)
            if step.tool_kind is ToolKind.FS_READ:
                if any(set(e.paths) & set(step.paths) for e in edits):
                    continue
            elif edits:
                continue
            before = view.completion_of.get(earlier.index)
            after = view.completion_of.get(step.index)
            if before is None or after is None:
                confidence = 0.55
                note = "One of the two results was not observed."
            elif before.output_hash != after.output_hash:
                continue
            else:
                confidence = 0.85
                note = "The output was identical and the agent had not changed it."
            verb = "Re-read" if step.tool_kind is ToolKind.FS_READ else "Re-ran search"
            yield _finding(
                self,
                view,
                confidence=confidence,
                title=f"{verb} {_short(step.subject)}",
                explanation=(
                    f"{verb} {_short(step.subject)} {step.index - earlier.index} "
                    f"events after the first time. {note} Its "
                    f"{view.completion_of[step.index].context_tokens if step.index in view.completion_of else 0}"
                    " tokens of output re-entered the context."
                ),
                evidence=(*view.pair(earlier), *view.pair(step)),
                waste=view.pair(step),
                subject=step.subject,
            )


def validation_cycles(view: SessionView) -> tuple[ValidationCycle, ...]:
    """Pair every failed validation run with the next run of the same command.

    Only pairs with at least one edit in between are cycles: the edits were an
    attempt to fix the failure. The verdict is iteration when the next run
    passed or failed differently, and rework when it failed identically. Runs
    whose outcome cannot be read form no cycle.
    """
    runs = [
        (s, view.completion_of.get(s.index)) for s in view.requests() if s.is_validation
    ]
    cycles: list[ValidationCycle] = []
    for i, (run, result) in enumerate(runs):
        if result is None or result.outcome is not Outcome.FAILED:
            continue
        following = next(
            ((r, res) for r, res in runs[i + 1 :] if r.command == run.command), None
        )
        if following is None:
            continue
        next_run, next_result = following
        fixes = _edits_between(view, run.index, next_run.index)
        outcome = next_result.outcome if next_result is not None else None
        if (
            not fixes
            or next_result is None
            or outcome is None
            or outcome is Outcome.UNKNOWN
        ):
            continue
        if next_result.outcome is Outcome.PASSED:
            verdict, reason = CycleVerdict.ITERATION, "the next run passed"
        elif next_result.failure_signature != result.failure_signature:
            verdict, reason = CycleVerdict.ITERATION, "the next run failed differently"
        else:
            verdict, reason = CycleVerdict.REWORK, "the next run failed identically"
        cycles.append(
            ValidationCycle(
                failed_run=run.event_id,
                failure=result.event_id,
                fixes=tuple(f.event_id for f in fixes),
                next_run=next_run.event_id,
                next_result=next_result.event_id,
                next_outcome=outcome,
                verdict=verdict,
                command=run.command or "",
                reason=reason,
            )
        )
    return tuple(cycles)


@dataclass(frozen=True)
class ReworkCycle:
    """Points 26, 37: an edit-test-fail cycle that did not move the failure."""

    id: str = "rework_cycle"
    waste: LeanWaste = LeanWaste.REWORK
    kind: FindingKind = FindingKind.WASTE
    summary: str = (
        "Edits after a failing check that left the check failing the same way."
    )
    confidence_rule: str = (
        "Only cycles of the same validation command with edits in between. "
        "0.80 when the next run fails with the same failure signature (timings "
        "and addresses ignored); 0.90 when that is the second or later identical "
        "failure in a row. A cycle whose next run passes or fails differently "
        "is productive iteration and is not a finding."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        streak: dict[str, int] = {}
        for cycle in validation_cycles(view):
            if cycle.verdict is not CycleVerdict.REWORK:
                streak[cycle.command] = 0
                continue
            streak[cycle.command] = streak.get(cycle.command, 0) + 1
            confidence = 0.8 if streak[cycle.command] == 1 else 0.9
            fixes = [view.by_id[e] for e in cycle.fixes]
            fix_pairs = [s for f in fixes for s in view.pair(f)]
            rerun = view.by_id[cycle.next_run]
            ids = (cycle.failed_run, cycle.failure, cycle.next_run)
            yield _finding(
                self,
                view,
                confidence=confidence,
                title=f"Fix attempt did not change the failure of {_short(cycle.command)}",
                explanation=(
                    f"{len(fixes)} edit(s) to {_files(fixes)} after `{_short(cycle.command)}` "
                    "failed, then the same command failed with the same failure. The "
                    "attempt was rework: it did not move the check."
                ),
                evidence=(*(view.by_id[e] for e in ids), *fix_pairs, *view.pair(rerun)),
                waste=(*fix_pairs, *view.pair(rerun)),
                subject=cycle.command,
            )


@dataclass(frozen=True)
class UnvalidatedImplementation:
    """Point 25: changes reported without a validation run after them."""

    id: str = "unvalidated_implementation"
    waste: LeanWaste = LeanWaste.DEFECTS
    kind: FindingKind = FindingKind.RISK
    summary: str = "Edits the agent reported on without running a check after them, or a last check that failed."
    confidence_rule: str = (
        "Judged per prompt, once the agent has responded after its edits. A "
        "check is a validation run or any shell line that runs a named check "
        "tool, even beside a change (sed -i … && pytest). 0.85 when no check "
        "ran in the whole session; 0.75 when checks ran earlier but not after "
        "these edits; 0.50 (uncertain) when every unvalidated edit is "
        "documentation and there are at least 3 of them; fewer than 3 "
        "documentation edits: no finding. 0.85 when the last validation before the response "
        "failed."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        any_validation = any(s.checks for s in view.steps)
        start = 0
        while start < len(view.steps):
            end = view.segment_end(start)
            yield from self._segment(view, view.steps[start : end + 1], any_validation)
            start = end + 1

    def _segment(
        self, view: SessionView, steps: Sequence[Step], any_validation: bool
    ) -> Iterator[Finding]:
        pending: list[Step] = []
        last_run: Step | None = None
        for step in steps:
            if step.is_edit:
                pending.append(step)
            elif step.checks:
                # A check chained after a change (``sed -i … && pytest``)
                # validates the edits too; StepLog reads its outcome, so a
                # failing chained check is reported below.
                pending = []
                last_run = step
        responses = [s for s in steps if s.kind is EventKind.RESPONSE]
        final = responses[-1] if responses else None
        if final is None:
            return
        if pending and final.index > pending[-1].index:
            docs = all(
                PurePosixPath(p).suffix.lower() in _DOC_SUFFIXES
                for e in pending
                for p in e.paths
            ) and any(e.paths for e in pending)
            # Judged on real sessions (10 Oct 2026): 8 of 8 findings for one
            # or two documentation edits were not waste, so they raise none.
            few_docs = docs and len(pending) < DOC_EDITS_WORTH_A_CHECK
            if docs:
                confidence, why = (
                    0.5,
                    "Only documentation changed, which may not need a check.",
                )
            elif any_validation:
                confidence, why = (
                    0.75,
                    "Checks ran earlier in the session, but not after these edits.",
                )
            else:
                confidence, why = 0.85, "No check ran anywhere in the session."
            if not few_docs:
                yield _finding(
                    self,
                    view,
                    confidence=confidence,
                    title=f"{len(pending)} edit(s) reported without validation",
                    explanation=(
                        f"The agent edited {_files(pending)} and then responded "
                        f"without running tests, a type check or the code. {why}"
                    ),
                    evidence=(*(s for e in pending for s in view.pair(e)), final),
                    subject=_files(pending),
                )
        if last_run is not None and final.index > last_run.index:
            result = view.completion_of.get(last_run.index)
            if result is not None and result.outcome is Outcome.FAILED:
                later_edits = [e for e in pending if e.index > last_run.index]
                if not later_edits:
                    yield _finding(
                        self,
                        view,
                        confidence=0.85,
                        title="Responded after a failing check",
                        explanation=(
                            f"The last validation, `{_short(last_run.command or '')}`, "
                            "failed and the agent responded without fixing or re-running it."
                        ),
                        evidence=(last_run, result, final),
                        subject=last_run.command or "",
                    )


@dataclass(frozen=True)
class PrematureImplementation:
    """Points 22, 23: changing code before looking at it."""

    id: str = "premature_implementation"
    waste: LeanWaste = LeanWaste.DEFECTS
    kind: FindingKind = FindingKind.RISK
    summary: str = (
        "An edit to a file the agent had not read, written or seen in a search."
    )
    confidence_rule: str = (
        "0.75 for an in-place edit of a file never read, written, named in an "
        "earlier shell command (cat, sed -n, head, grep …) or named in an "
        "earlier search or shell output; 0.45 (uncertain) for writing a new "
        "file before any exploration at all in the session. One finding per "
        "file."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        known: set[str] = set()
        seen_names: set[str] = set()
        explored = False
        flagged: set[str] = set()
        prompt: Step | None = None
        for step in view.steps:
            if step.kind is EventKind.PROMPT:
                prompt = step
            if step.is_completion and step.tool_kind is not ToolKind.FS_READ:
                seen_names |= step.words
            if not step.is_request:
                continue
            if step.stage is Stage.EXPLORE:
                explored = True
            if step.tool_kind is ToolKind.EXEC_SHELL:
                # A file read through the shell (``cat f``, ``sed -n 1,80p f``)
                # is named in the command, not in its output. On this
                # project's own transcripts the one confident finding was
                # such a file, read with ``cat`` before the edit.
                seen_names |= step.words
            if step.is_edit:
                for path in step.paths:
                    base = _basename(path).lower()
                    if path in known or path in flagged or base in seen_names:
                        continue
                    if step.tool_kind is ToolKind.FS_EDIT:
                        confidence = 0.75
                        why = "It was not read, written or named in any earlier output."
                    elif not explored:
                        confidence = 0.45
                        why = "Nothing in the repository had been explored yet; the file may be new."
                    else:
                        continue
                    flagged.add(path)
                    cited = [step] if prompt is None else [prompt, step]
                    yield _finding(
                        self,
                        view,
                        confidence=confidence,
                        title=f"Changed {_short(path)} before reading it",
                        explanation=f"{step.native_name} changed {path} with no evidence collected about it first. {why}",
                        evidence=cited,
                        subject=path,
                        anchor=step,
                        key=path if len(step.paths) > 1 else "",
                    )
            if step.tool_kind in (
                ToolKind.FS_READ,
                ToolKind.FS_EDIT,
                ToolKind.FS_WRITE,
            ):
                known.update(step.paths)


@dataclass(frozen=True)
class ExcessivePlanning:
    """Point 24 (and 8): planning that does not transition into action."""

    id: str = "excessive_planning"
    waste: LeanWaste = LeanWaste.OVER_PROCESSING
    kind: FindingKind = FindingKind.WASTE
    summary: str = "A run of planning steps (reasoning, to-do updates) with no action between them."
    confidence_rule: str = (
        "A run of at least 4 consecutive planning steps (reasoning blocks and "
        "plan/to-do updates) with no exploration, edit, check or response in "
        "between. Confidence 0.55 + 0.05 per step, capped at 0.90. The first two "
        "steps of the run are counted as necessary. A later step is waste only "
        "when it adds no decision: a reasoning block with at most 25% content "
        "words new to the prompt and the run so far, or a to-do update identical "
        "to an earlier one in the run. A step that adds a decision is never "
        "waste, and a run with no such restating step is not a finding."
    )
    minimum: int = 4

    def detect(self, view: SessionView) -> Iterable[Finding]:
        run: list[Step] = []
        prompt: frozenset[str] = frozenset()
        for step in (*view.steps, None):
            if step is not None and (
                step.is_completion and step.tool_kind is ToolKind.PLAN
            ):
                continue
            if step is not None and step.is_generated and step.stage is Stage.PLAN:
                run.append(step)
                continue
            if len(run) >= self.minimum:
                extra = _restating_steps(run, prompt)
                if extra:
                    yield _finding(
                        self,
                        view,
                        confidence=min(0.9, 0.55 + 0.05 * len(run)),
                        title=f"{len(run)} planning steps without acting",
                        explanation=(
                            f"{len(run)} reasoning or to-do steps in a row before the agent "
                            f"explored, edited, checked or answered anything; {len(extra)} "
                            "of them restated the plan without a new decision."
                        ),
                        evidence=[s for e in run for s in view.pair(e)],
                        waste=[s for e in extra for s in view.pair(e)],
                        subject=f"{len(run)} planning steps",
                    )
            run = []
            if step is not None and step.kind is EventKind.PROMPT:
                prompt = step.words


def _novelty(step: Step, known: frozenset[str]) -> float:
    """Share of a step's content words that are not in ``known``."""
    if not step.words:
        return 0.0
    return len(step.words - known) / len(step.words)


def _restating_steps(run: Sequence[Step], prompt: frozenset[str]) -> list[Step]:
    """Planning steps after the second that add no decision (TER-LEN-004)."""
    known = set(prompt)
    keys: set[str] = set()
    out: list[Step] = []
    for position, step in enumerate(run):
        if step.kind is EventKind.REASONING:
            restated = _novelty(step, frozenset(known)) <= DECISION_NOVELTY
        else:
            restated = step.call_key is not None and step.call_key in keys
        if position >= 2 and restated:
            out.append(step)
        known |= step.words
        if step.call_key is not None:
            keys.add(step.call_key)
    return out


@dataclass(frozen=True)
class FragmentedEdits:
    """Points 27, 28: one coherent change split into many round trips (motion).

    A round trip is one model turn: the agent waits for a result before it
    sends the next call. Edits sent together in one turn (parallel tool
    calls, all requested before the first result arrives) are one round trip,
    however many calls they take. Claude Code's Edit tool changes one string
    per call, so several calls are how a multi-hunk change is made; only the
    turns between them are overhead. Calibrated on this project's own
    transcripts: all 4 confident findings there were 3 or 4 Edit calls sent
    in one turn, which is the batching this detector recommends.
    """

    id: str = "fragmented_edits"
    waste: LeanWaste = LeanWaste.MOTION
    kind: FindingKind = FindingKind.WASTE
    summary: str = "Edits to one file spread over three or more round trips in a row."
    confidence_rule: str = (
        "Consecutive edit calls to one file, with only reasoning and the "
        "results of those edits in between, spread over at least 3 round "
        "trips (a round trip ends when a result arrives; calls sent together "
        "before any result are one round trip). 0.70 for 3 round trips, plus "
        "0.05 per extra, capped at 0.85. Only the round-trip overhead (the "
        "results of edits after the first round trip) is counted as waste, "
        "never the change itself."
    )
    minimum: int = 3

    def detect(self, view: SessionView) -> Iterable[Finding]:
        run: list[Step] = []
        for step in (*view.requests(), None):
            if (
                step is not None
                and _single_file_edit(step)
                and run
                and run[-1].paths == step.paths
                and _only_run_results_between(view, run, step)
            ):
                run.append(step)
                continue
            if run:
                yield from self._judge(view, run)
            run = [step] if step is not None and _single_file_edit(step) else []

    def _judge(self, view: SessionView, run: list[Step]) -> Iterable[Finding]:
        trips = _round_trips(view, run)
        if len(trips) < self.minimum:
            return
        path = run[0].paths[0]
        overhead = [
            view.completion_of[s.index]
            for trip in trips[1:]
            for s in trip
            if s.index in view.completion_of
        ]
        yield _finding(
            self,
            view,
            confidence=min(0.85, 0.7 + 0.05 * (len(trips) - self.minimum)),
            title=f"{len(run)} edits to {_short(path)} over {len(trips)} round trips",
            explanation=(
                f"{len(run)} consecutive edits to {path} took {len(trips)} round "
                "trips, each waiting for the previous result. Edits planned "
                "together can be sent in one turn as parallel Edit calls."
            ),
            evidence=[s for e in run for s in view.pair(e)],
            waste=overhead,
            subject=path,
        )


def _single_file_edit(step: Step) -> bool:
    return step.tool_kind is ToolKind.FS_EDIT and len(step.paths) == 1


def _only_run_results_between(view: SessionView, run: list[Step], b: Step) -> bool:
    """Only reasoning and results of the run's own edits since its last edit."""
    mine = {s.index for s in run}
    return all(
        s.kind is EventKind.REASONING or (s.is_completion and s.request_index in mine)
        for s in view.steps[run[-1].index + 1 : b.index]
    )


def _round_trips(view: SessionView, run: list[Step]) -> list[list[Step]]:
    """The run split into model turns: a result between two calls ends a turn."""
    trips: list[list[Step]] = [[run[0]]]
    for before, step in zip(run, run[1:], strict=False):
        if any(s.is_completion for s in view.steps[before.index + 1 : step.index]):
            trips.append([step])
        else:
            trips[-1].append(step)
    return trips


@dataclass(frozen=True)
class UnusedContext:
    """Points 21, 34, 35: file contents read and never used again."""

    id: str = "unused_context"
    waste: LeanWaste = LeanWaste.INVENTORY
    kind: FindingKind = FindingKind.WASTE
    summary: str = "A file read whose name and definitions never appear in anything the agent did afterwards."
    confidence_rule: str = (
        "Judged only once the agent has responded after the read. A read counts "
        "as used when a later reasoning, response or tool call names the file, "
        "or names something the file defines, or edits it. Otherwise 0.72. "
        "Calibrated on real sessions: 10 of 10 judged findings were waste "
        "(10 Oct 2026), so it counts at 0.72, the 95% lower bound of that "
        "sample. Reading to rule something out is legitimate, so a larger "
        "sample or repository evidence (L3) may lower it again."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        last_response = max(
            (s.index for s in view.steps if s.kind is EventKind.RESPONSE), default=-1
        )
        judged: set[str] = set()
        for step in view.steps:
            if not (
                step.is_completion and step.tool_kind is ToolKind.FS_READ and step.paths
            ):
                continue
            path = step.paths[0]
            if path in judged or step.index > last_response:
                continue
            judged.add(path)
            if self._used(view, step, path):
                continue
            request = (
                view.steps[step.request_index]
                if step.request_index is not None
                else step
            )
            yield _finding(
                self,
                view,
                confidence=JUDGED_CONFIDENCE,
                title=f"Read {_short(path)} and never used it",
                explanation=(
                    f"Nothing the agent did after reading {path} names the file"
                    + (
                        f" or any of the {len(step.identifiers)} name(s) it defines"
                        if step.identifiers
                        else ""
                    )
                    + f". Its {step.context_tokens} tokens stayed in the context as inventory."
                ),
                evidence=(request, step),
                waste=(request, step),
                subject=path,
            )

    @staticmethod
    def _used(view: SessionView, read: Step, path: str) -> bool:
        base = _basename(path).lower()
        stem = base.rsplit(".", 1)[0]
        names = {base} | ({stem} if len(stem) >= 4 else set())
        names |= {i.lower() for i in read.identifiers}
        for later in view.steps[read.index + 1 :]:
            if (
                later.is_request
                and path in later.paths
                and later.tool_kind is not ToolKind.FS_READ
            ):
                return True
            if (
                later.is_generated
                and later.tool_kind is not ToolKind.FS_READ
                and names & later.words
            ):
                return True
        return False


@dataclass(frozen=True)
class UnnecessaryHandoff:
    """Point 29 (and 30): delegating work, then doing it anyway."""

    id: str = "unnecessary_handoff"
    waste: LeanWaste = LeanWaste.HANDOFFS
    kind: FindingKind = FindingKind.WASTE
    summary: str = "A subagent handoff whose task the agent then did itself."
    confidence_rule: str = (
        "A later tool call by the agent itself shares at least 3 content words "
        "with the handoff's task and covers at least half of the task's words. "
        "Confidence 0.45 + 0.40 × coverage, at least 0.72 and capped at 0.85. "
        "Calibrated on real sessions: 10 of 10 judged findings were waste "
        "(10 Oct 2026); 0.72 is the 95% lower bound of that sample. The "
        "handoff (and its waiting time) is the waste; the agent's own call is "
        "kept."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        for handoff in view.requests():
            if handoff.tool_kind is not ToolKind.AGENT_HANDOFF or not handoff.words:
                continue
            best: tuple[float, Step] | None = None
            for later in view.steps[handoff.index + 1 :]:
                if not later.is_request or later.tool_kind is ToolKind.AGENT_HANDOFF:
                    continue
                # Coverage of the delegated task, not overlap with the
                # smaller set: a short command (``sed -n … analysis.py``, a
                # branch merge) always sits mostly inside a long brief. On
                # this project's own transcripts the smaller-set overlap
                # flagged all 9 handoffs of an orchestrator that reviewed and
                # merged its parallel workers' output (5 confident); none was
                # redone.
                shared = handoff.words & later.words
                score = len(shared) / len(handoff.words)
                if (
                    len(shared) >= 3
                    and score >= 0.5
                    and (best is None or score > best[0])
                ):
                    best = (score, later)
            if best is None:
                continue
            score, own = best
            yield _finding(
                self,
                view,
                confidence=min(0.85, max(JUDGED_CONFIDENCE, 0.45 + 0.4 * score)),
                title=f"Delegated {_short(handoff.subject)}, then did it directly",
                explanation=(
                    f"The agent handed '{_short(handoff.subject)}' to a subagent, then ran "
                    f"{own.native_name} on the same subject ({_short(own.subject)}), sharing "
                    f"{len(handoff.words & own.words)} key words. The handoff added a round "
                    "trip the agent did not rely on."
                ),
                evidence=(*view.pair(handoff), *view.pair(own)),
                waste=view.pair(handoff),
                subject=handoff.subject,
            )


@dataclass(frozen=True)
class RepeatedReasoning:
    """Point 17: reasoning that restates earlier reasoning without new decisions."""

    id: str = "repeated_reasoning"
    waste: LeanWaste = LeanWaste.OVER_PROCESSING
    kind: FindingKind = FindingKind.WASTE
    summary: str = "A reasoning block that restates an earlier one for the same prompt."
    confidence_rule: str = (
        "Same prompt and no edit in between. The later block shares at least 3 "
        "content words with the earlier one, at most 20% of its content words "
        "are new (in neither the earlier block nor the prompt), and none of its "
        "new words comes from a tool result observed since the earlier block "
        "(that would be new evidence). Confidence 0.85 − "
        "novelty, minus 0.10 when the later block has fewer than 6 content words. "
        "The restated share (1 − novelty) of the later block is waste."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        segment: list[Step] = []
        prompt: frozenset[str] = frozenset()
        # Words each tool result showed, by step index: new evidence.
        observed: list[tuple[int, frozenset[str]]] = []
        for step in view.steps:
            if step.kind is EventKind.PROMPT:
                segment, prompt, observed = [], step.words, []
                continue
            if step.is_edit:
                segment = []
                continue
            if step.is_completion and step.tool_kind is not ToolKind.PLAN:
                observed.append((step.index, step.words))
                continue
            if step.kind is not EventKind.REASONING or len(step.words) < 4:
                continue
            best: tuple[float, Step] | None = None
            for earlier in segment:
                if len(earlier.words & step.words) < 3:
                    continue
                new = step.words - earlier.words - prompt
                novelty = len(new) / len(step.words)
                if novelty > RESTATED_NOVELTY:
                    continue
                if any(new & words for i, words in observed if i > earlier.index):
                    # It names something a tool showed since: new evidence.
                    continue
                if best is None or novelty < best[0]:
                    best = (novelty, earlier)
            segment.append(step)
            if best is None:
                continue
            novelty, earlier = best
            yield _finding(
                self,
                view,
                confidence=0.85 - novelty - (0.1 if len(step.words) < 6 else 0.0),
                title="Reasoning restated",
                explanation=(
                    f"This reasoning restates reasoning from {step.index - earlier.index} "
                    f"events earlier: {1 - novelty:.0%} of its key words were already there "
                    "or in the prompt, and nothing was edited in between."
                ),
                evidence=(earlier, step),
                waste=(step,),
                share=1 - novelty,
                subject="reasoning",
            )


@dataclass(frozen=True)
class Regeneration:
    """Point 20: rewriting work that already existed."""

    id: str = "regeneration"
    waste: LeanWaste = LeanWaste.OVERPRODUCTION
    kind: FindingKind = FindingKind.WASTE
    summary: str = "A whole-file write that mostly regenerates content already there."
    confidence_rule: str = (
        "A whole-file write of a file the agent had itself written, keeping at "
        "least 80% of its lines: 0.80. A whole-file write of a file it had read "
        "(3+ lines), keeping at least 60% of them: 0.60 (uncertain; small files "
        "are often rewritten on purpose). No finding when under 30% of the new "
        "file repeats existing content: that rewrite is mostly new work. The "
        "retained share of the write is waste."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        written: dict[str, Step] = {}
        read: dict[str, Step] = {}
        for step in view.steps:
            if step.is_completion and step.tool_kind is ToolKind.FS_READ and step.paths:
                read[step.paths[0]] = step
            if not (
                step.is_request and step.tool_kind is ToolKind.FS_WRITE and step.paths
            ):
                continue
            path = step.paths[0]
            new = step.lines
            prior_write = written.get(path)
            prior_read = read.get(path)
            written[path] = step
            if not new:
                continue
            if prior_write is not None and prior_write.lines:
                if _repeated(prior_write, step) < REGENERATED_SHARE:
                    continue
                kept = len(prior_write.lines & new) / len(prior_write.lines)
                if kept >= 0.8:
                    yield self._found(
                        view, step, prior_write, kept, 0.8, "it had written itself"
                    )
                continue
            if prior_read is not None and len(prior_read.lines) >= 3:
                kept = len(prior_read.lines & new) / len(prior_read.lines)
                if kept >= 0.6 and _repeated(prior_read, step) >= REGENERATED_SHARE:
                    yield self._found(
                        view, step, prior_read, kept, 0.6, "it had just read"
                    )

    def _found(
        self,
        view: SessionView,
        step: Step,
        prior: Step,
        kept: float,
        confidence: float,
        what: str,
    ) -> Finding:
        path = step.paths[0]
        share = _repeated(prior, step)
        return _finding(
            self,
            view,
            confidence=confidence,
            title=f"Rewrote {_short(path)} in full",
            explanation=(
                f"The agent rewrote {path}, a file {what}, keeping {kept:.0%} of its "
                f"lines; {share:.0%} of the new file repeats existing content. An "
                "in-place edit would have produced only the change."
            ),
            evidence=(prior, step),
            waste=(step,),
            share=share,
            subject=path,
        )


def _repeated(prior: Step, step: Step) -> float:
    """The share of a write's lines that were already in ``prior``."""
    return len(prior.lines & step.lines) / len(step.lines)


@dataclass(frozen=True)
class IntentDrift:
    """Point 48 (TER-ITN-003): a change departing from the intent in force.

    Only edits and writes are judged: they are what moves the software away
    from the requested outcome. A departure that follows a recorded intent
    change is measured against the new intent, so it is drift only if it
    departs from that one too.
    """

    id: str = "intent_drift"
    waste: LeanWaste = LeanWaste.OVERPRODUCTION
    kind: FindingKind = FindingKind.WASTE
    summary: str = (
        "An edit or write about something the current intent does not ask for, "
        "with no intent change recorded."
    )
    confidence_rule: str = (
        "An edit or write whose alignment to the intent in force is below the "
        "drift band (default 0.25), against an intent of at least 3 key terms: "
        "0.85 when the names it defines continue a goal the developer dropped, "
        "or the agent's own reasoning or narration since the prompt called it "
        "additional ('also', 'while I'm at it') and names it; 0.55 (uncertain) "
        "when it only defines new names the intent does not mention, or defines "
        "no names and only its added words (at least 3) depart. "
        "In a file the session created (first touched by a write whose "
        "result was observed and did not report an existing file or an "
        "error, and not a repository file at the start when grounded), only "
        "the 0.85 cases are findings. "
        "Before any prompt, or against a shorter intent: no finding."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        timeline = view.intent
        band = timeline.config.drift_below
        record = timeline.record
        created = _created_files(view)
        for step in view.requests():
            if not step.is_edit:
                continue
            a = timeline.alignment_of(step.event_id)
            if a is None or a.score is None or a.revision is None:
                continue
            if a.score >= band:
                continue
            intent = record.revision(a.revision)
            if len(intent.terms) < _MIN_INTENT_TERMS:
                continue
            opened = intent.event_id
            dropped = frozenset(
                t
                for r in record.revisions[: a.revision]
                if r.relation in (IntentRelation.CHANGED, IntentRelation.ACKNOWLEDGED)
                for t in r.abandoned
            )
            announced = [
                s
                for s in view.steps[view.by_id[opened].index + 1 : step.index]
                if (x := timeline.alignment_of(s.event_id)) is not None
                and x.extra
                and not s.is_request
                and x.subject & a.subject
            ]
            evidence: list[Step] = [view.by_id[opened], *announced, *view.pair(step)]
            # Judged on a real greenfield session (10 Oct 2026): building a
            # file the session created defines names the intent never mentions
            # (132 findings) and adds words it never uses (46 of 55 judged
            # not waste), so only a dropped goal or announced extra work
            # counts there.
            in_created = bool(step.paths) and created.issuperset(
                _file_key(p, view.repository) for p in step.paths
            )
            what = ", ".join(a.names) if a.names else ", ".join(sorted(a.subject)[:5])
            if a.basis is SubjectBasis.NAMES:
                if a.subject & dropped:
                    confidence = 0.85
                    why = "It continues a goal the developer dropped."
                elif announced:
                    confidence = 0.85
                    why = "The agent itself called it additional work."
                elif in_created:
                    continue
                else:
                    # Calibrated on real sessions (9 Oct 2026): new names alone
                    # were 17 of 17 false positives (the requested new module,
                    # helper scripts), since implementing anything defines names.
                    confidence = 0.55
                    why = (
                        "The intent does not mention what it defines, but new "
                        "code always defines new names, so this may be the "
                        "requested change."
                    )
            else:
                if in_created or len(a.subject) < _MIN_DRIFT_WORDS:
                    continue
                confidence = 0.55
                why = (
                    "It defines no names, so this rests on the words it adds "
                    "and may be a necessary detail."
                )
            yield _finding(
                self,
                view,
                confidence=confidence,
                title=f"Edit departs from the intent: {_short(what)}",
                explanation=(
                    f"{step.native_name} of {_files([step])} is about {what}, which "
                    f"scores {a.score:.2f} against intent revision {a.revision} "
                    f"(below {band:.2f}), and no intent change was recorded. {why}"
                ),
                evidence=evidence,
                waste=view.pair(step),
                subject=what,
            )


_MIN_INTENT_TERMS = 3
_MIN_DRIFT_WORDS = 3


def _created_files(view: SessionView) -> frozenset[str]:
    """Files the session created: their first touch is a write that succeeded.

    A refused write changed nothing, and one with no observed result may not
    have run, so neither is a touch. A write whose result says it replaced an
    existing file did not create it. Paths are compared in
    one spelling: the repository path when the session is grounded, else the
    normalised path. A grounded file that existed at the start is never
    created, whatever touched it first."""
    g = view.repository
    seen: set[str] = set()
    created: set[str] = set()
    for step in view.requests():
        done = view.completion_of.get(step.index)
        if done is None or done.tool_failed:
            continue
        for path in step.paths:
            key = _file_key(path, g)
            if key in seen:
                continue
            seen.add(key)
            if (
                step.tool_kind is ToolKind.FS_WRITE
                and done.write_created is not False
                and (g is None or key not in g.files)
            ):
                created.add(key)
    return frozenset(created)


def _file_key(path: str, g: RepositoryGrounding | None) -> str:
    """One spelling of a session path: its repository path when known."""
    if g is not None:
        known = g.repository_path(path)
        if known is not None:
            return known
    return posixpath.normpath(path.replace("\\", "/"))


# ---------------------------------------------------------------------------
# Context band (TER-DET-003), traversal motion (TER-DET-007), failed routes
# (TER-DET-008) and exploration drivers (TER-DET-009)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextBand:
    """The configured band of context to acquire before a change (points 21, 22).

    Context is counted structurally, never in tokens: one *item* per distinct
    file read, search, fetch, handoff or exploring shell command in the task
    (from the prompt in force). A change is the set of files the task edits
    or writes. The band runs from ``min_per_file`` items per file edited in
    place to ``per_file`` items per file changed plus ``slack``.
    """

    min_per_file: int = 1
    per_file: int = 3
    slack: int = 3

    def upper(self, changed: int) -> int:
        return self.per_file * changed + self.slack

    def describe(self) -> str:
        return (
            f"at least {self.min_per_file} item(s) per file edited in place, at most "
            f"{self.per_file} per file changed plus {self.slack}"
        )


@dataclass(frozen=True)
class _Change:
    """One task's change: its prompt, context items and edits, in order."""

    prompt: Step | None
    #: Context items (the first request of each distinct target), in order.
    items: tuple[Step, ...]
    #: Edit and write requests, in order.
    edits: tuple[Step, ...]

    @property
    def changed(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p for e in self.edits for p in e.paths))

    def items_before(self, index: int) -> list[Step]:
        return [i for i in self.items if i.index < index]


def _item_key(step: Step) -> str | None:
    """What makes a context item distinct, or None when the step is not one."""
    if not step.is_request or step.stage is not Stage.EXPLORE:
        return None
    if step.tool_kind is ToolKind.FS_READ and step.paths:
        return "read\0" + step.paths[0]
    return step.call_key


def _changes(view: SessionView) -> Iterator[_Change]:
    """Every task (prompt segment) that edits or writes at least one file."""
    prompt: Step | None = None
    items: dict[str, Step] = {}
    edits: list[Step] = []
    for step in (*view.steps, None):
        if step is None or step.kind is EventKind.PROMPT:
            if edits:
                yield _Change(prompt, tuple(items.values()), tuple(edits))
            prompt, items, edits = step, {}, []
            continue
        if step.is_edit:
            edits.append(step)
            continue
        key = _item_key(step)
        if key is not None and key not in items:
            items[key] = step


@dataclass(frozen=True)
class ExcessiveContext:
    """Point 21: more context acquired before a change than its band allows."""

    id: str = "excessive_context"
    waste: LeanWaste = LeanWaste.INVENTORY
    kind: FindingKind = FindingKind.WASTE
    summary: str = "More files, searches and fetches before the first edit than the change's band allows."
    band: ContextBand = ContextBand()
    confidence_rule: str = (
        "Per task (prompt), counting distinct context items (file reads, "
        "searches, fetches, handoffs, exploring shell commands) before the first "
        "edit: a finding when they exceed per_file (3) × files the task changes "
        "+ slack (3). Items past the band that are not reads of a changed file "
        "are the waste. Confidence 0.72. Calibrated on real sessions: 10 of "
        "10 judged findings were waste (10 Oct 2026); 0.72 is the 95% lower "
        "bound of that sample."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        for change in _changes(view):
            first = change.edits[0]
            before = change.items_before(first.index)
            changed = change.changed
            upper = self.band.upper(len(changed))
            if len(before) <= upper:
                continue
            excess = [
                s
                for s in before[upper:]
                if not (s.tool_kind is ToolKind.FS_READ and set(s.paths) & set(changed))
            ]
            over = len(before) - upper
            cited = [] if change.prompt is None else [change.prompt]
            yield _finding(
                self,
                view,
                confidence=JUDGED_CONFIDENCE,
                title=f"{len(before)} context items before changing {len(changed)} file(s)",
                explanation=(
                    f"The task acquired {len(before)} distinct context items before its "
                    f"first edit, for a change to {_files(change.edits)}; the band allows "
                    f"{upper} ({self.band.describe()}). The {over} item(s) past the band "
                    "were carried as inventory. Reading widely can be justified; "
                    "repository evidence (L3) decides."
                ),
                evidence=(
                    *cited,
                    *(s for i in before for s in view.pair(i)),
                    *view.pair(first),
                ),
                waste=[s for i in excess for s in view.pair(i)],
                subject=_files(change.edits),
                anchor=first,
            )


@dataclass(frozen=True)
class InsufficientContext:
    """Point 22: a change made with less context than its band requires."""

    id: str = "insufficient_context"
    waste: LeanWaste = LeanWaste.DEFECTS
    kind: FindingKind = FindingKind.RISK
    summary: str = (
        "Files edited in place with fewer context items in the task than files edited."
    )
    band: ContextBand = ContextBand()
    confidence_rule: str = (
        "Per task (prompt): when the task first edits its n-th distinct file in "
        "place (Edit, not a new-file Write), it must have acquired at least "
        "min_per_file (1) × n distinct context items (reads, searches, fetches, "
        "handoffs, exploring shell). The first edit below that bound is the "
        "finding: 0.70 when that file was never read or written earlier in the "
        "session; 0.55 (uncertain) when it was, since context carried from an "
        "earlier task may suffice. A risk: it claims no cost."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        changes = {c.edits[0].index: c for c in _changes(view)}
        known: set[str] = set()
        current: _Change | None = None
        in_place: list[str] = []
        flagged = False
        for step in view.steps:
            if step.kind is EventKind.PROMPT:
                current, in_place, flagged = None, [], False
            if step.is_edit and current is None:
                current = changes.get(step.index)
            if (
                current is not None
                and not flagged
                and step.is_request
                and step.tool_kind is ToolKind.FS_EDIT
            ):
                new = [p for p in step.paths if p not in in_place]
                in_place.extend(new)
                acquired = current.items_before(step.index)
                need = self.band.min_per_file * len(in_place)
                if new and len(acquired) < need:
                    flagged = True
                    yield self._found(
                        view, current, step, new[0], acquired, need, known
                    )
            if step.is_request and step.tool_kind in (
                ToolKind.FS_READ,
                ToolKind.FS_EDIT,
                ToolKind.FS_WRITE,
            ):
                known.update(step.paths)

    def _found(
        self,
        view: SessionView,
        change: _Change,
        edit: Step,
        path: str,
        acquired: Sequence[Step],
        need: int,
        known: set[str],
    ) -> Finding:
        carried = path in known
        cited = [] if change.prompt is None else [change.prompt]
        return _finding(
            self,
            view,
            confidence=0.55 if carried else 0.7,
            title=f"Changed {_short(path)} with too little context",
            explanation=(
                f"The task had acquired {len(acquired)} context item(s) when it edited "
                f"{path} in place; the band asks for {need} ({self.band.describe()})."
                + (
                    " The file was read or written in an earlier task, so carried "
                    "context may be enough."
                    if carried
                    else ""
                )
            ),
            evidence=(
                *cited,
                *(s for i in acquired for s in view.pair(i)),
                *view.pair(edit),
            ),
            subject=path,
            anchor=edit,
        )


#: Shell commands that walk the file tree (a subset of exploring shell).
_TRAVERSAL = re.compile(
    r"(?:ls|find|tree|fd|rg|grep|ag|git\s+(?:ls-files|grep))(?:\s|$)"
)


def _file_names(words: frozenset[str]) -> frozenset[str]:
    """Words that look like file names (``a.py``, ``readme.md``)."""
    return frozenset(w for w in words if "." in w.strip("."))


def _path_words(step: Step) -> frozenset[str]:
    """A step's words plus the base names and stems of the paths it names."""
    names = {_basename(p).lower() for p in step.paths}
    stems = {n.rsplit(".", 1)[0] for n in names}
    return step.words | names | {s for s in stems if len(s) > 2}


@dataclass(frozen=True)
class UnusedTraversal:
    """Point 28: walking the file tree for names nothing later uses (motion)."""

    id: str = "unused_traversal"
    waste: LeanWaste = LeanWaste.MOTION
    kind: FindingKind = FindingKind.WASTE
    summary: str = "A search or directory walk whose listed files nothing later reads, edits or names."
    confidence_rule: str = (
        "Judged only once the agent has responded after it. A search (Grep, "
        "Glob) or traversing shell command (ls, find, tree, rg, grep) whose "
        "output lists file names, none of which a later event reads, edits or "
        "names: 0.72. Calibrated on real sessions: 10 of 10 judged findings "
        "were waste (10 Oct 2026); 0.72 is the 95% lower bound of that "
        "sample. Empty or unobserved output "
        "and repeats of an earlier traversal (left to repeated_exploration) are "
        "not findings."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        last_response = max(
            (s.index for s in view.steps if s.kind is EventKind.RESPONSE), default=-1
        )
        seen: set[str] = set()
        for step in view.requests():
            if not self._traverses(step) or step.call_key is None:
                continue
            if step.call_key in seen:
                continue
            seen.add(step.call_key)
            result = view.completion_of.get(step.index)
            if result is None or result.index > last_response:
                continue
            listed = _file_names(result.words) - step.words
            if not listed or self._used(view, result, listed):
                continue
            shown = ", ".join(sorted(listed)[:3]) + ("…" if len(listed) > 3 else "")
            yield _finding(
                self,
                view,
                confidence=JUDGED_CONFIDENCE,
                title=f"Traversal {_short(step.subject)} led nowhere",
                explanation=(
                    f"{step.native_name} listed {len(listed)} file name(s) ({shown}) "
                    "and nothing the agent did afterwards read, edited or named any of "
                    "them. The walk was motion without downstream use."
                ),
                evidence=view.pair(step),
                waste=view.pair(step),
                subject=step.subject,
            )

    @staticmethod
    def _traverses(step: Step) -> bool:
        if step.tool_kind is ToolKind.FS_SEARCH:
            return True
        return (
            step.tool_kind is ToolKind.EXEC_SHELL
            and step.command is not None
            and _TRAVERSAL.match(step.command) is not None
        )

    @staticmethod
    def _used(view: SessionView, result: Step, listed: frozenset[str]) -> bool:
        stems = {n.rsplit(".", 1)[0] for n in listed}
        stems = {s for s in stems if len(s) >= 4}
        for later in view.steps[result.index + 1 :]:
            if later.is_completion:
                continue
            words = _path_words(later)
            if listed & words or stems & words:
                return True
        return False


@dataclass(frozen=True)
class FailedRoute:
    """Points 29, 30: waiting on a model route that failed and returned nothing."""

    id: str = "failed_route"
    waste: LeanWaste = LeanWaste.WAITING
    kind: FindingKind = FindingKind.WASTE
    summary: str = "A model call that failed over to another route: waiting time that bought no evidence."
    confidence_rule: str = (
        "A route.failover event (a model call that returned no tokens and did "
        "not succeed). 0.80 when a later response completed the work on "
        "another route, so the failed call added only waiting; 0.55 "
        "(uncertain) when no later response followed. The failover's wall time "
        "is the waste; it generated no tokens."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        for step in view.steps:
            if not step.is_failover:
                continue
            later = [
                s for s in view.steps[step.index + 1 :] if s.kind is EventKind.RESPONSE
            ]
            # Prefer the response for the same task (it shares the task's words).
            rerouted = next((s for s in later if s.words & step.words), None)
            if rerouted is None and later:
                rerouted = later[0]
            yield _finding(
                self,
                view,
                confidence=0.8 if rerouted is not None else 0.55,
                title=f"Waited on a failed route: {_short(step.subject)}",
                explanation=(
                    f"The model call '{_short(step.subject)}' failed and returned "
                    f"nothing after {step.seconds:.1f}s"
                    + (
                        ", and the work was then done on another route. The wait "
                        "added no evidence."
                        if rerouted is not None
                        else "; no later response completed the work."
                    )
                ),
                evidence=[step] if rerouted is None else [step, rerouted],
                waste=(step,),
                subject=step.subject,
            )


@dataclass(frozen=True)
class UnearnedEscalation:
    """Point 30: waiting on a model escalation that added no evidence (TER-DET-011).

    An escalation is a ``route.escalated`` step, or a new attempt
    (``attempt.started``) whose first response is served by another model
    than the response before it. It counts only after a completed model call
    in the same task: a response before it, since the task's prompt.

    The escalated call *adds evidence* when, before the next prompt or the
    next escalation, the session gains something no step before the
    escalation held: a file read or searched that was not read before, a
    check result (command, outcome and failure signature) not seen before,
    or any other tool output with a fingerprint not seen before. Response
    text is not compared: the escalated model restating an answer adds
    nothing a later step can cite.
    """

    id: str = "unearned_escalation"
    waste: LeanWaste = LeanWaste.WAITING
    kind: FindingKind = FindingKind.WASTE
    summary: str = (
        "An escalation to another model after a completed call, where the "
        "escalated call added no evidence: waiting that bought nothing new."
    )
    confidence_rule: str = (
        "A route.escalated step (or a re-attempt whose first response another "
        "model served) after a completed response of the same task, followed "
        "before the next prompt or escalation by no new file read, no new "
        "check result and no new tool output. 0.80 for a recorded "
        "route.escalated with an escalated response; 0.60 (uncertain) for a "
        "re-attempt on another model, since a routing harness's verification "
        "results are not steps and the change may be lateral; 0.50 "
        "(uncertain) when no escalated response was recorded. The escalation "
        "marker and the escalated response are the waste."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        steps = view.steps
        marks = [s for s in steps if self._escalates(view, s)]
        for n, mark in enumerate(marks):
            prompt = max(
                (s.index for s in steps[: mark.index] if s.kind is EventKind.PROMPT),
                default=-1,
            )
            earlier = next(
                (
                    s
                    for s in reversed(steps[prompt + 1 : mark.index])
                    if s.kind is EventKind.RESPONSE
                ),
                None,
            )
            if earlier is None:
                continue  # no completed model call before it
            end = view.segment_end(mark.index)
            if n + 1 < len(marks) and marks[n + 1].index <= end:
                end = marks[n + 1].index - 1
            window = steps[mark.index : end + 1]
            if self._adds_evidence(steps[: mark.index], window):
                continue
            implicit = not mark.is_escalation
            call = (
                mark
                if implicit
                else next((s for s in window if s.kind is EventKind.RESPONSE), None)
            )
            confidence = 0.6 if implicit else 0.8 if call is not None else 0.5
            served = (
                f" ({call.usage.model})"
                if call and call.usage and call.usage.model
                else ""
            )
            yield _finding(
                self,
                view,
                confidence=confidence,
                title=f"Escalation added no evidence: {_short(mark.subject or 'new attempt')}",
                explanation=(
                    "After a completed model call the task moved to another model"
                    f"{served}, and the escalated call read no new file, ran no new "
                    "check and produced no new tool output before the next prompt."
                    + (
                        " The escalation was not followed by a recorded response."
                        if call is None
                        else ""
                    )
                ),
                evidence=[earlier, mark, *([call] if call is not None else [])],
                waste=[mark, *([call] if call is not None else [])],
                subject=mark.subject,
                anchor=mark,
            )

    @staticmethod
    def _escalates(view: SessionView, step: Step) -> bool:
        if step.is_escalation:
            return True
        if not (step.opens_attempt and step.kind is EventKind.RESPONSE):
            return False
        model = step.usage.model if step.usage else None
        before = next(
            (
                s
                for s in reversed(view.steps[: step.index])
                if s.kind is EventKind.RESPONSE and s.usage and s.usage.model
            ),
            None,
        )
        return (
            model is not None
            and before is not None
            and before.usage is not None
            and before.usage.model != model
        )

    @staticmethod
    def _adds_evidence(prior: Sequence[Step], window: Sequence[Step]) -> bool:
        read = {
            p
            for s in prior
            if s.is_request and s.tool_kind in _EXPLORE_KINDS
            for p in s.paths
        }
        checks = {
            (s.command, s.outcome, s.failure_signature)
            for s in prior
            if s.is_completion and s.outcome is not None
        }
        outputs = {s.output_hash for s in prior if s.is_completion and s.output_hash}
        for s in window:
            if s.is_request and s.tool_kind in _EXPLORE_KINDS and set(s.paths) - read:
                return True
            if not s.is_completion or s.tool_kind in _CHANGE_KINDS:
                continue  # an edit's own result is not evidence
            if s.outcome is not None:
                if (s.command, s.outcome, s.failure_signature) not in checks:
                    return True
            elif s.output_hash and s.output_hash not in outputs:
                return True
        return False


def exploration_labels(view: SessionView) -> tuple[ExplorationLabel, ...]:
    """Label every exploration request by what drove it (TER-DET-009, point 38).

    Questions (sentences ending in ``?``) in the prompt, reasoning or
    responses of a task are its open questions until the next prompt. An
    exploration request is *uncertainty-driven* only when it names a word of
    an open question recorded before it; otherwise *intent-directed* when it
    names a word or file of the prompt in force; otherwise *aimless*.
    """
    out: list[ExplorationLabel] = []
    prompt: Step | None = None
    questions: list[Step] = []
    for step in view.steps:
        if step.kind is EventKind.PROMPT:
            prompt, questions = step, []
        if step.questions:
            questions.append(step)
            continue
        if not step.is_request or step.stage is not Stage.EXPLORE:
            continue
        words = _path_words(step)
        label: ExplorationLabel | None = None
        for question in reversed(questions):
            shared = words & question.questions
            if shared:
                label = ExplorationLabel(
                    step.event_id,
                    ExplorationDriver.UNCERTAINTY_DRIVEN,
                    question.event_id,
                    tuple(sorted(shared)),
                )
                break
        if label is None and prompt is not None and words & prompt.words:
            label = ExplorationLabel(
                step.event_id,
                ExplorationDriver.INTENT_DIRECTED,
                prompt.event_id,
                tuple(sorted(words & prompt.words)),
            )
        out.append(
            label
            or ExplorationLabel(step.event_id, ExplorationDriver.AIMLESS, None, ())
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class DetectorRegistry:
    """An ordered set of detectors, keyed by id. Order breaks ties in reports."""

    def __init__(self, detectors: Iterable[WasteDetector] = ()) -> None:
        self._detectors: dict[str, WasteDetector] = {}
        for detector in detectors:
            self.register(detector)

    def register(self, detector: WasteDetector) -> None:
        if detector.id in self._detectors:
            raise ValueError(f"Detector {detector.id!r} is already registered")
        self._detectors[detector.id] = detector

    def __iter__(self) -> Iterator[WasteDetector]:
        return iter(self._detectors.values())

    def __len__(self) -> int:
        return len(self._detectors)

    def __contains__(self, detector_id: object) -> bool:
        return detector_id in self._detectors

    def get(self, detector_id: str) -> WasteDetector:
        return self._detectors[detector_id]

    def extended(self, detectors: Iterable[WasteDetector]) -> DetectorRegistry:
        """This registry, then each of ``detectors`` whose id it lacks."""
        out = DetectorRegistry(self)
        for detector in detectors:
            if detector.id not in out:
                out.register(detector)
        return out

    def run(self, view: SessionView) -> tuple[Finding, ...]:
        """Every detector's findings, largest cost first, then in session order."""
        order = {d.id: i for i, d in enumerate(self)}
        index = {s.event_id: s.index for s in view.steps}
        found = [f for d in self for f in d.detect(view)]
        return tuple(
            sorted(
                found,
                key=lambda f: (
                    f.kind is not FindingKind.WASTE,
                    -(f.tokens + f.context_tokens),
                    index[f.evidence[0]],
                    order[f.detector],
                    f.id,
                ),
            )
        )


#: The L2 detector set, in catalogue order.
DEFAULT_REGISTRY = DetectorRegistry(
    (
        RepeatedToolCall(),
        RepeatedExploration(),
        ReworkCycle(),
        UnvalidatedImplementation(),
        PrematureImplementation(),
        ExcessivePlanning(),
        FragmentedEdits(),
        UnusedContext(),
        UnnecessaryHandoff(),
        RepeatedReasoning(),
        Regeneration(),
        IntentDrift(),
        ExcessiveContext(),
        InsufficientContext(),
        UnusedTraversal(),
        FailedRoute(),
        UnearnedEscalation(),
    )
)


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _files(steps: Sequence[Step]) -> str:
    paths = sorted({p for s in steps for p in s.paths})
    if not paths:
        return "files"
    if len(paths) <= 3:
        return ", ".join(paths)
    return f"{', '.join(paths[:3])} and {len(paths) - 3} more"
