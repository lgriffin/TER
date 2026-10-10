"""Exploration drift (L3, TER-ITN-006, point 48): exploration or reasoning
that departs from the current intent where repository evidence shows it
touches nothing the intended change depends on.

L2 judges only edits and writes for drift (``intent_drift``, TER-ITN-003),
because exploration that looks unrelated in words is often needed. Here the
repository decides. Within a task (a prompt and the work up to the next one,
so a recorded intent change starts a new task and is never drift), a read
or a reasoning block is a **departure** when:

1. the task has an expected change surface (it changed something), and its
   alignment to the intent in force is not in the aligned band (below the
   drift band, or unscored);
2. every repository file it reads or names lies outside what the change
   depends on. A file is **dependent** when it is on the surface (a seed,
   an import neighbour or a test), edited by the task, one import link from
   a file of the surface, or used later by a change, a command or a check
   (evidence usage, TER-EVD-008), or when it is a **test, doc, CI or config
   file**. That last rule is the lesson of the ``unrelated_modification``
   calibration (``docs/ter4/l3-grounded.md``): import-graph evidence alone
   produced 0 true positives in 29 judged findings, because tasks need
   tests, docs, CI, config and modules an issue names, and the import graph
   cannot see those links;
3. no response is still to come after it (a live read is not judged early).

Confidence is capped below 0.70 (uncertain: counted as waste until
verified, ADR 0006) unless the departure is unambiguous: a read, with seeds the prompt named, of
a source file in a different top-level package from every seed and every
file the task edited, that nothing later named at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath

from ..events import EventKind
from .detectors import SessionView, _finding
from .grounding import RepositoryGrounding
from .model import Finding, FindingKind, LeanWaste, Step
from .surface import SeedBasis, change_surfaces
from .usage import (
    EvidenceUsage,
    FileNames,
    FileRole,
    evidence_usage,
    file_role,
    read_targets,
)
from .value import Task, tasks_of

__all__ = ["EVIDENCE_DETECTORS", "ExplorationDrift", "dependent", "package_of_path"]

#: Directories that hold packages rather than being one (``src/app``,
#: ``packages/web``): the top-level package is the next component.
_CONTAINERS = frozenset(
    {
        "src",
        "lib",
        "libs",
        "packages",
        "apps",
        "modules",
        "crates",
        "services",
        "pkg",
        "internal",
        "cmd",
    }
)


def package_of_path(path: str) -> str:
    """The top-level package a repository path belongs to: its first
    directory, or the first two when the first only holds packages
    (``src/app/x.py`` -> ``src/app``). A file at the root has none (``""``)."""
    parts = PurePosixPath(path).parts
    if len(parts) <= 1:
        return ""
    if parts[0].lower() in _CONTAINERS and len(parts) > 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def dependent(
    path: str, task: Task, g: RepositoryGrounding, usage: EvidenceUsage
) -> str | None:
    """Why the intended change of ``task`` depends on ``path``, or ``None``."""
    if path in task.required:
        return "on the change surface"
    role = file_role(path)
    if role is not FileRole.SOURCE:
        return f"a {role.value} file"
    linked = set(g.links.get(path, ())) | set(g.importers.get(path, ()))
    for step in task.steps:
        e = g.edits.get(step.event_id)
        if e is not None:
            if path in e.links:
                linked.add(e.path)
            if e.path == path:
                return "edited by the task"
    if linked & task.required:
        return "one import link from the change surface"
    if any(r.path == path and r.material for r in usage.reads):
        return "used later by a change, command or check"
    return None


@dataclass(frozen=True)
class ExplorationDrift:
    """Point 48 (TER-ITN-006): exploration or reasoning off what the change
    depends on, with no intent change recorded."""

    id: str = "exploration_drift"
    waste: LeanWaste = LeanWaste.MOTION
    kind: FindingKind = FindingKind.WASTE
    summary: str = (
        "Reads or reasoning about repository files the intended change does "
        "not depend on, with no intent change recorded."
    )
    confidence_rule: str = (
        "Needs repository evidence (L3). Per task that changed something and "
        "repository file: reads (a read tool, or an exploring shell command naming the file) or reasoning naming files not in the aligned "
        "intent band whose every file is off the change surface, not edited, "
        "not one import link from it, not used later by a change, command or "
        "check, and not a test, doc, CI or config file (calibration: import "
        "evidence alone gave 0 of 29 true positives), with a response after "
        "it: 0.75 for a read when the prompt named the seeds, the file is "
        "source in a different top-level package from every seed and edited "
        "file, and nothing later named it at all; 0.60 (uncertain) for such a "
        "read in the same package or named later by a decision; 0.60 "
        "(uncertain) when the seeds were inherited; 0.50 (uncertain) when the "
        "surface grew from the task's first edit; reasoning 0.50 (uncertain). "
        "A new prompt (a recorded intent change) starts a new task: never drift."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        g = view.repository
        if g is None:
            return
        surfaces = change_surfaces(view)
        names = FileNames(g)
        usage = evidence_usage(view.steps, g, view.completion_of, names)
        last_response = max(
            (s.index for s in view.steps if s.kind is EventKind.RESPONSE), default=-1
        )
        band = view.intent.config.drift_below
        for task in tasks_of(view.steps, g, surfaces):
            if task.surface is None or not task.surface.seeds:
                continue
            yield from self._task(view, g, task, usage, names, last_response, band)

    def _departs(self, view: SessionView, step: Step, band: float) -> bool:
        a = view.intent.alignment_of(step.event_id)
        return a is None or a.score is None or a.score < band

    def _task(
        self,
        view: SessionView,
        g: RepositoryGrounding,
        task: Task,
        usage: EvidenceUsage,
        names: FileNames,
        last_response: int,
        band: float,
    ) -> Iterator[Finding]:
        assert task.surface is not None
        reads: dict[str, list[Step]] = {}
        thoughts: list[tuple[Step, tuple[str, ...]]] = []
        for step in task.steps:
            end = view.completion_of.get(step.index, step)
            if end.index >= last_response or not self._departs(view, step, band):
                continue
            targets, _ = read_targets(step, g)
            if targets:
                # A command reading several files drifts only if all do.
                if all(dependent(p, task, g, usage) is None for p in targets):
                    for path in targets:
                        reads.setdefault(path, []).append(step)
            elif step.kind is EventKind.REASONING:
                files = tuple(sorted(names.named(step.words)))
                if files and all(dependent(p, task, g, usage) is None for p in files):
                    thoughts.append((step, files))
        anchors = [view.by_id[e] for e in task.anchors()]
        packages = {package_of_path(p) for p in (*task.surface.seeds, *task.edited)}
        for path, steps in reads.items():
            ids = {s.event_id for s in steps}
            named_later = any(r.uses for r in usage.reads if r.event_id in ids)
            confidence, why = self._confidence(task, path, packages, named_later)
            pairs = [p for s in steps for p in view.pair(s)]
            yield _finding(
                self,
                view,
                confidence=confidence,
                title=f"Exploration drifted off the change: {path}",
                explanation=(
                    f"{len(steps)} read(s) of {path}, which the change growing "
                    f"from {', '.join(task.surface.seeds[:3])} does not depend "
                    "on: off its surface, no import link to it, not a test, "
                    "doc, CI or config file, and no later change, command or "
                    f"check used it. No intent change was recorded. {why}"
                ),
                evidence=(*anchors, *pairs),
                waste=pairs,
                subject=path,
            )
        for step, files in thoughts:
            yield _finding(
                self,
                view,
                confidence=0.5,
                title=f"Reasoning drifted off the change: {', '.join(files[:3])}",
                explanation=(
                    f"The reasoning names {', '.join(files)}, none of which the "
                    "change depends on, and no intent change was recorded. "
                    "Reasoning about a file is often how it is ruled out, so "
                    "this is a pointer for review."
                ),
                evidence=(*anchors, step),
                waste=(step,),
                subject=", ".join(files),
            )

    @staticmethod
    def _confidence(
        task: Task, path: str, packages: set[str], named_later: bool
    ) -> tuple[float, str]:
        assert task.surface is not None
        if task.surface.basis is SeedBasis.FIRST_EDIT:
            return 0.5, "The prompt named no file, so the surface may miss the target."
        if task.surface.basis is SeedBasis.INHERITED:
            return 0.6, "The surface comes from an earlier prompt this one may widen."
        if named_later:
            return 0.6, "A later decision named it, so it may have informed one."
        if package_of_path(path) not in packages:
            return 0.75, (
                f"It lies in another top-level package "
                f"({package_of_path(path) or 'the root'}) than every seed and "
                "edited file, and nothing later named it."
            )
        return 0.6, (
            "It shares a package with the change, where links the import graph "
            "cannot see are common."
        )


#: Detectors that need evidence usage as well as the change surface; the
#: analysis adds them, with :data:`.surface.GROUNDED_DETECTORS`, only when
#: it is given repository evidence.
EVIDENCE_DETECTORS: tuple[ExplorationDrift, ...] = (ExplorationDrift(),)
