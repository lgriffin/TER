"""Grounded detectors (L3): the change surface and architecture boundaries.

These read the session's :class:`~.grounding.RepositoryGrounding` (repository
evidence at the start commit plus the replayed edits) and need nothing else
from outside, so they are plugins like the L2 detectors
(``docs/ter4/l3-grounded.md``). Without repository evidence they find nothing; the
analysis adds them to the registry only when it is given that evidence, so
an L2 analysis is unchanged.

**The expected change surface of a task** (TER-EVD-006, point 63) is built
from structure only, never from token counts or word overlap scores:

1. *Seeds*: the repository files the task's prompt names, by path or file
   name (``core.py``), dotted module name (``pkg.core``) or a distinctive
   symbol the file defines (``tests_importing``, ``ExplainSession``). A
   prompt that names none inherits the seeds of the last prompt that did,
   when the intent record reads it as a refinement or an acknowledgement of
   that intent; otherwise (a changed intent, or nothing named yet) the seed
   is the task's first edited repository file.
2. *Neighbours*: every repository file a seed imports or is imported by, at
   the start commit or after the task's own edits (test modules excepted).
3. *Tests*: every test module that imports a seed or a neighbour, or that a
   seed imports.

Files the task creates count like any other: a new module a seed now
imports, or a new test of a seed, is inside.

Every edit is then marked *inside* the surface; *expansion*, outside it but
one import link from a file inside (point 64); *unrelated*, with no import
link to it (point 66); *harness state*, under a ``.claude`` directory that
is not a worktree checkout and outside every root (the agent's plans, memory
and settings; not judged); or *outside the repository* (not judged).

**Boundary violations** (TER-EVD-007, point 65): an edit whose replayed
result imports a module it did not import before, where that import breaks a
contract the repository declares (import-linter's ``forbidden``, ``layers``
and ``independence``), and, TER-EVD-015, import-linter's ``protected`` and
``acyclic_siblings`` and dependency-cruiser's ``forbidden`` path rules for
TypeScript, JavaScript, Svelte and Vue files.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from ..events import EventId, EventKind
from ..repository import (
    ArchitectureContract,
    ContractKind,
    ContractViolation,
    contract_violations,
    is_harness_state,
    is_test_module,
)
from .detectors import SessionView, WasteDetector, _finding
from .grounding import EditGrounding, RepositoryGrounding
from .intent import IntentRelation
from .model import ActivityClass, Finding, FindingKind, LeanWaste, Step

__all__ = [
    "GROUNDED_DETECTORS",
    "BoundaryViolation",
    "ChangeSurface",
    "EditPlacement",
    "SeedBasis",
    "SurfaceEdit",
    "SurfaceExpansion",
    "UnrelatedModification",
    "change_surfaces",
    "files_named",
    "surface_of",
]


class SeedBasis(StrEnum):
    """Where a task's seed files came from."""

    NAMED = "named"
    INHERITED = "inherited"
    FIRST_EDIT = "first_edit"


class EditPlacement(StrEnum):
    INSIDE = "inside"
    EXPANSION = "expansion"
    UNRELATED = "unrelated"
    OUTSIDE_REPOSITORY = "outside_repository"
    #: The agent harness's own state under a ``.claude`` directory (plans,
    #: memory, settings), outside every root: not judged (TER-EVD-019).
    HARNESS_STATE = "harness_state"


@dataclass(frozen=True)
class SurfaceEdit:
    """One edit or write of a task, placed against the task's surface."""

    event_id: EventId
    path: str
    placement: EditPlacement
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "path": self.path,
            "placement": self.placement.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ChangeSurface:
    """The expected change surface of one task and where its edits fell."""

    prompt: EventId | None
    basis: SeedBasis
    #: The prompt whose words named the seeds (``None`` for a first edit).
    named_by: EventId | None
    seeds: tuple[str, ...]
    neighbours: tuple[str, ...]
    tests: tuple[str, ...]
    #: Files of the surface that the task itself created.
    created: tuple[str, ...]
    edits: tuple[SurfaceEdit, ...]

    @property
    def files(self) -> frozenset[str]:
        return frozenset((*self.seeds, *self.neighbours, *self.tests))

    def outside(self) -> tuple[SurfaceEdit, ...]:
        """Edits of repository files outside the surface."""
        return tuple(
            e
            for e in self.edits
            if e.placement in (EditPlacement.EXPANSION, EditPlacement.UNRELATED)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "basis": self.basis.value,
            "named_by": self.named_by,
            "seeds": list(self.seeds),
            "neighbours": list(self.neighbours),
            "tests": list(self.tests),
            "created": list(self.created),
            "edits": [e.as_dict() for e in self.edits],
        }


# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Task:
    prompt: Step | None
    edits: tuple[Step, ...]


def _tasks(
    view: SessionView, failed: frozenset[EventId] = frozenset()
) -> Iterator[_Task]:
    """Every prompt segment (and the edits before the first prompt); an
    edit whose tool failed (``failed``) changed nothing and is left out."""
    prompt: Step | None = None
    edits: list[Step] = []
    started = False
    for step in (*view.steps, None):
        if step is None or step.kind is EventKind.PROMPT:
            if started or edits:
                yield _Task(prompt, tuple(edits))
            prompt, edits, started = step, [], True
            continue
        if step.is_edit and step.event_id not in failed:
            edits.append(step)


class _Names:
    """Which repository files a prompt's words name."""

    def __init__(self, g: RepositoryGrounding) -> None:
        created = {e.path for e in g.edits.values() if e.created}
        self.by_word: dict[str, set[str]] = {}
        for path in sorted(g.files | created):
            self._add(PurePosixPath(path).name.lower(), path)
            module = g.modules.get(path)
            if module and "." in module:
                self._add(module.lower(), path)
        for name, paths in g.symbols.items():
            for path in paths:
                self._add(name, path)

    def _add(self, word: str, path: str) -> None:
        self.by_word.setdefault(word, set()).add(path)

    def named(self, words: frozenset[str]) -> tuple[str, ...]:
        return tuple(sorted({p for w in words for p in self.by_word.get(w, ())}))


class _Graph:
    """Import links at the start commit plus those the task's edits made."""

    def __init__(self, g: RepositoryGrounding, edits: Sequence[EditGrounding]):
        self.g = g
        self.out: dict[str, set[str]] = {}
        self.into: dict[str, set[str]] = {}
        for e in edits:
            for target in e.links:
                self.out.setdefault(e.path, set()).add(target)
                self.into.setdefault(target, set()).add(e.path)

    def linked(self, path: str) -> set[str]:
        return (
            set(self.g.links.get(path, ()))
            | set(self.g.importers.get(path, ()))
            | self.out.get(path, set())
            | self.into.get(path, set())
        )

    def importers(self, path: str) -> set[str]:
        return set(self.g.importers.get(path, ())) | self.into.get(path, set())


def _expand(graph: _Graph, seeds: set[str]) -> tuple[set[str], set[str]]:
    """The neighbours and tests of a surface grown from ``seeds``."""
    linked = set().union(*(graph.linked(s) for s in seeds)) - seeds
    neighbours = {p for p in linked if not is_test_module(p)}
    core = seeds | neighbours
    tests = (linked - neighbours) | {
        t for p in core for t in graph.importers(p) if is_test_module(t)
    } - core
    return neighbours, tests


def files_named(g: RepositoryGrounding, words: frozenset[str]) -> tuple[str, ...]:
    """The repository files a prompt's content words name, by file name,
    dotted module name or distinctive symbol: a task's named seeds."""
    return _Names(g).named(words)


def surface_of(
    g: RepositoryGrounding, seeds: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The neighbours and tests of the surface grown from ``seeds`` over the
    start commit's import graph, each sorted (the rules of
    :func:`change_surfaces`, before any edit). Context bundles select from it
    (TER-CTX-001)."""
    neighbours, tests = _expand(_Graph(g, ()), set(seeds))
    return tuple(sorted(neighbours)), tuple(sorted(tests))


def _place(
    task: _Task,
    g: RepositoryGrounding,
    basis: SeedBasis,
    named_by: Step | None,
    seeds: tuple[str, ...],
) -> ChangeSurface:
    grounded = [g.edits[e.event_id] for e in task.edits if e.event_id in g.edits]
    graph = _Graph(g, grounded)
    seed_set = set(seeds)
    neighbours, tests = _expand(graph, seed_set)
    surface = seed_set | neighbours | tests
    created = {e.path for e in grounded if e.created} & surface
    role = {
        **{p: "a test of the surface" for p in tests},
        **{p: "imported by or importing a seed" for p in neighbours},
        **{p: f"a seed ({basis.value})" for p in seed_set},
    }
    placed: list[SurfaceEdit] = []
    for step in task.edits:
        if not step.paths:
            continue
        path = g.repository_path(step.paths[0])
        if path is None and is_harness_state(step.paths[0]):
            placed.append(
                SurfaceEdit(
                    step.event_id,
                    step.paths[0],
                    EditPlacement.HARNESS_STATE,
                    "the agent harness's own state (plans, memory, settings) "
                    "under a .claude directory",
                )
            )
        elif path is None:
            placed.append(
                SurfaceEdit(
                    step.event_id,
                    step.paths[0],
                    EditPlacement.OUTSIDE_REPOSITORY,
                    "not a path under the repository root",
                )
            )
        elif path in surface:
            placed.append(
                SurfaceEdit(step.event_id, path, EditPlacement.INSIDE, role[path])
            )
        elif near := sorted(graph.linked(path) & surface):
            placed.append(
                SurfaceEdit(
                    step.event_id,
                    path,
                    EditPlacement.EXPANSION,
                    f"one import link beyond the surface (via {', '.join(near[:3])})",
                )
            )
        else:
            placed.append(
                SurfaceEdit(
                    step.event_id,
                    path,
                    EditPlacement.UNRELATED,
                    "linked by no import to any file of the surface",
                )
            )
    return ChangeSurface(
        prompt=task.prompt.event_id if task.prompt is not None else None,
        basis=basis,
        named_by=named_by.event_id if named_by is not None else None,
        seeds=tuple(sorted(seed_set)),
        neighbours=tuple(sorted(neighbours)),
        tests=tuple(sorted(tests)),
        created=tuple(sorted(created)),
        edits=tuple(placed),
    )


_CONTINUES = (IntentRelation.REFINED, IntentRelation.ACKNOWLEDGED)


def change_surfaces(view: SessionView) -> tuple[ChangeSurface, ...]:
    """The expected change surface of every task that edits (TER-EVD-006).

    Empty without repository evidence.
    """
    g = view.repository
    if g is None:
        return ()
    names = _Names(g)
    relation = {r.event_id: r.relation for r in view.intent.record.revisions}
    last: tuple[tuple[str, ...], Step] | None = None
    out: list[ChangeSurface] = []
    for task in _tasks(view, g.failed_edits):
        named = names.named(task.prompt.words) if task.prompt is not None else ()
        named_by: Step | None
        if named and task.prompt is not None:
            basis, seeds, named_by = SeedBasis.NAMED, named, task.prompt
            last = (named, task.prompt)
        elif (
            last is not None
            and task.prompt is not None
            and relation.get(task.prompt.event_id) in _CONTINUES
        ):
            basis, (seeds, named_by) = SeedBasis.INHERITED, last
        else:
            last = None
            first = next(
                (
                    p
                    for e in task.edits
                    if e.paths and (p := g.repository_path(e.paths[0])) is not None
                ),
                None,
            )
            basis, named_by = SeedBasis.FIRST_EDIT, None
            seeds = () if first is None else (first,)
        if task.edits:
            out.append(_place(task, g, basis, named_by, seeds))
    return tuple(out)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _imports_read(g: RepositoryGrounding, view: SessionView, path: str) -> bool:
    """Whether the engine read the file's imports from a syntax tree, at the
    start commit or after one of the session's edits: a Python file, or a
    TypeScript, JavaScript, Svelte or Vue file under an engine that reads
    them (``syntax``)."""
    if not g.syntax:
        return False
    return path in g.links or any(e.parsed for e in g.edits.values() if e.path == path)


def _outside(
    detector: WasteDetector,
    view: SessionView,
    placement: EditPlacement,
    confidence: Callable[[ChangeSurface, str], tuple[float, str]],
) -> Iterator[Finding]:
    for surface in change_surfaces(view):
        groups: dict[str, list[SurfaceEdit]] = {}
        for edit in surface.edits:
            if edit.placement is placement:
                groups.setdefault(edit.path, []).append(edit)
        prompts = [
            view.by_id[e]
            for e in dict.fromkeys((surface.named_by, surface.prompt))
            if e is not None
        ]
        for path, edits in groups.items():
            steps = [s for e in edits for s in view.pair(view.by_id[e.event_id])]
            value, why = confidence(surface, path)
            where = (
                ", ".join(surface.seeds[:3])
                + (
                    f" and {len(surface.seeds) - 3} more"
                    if len(surface.seeds) > 3
                    else ""
                )
                if surface.seeds
                else "nothing"
            )
            yield _finding(
                detector,
                view,
                confidence=value,
                title=f"Edit outside the change surface: {path}",
                explanation=(
                    f"The task's change surface grows from {where} "
                    f"({surface.basis.value.replace('_', ' ')}) and holds "
                    f"{len(surface.files)} file(s); {path} is {edits[0].reason}. "
                    f"{len(edits)} edit(s) changed it. {why}"
                ),
                evidence=(*prompts, *steps),
                waste=steps,
                subject=path,
            )


@dataclass(frozen=True)
class UnrelatedModification:
    """Points 63, 66 (TER-EVD-006): an edit with no import link to the task."""

    id: str = "unrelated_modification"
    waste: LeanWaste = LeanWaste.OVERPRODUCTION
    # Judged not waste on real sessions (0 of 29), so it is a pointer for
    # review that claims no cost, not uncertain waste (ADR 0006).
    kind: FindingKind = FindingKind.RISK
    summary: str = (
        "An edit to a repository file with no import link to the task's "
        "expected change surface."
    )
    confidence_rule: str = (
        "Needs repository evidence (L3). Per task and file, edits outside the "
        "expected change surface (seed files the prompt names, the files they "
        "import or are imported by at the start commit or after the task's "
        "edits, and the tests of both) and not one import link beyond it: "
        "0.60 (uncertain) when the prompt itself "
        "named the seeds and the file's imports were read from a syntax tree "
        "(calibrated on real sessions: none of 29 judged findings, 9 of them "
        "previously confident, was truly unrelated); "
        "0.65 (uncertain) when the seeds were inherited from an earlier prompt "
        "the intent continues; 0.55 (uncertain) when the file has no import "
        "evidence (a language the engine does not read, or it did not parse) "
        "or the seed is only the "
        "task's first edit, or when the file is a test module the task "
        "created. Files outside the repository: no finding."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        g = view.repository
        if g is None:
            return

        def confidence(surface: ChangeSurface, path: str) -> tuple[float, str]:
            # Calibration on a real session (9 Oct 2026): a test module the
            # task created to verify its change imported the code under test
            # only through a facade (the CLI), so it showed no import link to
            # the seeds. A new test is how a change gets verified, not extra
            # work, so it never counts as confident waste.
            if path not in g.files and is_test_module(path):
                return 0.55, (
                    "The task created this test module; a new test usually "
                    "verifies the change through an entry point the import "
                    "graph does not tie to the files asked for."
                )
            if not _imports_read(g, view, path):
                return 0.55, (
                    "The repository has no import evidence for this file, so it "
                    "may belong to the change in a way structure cannot show."
                )
            if surface.basis is SeedBasis.FIRST_EDIT:
                return 0.55, (
                    "The prompt names no repository file, so the surface grows "
                    "from the task's first edit, which may not be its target."
                )
            if surface.basis is SeedBasis.INHERITED:
                return 0.65, (
                    "The prompt names no file; the surface is the one an earlier "
                    "prompt named, which this prompt may have widened."
                )
            # Calibration on real sessions (9 Oct 2026): the owner's judging of
            # 29 findings (all 9 confident ones and 20 sampled uncertain ones)
            # found none truly unrelated. Tasks also need tests, fixtures,
            # docs, CI and config, and modules an issue names but the prompt
            # does not; the import graph cannot see those links. The finding
            # stays a pointer for review and never counts as waste until a
            # judged corpus shows true positives.
            return 0.6, (
                "Nothing in the import graph ties it to what was asked, but "
                "on real sessions such edits were tests, docs, CI, config or "
                "modules an issue named, so this is a pointer for review."
            )

        yield from _outside(self, view, EditPlacement.UNRELATED, confidence)


@dataclass(frozen=True)
class SurfaceExpansion:
    """Point 64 (TER-EVD-006): the change rippled one import link further."""

    id: str = "surface_expansion"
    waste: LeanWaste = LeanWaste.OVERPRODUCTION
    kind: FindingKind = FindingKind.WASTE
    summary: str = "An edit one import link beyond the task's expected change surface."
    confidence_rule: str = (
        "Needs repository evidence (L3). Per task and file, edits outside the "
        "expected change surface that import, or are imported by, a file "
        "inside it: 0.60 (uncertain) when the prompt named the seeds, 0.50 "
        "(uncertain) otherwise. Always uncertain: a change often has to ripple "
        "to the callers of what it changes, and structure alone cannot tell a "
        "needed ripple from an unneeded one."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        if view.repository is None:
            return

        def confidence(surface: ChangeSurface, path: str) -> tuple[float, str]:
            if surface.basis is SeedBasis.NAMED:
                return 0.6, (
                    "Callers sometimes have to change with what they call; check "
                    "whether this one had to."
                )
            return 0.5, "The surface itself was not named by the prompt."

        yield from _outside(self, view, EditPlacement.EXPANSION, confidence)


@dataclass(frozen=True)
class BoundaryViolation:
    """Point 65 (TER-EVD-007, TER-EVD-015): an added import breaks a declared
    contract."""

    id: str = "boundary_violation"
    waste: LeanWaste = LeanWaste.DEFECTS
    kind: FindingKind = FindingKind.RISK
    summary: str = (
        "An edit adds an import that breaks an architecture contract the "
        "repository declares."
    )
    confidence_rule: str = (
        "Needs repository evidence (L3) and declared architecture contracts: "
        "import-linter forbidden, layers, independence, protected and "
        "acyclic_siblings contracts on Python modules, or dependency-cruiser "
        "forbidden path rules on TypeScript, JavaScript, Svelte and Vue files. "
        "The edited file is replayed from the start commit and parsed; an "
        "import it holds after the edit and did not hold before, that a "
        "contract forbids for the file: 0.90 when the import is still there "
        "after the session's last edit of the file; 0.70 when a later edit of "
        "the file could not be replayed, so its final imports are unknown; "
        "0.55 (uncertain) when a later edit removed it. Only direct imports "
        "are judged; an acyclic_siblings contract is broken by an import that "
        "closes a cycle among siblings over the start commit's import graph "
        "and the session's earlier edits. A risk: it claims no cost."
    )

    def detect(self, view: SessionView) -> Iterable[Finding]:
        g = view.repository
        if g is None or not g.contracts:
            return
        order = [g.edits[s.event_id] for s in view.requests() if s.event_id in g.edits]
        graph = _module_graph(g) if _needs_graph(g.contracts) else None
        for n, edit in enumerate(order):
            by_path = edit.module is None
            importer = edit.path if by_path else edit.module
            if edit.added and importer is not None:
                later = [e for e in order[n + 1 :] if e.path == edit.path]
                for added in edit.added:
                    violations = contract_violations(
                        importer,
                        added.module,
                        g.contracts,
                        by_path=by_path,
                        graph=graph,
                    )
                    if not violations:
                        continue
                    confidence, fate, closing = self._fate(added.module, later)
                    for violation in violations:
                        yield self._found(
                            view, edit, added.line, violation, confidence, fate, closing
                        )
            if graph is not None and edit.module is not None and edit.parsed:
                graph[edit.module] = frozenset(edit.imports)

    @staticmethod
    def _fate(
        module: str, later: Sequence[EditGrounding]
    ) -> tuple[float, str, EditGrounding | None]:
        for e in later:
            if not e.applied or not e.parsed:
                return (
                    0.7,
                    "A later edit of the file could not be replayed, so whether "
                    "the import stayed is unknown.",
                    None,
                )
            if module not in e.imports:
                return 0.55, "A later edit removed the import again.", e
        return 0.9, "The import is still there after the session's last edit.", None

    def _found(
        self,
        view: SessionView,
        edit: EditGrounding,
        line: int,
        violation: ContractViolation,
        confidence: float,
        fate: str,
        closing: EditGrounding | None,
    ) -> Finding:
        step = view.by_id[edit.event_id]
        cited = [*view.pair(step)]
        if closing is not None:
            cited.extend(view.pair(view.by_id[closing.event_id]))
        return _finding(
            self,
            view,
            confidence=confidence,
            title=(
                f"Import breaks contract {violation.contract}: "
                f"{violation.importer} -> {violation.imported}"
            ),
            explanation=(
                f"{step.native_name} of {edit.path} adds `import "
                f"{violation.imported}` (line {line}) to {violation.importer}; "
                f"contract {violation.contract} ({violation.contract_name}) says "
                f"{violation.rule}. {fate}"
            ),
            evidence=cited,
            subject=violation.contract,
            activity_class=ActivityClass.AVOIDABLE,
            anchor=step,
            key=f"{violation.contract}:{violation.imported}",
        )


def _needs_graph(contracts: Iterable[ArchitectureContract]) -> bool:
    return any(c.kind is ContractKind.ACYCLIC_SIBLINGS for c in contracts)


def _module_graph(g: RepositoryGrounding) -> dict[str, frozenset[str]]:
    """Python module -> the repository modules it imports at the start
    commit (what an ``acyclic_siblings`` contract is judged against)."""
    return {
        g.modules[path]: frozenset(g.modules[t] for t in targets if t in g.modules)
        for path, targets in g.links.items()
        if path in g.modules
    }


#: Detectors that need repository evidence; the analysis adds them to its
#: registry when it is given a :class:`~.grounding.RepositoryGrounding`.
GROUNDED_DETECTORS: tuple[WasteDetector, ...] = (
    UnrelatedModification(),
    SurfaceExpansion(),
    BoundaryViolation(),
)
