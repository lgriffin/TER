"""In-memory adapters: fakes for tests that must pass the same contract suites.

A fake that drifts from the real adapter's obligations fails
``tests/contract``, so tests that use these fakes stay honest.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

from ...domain.events import Event, SessionTrace
from ...domain.lean.control import ControlLimits, ControlLimitsError
from ...domain.outcome import OutcomeEvidence, OutcomeFormatError
from ...domain.pricing import PriceEntry, PriceSchedule, Rates
from ...domain.routing import RoutingProfile, RoutingProfileError
from ...domain.repository import (
    ArchitectureContract,
    ContractFormatError,
    FileCommit,
    RepositoryDiff,
    RepositoryEvidenceError,
    SourceStructure,
    TextMatch,
    UnknownPathError,
    module_name,
)
from ...domain.repository import tests_importing as repository_tests_importing


class InMemorySessionSource:
    """Serves pre-built traces by reference, renumbering sequences on the way out."""

    format_name = "in-memory"

    def __init__(self, traces: Mapping[str, SessionTrace]) -> None:
        self._traces = dict(traces)

    def read(self, ref: str | Path) -> SessionTrace:
        key = str(ref)
        if key not in self._traces:
            raise FileNotFoundError(f"No in-memory session named {key!r}")
        trace = self._traces[key]
        events: tuple[Event, ...] = tuple(
            replace(event, sequence=index) for index, event in enumerate(trace.events)
        )
        return replace(trace, events=events, source_format=self.format_name)


class InMemoryPriceBook:
    """Serves prices from entries built in code, with the real book's semantics."""

    name = "in-memory"

    def __init__(self, entries: Iterable[PriceEntry]) -> None:
        self._schedule = PriceSchedule(entries)

    def models(self) -> tuple[str, ...]:
        return self._schedule.models()

    def rate(self, model: str, at: date | None = None) -> Rates:
        return self._schedule.rate(model, at)


class FixedClock:
    """A clock that returns a fixed time, advanced explicitly by tests."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now = self._now + delta


class SystemClock:
    """The real wall clock."""

    def now(self) -> datetime:
        return datetime.now().astimezone()


class InMemoryEventLog:
    """An :class:`~ter.ports.driven.EventLog` held in a dict, for tests."""

    def __init__(self) -> None:
        self._events: dict[str, list[Event]] = {}

    def append(self, event: Event) -> None:
        self._events.setdefault(event.session_id, []).append(event)

    def events(self, session_id: str) -> tuple[Event, ...]:
        return tuple(self._events.get(session_id, ()))

    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted(self._events))


class FixedTerScorer:
    """A :class:`~ter.ports.driven.TerScorer` fake returning preset scores by reference."""

    def __init__(self, scores: Mapping[str, float], method: str = "fixed") -> None:
        self._scores = dict(scores)
        self.method = method

    def score(self, ref: str | Path) -> float:
        key = str(ref)
        if key not in self._scores:
            raise FileNotFoundError(f"No fixed TER for {key!r}")
        return self._scores[key]


class InMemoryOutcomeSource:
    """An :class:`~ter.ports.driven.OutcomeSource` serving evidence built in code.

    ``malformed`` names references whose record exists but cannot be read,
    so the format-error obligation is testable without files.
    """

    name = "in-memory"

    def __init__(
        self,
        records: Mapping[str, OutcomeEvidence],
        malformed: Iterable[str] = (),
    ) -> None:
        self._records = dict(records)
        self._malformed = frozenset(malformed)

    def outcome(self, ref: str | Path) -> OutcomeEvidence | None:
        key = str(ref)
        if key in self._malformed:
            raise OutcomeFormatError(f"{key}: in-memory record marked malformed")
        return self._records.get(key)


class InMemoryRepositoryEvidence:
    """A :class:`~ter.ports.driven.RepositoryEvidence` over files held in code.

    ``imports`` gives, per Python file, the absolute modules its import
    statements name (``import a.b`` -> ``"a.b"``; ``from a import b`` ->
    ``"a"`` and ``"a.b"``), in place of reading them from text; the
    test-to-source rule itself is the domain's, shared with the real engines.
    ``structures``, ``diff`` and ``histories`` are served as given; an engine
    built without them answers ``None``, as one without that evidence must.
    ``parse`` stands in for a syntax tree of text that is not a listed
    file's own (``structure_of``).
    """

    def __init__(
        self,
        files: Mapping[str, str],
        *,
        imports: Mapping[str, Iterable[str]] | None = None,
        structures: Mapping[str, SourceStructure] | None = None,
        diff: RepositoryDiff | None = None,
        histories: Mapping[str, Iterable[FileCommit]] | None = None,
        parse: Callable[[str, str], SourceStructure | None] | None = None,
        name: str = "in-memory",
    ) -> None:
        self.name = name
        self._parse = parse
        self._files = dict(files)
        self._imports = {p: tuple(m) for p, m in (imports or {}).items()}
        self._structures = dict(structures) if structures is not None else None
        self._diff = diff
        self._histories = (
            {p: tuple(h) for p, h in histories.items()}
            if histories is not None
            else None
        )

    def files(self) -> tuple[str, ...]:
        return tuple(sorted(self._files))

    def text(self, path: str) -> str:
        if path not in self._files:
            raise UnknownPathError(f"{path!r} is not a file of the repository")
        return self._files[path]

    def search(self, needle: str, *, regex: bool = False) -> tuple[TextMatch, ...]:
        if not needle:
            raise RepositoryEvidenceError("search needs a non-empty needle")
        pattern = re.compile(needle if regex else re.escape(needle))
        return tuple(
            TextMatch(path, number, line)
            for path in self.files()
            for number, line in enumerate(
                self._files[path].removesuffix("\n").split("\n"), start=1
            )
            if self._files[path] and pattern.search(line.removesuffix("\r"))
        )

    def tests_importing(self, path: str) -> tuple[str, ...]:
        self.text(path)
        files = frozenset(self._files)
        return repository_tests_importing(module_name(path, files), self._imports)

    def structure(self, path: str) -> SourceStructure | None:
        self.text(path)
        return None if self._structures is None else self._structures.get(path)

    def structure_of(self, path: str, text: str) -> SourceStructure | None:
        """A listed file's structure for its own text; ``parse`` (when given)
        for any other text."""
        if path in self._files and text == self._files[path]:
            return self.structure(path)
        return None if self._parse is None else self._parse(path, text)

    def diff(self) -> RepositoryDiff | None:
        return self._diff

    def history(self, path: str) -> tuple[FileCommit, ...] | None:
        self.text(path)
        return None if self._histories is None else self._histories.get(path, ())


class InMemoryArchitectureContracts:
    """A :class:`~ter.ports.driven.ArchitectureContracts` serving contracts
    held in code, by the text they were declared in.

    ``declared`` maps a repository path to ``{text: contracts}``; any other
    text of a source file declares nothing, and a text listed in ``broken``
    raises :class:`ContractFormatError`, as a real reader does.
    """

    name = "in-memory"

    def __init__(
        self,
        declared: Mapping[str, Mapping[str, Iterable[ArchitectureContract]]],
        *,
        broken: Iterable[str] = (),
    ) -> None:
        self._declared = {
            path: {text: tuple(c) for text, c in by_text.items()}
            for path, by_text in declared.items()
        }
        self._broken = frozenset(broken)

    def sources(self) -> tuple[str, ...]:
        return tuple(self._declared)

    def read(self, path: str, text: str) -> tuple[ArchitectureContract, ...]:
        if text in self._broken:
            raise ContractFormatError(f"{path}: cannot read its contracts")
        return self._declared.get(path, {}).get(text, ())


class InMemoryRoutingProfiles:
    """A :class:`~ter.ports.driven.RoutingProfiles` serving profiles built in
    code, with the real adapter's semantics (TER-RTE-001)."""

    name = "in-memory"

    def __init__(self, profiles: Iterable[RoutingProfile], default: str | None = None):
        self._profiles = {p.name: p for p in profiles}
        if not self._profiles:
            raise RoutingProfileError("in-memory: no routing profiles")
        self._default = default if default is not None else min(self._profiles)
        if self._default not in self._profiles:
            raise RoutingProfileError(f"in-memory: no profile {self._default!r}")

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def default(self) -> str:
        return self._default

    def profile(self, name: str) -> RoutingProfile:
        try:
            return self._profiles[name]
        except KeyError:
            raise RoutingProfileError(f"Unknown routing profile {name!r}") from None


class InMemoryControlLimits:
    """A :class:`~ter.ports.driven.ControlLimitsSource` serving limits built in
    code, keyed by ref, with the real adapter's semantics."""

    name = "in-memory"

    def __init__(self, documents: Mapping[str, ControlLimits]) -> None:
        self._documents = dict(documents)

    def limits(self, ref: str | Path) -> ControlLimits:
        try:
            return self._documents[str(ref)]
        except KeyError:
            raise ControlLimitsError(f"{ref}: no such limits document") from None
