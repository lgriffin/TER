"""Driven ports: what the TER core needs from the outside world."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..domain.calibration import Calibration
from ..domain.events import Event, SessionTrace

# Declared beside the intent model it serves (the analysis takes it as an
# argument); re-exported here as the driven port adapters implement.
from ..domain.lean.intent import AlignmentScorer as AlignmentScorer
from ..domain.lean.control import ControlLimits
from ..domain.ledger import Intervention, LedgerRecord, Observation
from ..domain.policy_set import PolicySet
from ..domain.outcome import OutcomeEvidence
from ..domain.pricing import Rates
from ..domain.routing import RoutingProfile
from ..domain.repository import (
    ArchitectureContract,
    FileCommit,
    RepositoryDiff,
    SourceStructure,
    TextMatch,
)

if TYPE_CHECKING:
    # Annotation-only, so short-lived entry points (hooks) do not pay for
    # importing numpy just to declare the Embedder port.
    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class SessionSource(Protocol):
    """Reads a recorded agent session and returns it as normalised events.

    Obligations, verified by ``tests/contract/test_session_source.py``:

    * the same input yields an identical trace on every call;
    * event ids are unique within a trace and sequences run 0..n-1;
    * records the adapter cannot map are counted, never silently dropped;
    * input that is not a session of its format raises
      :class:`ter.domain.events.SessionFormatError`.
    """

    format_name: str

    def read(self, ref: str | Path) -> SessionTrace: ...


@runtime_checkable
class Tokenizer(Protocol):
    """Counts tokens in text.

    ``exact`` is True only when the count matches the target model's own
    tokenizer, so reports can state how much to trust token figures.
    """

    name: str
    exact: bool

    def count(self, text: str) -> int: ...


@runtime_checkable
class Embedder(Protocol):
    """Embeds texts as unit-length vectors of a fixed dimension."""

    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return an array of shape ``(len(texts), dimension)``."""
        ...


@runtime_checkable
class Clock(Protocol):
    """Supplies the current time, so time-dependent logic is testable."""

    def now(self) -> datetime: ...


@runtime_checkable
class PriceBook(Protocol):
    """Supplies per-model token prices, by date.

    Obligations, verified by ``tests/contract/test_price_book.py``:

    * ``rate`` accepts every name in ``models()`` and returns the same rates
      on every call;
    * with ``at=None`` it returns the latest rates; with a date, the rates in
      effect on that date;
    * an unknown model, or a date before a model's first price, raises
      ``ter.domain.UnknownModelError``.
    """

    name: str

    def models(self) -> tuple[str, ...]: ...

    def rate(self, model: str, at: date | None = None) -> Rates: ...


@runtime_checkable
class CalibrationSource(Protocol):
    """Supplies the judged samples behind the detectors (TER-REC-004).

    Obligations, verified by ``tests/contract/test_calibration_source.py``:

    * ``calibration()`` returns the same samples on every call;
    * a detector no sample judged reports zero judged findings;
    * a document that is not ``ter.calibration/1`` is refused with
      ``ter.domain.calibration.CalibrationError``, never another error.
    """

    name: str

    def calibration(self) -> Calibration: ...


@runtime_checkable
class EventLog(Protocol):
    """An append-only store of normalised events, grouped by session.

    Live mode appends each event as it arrives; analysis replays the log. The
    log may hold the same event twice (a retried append): consumers rely on
    event ids, not on the log, for idempotency.

    Obligations, verified by ``tests/contract/test_event_log.py``:

    * ``events(session_id)`` returns appended events in append order, equal
      field for field to what was appended;
    * sessions never leak into each other, and an unknown session is empty;
    * ``sessions()`` lists every session with at least one event, sorted.
    """

    def append(self, event: Event) -> None: ...

    def events(self, session_id: str) -> tuple[Event, ...]: ...

    def sessions(self) -> tuple[str, ...]: ...


@runtime_checkable
class TerScorer(Protocol):
    """Scores a recorded session with the TER 3 ratio (point 4: TER is retained).

    ``method`` names how the score was computed (tokenizer and embedder), so
    a report can say how far to trust it next to the Lean scorecard.
    """

    method: str

    def score(self, ref: str | Path) -> float: ...


@runtime_checkable
class OutcomeSource(Protocol):
    """Supplies the outcome evidence a run recorded: one result per check.

    This is the judgement side of point 5. Behaviour measures never read it;
    ``ter.domain.outcome.judge`` turns its evidence into a verdict that
    reports show beside them.

    Obligations, verified by ``tests/contract/test_outcome_source.py``:

    * ``outcome(ref)`` returns the evidence recorded for the run ``ref``
      names, or ``None`` when no outcome is recorded for it. No outcome is
      not an error and never reads as a failing outcome;
    * the same ``ref`` yields equal evidence on every call;
    * the evidence names its run (``run_ref``) and source, and holds one
      ``CheckEvidence`` per recorded result, in record order, each with a
      non-empty check id and where it was recorded; failing, erroring and
      skipped results are kept with that status, never dropped;
    * a record that exists but cannot be read as outcome evidence raises
      ``ter.domain.outcome.OutcomeFormatError`` naming the record.
    """

    name: str

    def outcome(self, ref: str | Path) -> OutcomeEvidence | None: ...


@runtime_checkable
class RepositoryEvidence(Protocol):
    """Evidence about one repository, independent of any model provider.

    TER obtains repository evidence only through this port (TER-EVD-001).
    An adapter is built for one repository root; engines differ in depth
    (lexical text, Git history, syntax trees) and say what they cannot do by
    returning ``None``, never by guessing. Engines load as capabilities,
    ``RepositoryEvidence.<engine>`` (TER-ARC-007).

    Obligations, verified by ``tests/contract/test_repository_evidence.py``:

    * paths are relative to the root, use ``/`` and come sorted; nothing in
      a version control directory is listed; every answer is a pure
      function of the repository content, so equal content yields equal
      evidence wherever the repository lives (TER-EVD-002);
    * ``text(path)`` returns a listed file's text; a path the repository
      does not list raises ``UnknownPathError`` (every method that takes a
      path does);
    * ``search(needle)`` returns every line containing ``needle`` (a regular
      expression when ``regex`` is true), sorted by path then line, from
      every UTF-8 text file;
    * ``tests_importing(path)`` returns every test module (``test_*.py`` or
      ``*_test.py``) whose import statements load the Python module at
      ``path``, sorted (TER-EVD-003); an engine that reads another language
      (``syntax``: TypeScript, JavaScript, Svelte, Vue) does the same for it
      with that language's test conventions (``*.test.*``, ``*.spec.*``,
      ``__tests__/``, ``tests/``) and import resolution (TER-EVD-014); a
      path in a language the engine has no import rule for raises
      ``UnsupportedLanguageError``;
    * ``structure(path)`` returns the file's symbols, imports and call edges
      from its syntax tree, or ``None`` when the engine does not support the
      file's language (TER-EVD-012). An import of a language that imports
      files by path carries the repository paths it may load
      (``ImportEdge.candidates``);
    * ``structure_of(path, text)`` returns what ``structure(path)`` would
      return if the file at ``path`` held ``text``: equal to
      ``structure(path)`` for a listed file's own text, and also served for a
      path the repository does not list (a file a session creates), whose
      module name is read as if it were added. It reads nothing from the
      repository but the file list (and, for import resolution, the
      project configuration such as ``tsconfig.json`` and ``package.json``
      files), so an edit can be judged on its result (TER-EVD-007);
    * ``diff()`` and ``history(path)`` return the working tree's changes and
      a file's commits (newest first), or ``None`` when the engine has no
      version control evidence (TER-EVD-013).
    """

    name: str

    def files(self) -> tuple[str, ...]: ...

    def text(self, path: str) -> str: ...

    def search(self, needle: str, *, regex: bool = False) -> tuple[TextMatch, ...]: ...

    def tests_importing(self, path: str) -> tuple[str, ...]: ...

    def structure(self, path: str) -> SourceStructure | None: ...

    def structure_of(self, path: str, text: str) -> SourceStructure | None: ...

    def diff(self) -> RepositoryDiff | None: ...

    def history(self, path: str) -> tuple[FileCommit, ...] | None: ...


@runtime_checkable
class ArchitectureContracts(Protocol):
    """Reads the architecture contracts a repository declares (import-linter
    style: forbidden imports, layers, independence).

    The reader does no IO: it names the repository files it reads, in order
    of precedence, and parses the text the caller obtained through
    :class:`RepositoryEvidence` (TER-EVD-001), so contracts are read at the
    same commit as the rest of the evidence.

    Obligations, verified by ``tests/contract/test_architecture_contracts.py``:

    * ``sources()`` names the files that may declare contracts, most
      specific first;
    * ``read(path, text)`` returns every contract the text declares, in
      declaration order, with absolute module names; a file that declares
      none returns ``()``;
    * a declaration that cannot be read raises
      ``ter.domain.repository.ContractFormatError`` naming the file;
    * the same text yields equal contracts on every call.
    """

    name: str

    def sources(self) -> tuple[str, ...]: ...

    def read(self, path: str, text: str) -> tuple[ArchitectureContract, ...]: ...


@runtime_checkable
class RoutingProfiles(Protocol):
    """Supplies routing profiles: role names and the models they mean
    (TER-RTE-001).

    Obligations, verified by ``tests/contract/test_routing_profiles.py``:

    * ``names()`` lists every profile, sorted, and includes ``default()``;
    * ``profile(name)`` returns an equal profile on every call, whose task
      kinds and escalations name only roles it defines;
    * an unknown profile name raises ``ter.domain.routing.RoutingProfileError``.
    """

    name: str

    def names(self) -> tuple[str, ...]: ...

    def default(self) -> str: ...

    def profile(self, name: str) -> RoutingProfile: ...


@runtime_checkable
class InterventionChannel(Protocol):
    """Delivers advice to the developer (TER-INT-002, TER-INT-003, TER-INT-014).

    At the L4 ceiling a channel delivers advice only: nothing it does changes
    what the agent reads. Each delivery shows the advice with the signal and
    evidence behind it (:meth:`ter.domain.ledger.Intervention.message`).

    Obligations, verified by ``tests/contract/test_intervention_channel.py``:

    * ``deliver`` takes only an advised record (an
      :class:`~ter.domain.ledger.Intervention` cannot be built from a
      suppressed one) and keeps deliveries in order;
    * ``delivered()`` returns the messages delivered so far, each naming the
      policy's advice, the signal's detector and confidence, and its evidence.
    """

    name: str

    def deliver(self, intervention: Intervention) -> None: ...

    def delivered(self) -> tuple[str, ...]: ...


@runtime_checkable
class InterventionLedger(Protocol):
    """The intervention ledger: one record per engine decision (TER-INT-009,
    points 175 to 178, 180), each delivered intervention's outcome attached
    once it is observed.

    Obligations, verified by ``tests/contract/test_intervention_ledger.py``:

    * ``entries(session_id)`` returns recorded records in record order, equal
      field for field to what was recorded, each with its observation if one
      was attached;
    * recording a record id again changes nothing: the first record stays;
    * ``observe`` attaches an outcome to a delivered record once, and raises
      ``ter.domain.ledger.LedgerError`` for an unknown record, a suppressed
      one, or one already observed;
    * sessions never leak into each other, an unknown session is empty, and
      ``sessions()`` lists every session with a record, sorted.
    """

    name: str

    def record(self, entry: LedgerRecord) -> None: ...

    def observe(
        self, session_id: str, record_id: str, observation: Observation
    ) -> None: ...

    def entries(self, session_id: str) -> tuple[LedgerRecord, ...]: ...

    def sessions(self) -> tuple[str, ...]: ...


@runtime_checkable
class InterventionPolicies(Protocol):
    """Supplies an intervention policy file (``ter.intervention-policies/1``):
    declarative policies, the eligibility list and the human approval record
    (TER-INT-008, TER-INT-015, points 163, 164, 173).

    Obligations, verified by ``tests/contract/test_intervention_policies.py``:

    * ``policies(ref)`` returns equal policy sets on every call for the same
      ref, approval record included;
    * a ref it cannot find, or a document that is not a valid policy file,
      raises ``ter.domain.intervention.PolicyError``; it never returns some
      of the policies;
    * it never approves: a file without an approval record, or changed since
      it was approved, is returned as such, and ``PolicySet.engine`` refuses
      it.
    """

    name: str

    def policies(self, ref: str | Path) -> PolicySet: ...


@runtime_checkable
class ControlLimitsSource(Protocol):
    """Supplies a developer's control limits document (``ter.control-limits/1``):
    natural process limits per measure, any tuned action limits with their
    reasons, and the rules switched on (point 201).

    Obligations, verified by ``tests/contract/test_control_limits.py``:

    * ``limits(ref)`` returns equal limits on every call for the same ref;
    * a ref it cannot find, or a document that is not valid limits, raises
      ``ter.domain.lean.control.ControlLimitsError``; it never returns
      partial limits.
    """

    name: str

    def limits(self, ref: str | Path) -> ControlLimits: ...
