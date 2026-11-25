"""Persistent intent and alignment (points 6, 7, 45 to 50).

The developer's intent is a first-class, evolving record rather than the
first prompt. Every ``intent.stated`` event updates the session's one
:class:`IntentRecord` (TER-ITN-001): it opens the record, refines it, is
acknowledged without changing it, or *changes* it, in which case the change
is recorded with the prompt that caused it (TER-ITN-002).

Each agent event (reasoning, tool request, response) is scored against the
intent in force when it happened (TER-ITN-004). Scoring sits behind the
:class:`AlignmentScorer` protocol, so an embedding adapter can replace the
deterministic default, :class:`LexicalAlignment`, which compares *key terms*:
the content words, identifiers and file-name parts of the intent and of the
event. A run of agent events scoring below a configured band, at least a
configured number of events long, is a :class:`LowAlignmentPeriod`
(TER-ITN-005). Every threshold is a similarity band or a count of events,
never a token count.

Drift (TER-ITN-003) is a waste finding: the ``intent_drift`` detector in
:mod:`.detectors` reads this timeline through
:attr:`~.detectors.SessionView.intent`.

:class:`IntentLog` is the per-event fold (O(size of the event), idempotent
through :class:`~.steps.StepLog`); :func:`build_intent` replays the stored
terms into the timeline when an analysis is asked for, so batch and
incremental analysis agree by construction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ..events import Actor, Event, EventId, EventKind, ToolKind
from .facts import STOPWORDS, defined_identifiers, injected_prompt, overlap
from .model import Step

__all__ = [
    "DEFAULT_INTENT_CONFIG",
    "LEXICAL_ALIGNMENT",
    "Alignment",
    "AlignmentBand",
    "AlignmentScorer",
    "IntentChange",
    "IntentConfig",
    "IntentLog",
    "IntentRecord",
    "IntentRelation",
    "IntentRevision",
    "IntentTimeline",
    "LexicalAlignment",
    "LowAlignmentPeriod",
    "SubjectBasis",
    "build_intent",
    "key_terms",
]

#: A prompt with fewer goal terms than this states no goal (``ok``, ``go on``).
MIN_GOAL_TERMS = 2
#: An unmarked prompt with at least this many goal terms, scoring below
#: :data:`CHANGE_BELOW` against the current intent, states a different goal.
UNRELATED_GOAL_TERMS = 3
CHANGE_BELOW = 0.2
#: A redirect marker (``Actually,``, ``Instead``…) changes the intent unless
#: the prompt still scores at least this against it (then it refines it).
REDIRECT_KEEPS_ABOVE = 0.5
#: Alignment at or above this is *aligned*.
ALIGNED_FROM = 0.5


# ---------------------------------------------------------------------------
# Key terms
# ---------------------------------------------------------------------------

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ALNUM = re.compile(r"[A-Za-z][A-Za-z0-9]*")
# Words that say how, not what: generic verbs of programming, language
# keywords, path noise, and the redirect vocabulary itself.
_GENERIC = STOPWORDS | frozenset(
    """
    add write create implement change update fix function method class module
    code new also instead actually please want like src lib def self none true
    false return import from print python elif pass lambda forget drop stop
    scratch mind longer rather okay yes yeah thanks thank has have had
    """.split()
)
_SENTENCE = re.compile(r"(?<=[.!?;\n])\s+|\n+")
_REDIRECT = re.compile(
    r"^\W*(?:actually|instead|forget|never\s*mind|scratch\s+that|change\s+of\s+plans?"
    r"|new\s+task|different\s+task|stop|drop|let'?s\s+switch|switch\s+to|rather)\b",
    re.IGNORECASE,
)
#: A sentence that gives a goal up: 'forget the cache', 'never mind the
#: docs', 'we no longer need X', 'stop working on Y'. A bare 'don't', 'do
#: not' or 'stop' is a rule for doing the work ('don't push', 'stop at the
#: first failure'), not a dropped goal: read that way on the owner's
#: sessions (25 Nov 2026) its words became 'dropped goals', and two later
#: test edits that shared one were confident drift.
_ABANDON = re.compile(
    r"\b(?:forget(?:\s+about)?|never\s*mind|scratch\s+that"
    r"|drop\s+(?:it|that|this|the\s+idea)"
    r"|no\s+longer\s+(?:need|want|needed|wanted|necessary|required)"
    r"|(?:don'?t|do\s+not)\s+(?:bother|need\s+to|worry\s+about)"
    r"|stop\s+(?:working\s+on|doing|trying|with))\b",
    re.IGNORECASE,
)
_EXTRA = re.compile(
    r"\b(?:also|additionally|as\s+well|while\s+(?:i'?m|we'?re|i\s+am)\s+at\s+it"
    r"|for\s+completeness|bonus|extra)\b",
    re.IGNORECASE,
)


def _stem(word: str) -> str:
    """A light, consistent stem: ``parsing``, ``parsed``, ``parses`` and
    ``parse`` all become ``pars``; ``retries`` and ``retry`` become ``retry``."""
    if len(word) > 4 and word.endswith(("ies", "ied")):
        return word[:-3] + "y"
    for suffix, keep in (("ing", 3), ("ed", 3), ("es", 3), ("s", 3)):
        if (
            word.endswith(suffix)
            and len(word) - len(suffix) >= keep
            and not (suffix == "s" and word.endswith(("ss", "us", "is")))
        ):
            word = word[: -len(suffix)]
            if suffix in ("ing", "ed") and len(word) > 3 and word[-1] == word[-2]:
                if word[-1] not in "lsz":
                    word = word[:-1]
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


_GENERIC_STEMS = _GENERIC | frozenset(_stem(w) for w in _GENERIC)


def key_terms(text: str) -> frozenset[str]:
    """What a text is about: lower-cased, lightly stemmed word parts.

    Identifiers split at ``_``, ``.``, ``/`` and camel case
    (``ValueError`` → ``value``, ``error``); stopwords, generic programming
    words and parts shorter than three letters are dropped.
    """
    out: set[str] = set()
    for match in _ALNUM.finditer(_CAMEL.sub(" ", text)):
        word = match.group(0).lower()
        if word in _GENERIC:
            continue
        stem = _stem(word)
        if len(stem) >= 3 and stem not in _GENERIC_STEMS:
            out.add(stem)
    return frozenset(out)


# ---------------------------------------------------------------------------
# The scoring port and its deterministic default
# ---------------------------------------------------------------------------


@runtime_checkable
class AlignmentScorer(Protocol):
    """Scores how far an agent event's key terms are about the intent's.

    ``score`` returns a value in [0, 1]: 0 when either side is empty, and the
    same value for the same inputs every time. ``rule`` is published with
    the analysis, like a detector's confidence rule.
    """

    @property
    def name(self) -> str: ...

    @property
    def rule(self) -> str: ...

    def score(self, intent: frozenset[str], activity: frozenset[str]) -> float: ...


@dataclass(frozen=True)
class LexicalAlignment:
    """The default scorer: the overlap coefficient of the two term sets."""

    name: str = "lexical"
    rule: str = (
        "Overlap coefficient |intent ∩ activity| / min(|intent|, |activity|) "
        "of key terms (content words, identifier parts and file-name parts, "
        "stemmed, without stopwords or generic programming words); 0 when "
        "either side has no terms."
    )

    def score(self, intent: frozenset[str], activity: frozenset[str]) -> float:
        return overlap(intent, activity)


LEXICAL_ALIGNMENT = LexicalAlignment()


@dataclass(frozen=True)
class IntentConfig:
    """Structural thresholds: similarity bands and counts of events."""

    low_below: float = 0.25
    min_events: int = 3
    drift_below: float = 0.25

    def __post_init__(self) -> None:
        for name in ("low_below", "drift_below"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if self.min_events < 1:
            raise ValueError(f"min_events must be at least 1, got {self.min_events}")

    def as_dict(self) -> dict[str, object]:
        return {
            "low_below": self.low_below,
            "min_events": self.min_events,
            "drift_below": self.drift_below,
        }


DEFAULT_INTENT_CONFIG = IntentConfig()


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class IntentRelation(StrEnum):
    """How one ``intent.stated`` event updated the intent record."""

    OPENED = "opened"
    REFINED = "refined"
    ACKNOWLEDGED = "acknowledged"
    CHANGED = "changed"


class SubjectBasis(StrEnum):
    """What an agent event's alignment was read from."""

    NAMES = "names"  # names an edit or write defines that were not there before
    CHANGED_WORDS = "changed_words"  # words an edit adds
    TEXT = "text"  # the event's text or arguments


class AlignmentBand(StrEnum):
    ALIGNED = "aligned"
    PARTIAL = "partial"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentRevision:
    """The intent after one ``intent.stated`` event."""

    revision: int
    event_id: EventId
    relation: IntentRelation
    terms: frozenset[str]
    abandoned: frozenset[str]
    similarity: float | None
    prompt: str

    def as_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "event_id": self.event_id,
            "relation": self.relation.value,
            "terms": sorted(self.terms),
            "abandoned": sorted(self.abandoned),
            "similarity": None
            if self.similarity is None
            else round(self.similarity, 4),
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class IntentChange:
    """A goal the developer stated that differs from the intent in force."""

    event_id: EventId
    from_revision: int
    to_revision: int
    reason: str
    abandoned: frozenset[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "from_revision": self.from_revision,
            "to_revision": self.to_revision,
            "reason": self.reason,
            "abandoned": sorted(self.abandoned),
        }


@dataclass(frozen=True)
class Alignment:
    """One agent event scored against the intent revision in force."""

    event_id: EventId
    index: int
    revision: int | None
    score: float | None
    band: AlignmentBand
    basis: SubjectBasis
    subject: frozenset[str]
    names: tuple[str, ...]
    extra: bool = False
    #: Key terms and written words of the sentences announcing extra work.
    extra_terms: frozenset[str] = frozenset()
    extra_words: frozenset[str] = frozenset()

    def announces(self, change: Alignment) -> bool:
        """Whether this narration announced ``change`` as extra work.

        The announcing sentence must name it: spell out a distinctive name
        the change defines (:func:`distinctive`) or at least two of its
        plain-word names that are at least half of its names ('also add mean
        and variance'), or, for a change that defines
        none, share at least two of its key terms. Reviewed on the owner's sessions (25 Nov 2026,
        by Claude; not yet the owner's judge): 14 of 14 findings matched on
        any word of the whole block were not waste (a summary that 'also
        checked' something, one shared word such as 'send').
        """
        if not self.extra:
            return False
        if change.names:
            named = [n for n in change.names if n in self.extra_words]
            if any(distinctive(n) for n in named):
                return True
            # Plain words name the change only when they are most of it: a
            # new module defining dozens of plain names ('out', 'fees')
            # shares two with almost any sentence.
            return len(named) >= 2 and 2 * len(named) >= len(change.names)
        return len(self.extra_terms & change.subject) >= 2

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "revision": self.revision,
            "score": None if self.score is None else round(self.score, 4),
            "band": self.band.value,
            "basis": self.basis.value,
            "names": list(self.names),
        }


@dataclass(frozen=True)
class LowAlignmentPeriod:
    """Consecutive agent events below the low band, long enough to report."""

    revision: int
    events: tuple[EventId, ...]
    mean_score: float

    @property
    def start(self) -> EventId:
        return self.events[0]

    @property
    def end(self) -> EventId:
        return self.events[-1]

    def as_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "start": self.start,
            "end": self.end,
            "events": list(self.events),
            "mean_score": round(self.mean_score, 4),
        }


@dataclass(frozen=True)
class IntentRecord:
    """The session's one intent record and its full history (TER-ITN-001)."""

    revisions: tuple[IntentRevision, ...] = ()

    @property
    def current(self) -> IntentRevision | None:
        return self.revisions[-1] if self.revisions else None

    def revision(self, number: int) -> IntentRevision:
        return self.revisions[number - 1]


@dataclass(frozen=True)
class IntentTimeline:
    """Intent record, changes, per-event alignment and low-alignment periods."""

    record: IntentRecord = field(default_factory=IntentRecord)
    changes: tuple[IntentChange, ...] = ()
    alignments: tuple[Alignment, ...] = ()
    periods: tuple[LowAlignmentPeriod, ...] = ()
    scorer: str = LEXICAL_ALIGNMENT.name
    rule: str = LEXICAL_ALIGNMENT.rule
    config: IntentConfig = DEFAULT_INTENT_CONFIG
    _by_id: Mapping[EventId, Alignment] = field(
        default_factory=dict, repr=False, compare=False
    )

    def alignment_of(self, event_id: EventId) -> Alignment | None:
        return self._by_id.get(event_id)

    def as_dict(self, drift: Sequence[str] = ()) -> dict[str, object]:
        return {
            "scorer": self.scorer,
            "rule": self.rule,
            "config": self.config.as_dict(),
            "revisions": [r.as_dict() for r in self.record.revisions],
            "changes": [c.as_dict() for c in self.changes],
            "alignment": [a.as_dict() for a in self.alignments],
            "low_alignment_periods": [p.as_dict() for p in self.periods],
            "drift": list(drift),
        }


# ---------------------------------------------------------------------------
# The fold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Read:
    """Intent facts of one step, read from its event alone."""

    terms: frozenset[str]
    basis: SubjectBasis = SubjectBasis.TEXT
    names: tuple[str, ...] = ()
    abandoned: frozenset[str] = frozenset()
    redirect: str | None = None
    extra: bool = False
    #: Key terms of the sentences that announce extra work.
    extra_terms: frozenset[str] = frozenset()
    #: Their words as written, to find a name they spell out.
    extra_words: frozenset[str] = frozenset()


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: A name no sentence uses as a plain word: snake_case, an inner capital
#: (camelCase, PascalCase), a digit, or a CONSTANT.
_DISTINCTIVE = re.compile(r"^_*[A-Za-z]\w*(?:_|\d|[a-z][A-Z])|^[A-Z][A-Z0-9_]{2,}$")


def distinctive(name: str) -> bool:
    """Whether a defined name can only mean the code it names.

    ``out``, ``item`` or ``fees`` are words any sentence uses; ``FeeModel``,
    ``parse_auctions`` or ``MAX_QTY`` are not, and neither is a dunder such
    as ``__init__``. Reviewed 25 Nov 2026 on the owner's sessions: a
    greenfield build matched announcements on plain-word names (an agent
    'also checking the fees' while writing ``fees``), none of it extra work.
    """
    if name.startswith("__") and name.endswith("__"):
        return False
    return len(name) >= 4 and bool(_DISTINCTIVE.search(name))


def _read_narration(text: str) -> _Read:
    """Key terms of a reasoning or response block, and of the sentences in it
    that announce extra work ('also', 'while I'm at it').

    Only those sentences say what the extra work is: a long block that uses
    'also' once is not about everything it mentions (judged 25 Nov 2026).
    """
    cues = [s for s in _SENTENCE.split(text) if _EXTRA.search(s)]
    return _Read(
        key_terms(text),
        extra=bool(cues),
        extra_terms=key_terms(" ".join(cues)),
        extra_words=frozenset(w for s in cues for w in _IDENTIFIER.findall(s)),
    )


def _text_values(arguments: Mapping[str, object]) -> Iterable[str]:
    for value in arguments.values():
        if isinstance(value, str):
            yield value
        elif isinstance(value, (int, float)):
            yield str(value)


def _edit_pairs(arguments: Mapping[str, object]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    old, new = arguments.get("old_string"), arguments.get("new_string")
    if isinstance(new, str):
        pairs.append((old if isinstance(old, str) else "", new))
    edits = arguments.get("edits")
    if isinstance(edits, list):
        for edit in edits:
            if isinstance(edit, Mapping):
                o, n = edit.get("old_string"), edit.get("new_string")
                if isinstance(n, str):
                    pairs.append((o if isinstance(o, str) else "", n))
    return pairs


def _change_terms(old: str, new: str) -> _Read:
    """The subject of a change: names it defines, else words it adds."""
    names = tuple(sorted(defined_identifiers(new) - defined_identifiers(old)))
    if names:
        return _Read(key_terms(" ".join(names)), SubjectBasis.NAMES, names)
    added = key_terms(new) - key_terms(old)
    if added:
        return _Read(added, SubjectBasis.CHANGED_WORDS)
    return _Read(key_terms(new))


def _read_request(kind: ToolKind, arguments: Mapping[str, object]) -> _Read:
    if kind is ToolKind.FS_EDIT:
        pairs = _edit_pairs(arguments)
        if pairs:
            return _change_terms(
                "\n".join(o for o, _ in pairs), "\n".join(n for _, n in pairs)
            )
    if kind is ToolKind.FS_WRITE:
        content = arguments.get("content")
        if isinstance(content, str):
            return _change_terms("", content)
    return _Read(key_terms(" ".join(_text_values(arguments))))


def _read_prompt(text: str) -> _Read:
    sentences = [s for s in _SENTENCE.split(text) if s.strip()]
    kept: set[str] = set()
    dropped: set[str] = set()
    for sentence in sentences:
        (dropped if _ABANDON.search(sentence) else kept).update(key_terms(sentence))
    redirect = next(
        (m.group(0).strip(" ,") for s in sentences if (m := _REDIRECT.match(s))),
        None,
    )
    return _Read(
        frozenset(kept),
        abandoned=frozenset(dropped - kept),
        redirect=redirect,
    )


class IntentLog:
    """Reads each step's intent facts as it arrives: one entry per step."""

    def __init__(self) -> None:
        self._reads: list[_Read | None] = []

    def __len__(self) -> int:
        return len(self._reads)

    def add(self, event: Event) -> None:
        """Read one event that :class:`~.steps.StepLog` turned into a step."""
        self._reads.append(self._read(event))

    @staticmethod
    def _read(event: Event) -> _Read | None:
        if event.kind is EventKind.PROMPT:
            # A notification, slash-command wrapper or interruption marker
            # the harness wrote states no intent.
            if injected_prompt(event.text):
                return None
            return _read_prompt(event.text)
        if event.actor is not Actor.ASSISTANT or not event.kind.is_generated:
            return None
        if event.kind is EventKind.TOOL_REQUESTED:
            if event.tool is None:
                return _Read(key_terms(event.text))
            return _read_request(event.tool.kind, event.tool.arguments)
        return _read_narration(event.text)

    def reads(self) -> tuple[_Read | None, ...]:
        return tuple(self._reads)


# ---------------------------------------------------------------------------
# The timeline
# ---------------------------------------------------------------------------


def _band(score: float | None, config: IntentConfig) -> AlignmentBand:
    if score is None:
        return AlignmentBand.UNKNOWN
    if score >= ALIGNED_FROM:
        return AlignmentBand.ALIGNED
    if score >= config.low_below:
        return AlignmentBand.PARTIAL
    return AlignmentBand.LOW


def _update(
    current: IntentRevision | None,
    step: Step,
    read: _Read,
    scorer: AlignmentScorer,
) -> tuple[IntentRevision, IntentChange | None]:
    revision = 1 if current is None else current.revision + 1
    prompt = " ".join(step.subject.split())

    def make(
        relation: IntentRelation, terms: frozenset[str], similarity: float | None
    ) -> IntentRevision:
        return IntentRevision(
            revision, step.event_id, relation, terms, read.abandoned, similarity, prompt
        )

    if current is None:
        return make(IntentRelation.OPENED, read.terms, None), None
    goal = read.terms
    similarity = scorer.score(current.terms, goal) if current.terms and goal else None
    if not current.terms:
        return make(IntentRelation.REFINED, goal, similarity), None
    if len(goal) < MIN_GOAL_TERMS:
        # A short prompt states no new goal, but what it explicitly abandons
        # ("Forget the median.") still leaves the intent in force.
        kept = current.terms - read.abandoned
        return make(IntentRelation.ACKNOWLEDGED, kept, similarity), None
    if read.redirect is not None and (
        similarity is None or similarity < REDIRECT_KEEPS_ABOVE or read.abandoned
    ):
        reason = f"explicit redirect ({read.redirect!r})"
    elif (
        read.redirect is None
        and len(goal) >= UNRELATED_GOAL_TERMS
        and similarity is not None
        and similarity < CHANGE_BELOW
    ):
        reason = f"unrelated goal (similarity {similarity:.2f} < {CHANGE_BELOW})"
    else:
        terms = (current.terms | goal) - read.abandoned
        return make(IntentRelation.REFINED, terms, similarity), None
    changed = make(IntentRelation.CHANGED, goal, similarity)
    return changed, IntentChange(
        step.event_id, current.revision, revision, reason, read.abandoned
    )


def _periods(
    alignments: Sequence[Alignment],
    steps: Sequence[Step],
    config: IntentConfig,
    stated: frozenset[int],
) -> tuple[LowAlignmentPeriod, ...]:
    out: list[LowAlignmentPeriod] = []
    run: list[Alignment] = []

    def close() -> None:
        if len(run) >= config.min_events and run[0].revision is not None:
            mean = sum(a.score or 0.0 for a in run) / len(run)
            out.append(
                LowAlignmentPeriod(
                    run[0].revision, tuple(a.event_id for a in run), mean
                )
            )
        run.clear()

    by_index = {a.index: a for a in alignments}
    for step in steps:
        if step.kind is EventKind.PROMPT:
            # Only a prompt that stated intent ends a period: a harness
            # notification arriving mid-drift does not (TER-ITN-008).
            if step.index in stated:
                close()
            continue
        a = by_index.get(step.index)
        if a is None or a.band is AlignmentBand.UNKNOWN:
            continue  # tool results and unscorable events neither extend nor break
        if a.band is AlignmentBand.LOW:
            run.append(a)
        else:
            close()
    close()
    return tuple(out)


def build_intent(
    steps: Sequence[Step],
    reads: Sequence[_Read | None],
    *,
    scorer: AlignmentScorer = LEXICAL_ALIGNMENT,
    config: IntentConfig = DEFAULT_INTENT_CONFIG,
) -> IntentTimeline:
    """Replay the stored intent facts into the session's intent timeline."""
    revisions: list[IntentRevision] = []
    changes: list[IntentChange] = []
    alignments: list[Alignment] = []
    current: IntentRevision | None = None
    for step, read in zip(steps, reads, strict=True):
        if read is None:
            continue
        if step.kind is EventKind.PROMPT:
            current, change = _update(current, step, read, scorer)
            revisions.append(current)
            if change is not None:
                changes.append(change)
            continue
        score: float | None = None
        if current is not None and current.terms and read.terms:
            score = max(0.0, min(1.0, scorer.score(current.terms, read.terms)))
        alignments.append(
            Alignment(
                event_id=step.event_id,
                index=step.index,
                revision=None if current is None else current.revision,
                score=score,
                band=_band(score, config),
                basis=read.basis,
                subject=read.terms,
                names=read.names,
                extra=read.extra,
                extra_terms=read.extra_terms,
                extra_words=read.extra_words,
            )
        )
    return IntentTimeline(
        record=IntentRecord(tuple(revisions)),
        changes=tuple(changes),
        alignments=tuple(alignments),
        periods=_periods(
            alignments,
            steps,
            config,
            frozenset(
                s.index
                for s, r in zip(steps, reads, strict=True)
                if s.kind is EventKind.PROMPT and r is not None
            ),
        ),
        scorer=scorer.name,
        rule=scorer.rule,
        config=config,
        _by_id={a.event_id: a for a in alignments},
    )
