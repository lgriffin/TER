"""The adapter conformance kit (Stage U3, ADR 0008 decision 7, TER-CNF-001 to
TER-CNF-004).

A port's contract is a list of :class:`Obligation`\\ s, each a named check
that raises :class:`Breach` when an adapter falls short. TER's own contract
suites in ``tests/contract/`` run these same obligations against every
built-in adapter and fake, so the kit an adapter author runs is the suite
TER holds itself to, unchanged:

    python -m ter conformance SessionSource --adapter my_pkg:MySource \\
        --sample sessions/one.log --sample sessions/two.log

Ports covered: ``SessionSource`` and ``PriceBook``. ``OutcomeSource`` comes
next (its obligations need runs with known results).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Generic, TypeVar

from ...domain import Actor, EventKind, SessionFormatError
from ...domain.pricing import Rates, UnknownModelError
from ...ports import PriceBook, SessionSource

__all__ = [
    "PORTS",
    "PRICE_BOOK",
    "SESSION_SOURCE",
    "Breach",
    "Obligation",
    "PriceSubject",
    "SessionSubject",
    "Verdict",
    "check",
    "format_verdicts",
]

S = TypeVar("S")


class Breach(Exception):
    """An adapter does not meet an obligation; the message says how."""


def require(condition: object, message: str) -> None:
    """Raise :class:`Breach` with ``message`` unless ``condition`` holds.

    Obligations never use ``assert``, which ``python -O`` removes."""
    if not condition:
        raise Breach(message)


@dataclass(frozen=True)
class Obligation(Generic[S]):
    """One obligation of a port's contract."""

    id: str
    text: str
    check: Callable[[S], None]

    def __call__(self, subject: S) -> None:
        self.check(subject)


@dataclass(frozen=True)
class Verdict:
    obligation: str
    text: str
    met: bool
    detail: str = ""


def check(obligations: Sequence[Obligation[S]], subject: S) -> tuple[Verdict, ...]:
    """Every obligation's verdict on ``subject``. An unexpected exception is
    a failed obligation, never the end of the run."""
    verdicts = []
    for obligation in obligations:
        try:
            obligation(subject)
        except Breach as exc:
            verdicts.append(Verdict(obligation.id, obligation.text, False, str(exc)))
        except Exception as exc:  # noqa: BLE001 - an adapter's crash is a finding
            detail = f"raised {type(exc).__name__}: {exc}"
            verdicts.append(Verdict(obligation.id, obligation.text, False, detail))
        else:
            verdicts.append(Verdict(obligation.id, obligation.text, True))
    return tuple(verdicts)


def format_verdicts(port: str, verdicts: Sequence[Verdict]) -> str:
    lines = [f"{port} conformance"]
    for v in verdicts:
        lines.append(f"  {'PASS' if v.met else 'FAIL'}  {v.obligation:<28} {v.text}")
        if v.detail:
            lines.append(f"        {v.detail}")
    met = sum(v.met for v in verdicts)
    lines.append(f"{met} of {len(verdicts)} obligations met")
    return "\n".join(lines) + "\n"


# -- SessionSource -------------------------------------------------------------


@dataclass(frozen=True)
class SessionSubject:
    """A session source and sessions it should read."""

    adapter: Any
    samples: tuple[str, ...]


def _traces(subject: SessionSubject) -> list[tuple[str, Any]]:
    require(subject.samples, "no sample sessions given")
    return [(ref, subject.adapter.read(ref)) for ref in subject.samples]


def _protocol(subject: SessionSubject) -> None:
    require(
        isinstance(subject.adapter, SessionSource),
        "has no format_name attribute and read(ref) method",
    )
    require(
        isinstance(subject.adapter.format_name, str) and subject.adapter.format_name,
        "format_name is empty",
    )


def _deterministic(subject: SessionSubject) -> None:
    for ref, trace in _traces(subject):
        require(subject.adapter.read(ref) == trace, f"{ref}: a second read differs")


def _ids(subject: SessionSubject) -> None:
    for ref, trace in _traces(subject):
        ids = [e.id for e in trace.events]
        require(len(ids) == len(set(ids)), f"{ref}: event ids repeat")
        require(
            [e.sequence for e in trace.events] == list(range(len(trace.events))),
            f"{ref}: sequences do not run 0..n-1",
        )
        require(
            trace.source_format == subject.adapter.format_name,
            f"{ref}: source_format {trace.source_format!r} is not the adapter's "
            f"format_name {subject.adapter.format_name!r}",
        )


def _chain(subject: SessionSubject) -> None:
    for ref, trace in _traces(subject):
        require(trace.events, f"{ref}: no events")
        sessions = {e.session_id for e in trace.events}
        require(
            sessions == {trace.session_id},
            f"{ref}: events name sessions {sorted(sessions)}, not {trace.session_id!r}",
        )
        require(
            trace.events[0].parent_id is None, f"{ref}: the first event has a parent"
        )
        for earlier, later in zip(trace.events, trace.events[1:]):
            require(
                later.parent_id == earlier.id,
                f"{ref}: event {later.sequence} does not follow event {earlier.sequence}",
            )


def _generated(subject: SessionSubject) -> None:
    for ref, trace in _traces(subject):
        generated = trace.generated()
        require(generated, f"{ref}: no generated (assistant) events")
        require(
            all(e.actor is Actor.ASSISTANT for e in generated),
            f"{ref}: a generated event is not the assistant's",
        )
        kinds = {e.kind for e in generated} & {
            EventKind.PROMPT,
            EventKind.TOOL_COMPLETED,
        }
        require(
            not kinds,
            f"{ref}: generated events include {sorted(k.value for k in kinds)}",
        )


def _missing(subject: SessionSubject) -> None:
    with tempfile.TemporaryDirectory() as folder:
        ref = str(Path(folder) / "missing.jsonl")
        try:
            subject.adapter.read(ref)
        except FileNotFoundError:
            return
        raise Breach("reading a missing file did not raise FileNotFoundError")


#: Inputs no session format accepts.
NOT_SESSIONS: tuple[tuple[str, bytes], ...] = (
    ("empty", b""),
    ("text", b"this is not a session\n"),
    ("not-utf8", b"\xff\xfe\x00\x00{}\n"),
)


def _refuses(subject: SessionSubject) -> None:
    suffixes = sorted({Path(ref).suffix for ref in subject.samples} - {""}) or [
        ".jsonl"
    ]
    with tempfile.TemporaryDirectory() as folder:
        for suffix in suffixes:
            for name, content in NOT_SESSIONS:
                path = Path(folder) / f"{name}{suffix}"
                path.write_bytes(content)
                try:
                    subject.adapter.read(path)
                except SessionFormatError:
                    continue
                except Exception as exc:  # noqa: BLE001
                    raise Breach(
                        f"a {name} {suffix} file raised {type(exc).__name__}, "
                        "not SessionFormatError"
                    ) from None
                raise Breach(f"a {name} {suffix} file was read as a session")


SESSION_SOURCE: tuple[Obligation[SessionSubject], ...] = (
    Obligation("protocol", "fits the SessionSource port", _protocol),
    Obligation(
        "deterministic", "the same input gives an identical trace", _deterministic
    ),
    Obligation("ids", "unique event ids, sequences 0..n-1, its format named", _ids),
    Obligation("chain", "events belong to their session and chain in order", _chain),
    Obligation(
        "generated", "generated events exclude user input and tool output", _generated
    ),
    Obligation("missing", "a missing file raises FileNotFoundError", _missing),
    Obligation(
        "refuses", "a file that is not a session raises SessionFormatError", _refuses
    ),
)


# -- PriceBook -----------------------------------------------------------------


@dataclass(frozen=True)
class PriceSubject:
    """A price book and the day its latest rates are in effect."""

    book: Any
    today: date


#: A day before any model was priced.
BEFORE_PRICES = date(1999, 1, 1)


def _book_protocol(subject: PriceSubject) -> None:
    require(isinstance(subject.book, PriceBook), "has no name, models() and rate()")
    require(isinstance(subject.book.name, str) and subject.book.name, "name is empty")
    require(subject.book.models(), "models() lists no model")


def _valid(subject: PriceSubject) -> None:
    for model in subject.book.models():
        rates = subject.book.rate(model)
        require(
            isinstance(rates, Rates), f"{model}: rate() returned {type(rates).__name__}"
        )
        require(
            rates.output >= rates.input >= rates.cache_read >= 0,
            f"{model}: rates are not output >= input >= cache read >= 0 ({rates})",
        )


def _book_deterministic(subject: PriceSubject) -> None:
    for model in subject.book.models():
        require(
            subject.book.rate(model) == subject.book.rate(model),
            f"{model}: a second lookup differs",
        )


def _latest(subject: PriceSubject) -> None:
    for model in subject.book.models():
        require(
            subject.book.rate(model, subject.today) == subject.book.rate(model),
            f"{model}: the latest rates are not the ones in effect on {subject.today}",
        )


def _unknown(subject: PriceSubject) -> None:
    try:
        subject.book.rate("no-such-model-ter-conformance")
    except UnknownModelError:
        return
    raise Breach("an unknown model did not raise UnknownModelError")


def _before(subject: PriceSubject) -> None:
    for model in subject.book.models():
        try:
            subject.book.rate(model, BEFORE_PRICES)
        except UnknownModelError:
            continue
        raise Breach(
            f"{model}: a date before any price did not raise UnknownModelError"
        )


PRICE_BOOK: tuple[Obligation[PriceSubject], ...] = (
    Obligation("protocol", "fits the PriceBook port and lists models", _book_protocol),
    Obligation("valid", "every listed model has valid rates", _valid),
    Obligation(
        "deterministic", "the same lookup gives the same rates", _book_deterministic
    ),
    Obligation("latest", "the latest rates are in effect today", _latest),
    Obligation("unknown", "an unknown model raises UnknownModelError", _unknown),
    Obligation("before", "a date before any price raises UnknownModelError", _before),
)

#: The ports the kit covers.
PORTS: tuple[str, ...] = ("SessionSource", "PriceBook")
