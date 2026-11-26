"""The adapter conformance kit (Stage U3, ADR 0008 decision 7, P211).

The contract suites in tests/contract run the kit's obligations against
TER's own adapters; these tests show the kit failing adapters that fall
short, naming the obligation and how.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.in_memory import InMemoryPriceBook
from ter.adapters.driving.cli import CliServices
from ter.adapters.driving.cli import main as cli_main
from ter.adapters.driving.conformance import (
    PORTS,
    PRICE_BOOK,
    SESSION_SOURCE,
    Breach,
    Obligation,
    PriceSubject,
    SessionSubject,
    Verdict,
    check,
    format_verdicts,
    require,
)
from ter.domain import SessionFormatError
from ter.domain.capabilities import CapabilityError
from ter.domain.pricing import PriceEntry, Rates, UnknownModelError

from tests.golden.corpus import CORPUS

SAMPLES = tuple(str(p) for p in list(CORPUS.values())[:2])
RATES = Rates(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
PRICED = date(2026, 1, 1)
TODAY = date(2026, 11, 25)


def _verdicts(adapter: Any, samples: tuple[str, ...] = SAMPLES) -> dict[str, Verdict]:
    return {
        v.obligation: v for v in check(SESSION_SOURCE, SessionSubject(adapter, samples))
    }


def _failed(verdicts: dict[str, Verdict]) -> dict[str, str]:
    return {k: v.detail for k, v in verdicts.items() if not v.met}


# -- the kit itself ------------------------------------------------------------


@pytest.mark.req("TER-CNF-001")
def test_a_real_session_source_meets_every_obligation() -> None:
    verdicts = _verdicts(ClaudeCodeJsonlSource())
    assert list(verdicts) == [o.id for o in SESSION_SOURCE]
    assert _failed(verdicts) == {}
    assert len(SESSION_SOURCE) == 7


@pytest.mark.req("TER-CNF-003")
def test_require_raises_a_breach_only_when_the_condition_fails() -> None:
    require(True, "never")
    require([1], "never")
    with pytest.raises(Breach, match="^said so$"):
        require([], "said so")


@pytest.mark.req("TER-CNF-003")
def test_check_runs_every_obligation_whatever_the_others_do() -> None:
    def breach(subject: int) -> None:
        raise Breach(f"short by {subject}")

    def crash(subject: int) -> None:
        raise KeyError("boom")

    def fine(subject: int) -> None:
        return None

    verdicts = check(
        [
            Obligation("a", "A", breach),
            Obligation("b", "B", crash),
            Obligation("c", "C", fine),
        ],
        3,
    )
    assert verdicts == (
        Verdict("a", "A", False, "short by 3"),
        Verdict("b", "B", False, "raised KeyError: 'boom'"),
        Verdict("c", "C", True, ""),
    )


@pytest.mark.req("TER-CNF-003")
def test_the_report_names_each_obligation_and_counts_them() -> None:
    text = format_verdicts(
        "SessionSource x",
        [
            Verdict("ids", "unique ids", True),
            Verdict("chain", "in order", False, "event 3"),
        ],
    )
    assert text == (
        "SessionSource x conformance\n"
        "  PASS  ids                          unique ids\n"
        "  FAIL  chain                        in order\n"
        "        event 3\n"
        "1 of 2 obligations met\n"
    )


def test_the_kit_covers_the_ports_the_adr_names_first() -> None:
    assert PORTS == ("SessionSource", "PriceBook")


# -- SessionSource adapters that fall short ------------------------------------


class _Wrapping:
    """A session source that changes what a real one reads."""

    format_name = "claude-code-jsonl"

    def __init__(self, change: Any = None, read: Any = None) -> None:
        self._real = ClaudeCodeJsonlSource()
        self._change = change
        self._read = read

    def read(self, ref: str | Path) -> Any:
        if self._read is not None:
            return self._read(ref)
        trace = self._real.read(ref)
        return trace if self._change is None else self._change(trace)


@pytest.mark.req("TER-CNF-003")
def test_an_object_without_the_port_shape_fails_the_protocol() -> None:
    class NoRead:
        format_name = "x"

    assert _failed(_verdicts(NoRead()))["protocol"] == (
        "has no format_name attribute and read(ref) method"
    )
    nameless = _Wrapping()
    nameless.format_name = ""
    assert _failed(_verdicts(nameless))["protocol"] == "format_name is empty"


@pytest.mark.req("TER-CNF-003")
def test_a_source_whose_reads_differ_fails_determinism() -> None:
    count = iter(range(1000))

    def change(trace: Any) -> Any:
        return replace(trace, session_id=f"s{next(count)}")

    failed = _failed(_verdicts(_Wrapping(change)))
    assert failed["deterministic"] == f"{SAMPLES[0]}: a second read differs"


@pytest.mark.req("TER-CNF-003")
def test_repeated_ids_or_broken_sequences_fail_the_ids_obligation() -> None:
    def repeat(trace: Any) -> Any:
        events = list(trace.events)
        events[1] = replace(events[1], id=events[0].id)
        return replace(trace, events=tuple(events))

    def renumber(trace: Any) -> Any:
        events = [replace(e, sequence=e.sequence + 1) for e in trace.events]
        return replace(trace, events=tuple(events))

    def rename(trace: Any) -> Any:
        return replace(trace, source_format="other")

    ref = SAMPLES[0]
    assert _failed(_verdicts(_Wrapping(repeat)))["ids"] == f"{ref}: event ids repeat"
    assert (
        _failed(_verdicts(_Wrapping(renumber)))["ids"]
        == f"{ref}: sequences do not run 0..n-1"
    )
    assert _failed(_verdicts(_Wrapping(rename)))["ids"] == (
        f"{ref}: source_format 'other' is not the adapter's format_name 'claude-code-jsonl'"
    )


@pytest.mark.req("TER-CNF-003")
def test_events_outside_their_session_or_order_fail_the_chain() -> None:
    def stray(trace: Any) -> Any:
        events = list(trace.events)
        events[-1] = replace(events[-1], session_id="elsewhere")
        return replace(trace, events=tuple(events))

    def orphan(trace: Any) -> Any:
        events = list(trace.events)
        events[2] = replace(events[2], parent_id=None)
        return replace(trace, events=tuple(events))

    def rooted(trace: Any) -> Any:
        events = list(trace.events)
        events[0] = replace(events[0], parent_id="x")
        return replace(trace, events=tuple(events))

    def empty(trace: Any) -> Any:
        return replace(trace, events=())

    ref = SAMPLES[0]
    assert "events name sessions" in _failed(_verdicts(_Wrapping(stray)))["chain"]
    assert (
        _failed(_verdicts(_Wrapping(orphan)))["chain"]
        == f"{ref}: event 2 does not follow event 1"
    )
    assert (
        _failed(_verdicts(_Wrapping(rooted)))["chain"]
        == f"{ref}: the first event has a parent"
    )
    assert _failed(_verdicts(_Wrapping(empty)))["chain"] == f"{ref}: no events"


@pytest.mark.req("TER-CNF-003")
def test_user_input_counted_as_generated_fails() -> None:
    from ter.domain import Actor

    def no_assistant(trace: Any) -> Any:
        events = [
            replace(e, actor=Actor.USER) if e.actor is Actor.ASSISTANT else e
            for e in trace.events
        ]
        return replace(trace, events=tuple(events))

    failed = _failed(_verdicts(_Wrapping(no_assistant)))
    assert (
        failed["generated"] == f"{SAMPLES[0]}: a generated event is not the assistant's"
    )


@pytest.mark.req("TER-CNF-003")
def test_a_source_that_reads_anything_fails_the_refusals() -> None:
    real = ClaudeCodeJsonlSource()

    def lenient(ref: str | Path) -> Any:
        if not Path(ref).exists():
            raise KeyError(ref)
        if Path(ref).name.startswith(("empty", "text", "not-utf8")):
            return real.read(SAMPLES[0])
        return real.read(ref)

    failed = _failed(_verdicts(_Wrapping(read=lenient)))
    assert failed["missing"].startswith("raised KeyError")
    assert failed["refuses"] == "a empty .jsonl file was read as a session"


@pytest.mark.req("TER-CNF-003")
def test_a_reader_that_crashes_on_bad_bytes_is_named() -> None:
    real = ClaudeCodeJsonlSource()

    def crashy(ref: str | Path) -> Any:
        if Path(ref).name.startswith("not-utf8"):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        return real.read(ref)

    failed = _failed(_verdicts(_Wrapping(read=crashy)))
    assert failed == {
        "refuses": "a not-utf8 .jsonl file raised UnicodeDecodeError, not SessionFormatError"
    }


def test_refusals_use_the_samples_suffixes(tmp_path: Path) -> None:
    seen: list[str] = []

    def record(ref: str | Path) -> Any:
        seen.append(Path(ref).name)
        raise SessionFormatError("no")

    subject = SessionSubject(
        _Wrapping(read=record), ("a.log", "b.log", "c.json", "folder")
    )
    refuses = next(o for o in SESSION_SOURCE if o.id == "refuses")
    refuses(subject)
    assert sorted(seen) == sorted(
        f"{n}{s}" for s in (".json", ".log") for n in ("empty", "text", "not-utf8")
    )


def test_no_samples_fails_every_reading_obligation() -> None:
    failed = _failed(_verdicts(ClaudeCodeJsonlSource(), samples=()))
    assert set(failed) == {"deterministic", "ids", "chain", "generated"}
    assert set(failed.values()) == {"no sample sessions given"}


# -- PriceBook -----------------------------------------------------------------


def _book(entries: list[PriceEntry]) -> Any:
    return InMemoryPriceBook(entries)


@pytest.mark.req("TER-CNF-002")
def test_a_price_book_meets_every_obligation() -> None:
    verdicts = check(
        PRICE_BOOK, PriceSubject(_book([PriceEntry("m", PRICED, RATES)]), TODAY)
    )
    assert [v.obligation for v in verdicts] == [o.id for o in PRICE_BOOK]
    assert all(v.met for v in verdicts)


@pytest.mark.req("TER-CNF-002")
def test_a_book_priced_long_ago_still_meets_every_obligation() -> None:
    old = InMemoryPriceBook([PriceEntry("m", date(1990, 1, 1), RATES)])
    assert all(v.met for v in check(PRICE_BOOK, PriceSubject(old, TODAY)))


@dataclass
class _Book:
    """A price book whose answers a test sets."""

    name: str = "fake"
    listed: tuple[str, ...] = ("m",)
    latest: Any = RATES
    dated: Any = RATES
    unknown: Any = None

    def models(self) -> tuple[str, ...]:
        return self.listed

    def rate(self, model: str, at: date | None = None) -> Any:
        if model not in self.listed:
            if self.unknown is not None:
                return self.unknown
            raise UnknownModelError(model)
        if at is None:
            return self.latest() if callable(self.latest) else self.latest
        if at < PRICED:
            if self.dated is None:
                return RATES
            raise UnknownModelError(model)
        return self.dated


def _book_failures(book: Any) -> dict[str, str]:
    return {
        v.obligation: v.detail
        for v in check(PRICE_BOOK, PriceSubject(book, TODAY))
        if not v.met
    }


@pytest.mark.req("TER-CNF-003")
def test_a_price_book_that_falls_short_is_named() -> None:
    assert _book_failures(_Book()) == {}
    assert _book_failures(_Book(name=""))["protocol"] == "name is empty"
    assert _book_failures(_Book(listed=()))["protocol"] == "models() lists no model"
    assert (
        _book_failures(_Book(latest=1.0, dated=1.0))["valid"]
        == "m: rate() returned float"
    )
    # Any finite, non-negative prices are valid, in any order, and a later
    # price than today's leaves the undated lookup free to differ.
    cheap_output = Rates(input=3.0, output=1.0, cache_read=0.3, cache_write=3.75)
    assert _book_failures(_Book(latest=cheap_output, dated=cheap_output)) == {}
    later = Rates(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)
    assert _book_failures(_Book(dated=later)) == {}
    assert _book_failures(_Book(dated=2.0))["dated"] == (
        f"m: rate() on {TODAY} returned float"
    )
    assert (
        _book_failures(_Book(unknown=RATES))["unknown"]
        == "an unknown model did not raise UnknownModelError"
    )
    assert (
        _book_failures(_Book(dated=None))["before"]
        == "m: a date before any price did not raise UnknownModelError"
    )
    counter = iter(range(100))
    drifting = _Book(
        latest=lambda: Rates(
            input=1.0, output=float(10 + next(counter)), cache_read=0, cache_write=0
        )
    )
    assert _book_failures(drifting)["deterministic"] == "m: a second lookup differs"


# -- the command ---------------------------------------------------------------


def _run(
    argv: list[str], adapter: Any = None, error: str | None = None
) -> tuple[int, str, str]:
    asked: list[tuple[str, str, tuple[str, ...]]] = []

    def build(port: str, spec: str, args: Any) -> object:
        asked.append((port, spec, tuple(args)))
        if error is not None:
            raise CapabilityError(error)
        return adapter

    services = CliServices(
        analyse_transcript=lambda p, t: None,  # type: ignore[arg-type,return-value]
        log_sessions=lambda d: (),
        analyse_log=lambda d, s, t: None,  # type: ignore[arg-type,return-value]
        hook_ingest=lambda d: None,  # type: ignore[arg-type,return-value]
        default_log_dir=Path("."),
        conformance_adapter=build,
    )
    out, err = io.StringIO(), io.StringIO()
    code = cli_main(argv, services, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.req("TER-CNF-001")
def test_the_command_passes_a_conforming_session_source() -> None:
    argv = ["conformance", "SessionSource", "--adapter", "claude-code"]
    for ref in SAMPLES:
        argv += ["--sample", ref]
    code, out, err = _run(argv, ClaudeCodeJsonlSource())
    assert code == 0, err
    assert out.startswith("SessionSource claude-code conformance\n")
    assert out.endswith("7 of 7 obligations met\n")


@pytest.mark.req("TER-CNF-003")
def test_the_command_exits_one_when_an_obligation_is_not_met() -> None:
    nameless = _Wrapping()
    nameless.format_name = ""
    code, out, _ = _run(
        ["conformance", "SessionSource", "--adapter", "x:Y", "--sample", SAMPLES[0]],
        nameless,
    )
    assert code == 1
    assert "  FAIL  protocol" in out and "format_name is empty" in out


@pytest.mark.req("TER-CNF-002")
def test_the_command_checks_a_price_book_on_the_day_given() -> None:
    book = _book([PriceEntry("m", PRICED, RATES)])
    code, out, _ = _run(
        [
            "conformance",
            "PriceBook",
            "--adapter",
            "a:B",
            "--arg",
            "f.csv",
            "--today",
            "2026-06-01",
        ],
        book,
    )
    assert code == 0
    assert out.endswith("6 of 6 obligations met\n")
    code, out, _ = _run(
        ["conformance", "PriceBook", "--adapter", "a:B", "--today", "2026-06-01"],
        _Book(dated=2.0),
    )
    assert code == 1
    assert "m: rate() on 2026-06-01 returned float" in out


@pytest.mark.req("TER-CNF-004")
def test_an_adapter_that_cannot_be_built_is_reported_without_checking() -> None:
    code, out, err = _run(
        ["conformance", "PriceBook", "--adapter", "nope:X"], error="cannot load nope:X"
    )
    assert (code, out) == (2, "")
    assert err == "Cannot build the adapter: cannot load nope:X\n"


@pytest.mark.req("TER-CNF-004")
def test_a_session_source_without_samples_is_refused_before_loading(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit:
        _run(["conformance", "SessionSource", "--adapter", "x:Y"], object())
    assert exit.value.code == 2
    assert "--sample" in capsys.readouterr().err


@pytest.mark.req("TER-CNF-004")
def test_the_composition_root_loads_by_entry_point_or_capability_name(
    tmp_path: Path,
) -> None:
    from ter.bootstrap import cli_services

    build = cli_services().conformance_adapter
    assert build is not None
    by_value = build(
        "SessionSource", "ter.adapters.driven.otel_genai:OtelGenAiSource", []
    )
    by_name = build("SessionSource", "otel-genai", [])
    by_key = build("SessionSource", "SessionSource.otel-genai", [])
    assert {type(a).__name__ for a in (by_value, by_name, by_key)} == {
        "OtelGenAiSource"
    }
    prices = (
        Path(__file__).resolve().parents[1] / "fixtures" / "focus" / "sku_price.csv"
    )
    book = build("PriceBook", "focus-sku-price", [str(prices)])
    assert getattr(book, "name") == "focus-sku-price-1.5"
    for spec, args, reason in [
        ("no_such_module:X", [], "cannot load no_such_module:X: ModuleNotFoundError"),
        ("ter.adapters.driven.otel_genai:Nope", [], "cannot load"),
        (
            "focus-sku-price",
            [str(tmp_path / "missing.csv")],
            "cannot build focus-sku-price(",
        ),
        ("no-such-capability", [], "no-such-capability"),
    ]:
        with pytest.raises(CapabilityError, match=reason.replace("(", r"\(")):
            build("PriceBook", spec, args)
