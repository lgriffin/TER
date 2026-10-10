"""Context inventory in tokens, and a session priced at its date's prices.

TER-DET-004: unused retrieved context and context read more than once are
reported in tokens. TER-ANL-040: a session is priced with the price book entry
whose date is the latest not after the session date. TER-ANL-041: a session
whose usage has no cache fields has its cost figures marked estimated.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.event_log.codec import event_from_record, event_to_record
from ter.adapters.driven.gare import GareRunSource
from ter.adapters.driven.in_memory import InMemoryPriceBook
from ter.adapters.driven.pricing import default_price_book
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import Event, PriceEntry, Rates, TokenUsage, analyse_batch
from ter.domain.costing import EstimateReason, price_session, session_date
from ter.domain.lean import build_a3
from ter.domain.lean.analysis import LeanAnalysis, explain
from ter.domain.lean.inventory import context_inventory

from tests.golden.corpus import CORPUS

from ter4_lean_builder import Script

REPO = Path(__file__).resolve().parents[2]
GARE_FIXTURES = REPO / "tests" / "fixtures" / "gare"

OLD = Rates(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
NEW = Rates(input=4.0, output=20.0, cache_read=0.4, cache_write=5.0)
SWITCH = date(2026, 3, 1)


def _book() -> InMemoryPriceBook:
    return InMemoryPriceBook(
        [
            PriceEntry("model-x", date(2026, 1, 1), OLD, aliases=("x",)),
            PriceEntry("model-x", SWITCH, NEW, aliases=("x",)),
        ]
    )


def _usage(
    *,
    input: int = 1_000_000,
    output: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    model: str | None = "model-x",
    cache_reported: bool = True,
) -> TokenUsage:
    return TokenUsage(
        input_tokens=input,
        output_tokens=output,
        cache_creation_tokens=cache_write,
        cache_read_tokens=cache_read,
        model=model,
        cache_reported=cache_reported,
    )


def _with_usage(events: list[Event], usage: dict[int, TokenUsage]) -> list[Event]:
    return [replace(e, usage=usage.get(i, e.usage)) for i, e in enumerate(events)]


def _on(events: list[Event], day: date) -> list[Event]:
    start = datetime(day.year, day.month, day.day, 9, 0, tzinfo=UTC)
    return [
        replace(e, timestamp=start + timedelta(seconds=i)) for i, e in enumerate(events)
    ]


def _analysis(events: list[Event]) -> LeanAnalysis:
    return explain(events, RegexTokenizer())


def _turn_session() -> list[Event]:
    """A session with one model turn: the response carries 1M input tokens."""
    s = Script()
    s.prompt("explain the module")
    s.say("It parses arguments.")
    return _with_usage(s.events, {1: _usage()})


# --- TER-DET-004: unused and re-read context, in tokens ---------------------


def _inventory_script() -> Script:
    s = Script()
    s.prompt("add a flag to the cli")
    s.think("look at the cli")  # turn 1
    s.read("src/cli.py", "def main(): pass")
    s.read("src/utils.py", "def slugify(value): return value.lower()")
    s.think("now edit")  # turn 2
    s.read("src/cli.py", "def main(): pass")  # unchanged re-read
    s.edit("src/cli.py")
    s.read("src/cli.py", "def main(flag): pass")  # re-read after an edit
    s.say("Added the flag in cli.py.")  # turn 3
    return s


@pytest.mark.req("TER-DET-004")
class TestContextInventory:
    def _inventory(self, s: Script) -> tuple[LeanAnalysis, list[Event]]:
        events = _with_usage(
            s.events, {1: _usage(), 6: _usage(), len(s.events) - 1: _usage()}
        )
        return _analysis(events), events

    def test_unused_read_reports_its_tokens(self) -> None:
        analysis, _ = self._inventory(_inventory_script())
        inv = context_inventory(analysis.steps, analysis.findings)
        [item] = inv.unused
        result = next(s for s in analysis.steps if s.event_id == item.event_id)
        assert item.path == "src/utils.py"
        assert item.tokens == result.context_tokens > 0
        assert inv.unused_tokens == item.tokens
        # Cited by the unused_context finding, promoted by the judged sample.
        assert item.finding is not None and item.finding.startswith("unused_context:")
        assert not item.uncertain
        assert item.request_id is not None

    def test_reads_after_the_first_are_reread_tokens(self) -> None:
        analysis, events = self._inventory(_inventory_script())
        inv = context_inventory(analysis.steps, analysis.findings)
        assert [i.path for i in inv.reread] == ["src/cli.py", "src/cli.py"]
        assert [i.changed for i in inv.reread] == [False, True]
        assert inv.reread_tokens == sum(i.tokens for i in inv.reread)
        assert inv.unchanged_reread_tokens == inv.reread[0].tokens
        # The same re-reads the L1 report counts.
        stream = analyse_batch(events, RegexTokenizer())
        assert len(inv.reread) == stream.repeated_read_count

    def test_carried_turns_count_the_model_turns_after_the_read(self) -> None:
        analysis, _ = self._inventory(_inventory_script())
        inv = context_inventory(analysis.steps, analysis.findings)
        assert inv.turns == 3
        assert inv.unused[0].carried_turns == 2  # turns 2 and 3
        assert [i.carried_turns for i in inv.reread] == [1, 1]
        assert inv.unused_carried_tokens == inv.unused[0].tokens * 2

    def test_negative_a_used_single_read_is_no_inventory(self) -> None:
        s = Script()
        s.read("src/utils.py", "def slugify(value): return value")
        s.think("I can reuse slugify here")
        s.say("done")
        analysis = _analysis(s.events)
        inv = context_inventory(analysis.steps, analysis.findings)
        assert inv.unused == () and inv.reread == ()
        assert inv.unused_tokens == inv.reread_tokens == 0
        assert inv.retrieved_tokens > 0

    def test_boundary_second_read_of_another_path_is_not_a_reread(self) -> None:
        s = Script()
        s.read("src/a.py", "def alpha(): pass")
        s.read("src/b.py", "def beta(): pass")
        s.say("alpha and beta are fine")
        analysis = _analysis(s.events)
        assert context_inventory(analysis.steps, analysis.findings).reread == ()

    def test_the_a3_reports_the_inventory(self) -> None:
        analysis, _ = self._inventory(_inventory_script())
        payload = build_a3(analysis).as_dict()["context_inventory"]
        assert isinstance(payload, dict)
        inv = context_inventory(analysis.steps, analysis.findings)
        assert payload["unused_tokens"] == inv.unused_tokens
        assert payload["reread_tokens"] == inv.reread_tokens
        assert payload["unused"][0]["event_id"] == inv.unused[0].event_id  # type: ignore[index]

    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_corpus_rereads_match_the_l1_count(self, name: str) -> None:
        trace = ClaudeCodeJsonlSource().read(CORPUS[name])
        analysis = _analysis(list(trace.events))
        inv = context_inventory(analysis.steps, analysis.findings)
        stream = analyse_batch(trace.events, RegexTokenizer())
        assert len(inv.reread) == stream.repeated_read_count
        assert inv.unused_tokens + inv.reread_tokens <= inv.retrieved_tokens * 2


@pytest.mark.req("TER-DET-004", "TER-ANL-040")
class TestInventoryCost:
    def test_carrying_cost_uses_cache_write_then_cache_read_rates(self) -> None:
        s = Script()
        s.prompt("look")
        s.read("notes.txt", "alpha beta gamma delta")
        s.think("hm")  # turn 1 ingests the read
        s.think("hm hm")  # turn 2 carries it
        s.say("done")  # turn 3 carries it
        caching = _usage(input=10, cache_read=5, cache_write=5)
        events = _on(
            _with_usage(s.events, {3: caching, 4: caching, 5: caching}), SWITCH
        )
        analysis = _analysis(events)
        inv = context_inventory(analysis.steps, analysis.findings)
        [item] = inv.unused
        cost = price_session(analysis.steps, inv, _book())
        expected = item.tokens * (NEW.cache_write + 2 * NEW.cache_read) / 1_000_000
        assert cost.unused_context_usd == pytest.approx(expected)
        assert cost.reread_context_usd == 0

    def test_without_caching_context_is_carried_at_the_input_rate(self) -> None:
        s = Script()
        s.prompt("look")
        s.read("notes.txt", "alpha beta gamma delta")
        s.think("hm")
        s.say("done")
        plain = _usage(input=10)
        events = _on(_with_usage(s.events, {3: plain, 4: plain}), SWITCH)
        analysis = _analysis(events)
        inv = context_inventory(analysis.steps, analysis.findings)
        cost = price_session(analysis.steps, inv, _book())
        assert cost.unused_context_usd == pytest.approx(
            inv.unused[0].tokens * 2 * NEW.input / 1_000_000
        )


# --- TER-ANL-040: the price book in force on the session date ---------------


@pytest.mark.req("TER-ANL-040")
class TestDatedPricing:
    @pytest.mark.parametrize(
        ("day", "rates"),
        [
            (SWITCH, NEW),  # exactly the effective date: the new entry
            (SWITCH - timedelta(days=1), OLD),  # the day before: the old one
            (date(2026, 2, 10), OLD),  # between the entries
            (date(2026, 9, 1), NEW),  # after the latest entry
        ],
        ids=["exact-date", "day-before", "between", "after-latest"],
    )
    def test_session_is_priced_at_its_dates_prices(
        self, day: date, rates: Rates
    ) -> None:
        events = _on(_turn_session(), day)
        analysis = _analysis(events)
        inv = context_inventory(analysis.steps, analysis.findings)
        cost = price_session(analysis.steps, inv, _book())
        assert cost.priced_on == day
        assert cost.usd == pytest.approx(rates.input)  # 1M input tokens
        assert cost.unpriced_turns == 0
        assert not cost.estimated

    def test_before_the_earliest_entry_the_session_is_unpriced(self) -> None:
        events = _on(_turn_session(), date(2025, 12, 31))
        analysis = _analysis(events)
        inv = context_inventory(analysis.steps, analysis.findings)
        cost = price_session(analysis.steps, inv, _book())
        assert cost.usd == 0 and cost.unpriced_turns == 1
        assert cost.unpriced_models == ("model-x",)
        assert EstimateReason.UNPRICED_TURNS in cost.estimate_reasons

    def test_an_alias_is_priced_as_its_model(self) -> None:
        events = _on(_with_usage(_turn_session(), {1: _usage(model="x")}), SWITCH)
        analysis = _analysis(events)
        cost = price_session(
            analysis.steps, context_inventory(analysis.steps, ()), _book()
        )
        assert cost.usd == pytest.approx(NEW.input)

    def test_session_date_is_the_first_timestamp_in_utc(self) -> None:
        late = datetime(2026, 2, 28, 23, 30, tzinfo=UTC).astimezone()
        assert session_date([None, late, datetime(2026, 5, 1)]) == date(2026, 2, 28)
        assert session_date([None]) is None

    def test_an_undated_session_uses_the_latest_prices_and_says_so(self) -> None:
        events = [replace(e, timestamp=None) for e in _turn_session()]
        analysis = _analysis(events)
        cost = price_session(
            analysis.steps, context_inventory(analysis.steps, ()), _book()
        )
        assert cost.priced_on is None
        assert cost.usd == pytest.approx(NEW.input)
        assert EstimateReason.NO_SESSION_DATE in cost.estimate_reasons

    def test_the_a3_prices_through_the_shipped_book(self) -> None:
        book = default_price_book()
        model = book.models()[0]
        events = _on(_with_usage(_turn_session(), {1: _usage(model=model)}), SWITCH)
        a3 = build_a3(_analysis(events), prices=book)
        assert a3.cost is not None
        assert a3.cost.usd == pytest.approx(book.rate(model, SWITCH).input)
        payload = a3.as_dict()["cost"]
        assert isinstance(payload, dict)
        assert payload["priced_on"] == SWITCH.isoformat()
        assert payload["price_book"] == book.name


# --- TER-ANL-041: no cache fields, estimated costs --------------------------


@pytest.mark.req("TER-ANL-041")
class TestEstimatedCosts:
    def _cost(self, usage: TokenUsage | None) -> tuple[bool, tuple[str, ...]]:
        events = _on(_with_usage(_turn_session(), {1: usage} if usage else {}), SWITCH)
        if usage is None:
            events = [replace(e, usage=None) for e in events]
        analysis = _analysis(events)
        a3 = build_a3(analysis, prices=_book())
        assert a3.cost is not None
        payload = a3.as_dict()["cost"]
        assert isinstance(payload, dict)
        assert payload["estimated"] is a3.cost.estimated
        return a3.cost.estimated, a3.cost.estimate_reasons

    def test_usage_without_cache_fields_is_estimated(self) -> None:
        estimated, reasons = self._cost(_usage(cache_reported=False))
        assert estimated and reasons == (EstimateReason.NO_CACHE_FIELDS,)

    def test_usage_with_cache_fields_is_not_estimated(self) -> None:
        assert self._cost(_usage(cache_reported=True)) == (False, ())

    def test_boundary_reported_zero_cache_fields_are_not_an_estimate(self) -> None:
        # Fields present but zero: nothing was cached, which is a fact.
        assert self._cost(_usage(cache_read=0, cache_write=0)) == (False, ())

    def test_no_usage_at_all_is_estimated(self) -> None:
        estimated, reasons = self._cost(None)
        assert estimated
        assert EstimateReason.NO_CACHE_FIELDS in reasons
        assert EstimateReason.NO_USAGE in reasons

    def test_claude_code_records_without_cache_fields(self, tmp_path: Path) -> None:
        records = [
            {
                "type": "user",
                "uuid": "u1",
                "sessionId": "s1",
                "timestamp": "2026-03-01T09:00:00Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                },
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "sessionId": "s1",
                "requestId": "r1",
                "timestamp": "2026-03-01T09:00:01Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "hello"}],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            },
        ]
        path = tmp_path / "s1.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        [usage] = [
            e.usage for e in ClaudeCodeJsonlSource().read(path).events if e.usage
        ]
        assert usage.model == "claude-sonnet-4-6"
        assert usage.cache_reported is False

    def test_claude_code_records_with_cache_fields(self) -> None:
        trace = ClaudeCodeJsonlSource().read(CORPUS["example_session"])
        usages = [e.usage for e in trace.events if e.usage is not None]
        assert usages and all(u.cache_reported for u in usages)
        assert {u.model for u in usages} == {"claude-sonnet-demo"}

    def test_gare_usage_reports_no_cache_fields(self) -> None:
        run = next(p for p in sorted(GARE_FIXTURES.iterdir()) if p.is_dir())
        trace = GareRunSource().read(run)
        usages = [e.usage for e in trace.events if e.usage is not None]
        assert usages and not any(u.cache_reported for u in usages)

    def test_records_before_0_4_decode_cache_presence_from_the_figures(self) -> None:
        [event] = _with_usage(_turn_session()[1:2], {0: _usage(cache_read=3)})
        record = event_to_record(event)
        assert record["usage"]["model"] == "model-x"
        assert event_from_record(record) == event
        old = {**record, "schema": "ter.event/0.3"}
        old["usage"] = {
            k: v
            for k, v in record["usage"].items()
            if k not in ("model", "cache_reported")
        }
        assert event_from_record(old).usage == replace(event.usage, model=None)  # type: ignore[arg-type]
        old["usage"]["cache_read_tokens"] = 0
        decoded = event_from_record(old).usage
        assert decoded is not None and decoded.cache_reported is False
