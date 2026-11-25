"""Unit tests for TER 4 pricing: domain arithmetic, schedules and the JSON book."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ter.adapters.driven.pricing import (
    JsonPriceBook,
    default_price_book,
    parse_price_book,
)
from ter.domain import (
    PriceEntry,
    PriceSchedule,
    Rates,
    TokenCounts,
    UnknownModelError,
    token_cost,
    usage_cost,
)
from ter_calculator.config_parse import parse_cost_model
from ter_calculator.cost_model import PRICING
from ter_calculator.models import CostModel

SONNET = Rates(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)


def _entry(model: str, day: str, rate: float, *aliases: str) -> PriceEntry:
    return PriceEntry(
        model=model,
        effective_from=date.fromisoformat(day),
        rates=Rates(rate, rate * 5, rate / 10, rate * 1.25),
        source="test",
        aliases=aliases,
    )


class TestArithmetic:
    def test_token_cost_is_per_million(self) -> None:
        assert token_cost(1_000_000, 3.0) == 3.0
        assert token_cost(0, 15.0) == 0.0

    def test_token_cost_keeps_ter3_operation_order(self) -> None:
        tokens, rate = 12_345, 0.3
        assert token_cost(tokens, rate) == tokens * rate / 1_000_000

    def test_usage_cost_sums_every_category(self) -> None:
        counts = TokenCounts(
            input=1_000, output=2_000, cache_read=10_000, cache_write=500
        )
        expected = (
            1_000 * 3.0 / 1e6
            + 2_000 * 15.0 / 1e6
            + 10_000 * 0.3 / 1e6
            + 500 * 3.75 / 1e6
        )
        assert usage_cost(counts, SONNET) == pytest.approx(expected)
        assert usage_cost(TokenCounts(), SONNET) == 0.0

    def test_thinking_is_billed_as_output(self) -> None:
        assert SONNET.thinking == SONNET.output

    @pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
    def test_rates_must_be_finite_and_non_negative(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            Rates(input=bad, output=1.0, cache_read=0.1, cache_write=1.0)


class TestSchedule:
    def test_dated_entries_resolve_by_effective_date(self) -> None:
        schedule = PriceSchedule(
            [_entry("m", "2025-06-01", 2.0), _entry("m", "2025-01-01", 1.0, "alias")]
        )
        assert schedule.rate("m").input == 2.0
        assert schedule.rate("m", date(2025, 3, 1)).input == 1.0
        assert schedule.rate("m", date(2025, 6, 1)).input == 2.0
        assert schedule.rate("alias", date(2025, 1, 1)).input == 1.0
        assert schedule.resolve("alias", date(2025, 3, 1)) == "m"
        assert schedule.models() == ("m",)

    def test_before_first_price_is_unknown(self) -> None:
        schedule = PriceSchedule([_entry("m", "2025-06-01", 2.0)])
        with pytest.raises(UnknownModelError, match="no price in effect"):
            schedule.rate("m", date(2025, 5, 31))

    def test_unknown_model(self) -> None:
        with pytest.raises(UnknownModelError):
            PriceSchedule([]).rate("nope")

    def test_duplicate_dates_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            PriceSchedule(
                [_entry("m", "2025-01-01", 1.0), _entry("m", "2025-01-01", 2.0)]
            )

    def test_alias_moves_to_a_successor_on_its_effective_date(self) -> None:
        schedule = PriceSchedule(
            [
                _entry("sonnet-4", "2025-01-01", 3.0, "sonnet"),
                _entry("sonnet-5", "2026-01-01", 4.0, "sonnet"),
            ]
        )
        assert schedule.resolve("sonnet", date(2025, 12, 31)) == "sonnet-4"
        assert schedule.resolve("sonnet", date(2026, 1, 1)) == "sonnet-5"
        assert schedule.resolve("sonnet") == "sonnet-5"
        assert schedule.rate("sonnet", date(2025, 6, 1)).input == 3.0
        assert schedule.rate("sonnet").input == 4.0
        # The old model keeps its own price under its canonical name.
        assert schedule.rate("sonnet-4").input == 3.0

    def test_alias_added_later_does_not_resolve_earlier(self) -> None:
        schedule = PriceSchedule(
            [_entry("m", "2025-01-01", 1.0), _entry("m", "2025-06-01", 2.0, "new")]
        )
        assert schedule.rate("new", date(2025, 6, 1)).input == 2.0
        with pytest.raises(UnknownModelError, match="no priced model"):
            schedule.rate("new", date(2025, 3, 1))

    def test_alias_dropped_later_does_not_resolve_after(self) -> None:
        schedule = PriceSchedule(
            [_entry("m", "2025-01-01", 1.0, "old"), _entry("m", "2025-06-01", 2.0)]
        )
        assert schedule.rate("old", date(2025, 3, 1)).input == 1.0
        with pytest.raises(UnknownModelError, match="no priced model"):
            schedule.rate("old", date(2025, 6, 1))
        with pytest.raises(UnknownModelError):
            schedule.rate("old")

    @pytest.mark.parametrize(
        "entries",
        [
            [_entry("a", "2025-01-01", 1.0, "x"), _entry("b", "2025-01-01", 1.0, "x")],
            [_entry("a", "2025-01-01", 1.0, "b"), _entry("b", "2025-01-01", 1.0)],
        ],
    )
    def test_ambiguous_aliases_are_rejected(self, entries: list[PriceEntry]) -> None:
        with pytest.raises(ValueError, match="Alias"):
            PriceSchedule(entries)


def _book(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": "ter.price_book/1",
        "currency": "USD",
        "unit": "per_million_tokens",
        "prices": [
            {
                "model": "m",
                "aliases": ["short"],
                "effective_from": "2025-01-01",
                "source": "unit test",
                "rates": {
                    "input": 1,
                    "output": 5,
                    "cache_read": 0.1,
                    "cache_write": 1.25,
                },
            }
        ],
    }
    document.update(overrides)
    return document


def _price(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = dict(_book()["prices"][0])
    entry.update(overrides)
    return _book(prices=[entry])


class TestJsonPriceBook:
    def test_reads_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "prices.json"
        path.write_text(json.dumps(_book()), encoding="utf-8")
        book = JsonPriceBook(path)
        assert book.name == str(path)
        assert book.rate("short") == Rates(1.0, 5.0, 0.1, 1.25)
        assert book.entry("m").source == "unit test"

    @pytest.mark.parametrize(
        "document, message",
        [
            ([], "JSON object"),
            (_book(schema="other/1"), "schema"),
            (_book(currency="EUR"), "USD"),
            (_book(unit="per_token"), "USD per million"),
            (_book(prices=[]), "non-empty"),
            (_book(prices=["x"]), "must be an object"),
            (_price(effective_from="soon"), "malformed"),
            (_price(rates={"input": 1}), "malformed"),
            (_price(rates={**_book()["prices"][0]["rates"], "input": True}), "number"),
            (
                _price(rates={**_book()["prices"][0]["rates"], "output": False}),
                "number",
            ),
            (_price(rates={**_book()["prices"][0]["rates"], "input": "3"}), "number"),
            (_price(model=""), "model"),
            (_price(source=""), "source"),
            (_price(aliases="sonnet"), "aliases"),
        ],
    )
    def test_rejects_malformed_books(self, document: Any, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_price_book(document)

    def test_shipped_book_is_cached_and_sourced(self) -> None:
        book = default_price_book()
        assert book is default_price_book()
        assert book.name == "ter:data/price_book.json"
        for model in book.models():
            assert book.entry(model).source


@pytest.mark.req("TER-ANL-000")
class TestTer3ReadsThePriceBook:
    """TER 3's hard-coded rates now come from the shipped book.

    Sonnet is unchanged; haiku and opus were corrected to the published list
    prices on 25 Nov 2026 (TER 3 carried the earlier generation's rates).
    """

    def test_default_cost_model_is_sonnet(self) -> None:
        assert CostModel() == CostModel(3.00, 15.00, 0.30, 3.75)

    def test_parse_cost_model_sonnet_matches(self) -> None:
        assert parse_cost_model("sonnet") == CostModel(3.00, 15.00, 0.30, 3.75)
        assert parse_cost_model("SONNET") == CostModel()

    @pytest.mark.parametrize(
        "tier, name, rates",
        [
            ("haiku", "claude-haiku-4-5", (1.00, 5.00, 0.10, 1.25)),
            ("sonnet", "claude-sonnet-4-6", (3.00, 15.00, 0.30, 3.75)),
            ("opus", "claude-opus-4-6", (5.00, 25.00, 0.50, 6.25)),
        ],
    )
    def test_pricing_tiers_match_ter3(
        self, tier: str, name: str, rates: tuple[float, float, float, float]
    ) -> None:
        t = PRICING[tier]
        assert t.name == name
        assert (
            t.input_per_mtok,
            t.output_per_mtok,
            t.cached_read_per_mtok,
            t.cached_write_per_mtok,
        ) == rates
        assert default_price_book().rate(tier) == Rates(*rates)


@pytest.mark.req("TER-ANL-047")
def test_an_entry_that_ends_leaves_its_model_unpriced_from_its_end() -> None:
    rates = Rates(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)
    schedule = PriceSchedule(
        [
            PriceEntry(
                "m",
                date(2026, 1, 1),
                rates,
                aliases=("fam",),
                effective_to=date(2026, 3, 1),
            )
        ]
    )
    assert schedule.rate("m", date(2026, 2, 28)) == rates
    assert schedule.rate("fam", date(2026, 2, 28)) == rates
    for day in (date(2026, 3, 1), date(2026, 3, 2)):
        with pytest.raises(UnknownModelError, match="its price ended on 2026-03-01"):
            schedule.rate("m", day)
    # With no date, the latest rates are still the latest published.
    assert schedule.rate("m") == rates
    # An entry with no end stays in effect.
    open_ended = PriceSchedule([PriceEntry("m", date(2026, 1, 1), rates)])
    assert open_ended.rate("m", date(2099, 1, 1)) == rates
