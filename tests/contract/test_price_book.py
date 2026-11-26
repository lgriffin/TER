"""Contract suite for the ``PriceBook`` port.

The shipped JSON price book and the in-memory fake run the same assertions,
so the fake cannot drift from the obligations the real adapter meets.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pytest

from ter.adapters.driving.conformance import PRICE_BOOK, Obligation, PriceSubject
from ter.adapters.driven.in_memory import InMemoryPriceBook
from ter.adapters.driven.pricing import JsonPriceBook
from ter.adapters.driven.pricing.focus import FocusSkuPriceBook
from ter.domain import PriceEntry, Rates, UnknownModelError
from ter.ports import PriceBook

Factory = Callable[[], PriceBook]


def _json() -> PriceBook:
    return JsonPriceBook()


def _memory() -> PriceBook:
    real = JsonPriceBook()
    return InMemoryPriceBook(
        PriceEntry(
            model=model,
            effective_from=real.entry(model).effective_from,
            rates=real.rate(model),
            source="copied from the shipped book",
        )
        for model in real.models()
    )


FOCUS = Path(__file__).resolve().parents[1] / "fixtures" / "focus"


def _focus() -> PriceBook:
    return FocusSkuPriceBook(FOCUS / "sku_price.csv")


def _focus_details() -> PriceBook:
    return FocusSkuPriceBook(
        FOCUS / "sku_price.no-details.csv", details=FOCUS / "cost_and_usage.csv"
    )


def focus_csv(entries: list[PriceEntry], path: Path) -> Path:
    """``entries`` as a FOCUS 1.5 SKU Price file, one row per token rate."""
    columns = [
        "ServiceProviderName",
        "SkuId",
        "SkuPriceId",
        "ContractId",
        "PricingCurrency",
        "PricingUnit",
        "UnitPrice",
        "UnitPriceType",
        "SkuPriceEffectiveStart",
        "SkuPriceEffectiveEnd",
        "SkuPriceDetails",
    ]
    labels = {
        "input": ("Input", "Uncached"),
        "cache_read": ("Input", "Read"),
        "cache_write": ("Input", "Write"),
        "output": ("Output", None),
    }
    lines = [",".join(columns)]
    for n, entry in enumerate(entries):
        for key, (direction, action) in labels.items():
            details = {"ModelId": entry.model, "TokenDirection": direction}
            if action:
                details["TokenCacheAction"] = action
            cells = [
                "Example",
                entry.model,
                f"{entry.model}-{key}-{n}",
                "",
                "USD",
                "1M Tokens",
                repr(getattr(entry.rates, key)),
                "List",
                entry.effective_from.isoformat(),
                "",
                '"' + json.dumps(details).replace('"', '""') + '"',
            ]
            lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(
    params=[_json, _memory, _focus, _focus_details],
    ids=["json", "in-memory", "focus-sku-price", "focus-sku-price-details"],
)
def book(request: pytest.FixtureRequest) -> PriceBook:
    factory: Factory = request.param
    return factory()


@pytest.mark.req("TER-CNF-002")
@pytest.mark.parametrize("obligation", [pytest.param(o, id=o.id) for o in PRICE_BOOK])
def test_meets_the_conformance_kit_obligation(
    book: PriceBook, obligation: Obligation[PriceSubject]
) -> None:
    """The kit adapter authors run (``python -m ter conformance PriceBook``)
    is this suite, unchanged (ADR 0008 decision 7)."""
    obligation(PriceSubject(book, today=date(2026, 9, 29)))


# --- Dated lookup at the boundaries (TER-ANL-040) ----------------------------

_EARLY = Rates(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)
_LATE = Rates(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)
_FIRST = date(2026, 1, 1)
_SECOND = date(2026, 4, 1)


def _dated_entries() -> list[PriceEntry]:
    return [
        PriceEntry("dated-model", _FIRST, _EARLY, source="contract test"),
        PriceEntry("dated-model", _SECOND, _LATE, source="contract test"),
    ]


def _dated_json(tmp_path: Path) -> PriceBook:
    def rates(r: Rates) -> dict[str, float]:
        return {
            "input": r.input,
            "output": r.output,
            "cache_read": r.cache_read,
            "cache_write": r.cache_write,
        }

    document = {
        "schema": "ter.price_book/1",
        "currency": "USD",
        "unit": "per_million_tokens",
        "prices": [
            {
                "model": e.model,
                "effective_from": e.effective_from.isoformat(),
                "source": e.source,
                "rates": rates(e.rates),
            }
            for e in _dated_entries()
        ],
    }
    path = tmp_path / "dated.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return JsonPriceBook(path)


@pytest.fixture(params=["json", "in-memory", "focus-sku-price"])
def dated_book(request: pytest.FixtureRequest, tmp_path: Path) -> PriceBook:
    if request.param == "json":
        return _dated_json(tmp_path)
    if request.param == "focus-sku-price":
        return FocusSkuPriceBook(focus_csv(_dated_entries(), tmp_path / "dated.csv"))
    return InMemoryPriceBook(_dated_entries())


@pytest.mark.req("TER-ANL-040")
@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (_SECOND, _LATE),
        (_SECOND - timedelta(days=1), _EARLY),
        (date(2026, 2, 15), _EARLY),
        (_FIRST, _EARLY),
        (date(2027, 1, 1), _LATE),
    ],
    ids=["exact-date", "day-before", "between", "first-date", "after-latest"],
)
def test_the_latest_entry_not_after_the_date_wins(
    dated_book: PriceBook, day: date, expected: Rates
) -> None:
    assert dated_book.rate("dated-model", day) == expected


@pytest.mark.req("TER-ANL-040")
def test_before_the_earliest_entry_there_is_no_price(dated_book: PriceBook) -> None:
    with pytest.raises(UnknownModelError):
        dated_book.rate("dated-model", _FIRST - timedelta(days=1))
