"""A FOCUS 1.5 SKU Price file as a price book (Stage U2, ADR 0008)."""

from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ter import bootstrap
from ter.adapters.driven.pricing import JsonPriceBook, default_price_book
from ter.adapters.driven.pricing.focus import (
    REQUIRED_COLUMNS,
    FocusPriceError,
    FocusSkuPriceBook,
    tokens_per_unit,
)
from ter.adapters.driving.cli import main as cli_main
from ter.domain import PriceEntry, Rates, UnknownModelError

from tests.contract.test_price_book import focus_csv

FOCUS = Path(__file__).resolve().parents[1] / "fixtures" / "focus"
RATES = Rates(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)


def _write(tmp_path: Path, rows: list[dict[str, str]], name: str = "p.csv") -> Path:
    columns = [*REQUIRED_COLUMNS, "SkuPriceDetails"]
    out = io.StringIO()
    import csv

    writer = csv.DictWriter(out, columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    path = tmp_path / name
    path.write_text(out.getvalue(), encoding="utf-8")
    return path


def _row(rate: str, direction: str, action: str | None, **over: str) -> dict[str, str]:
    details: dict[str, str] = {"ModelId": "m", "TokenDirection": direction}
    if action is not None:
        details["TokenCacheAction"] = action
    row = {
        "ServiceProviderName": "P",
        "SkuId": "m",
        "SkuPriceId": f"m-{direction}-{action}",
        "PricingCurrency": "USD",
        "PricingUnit": "1M Tokens",
        "UnitPrice": rate,
        "UnitPriceType": "List",
        "SkuPriceEffectiveStart": "2026-01-01T00:00:00Z",
        "SkuPriceDetails": json.dumps(details),
    }
    row.update(over)
    return row


def _full(**over: str) -> list[dict[str, str]]:
    return [
        _row("3", "Input", "Uncached", **over),
        _row("0.3", "Input", "Read", **over),
        _row("3.75", "Input", "Write", **over),
        _row("15", "Output", None, **over),
    ]


@pytest.mark.req("TER-ANL-045")
def test_token_rows_become_rates_by_direction_and_cache_action(tmp_path: Path) -> None:
    book = FocusSkuPriceBook(_write(tmp_path, _full()))
    assert book.models() == ("m",)
    assert book.rate("m") == RATES
    assert book.entry("m").effective_from == date(2026, 1, 1)
    assert "m-Input-Uncached" in book.entry("m").source
    assert book.name == "focus-sku-price-1.5"  # no file name: Records carry no paths


@pytest.mark.req("TER-ANL-045")
@pytest.mark.parametrize(
    ("unit", "price", "per_million"),
    [
        ("Tokens", "0.000003", 3.0),
        ("1000 Tokens", "0.003", 3.0),
        ("1K Tokens", "0.003", 3.0),
        ("1,000,000 Tokens", "3", 3.0),
        ("Million Tokens", "3", 3.0),
    ],
)
def test_a_pricing_unit_of_any_token_multiple_becomes_a_per_million_rate(
    tmp_path: Path, unit: str, price: str, per_million: float
) -> None:
    rows = _full(PricingUnit=unit)
    rows[0]["UnitPrice"] = price
    assert FocusSkuPriceBook(_write(tmp_path, rows)).rate("m").input == pytest.approx(
        per_million
    )


@pytest.mark.parametrize(
    "unit", ["Hours", "1 Request", "0.5 Tokens", "Tokens per day", ""]
)
def test_a_unit_that_is_not_a_count_of_tokens_is_not_read(unit: str) -> None:
    assert tokens_per_unit(unit) is None


@pytest.mark.req("TER-ANL-045")
def test_only_public_usd_list_prices_are_read(tmp_path: Path) -> None:
    rows = _full()
    rows += [
        _row("1", "Input", "Uncached", UnitPriceType="Contracted", SkuPriceId="c1"),
        _row("1", "Input", "Uncached", ContractId="C-9", SkuPriceId="c2"),
        _row("1", "Input", "Uncached", PricingCurrency="EUR", SkuPriceId="c3"),
    ]
    book = FocusSkuPriceBook(_write(tmp_path, rows))
    assert book.rate("m") == RATES
    assert book.skipped == {"not a public USD list price": 3}


@pytest.mark.req("TER-ANL-045")
def test_labels_come_from_cost_and_usage_rows_when_the_price_row_has_none(
    tmp_path: Path,
) -> None:
    book = FocusSkuPriceBook(
        FOCUS / "sku_price.no-details.csv", details=FOCUS / "cost_and_usage.csv"
    )
    shipped = default_price_book()
    for model in book.models():
        assert book.rate(model) == shipped.rate(model)
    without = _write(tmp_path, [{**r, "SkuPriceDetails": ""} for r in _full()])
    with pytest.raises(FocusPriceError, match="no SKU Price Details"):
        FocusSkuPriceBook(without)


@pytest.mark.req("TER-ANL-045")
def test_a_meter_that_bills_several_token_kinds_is_skipped(tmp_path: Path) -> None:
    mixed = _row("9", "Input", "Uncached", SkuPriceId="mixed")
    mixed["SkuPriceDetails"] = json.dumps({"ModelId": "m"})
    other = _row("9", "Input", "Other", SkuPriceId="other")
    book = FocusSkuPriceBook(_write(tmp_path, [*_full(), mixed, other]))
    assert book.rate("m") == RATES
    assert book.skipped == {"not a token price of one kind": 2}


@pytest.mark.req("TER-ANL-046")
def test_a_file_missing_a_column_of_the_version_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "p.csv"
    path.write_text("SkuPriceId,UnitPrice\nx,1\n", encoding="utf-8")
    with pytest.raises(FocusPriceError, match="ServiceProviderName.*UnitPriceType"):
        FocusSkuPriceBook(path)


@pytest.mark.req("TER-ANL-046")
def test_an_unknown_focus_version_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FocusPriceError, match="1.4"):
        FocusSkuPriceBook(_write(tmp_path, _full()), focus_version="1.4")


@pytest.mark.req("TER-ANL-047")
def test_a_date_that_restates_some_rates_carries_the_others_forward(
    tmp_path: Path,
) -> None:
    """A provider that changes only its output price publishes one row from
    the new date; the other three rates stay in force (Qodo #115 finding 9)."""
    output = _row("20", "Output", None, SkuPriceEffectiveStart="2026-06-01")
    output["SkuPriceId"] += "-late"
    book = FocusSkuPriceBook(_write(tmp_path, [*_full(), output]))
    assert book.incomplete == []
    assert book.rate("m", date(2026, 5, 31)) == RATES
    later = Rates(input=3.0, output=20.0, cache_read=0.3, cache_write=3.75)
    assert book.rate("m", date(2026, 6, 1)) == later
    assert book.rate("m") == later
    assert "m-Output-None-late" in book.entry("m").source


@pytest.mark.req("TER-ANL-047")
def test_a_model_missing_a_rate_on_a_date_is_not_priced_on_that_date(
    tmp_path: Path,
) -> None:
    three = [r for r in _full() if "Write" not in r["SkuPriceId"]]
    write = _row("3.75", "Input", "Write", SkuPriceEffectiveStart="2026-06-01")
    book = FocusSkuPriceBook(_write(tmp_path, [*three, write]))
    assert book.incomplete == [("m", date(2026, 1, 1), ("cache_write",))]
    with pytest.raises(UnknownModelError):
        book.rate("m", date(2026, 5, 31))
    assert book.rate("m", date(2026, 6, 1)) == RATES
    with pytest.raises(FocusPriceError, match="prices no model"):
        FocusSkuPriceBook(_write(tmp_path, three, "three.csv"))


@pytest.mark.req("TER-ANL-047")
def test_a_price_that_ends_with_no_successor_leaves_the_model_unpriced(
    tmp_path: Path,
) -> None:
    """SkuPriceEffectiveEnd is read: after it, sessions are unpriced rather
    than priced from an expired rate (Qodo #115 finding 8)."""
    ended = _full(SkuPriceEffectiveEnd="2026-06-01T00:00:00Z")
    book = FocusSkuPriceBook(_write(tmp_path, ended))
    assert book.rate("m", date(2026, 5, 31)) == RATES
    with pytest.raises(UnknownModelError, match="ended on 2026-06-01"):
        book.rate("m", date(2026, 6, 1))
    assert book.incomplete == [
        ("m", date(2026, 6, 1), ("input", "output", "cache_read", "cache_write"))
    ]
    # One kind ending is enough to leave the model unpriced from that day,
    # until a later complete set starts.
    one = _full()
    one[3]["SkuPriceEffectiveEnd"] = "2026-03-01"
    again = _row("16", "Output", None, SkuPriceEffectiveStart="2026-04-01")
    again["SkuPriceId"] += "-again"
    book = FocusSkuPriceBook(_write(tmp_path, [*one, again], "gap.csv"))
    assert book.rate("m", date(2026, 2, 28)) == RATES
    with pytest.raises(UnknownModelError):
        book.rate("m", date(2026, 3, 15))
    assert book.rate("m", date(2026, 4, 1)).output == 16.0


@pytest.mark.req("TER-ANL-047")
@pytest.mark.parametrize(
    ("end", "reason"),
    [
        ("2026-01-01", "effective end is not after its start"),
        ("2025-12-01", "effective end is not after its start"),
        ("soon", "effective dates are not dates"),
    ],
)
def test_a_row_whose_end_is_not_after_its_start_is_skipped(
    tmp_path: Path, end: str, reason: str
) -> None:
    rows = _full()
    rows.append(_row("99", "Output", None, SkuPriceEffectiveEnd=end, SkuPriceId="bad"))
    rows[-1]["SkuPriceEffectiveStart"] = "2026-01-01"
    rows[-1]["SkuPriceId"] = "bad"
    book = FocusSkuPriceBook(_write(tmp_path, rows))
    assert book.skipped == {reason: 1}
    assert book.rate("m") == RATES


@pytest.mark.req("TER-ANL-047")
def test_two_different_prices_for_one_kind_and_day_are_refused(tmp_path: Path) -> None:
    """Reordering the file must not change a rate (Qodo #115 finding 3)."""
    clash = _row("16", "Output", None)
    clash["SkuPriceId"] = "m-out-2"
    with pytest.raises(FocusPriceError, match="two output prices from 2026-01-01"):
        FocusSkuPriceBook(_write(tmp_path, [*_full(), clash]))
    same = _row("15", "Output", None)
    same["SkuPriceId"] = "m-out-copy"
    assert (
        FocusSkuPriceBook(_write(tmp_path, [*_full(), same], "s.csv")).rate("m")
        == RATES
    )


@pytest.mark.req("TER-ANL-045")
def test_a_dated_model_is_priced_as_the_model_its_provider_bills(
    tmp_path: Path,
) -> None:
    """Qodo #115 finding 1: the session names claude-x-1-20261001, the price
    file names claude-x-1."""
    rows = [
        dict(r, SkuPriceDetails=r["SkuPriceDetails"].replace('"m"', '"claude-x-1"'))
        for r in _full()
    ]
    book = FocusSkuPriceBook(_write(tmp_path, rows))
    assert book.rate("claude-x-1-20261001") == RATES
    assert book.entry("claude-x-1-20261001", date(2026, 2, 1)).model == "claude-x-1"
    for other in (
        "claude-x-1-2026",
        "claude-x-1-2026100a",
        "claude-x",
        "claude-x-1-x20261001",
        "20261001",
    ):
        with pytest.raises(UnknownModelError):
            book.rate(other)


def test_a_null_effective_start_is_in_force_from_the_start(tmp_path: Path) -> None:
    book = FocusSkuPriceBook(_write(tmp_path, _full(SkuPriceEffectiveStart="")))
    assert book.rate("m", date(2000, 1, 1)) == RATES


@pytest.mark.parametrize(
    "content",
    [b"", b"\xff\xfe", b"a,b\n1,2\n", b'ServiceProviderName,"unclosed\n'],
    ids=["empty", "not-utf8", "other-csv", "torn"],
)
def test_a_file_that_is_not_a_sku_price_file_is_refused(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(content)
    with pytest.raises(FocusPriceError):
        FocusSkuPriceBook(path)


def test_bad_values_are_skipped_not_raised(tmp_path: Path) -> None:
    rows = _full()
    rows += [
        _row("lots", "Input", "Uncached", SkuPriceId="b1"),
        _row("-1", "Input", "Uncached", SkuPriceId="b2"),
        _row("1", "Input", "Uncached", SkuPriceId="b3", SkuPriceEffectiveStart="soon"),
        _row("1", "Input", "Uncached", SkuPriceId="b4", PricingUnit="Hours"),
        _row("1", "Input", "Uncached", SkuPriceId="b5", SkuPriceDetails="[1]"),
    ]
    book = FocusSkuPriceBook(_write(tmp_path, rows))
    assert book.rate("m") == RATES
    assert sum(book.skipped.values()) == 5


def test_the_bootstrap_picks_the_book_by_file_type(tmp_path: Path) -> None:
    assert bootstrap.make_price_book() is default_price_book()
    assert isinstance(
        bootstrap.make_price_book(FOCUS / "sku_price.csv"), FocusSkuPriceBook
    )
    shipped = (
        Path(__file__).resolve().parents[2] / "src" / "ter" / "data" / "price_book.json"
    )
    assert isinstance(bootstrap.make_price_book(shipped), JsonPriceBook)
    with pytest.raises(ValueError, match="price-details"):
        bootstrap.make_price_book(None, FOCUS / "cost_and_usage.csv")
    with pytest.raises(ValueError, match="price-details"):
        bootstrap.make_price_book(shipped, FOCUS / "cost_and_usage.csv")


# -- the Record priced from FOCUS -------------------------------------------------------


def _session(tmp_path: Path) -> Path:
    """An OpenTelemetry export of two turns of a model the shipped book prices."""

    def attr(key: str, value: object) -> dict[str, Any]:
        if isinstance(value, int):
            return {"key": key, "value": {"intValue": str(value)}}
        return {"key": key, "value": {"stringValue": str(value)}}

    spans = [
        {
            "traceId": "a" * 32,
            "spanId": f"{n:016x}",
            "startTimeUnixNano": str((1_790_000_000 + n) * 10**9),
            "endTimeUnixNano": str((1_790_000_000 + n) * 10**9 + 1),
            "attributes": [
                attr("gen_ai.operation.name", "chat"),
                attr("gen_ai.request.model", "claude-sonnet-5-5"),
                attr("gen_ai.usage.input_tokens", 120_000),
                attr("gen_ai.usage.output_tokens", 2_000),
                attr("gen_ai.usage.cache_read.input_tokens", 100_000),
                attr("gen_ai.usage.cache_creation.input_tokens", 10_000),
            ],
        }
        for n in (1, 2)
    ]
    path = tmp_path / "spans.json"
    path.write_text(
        json.dumps(
            {
                "resourceSpans": [
                    {
                        "scopeSpans": [
                            {
                                "schemaUrl": "https://opentelemetry.io/schemas/1.45.0",
                                "spans": spans,
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli_main(list(argv), bootstrap.cli_services(), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.req("TER-ANL-045")
def test_a_record_priced_from_focus_matches_the_same_rates_in_the_shipped_book(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    code, shipped, err = _run("record", "export", str(session), "--ter", "off")
    assert code == 0, err
    code, focus, err = _run(
        "record",
        "export",
        str(session),
        "--ter",
        "off",
        "--prices",
        str(FOCUS / "sku_price.csv"),
    )
    assert code == 0, err
    a, b = json.loads(shipped)["cost"], json.loads(focus)["cost"]
    assert b["price_book"] == "focus-sku-price-1.5"
    assert b["usd"] == a["usd"] > 0
    assert b["unpriced_turns"] == 0
    record = tmp_path / "r.json"
    record.write_text(focus, encoding="utf-8")
    code, out, err = _run(
        "record",
        "verify",
        str(record),
        str(session),
        "--prices",
        str(FOCUS / "sku_price.csv"),
    )
    assert code == 0, out + err
    # Verified against the shipped book instead, the price book differs.
    code, out, _ = _run("record", "verify", str(record), str(session))
    assert code == 1 and "cost.price_book" in out


def test_the_record_command_reports_an_unreadable_price_file(tmp_path: Path) -> None:
    session = _session(tmp_path)
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n", encoding="utf-8")
    code, _, err = _run("record", "export", str(session), "--prices", str(bad))
    assert code == 2 and "Cannot read prices" in err
    code, _, err = _run(
        "record", "export", str(session), "--prices", str(tmp_path / "missing.csv")
    )
    assert code == 2 and "No such price file" in err


def test_a_model_the_focus_file_does_not_price_stays_unpriced(tmp_path: Path) -> None:
    other = focus_csv(
        [PriceEntry("other-model", date(2025, 1, 1), RATES, source="t")],
        tmp_path / "other.csv",
    )
    book = FocusSkuPriceBook(other)
    with pytest.raises(UnknownModelError):
        book.rate("claude-sonnet-5-5")
