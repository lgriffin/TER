"""A FOCUS SKU Price file as a :class:`~ter.ports.driven.PriceBook`
(Stage U2, ADR 0008, TER-ANL-045 to TER-ANL-047).

FOCUS 1.5 adds a SKU Price dataset (``SkuPrice``, a preview) in which a
provider publishes its prices as rows: one unit price per row, with the
price's type, currency, pricing unit and effective dates. TER reads it as a
CSV file by its 1.5 column contract and turns the rows that price tokens into
dated per-million-token rates.

What a row prices comes from its SKU Price Details, the properties FOCUS 1.5
recommends for token prices: ``TokenDirection`` (``Input``, ``Output``),
``TokenCacheAction`` (``Uncached``, ``Read``, ``Write``, ``Other``) and
``ModelId``, the model as it appears in billing. The 1.5 SKU Price dataset
defers its own SKU Price Details column, so the details are taken from the
row when it carries the column, else from the provider's FOCUS Cost and Usage
file given beside it, matched by ``SkuPriceId``.

======================================  =================
SKU Price Details                       TER rate
======================================  =================
Input, Uncached                         ``input``
Input, Read                             ``cache_read``
Input, Write                            ``cache_write``
Output (any cache action, or none)      ``output``
======================================  =================

Only public list prices are read: ``UnitPriceType`` ``List`` with no
``ContractId``, in US dollars (TER's rates are USD list prices, ADR 0003).
A row that prices something else, or several token kinds on one meter (no
``TokenDirection``), is skipped and counted in :attr:`FocusSkuPriceBook.skipped`.
A model whose rows lack one of the four rates on a date is not priced on that
date rather than priced by guess; it is listed in
:attr:`FocusSkuPriceBook.incomplete`.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ....domain.focus_join import focus_token_kind, tokens_per_unit
from ....domain.pricing import (
    TOKENS_PER_RATE_UNIT,
    PriceEntry,
    PriceSchedule,
    Rates,
)

__all__ = [
    "FOCUS_VERSIONS",
    "REQUIRED_COLUMNS",
    "FocusPriceError",
    "FocusSkuPriceBook",
    "tokens_per_unit",
]

#: FOCUS versions whose SKU Price column contract this adapter reads.
FOCUS_VERSIONS: tuple[str, ...] = ("1.5",)
#: The 1.5 preview's columns this adapter needs (TER-ANL-046).
REQUIRED_COLUMNS: tuple[str, ...] = (
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
)
_DETAILS = "SkuPriceDetails"
_EPOCH = date(1970, 1, 1)


class FocusPriceError(ValueError):
    """A file that cannot be read as a FOCUS SKU Price file of a known version."""


class FocusSkuPriceBook:
    """Dated token rates read from a FOCUS 1.5 SKU Price CSV file."""

    def __init__(
        self,
        path: str | Path,
        *,
        details: str | Path | None = None,
        focus_version: str = "1.5",
    ) -> None:
        if focus_version not in FOCUS_VERSIONS:
            raise FocusPriceError(
                f"FOCUS {focus_version} is not a version this price book reads "
                f"({', '.join(FOCUS_VERSIONS)})"
            )
        self.focus_version = focus_version
        # No file name: the name travels into Records, which carry no paths.
        self.name = f"focus-sku-price-{focus_version}"
        self.path = Path(path)
        rows = _rows(Path(path), REQUIRED_COLUMNS)
        by_price = _details_by_price(Path(details)) if details is not None else {}
        #: Rows skipped, by reason (not a public USD list price, not a token
        #: price, no details, a unit that is not tokens).
        self.skipped: Counter[str] = Counter()
        #: ``(model, effective date, missing rates)`` not priced on that date.
        self.incomplete: list[tuple[str, date, tuple[str, ...]]] = []
        self._schedule = PriceSchedule(self._entries(rows, by_price))

    def models(self) -> tuple[str, ...]:
        return self._schedule.models()

    def entry(self, model: str, at: date | None = None) -> PriceEntry:
        return self._schedule.entry(model, at)

    def rate(self, model: str, at: date | None = None) -> Rates:
        return self._schedule.rate(model, at)

    def _entries(
        self,
        rows: Iterable[Mapping[str, str]],
        by_price: Mapping[str, Mapping[str, Any]],
    ) -> list[PriceEntry]:
        found: dict[tuple[str, date], dict[str, tuple[float, str]]] = {}
        for row in rows:
            if (
                row["UnitPriceType"].strip() != "List"
                or row["ContractId"].strip()
                or row["PricingCurrency"].strip() != "USD"
            ):
                self.skipped["not a public USD list price"] += 1
                continue
            price_id = row["SkuPriceId"].strip()
            details = _details(row.get(_DETAILS)) or by_price.get(price_id)
            if not details:
                self.skipped["no SKU Price Details"] += 1
                continue
            model = details.get("ModelId")
            direction = details.get("TokenDirection")
            action = details.get("TokenCacheAction")
            if (
                not isinstance(model, str)
                or not model
                or direction not in ("Input", "Output")
            ):
                self.skipped["not a token price of one kind"] += 1
                continue
            key = focus_token_kind(direction, action)
            if key is None:
                self.skipped["not a token price of one kind"] += 1
                continue
            per_unit = tokens_per_unit(row["PricingUnit"])
            if per_unit is None:
                self.skipped["pricing unit is not tokens"] += 1
                continue
            try:
                price = float(row["UnitPrice"])
            except ValueError:
                self.skipped["unit price is not a number"] += 1
                continue
            if price < 0 or price != price or price in (float("inf"),):
                self.skipped["unit price is not a number"] += 1
                continue
            start = _day(row["SkuPriceEffectiveStart"])
            if start is None:
                self.skipped["effective start is not a date"] += 1
                continue
            rate = price * TOKENS_PER_RATE_UNIT / per_unit
            found.setdefault((model, start), {})[key] = (rate, price_id)
        entries: list[PriceEntry] = []
        for (model, start), rates in sorted(found.items()):
            missing = tuple(
                k
                for k in ("input", "output", "cache_read", "cache_write")
                if k not in rates
            )
            if missing:
                self.incomplete.append((model, start, missing))
                continue
            ids = ", ".join(sorted({price_id for _, price_id in rates.values()}))
            entries.append(
                PriceEntry(
                    model=model,
                    effective_from=start,
                    rates=Rates(**{k: v for k, (v, _) in rates.items()}),
                    source=f"FOCUS {self.focus_version} SKU Price ({ids})",
                )
            )
        if not entries:
            raise FocusPriceError(
                f"{self.name} prices no model with all four token rates "
                f"(skipped: {dict(self.skipped) or 'none'}, "
                f"incomplete: {len(self.incomplete)})"
            )
        return entries


def _rows(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FocusPriceError(f"{path} is not UTF-8") from exc
    try:
        reader = csv.DictReader(io.StringIO(text))
        header = reader.fieldnames or []
        missing = [c for c in required if c not in header]
        if missing:
            raise FocusPriceError(
                f"{path} is not a FOCUS SKU Price file: no column {', '.join(missing)}"
            )
        rows = [
            {
                k: v if isinstance(v, str) else ""
                for k, v in row.items()
                if isinstance(k, str)
            }
            for row in reader
        ]
    except csv.Error as exc:
        raise FocusPriceError(f"{path} is not CSV: {exc}") from exc
    for row in rows:
        for column in required:
            row.setdefault(column, "")
    return rows


def _details_by_price(path: Path) -> dict[str, Mapping[str, Any]]:
    """SKU Price Details by SKU Price ID, from a FOCUS Cost and Usage file."""
    found: dict[str, Mapping[str, Any]] = {}
    for row in _rows(path, ("SkuPriceId", _DETAILS)):
        details = _details(row.get(_DETAILS))
        price_id = row["SkuPriceId"].strip()
        if details and price_id:
            found.setdefault(price_id, details)
    return found


def _details(raw: object) -> Mapping[str, Any] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _day(raw: str) -> date | None:
    text = raw.strip()
    if not text:
        return _EPOCH  # null: in effect for as long as the file says
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None
