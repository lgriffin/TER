"""``python -m ter record focus``: TER Records joined to FOCUS billing rows
(Stage U2, ADR 0008, TER-EXP-011 to TER-EXP-013).

Reads a FOCUS Cost and Usage CSV file and any number of ``ter.record/0.1``
files, joins the Records to the rows of their models and charge periods
(:func:`ter.domain.focus_join.join_records`) and writes the billing file back
with efficiency columns added. Every added column carries the ``x_`` prefix
FOCUS reserves for custom columns, and every row and FOCUS column is written
as it was read (TER-EXP-011).
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ...domain.focus_join import (
    TOKEN_KINDS,
    BillingRow,
    JoinResult,
    RecordFacts,
    RowEfficiency,
    focus_token_kind,
    join_records,
    tokens_per_unit,
)
from ...domain.ter_record import RECORD_SCHEMA

__all__ = [
    "BILLING_COLUMNS",
    "EFFICIENCY_COLUMNS",
    "FocusExport",
    "FocusExportError",
    "export_focus",
]

#: The FOCUS Cost and Usage columns the join reads.
BILLING_COLUMNS: tuple[str, ...] = (
    "ChargePeriodStart",
    "ChargePeriodEnd",
    "ConsumedQuantity",
    "ConsumedUnit",
    "SkuId",
)
_CLASSES = (
    ("value_adding", "x_TerValueAddingCostShare"),
    ("necessary_non_value_adding", "x_TerNecessaryCostShare"),
    ("avoidable", "x_TerAvoidableCostShare"),
    ("uncertain", "x_TerUncertainCostShare"),
)
#: The columns the exporter adds, all ``x_`` custom columns (TER-EXP-011).
EFFICIENCY_COLUMNS: tuple[str, ...] = (
    "x_TerRecordCount",
    "x_TerRecordTokens",
    "x_TerTokenShare",
    *(column for _, column in _CLASSES),
    "x_TerVerifiedOutcomes",
)


class FocusExportError(ValueError):
    """A billing file or record that cannot be joined."""


@dataclass(frozen=True)
class FocusExport:
    """The joined billing file and what the join found."""

    text: str
    rows: int
    rows_joined: int
    records: int
    result: JoinResult


def export_focus(billing: Path, records: Sequence[Path]) -> FocusExport:
    """Join ``records`` to the FOCUS Cost and Usage file ``billing``."""
    header, raw_rows = _billing(billing)
    clash = [c for c in EFFICIENCY_COLUMNS if c in header]
    if clash:
        raise FocusExportError(
            f"{billing} already has column(s) {', '.join(clash)}; "
            "export from the provider's file, not from a joined one"
        )
    rows: list[BillingRow | None] = [_row(raw) for raw in raw_rows]
    joinable = [r for r in rows if r is not None]
    facts = [_facts(path) for path in records]
    result = join_records(joinable, facts)
    figures = iter(result.rows)
    out = io.StringIO()
    writer = csv.DictWriter(out, [*header, *EFFICIENCY_COLUMNS], lineterminator="\n")
    writer.writeheader()
    joined = 0
    for raw, row in zip(raw_rows, rows):
        efficiency = next(figures) if row is not None else None
        joined += efficiency is not None
        writer.writerow({**raw, **_columns(efficiency)})
    return FocusExport(
        text=out.getvalue(),
        rows=len(raw_rows),
        rows_joined=joined,
        records=len(records),
        result=result,
    )


def _billing(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FocusExportError(f"{path} is not UTF-8") from exc
    try:
        reader = csv.DictReader(io.StringIO(text))
        header = [h for h in (reader.fieldnames or []) if isinstance(h, str)]
        missing = [c for c in BILLING_COLUMNS if c not in header]
        if missing:
            raise FocusExportError(
                f"{path} is not a FOCUS Cost and Usage file: no column {', '.join(missing)}"
            )
        rows = [
            {
                k: (v if isinstance(v, str) else "")
                for k, v in row.items()
                if k in header
            }
            for row in reader
        ]
    except csv.Error as exc:
        raise FocusExportError(f"{path} is not CSV: {exc}") from exc
    return header, rows


def _row(raw: Mapping[str, str]) -> BillingRow | None:
    """The joinable part of a billing row, or None for a row the join cannot
    place (no period, no tokens). Such a row is still written unchanged."""
    start, end = (
        _time(raw.get("ChargePeriodStart", "")),
        _time(raw.get("ChargePeriodEnd", "")),
    )
    if start is None or end is None or end <= start:
        return None
    details = _details(raw.get("SkuPriceDetails", ""))
    model = details.get("ModelId") if details else None
    if not isinstance(model, str) or not model:
        model = raw.get("SkuId", "").strip()
    if not model:
        return None
    per_unit = tokens_per_unit(raw.get("ConsumedUnit", ""))
    tokens: float | None = None
    if per_unit is not None:
        try:
            quantity = float(raw.get("ConsumedQuantity", ""))
        except ValueError:
            quantity = float("nan")
        if quantity == quantity and 0 <= quantity < float("inf"):
            tokens = quantity * per_unit
    if tokens is None:
        return None
    kind = None
    if details and "TokenDirection" in details:
        kind = focus_token_kind(
            details.get("TokenDirection"), details.get("TokenCacheAction")
        )
        if kind is None:
            return None  # a token kind TER does not count (Other)
    principal = raw.get("PrincipalId", "").strip() or None
    return BillingRow(
        model=model, start=start, end=end, kind=kind, tokens=tokens, principal=principal
    )


def _facts(path: Path) -> RecordFacts:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise FocusExportError(f"cannot read record {path}: {exc}") from None
    if not isinstance(doc, dict) or doc.get("schema") != RECORD_SCHEMA:
        raise FocusExportError(f"{path} is not a {RECORD_SCHEMA} record")
    usage = doc.get("usage")
    by_model = usage.get("by_model") if isinstance(usage, dict) else None
    if not isinstance(by_model, list):
        raise FocusExportError(
            f"{path} has no usage.by_model; export it again with this version of TER"
        )
    models: dict[str, dict[str, int]] = {}
    for item in by_model:
        if not isinstance(item, dict):
            raise FocusExportError(f"{path}: usage.by_model holds a non-object")
        model = item.get("model")
        if model is None:
            continue  # a turn whose source named no model joins no model's row
        if not isinstance(model, str):
            raise FocusExportError(f"{path}: a usage.by_model model is not a string")
        models[model] = {k: _count(item.get(k)) for k in TOKEN_KINDS}
    join = doc.get("join")
    day = _date(join.get("charge_period") if isinstance(join, dict) else None)
    cost = doc.get("cost")
    usd = 0.0
    classes: dict[str, float] = {}
    if isinstance(cost, dict):
        usd = _number(cost.get("usd"))
        raw = cost.get("by_activity_class")
        if isinstance(raw, dict):
            classes = {str(k): _number(v) for k, v in raw.items()}
    outcome = doc.get("outcome")
    accepted = isinstance(outcome, dict) and outcome.get("status") == "accepted"
    return RecordFacts(
        usage=models, day=day, usd=usd, cost_by_class=classes, accepted=accepted
    )


def _columns(efficiency: RowEfficiency | None) -> dict[str, str]:
    if efficiency is None:
        return {column: "" for column in EFFICIENCY_COLUMNS}
    shares = efficiency.cost_shares
    out = {
        "x_TerRecordCount": str(efficiency.records),
        "x_TerRecordTokens": _fmt(efficiency.record_tokens),
        "x_TerTokenShare": ""
        if efficiency.token_share is None
        else _fmt(efficiency.token_share),
        "x_TerVerifiedOutcomes": str(efficiency.verified_outcomes),
    }
    for cls, column in _CLASSES:
        out[column] = "" if shares is None else _fmt(shares.get(cls, 0.0))
    return out


def _fmt(value: float) -> str:
    rounded = round(value, 6)
    return (
        str(int(rounded)) if rounded == int(rounded) else f"{rounded:.6f}".rstrip("0")
    )


def _details(raw: str) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _time(raw: str) -> datetime | None:
    text = raw.strip()
    if not text:
        return None
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value


def _date(raw: object) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _count(value: object) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    number = float(value)
    return number if number == number and 0 <= number < float("inf") else 0.0
