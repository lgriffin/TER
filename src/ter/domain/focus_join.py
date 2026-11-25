"""TER Records joined to FOCUS billing rows (Stage U2, ADR 0008,
TER-EXP-011 to TER-EXP-013).

FOCUS keeps the session out of billing: a Cost and Usage row says which
model was used, by whom and in which charge period, never in which session.
So the join is many to one. A Record joins every row of a model it used whose
charge period holds the Record's day, and its usage of that model gives its
share of the row's tokens:

- a row that prices one kind of token (its SKU Price Details name a token
  direction and cache action) takes the Record's tokens of that kind; a row
  with no kind takes all of the Record's tokens of the model;
- Records carry no principal (ADR 0007), so when rows of one model, kind
  and period differ only by principal, a Record's tokens are shared among
  them in proportion to their consumed tokens, never counted in each.

Each joined row gets efficiency figures from the Records behind it: how
many, their tokens and share of the row, the share of their cost by activity
class, and how many have an accepted outcome. A Record that joins no row is
counted with its reason. Every figure is a count, a share or a sum of the
Records' own figures: nothing here reads a session.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

__all__ = [
    "TOKEN_KINDS",
    "focus_token_kind",
    "tokens_per_unit",
    "BillingRow",
    "JoinResult",
    "RecordFacts",
    "RowEfficiency",
    "join_records",
    "model_matches",
]

#: The Record's usage fields, which are also the row kinds a FOCUS token price
#: names (Input/Uncached, Input/Read, Input/Write, Output).
TOKEN_KINDS: tuple[str, ...] = ("input", "cache_read", "cache_write", "output")


_UNIT = re.compile(
    r"^\s*(?:(?P<n>[\d,]+(?:\.\d+)?)\s*(?P<scale>[KkMm])?\s*|(?P<word>thousand|million)\s+)?"
    r"tokens?\s*$",
    re.IGNORECASE,
)
_SCALE = {"k": 1_000, "m": 1_000_000, "thousand": 1_000, "million": 1_000_000}
_KIND_OF = {
    ("Input", "Uncached"): "input",
    ("Input", "Read"): "cache_read",
    ("Input", "Write"): "cache_write",
}


def tokens_per_unit(unit: str) -> int | None:
    """How many tokens one FOCUS unit holds (``"1M Tokens"`` is 1,000,000,
    ``"Tokens"`` 1), or None when the unit is not a whole count of tokens."""
    match = _UNIT.match(unit)
    if match is None:
        return None
    if match.group("word"):
        return _SCALE[match.group("word").lower()]
    raw = match.group("n")
    count = 1.0 if raw is None else float(raw.replace(",", ""))
    scale = match.group("scale")
    if scale:
        count *= _SCALE[scale.lower()]
    whole = round(count)
    return whole if whole >= 1 and abs(whole - count) < 1e-9 else None


def focus_token_kind(direction: object, action: object) -> str | None:
    """The token kind a FOCUS 1.5 price prices, from its SKU Price Details
    ``TokenDirection`` and ``TokenCacheAction``: one of :data:`TOKEN_KINDS`,
    or None when the labels name no single kind (``Other``, missing).
    Output is one kind whatever its cache action; reasoning tokens metered
    separately are output too."""
    if direction == "Output":
        return "output"
    if direction == "Input":
        return _KIND_OF.get(("Input", str(action)))
    return None


@dataclass(frozen=True)
class BillingRow:
    """What the join needs from one FOCUS Cost and Usage row."""

    model: str
    start: datetime
    end: datetime
    #: One of :data:`TOKEN_KINDS`, or None for a row that bills every kind.
    kind: str | None
    #: Tokens the row consumed, or None when its unit is not tokens.
    tokens: float | None
    principal: str | None = None


@dataclass(frozen=True)
class RecordFacts:
    """What the join needs from one TER Record."""

    #: Tokens by model and kind, from the Record's ``usage.by_model``.
    usage: Mapping[str, Mapping[str, int]]
    day: date | None
    usd: float = 0.0
    #: Cost by activity class (``value_adding``, ``avoidable``, ...).
    cost_by_class: Mapping[str, float] = field(default_factory=dict)
    accepted: bool = False


@dataclass(frozen=True)
class RowEfficiency:
    """The efficiency figures a billing row gains from its Records."""

    records: int
    record_tokens: float
    #: ``record_tokens`` over the row's consumed tokens; None without them.
    token_share: float | None
    #: Share of the joined Records' priced cost by activity class; None when
    #: none of them was priced.
    cost_shares: Mapping[str, float] | None
    verified_outcomes: int


@dataclass(frozen=True)
class JoinResult:
    rows: tuple[RowEfficiency | None, ...]
    joined: int
    #: Records that joined no row, by reason.
    unjoined: Mapping[str, int]


def model_matches(record_model: str, row_model: str) -> bool:
    """A Record's model is the row's, or a dated version of it
    (``claude-sonnet-5-5-20261001`` bills as ``claude-sonnet-5-5``)."""
    if record_model == row_model:
        return True
    prefix = row_model + "-"
    suffix = record_model[len(prefix) :]
    # Only a date suffix: claude-sonnet-5-5 is not a version of claude-sonnet-5.
    return record_model.startswith(prefix) and len(suffix) == 8 and suffix.isdigit()


def _midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _covers(row: BillingRow, day: date) -> bool:
    at = _midnight(day)
    return row.start <= at < row.end


def _tokens(usage: Mapping[str, int], kind: str | None) -> int:
    if kind is None:
        return sum(int(usage.get(k, 0)) for k in TOKEN_KINDS)
    return int(usage.get(kind, 0))


def join_records(
    rows: Sequence[BillingRow], records: Sequence[RecordFacts]
) -> JoinResult:
    """Join ``records`` to ``rows`` (TER-EXP-012, TER-EXP-013)."""
    groups: dict[tuple[str, str | None, datetime, datetime], list[int]] = {}
    for index, row in enumerate(rows):
        if row.tokens is not None:
            groups.setdefault((row.model, row.kind, row.start, row.end), []).append(
                index
            )

    tokens = [0.0] * len(rows)
    members: list[list[int]] = [[] for _ in rows]
    unjoined: Counter[str] = Counter()
    joined = 0
    for number, record in enumerate(records):
        if record.day is None:
            unjoined["no charge period"] += 1
            continue
        if not record.usage:
            unjoined["no model turns"] += 1
            continue
        hit = False
        for (model, kind, _, _), indexes in groups.items():
            if not _covers(rows[indexes[0]], record.day):
                continue
            matching = [u for m, u in record.usage.items() if model_matches(m, model)]
            if not matching:
                continue
            used = sum(_tokens(u, kind) for u in matching)
            hit = True
            weights = [rows[i].tokens or 0.0 for i in indexes]
            total = sum(weights)
            for i, weight in zip(indexes, weights):
                part = used / len(indexes) if total <= 0 else used * weight / total
                tokens[i] += part
                members[i].append(number)
        if hit:
            joined += 1
        else:
            unjoined["no billing row for its models and charge period"] += 1

    out: list[RowEfficiency | None] = []
    for index, row in enumerate(rows):
        if not members[index]:
            out.append(None)
            continue
        behind = [records[n] for n in members[index]]
        priced = [r for r in behind if r.usd > 0]
        spent = sum(r.usd for r in priced)
        shares: dict[str, float] | None = None
        if spent > 0:
            classes = sorted({c for r in priced for c in r.cost_by_class})
            shares = {
                c: sum(r.cost_by_class.get(c, 0.0) for r in priced) / spent
                for c in classes
            }
        consumed = row.tokens
        out.append(
            RowEfficiency(
                records=len(behind),
                record_tokens=tokens[index],
                token_share=tokens[index] / consumed if consumed else None,
                cost_shares=shares,
                verified_outcomes=sum(1 for r in behind if r.accepted),
            )
        )
    return JoinResult(rows=tuple(out), joined=joined, unjoined=dict(unjoined))
