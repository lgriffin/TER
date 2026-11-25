"""What a session cost, priced from its events and a dated price book.

A session is priced at the prices in force when it ran (TER-ANL-040): the
session date is the UTC date of its first timestamped event, and every model
turn is priced with the price book entry whose ``effective_from`` is the
latest that is not after that date. Turns whose model is unnamed, or has no
price on that date, stay unpriced and are counted, never guessed.

Cost figures are marked *estimated* (TER-ANL-041) when a turn's usage had no
cache fields (the provider did not say what was cached, so cache-priced
figures are guesses), when the session has no model turns at all, when it has
no timestamp to date it by (the latest prices are used), or when some turns
could not be priced. Each reason is listed, so a reader sees why.

Context inventory (``ter.domain.lean.inventory``) is priced as carrying cost:
a context item is paid for on the first model turn after it entered (at the
cache-write rate when that turn shows caching, else the input rate) and again
on every later turn (the cache-read rate when caching, else input). Only
the price book supplies rates; nothing here is hard-coded (ADR 0003).

With the analysis's classifications, the session cost is also split by
activity class (TER-ANL-042, ADR 0007). Provider usage is reported once per
model turn: the turn opens at the event that carries the usage and runs to
the next one. Its whole cost (input, cache read, cache write, output) is
shared among the generated events of the turn in proportion to their
generated tokens (equally when none has any), and each event's part follows
its classification: the confident waste share to ``avoidable``, the
uncertain share to ``uncertain`` (never avoidable, ADR 0006), the rest to its
base class. The class costs sum to the session cost (TER-ANL-043). The same
split of output cost alone is kept beside it, so waste priced by whole turn
can be compared with waste priced by its output tokens.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Protocol

from .events import TokenUsage
from .pricing import (
    TOKENS_PER_RATE_UNIT,
    Rates,
    TokenCounts,
    UnknownModelError,
    token_cost,
    usage_cost,
)

if TYPE_CHECKING:  # the lean package imports this module at runtime
    from .lean.analysis import Classification
    from .lean.inventory import ContextInventory, InventoryItem
    from .lean.model import Step

__all__ = [
    "COST_CLASSES",
    "EstimateReason",
    "ModelCost",
    "Prices",
    "SessionCost",
    "TOKEN_TYPES",
    "price_session",
    "session_date",
]


#: The buckets session cost is split into by activity class (TER-ANL-042):
#: the three Lean classes and the uncertain share kept apart from avoidable.
COST_CLASSES = (
    "value_adding",
    "necessary_non_value_adding",
    "avoidable",
    "uncertain",
)

#: The provider's token types, in the order FOCUS labels them.
TOKEN_TYPES = ("input", "cache_read", "cache_write", "output")


class Prices(Protocol):
    """What costing needs from a price book.

    Structurally the :class:`ter.ports.driven.PriceBook` port, which the
    domain may not import: every price book adapter satisfies both.
    """

    @property
    def name(self) -> str: ...

    def rate(self, model: str, at: date | None = None) -> Rates: ...


class EstimateReason:
    """Stable ids for why a cost figure is an estimate."""

    NO_CACHE_FIELDS = "no-cache-fields"
    NO_USAGE = "no-usage"
    NO_SESSION_DATE = "no-session-date"
    UNPRICED_TURNS = "unpriced-turns"

    TEXT = {
        NO_CACHE_FIELDS: (
            "usage without cache fields: cache reads and writes are unknown, "
            "so context is priced at the input rate"
        ),
        NO_USAGE: "the session reports no provider usage",
        NO_SESSION_DATE: "no event has a timestamp; the latest prices were used",
        UNPRICED_TURNS: "some model turns name no model or one with no price",
    }


@dataclass(frozen=True)
class ModelCost:
    """Turns and cost of one model in the session."""

    model: str
    turns: int
    usd: float


@dataclass(frozen=True)
class SessionCost:
    """A session's cost and the cost of its context inventory, in USD."""

    price_book: str
    priced_on: date | None
    turns: int
    usd: float
    by_model: tuple[ModelCost, ...]
    unpriced_turns: int
    unpriced_models: tuple[str, ...]
    unused_context_usd: float
    reread_context_usd: float
    estimate_reasons: tuple[str, ...]
    #: USD by token type (``TOKEN_TYPES``) over the priced turns.
    by_token_type: tuple[tuple[str, float], ...] = ()
    #: Whole-turn cost by activity class (``COST_CLASSES``), when the
    #: session was priced with its classifications (TER-ANL-042).
    by_activity_class: tuple[tuple[str, float], ...] = ()
    #: The same split of output cost alone: waste priced by its output.
    output_by_activity_class: tuple[tuple[str, float], ...] = ()

    @property
    def estimated(self) -> bool:
        return bool(self.estimate_reasons)

    def class_usd(self, activity_class: str) -> float:
        """Whole-turn cost of one activity class; 0.0 when not split."""
        return dict(self.by_activity_class).get(activity_class, 0.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "price_book": self.price_book,
            "priced_on": None if self.priced_on is None else self.priced_on.isoformat(),
            "estimated": self.estimated,
            "estimate_reasons": list(self.estimate_reasons),
            "turns": self.turns,
            "usd": round(self.usd, 6),
            "by_model": [
                {"model": m.model, "turns": m.turns, "usd": round(m.usd, 6)}
                for m in self.by_model
            ],
            "unpriced_turns": self.unpriced_turns,
            "unpriced_models": list(self.unpriced_models),
            "unused_context_usd": round(self.unused_context_usd, 6),
            "reread_context_usd": round(self.reread_context_usd, 6),
            **(
                {"by_token_type": {k: round(v, 6) for k, v in self.by_token_type}}
                if self.by_token_type
                else {}
            ),
            **(
                {
                    "by_activity_class": {
                        k: round(v, 6) for k, v in self.by_activity_class
                    },
                    "output_by_activity_class": {
                        k: round(v, 6) for k, v in self.output_by_activity_class
                    },
                }
                if self.by_activity_class
                else {}
            ),
        }


def session_date(timestamps: Iterable[datetime | None]) -> date | None:
    """The UTC date of the first timestamp, or None when there is none.

    A naive timestamp is taken as UTC.
    """
    for ts in timestamps:
        if ts is None:
            continue
        if ts.tzinfo is not None:
            ts = ts.astimezone(UTC)
        return ts.date()
    return None


def _caching(usage: TokenUsage) -> bool:
    return usage.cache_creation_tokens > 0 or usage.cache_read_tokens > 0


def price_session(
    steps: tuple[Step, ...],
    inventory: ContextInventory,
    prices: Prices,
    classifications: tuple[Classification, ...] | None = None,
) -> SessionCost:
    """Price every model turn of ``steps`` and the context ``inventory``.

    With ``classifications`` (one per step, as the analysis made them) the
    cost is also split by activity class (TER-ANL-042).

    Linear in the session: one pass prices the turns and builds, per step,
    the per-token cost of carrying one token from that step to the end.
    """
    if classifications is not None and len(classifications) != len(steps):
        raise ValueError(
            f"{len(classifications)} classifications for {len(steps)} steps"
        )
    on = session_date(s.timestamp for s in steps)
    turns = 0
    unpriced = 0
    unpriced_models: set[str] = set()
    no_cache = False
    total = 0.0
    by_model: dict[str, tuple[int, float]] = {}
    # Per-token prices of each turn, for carrying context: (ingest, carry).
    turn_rates: dict[int, tuple[float, float]] = {}
    by_type: dict[str, float] = dict.fromkeys(TOKEN_TYPES, 0.0)
    # Priced turns as (whole cost, output cost, member step indexes).
    priced: list[tuple[float, float, list[int]]] = []
    members: list[int] | None = None  # the open turn's steps; None if unpriced
    for step in steps:
        usage = step.usage
        if usage is None:
            if members is not None and step.is_generated:
                members.append(step.index)
            continue
        members = None
        turns += 1
        no_cache = no_cache or not usage.cache_reported
        rates = _rates(prices, usage.model, on)
        if rates is None:
            unpriced += 1
            unpriced_models.add(usage.model or "<unnamed>")
            continue
        cost = usage_cost(
            TokenCounts(
                input=usage.input_tokens,
                output=usage.output_tokens,
                cache_read=usage.cache_read_tokens,
                cache_write=usage.cache_creation_tokens,
            ),
            rates,
        )
        total += cost
        parts = {
            "input": token_cost(usage.input_tokens, rates.input),
            "cache_read": token_cost(usage.cache_read_tokens, rates.cache_read),
            "cache_write": token_cost(usage.cache_creation_tokens, rates.cache_write),
            "output": token_cost(usage.output_tokens, rates.output),
        }
        for key, usd in parts.items():
            by_type[key] += usd
        # The step carrying the usage always belongs to its turn, so every
        # priced turn has a member and the class costs add up (TER-ANL-043).
        members = [step.index]
        priced.append((cost, parts["output"], members))
        model = usage.model or ""
        n, usd = by_model.get(model, (0, 0.0))
        by_model[model] = (n + 1, usd + cost)
        caching = _caching(usage)
        turn_rates[step.index] = (
            (rates.cache_write if caching else rates.input) / TOKENS_PER_RATE_UNIT,
            (rates.cache_read if caching else rates.input) / TOKENS_PER_RATE_UNIT,
        )

    # first_after[i]: ingest price of the first priced turn after step i;
    # carry_after[i]: summed carry prices of the priced turns after that one.
    first_after = [0.0] * len(steps)
    carry_after = [0.0] * len(steps)
    first = 0.0
    carry = 0.0
    nearest: tuple[float, float] | None = None
    for step in reversed(steps):
        first_after[step.index] = first
        carry_after[step.index] = carry
        rate = turn_rates.get(step.index)
        if rate is not None:
            if nearest is not None:
                carry += nearest[1]
            nearest = rate
            first = rate[0]
    index_of = {s.event_id: s.index for s in steps}

    def held(items: tuple[InventoryItem, ...]) -> float:
        return sum(
            item.tokens * (first_after[i] + carry_after[i])
            for item in items
            for i in (index_of[item.event_id],)
        )

    whole: tuple[tuple[str, float], ...] = ()
    output: tuple[tuple[str, float], ...] = ()
    if classifications is not None:
        whole, output = _split_by_class(steps, classifications, priced)

    reasons: list[str] = []
    if no_cache or turns == 0:
        reasons.append(EstimateReason.NO_CACHE_FIELDS)
    if turns == 0:
        reasons.append(EstimateReason.NO_USAGE)
    if on is None:
        reasons.append(EstimateReason.NO_SESSION_DATE)
    if unpriced:
        reasons.append(EstimateReason.UNPRICED_TURNS)
    return SessionCost(
        price_book=prices.name,
        priced_on=on,
        turns=turns,
        usd=total,
        by_model=tuple(
            ModelCost(m, n, usd) for m, (n, usd) in sorted(by_model.items())
        ),
        unpriced_turns=unpriced,
        unpriced_models=tuple(sorted(unpriced_models)),
        unused_context_usd=held(inventory.unused),
        reread_context_usd=held(inventory.reread),
        estimate_reasons=tuple(reasons),
        by_token_type=tuple(by_type.items()) if turns else (),
        by_activity_class=whole,
        output_by_activity_class=output,
    )


def _class_shares(c: Classification) -> dict[str, float]:
    """How one event's cost divides among ``COST_CLASSES``.

    The confident waste share is avoidable, the uncertain share uncertain
    (ADR 0006: never avoidable), and the rest the event's base class: an
    event wholly claimed by waste is necessary work done needlessly, so a
    remainder of it (none, at a full claim) stays necessary.
    """
    shares = dict.fromkeys(COST_CLASSES, 0.0)
    avoid = c.avoidable_share
    unsure = c.uncertain_share
    shares["avoidable"] += avoid
    shares["uncertain"] += unsure
    cls = c.activity_class
    base = (
        "necessary_non_value_adding"
        if cls is None or cls.value == "avoidable"
        else cls.value
    )
    shares[base] += max(0.0, 1.0 - avoid - unsure)
    return shares


def _split_by_class(
    steps: tuple[Step, ...],
    classifications: tuple[Classification, ...],
    priced: list[tuple[float, float, list[int]]],
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    """Whole-turn and output-only cost by activity class (TER-ANL-042)."""
    whole = dict.fromkeys(COST_CLASSES, 0.0)
    output = dict.fromkeys(COST_CLASSES, 0.0)
    for cost, out_cost, members in priced:
        weights = [float(steps[i].tokens) for i in members]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(members)
            total = float(len(members))
        for i, weight in zip(members, weights, strict=True):
            part = weight / total
            for key, share in _class_shares(classifications[i]).items():
                whole[key] += cost * part * share
                output[key] += out_cost * part * share
    return tuple(whole.items()), tuple(output.items())


def _rates(prices: Prices, model: str | None, on: date | None) -> Rates | None:
    if not model:
        return None
    try:
        return prices.rate(model, on)
    except UnknownModelError:
        return None
