"""Model prices and the cost arithmetic built on them.

Prices are data, not code: they change on a vendor's schedule, differ by
model and by date, and must be traceable to a source. The domain defines what
a price is (:class:`Rates`), how dated prices resolve (:class:`PriceSchedule`)
and how token counts become dollars. Where the prices come from is a
``PriceBook`` port (``ter.ports``); the shipped adapter reads a dated JSON
file.

All rates are US dollars per million tokens. Cost arithmetic uses the
operation order TER 3 used (``tokens * rate / 1_000_000``) so dollar figures
are bit-for-bit unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

__all__ = [
    "PriceBookError",
    "PriceEntry",
    "PriceSchedule",
    "Rates",
    "TokenCounts",
    "UnknownModelError",
    "token_cost",
    "usage_cost",
]

TOKENS_PER_RATE_UNIT = 1_000_000


class PriceBookError(ValueError):
    """A price file that cannot be read as a price book."""


class UnknownModelError(KeyError):
    """No price is known for the model (or none was in effect on the date)."""


@dataclass(frozen=True, slots=True)
class Rates:
    """Per-million-token prices in USD for one model."""

    input: float
    output: float
    cache_read: float
    cache_write: float

    def __post_init__(self) -> None:
        for name in ("input", "output", "cache_read", "cache_write"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Rate {name} must be finite and >= 0, got {value}")

    @property
    def thinking(self) -> float:
        """Extended-thinking tokens are billed as output."""
        return self.output


@dataclass(frozen=True, slots=True)
class TokenCounts:
    """Billed token counts for a session or turn."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0


def token_cost(tokens: int, rate_per_mtok: float) -> float:
    """Dollar cost of ``tokens`` at ``rate_per_mtok`` dollars per million."""
    return tokens * rate_per_mtok / TOKENS_PER_RATE_UNIT


def usage_cost(counts: TokenCounts, rates: Rates) -> float:
    """Dollar cost of a usage record: input, output, cache reads and writes."""
    return (
        token_cost(counts.input, rates.input)
        + token_cost(counts.output, rates.output)
        + token_cost(counts.cache_read, rates.cache_read)
        + token_cost(counts.cache_write, rates.cache_write)
    )


@dataclass(frozen=True, slots=True)
class PriceEntry:
    """A model's rates from a given date, with where the numbers came from."""

    model: str
    effective_from: date
    rates: Rates
    source: str = ""
    aliases: tuple[str, ...] = ()
    #: The first day the entry is no longer in effect, when its source ends
    #: it with no successor; None while it stays in effect until the next
    #: entry. On and after it the model has no price.
    effective_to: date | None = None


class PriceSchedule:
    """Resolves a model name (or alias) and a date to the rates then in effect.

    Each model may have several dated entries; the latest one whose
    ``effective_from`` is on or before the requested date wins. With no date,
    the latest entry wins.

    Aliases belong to dated entries, not to models. An alias names the model
    whose entry in effect on the requested date carries it; when several do
    (a family alias such as ``sonnet`` moving to a successor model), the most
    recently effective entry wins. So an alias can move to a new model on its
    effective date, and an alias added or dropped in a later entry only
    resolves while an entry carrying it is in effect.
    """

    def __init__(self, entries: Iterable[PriceEntry]) -> None:
        self._entries: dict[str, list[PriceEntry]] = {}
        for entry in entries:
            self._entries.setdefault(entry.model, []).append(entry)
        claims: dict[str, dict[date, str]] = {}
        for model, dated in self._entries.items():
            dated.sort(key=lambda e: e.effective_from)
            days = [e.effective_from for e in dated]
            if len(set(days)) != len(days):
                raise ValueError(f"Duplicate effective_from dates for {model}")
            for entry in dated:
                for alias in entry.aliases:
                    if alias in self._entries:
                        raise ValueError(f"Alias {alias!r} is also a model name")
                    owner = claims.setdefault(alias, {}).setdefault(
                        entry.effective_from, model
                    )
                    if owner != model:
                        raise ValueError(
                            f"Alias {alias!r} names both {owner} and {model} "
                            f"from {entry.effective_from}"
                        )
        self._alias_models: dict[str, tuple[str, ...]] = {
            alias: tuple(sorted(set(by_day.values())))
            for alias, by_day in claims.items()
        }
        self._latest = max(
            (e.effective_from for dated in self._entries.values() for e in dated),
            default=date.min,
        )

    def models(self) -> tuple[str, ...]:
        """Canonical model names with at least one price, sorted."""
        return tuple(sorted(self._entries))

    def _in_effect(self, model: str, at: date) -> PriceEntry | None:
        in_effect = [e for e in self._entries[model] if e.effective_from <= at]
        return in_effect[-1] if in_effect else None

    def resolve(self, model: str, at: date | None = None) -> str:
        """Return the canonical model name for a name or alias on ``at``.

        With no date, an alias resolves as of the book's latest entry.
        """
        if model in self._entries:
            return model
        if model not in self._alias_models:
            raise UnknownModelError(model)
        when = self._latest if at is None else at
        carriers = [
            entry
            for candidate in self._alias_models[model]
            if (entry := self._in_effect(candidate, when)) is not None
            and model in entry.aliases
        ]
        if not carriers:
            raise UnknownModelError(f"{model} names no priced model on {when}")
        return max(carriers, key=lambda e: e.effective_from).model

    def entry(self, model: str, at: date | None = None) -> PriceEntry:
        """Return the price entry in effect for ``model`` on ``at``."""
        canonical = self.resolve(model, at)
        if at is None:
            return self._entries[canonical][-1]
        found = self._in_effect(canonical, at)
        if found is None:
            raise UnknownModelError(f"{model} has no price in effect on {at}")
        if found.effective_to is not None and at >= found.effective_to:
            raise UnknownModelError(
                f"{model} has no price in effect on {at} "
                f"(its price ended on {found.effective_to})"
            )
        return found

    def rate(self, model: str, at: date | None = None) -> Rates:
        """Return the rates in effect for ``model`` on ``at`` (latest if None)."""
        return self.entry(model, at).rates
