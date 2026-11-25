"""Golden snapshots of each corpus session priced, with cost by activity class.

The synthetic sessions name a demo model with no entry in the shipped price
book, so TER leaves their turns unpriced. Here every demo model is priced at
the book's ``sonnet`` rates through an in-memory book, which reproduces each
TER 3 snapshot's ``cost_usd``, so the split by activity class (TER-ANL-042,
ADR 0007) can be frozen and its sum checked (TER-ANL-043). The figures
match the prototype behind ADR 0007 ("Upstream of the Bill", 25 Nov 2026).
A diff is a behaviour change: regenerate with ``TER_UPDATE_GOLDEN=1``.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.in_memory import InMemoryPriceBook
from ter.adapters.driven.pricing import default_price_book
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.application import ExplainSession
from ter.domain import PriceEntry
from ter.domain.costing import COST_CLASSES, SessionCost

from .conftest import CORPUS, SNAPSHOT_DIR, assert_matches_snapshot


def _demo_models() -> set[str]:
    models: set[str] = set()
    for name in CORPUS:
        trace = ClaudeCodeJsonlSource().read(CORPUS[name])
        models |= {e.usage.model for e in trace.events if e.usage and e.usage.model}
    return models


def _book() -> InMemoryPriceBook:
    sonnet = default_price_book().rate("sonnet")
    return InMemoryPriceBook(
        PriceEntry(model, date(2000, 1, 1), sonnet, source="test: sonnet rates")
        for model in sorted(_demo_models())
    )


def _cost(name: str) -> SessionCost:
    use_case = ExplainSession(ClaudeCodeJsonlSource(), RegexTokenizer(), prices=_book())
    cost = use_case(CORPUS[name]).a3.cost
    assert cost is not None
    return cost


@pytest.mark.req("TER-ANL-042")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_session_cost_by_class_matches_golden_snapshot(name: str) -> None:
    assert_matches_snapshot(
        f"cost/{name}.cost", json.loads(json.dumps(_cost(name).as_dict()))
    )


@pytest.mark.req("TER-ANL-043")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_class_costs_sum_to_the_session_cost(name: str) -> None:
    cost = _cost(name)
    assert [k for k, _ in cost.by_activity_class] == list(COST_CLASSES)
    assert sum(v for _, v in cost.by_activity_class) == pytest.approx(
        cost.usd, abs=1e-12
    )
    assert sum(v for _, v in cost.by_token_type) == pytest.approx(cost.usd, abs=1e-12)
    output = dict(cost.by_token_type)["output"]
    assert sum(v for _, v in cost.output_by_activity_class) == pytest.approx(
        output, abs=1e-12
    )


@pytest.mark.req("TER-ANL-042")
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_session_cost_reproduces_the_ter3_snapshot(name: str) -> None:
    frozen = json.loads(
        (SNAPSHOT_DIR / f"{name}.default.json").read_text(encoding="utf-8")
    )
    assert round(_cost(name).usd, 4) == pytest.approx(frozen["economics"]["cost_usd"])
