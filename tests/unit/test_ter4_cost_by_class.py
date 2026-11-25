"""Session cost split by activity class (TER-ANL-042, TER-ANL-043, ADR 0007).

A model turn opens at the event carrying its usage and runs to the next one.
Its whole cost is shared among the turn's generated events by generated
tokens, and each part follows the event's classification.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from ter.adapters.driven.in_memory import InMemoryPriceBook
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import Event, PriceEntry, Rates, TokenUsage
from ter.domain.costing import COST_CLASSES, price_session
from ter.domain.lean import build_a3
from ter.domain.lean.analysis import Classification, LeanAnalysis, explain
from ter.domain.lean.inventory import context_inventory
from ter.domain.lean.model import ActivityClass

from ter4_lean_builder import Script

#: One million input tokens cost $1, one million output tokens $10.
RATES = Rates(input=1.0, output=10.0, cache_read=0.1, cache_write=1.25)
BOOK = InMemoryPriceBook([PriceEntry("m", date(2020, 1, 1), RATES)])


def _usage(input: int = 1_000_000, output: int = 0, model: str = "m") -> TokenUsage:
    return TokenUsage(input_tokens=input, output_tokens=output, model=model)


def _with_usage(events: list[Event], usage: dict[int, TokenUsage]) -> list[Event]:
    return [replace(e, usage=usage.get(i, e.usage)) for i, e in enumerate(events)]


def _session() -> tuple[LeanAnalysis, list[Event]]:
    """Turn 1: reasoning and an edit; turn 2: the final response."""
    s = Script()
    s.prompt("rename the flag in cli.py")
    s.think("I will edit the flag name")  # 1: opens turn 1
    s.edit("src/cli.py")  # 2: request, same turn; 3: its result
    s.say("Renamed the flag.")  # 4: opens turn 2
    events = _with_usage(s.events, {1: _usage(output=100_000), 4: _usage()})
    return explain(events, RegexTokenizer()), events


def _price(
    analysis: LeanAnalysis, classes: tuple[Classification, ...] | None = None
) -> dict[str, float]:
    inventory = context_inventory(analysis.steps, analysis.findings)
    cost = price_session(
        analysis.steps,
        inventory,
        BOOK,
        analysis.classifications if classes is None else classes,
    )
    return dict(cost.by_activity_class)


def _set(
    analysis: LeanAnalysis, index: int, **changes: object
) -> tuple[Classification, ...]:
    out = list(analysis.classifications)
    out[index] = replace(out[index], **changes)  # type: ignore[arg-type]
    return tuple(out)


@pytest.mark.req("TER-ANL-042")
class TestWholeTurnSplit:
    def test_turn_cost_is_shared_by_generated_tokens(self) -> None:
        analysis, _ = _session()
        steps = analysis.steps
        think, edit = steps[1].tokens, steps[2].tokens
        assert think > 0 and edit > 0
        by_class = _price(analysis)
        turn1 = 1.0 + 1.0  # $1 input + $1 output
        assert by_class["value_adding"] == pytest.approx(
            turn1 * edit / (think + edit) + 1.0  # the final response's turn
        )
        assert by_class["necessary_non_value_adding"] == pytest.approx(
            turn1 * think / (think + edit)
        )

    def test_output_split_prices_only_output_tokens(self) -> None:
        analysis, _ = _session()
        inventory = context_inventory(analysis.steps, analysis.findings)
        cost = price_session(analysis.steps, inventory, BOOK, analysis.classifications)
        output = dict(cost.output_by_activity_class)
        assert sum(output.values()) == pytest.approx(1.0)
        assert dict(cost.by_token_type) == pytest.approx(
            {"input": 2.0, "cache_read": 0.0, "cache_write": 0.0, "output": 1.0}
        )

    def test_avoidable_share_goes_to_avoidable_and_rest_to_base_class(self) -> None:
        analysis, _ = _session()
        classes = _set(analysis, 4, avoidable_share=0.25)
        by_class = _price(analysis, classes)
        base = _price(analysis)
        assert by_class["avoidable"] == pytest.approx(0.25)
        assert by_class["value_adding"] == pytest.approx(base["value_adding"] - 0.25)

    def test_fully_avoidable_event_leaves_nothing_in_its_base_class(self) -> None:
        analysis, _ = _session()
        classes = _set(
            analysis, 4, activity_class=ActivityClass.AVOIDABLE, avoidable_share=1.0
        )
        by_class = _price(analysis, classes)
        assert by_class["avoidable"] == pytest.approx(1.0)
        assert by_class["necessary_non_value_adding"] == pytest.approx(
            _price(analysis)["necessary_non_value_adding"]
        )

    def test_boundary_turn_with_no_generated_tokens_is_shared_equally(self) -> None:
        analysis, _ = _session()
        steps = tuple(replace(s, tokens=0) for s in analysis.steps)
        inventory = context_inventory(steps, analysis.findings)
        cost = price_session(steps, inventory, BOOK, analysis.classifications)
        by_class = dict(cost.by_activity_class)
        assert by_class["value_adding"] == pytest.approx(1.0 + 1.0)
        assert by_class["necessary_non_value_adding"] == pytest.approx(1.0)

    def test_negative_events_before_the_first_usage_carry_no_cost(self) -> None:
        s = Script()
        s.prompt("look")
        s.think("first thought, before any reported usage")
        s.say("done")
        events = _with_usage(s.events, {2: _usage()})
        analysis = explain(events, RegexTokenizer())
        by_class = _price(analysis)
        assert by_class["value_adding"] == pytest.approx(1.0)
        assert by_class["necessary_non_value_adding"] == 0.0

    def test_negative_unpriced_turn_is_in_no_class(self) -> None:
        analysis, events = _session()
        events = _with_usage(events, {1: _usage(model="unknown")})
        analysis = explain(events, RegexTokenizer())
        by_class = _price(analysis)
        assert sum(by_class.values()) == pytest.approx(1.0)  # turn 2 only

    def test_without_classifications_the_cost_is_not_split(self) -> None:
        analysis, _ = _session()
        inventory = context_inventory(analysis.steps, analysis.findings)
        cost = price_session(analysis.steps, inventory, BOOK)
        assert cost.by_activity_class == ()
        assert "by_activity_class" not in cost.as_dict()
        assert cost.class_usd("avoidable") == 0.0

    def test_classifications_must_match_the_steps(self) -> None:
        analysis, _ = _session()
        inventory = context_inventory(analysis.steps, analysis.findings)
        with pytest.raises(ValueError, match="classifications"):
            price_session(
                analysis.steps, inventory, BOOK, analysis.classifications[:-1]
            )

    def test_the_a3_reports_cost_by_class(self) -> None:
        analysis, _ = _session()
        payload = build_a3(analysis, prices=BOOK).as_dict()["cost"]
        assert isinstance(payload, dict)
        assert list(payload["by_activity_class"]) == list(COST_CLASSES)
        assert list(payload["by_token_type"]) == [
            "input",
            "cache_read",
            "cache_write",
            "output",
        ]


@pytest.mark.req("TER-ANL-043")
class TestClassCostsAddUp:
    def test_uncertain_share_is_never_avoidable(self) -> None:
        analysis, _ = _session()
        classes = _set(analysis, 1, uncertain_share=1.0)
        by_class = _price(analysis, classes)
        assert by_class["avoidable"] == 0.0
        assert by_class["uncertain"] > 0.0
        assert by_class["necessary_non_value_adding"] == 0.0

    def test_split_sums_to_the_session_cost(self) -> None:
        analysis, _ = _session()
        classes = _set(analysis, 2, avoidable_share=0.5, uncertain_share=0.25)
        inventory = context_inventory(analysis.steps, analysis.findings)
        cost = price_session(analysis.steps, inventory, BOOK, classes)
        assert cost.usd == pytest.approx(3.0)
        assert sum(v for _, v in cost.by_activity_class) == pytest.approx(cost.usd)
        assert cost.class_usd("uncertain") > 0
