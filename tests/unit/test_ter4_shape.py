"""Observed consumption shape (TER-SHP-001, TER-SHP-002, ADR 0007)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import Event, TokenUsage
from ter.domain.lean import build_a3
from ter.domain.lean.analysis import explain
from ter.domain.lean.shape import ObservedShape, observed_shape

from tests.golden.corpus import CORPUS

from ter4_lean_builder import Script

USAGE = TokenUsage(input_tokens=10, output_tokens=5, model="m")


def _shape(events: list[Event], turns: set[int], **kw: bool) -> ObservedShape:
    events = [
        replace(e, usage=USAGE) if i in turns else e for i, e in enumerate(events)
    ]
    return observed_shape(explain(events, RegexTokenizer()).steps, **kw)


@pytest.mark.req("TER-SHP-001")
class TestObservedShape:
    def test_one_turn_per_intent_is_t_n(self) -> None:
        s = Script()
        s.prompt("explain the cli")
        s.say("It parses arguments.")
        shape = _shape(s.events, {1})
        assert (shape.intents, shape.model_turns, shape.agent_depth) == (1, 1, 0)
        assert shape.turns_per_intent == 1.0
        assert shape.observed_class == "T(n)"

    def test_several_turns_per_intent_is_t_n_k(self) -> None:
        s = Script()
        s.prompt("fix the flag")
        s.think("read first")  # 1
        s.read("src/cli.py", "def main(): pass")  # 2, 3
        s.say("Fixed.")  # 4
        s.prompt("and the docs")  # 5
        s.say("Done.")  # 6
        shape = _shape(s.events, {1, 4, 6})
        assert shape.intents == 2
        assert shape.turns_per_intent == pytest.approx(1.5)
        assert shape.observed_class == "T(n·k)"

    def test_boundary_two_intents_two_turns_is_still_t_n(self) -> None:
        s = Script()
        s.prompt("one")
        s.say("a")
        s.prompt("two")
        s.say("b")
        assert _shape(s.events, {1, 3}).observed_class == "T(n)"

    def test_a_handoff_is_agent_depth_one(self) -> None:
        s = Script()
        s.prompt("research the api")
        s.task("look up the api", "find the endpoint")  # 1, 2
        s.say("Found it.")  # 3
        shape = _shape(s.events, {1, 3})
        assert (shape.handoffs, shape.agent_depth) == (1, 1)
        assert shape.observed_class == "T(n·k·a)"

    def test_negative_no_model_turn_has_no_class(self) -> None:
        s = Script()
        s.prompt("hello")
        s.say("hi")
        shape = _shape(s.events, set())
        assert shape.model_turns == 0 and shape.observed_class is None

    def test_boundary_no_prompt_has_no_turns_per_intent(self) -> None:
        s = Script()
        s.say("unprompted")
        shape = _shape(s.events, {0})
        assert shape.turns_per_intent is None
        assert shape.observed_class == "T(n·k)"
        assert shape.as_dict()["turns_per_intent"] is None

    @pytest.mark.parametrize("name", sorted(CORPUS))
    def test_the_a3_reports_the_shape_of_every_corpus_session(self, name: str) -> None:
        trace = ClaudeCodeJsonlSource().read(CORPUS[name])
        analysis = explain(list(trace.events), RegexTokenizer())
        payload = json.loads(json.dumps(build_a3(analysis).as_dict()))["shape"]
        assert payload == observed_shape(analysis.steps).as_dict()
        turns = sum(1 for e in trace.events if e.usage is not None)
        assert payload["model_turns"] == turns
        expected = "T(n·k·a)" if name == "handoff_fetch" else payload["observed_class"]
        assert payload["observed_class"] == expected


@pytest.mark.req("TER-SHP-002")
class TestDepthLowerBound:
    def _handoff(self) -> list[Event]:
        s = Script()
        s.prompt("research the api")
        s.task("look up the api", "find the endpoint")
        s.say("Found it.")
        return s.events

    def test_depth_is_a_lower_bound_when_nesting_is_not_recorded(self) -> None:
        assert _shape(self._handoff(), {1, 3}).depth_lower_bound is True

    def test_negative_a_source_that_records_nesting_gives_exact_depth(self) -> None:
        shape = _shape(self._handoff(), {1, 3}, records_nesting=True)
        assert shape.depth_lower_bound is False

    def test_negative_no_handoff_is_exact_depth_zero(self) -> None:
        s = Script()
        s.prompt("one")
        s.say("a")
        shape = _shape(s.events, {1})
        assert (shape.agent_depth, shape.depth_lower_bound) == (0, False)
