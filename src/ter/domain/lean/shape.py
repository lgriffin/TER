"""A session's observed consumption shape (TER-SHP-001, TER-SHP-002, ADR 0007).

Big-T notation classifies how an agentic workload consumes tokens: over
requests ``n``, model calls per request ``k`` and agent depth ``a``, from
T(n) (one call per request) through T(n·k) (several calls per request) to
T(n·k·a) (agents calling agents). It asks teams to find the class by reading
the architecture. TER counts it from the events of what ran:

- ``n``: intents, the developer's prompts (``intent.stated``). Whether a
  later prompt corrects the task or starts a new one is not decided here;
  every prompt counts.
- ``k``: model turns per intent, a model turn being an event that carries
  the provider's usage (the turns a session is priced from, TER-ANL-040).
- ``a``: agent depth, 1 when the session handed work to a subagent
  (an ``agent.handoff`` request), else 0. A source that does not record
  subagent nesting cannot show depth below the first handoff, so there
  depth is a lower bound and says so (TER-SHP-002).

The class is the observed one: ``T(n·k·a)`` with any handoff, ``T(n·k)``
when an intent took more than one model turn, else ``T(n)``. With no model
turns there is nothing to classify and the class is None.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..events import EventKind, ToolKind
from .model import Step

__all__ = ["ObservedShape", "observed_shape"]


@dataclass(frozen=True)
class ObservedShape:
    """Intents, model turns per intent and agent depth, counted from events."""

    intents: int
    model_turns: int
    handoffs: int
    agent_depth: int
    #: True when depth may be deeper than shown: the session handed work to
    #: a subagent and its source does not record nesting (TER-SHP-002).
    depth_lower_bound: bool

    @property
    def turns_per_intent(self) -> float | None:
        """``k``: model turns per intent; None for a session with no prompt."""
        if self.intents == 0:
            return None
        return self.model_turns / self.intents

    @property
    def observed_class(self) -> str | None:
        """The Big-T class the counts imply, or None with no model turn."""
        if self.model_turns == 0:
            return None
        if self.agent_depth > 0:
            return "T(n·k·a)"
        k = self.turns_per_intent
        if k is None or k > 1:
            return "T(n·k)"
        return "T(n)"

    def as_dict(self) -> dict[str, object]:
        k = self.turns_per_intent
        return {
            "intents": self.intents,
            "model_turns": self.model_turns,
            "turns_per_intent": None if k is None else round(k, 4),
            "handoffs": self.handoffs,
            "agent_depth": self.agent_depth,
            "depth_lower_bound": self.depth_lower_bound,
            "observed_class": self.observed_class,
        }


def observed_shape(
    steps: Iterable[Step], *, records_nesting: bool = False
) -> ObservedShape:
    """Count a session's shape from its steps (TER-SHP-001).

    ``records_nesting`` says whether the source records subagent nesting;
    none of the shipped sources does, so a handoff makes depth a lower bound.
    """
    intents = turns = handoffs = 0
    for step in steps:
        if step.kind is EventKind.PROMPT:
            intents += 1
        if step.usage is not None:
            turns += 1
        if step.is_request and step.tool_kind is ToolKind.AGENT_HANDOFF:
            handoffs += 1
    depth = 1 if handoffs else 0
    return ObservedShape(
        intents=intents,
        model_turns=turns,
        handoffs=handoffs,
        agent_depth=depth,
        depth_lower_bound=bool(handoffs) and not records_nesting,
    )
