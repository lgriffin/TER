"""Context inventory: retrieved context nothing used, and context read twice.

Lean treats context like stock on a shelf: every token a tool returns stays
in the agent's context and is carried, at a price, through every later model
turn. Two kinds of that stock are measurable from the event stream alone
(TER-DET-004, points 34 to 36):

* **unused** context: a file read that no later event names or edits. These
  are the reads the ``unused_context`` detector flags (counted as waste
  since the first judged sample found 10 of 10 waste, ADR 0006).
* **re-read** context: every read of a path after its first read in the
  session, the tokens behind :attr:`StreamReport.repeated_read_count`. A
  re-read after an edit of that file brings in changed content and is marked
  ``changed``; the rest re-entered the context unchanged.

Tokens are the tokenizer's count of the tool output (``Step.context_tokens``).
``carried_turns`` is how many model turns (events carrying provider usage)
came after the context entered: the turns that paid for carrying it. Pricing
those turns is :mod:`ter.domain.costing`'s job.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..events import EventId, ToolKind
from .model import Finding, Step

__all__ = [
    "ContextInventory",
    "InventoryItem",
    "context_inventory",
    "turns_after",
]

#: The detector whose findings name unused retrieved context.
UNUSED_CONTEXT_DETECTOR = "unused_context"


@dataclass(frozen=True)
class InventoryItem:
    """One tool result held in the context as inventory."""

    event_id: EventId
    request_id: EventId | None
    path: str
    tokens: int
    carried_turns: int
    #: The ``unused_context`` finding that names it (unused items only).
    finding: str | None = None
    #: True when the finding is below the confidence threshold.
    uncertain: bool = False
    #: Re-reads only: the file was edited since it was last read.
    changed: bool = False

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "event_id": self.event_id,
            "request_id": self.request_id,
            "path": self.path,
            "tokens": self.tokens,
            "carried_turns": self.carried_turns,
        }
        if self.finding is not None:
            out["finding"] = self.finding
            out["uncertain"] = self.uncertain
        else:
            out["changed"] = self.changed
        return out


@dataclass(frozen=True)
class ContextInventory:
    """The session's context inventory, all traceable to event ids."""

    retrieved_tokens: int
    turns: int
    unused: tuple[InventoryItem, ...]
    reread: tuple[InventoryItem, ...]

    @property
    def unused_tokens(self) -> int:
        """Tokens of retrieved context no later event used (TER-DET-004)."""
        return sum(i.tokens for i in self.unused)

    @property
    def reread_tokens(self) -> int:
        """Tokens of context read more than once: every read after the first."""
        return sum(i.tokens for i in self.reread)

    @property
    def unchanged_reread_tokens(self) -> int:
        """Re-read tokens of files not edited since they were last read."""
        return sum(i.tokens for i in self.reread if not i.changed)

    @property
    def unused_carried_tokens(self) -> int:
        """Unused tokens times the model turns that carried them."""
        return sum(i.tokens * i.carried_turns for i in self.unused)

    @property
    def reread_carried_tokens(self) -> int:
        """Re-read tokens times the model turns that carried them."""
        return sum(i.tokens * i.carried_turns for i in self.reread)

    def as_dict(self) -> dict[str, object]:
        return {
            "retrieved_tokens": self.retrieved_tokens,
            "model_turns": self.turns,
            "unused_tokens": self.unused_tokens,
            "unused_carried_tokens": self.unused_carried_tokens,
            "reread_tokens": self.reread_tokens,
            "unchanged_reread_tokens": self.unchanged_reread_tokens,
            "reread_carried_tokens": self.reread_carried_tokens,
            "unused": [i.as_dict() for i in self.unused],
            "reread": [i.as_dict() for i in self.reread],
        }


def turns_after(steps: tuple[Step, ...]) -> list[int]:
    """For each step index, the number of model turns strictly after it.

    A model turn is a step carrying provider usage (adapters attach a turn's
    usage to its first event, once).
    """
    after = [0] * len(steps)
    seen = 0
    for step in reversed(steps):
        after[step.index] = seen
        if step.usage is not None:
            seen += 1
    return after


def _is_read_result(step: Step) -> bool:
    return (
        step.is_completion and step.tool_kind is ToolKind.FS_READ and bool(step.paths)
    )


def context_inventory(
    steps: tuple[Step, ...], findings: Iterable[Finding]
) -> ContextInventory:
    """Measure the inventory of ``steps`` given the analysis's ``findings``.

    Linear in the session: one pass for turn counts, one over the reads.
    """
    after = turns_after(steps)
    by_id = {s.event_id: s for s in steps}

    unused: list[InventoryItem] = []
    for finding in findings:
        if finding.detector != UNUSED_CONTEXT_DETECTOR:
            continue
        result = next(
            (by_id[e] for e in finding.waste_events if by_id[e].is_completion), None
        )
        if result is None:
            continue
        unused.append(
            InventoryItem(
                event_id=result.event_id,
                request_id=_request_id(steps, result),
                path=result.paths[0],
                tokens=result.context_tokens,
                carried_turns=after[result.index],
                finding=finding.id,
                uncertain=finding.uncertain,
            )
        )
    unused.sort(key=lambda i: by_id[i.event_id].index)

    reread: list[InventoryItem] = []
    last_read: dict[str, int] = {}
    last_edit: dict[str, int] = {}
    retrieved = 0
    for step in steps:
        if step.is_edit:
            for path in step.paths:
                last_edit[path] = step.index
        if not _is_read_result(step):
            continue
        retrieved += step.context_tokens
        path = step.paths[0]
        previous = last_read.get(path)
        last_read[path] = step.index
        if previous is None:
            continue
        reread.append(
            InventoryItem(
                event_id=step.event_id,
                request_id=_request_id(steps, step),
                path=path,
                tokens=step.context_tokens,
                carried_turns=after[step.index],
                changed=last_edit.get(path, -1) > previous,
            )
        )
    turns = sum(1 for s in steps if s.usage is not None)
    return ContextInventory(retrieved, turns, tuple(unused), tuple(reread))


def _request_id(steps: tuple[Step, ...], result: Step) -> EventId | None:
    if result.request_index is None:
        return None
    return steps[result.request_index].event_id
