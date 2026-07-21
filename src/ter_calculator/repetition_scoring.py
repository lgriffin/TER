"""Blended repetition scoring for reasoning, generation, and tool actions.

The scorer combines semantic similarity with lexical, entity, action, and
parameter-novelty evidence.  It is deliberately conservative: exact structured
tool duplicates score 1.0, while changed tool parameters reduce the score even
when rendered text embeddings are nearly identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .tool_fingerprints import ToolCallComparison

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*|\d+")
_ENTITY_RE = re.compile(
    r"(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)|(?:\b[A-Za-z_][A-Za-z0-9_]{2,}\b)|(?:\b\d{2,}\b)"
)
_ACTION_WORDS = {
    "add",
    "analyze",
    "build",
    "check",
    "compare",
    "create",
    "delete",
    "edit",
    "find",
    "fix",
    "generate",
    "inspect",
    "list",
    "load",
    "move",
    "open",
    "read",
    "remove",
    "replace",
    "report",
    "run",
    "search",
    "test",
    "update",
    "validate",
    "verify",
    "write",
}


@dataclass(frozen=True)
class RepetitionScore:
    """Explainable repetition evidence for a pair of actions or spans."""

    score: float
    semantic: float
    lexical: float
    entity: float
    action: float
    parameter_novelty: float
    exact_duplicate: bool = False


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _WORD_RE.findall(text)}


def _entities(text: str) -> set[str]:
    return {token.casefold() for token in _ENTITY_RE.findall(text)}


def _jaccard(first: Iterable[str], second: Iterable[str]) -> float:
    left, right = set(first), set(second)
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def lexical_similarity(first: str, second: str) -> float:
    """Return case-insensitive token Jaccard similarity."""
    return _jaccard(_tokens(first), _tokens(second))


def entity_similarity(first: str, second: str) -> float:
    """Compare paths, identifiers, and numeric references."""
    return _jaccard(_entities(first), _entities(second))


def action_similarity(first: str, second: str) -> float:
    """Compare coarse action verbs used by two spans."""
    left = _tokens(first) & _ACTION_WORDS
    right = _tokens(second) & _ACTION_WORDS
    if not left and not right:
        return 0.0
    return _jaccard(left, right)


def infer_parameter_novelty(first: str, second: str) -> float:
    """Estimate novelty from entities introduced or removed in the second span."""
    left, right = _entities(first), _entities(second)
    if not left and not right:
        return 0.0
    return 1.0 - _jaccard(left, right)


def score_text_repetition(first: str, second: str, semantic: float) -> RepetitionScore:
    """Blend text signals while preserving semantic similarity as the anchor."""
    lexical = lexical_similarity(first, second)
    entity = entity_similarity(first, second)
    action = action_similarity(first, second)
    novelty = infer_parameter_novelty(first, second)

    corroboration = 0.45 * lexical + 0.30 * entity + 0.25 * action
    # Semantic similarity remains dominant, but new entities/specifics lower the
    # result.  Identical text still reaches 1.0.
    blended = 0.82 * semantic + 0.18 * corroboration
    score = blended * (1.0 - 0.30 * novelty)
    return RepetitionScore(
        score=max(0.0, min(1.0, score)),
        semantic=semantic,
        lexical=lexical,
        entity=entity,
        action=action,
        parameter_novelty=novelty,
        exact_duplicate=first.strip() == second.strip(),
    )


def score_tool_repetition(
    first_text: str,
    second_text: str,
    semantic: float,
    comparison: ToolCallComparison,
) -> RepetitionScore:
    """Blend structured and textual evidence for two tool calls."""
    lexical = lexical_similarity(first_text, second_text)
    entity = entity_similarity(first_text, second_text)
    action = 1.0 if comparison.same_tool else 0.0
    novelty = comparison.parameter_novelty

    if comparison.exact_duplicate:
        return RepetitionScore(1.0, semantic, lexical, entity, action, 0.0, True)

    base = 0.45 * semantic + 0.15 * lexical + 0.10 * entity + 0.30 * action
    # Structured parameter novelty is decisive for tools. A changed range/query
    # must not be classified as duplicate merely because rendered JSON is alike.
    score = base * (1.0 - 0.85 * novelty)
    return RepetitionScore(
        score=max(0.0, min(1.0, score)),
        semantic=semantic,
        lexical=lexical,
        entity=entity,
        action=action,
        parameter_novelty=novelty,
        exact_duplicate=False,
    )
