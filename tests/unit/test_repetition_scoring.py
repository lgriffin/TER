"""Tests for blended repetition scoring."""

from __future__ import annotations

import numpy as np

import ter_calculator.classifier as classifier
from ter_calculator.models import IntentVector, SpanLabel, SpanPhase, TokenSpan
from ter_calculator.repetition_scoring import (
    action_similarity,
    entity_similarity,
    lexical_similarity,
    score_text_repetition,
    score_tool_repetition,
)
from ter_calculator.tool_fingerprints import build_tool_fingerprint, compare_tool_calls


def test_text_components_are_explainable() -> None:
    assert lexical_similarity("read src/app.py", "read src/app.py") == 1.0
    assert entity_similarity("inspect line 120", "inspect line 120") == 1.0
    assert action_similarity("read the file", "read config") == 1.0


def test_new_specific_entities_reduce_repetition_score() -> None:
    repeated = score_text_repetition("inspect parser", "inspect parser", 0.95)
    novel = score_text_repetition(
        "inspect parser line 120", "inspect loader line 900", 0.95
    )
    assert repeated.score > novel.score
    assert novel.parameter_novelty > 0


def test_exact_tool_duplicate_scores_one() -> None:
    first = build_tool_fingerprint("Read", {"path": "a.py", "offset": 1, "limit": 20})
    second = build_tool_fingerprint("read", {"limit": 20, "offset": 1, "path": "a.py"})
    result = score_tool_repetition(
        "read a", "read a", 0.8, compare_tool_calls(first, second)
    )
    assert result.exact_duplicate
    assert result.score == 1.0


def test_changed_tool_range_is_penalized_despite_identical_semantics() -> None:
    first = build_tool_fingerprint("Read", {"path": "a.py", "offset": 1, "limit": 20})
    second = build_tool_fingerprint("Read", {"path": "a.py", "offset": 21, "limit": 20})
    result = score_tool_repetition(
        "read a", "read a", 1.0, compare_tool_calls(first, second)
    )
    assert result.parameter_novelty > 0
    assert result.score < 0.93


def _reasoning(text: str, position: int) -> TokenSpan:
    return TokenSpan(text, SpanPhase.REASONING, position, 40, str(position))


def test_classifier_flags_identical_reasoning_with_blended_score(monkeypatch) -> None:
    monkeypatch.setattr(
        classifier,
        "embed_texts",
        lambda texts: np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)
        ),
    )
    spans = [
        _reasoning("Analyze the parser architecture carefully.", 0),
        _reasoning("Analyze the parser architecture carefully.", 1),
    ]
    intent = IntentVector("parser architecture", np.array([1.0, 0.0], dtype=np.float32))
    result = classifier.classify_spans(spans, intent)
    assert result[1].label == SpanLabel.REDUNDANT_REASONING


def test_classifier_protects_semantically_similar_new_specifics(monkeypatch) -> None:
    monkeypatch.setattr(
        classifier,
        "embed_texts",
        lambda texts: np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)
        ),
    )
    spans = [
        _reasoning("Inspect parser.py line 120 for request merging behavior.", 0),
        _reasoning("Inspect loader.py line 900 for session discovery behavior.", 1),
    ]
    intent = IntentVector(
        "inspect implementation", np.array([1.0, 0.0], dtype=np.float32)
    )
    result = classifier.classify_spans(spans, intent)
    assert result[1].label == SpanLabel.ALIGNED_REASONING
