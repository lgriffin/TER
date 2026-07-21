"""Regression tests for structured tool-call repetition detection."""

from __future__ import annotations

import numpy as np

import ter_calculator.classifier as classifier
from ter_calculator.models import IntentVector, SpanLabel, SpanPhase, TokenSpan
from ter_calculator.tool_fingerprints import (
    build_tool_fingerprint,
    compare_tool_calls,
    normalize_tool_arguments,
)


def test_fingerprint_is_stable_across_key_order_and_volatile_metadata() -> None:
    first = build_tool_fingerprint(
        "Read",
        {"file_path": "./src/../src/app.py", "offset": 10, "limit": 20, "id": "a"},
    )
    second = build_tool_fingerprint(
        "read",
        {"limit": 20, "offset": 10, "file_path": "src/app.py", "id": "b"},
    )
    assert first.digest == second.digest
    assert first.path == "src/app.py"
    assert first.line_range == (10, 29)


def test_different_file_range_is_parameter_novelty_not_duplicate() -> None:
    first = build_tool_fingerprint(
        "Read", {"file_path": "src/app.py", "offset": 1, "limit": 100}
    )
    second = build_tool_fingerprint(
        "Read", {"file_path": "src/app.py", "offset": 101, "limit": 100}
    )
    comparison = compare_tool_calls(first, second)
    assert not comparison.exact_duplicate
    assert comparison.same_tool
    assert comparison.parameter_novelty > 0
    assert "line_range" in comparison.changed_fields


def test_query_and_command_changes_are_not_duplicates() -> None:
    search_a = build_tool_fingerprint("Grep", {"query": "TokenSpan", "path": "src"})
    search_b = build_tool_fingerprint("Grep", {"query": "ContentBlock", "path": "src"})
    bash_a = build_tool_fingerprint("Bash", {"command": "pytest tests/unit/test_a.py"})
    bash_b = build_tool_fingerprint("Bash", {"command": "pytest tests/unit/test_b.py"})
    assert not compare_tool_calls(search_a, search_b).exact_duplicate
    assert not compare_tool_calls(bash_a, bash_b).exact_duplicate


def test_nested_arguments_are_normalized_deterministically() -> None:
    assert normalize_tool_arguments({"options": {"b": 2, "a": "  x   y "}}) == {
        "options": {"a": "x y", "b": 2}
    }


def _span(position: int, tool_input: dict[str, object]) -> TokenSpan:
    return TokenSpan(
        text=f"Read {tool_input}",
        phase=SpanPhase.TOOL_USE,
        position=position,
        token_count=10,
        source_message_uuid=str(position),
        block_type="tool_use",
        tool_name="Read",
        tool_input=tool_input,
    )


def test_classifier_flags_exact_structured_duplicate(monkeypatch) -> None:
    monkeypatch.setattr(
        classifier,
        "embed_texts",
        lambda texts: np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)
        ),
    )
    spans = [
        _span(0, {"file_path": "src/app.py", "offset": 1, "limit": 100}),
        _span(1, {"limit": 100, "offset": 1, "file_path": "./src/app.py"}),
    ]
    intent = IntentVector("inspect app", np.array([1.0, 0.0], dtype=np.float32))
    result = classifier.classify_spans(spans, intent)
    assert result[0].label == SpanLabel.ALIGNED_TOOL_CALL
    assert result[1].label == SpanLabel.UNNECESSARY_TOOL_CALL


def test_classifier_protects_changed_parameters_even_with_identical_embeddings(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        classifier,
        "embed_texts",
        lambda texts: np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1)
        ),
    )
    spans = [
        _span(0, {"file_path": "src/app.py", "offset": 1, "limit": 100}),
        _span(1, {"file_path": "src/app.py", "offset": 101, "limit": 100}),
    ]
    intent = IntentVector("inspect app", np.array([1.0, 0.0], dtype=np.float32))
    result = classifier.classify_spans(spans, intent)
    assert [item.label for item in result] == [
        SpanLabel.ALIGNED_TOOL_CALL,
        SpanLabel.ALIGNED_TOOL_CALL,
    ]
