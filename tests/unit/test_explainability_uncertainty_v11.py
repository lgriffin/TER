from __future__ import annotations

import json

import numpy as np

from ter_calculator.classifier import classify_spans
from ter_calculator.compute import compute_ter
from ter_calculator.formatter_json import format_json
from ter_calculator.formatter_text import format_text
from ter_calculator.models import IntentVector, SpanLabel, SpanPhase, TokenSpan
from ter_calculator.uncertainty import estimate_uncertainty


def _span(
    text: str, position: int, phase: SpanPhase = SpanPhase.REASONING
) -> TokenSpan:
    return TokenSpan(text, phase, position, 10, f"m-{position}")


def test_repetition_explanation_identifies_matched_prior(monkeypatch):
    monkeypatch.setattr(
        "ter_calculator.classifier.embed_texts",
        lambda texts: np.array([[1.0, 0.0] for _ in texts]),
    )
    intent = IntentVector("task", np.array([1.0, 0.0]))
    spans = [
        _span("Run the same validation now", 0),
        _span("Run the same validation now", 1),
    ]

    result = classify_spans(spans, intent)

    assert result[1].label == SpanLabel.REDUNDANT_REASONING
    assert result[1].explanation is not None
    assert result[1].explanation.reason_code == "repetition"
    assert result[1].explanation.matched_prior_position == 0
    assert result[1].explanation.signals["repetition_score"] >= 0.88


def test_parameter_novelty_is_exposed_for_aligned_tool_call(monkeypatch):
    monkeypatch.setattr(
        "ter_calculator.classifier.embed_texts",
        lambda texts: np.array([[1.0, 0.0] for _ in texts]),
    )
    intent = IntentVector("inspect files", np.array([1.0, 0.0]))
    spans = [
        TokenSpan(
            "read a",
            SpanPhase.TOOL_USE,
            0,
            5,
            "m0",
            tool_name="Read",
            tool_input={"path": "a.py"},
        ),
        TokenSpan(
            "read b",
            SpanPhase.TOOL_USE,
            1,
            5,
            "m1",
            tool_name="Read",
            tool_input={"path": "b.py"},
        ),
    ]

    result = classify_spans(spans, intent)

    assert result[1].label == SpanLabel.ALIGNED_TOOL_CALL
    assert result[1].explanation is not None
    assert result[1].explanation.signals["parameter_novelty"] > 0.0


def test_uncertainty_is_deterministic(monkeypatch):
    monkeypatch.setattr(
        "ter_calculator.classifier.embed_texts",
        lambda texts: np.array([[1.0, 0.0] for _ in texts]),
    )
    intent = IntentVector("task", np.array([1.0, 0.0]))
    classified = classify_spans([_span("Useful detailed analysis", 0)], intent)

    first = estimate_uncertainty(classified, bootstrap_samples=50, seed=3)
    second = estimate_uncertainty(classified, bootstrap_samples=50, seed=3)

    assert first == second
    assert first.reliability == "low"


def test_result_formatters_expose_uncertainty_and_explanations(monkeypatch):
    monkeypatch.setattr(
        "ter_calculator.classifier.embed_texts",
        lambda texts: np.array([[1.0, 0.0] for _ in texts]),
    )
    intent = IntentVector("task", np.array([1.0, 0.0]))
    classified = classify_spans([_span("Useful detailed analysis", 0)], intent)
    result = compute_ter(classified, "s", intent)

    payload = json.loads(format_json(result))
    assert payload["classifier_version"] == "v11"
    assert payload["uncertainty"]["method"] == "deterministic_span_bootstrap"
    assert payload["classified_spans"][0]["explanation"]["reason_code"]

    rendered = format_text(result)
    assert "95% interval:" in rendered
    assert "Classification Evidence:" in rendered
