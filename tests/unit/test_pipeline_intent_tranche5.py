from types import SimpleNamespace
import numpy as np
import pytest

import ter_calculator.intent as intent
import ter_calculator.analyze_pipeline as ap


class Model:
    def encode(self, x, **kwargs):
        if isinstance(x, list):
            return np.ones((len(x), 384))
        return np.ones(384)


def test_intent_empty_single_multiple_and_embedding(monkeypatch):
    monkeypatch.setattr(intent, "get_embedding_model", lambda: Model())
    empty = SimpleNamespace(user_prompts=[])
    out = intent.extract_intent(empty)
    assert out.confidence == 0 and out.embedding.shape == (384,)
    one = SimpleNamespace(user_prompts=["fix bug"])
    assert intent.extract_intent(one).text == "fix bug"
    many = SimpleNamespace(user_prompts=["fix", "add tests", "ship now"])
    out = intent.extract_intent(many)
    assert out.text.count("add tests") == 1 and out.confidence > 0
    assert intent.embed_text("x").shape == (384,)
    assert intent.embed_texts([]).shape == (0, 384)
    assert intent.embed_texts(["a", "b"]).shape == (2, 384)
    assert intent._compute_confidence(["x"]) == 0.2
    assert intent._compute_confidence(["one two three four six"]) == 0.5
    assert intent._compute_confidence(["one two three four five six seven"]) == 0.7
    assert intent._compute_confidence(["word " * 11]) == 0.85


def test_analyze_pipeline_optional_paths(monkeypatch):
    import ter_calculator.config_parse as cp, ter_calculator.loader as ld
    import ter_calculator.intent_extraction as ie, ter_calculator.classifier as cl
    import ter_calculator.compute as co, ter_calculator.economics as ec
    import ter_calculator.waste as wa, ter_calculator.input_analysis as ia
    import ter_calculator.cost_model as cm, ter_calculator.overthinking as ot
    from ter_calculator.models import SpanPhase, SpanLabel

    monkeypatch.setattr(
        cp,
        "parse_phase_weights",
        lambda x: {"reasoning": 0.3, "tool_use": 0.4, "generation": 0.3},
    )
    monkeypatch.setattr(cp, "parse_cost_model", lambda x: "cost")
    session = SimpleNamespace(session_id="s", user_prompts=["task"])
    monkeypatch.setattr(ld, "load_session", lambda p: session)
    spans = [
        SimpleNamespace(phase=SpanPhase.REASONING, token_count=2, text=f"r{i}")
        for i in range(3)
    ]
    monkeypatch.setattr(ld, "segment_spans", lambda s: spans)
    monkeypatch.setattr(
        ie,
        "SlidingIntentExtractor",
        lambda: SimpleNamespace(extract=lambda p: ["intent"]),
    )
    classified = [
        SimpleNamespace(span=s, label=SpanLabel.ALIGNED_REASONING) for s in spans
    ]
    monkeypatch.setattr(cl, "classify_spans", lambda *a, **k: classified)
    result = SimpleNamespace(
        aggregate_ter=0.8,
        economics=None,
        waste_patterns=None,
        input_analysis=None,
        cost_report=None,
        overthinking_result=None,
    )
    monkeypatch.setattr(co, "compute_ter", lambda *a, **k: result)
    result.economics = SimpleNamespace(
        total_input_tokens=1,
        total_output_tokens=2,
        total_cache_creation_tokens=3,
        total_cache_read_tokens=4,
    )
    monkeypatch.setattr(ec, "compute_economics", lambda *a, **k: result.economics)
    monkeypatch.setattr(wa, "detect_waste_patterns", lambda *a, **k: ["w"])
    monkeypatch.setattr(ia, "analyze_input", lambda *a, **k: "input")
    monkeypatch.setattr(cm, "generate_cost_report", lambda **k: "cost-report")
    monkeypatch.setattr(ot, "analyze_overthinking", lambda x: "over")
    args = SimpleNamespace(
        session_path="x",
        phase_weights="x",
        similarity_threshold=0.4,
        confidence_threshold=0.7,
        no_waste_patterns=False,
        restatement_threshold=0.8,
        cost_model="sonnet",
        no_input_analysis=False,
        prompt_similarity_threshold=0.7,
        cost_weighted=True,
        check_overthinking=True,
    )
    out = ap.analyze_session(args)
    assert out.waste_patterns == ["w"] and out.input_analysis == "input"
    assert out.cost_report == "cost-report" and out.overthinking_result == "over"


def test_default_args_and_pipeline_skip_optionals(monkeypatch):
    d = ap.default_analyze_args("x")
    assert d.session_path == "x"
    # Ensure optional paths can be skipped with a minimal fake module graph
    import ter_calculator.config_parse as cp, ter_calculator.loader as ld
    import ter_calculator.intent_extraction as ie, ter_calculator.classifier as cl
    import ter_calculator.compute as co, ter_calculator.economics as ec

    monkeypatch.setattr(cp, "parse_phase_weights", lambda x: {})
    monkeypatch.setattr(cp, "parse_cost_model", lambda x: None)
    monkeypatch.setattr(
        ld, "load_session", lambda p: SimpleNamespace(session_id="s", user_prompts=[])
    )
    monkeypatch.setattr(ld, "segment_spans", lambda s: [])
    monkeypatch.setattr(
        ie, "SlidingIntentExtractor", lambda: SimpleNamespace(extract=lambda p: ["i"])
    )
    monkeypatch.setattr(cl, "classify_spans", lambda *a, **k: [])
    res = SimpleNamespace(economics=None)
    monkeypatch.setattr(co, "compute_ter", lambda *a, **k: res)
    monkeypatch.setattr(ec, "compute_economics", lambda *a, **k: None)
    args = SimpleNamespace(
        session_path="x",
        phase_weights="x",
        similarity_threshold=0.4,
        confidence_threshold=0.7,
        no_waste_patterns=True,
        restatement_threshold=0.8,
        cost_model="x",
        no_input_analysis=True,
        prompt_similarity_threshold=0.7,
        cost_weighted=False,
        check_overthinking=False,
    )
    assert ap.analyze_session(args) is res


def test_analyze_pipeline_passes_segmentation_config_only_when_enabled(monkeypatch):
    import ter_calculator.config_parse as cp
    import ter_calculator.loader as ld
    import ter_calculator.intent_extraction as ie
    import ter_calculator.classifier as cl
    import ter_calculator.compute as co
    import ter_calculator.economics as ec

    monkeypatch.setattr(cp, "parse_phase_weights", lambda value: {})
    monkeypatch.setattr(cp, "parse_cost_model", lambda value: None)
    session = SimpleNamespace(session_id="s", user_prompts=["task"])
    monkeypatch.setattr(ld, "load_session", lambda path: session)
    captured = {}

    def fake_segment(current_session, config):
        captured["session"] = current_session
        captured["config"] = config
        return []

    monkeypatch.setattr(ld, "segment_spans", fake_segment)
    monkeypatch.setattr(
        ie,
        "SlidingIntentExtractor",
        lambda: SimpleNamespace(extract=lambda prompts: ["intent"]),
    )
    monkeypatch.setattr(cl, "classify_spans", lambda *args, **kwargs: [])
    result = SimpleNamespace(economics=None)
    monkeypatch.setattr(co, "compute_ter", lambda *args, **kwargs: result)
    monkeypatch.setattr(ec, "compute_economics", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        session_path="x",
        phase_weights="x",
        similarity_threshold=0.4,
        confidence_threshold=0.7,
        no_waste_patterns=True,
        restatement_threshold=0.8,
        cost_model="x",
        no_input_analysis=True,
        prompt_similarity_threshold=0.7,
        cost_weighted=False,
        check_overthinking=False,
        fine_segmentation=True,
        segment_min_tokens=15,
        segment_max_tokens=120,
    )

    assert ap.analyze_session(args) is result
    assert captured["session"] is session
    assert captured["config"].enabled is True
    assert captured["config"].min_tokens == 15
    assert captured["config"].max_tokens == 120
