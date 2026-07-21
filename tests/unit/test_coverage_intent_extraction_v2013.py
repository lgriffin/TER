from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from ter_calculator import intent_extraction as ie


def _vec(text: str) -> np.ndarray:
    # deterministic orthogonal-ish vectors based on keywords
    v = np.zeros(384, dtype=np.float32)
    v[0] = 1 if "alpha" in text else 0
    v[1] = 1 if "beta" in text else 0
    v[2] = 1 if not v.any() else 0
    return v


def test_helpers_and_structured_goal(monkeypatch):
    monkeypatch.setattr(ie, "embed_text", _vec)
    monkeypatch.setattr(
        ie, "embed_texts", lambda texts: np.stack([_vec(t) for t in texts])
    )
    assert ie._cosine_similarity(np.zeros(2), np.ones(2)) == 0
    assert ie._cosine_similarity(np.array([1.0, 0]), np.array([1.0, 0])) == 1
    goal = ie.StructuredGoal("Build alpha", ["test"], ["fast"], ["archive"])
    assert "Sub-goals" in goal.to_embedding_text()
    assert ie.StructuredGoal("x").to_embedding_text() == "x"
    assert ie._prompt_confidence("x") == 0.2
    assert ie._prompt_confidence("two words") == 0.3
    assert ie._prompt_confidence("one two three four") == 0.5
    assert ie._prompt_confidence("one two three four five six") == 0.7
    assert ie._prompt_confidence(" ".join(["w"] * 11)) == 0.85
    assert ie._segment_confidence([]) == 0
    assert ie._segment_confidence(["alpha task", "alpha task again"]) > 0.3


def test_sliding_extract_all_paths(monkeypatch):
    monkeypatch.setattr(ie, "_embed", _vec)
    monkeypatch.setattr(
        ie,
        "_embed_batch",
        lambda texts: (
            np.stack([_vec(t) for t in texts])
            if texts
            else np.zeros((0, 384), dtype=np.float32)
        ),
    )
    ex = ie.SlidingIntentExtractor(window_size=2, split_threshold=0.5)
    assert ex.extract([])[0].confidence == 0
    assert ex.extract(["alpha task"])[0].source_prompts == ["alpha task"]
    out = ex.extract(["alpha one", "alpha two", "beta one", "beta two", "other"])
    assert len(out) >= 3
    assert all(x.embedding.shape == (384,) for x in out)


def test_hierarchical_extract_and_score(monkeypatch):
    monkeypatch.setattr(ie, "_embed", _vec)
    monkeypatch.setattr(
        ie,
        "_embed_batch",
        lambda texts: (
            np.stack([_vec(t) for t in texts])
            if texts
            else np.zeros((0, 384), dtype=np.float32)
        ),
    )
    ex = ie.HierarchicalIntentExtractor(sub_intent_weight=2)
    assert ex.sub_intent_weight == 1
    assert ex.extract([])[0].text == ""
    intents = ex.extract(["alpha overall", "beta detail", "alpha followup"])
    assert len(intents) == 3
    score, best = ex.score_span(_vec("beta"), intents)
    assert score == pytest.approx(1) and best.text == "beta detail"
    score2, best2 = ie.HierarchicalIntentExtractor(0).score_span(_vec("alpha"), intents)
    assert best2.text in {
        "alpha overall",
        "alpha followup",
    } and score2 == pytest.approx(1)
    assert ex.score_span(_vec("alpha"), [])[0] == 0
    one = [intents[0]]
    assert ex.score_span(_vec("alpha"), one)[1] is intents[0]
    assert ex._build_sub_intents([]) == []


def test_llm_success_fallback_and_import_error(monkeypatch):
    monkeypatch.setattr(ie, "_embed", _vec)
    ex = ie.LLMIntentExtractor(api_key=None)
    assert ex.extract([])[0].text == ""
    fallback = ex.extract(["alpha task", "more"])[0]
    assert fallback.text == "alpha task more"

    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                text='{"primary_goal":"alpha","sub_goals":["beta"],"constraints":[],"expected_outputs":["tests"]}'
            )
        ]
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response))
    ex2 = ie.LLMIntentExtractor(api_key="k")
    ex2._client = client
    result = ex2.extract(["do it"])[0]
    assert result.confidence == 0.95 and ex2.structured_goal.primary_goal == "alpha"

    ex3 = ie.LLMIntentExtractor(api_key="k")
    ex3._client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **k: (_ for _ in ()).throw(RuntimeError("bad"))
        )
    )
    assert ex3.extract(["alpha"])[0].text == "alpha"

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "anthropic":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        ie.LLMIntentExtractor(api_key="k")._get_client()


def test_factory_and_protocol():
    assert isinstance(ie.create_intent_extractor("sliding"), ie.IntentStrategy)
    assert isinstance(ie.create_intent_extractor("hierarchical"), ie.IntentStrategy)
    assert isinstance(ie.create_intent_extractor("llm"), ie.IntentStrategy)
    with pytest.raises(ValueError, match="Unknown intent"):
        ie.create_intent_extractor("bad")
