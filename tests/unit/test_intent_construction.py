"""Regression tests for v08 weighted intent construction."""

from types import SimpleNamespace

import numpy as np
import pytest

import ter_calculator.intent as intent
from ter_calculator.intent_construction import (
    compute_prompt_weights,
    detect_topic_shifts,
    intent_display_text,
    is_correction_prompt,
    is_low_information_prompt,
    split_prompt_topics,
    weighted_centroid,
)


class MappingModel:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, value, **kwargs):
        if isinstance(value, list):
            return np.asarray([self.vectors[item] for item in value], dtype=float)
        return np.asarray(self.vectors[value], dtype=float)


def test_operational_prompts_are_detected_and_downweighted() -> None:
    prompts = ["Implement JWT authentication", "continue"]
    weights = compute_prompt_weights(prompts)
    assert is_low_information_prompt("Continue.")
    assert weights[1].is_operational
    assert weights[1].weight < weights[0].weight / 5
    assert intent_display_text(prompts) == "Implement JWT authentication"


def test_corrections_receive_more_weight_than_similar_followups() -> None:
    prompts = ["Use cookies", "Add tests", "Actually, use JWT instead"]
    weights = compute_prompt_weights(prompts)
    assert is_correction_prompt(prompts[-1])
    assert weights[-1].is_correction
    assert weights[-1].weight > weights[1].weight


def test_weighted_centroid_is_normalized() -> None:
    prompts = ["Build API", "Add validation"]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    result = weighted_centroid(embeddings, compute_prompt_weights(prompts))
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert result.shape == (2,)


def test_weighted_centroid_validates_shape() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        weighted_centroid(np.asarray([1.0, 2.0]), compute_prompt_weights(["task"]))
    with pytest.raises(ValueError, match="count"):
        weighted_centroid(np.ones((2, 3)), compute_prompt_weights(["task"]))


def test_topic_shift_ignores_operational_prompt() -> None:
    prompts = ["Fix database migration", "continue", "Write marketing documentation"]
    embeddings = np.asarray([[1.0, 0.0], [0.2, 0.2], [0.0, 1.0]])
    shifts = detect_topic_shifts(prompts, embeddings, threshold=0.5)
    assert shifts == [2]
    assert split_prompt_topics(prompts, shifts) == [
        ["Fix database migration", "continue"],
        ["Write marketing documentation"],
    ]


def test_correction_is_treated_as_refinement_at_moderate_similarity() -> None:
    prompts = ["Use cookies for authentication", "Actually, use JWT instead"]
    embeddings = np.asarray([[1.0, 0.0], [0.45, 0.89]])
    assert detect_topic_shifts(prompts, embeddings, threshold=0.6) == []


def test_extract_intent_uses_latest_topic_centroid(monkeypatch) -> None:
    prompts = ["Fix database migration", "continue", "Write marketing documentation"]
    model = MappingModel(
        {
            prompts[0]: [1.0, 0.0, 0.0],
            prompts[1]: [0.2, 0.2, 0.0],
            prompts[2]: [0.0, 1.0, 0.0],
        }
    )
    monkeypatch.setattr(intent, "get_embedding_model", lambda: model)
    result = intent.extract_intent(SimpleNamespace(user_prompts=prompts))
    assert result.embedding == pytest.approx(np.asarray([0.0, 1.0, 0.0]))
    assert result.text == "Fix database migration | Write marketing documentation"
    assert result.source_prompts == prompts


def test_extract_intent_topics_returns_one_vector_per_topic(monkeypatch) -> None:
    prompts = ["Fix database migration", "Write marketing documentation"]
    model = MappingModel({prompts[0]: [1.0, 0.0], prompts[1]: [0.0, 1.0]})
    monkeypatch.setattr(intent, "get_embedding_model", lambda: model)
    topics = intent.extract_intent_topics(
        SimpleNamespace(user_prompts=prompts), split_threshold=0.5
    )
    assert len(topics) == 2
    assert topics[0].source_prompts == [prompts[0]]
    assert topics[1].source_prompts == [prompts[1]]
