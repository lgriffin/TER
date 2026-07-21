"""Weighted intent construction for multi-turn user prompts.

The helpers in this module keep prompt weighting explicit and testable.  They
avoid the legacy technique of repeating prompt text before embedding, filter
low-information operational messages, boost corrections and constraints, and
identify semantic topic shifts between adjacent informative prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from numpy.typing import NDArray

_OPERATIONAL_PATTERNS = (
    r"^(?:ok(?:ay)?|yes|yep|sure|continue|go ahead|proceed|retry|again|do it|do that|ship it|thanks?|please)$",
    r"^(?:keep going|carry on|sounds good|looks good)[.!\s]*$",
)
_CORRECTION_CUES = (
    "actually",
    "instead",
    "correction",
    "change that",
    "not ",
    "don't ",
    "do not ",
    "rather than",
)
_GOAL_CUES = (
    "add ",
    "build ",
    "create ",
    "implement ",
    "fix ",
    "refactor ",
    "update ",
    "remove ",
    "write ",
    "make ",
)
_CONSTRAINT_CUES = (
    "must",
    "should",
    "ensure",
    "without",
    "only",
    "minimum",
    "maximum",
    "use ",
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")


@dataclass(frozen=True)
class PromptWeight:
    """Explainable weight assigned to one user prompt."""

    prompt: str
    weight: float
    information_score: float
    recency_score: float
    is_operational: bool
    is_correction: bool


def is_low_information_prompt(prompt: str) -> bool:
    """Return whether *prompt* is an operational acknowledgement with no goal."""
    normalized = " ".join(prompt.lower().strip().split()).strip(".!?,;:")
    if not normalized:
        return True
    return any(re.fullmatch(pattern, normalized) for pattern in _OPERATIONAL_PATTERNS)


def is_correction_prompt(prompt: str) -> bool:
    """Return whether *prompt* appears to revise or correct an earlier goal."""
    normalized = prompt.lower().strip()
    return any(cue in normalized for cue in _CORRECTION_CUES)


def prompt_information_score(prompt: str) -> float:
    """Estimate how much task information a prompt contributes, in ``[0, 1]``."""
    if is_low_information_prompt(prompt):
        return 0.05

    tokens = _TOKEN_RE.findall(prompt)
    token_score = min(1.0, len(tokens) / 18.0)
    normalized = prompt.lower().strip()
    cue_bonus = 0.0
    if any(normalized.startswith(cue) for cue in _GOAL_CUES):
        cue_bonus += 0.18
    if any(cue in normalized for cue in _CONSTRAINT_CUES):
        cue_bonus += 0.12
    if any(char in prompt for char in ("/", ".py", ".json", "--")):
        cue_bonus += 0.08
    return min(1.0, 0.25 + token_score * 0.65 + cue_bonus)


def compute_prompt_weights(prompts: list[str]) -> list[PromptWeight]:
    """Compute normalized, explicit weights for a sequence of prompts."""
    if not prompts:
        return []

    raw: list[tuple[str, float, float, bool, bool]] = []
    denominator = max(1, len(prompts) - 1)
    for index, prompt in enumerate(prompts):
        operational = is_low_information_prompt(prompt)
        correction = is_correction_prompt(prompt)
        information = prompt_information_score(prompt)
        recency = 0.75 + 0.25 * (index / denominator)
        correction_multiplier = 1.35 if correction else 1.0
        operational_multiplier = 0.10 if operational else 1.0
        value = information * recency * correction_multiplier * operational_multiplier
        raw.append((prompt, value, information, operational, correction))

    total = sum(item[1] for item in raw)
    if total <= 0.0:
        total = float(len(raw))
        raw = [(p, 1.0, info, op, corr) for p, _, info, op, corr in raw]

    return [
        PromptWeight(
            prompt=prompt,
            weight=value / total,
            information_score=information,
            recency_score=0.75 + 0.25 * (index / denominator),
            is_operational=operational,
            is_correction=correction,
        )
        for index, (prompt, value, information, operational, correction) in enumerate(
            raw
        )
    ]


def weighted_centroid(
    embeddings: NDArray[np.floating],
    weights: list[PromptWeight],
) -> NDArray[np.float64]:
    """Return the L2-normalized weighted centroid of independently embedded prompts."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if matrix.shape[0] != len(weights):
        raise ValueError("embedding count must match prompt weight count")
    if matrix.shape[0] == 0:
        return np.zeros(384, dtype=np.float64)

    values = np.asarray([item.weight for item in weights], dtype=np.float64)
    centroid = np.asarray(np.average(matrix, axis=0, weights=values), dtype=np.float64)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        return centroid
    return np.asarray(centroid / norm, dtype=np.float64)


def cosine_similarity(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Return cosine similarity for two vectors, safely handling zero vectors."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def detect_topic_shifts(
    prompts: list[str],
    embeddings: NDArray[np.floating],
    *,
    threshold: float = 0.42,
) -> list[int]:
    """Return indices that begin a new semantic topic.

    Operational prompts are ignored as shift evidence. A correction is treated
    as a refinement of the active topic unless its embedding is very distant.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    if len(prompts) != matrix.shape[0]:
        raise ValueError("embedding count must match prompt count")

    shifts: list[int] = []
    previous_index: int | None = None
    for index, prompt in enumerate(prompts):
        if is_low_information_prompt(prompt):
            continue
        if previous_index is None:
            previous_index = index
            continue
        similarity = cosine_similarity(matrix[previous_index], matrix[index])
        effective_threshold = threshold * (
            0.65 if is_correction_prompt(prompt) else 1.0
        )
        if similarity < effective_threshold:
            shifts.append(index)
        previous_index = index
    return shifts


def split_prompt_topics(
    prompts: list[str], shift_indices: list[int]
) -> list[list[str]]:
    """Split prompts at topic-shift indices while retaining operational messages."""
    if not prompts:
        return []
    boundaries = [
        0,
        *sorted(set(i for i in shift_indices if 0 < i < len(prompts))),
        len(prompts),
    ]
    return [prompts[start:end] for start, end in zip(boundaries, boundaries[1:])]


def intent_display_text(prompts: list[str]) -> str:
    """Create readable intent text without artificial prompt repetition."""
    informative = [
        prompt.strip() for prompt in prompts if not is_low_information_prompt(prompt)
    ]
    selected = informative or [prompt.strip() for prompt in prompts if prompt.strip()]
    return " | ".join(selected)
