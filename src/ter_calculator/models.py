"""Data models for TER Calculator."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SpanPhase(Enum):
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    GENERATION = "generation"


class SpanLabel(Enum):
    ALIGNED_REASONING = "aligned_reasoning"
    REDUNDANT_REASONING = "redundant_reasoning"
    ALIGNED_TOOL_CALL = "aligned_tool_call"
    UNNECESSARY_TOOL_CALL = "unnecessary_tool_call"
    ALIGNED_RESPONSE = "aligned_response"
    OVER_EXPLANATION = "over_explanation"


ALIGNED_LABELS = {
    SpanLabel.ALIGNED_REASONING,
    SpanLabel.ALIGNED_TOOL_CALL,
    SpanLabel.ALIGNED_RESPONSE,
}

PHASE_WEIGHTS_DEFAULT = {
    SpanPhase.REASONING: 0.3,
    SpanPhase.TOOL_USE: 0.4,
    SpanPhase.GENERATION: 0.3,
}


@dataclass
class ContentBlock:
    block_type: str  # text, tool_use, tool_result, thinking
    text: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_use_id: str | None = None


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class Message:
    uuid: str
    role: str  # user, assistant
    content_blocks: list[ContentBlock] = field(default_factory=list)
    parent_uuid: str | None = None
    timestamp: datetime | None = None
    request_id: str | None = None
    usage: TokenUsage | None = None
    stop_reason: str | None = None


@dataclass
class Session:
    session_id: str
    file_path: str
    messages: list[Message] = field(default_factory=list)
    timestamp: datetime | None = None
    total_tokens: int = 0
    user_prompts: list[str] = field(default_factory=list)


@dataclass
class TokenSpan:
    text: str
    phase: SpanPhase
    position: int
    token_count: int
    source_message_uuid: str
    block_type: str = ""  # e.g. "tool_use", "tool_result", "text", "thinking"
    embedding: np.ndarray | None = None


@dataclass
class IntentVector:
    text: str
    embedding: np.ndarray
    confidence: float = 1.0
    source_prompts: list[str] = field(default_factory=list)


@dataclass
class ClassifiedSpan:
    span: TokenSpan
    label: SpanLabel
    confidence: float
    cosine_similarity: float


@dataclass
class WastePattern:
    pattern_type: str  # reasoning_loop, duplicate_tool_call, context_restatement
    description: str
    start_position: int
    end_position: int
    spans_involved: int
    tokens_wasted: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TERResult:
    session_id: str
    aggregate_ter: float
    raw_ratio: float
    phase_scores: dict[str, float]
    total_tokens: int
    aligned_tokens: int
    waste_tokens: int
    waste_patterns: list[WastePattern] = field(default_factory=list)
    intent: IntentVector | None = None
    classified_spans: list[ClassifiedSpan] = field(default_factory=list)
