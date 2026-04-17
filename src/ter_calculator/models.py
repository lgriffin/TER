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


class TokenOrigin(Enum):
    USER = "user"
    MODEL = "model"


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
    # assistant | user — aligns waste $ with billed output vs input context.
    source_role: str = "assistant"


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
class CostModel:
    """Pricing rates per million tokens."""
    input_rate: float = 3.00
    output_rate: float = 15.00
    cache_read_rate: float = 0.30
    cache_write_rate: float = 3.75


@dataclass
class PositionalBreakdown:
    """TER computed over session thirds (early/mid/late)."""
    early_ter: float
    mid_ter: float
    late_ter: float
    early_span_count: int
    mid_span_count: int
    late_span_count: int


@dataclass
class InputGrowth:
    """Turn-over-turn input token growth tracking."""
    turn_input_tokens: list[int]
    growth_rate: float
    is_superlinear: bool
    context_bloat_detected: bool


@dataclass
class SessionEconomics:
    """Aggregated token usage, cost, positional analysis, and growth."""
    total_input_tokens: int
    total_output_tokens: int
    total_cache_creation_tokens: int
    total_cache_read_tokens: int
    input_output_ratio: float
    cache_hit_rate: float
    estimated_cost_usd: float
    estimated_waste_cost_usd: float
    cost_model: CostModel
    positional: PositionalBreakdown
    input_growth: InputGrowth
    """Billed output_tokens / heuristic assistant span tokens (1.0 if N/A)."""
    waste_output_calibration_ratio: float = 1.0


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
    economics: SessionEconomics | None = None
    input_analysis: InputAnalysis | None = None


@dataclass
class TokenBreakdown:
    """Token counts categorised by origin (user vs model)."""
    user_input_tokens: int = 0
    user_result_tokens: int = 0
    model_reasoning_tokens: int = 0
    model_tool_tokens: int = 0
    model_generation_tokens: int = 0
    total_user_tokens: int = 0
    total_model_tokens: int = 0
    user_ratio: float = 0.0


@dataclass
class PromptPair:
    """A pair of user prompts with their cosine similarity."""
    prompt_a_index: int
    prompt_b_index: int
    similarity: float
    prompt_a_text: str
    prompt_b_text: str


@dataclass
class PromptSimilarityResult:
    """Pairwise similarity analysis of user prompts."""
    similarity_matrix: list[list[float]] = field(default_factory=list)
    similar_pairs: list[PromptPair] = field(default_factory=list)
    prompt_redundancy_score: float = 0.0
    prompt_count: int = 0


@dataclass
class IntentDriftStep:
    """A single step in the intent drift sequence."""
    from_index: int
    to_index: int
    similarity: float
    drift_type: str  # "convergent", "evolving", "divergent"


@dataclass
class IntentDrift:
    """Turn-over-turn intent drift analysis."""
    steps: list[IntentDriftStep] = field(default_factory=list)
    overall_trajectory: str = "stable"  # "convergent", "divergent", "stable", "mixed"
    average_drift: float = 0.0


@dataclass
class PromptResponsePair:
    """A user prompt paired with the model's response and their alignment."""
    prompt_index: int
    prompt_text: str
    response_text: str
    alignment: float


@dataclass
class PromptResponseAlignment:
    """Alignment analysis between user prompts and model responses."""
    pairs: list[PromptResponsePair] = field(default_factory=list)
    average_alignment: float = 0.0
    low_alignment_count: int = 0


@dataclass
class InputAnalysis:
    """Combined input-side analysis: token breakdown + prompt similarity."""
    token_breakdown: TokenBreakdown = field(default_factory=TokenBreakdown)
    prompt_similarity: PromptSimilarityResult = field(
        default_factory=PromptSimilarityResult
    )
    intent_drift: IntentDrift = field(default_factory=IntentDrift)
    prompt_response_alignment: PromptResponseAlignment = field(
        default_factory=PromptResponseAlignment
    )
