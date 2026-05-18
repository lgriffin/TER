"""Real-time TER streaming monitor.

Watches active Claude Code sessions and computes rolling TER as new
messages arrive.  This bridges TER from post-hoc batch analysis toward
live efficiency signalling.

Key components:

- RollingTERState: maintains per-session running accumulators so TER
  can be updated incrementally without re-processing the entire session.
- TERSignal: an efficiency signal emitted after each new message, carrying
  the current TER, a drift indicator, and optional warnings.
- SessionMonitor: watches a single JSONL file via polling, detects new
  lines, and feeds them through a lightweight classification pipeline.
- LiveDashboard: coordinates monitoring of multiple active sessions
  with configurable callbacks for signal consumers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ter_calculator.models import TERResult

__all__ = [
    "DriftDirection",
    "LiveDashboard",
    "RollingTERState",
    "SessionMonitor",
    "TERSignal",
    "WarningLevel",
    "compute_rolling_ter",
    "detect_drift",
    "load_embedding_model",
]

logger = logging.getLogger(__name__)

# Global model cache for lazy loading
_EMBEDDING_MODEL_CACHE: Any | None = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_POLL_INTERVAL_SEC = 2.0
"""How often to check for new JSONL lines."""

DRIFT_WINDOW = 5
"""Number of recent signals over which to compute TER drift."""

DRIFT_THRESHOLD = 0.10
"""Absolute TER change within the drift window that triggers a drift warning."""

PHASE_WEIGHTS: dict[str, float] = {
    "reasoning": 0.3,
    "tool_use": 0.4,
    "generation": 0.3,
}

SIM_THRESHOLD = 0.40
CONF_THRESHOLD = 0.75
REPETITION_THRESHOLD = 0.88  # Matches classifier.py _check_repetition
REPETITION_WINDOW = 10       # How many recent same-phase spans to compare against

_SIM_REASONING = 0.55
"""Similarity floor for reasoning spans — trigram hashing yields ~0.3–0.5
between unrelated English text, so 0.40 barely fires."""

_SIM_GENERATION = 0.50
"""Similarity floor for generation spans."""

_SIM_FLOOR = 0.25
"""Below this, content is clearly off-topic regardless of length."""

_TOOL_DEDUP_WINDOW = 10
"""How many recent tool signatures to keep for duplicate detection."""

_REPEATED_CMD_THRESHOLD = 3
"""Bash command must repeat this many times to count as waste."""

_BASH_ANTIPATTERN_RES: list[re.Pattern[str]] = [
    re.compile(r"(?:^|\|\s*)cat\s+"),
    re.compile(r"(?:^|\|\s*)head\s+"),
    re.compile(r"(?:^|\|\s*)tail\s+"),
    re.compile(r"(?:^|\|\s*)grep\s+"),
    re.compile(r"(?:^|\|\s*)rg\s+"),
    re.compile(r"^find\s+"),
]

_ERROR_MARKERS = [
    "<tool_use_error>",
    "File does not exist",
    "command not found",
    "No such file or directory",
    "Permission denied",
    "File has not been read yet",
]

_BLOCK_TYPE_TO_PHASE: dict[str, str] = {
    "thinking": "reasoning",
    "tool_use": "tool_use",
    "tool_result": "tool_use",
    "text": "generation",
}


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """Lazy-load the sentence-transformers model for embeddings.

    The model is cached globally so it's only loaded once per process.
    This keeps the CLI fast for commands that don't need embeddings while
    ensuring live monitoring always uses accurate semantic embeddings.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Loaded SentenceTransformer model

    Raises:
        ImportError: If sentence-transformers is not installed
    """
    global _EMBEDDING_MODEL_CACHE

    if _EMBEDDING_MODEL_CACHE is not None:
        return _EMBEDDING_MODEL_CACHE

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for live monitoring. "
            "Install with: pip install sentence-transformers"
        ) from e

    logger.info("Loading embedding model: %s", model_name)
    _EMBEDDING_MODEL_CACHE = SentenceTransformer(model_name)
    logger.info("Model loaded successfully")

    return _EMBEDDING_MODEL_CACHE


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DriftDirection(Enum):
    """Direction of TER change within the rolling window."""

    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"


class WarningLevel(Enum):
    """Severity of a real-time TER warning."""

    INFO = "info"
    CAUTION = "caution"
    ALERT = "alert"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RollingTERState:
    """Incremental accumulators for a streaming TER computation.

    Instead of re-embedding and re-classifying every span on each new
    message, we maintain running totals per phase and update them as new
    content blocks arrive.
    """

    total_tokens: int = 0
    aligned_tokens: int = 0
    waste_tokens: int = 0

    phase_total: dict[str, int] = field(
        default_factory=lambda: {"reasoning": 0, "tool_use": 0, "generation": 0}
    )
    phase_aligned: dict[str, int] = field(
        default_factory=lambda: {"reasoning": 0, "tool_use": 0, "generation": 0}
    )
    phase_waste: dict[str, int] = field(
        default_factory=lambda: {"reasoning": 0, "tool_use": 0, "generation": 0}
    )

    message_count: int = 0
    span_count: int = 0
    recent_ter_values: list[float] = field(default_factory=list)

    intent_embedding: NDArray[np.float32] | None = None
    intent_text: str = ""
    intent_confidence: float = 0.0
    intent_embeddings: list[NDArray[np.float32]] = field(default_factory=list)

    last_request_ids: dict[str, int] = field(default_factory=dict)
    last_file_position: int = 0

    tool_signatures: list[str] = field(default_factory=list)
    file_read_counts: dict[str, int] = field(default_factory=dict)
    bash_command_counts: dict[str, int] = field(default_factory=dict)
    waste_by_type: dict[str, int] = field(default_factory=dict)

    # Economics tracking for live dashboard
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    turn_context_sizes: list[int] = field(default_factory=list)
    session_start_time: float | None = None

    # Waste tracking for cost calculation: assistant at output rate, user at input rate
    assistant_waste_tokens: int = 0
    user_waste_tokens: int = 0

    # Tool call deduplication (Phase 2B)
    tool_call_history: dict[str, list[str]] = field(default_factory=dict)

    # Repetition detection: recent embeddings per phase (reasoning, tool_use, generation)
    recent_phase_embeddings: dict[str, list] = field(
        default_factory=lambda: {"reasoning": [], "tool_use": [], "generation": []}
    )

    # Cross-message tool tracking for repetitive reads and failed retries
    pending_tool_calls: dict[str, tuple] = field(default_factory=dict)
    # tool_use_id → (tool_name, file_path)
    file_read_history: dict[str, list] = field(default_factory=dict)
    # file_path → [token_count_per_read]

    @property
    def aggregate_ter(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        phase_scores: dict[str, float] = {}
        for phase in ("reasoning", "tool_use", "generation"):
            total = self.phase_total[phase]
            phase_scores[phase] = (
                self.phase_aligned[phase] / total if total > 0 else 1.0
            )
        return sum(
            PHASE_WEIGHTS[p] * phase_scores[p] for p in PHASE_WEIGHTS
        )

    @property
    def raw_ratio(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.aligned_tokens / self.total_tokens

    def get_phase_ter_scores(self) -> dict[str, float]:
        """Get TER score for each phase."""
        scores = {}
        for phase in ("reasoning", "tool_use", "generation"):
            total = self.phase_total[phase]
            scores[phase] = (
                self.phase_aligned[phase] / total if total > 0 else 1.0
            )
        return scores

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate: cache_read / (cache_read + input)."""
        total = self.total_cache_read_tokens + self.total_input_tokens
        if total == 0:
            return 0.0
        return self.total_cache_read_tokens / total

    def get_estimated_cost(self, cost_model: Any = None) -> float:
        """Estimate session cost in USD using default Sonnet rates."""
        # Default Sonnet 4.5 rates (per million tokens)
        input_rate = 3.00
        output_rate = 15.00
        cache_read_rate = 0.30
        cache_write_rate = 3.75

        if cost_model is not None:
            input_rate = cost_model.input_rate
            output_rate = cost_model.output_rate
            cache_read_rate = cost_model.cache_read_rate
            cache_write_rate = cost_model.cache_write_rate

        return (
            self.total_input_tokens * input_rate / 1_000_000
            + self.total_output_tokens * output_rate / 1_000_000
            + self.total_cache_read_tokens * cache_read_rate / 1_000_000
            + self.total_cache_creation_tokens * cache_write_rate / 1_000_000
        )

    def get_estimated_waste_cost(self, cost_model: Any = None) -> float:
        """Estimate waste cost in USD.

        Matches post-hoc pricing: no calibration, char/4 tokens × rate directly.
          - Assistant waste (reasoning loops, bash antipatterns) → output rate
          - User-side waste (repetitive reads, error retries) → input rate
        """
        output_rate = 15.00 if cost_model is None else cost_model.output_rate
        input_rate = 3.00 if cost_model is None else cost_model.input_rate
        return (
            self.assistant_waste_tokens * output_rate / 1_000_000
            + self.user_waste_tokens * input_rate / 1_000_000
        )

    def get_context_growth_rate(self) -> float:
        """Calculate context growth rate: final / first context size."""
        if len(self.turn_context_sizes) < 2:
            return 1.0
        # Filter out tiny contexts (< 100 tokens)
        significant = [c for c in self.turn_context_sizes if c > 100]
        if len(significant) < 2:
            return 1.0
        return significant[-1] / significant[0] if significant[0] > 0 else 1.0

    def is_context_bloat_detected(self) -> bool:
        """Detect context bloat: super-linear growth AND >2x size increase."""
        if len(self.turn_context_sizes) < 3:
            return False
        # Detect super-linear growth via second differences
        deltas = [
            self.turn_context_sizes[i + 1] - self.turn_context_sizes[i]
            for i in range(len(self.turn_context_sizes) - 1)
        ]
        if len(deltas) < 2:
            return False
        second_deltas = [deltas[i + 1] - deltas[i] for i in range(len(deltas) - 1)]
        avg_second = sum(second_deltas) / len(second_deltas)
        is_superlinear = avg_second > 0
        growth_rate = self.get_context_growth_rate()
        return is_superlinear and growth_rate > 2.0

    def get_session_duration(self, current_time: float) -> float:
        """Get session duration in seconds."""
        if self.session_start_time is None:
            return 0.0
        return current_time - self.session_start_time


@dataclass(frozen=True, slots=True)
class TERSignal:
    """An efficiency signal emitted after processing a new message."""

    session_id: str
    timestamp: float
    aggregate_ter: float
    raw_ratio: float
    message_index: int
    total_tokens: int
    aligned_tokens: int
    waste_tokens: int
    drift: DriftDirection
    drift_magnitude: float
    is_live: bool = False
    warnings: list[str] = field(default_factory=list)
    warning_level: WarningLevel = WarningLevel.INFO
    phase_ter: dict[str, float] = field(default_factory=dict)
    waste_sources: dict[str, int] = field(default_factory=dict)

    # Economics fields for live dashboard
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cache_hit_rate: float = 0.0
    estimated_cost_usd: float = 0.0
    estimated_waste_cost_usd: float = 0.0
    context_growth_rate: float = 1.0
    context_bloat_detected: bool = False
    session_duration_seconds: float = 0.0

    @property
    def is_healthy(self) -> bool:
        return self.warning_level == WarningLevel.INFO and self.drift != DriftDirection.DEGRADING


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Cheap character heuristic: 1 token ~ 4 chars."""
    return max(1, len(text) // 4)


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _is_aligned(sim: float, phase: str, text: str) -> bool:
    """Classify a span as aligned or waste based on intent similarity.

    Aligned by default. Only waste if a specific signal fires:
    - Tool calls are always aligned (actions, not words).
    - Reasoning is waste only if below threshold AND short filler.
    - Generation is waste only if below threshold AND long verbose text.

    Thresholds match classifier.py's derived values so live and post-hoc
    classify generation/reasoning waste identically:
      filler_sim_max  = max(0.06, min(0.14, SIM_THRESHOLD * 0.28)) ≈ 0.11
      verbose_sim_max = max(0.05, min(0.12, SIM_THRESHOLD * 0.22)) ≈ 0.09
    """
    if phase == "tool_use":
        return True

    filler_sim_max = max(0.06, min(0.14, SIM_THRESHOLD * 0.28))
    verbose_sim_max = max(0.05, min(0.12, SIM_THRESHOLD * 0.22))

    if phase == "reasoning":
        if sim < filler_sim_max and len(text.split()) < 15:
            return False
        return True

    if phase == "generation":
        if sim < verbose_sim_max and len(text.split()) > 50:
            return False
        return True

    return True


def _is_duplicate_tool_call(
    tool_name: str,
    tool_input_json: str,
    state: RollingTERState,
    window: int = 10,
) -> bool:
    """Check if this tool call duplicates a recent one with the same name and input."""
    if tool_name not in state.tool_call_history:
        state.tool_call_history[tool_name] = []

    history = state.tool_call_history[tool_name]
    is_duplicate = tool_input_json in history
    history.append(tool_input_json)

    if len(history) > window:
        state.tool_call_history[tool_name] = history[-window:]

    return is_duplicate


# Bash anti-pattern rules: (regex, recommended tool, description)
_BASH_ANTIPATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"(?:^|\|\s*)cat\s+"), "Read", "cat → Read"),
    (re.compile(r"(?:^|\|\s*)head\s+"), "Read", "head → Read"),
    (re.compile(r"(?:^|\|\s*)tail\s+"), "Read", "tail → Read"),
    (re.compile(r"(?:^|\|\s*)grep\s+"), "Grep", "grep → Grep"),
    (re.compile(r"(?:^|\|\s*)rg\s+"), "Grep", "rg → Grep"),
    (re.compile(r"^find\s+"), "Glob", "find → Glob"),
]


def _is_bash_antipattern(tool_name: str, tool_input: dict[str, Any]) -> bool:
    """Check if a Bash command matches an anti-pattern."""
    if tool_name != "Bash":
        return False

    command = tool_input.get("command", "").strip()
    if not command:
        return False

    for pattern, _, _ in _BASH_ANTIPATTERNS:
        if pattern.search(command):
            return True

    return False


def _is_error_result_text(text: str) -> bool:
    """Check if tool_result text indicates a failure.

    Matches waste.py _is_error_result logic: prefix checks for generic
    markers, substring checks only for specific Claude Code error strings.
    """
    if "<tool_use_error>" in text:
        return True
    if text.startswith("Error:") or text.startswith("Exit code 1"):
        return True
    return any(marker in text for marker in (
        "File does not exist",
        "command not found",
        "No such file or directory",
        "Permission denied",
        "File has not been read yet",
    ))


def _is_repetition(
    embedding: "NDArray[np.float32]",
    recent: list,
    threshold: float = REPETITION_THRESHOLD,
) -> bool:
    """Check if embedding closely matches any recent same-phase embedding."""
    for prev in recent[-REPETITION_WINDOW:]:
        if _cosine_similarity(embedding, prev) >= threshold:
            return True
    return False


def _record_embedding(embedding: "NDArray[np.float32]", recent: list) -> None:
    """Append embedding to recent list, capping at REPETITION_WINDOW."""
    recent.append(embedding)
    if len(recent) > REPETITION_WINDOW:
        del recent[0]


def _extract_blocks_from_line(line_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull content blocks from a JSONL line."""
    msg = line_data.get("message", {})
    content = msg.get("content", [])
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return []


def _get_request_id(line_data: dict[str, Any]) -> str | None:
    """Extract requestId for deduplication."""
    return line_data.get("requestId") or line_data.get("request_id")


def _get_usage(line_data: dict[str, Any]) -> dict[str, int]:
    """Extract token usage dict."""
    msg = line_data.get("message", {})
    return msg.get("usage", {})


def _get_message_timestamp(line_data: dict[str, Any]) -> float:
    """Extract the message timestamp as a unix float, falling back to now."""
    ts = line_data.get("timestamp")
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            pass
    return time.time()


_LIVE_THRESHOLD_SEC = 30.0


def compute_rolling_ter(
    state: RollingTERState,
    new_lines: list[dict[str, Any]],
    *,
    model: Any,
) -> list[TERSignal]:
    """Process new JSONL lines and update rolling TER state.

    Uses the provided SentenceTransformer model for semantic embeddings,
    enabling accurate waste detection in real-time.

    Args:
        state: Rolling state accumulator
        new_lines: New JSONL entries to process
        model: SentenceTransformer model for embeddings

    Returns:
        List of TERSignal objects, one per assistant message processed
    """
    signals: list[TERSignal] = []
    embed_fn = model.encode

    for line_data in new_lines:
        request_id = _get_request_id(line_data)
        if request_id and request_id in state.last_request_ids:
            continue
        if request_id:
            state.last_request_ids[request_id] = _get_usage(line_data).get(
                "output_tokens", 0
            )

        msg = line_data.get("message", {})
        role = msg.get("role", "")
        blocks = _extract_blocks_from_line(line_data)

        if role == "user":
            for block in blocks:
                block_type = block.get("type", "")
                if block_type == "text" and block.get("text"):
                    user_text = block["text"]
                    if state.intent_text:
                        state.intent_text += " " + user_text
                    else:
                        state.intent_text = user_text
                    prompt_emb = embed_fn(
                        user_text, normalize_embeddings=True
                    ).astype(np.float32)
                    state.intent_embeddings.append(prompt_emb)
                    state.intent_embedding = np.mean(
                        state.intent_embeddings, axis=0
                    ).astype(np.float32)
                    norm = np.linalg.norm(state.intent_embedding)
                    if norm > 0:
                        state.intent_embedding /= norm
                    state.intent_confidence = min(
                        1.0, len(state.intent_text.split()) / 20
                    )
                elif block_type == "tool_result":
                    content = block.get("content", "")
                    text = content if isinstance(content, str) else json.dumps(content)
                    if text:
                        tokens = _estimate_tokens(text)
                        state.total_tokens += tokens
                        state.phase_total["tool_use"] += tokens
                        state.span_count += 1

                        # Embed and check repetition against recent tool_use spans
                        span_emb = embed_fn(
                            text, normalize_embeddings=True
                        ).astype(np.float32)
                        recent_tu = state.recent_phase_embeddings["tool_use"]
                        is_repeated = _is_repetition(span_emb, recent_tu)

                        # Check for error result (failed retry)
                        is_error = _is_error_result_text(text)

                        # Track file reads for context (not for waste detection —
                        # repetitive reads are caught by embedding repetition above)
                        tool_use_id = block.get("tool_use_id", "")
                        if tool_use_id in state.pending_tool_calls:
                            t_name, file_path = state.pending_tool_calls[tool_use_id]
                            if t_name == "Read" and file_path:
                                state.file_read_history.setdefault(file_path, []).append(tokens)

                        if is_repeated or is_error:
                            state.waste_tokens += tokens
                            state.phase_waste["tool_use"] += tokens
                            state.user_waste_tokens += tokens  # priced at input rate
                        else:
                            state.aligned_tokens += tokens
                            state.phase_aligned["tool_use"] += tokens
                            _record_embedding(span_emb, recent_tu)
            continue

        if role != "assistant":
            continue

        # Extract usage data for economics tracking
        usage = _get_usage(line_data)
        msg_ts = _get_message_timestamp(line_data)

        # Initialize session start time on first assistant message
        if state.session_start_time is None:
            state.session_start_time = msg_ts

        # Update economics accumulators
        if usage:
            state.total_input_tokens += usage.get("input_tokens", 0)
            state.total_output_tokens += usage.get("output_tokens", 0)
            state.total_cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
            state.total_cache_read_tokens += usage.get("cache_read_input_tokens", 0)

            # Track context size (input + cache_read)
            context_size = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
            if context_size > 0:
                state.turn_context_sizes.append(context_size)

        state.message_count += 1
        message_aligned = 0
        message_total = 0

        for block in blocks:
            block_type = block.get("type", "text")
            phase = _BLOCK_TYPE_TO_PHASE.get(block_type, "generation")

            text = block.get("text", "")
            tool_name = ""
            tool_input_json = ""

            if block_type == "thinking":
                text = block.get("thinking", "")
            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                tool_input_json = json.dumps(block.get("input", {}), sort_keys=True)
                text = f"{tool_name} {tool_input_json}"
                # Record for tool_result matching (repetitive reads / failed retries)
                tool_use_id = block.get("id", "")
                if tool_use_id and tool_name:
                    file_path = block.get("input", {}).get("file_path", "")
                    state.pending_tool_calls[tool_use_id] = (tool_name, file_path)
            elif block_type == "tool_result":
                content = block.get("content", "")
                text = content if isinstance(content, str) else json.dumps(content)

            if not text:
                continue

            tokens = _estimate_tokens(text)
            state.total_tokens += tokens
            state.phase_total[phase] += tokens
            state.span_count += 1
            message_total += tokens

            # Embed the span
            span_emb = embed_fn(text, normalize_embeddings=True).astype(np.float32)

            # Check for duplicate tool calls (Phase 2B) and bash antipatterns (Phase 2C)
            is_duplicate_tool = False
            is_bash_antipattern = False
            if block_type == "tool_use" and tool_name:
                if tool_input_json:
                    is_duplicate_tool = _is_duplicate_tool_call(
                        tool_name, tool_input_json, state
                    )
                tool_input_dict = block.get("input", {})
                is_bash_antipattern = _is_bash_antipattern(tool_name, tool_input_dict)

            # Check repetition against recent same-phase spans.
            # Only for reasoning/generation: tool_use uses exact-match dedup (Phase 2B)
            # and file-read tracking instead, since tool results are structurally similar
            # even when semantically different (e.g., different file contents).
            recent_embs = state.recent_phase_embeddings.get(phase, [])
            is_repeated = (
                phase in ("reasoning", "generation")
                and _is_repetition(span_emb, recent_embs)
            )

            if is_duplicate_tool or is_bash_antipattern or is_repeated:
                aligned = False
            elif state.intent_embedding is not None:
                sim = _cosine_similarity(span_emb, state.intent_embedding)
                aligned = _is_aligned(sim, phase, text)
            else:
                aligned = True

            waste_type = ""
            if block_type == "tool_use" and aligned:
                aligned, waste_type = _check_tool_patterns(state, block)

            if aligned:
                state.aligned_tokens += tokens
                state.phase_aligned[phase] += tokens
                message_aligned += tokens
                # Track embedding for future repetition detection (reasoning/generation only)
                if phase in ("reasoning", "generation"):
                    _record_embedding(span_emb, state.recent_phase_embeddings[phase])
            else:
                state.waste_tokens += tokens
                state.phase_waste[phase] += tokens
                state.assistant_waste_tokens += tokens

        if message_total == 0:
            continue

        current_ter = state.aggregate_ter
        state.recent_ter_values.append(current_ter)
        if len(state.recent_ter_values) > DRIFT_WINDOW * 2:
            state.recent_ter_values = state.recent_ter_values[-DRIFT_WINDOW * 2 :]

        drift_dir, drift_mag = detect_drift(state.recent_ter_values)

        warnings: list[str] = []
        level = WarningLevel.INFO

        if drift_dir == DriftDirection.DEGRADING and drift_mag > DRIFT_THRESHOLD:
            warnings.append(
                f"TER dropped {drift_mag:.2f} over last {DRIFT_WINDOW} messages"
            )
            level = WarningLevel.CAUTION

        if current_ter < 0.4:
            warnings.append(
                f"TER is critically low ({current_ter:.2f}) — session may be spiralling"
            )
            level = WarningLevel.ALERT

        ratio = state.raw_ratio
        if state.total_tokens > 5000 and ratio < 0.5:
            warnings.append(
                f"Over half of tokens ({state.waste_tokens}) classified as waste"
            )
            if level == WarningLevel.INFO:
                level = WarningLevel.CAUTION

        # Get phase-specific TER scores
        phase_ter = state.get_phase_ter_scores()

        # Build waste sources breakdown (phases + pattern types)
        waste_sources: dict[str, int] = {}
        for phase, waste in state.phase_waste.items():
            if waste > 0:
                waste_sources[phase] = waste
        for wtype, wtokens in state.waste_by_type.items():
            if wtokens > 0:
                waste_sources[wtype] = wtokens

        # Calculate economics metrics
        cache_hit_rate = state.get_cache_hit_rate()
        estimated_cost = state.get_estimated_cost()
        estimated_waste_cost = state.get_estimated_waste_cost()
        context_growth_rate = state.get_context_growth_rate()
        context_bloat = state.is_context_bloat_detected()
        session_duration = state.get_session_duration(msg_ts)

        # Add context bloat warning
        if context_bloat and "bloat" not in " ".join(warnings).lower():
            warnings.append(
                f"Context bloat detected: {context_growth_rate:.1f}x growth"
            )
            if level == WarningLevel.INFO:
                level = WarningLevel.CAUTION

        signal = TERSignal(
            session_id=line_data.get("sessionId", "unknown"),
            timestamp=msg_ts,
            aggregate_ter=current_ter,
            raw_ratio=ratio,
            message_index=state.message_count,
            total_tokens=state.total_tokens,
            aligned_tokens=state.aligned_tokens,
            waste_tokens=state.waste_tokens,
            drift=drift_dir,
            drift_magnitude=drift_mag,
            warnings=warnings,
            warning_level=level,
            is_live=(time.time() - msg_ts) < _LIVE_THRESHOLD_SEC,
            phase_ter=phase_ter,
            waste_sources=waste_sources,
            # Economics fields
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
            cache_creation_tokens=state.total_cache_creation_tokens,
            cache_read_tokens=state.total_cache_read_tokens,
            cache_hit_rate=cache_hit_rate,
            estimated_cost_usd=estimated_cost,
            estimated_waste_cost_usd=estimated_waste_cost,
            context_growth_rate=context_growth_rate,
            context_bloat_detected=context_bloat,
            session_duration_seconds=session_duration,
        )
        signals.append(signal)

    return signals


def detect_drift(
    recent_values: list[float],
    window: int = DRIFT_WINDOW,
    threshold: float = DRIFT_THRESHOLD,
) -> tuple[DriftDirection, float]:
    """Compute TER drift direction and magnitude from recent values.

    Uses a simple linear slope over the last *window* values.
    """
    if len(recent_values) < 2:
        return DriftDirection.STABLE, 0.0

    vals = recent_values[-window:]
    if len(vals) < 2:
        return DriftDirection.STABLE, 0.0

    xs = np.arange(len(vals), dtype=np.float64)
    ys = np.array(vals, dtype=np.float64)
    slope = float(np.polyfit(xs, ys, 1)[0])
    magnitude = abs(slope * len(vals))

    if magnitude < threshold:
        return DriftDirection.STABLE, magnitude
    if slope > 0:
        return DriftDirection.IMPROVING, magnitude
    return DriftDirection.DEGRADING, magnitude


# ---------------------------------------------------------------------------
# SessionMonitor — watches a single JSONL file
# ---------------------------------------------------------------------------


class SessionMonitor:
    """Polls a JSONL session file and emits TERSignals on new content."""

    def __init__(
        self,
        path: Path | str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SEC,
        model: Any | None = None,
        on_signal: Callable[[TERSignal], None] | None = None,
        skip_history: bool = True,
    ) -> None:
        self.path = Path(path)
        self.poll_interval = poll_interval
        self.model = model
        self.on_signal = on_signal
        self.state = RollingTERState()
        self._stop = False
        self._file_pos = 0
        self._caught_up = not skip_history

    def _read_new_lines(self) -> list[dict[str, Any]]:
        """Read lines appended since last poll using byte-offset seek."""
        if not self.path.exists():
            return []
        new_lines: list[dict[str, Any]] = []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                fh.seek(self._file_pos)
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        new_lines.append(json.loads(raw))
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed JSONL line")
                self._file_pos = fh.tell()
        except OSError as exc:
            logger.warning("Could not read %s: %s", self.path, exc)
        return new_lines

    def poll_once(self) -> list[TERSignal]:
        """Check for new lines and return any signals produced."""
        new_lines = self._read_new_lines()
        if not new_lines:
            return []
        signals = compute_rolling_ter(self.state, new_lines, model=self.model)
        if not self._caught_up:
            self._caught_up = True
            if signals:
                last = signals[-1]
                if self.on_signal:
                    self.on_signal(last)
            return signals
        if self.on_signal:
            for sig in signals:
                self.on_signal(sig)
        return signals

    def run(self) -> None:
        """Blocking poll loop.  Call ``stop()`` from another thread to exit."""
        logger.info("Monitoring %s (poll every %.1fs)", self.path, self.poll_interval)
        while not self._stop:
            self.poll_once()
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        self._stop = True

    @property
    def current_ter(self) -> float:
        return self.state.aggregate_ter

    @property
    def signal_history(self) -> list[float]:
        return list(self.state.recent_ter_values)


# ---------------------------------------------------------------------------
# LiveDashboard — coordinates multiple monitors
# ---------------------------------------------------------------------------


class LiveDashboard:
    """Manages multiple SessionMonitors and aggregates signals.

    Intended as the entry point for ``ter watch <project>`` in the CLI.
    """

    def __init__(
        self,
        project_dir: Path | str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SEC,
        model: Any | None = None,
        on_signal: Callable[[TERSignal], None] | None = None,
        skip_history: bool = True,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.poll_interval = poll_interval
        self.model = model
        self.on_signal = on_signal
        self.skip_history = skip_history
        self._monitors: dict[str, SessionMonitor] = {}
        self._stop = False

    def _discover_sessions(self) -> list[Path]:
        """Find JSONL files in the project directory tree."""
        if not self.project_dir.exists():
            return []
        return sorted(self.project_dir.rglob("*.jsonl"))

    def _ensure_monitor(self, path: Path) -> SessionMonitor:
        key = str(path)
        if key not in self._monitors:
            mon = SessionMonitor(
                path,
                poll_interval=self.poll_interval,
                model=self.model,
                on_signal=self.on_signal,
                skip_history=self.skip_history,
            )
            self._monitors[key] = mon
            logger.info("Tracking new session: %s", path.name)
        return self._monitors[key]

    def poll_once(self) -> list[TERSignal]:
        """Discover sessions, poll each, return all signals."""
        signals: list[TERSignal] = []
        for path in self._discover_sessions():
            mon = self._ensure_monitor(path)
            signals.extend(mon.poll_once())
        return signals

    def run(self) -> None:
        """Blocking poll loop across all sessions."""
        logger.info("Watching %s for active sessions", self.project_dir)
        while not self._stop:
            self.poll_once()
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        self._stop = True

    @property
    def active_sessions(self) -> dict[str, float]:
        """Map of session file -> current TER."""
        return {k: m.current_ter for k, m in self._monitors.items()}

    def get_summary(self) -> dict[str, Any]:
        """Snapshot summary of all monitored sessions."""
        sessions = []
        for path, mon in self._monitors.items():
            sessions.append(
                {
                    "path": path,
                    "ter": mon.current_ter,
                    "messages": mon.state.message_count,
                    "total_tokens": mon.state.total_tokens,
                    "waste_tokens": mon.state.waste_tokens,
                    "drift": detect_drift(mon.state.recent_ter_values)[0].value,
                }
            )
        total_tokens = sum(s["total_tokens"] for s in sessions)
        total_waste = sum(s["waste_tokens"] for s in sessions)
        avg_ter = (
            sum(s["ter"] for s in sessions) / len(sessions) if sessions else 0.0
        )
        return {
            "session_count": len(sessions),
            "average_ter": round(avg_ter, 4),
            "total_tokens": total_tokens,
            "total_waste": total_waste,
            "sessions": sessions,
        }
