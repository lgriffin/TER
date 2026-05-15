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
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_POLL_INTERVAL_SEC = 2.0
"""How often to check for new JSONL lines."""

DRIFT_WINDOW = 5
"""Number of recent signals over which to compute TER drift."""

DRIFT_THRESHOLD = 0.10
"""Absolute TER change within the drift window that triggers a drift warning."""

EMBEDDING_DIM = 384

PHASE_WEIGHTS: dict[str, float] = {
    "reasoning": 0.3,
    "tool_use": 0.4,
    "generation": 0.3,
}

SIM_THRESHOLD = 0.40
CONF_THRESHOLD = 0.75

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

    # Track assistant-only waste for cost calculation (matches post-hoc)
    assistant_waste_tokens: int = 0

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
        """Estimate waste cost in USD based on assistant-only waste tokens.

        Matches post-hoc calculation which only counts assistant-origin waste,
        excluding user-side waste like tool results.
        """
        output_rate = 15.00 if cost_model is None else cost_model.output_rate
        # Calibrate assistant waste to output_tokens if available
        if self.total_output_tokens > 0 and self.total_tokens > 0:
            calibration = self.total_output_tokens / self.total_tokens
        else:
            calibration = 1.0
        return self.assistant_waste_tokens * calibration * output_rate / 1_000_000

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


def _embed_text_fast(text: str) -> NDArray[np.float32]:
    """Lightweight pseudo-embedding using character trigram hashing.

    For real-time monitoring we cannot afford the full sentence-transformers
    model on every message.  Instead we produce a deterministic 384-dim vector
    from character trigram hashes.  This is less semantically rich but runs
    in <1ms and provides a usable similarity signal for drift detection.
    """
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    text_lower = text.lower()
    if len(text_lower) < 3:
        vec[0] = 1.0
        return vec
    for i in range(len(text_lower) - 2):
        trigram = text_lower[i : i + 3]
        idx = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % EMBEDDING_DIM
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _is_aligned(sim: float, phase: str, text: str) -> bool:
    """Classify a span as aligned or waste based on intent similarity.

    Thresholds are calibrated for the trigram-hash embedding, which yields
    moderate cosine similarity (~0.3–0.5) even between unrelated English
    texts.  Tool-use pattern waste (duplicates, antipatterns) is handled
    separately by _check_tool_patterns.
    """
    if phase == "tool_use":
        return True

    word_count = len(text.split())

    if sim < _SIM_FLOOR and word_count > 5:
        return False

    if phase == "reasoning":
        if sim < _SIM_REASONING and word_count > 25:
            return False
        return True

    if phase == "generation":
        if sim < _SIM_GENERATION and word_count > 20:
            return False
        return True

    return True


def _is_bash_antipattern(cmd: str) -> bool:
    """Check if a Bash command should use a dedicated tool instead."""
    cmd = cmd.strip()
    return any(p.search(cmd) for p in _BASH_ANTIPATTERN_RES)


def _normalize_bash_cmd(cmd: str) -> str:
    """Normalize a bash command for repeat detection."""
    cmd = cmd.strip()
    cmd = re.sub(r'\s*\|\s*(tail|head)\s+-\d+\s*$', '', cmd)
    cmd = re.sub(r'\s+', ' ', cmd)
    return cmd


def _has_error_markers(text: str) -> bool:
    """Check if tool result text indicates an error."""
    if text.startswith("Error:") or text.startswith("Exit code 1"):
        return True
    return any(marker in text for marker in _ERROR_MARKERS)


def _check_tool_patterns(
    state: RollingTERState,
    block: dict[str, Any],
) -> tuple[bool, str]:
    """Check a tool_use block for waste patterns.

    Returns (is_aligned, waste_type).  Mutates state to track history.
    """
    tool_name = block.get("name", "")
    tool_input = block.get("input", {})

    sig = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"

    recent = state.tool_signatures[-_TOOL_DEDUP_WINDOW:]
    is_duplicate = sig in recent
    state.tool_signatures.append(sig)
    if len(state.tool_signatures) > _TOOL_DEDUP_WINDOW * 2:
        state.tool_signatures = state.tool_signatures[-_TOOL_DEDUP_WINDOW * 2:]

    if is_duplicate:
        return False, "duplicate_tool_call"

    if tool_name == "Read":
        fp = tool_input.get("file_path", "")
        if fp:
            count = state.file_read_counts.get(fp, 0) + 1
            state.file_read_counts[fp] = count
            if count > 1:
                return False, "repetitive_read"

    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        if cmd:
            if _is_bash_antipattern(cmd):
                return False, "bash_antipattern"
            norm = _normalize_bash_cmd(cmd)
            if norm:
                count = state.bash_command_counts.get(norm, 0) + 1
                state.bash_command_counts[norm] = count
                if count >= _REPEATED_CMD_THRESHOLD:
                    return False, "repeated_command"

    return True, ""


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
    model: Any | None = None,
) -> list[TERSignal]:
    """Process new JSONL lines and update rolling TER state.

    If *model* is provided (a SentenceTransformer), it is used for proper
    semantic embedding.  Otherwise falls back to the fast trigram hash.

    Returns one TERSignal per assistant message processed.
    """
    signals: list[TERSignal] = []
    embed_fn = model.encode if model is not None else None

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
                    if embed_fn is not None:
                        prompt_emb = embed_fn(
                            user_text, normalize_embeddings=True
                        ).astype(np.float32)
                    else:
                        prompt_emb = _embed_text_fast(user_text)
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
                        phase = "tool_use"
                        state.total_tokens += tokens
                        state.phase_total[phase] += tokens
                        state.span_count += 1
                        waste_type = ""
                        if _has_error_markers(text):
                            aligned = False
                            waste_type = "failed_tool"
                        elif state.intent_embedding is not None:
                            if embed_fn is not None:
                                span_emb = embed_fn(
                                    text, normalize_embeddings=True
                                ).astype(np.float32)
                            else:
                                span_emb = _embed_text_fast(text)
                            sim = _cosine_similarity(span_emb, state.intent_embedding)
                            aligned = _is_aligned(sim, phase, text)
                        else:
                            aligned = True
                        if aligned:
                            state.aligned_tokens += tokens
                            state.phase_aligned[phase] += tokens
                        else:
                            state.waste_tokens += tokens
                            state.phase_waste[phase] += tokens
                            if waste_type:
                                state.waste_by_type[waste_type] = (
                                    state.waste_by_type.get(waste_type, 0) + tokens
                                )
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
            if block_type == "thinking":
                text = block.get("thinking", "")
            elif block_type == "tool_use":
                tool_name = block.get("name", "")
                tool_input = json.dumps(block.get("input", {}), sort_keys=True)
                text = f"{tool_name} {tool_input}"
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

            if state.intent_embedding is not None:
                if embed_fn is not None:
                    span_emb = embed_fn(text, normalize_embeddings=True).astype(
                        np.float32
                    )
                else:
                    span_emb = _embed_text_fast(text)
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
            else:
                state.waste_tokens += tokens
                state.phase_waste[phase] += tokens
                if waste_type:
                    state.waste_by_type[waste_type] = (
                        state.waste_by_type.get(waste_type, 0) + tokens
                    )
                # Track assistant-only waste for cost calculation (matches post-hoc)
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
