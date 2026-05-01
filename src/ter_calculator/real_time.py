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

    message_count: int = 0
    span_count: int = 0
    recent_ter_values: list[float] = field(default_factory=list)

    intent_embedding: NDArray[np.float32] | None = None
    intent_text: str = ""
    intent_confidence: float = 0.0

    last_request_ids: set[str] = field(default_factory=set)
    last_file_position: int = 0

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
    warnings: list[str] = field(default_factory=list)
    warning_level: WarningLevel = WarningLevel.INFO

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
    msg = line_data.get("message", {})
    return msg.get("requestId") or msg.get("request_id")


def _get_usage(line_data: dict[str, Any]) -> dict[str, int]:
    """Extract token usage dict."""
    msg = line_data.get("message", {})
    return msg.get("usage", {})


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
            usage = _get_usage(line_data)
            prev_output = 0
            if usage.get("output_tokens", 0) > prev_output:
                pass
            continue
        if request_id:
            state.last_request_ids.add(request_id)

        msg = line_data.get("message", {})
        role = msg.get("role", "")
        blocks = _extract_blocks_from_line(line_data)

        if role == "user":
            for block in blocks:
                if block.get("type") == "text" and block.get("text"):
                    user_text = block["text"]
                    if state.intent_text:
                        state.intent_text += " " + user_text
                    else:
                        state.intent_text = user_text
                    if embed_fn is not None:
                        state.intent_embedding = embed_fn(
                            state.intent_text, normalize_embeddings=True
                        ).astype(np.float32)
                    else:
                        state.intent_embedding = _embed_text_fast(state.intent_text)
                    state.intent_confidence = min(
                        1.0, len(state.intent_text.split()) / 20
                    )
            continue

        if role != "assistant":
            continue

        state.message_count += 1
        message_aligned = 0
        message_total = 0

        for block in blocks:
            block_type = block.get("type", "text")
            phase = _BLOCK_TYPE_TO_PHASE.get(block_type, "generation")

            text = block.get("text", "")
            if block_type == "tool_use":
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
                is_aligned = sim >= SIM_THRESHOLD
            else:
                is_aligned = True

            if is_aligned:
                state.aligned_tokens += tokens
                state.phase_aligned[phase] += tokens
                message_aligned += tokens
            else:
                state.waste_tokens += tokens

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

        signal = TERSignal(
            session_id=line_data.get("sessionId", "unknown"),
            timestamp=time.time(),
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
    ) -> None:
        self.path = Path(path)
        self.poll_interval = poll_interval
        self.model = model
        self.on_signal = on_signal
        self.state = RollingTERState()
        self._stop = False
        self._lines_read = 0

    def _read_new_lines(self) -> list[dict[str, Any]]:
        """Read lines added since last poll."""
        if not self.path.exists():
            return []
        new_lines: list[dict[str, Any]] = []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for i, raw in enumerate(fh):
                    if i < self._lines_read:
                        continue
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        new_lines.append(json.loads(raw))
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed JSONL line %d", i)
                self._lines_read = i + 1 if new_lines else self._lines_read
        except OSError as exc:
            logger.warning("Could not read %s: %s", self.path, exc)
        return new_lines

    def poll_once(self) -> list[TERSignal]:
        """Check for new lines and return any signals produced."""
        new_lines = self._read_new_lines()
        if not new_lines:
            return []
        signals = compute_rolling_ter(self.state, new_lines, model=self.model)
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
    ) -> None:
        self.project_dir = Path(project_dir)
        self.poll_interval = poll_interval
        self.model = model
        self.on_signal = on_signal
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
