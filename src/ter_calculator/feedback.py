"""Feedback loop module for TER analysis.

Provides prompt improvement hints, historical trending, session tagging,
and CI threshold checking to close the loop between TER analysis and
actionable improvement.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ter_calculator.models import TERResult

__all__ = [
    "CheckResult",
    "PromptHint",
    "TagStats",
    "TERHistory",
    "TERHistoryEntry",
    "TrendDirection",
    "TrendSummary",
    "check_threshold",
    "generate_prompt_hints",
    "get_stats_by_tag",
    "tag_session",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrendDirection(Enum):
    """Direction of TER trend over recent sessions."""

    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromptHint:
    """An actionable suggestion for improving prompt efficiency."""

    category: str
    suggestion: str
    estimated_impact: str  # "high", "medium", "low"
    related_pattern_type: str | None = None


@dataclass(frozen=True, slots=True)
class TERHistoryEntry:
    """A single recorded TER result in the history file."""

    session_id: str
    timestamp: float
    aggregate_ter: float
    total_tokens: int
    waste_tokens: int
    project_path: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TrendSummary:
    """Aggregated summary of TER trends across sessions."""

    avg_ter: float
    trend_direction: TrendDirection
    session_count: int
    best_ter: float
    worst_ter: float
    tokens_total: int


@dataclass(frozen=True, slots=True)
class TagStats:
    """Aggregate statistics for sessions sharing a tag."""

    tag: str
    session_count: int
    avg_ter: float
    avg_tokens: float
    avg_waste_ratio: float


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of a CI threshold check."""

    passed: bool
    actual_ter: float
    threshold: float
    message: str
    phase_failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt improvement hints
# ---------------------------------------------------------------------------

# Mapping from waste-pattern type names to hint templates.
_WASTE_PATTERN_HINTS: dict[str, tuple[str, str, str]] = {
    # pattern_type_value -> (category, suggestion, impact)
    "reasoning_loop": (
        "reasoning",
        "Try adding 'Be concise in your reasoning' to your prompt",
        "high",
    ),
    "duplicate_tool_call": (
        "tool_use",
        "Consider specifying expected outputs to reduce exploratory tool calls",
        "medium",
    ),
    "context_restatement": (
        "generation",
        "Add 'Do not restate context I already have' to your instructions",
        "medium",
    ),
}

# Phase score thresholds below which hints are generated.
_PHASE_SCORE_HINTS: dict[str, tuple[str, str, str]] = {
    # phase_key -> (category, suggestion, impact)
    "reasoning": (
        "reasoning",
        "Provide more specific requirements to reduce unfocused reasoning",
        "medium",
    ),
    "tool_use": (
        "tool_use",
        "List the specific files/tools needed upfront",
        "high",
    ),
    "generation": (
        "generation",
        "Request concise output format (e.g., 'respond in bullet points')",
        "low",
    ),
}

_LOW_PHASE_THRESHOLD = 0.6


def generate_prompt_hints(result: TERResult) -> list[PromptHint]:
    """Generate actionable prompt-improvement hints from a TER result.

    Analyses waste patterns and per-phase scores to produce specific
    suggestions that the user can apply to future prompts.

    Args:
        result: A computed TERResult with waste_patterns and phase_scores.

    Returns:
        A list of :class:`PromptHint` objects, ordered with higher-impact
        hints first.
    """
    hints: list[PromptHint] = []
    seen_pattern_types: set[str] = set()

    # --- Hints derived from detected waste patterns -----------------------
    for pattern in result.waste_patterns:
        # Normalise: pattern_type may be an enum or a plain string.
        ptype = (
            pattern.pattern_type.value
            if hasattr(pattern.pattern_type, "value")
            else str(pattern.pattern_type)
        )

        if ptype in seen_pattern_types:
            continue
        seen_pattern_types.add(ptype)

        if ptype in _WASTE_PATTERN_HINTS:
            category, suggestion, impact = _WASTE_PATTERN_HINTS[ptype]
            hints.append(
                PromptHint(
                    category=category,
                    suggestion=suggestion,
                    estimated_impact=impact,
                    related_pattern_type=ptype,
                )
            )

    # --- Hints derived from low per-phase scores -------------------------
    for phase_key, (category, suggestion, impact) in _PHASE_SCORE_HINTS.items():
        score = result.phase_scores.get(phase_key)
        if score is not None and score < _LOW_PHASE_THRESHOLD:
            hints.append(
                PromptHint(
                    category=category,
                    suggestion=suggestion,
                    estimated_impact=impact,
                    related_pattern_type=None,
                )
            )

    # Sort so that high-impact hints come first.
    impact_order = {"high": 0, "medium": 1, "low": 2}
    hints.sort(key=lambda h: impact_order.get(h.estimated_impact, 3))

    return hints


# ---------------------------------------------------------------------------
# Historical trending
# ---------------------------------------------------------------------------

_DEFAULT_HISTORY_PATH = Path.home() / ".cache" / "ter" / "history.json"


class TERHistory:
    """Persistent storage and trending analysis for TER results.

    Results are stored as a JSON array in *path* (default
    ``~/.cache/ter/history.json``).  The file is created on first
    :meth:`record` call if it does not exist.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_HISTORY_PATH

    # -- internal helpers --------------------------------------------------

    def _load(self) -> list[dict]:
        """Load history entries from disk."""
        if not self._path.exists():
            return []
        text = self._path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return data

    def _save(self, entries: list[dict]) -> None:
        """Persist entries to disk, creating parent dirs as needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _entry_to_dataclass(raw: dict) -> TERHistoryEntry:
        return TERHistoryEntry(
            session_id=raw["session_id"],
            timestamp=raw["timestamp"],
            aggregate_ter=raw["aggregate_ter"],
            total_tokens=raw["total_tokens"],
            waste_tokens=raw["waste_tokens"],
            project_path=raw["project_path"],
            tags=raw.get("tags", []),
        )

    # -- public API --------------------------------------------------------

    def record(
        self,
        result: TERResult,
        project_path: str,
        tags: list[str] | None = None,
    ) -> TERHistoryEntry:
        """Persist a TER result to the history file.

        Args:
            result: The TER result to record.
            project_path: Filesystem path to the project that was analysed.
            tags: Optional list of human-readable tags (e.g. ``["bug-fix"]``).

        Returns:
            The newly created :class:`TERHistoryEntry`.
        """
        entry = TERHistoryEntry(
            session_id=result.session_id,
            timestamp=time.time(),
            aggregate_ter=result.aggregate_ter,
            total_tokens=result.total_tokens,
            waste_tokens=result.waste_tokens,
            project_path=project_path,
            tags=tags or [],
        )
        entries = self._load()
        entries.append(asdict(entry))
        self._save(entries)
        return entry

    def get_trend(
        self,
        project_path: str | None = None,
        last_n: int = 20,
    ) -> list[TERHistoryEntry]:
        """Retrieve the most recent TER history entries.

        Args:
            project_path: If given, filter to entries matching this path.
            last_n: Maximum number of entries to return (most recent first).

        Returns:
            A list of :class:`TERHistoryEntry` ordered newest-first,
            up to *last_n* entries.
        """
        entries = self._load()
        if project_path is not None:
            entries = [e for e in entries if e.get("project_path") == project_path]
        # Sort by timestamp descending, take last_n.
        entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return [self._entry_to_dataclass(e) for e in entries[:last_n]]

    def get_summary(self, project_path: str | None = None) -> TrendSummary:
        """Compute an aggregated trend summary.

        Args:
            project_path: If given, restrict the summary to this project.

        Returns:
            A :class:`TrendSummary` with averages, extremes, and trend
            direction.

        Raises:
            ValueError: If there are no history entries to summarise.
        """
        all_entries = self._load()
        if project_path is not None:
            all_entries = [
                e for e in all_entries if e.get("project_path") == project_path
            ]
        if not all_entries:
            raise ValueError("No history entries to summarise")

        all_entries.sort(key=lambda e: e.get("timestamp", 0))

        ters = [e["aggregate_ter"] for e in all_entries]
        total_tokens = sum(e["total_tokens"] for e in all_entries)
        avg_ter = sum(ters) / len(ters)
        best_ter = max(ters)
        worst_ter = min(ters)

        # Determine trend direction using a simple comparison of the first
        # and second halves of the (time-ordered) TER list.
        trend_direction = _compute_trend_direction(ters)

        return TrendSummary(
            avg_ter=avg_ter,
            trend_direction=trend_direction,
            session_count=len(all_entries),
            best_ter=best_ter,
            worst_ter=worst_ter,
            tokens_total=total_tokens,
        )


def _compute_trend_direction(
    ters: list[float],
    stability_threshold: float = 0.02,
) -> TrendDirection:
    """Determine trend direction from a time-ordered list of TER scores.

    Splits the list into two halves and compares their averages.  A
    difference smaller than *stability_threshold* is treated as stable.
    """
    if len(ters) < 2:
        return TrendDirection.STABLE

    mid = len(ters) // 2
    first_half_avg = sum(ters[:mid]) / mid
    second_half_avg = sum(ters[mid:]) / (len(ters) - mid)

    diff = second_half_avg - first_half_avg
    if diff > stability_threshold:
        return TrendDirection.IMPROVING
    if diff < -stability_threshold:
        return TrendDirection.DECLINING
    return TrendDirection.STABLE


# ---------------------------------------------------------------------------
# Session tagging
# ---------------------------------------------------------------------------


def tag_session(
    session_id: str,
    tags: list[str],
    history_path: Path | None = None,
) -> None:
    """Attach tags to a previously recorded session.

    Tags are *merged* with any existing tags for the session (duplicates
    are removed).

    Args:
        session_id: The session to tag.
        tags: A list of human-readable tag strings (e.g. ``["bug-fix"]``).
        history_path: Override the default history file location.

    Raises:
        ValueError: If the session_id is not found in the history.
    """
    history = TERHistory(path=history_path)
    entries = history._load()

    found = False
    for entry in entries:
        if entry.get("session_id") == session_id:
            existing = set(entry.get("tags", []))
            existing.update(tags)
            entry["tags"] = sorted(existing)
            found = True

    if not found:
        raise ValueError(f"Session '{session_id}' not found in history")

    history._save(entries)


def get_stats_by_tag(
    tag: str,
    history_path: Path | None = None,
) -> TagStats:
    """Compute aggregate statistics for all sessions with a given tag.

    Args:
        tag: The tag to filter by.
        history_path: Override the default history file location.

    Returns:
        A :class:`TagStats` summarising sessions with this tag.

    Raises:
        ValueError: If no sessions match the tag.
    """
    history = TERHistory(path=history_path)
    entries = history._load()
    matching = [e for e in entries if tag in e.get("tags", [])]

    if not matching:
        raise ValueError(f"No sessions found with tag '{tag}'")

    count = len(matching)
    avg_ter = sum(e["aggregate_ter"] for e in matching) / count
    avg_tokens = sum(e["total_tokens"] for e in matching) / count
    total_waste = sum(e["waste_tokens"] for e in matching)
    total_tokens = sum(e["total_tokens"] for e in matching)
    avg_waste_ratio = total_waste / total_tokens if total_tokens > 0 else 0.0

    return TagStats(
        tag=tag,
        session_count=count,
        avg_ter=avg_ter,
        avg_tokens=avg_tokens,
        avg_waste_ratio=avg_waste_ratio,
    )


# ---------------------------------------------------------------------------
# CI integration — threshold checking
# ---------------------------------------------------------------------------


def check_threshold(
    result: TERResult,
    threshold: float,
    phase_threshold: float | None = None,
) -> CheckResult:
    """Check whether a TER result meets a minimum quality threshold.

    Designed to be called from ``ter check --threshold 0.6``.

    Args:
        result: The TER result to evaluate.
        threshold: Minimum acceptable aggregate TER (0.0 -- 1.0).
        phase_threshold: Optional per-phase minimum.  If ``None``, the
            aggregate *threshold* is applied to each phase as well.

    Returns:
        A :class:`CheckResult` indicating pass/fail and any per-phase
        failures.
    """
    per_phase = phase_threshold if phase_threshold is not None else threshold

    phase_failures: list[str] = []
    for phase, score in result.phase_scores.items():
        if score < per_phase:
            phase_failures.append(phase)

    aggregate_passed = result.aggregate_ter >= threshold
    passed = aggregate_passed and len(phase_failures) == 0

    if passed:
        message = (
            f"TER check passed: {result.aggregate_ter:.4f} "
            f">= {threshold:.4f}"
        )
    else:
        parts: list[str] = []
        if not aggregate_passed:
            parts.append(
                f"aggregate TER {result.aggregate_ter:.4f} "
                f"< {threshold:.4f}"
            )
        if phase_failures:
            failures_str = ", ".join(
                f"{p} ({result.phase_scores[p]:.4f})" for p in phase_failures
            )
            parts.append(f"phase failures: {failures_str}")
        message = "TER check failed: " + "; ".join(parts)

    return CheckResult(
        passed=passed,
        actual_ter=result.aggregate_ter,
        threshold=threshold,
        message=message,
        phase_failures=phase_failures,
    )
