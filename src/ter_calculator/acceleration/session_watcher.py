"""Extracted acceleration responsibility."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ter" / "analysis"
CACHE_VERSION = 1
DEFAULT_TTL_HOURS = 168
DEFAULT_WATCH_INTERVAL = 30
DEFAULT_WATCH_DIR = Path.home() / ".claude" / "projects"
EMBEDDING_DIM = 384
_DEFAULT_THRESHOLDS = {
    "similarity_threshold": 0.40,
    "confidence_threshold": 0.75,
    "restatement_threshold": 0.85,
}
_PHASE_WEIGHTS = {"reasoning": 0.3, "tool_use": 0.4, "generation": 0.3}
_MIN_KEYWORD_LEN = 3
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "shall",
        "may",
        "might",
        "can",
        "must",
        "need",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "no",
        "if",
        "then",
        "else",
        "for",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "by",
        "with",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "all",
        "each",
        "every",
        "any",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "also",
        "only",
        "so",
        "up",
        "out",
        "about",
        "into",
        "over",
        "after",
        "before",
    }
)


class WatchEventType(Enum):
    """Types of file-system events detected by SessionWatcher."""

    NEW_SESSION = "new_session"
    MODIFIED_SESSION = "modified_session"


@dataclass
class WatchEvent:
    """A detected session file event."""

    event_type: WatchEventType
    file_path: str
    timestamp: float


class SessionWatcher:
    """Polling-based watcher for new/modified JSONL session files.

    Monitors a directory (recursively) for ``.jsonl`` files and fires a
    callback when a new file appears or an existing file is modified.

    Uses simple polling with a configurable interval -- no external
    dependencies like ``watchdog`` required.

    Parameters
    ----------
    analyser_fn:
        Optional callable invoked with the file path when a session is
        detected.  If not provided, the default behaviour prints a summary
        line to stdout.
    """

    def __init__(self, analyser_fn: Callable[[str], None] | None = None) -> None:
        self._analyser_fn = analyser_fn
        self._known_files: dict[str, float] = {}
        self._running = False

    def watch(
        self,
        project_path: str | None = None,
        interval: int = DEFAULT_WATCH_INTERVAL,
        callback: Callable[[WatchEvent], None] | None = None,
    ) -> None:
        """Start a blocking watch loop over *project_path*.

        Polls every *interval* seconds for new or modified ``.jsonl`` files.
        Exits gracefully on ``KeyboardInterrupt``.

        Parameters
        ----------
        project_path:
            Directory to watch (recursively).  Defaults to
            ``~/.claude/projects/``.
        interval:
            Seconds between polls.
        callback:
            Optional callback invoked with a :class:`WatchEvent` for each
            detected change.  If not provided, the default behaviour logs
            the event and optionally calls the analyser function.
        """
        watch_dir = Path(project_path) if project_path else DEFAULT_WATCH_DIR

        if not watch_dir.exists():
            logger.warning(
                "Watch directory does not exist: %s -- creating it.", watch_dir
            )
            watch_dir.mkdir(parents=True, exist_ok=True)

        # Build initial file snapshot.
        self._known_files = self._snapshot(watch_dir)
        self._running = True

        logger.info(
            "Watching %s for JSONL sessions (interval=%ds, %d known files)",
            watch_dir,
            interval,
            len(self._known_files),
        )

        try:
            while self._running:
                time.sleep(interval)
                self._poll(watch_dir, callback)
        except KeyboardInterrupt:
            logger.info("Watch interrupted -- shutting down gracefully.")
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the watch loop to stop after the current poll cycle."""
        self._running = False

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _snapshot(directory: Path) -> dict[str, float]:
        """Build a {path: mtime} dict for all JSONL files under *directory*."""
        files: dict[str, float] = {}
        try:
            for p in directory.rglob("*.jsonl"):
                try:
                    files[str(p)] = p.stat().st_mtime
                except OSError:
                    continue
        except OSError:
            logger.warning("Failed to scan directory: %s", directory)
        return files

    def _poll(
        self,
        directory: Path,
        callback: Callable[[WatchEvent], None] | None,
    ) -> None:
        """Compare current snapshot against known files and fire events."""
        current = self._snapshot(directory)
        now = time.time()

        for path, mtime in current.items():
            if path not in self._known_files:
                event = WatchEvent(
                    event_type=WatchEventType.NEW_SESSION,
                    file_path=path,
                    timestamp=now,
                )
                self._handle_event(event, callback)
            elif mtime > self._known_files[path]:
                event = WatchEvent(
                    event_type=WatchEventType.MODIFIED_SESSION,
                    file_path=path,
                    timestamp=now,
                )
                self._handle_event(event, callback)

        self._known_files = current

    def _handle_event(
        self,
        event: WatchEvent,
        callback: Callable[[WatchEvent], None] | None,
    ) -> None:
        """Process a single watch event."""
        logger.info(
            "Detected %s: %s",
            event.event_type.value,
            event.file_path,
        )

        if callback is not None:
            try:
                callback(event)
            except Exception:
                logger.exception("Callback failed for event %s", event)
            return

        # Default behaviour: run the analyser if provided.
        if self._analyser_fn is not None:
            try:
                self._analyser_fn(event.file_path)
            except Exception:
                logger.exception("Analyser failed for %s", event.file_path)
