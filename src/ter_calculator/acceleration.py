"""Value Stream acceleration module -- reducing lead time for TER analysis.

Provides mechanisms to dramatically reduce analysis time:

- AnalysisCache: incremental cache for intermediate pipeline results so
  re-analysis with different thresholds skips expensive steps.
- QuickAnalyser: fast approximate TER using keyword heuristics instead of
  embedding, targeting ~1-2 seconds for any session size.
- SessionWatcher: polling-based directory watcher that auto-analyses new or
  modified JSONL sessions.
- parallel_embed: multiprocessing-based parallel span embedding.
- hash_file: fast content-based SHA-256 hashing for cache keys.

All components use only the Python standard library plus numpy for array
serialization.  No external dependencies (watchdog, etc.) are required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing
import os
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "AnalysisCache",
    "CacheStats",
    "QuickAnalyser",
    "SessionWatcher",
    "WatchEvent",
    "WatchEventType",
    "parallel_embed",
    "hash_file",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ter" / "analysis"
"""Default location for the incremental analysis cache."""

CACHE_VERSION = 1
"""Version tag stored in sidecar metadata -- used for invalidation on format changes."""

DEFAULT_TTL_HOURS = 168
"""Default time-to-live for cache entries (7 days)."""

DEFAULT_WATCH_INTERVAL = 30
"""Default polling interval in seconds for SessionWatcher."""

DEFAULT_WATCH_DIR = Path.home() / ".claude" / "projects"
"""Default directory to watch for new/modified JSONL session files."""

EMBEDDING_DIM = 384
"""Dimensionality of the all-MiniLM-L6-v2 embeddings."""

# QuickAnalyser default thresholds -- mirrors classifier.py defaults.
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "similarity_threshold": 0.40,
    "confidence_threshold": 0.75,
    "restatement_threshold": 0.85,
}

# Phase weights -- mirrors compute.py / models.py defaults.
_PHASE_WEIGHTS: dict[str, float] = {
    "reasoning": 0.3,
    "tool_use": 0.4,
    "generation": 0.3,
}

# Minimum keyword length for TF-based extraction.
_MIN_KEYWORD_LEN = 3

# Stop words removed during keyword extraction (common English).
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "shall", "may", "might", "can", "must", "need",
    "and", "but", "or", "nor", "not", "no", "if", "then", "else",
    "for", "of", "in", "on", "at", "to", "from", "by", "with",
    "this", "that", "these", "those", "it", "its", "my", "your",
    "his", "her", "our", "their", "what", "which", "who", "whom",
    "how", "when", "where", "why", "all", "each", "every", "any",
    "some", "such", "than", "too", "very", "just", "also", "only",
    "so", "up", "out", "about", "into", "over", "after", "before",
})


# ---------------------------------------------------------------------------
# File hashing utility
# ---------------------------------------------------------------------------


def hash_file(path: str | Path, *, chunk_size: int = 65536) -> str:
    """Compute a SHA-256 hex digest of a file's contents.

    Uses chunked reading so large files are hashed without loading them
    entirely into memory.

    Parameters
    ----------
    path:
        Filesystem path to the file.
    chunk_size:
        Number of bytes to read per chunk (default 64 KiB).

    Returns
    -------
    str
        Hex digest of the file contents.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CacheStats dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Statistics about the analysis cache."""

    hit_count: int
    miss_count: int
    total_size_bytes: int
    entry_count: int
    oldest_entry_age_hours: float


# ---------------------------------------------------------------------------
# AnalysisCache
# ---------------------------------------------------------------------------


class AnalysisCache:
    """Incremental analysis cache for intermediate pipeline results.

    Stores pickled Python objects alongside a JSON metadata sidecar so that
    expensive steps (parsing, span segmentation, embeddings, intent vectors)
    can be skipped when re-analysing with different thresholds.

    Cache layout::

        ~/.cache/ter/analysis/
            <key-prefix>/
                <key>.pkl      -- pickled artifact
                <key>.meta     -- JSON sidecar {timestamp, ttl, key, version}

    Parameters
    ----------
    cache_dir:
        Directory to store cached artifacts.  Created on first write if it
        does not exist.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hit_count: int = 0
        self._miss_count: int = 0

    # -- public API ----------------------------------------------------------

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ) -> Any:
        """Return a cached value for *key*, or call *compute_fn* to produce it.

        If the cached entry exists and has not expired, it is returned
        directly.  Otherwise *compute_fn* is called, the result is cached,
        and the value is returned.

        Parameters
        ----------
        key:
            Cache key -- typically a content hash or composite hash.
        compute_fn:
            Zero-argument callable that produces the value to cache.
        ttl_hours:
            Time-to-live for the cache entry in hours.

        Returns
        -------
        Any
            The cached or freshly computed value.
        """
        cached = self._read(key, ttl_hours)
        if cached is not None:
            self._hit_count += 1
            logger.debug("Cache HIT for key %s", key[:16])
            return cached

        self._miss_count += 1
        logger.debug("Cache MISS for key %s -- computing", key[:16])
        value = compute_fn()
        self._write(key, value, ttl_hours)
        return value

    def invalidate(self, session_path: str) -> None:
        """Clear all cache entries whose key matches *session_path*.

        Since cache keys are typically derived from the file's content hash,
        this method computes the hash of the file at *session_path* and
        removes any entries that start with that hash.  It also scans sidecar
        metadata for entries referencing the path directly.

        Parameters
        ----------
        session_path:
            Filesystem path to the session file to invalidate.
        """
        target_path = Path(session_path)
        removed = 0

        # Strategy 1: remove by content hash of the file (if it still exists).
        if target_path.exists():
            file_hash = hash_file(target_path)
            removed += self._remove_by_prefix(file_hash)

        # Strategy 2: scan sidecar metadata for matching source_path.
        for meta_path in self.cache_dir.rglob("*.meta"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("source_path") == str(target_path):
                    pkl_path = meta_path.with_suffix(".pkl")
                    pkl_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    removed += 1
            except (json.JSONDecodeError, OSError):
                continue

        logger.info("Invalidated %d cache entries for %s", removed, session_path)

    def clear_all(self) -> None:
        """Purge the entire cache directory."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hit_count = 0
        self._miss_count = 0
        logger.info("Analysis cache cleared.")

    def cache_stats(self) -> CacheStats:
        """Return statistics about the current cache state.

        Returns
        -------
        CacheStats
            Snapshot of hit/miss counts, total size, entry count, and age
            of the oldest entry.
        """
        total_size = 0
        entry_count = 0
        oldest_timestamp: float | None = None
        now = time.time()

        for pkl_path in self.cache_dir.rglob("*.pkl"):
            total_size += pkl_path.stat().st_size
            entry_count += 1

            meta_path = pkl_path.with_suffix(".meta")
            if meta_path.exists():
                total_size += meta_path.stat().st_size
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    ts = meta.get("timestamp", now)
                    if oldest_timestamp is None or ts < oldest_timestamp:
                        oldest_timestamp = ts
                except (json.JSONDecodeError, OSError):
                    pass

        oldest_age_hours = 0.0
        if oldest_timestamp is not None:
            oldest_age_hours = max(0.0, (now - oldest_timestamp) / 3600.0)

        return CacheStats(
            hit_count=self._hit_count,
            miss_count=self._miss_count,
            total_size_bytes=total_size,
            entry_count=entry_count,
            oldest_entry_age_hours=oldest_age_hours,
        )

    # -- internal helpers ----------------------------------------------------

    def _key_paths(self, key: str) -> tuple[Path, Path]:
        """Return (pkl_path, meta_path) for a given cache key."""
        prefix = key[:2]
        directory = self.cache_dir / prefix
        return directory / f"{key}.pkl", directory / f"{key}.meta"

    def _read(self, key: str, ttl_hours: int) -> Any | None:
        """Attempt to read a valid (non-expired) cache entry."""
        pkl_path, meta_path = self._key_paths(key)

        if not pkl_path.exists() or not meta_path.exists():
            return None

        # Check metadata for TTL expiry and version compatibility.
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.debug("Corrupt metadata for key %s -- treating as miss", key[:16])
            return None

        if meta.get("version", 0) != CACHE_VERSION:
            logger.debug("Version mismatch for key %s -- treating as miss", key[:16])
            return None

        stored_ts = meta.get("timestamp", 0)
        age_hours = (time.time() - stored_ts) / 3600.0
        if age_hours > ttl_hours:
            logger.debug("Expired entry for key %s (%.1fh > %dh)", key[:16], age_hours, ttl_hours)
            pkl_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

        # Deserialise the artifact.
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)  # noqa: S301
        except Exception:
            logger.warning("Failed to unpickle key %s -- treating as miss", key[:16])
            pkl_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

    def _write(self, key: str, value: Any, ttl_hours: int) -> None:
        """Persist a value and its metadata sidecar to disk."""
        pkl_path, meta_path = self._key_paths(key)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the pickled artifact.
        with open(pkl_path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Write the metadata sidecar.
        meta = {
            "key": key,
            "timestamp": time.time(),
            "ttl_hours": ttl_hours,
            "version": CACHE_VERSION,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _remove_by_prefix(self, prefix: str) -> int:
        """Remove all cache entries whose key starts with *prefix*."""
        removed = 0
        prefix_dir = self.cache_dir / prefix[:2]
        if not prefix_dir.exists():
            return 0

        for pkl_path in prefix_dir.glob(f"{prefix}*.pkl"):
            pkl_path.unlink(missing_ok=True)
            pkl_path.with_suffix(".meta").unlink(missing_ok=True)
            removed += 1

        return removed


# ---------------------------------------------------------------------------
# QuickAnalyser
# ---------------------------------------------------------------------------


class QuickAnalyser:
    """Fast approximate TER calculation using keyword heuristics.

    Skips the embedding step entirely, replacing cosine similarity with a
    keyword-overlap ratio.  This makes analysis near-instant (~1-2 seconds)
    at the cost of reduced accuracy.

    The keyword extraction uses simple TF-based scoring with no external
    dependencies:

    1. Tokenise user prompts into words, remove stop words, and count term
       frequencies.
    2. Select top-N keywords by frequency (ties broken alphabetically).
    3. For each span, compute ``keywords_found_in_span / total_keywords``
       as the alignment score (analogous to cosine similarity).
    4. Apply the standard threshold logic to classify spans and compute TER.
    """

    def __init__(self, top_n_keywords: int = 30) -> None:
        self.top_n_keywords = max(1, top_n_keywords)

    def analyse_quick(
        self,
        session_path: str,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run a fast approximate TER analysis on a session file.

        Parameters
        ----------
        session_path:
            Path to a JSONL session file.
        thresholds:
            Override thresholds.  Keys: ``similarity_threshold``,
            ``confidence_threshold``.  Defaults mirror the main classifier.

        Returns
        -------
        dict
            A dictionary matching the TERResult structure with fields:
            ``session_id``, ``aggregate_ter``, ``raw_ratio``,
            ``phase_scores``, ``total_tokens``, ``aligned_tokens``,
            ``waste_tokens``, ``waste_patterns``, ``method``.
        """
        effective_thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
        sim_threshold = effective_thresholds["similarity_threshold"]

        # -- 1. Parse session minimally -----------------------------------
        session_data = self._parse_session(session_path)
        if not session_data["spans"]:
            return self._empty_result(session_data["session_id"])

        # -- 2. Extract keywords from user prompts ------------------------
        keywords = self._extract_keywords(session_data["user_prompts"])
        if not keywords:
            # No meaningful keywords -- treat everything as aligned.
            return self._all_aligned_result(session_data)

        # -- 3. Score each span by keyword overlap ------------------------
        scored_spans: list[dict[str, Any]] = []
        for span in session_data["spans"]:
            score = self._keyword_overlap_score(span["text"], keywords)
            label = "aligned" if score >= sim_threshold else "waste"
            scored_spans.append({**span, "score": score, "label": label})

        # -- 4. Compute per-phase and aggregate TER -----------------------
        return self._compute_result(session_data["session_id"], scored_spans)

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _parse_session(session_path: str) -> dict[str, Any]:
        """Minimal JSONL parsing -- extracts spans and user prompts.

        This is a lightweight parser that avoids importing the full loader
        module, keeping QuickAnalyser self-contained for speed.
        """
        path = Path(session_path)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {session_path}")

        messages: list[dict[str, Any]] = []
        session_id = ""

        # Deduplicate by requestId -- keep highest output_tokens.
        seen_requests: dict[str, dict[str, Any]] = {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not session_id:
                    session_id = entry.get("sessionId", "")

                msg = entry.get("message", {})
                if not msg:
                    continue

                request_id = msg.get("requestId") or entry.get("requestId")
                usage = msg.get("usage", {})
                output_tokens = usage.get("output_tokens", 0) if usage else 0

                if request_id:
                    existing = seen_requests.get(request_id)
                    if existing is None or output_tokens > existing.get("_output_tokens", 0):
                        seen_requests[request_id] = {**entry, "_output_tokens": output_tokens}
                else:
                    messages.append(entry)

        messages.extend(seen_requests.values())
        # Sort by timestamp if available.
        messages.sort(key=lambda m: m.get("timestamp", ""))

        # Extract user prompts and content spans.
        user_prompts: list[str] = []
        spans: list[dict[str, Any]] = []
        position = 0

        for entry in messages:
            msg = entry.get("message", {})
            role = msg.get("role", entry.get("type", ""))
            content = msg.get("content", "")

            # Handle string content.
            if isinstance(content, str) and content.strip():
                if role == "user":
                    user_prompts.append(content)
                else:
                    spans.append({
                        "text": content,
                        "phase": "generation",
                        "position": position,
                        "token_count": max(1, len(content) // 4),
                    })
                    position += 1
                continue

            # Handle block-based content.
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue

                    block_type = block.get("type", "")
                    text = ""

                    if block_type == "text":
                        text = block.get("text", "")
                        phase = "generation"
                    elif block_type == "thinking":
                        text = block.get("thinking", "") or block.get("text", "")
                        phase = "reasoning"
                    elif block_type == "tool_use":
                        tool_input = block.get("input", {})
                        text = json.dumps(tool_input) if tool_input else ""
                        tool_name = block.get("name", "unknown")
                        text = f"{tool_name}: {text}"
                        phase = "tool_use"
                    elif block_type == "tool_result":
                        text = ""
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            text = result_content
                        elif isinstance(result_content, list):
                            text = " ".join(
                                b.get("text", "")
                                for b in result_content
                                if isinstance(b, dict)
                            )
                        phase = "tool_use"
                    else:
                        continue

                    if role == "user" and block_type == "text" and text.strip():
                        user_prompts.append(text)
                        continue

                    if text.strip():
                        spans.append({
                            "text": text,
                            "phase": phase,
                            "position": position,
                            "token_count": max(1, len(text) // 4),
                        })
                        position += 1

        if not session_id:
            session_id = path.stem

        return {
            "session_id": session_id,
            "user_prompts": user_prompts,
            "spans": spans,
        }

    def _extract_keywords(self, prompts: list[str]) -> set[str]:
        """Extract top-N keywords from user prompts using TF scoring.

        Words are lowercased, stop words and very short tokens are removed.
        The most frequent remaining words are selected as keywords.
        """
        if not prompts:
            return set()

        combined = " ".join(prompts)
        # Tokenise: split on non-alphanumeric characters.
        words = [
            w.lower()
            for w in combined.replace("-", " ").replace("_", " ").split()
            if len(w) >= _MIN_KEYWORD_LEN
        ]

        # Remove stop words.
        words = [w for w in words if w not in _STOP_WORDS]

        if not words:
            return set()

        counter = Counter(words)
        # Select top-N by frequency; break ties alphabetically for determinism.
        top = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return {word for word, _ in top[: self.top_n_keywords]}

    @staticmethod
    def _keyword_overlap_score(text: str, keywords: set[str]) -> float:
        """Compute the fraction of keywords found in *text*.

        Returns a float in [0.0, 1.0].
        """
        if not keywords:
            return 0.0

        text_lower = text.lower()
        found = sum(1 for kw in keywords if kw in text_lower)
        return found / len(keywords)

    @staticmethod
    def _compute_result(
        session_id: str,
        scored_spans: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate scored spans into a TERResult-compatible dict."""
        # Per-phase token counts.
        phase_aligned: dict[str, int] = {"reasoning": 0, "tool_use": 0, "generation": 0}
        phase_total: dict[str, int] = {"reasoning": 0, "tool_use": 0, "generation": 0}

        total_tokens = 0
        aligned_tokens = 0

        for span in scored_spans:
            phase = span["phase"]
            tc = span["token_count"]
            total_tokens += tc

            if phase not in phase_total:
                phase_total[phase] = 0
                phase_aligned[phase] = 0

            phase_total[phase] += tc

            if span["label"] == "aligned":
                aligned_tokens += tc
                phase_aligned[phase] += tc

        waste_tokens = total_tokens - aligned_tokens

        # Per-phase scores.
        phase_scores: dict[str, float] = {}
        for phase in ("reasoning", "tool_use", "generation"):
            pt = phase_total.get(phase, 0)
            pa = phase_aligned.get(phase, 0)
            phase_scores[phase] = pa / pt if pt > 0 else 1.0

        # Weighted aggregate TER.
        aggregate_ter = sum(
            _PHASE_WEIGHTS.get(phase, 0.0) * phase_scores[phase]
            for phase in phase_scores
        )

        # Raw ratio.
        raw_ratio = aligned_tokens / total_tokens if total_tokens > 0 else 1.0

        return {
            "session_id": session_id,
            "aggregate_ter": round(aggregate_ter, 4),
            "raw_ratio": round(raw_ratio, 4),
            "phase_scores": {k: round(v, 4) for k, v in phase_scores.items()},
            "total_tokens": total_tokens,
            "aligned_tokens": aligned_tokens,
            "waste_tokens": waste_tokens,
            "waste_patterns": [],
            "method": "quick_keyword",
        }

    @staticmethod
    def _empty_result(session_id: str) -> dict[str, Any]:
        """Return a TERResult-compatible dict for an empty session."""
        return {
            "session_id": session_id,
            "aggregate_ter": 1.0,
            "raw_ratio": 1.0,
            "phase_scores": {"reasoning": 1.0, "tool_use": 1.0, "generation": 1.0},
            "total_tokens": 0,
            "aligned_tokens": 0,
            "waste_tokens": 0,
            "waste_patterns": [],
            "method": "quick_keyword",
        }

    @staticmethod
    def _all_aligned_result(session_data: dict[str, Any]) -> dict[str, Any]:
        """Return a TERResult treating all tokens as aligned (no keywords)."""
        total = sum(s["token_count"] for s in session_data["spans"])
        return {
            "session_id": session_data["session_id"],
            "aggregate_ter": 1.0,
            "raw_ratio": 1.0,
            "phase_scores": {"reasoning": 1.0, "tool_use": 1.0, "generation": 1.0},
            "total_tokens": total,
            "aligned_tokens": total,
            "waste_tokens": 0,
            "waste_patterns": [],
            "method": "quick_keyword",
        }


# ---------------------------------------------------------------------------
# WatchEvent
# ---------------------------------------------------------------------------


class WatchEventType(Enum):
    """Types of file-system events detected by SessionWatcher."""

    NEW_SESSION = "new_session"
    MODIFIED_SESSION = "modified_session"


@dataclass(frozen=True, slots=True)
class WatchEvent:
    """A detected session file event."""

    event_type: WatchEventType
    file_path: str
    timestamp: float


# ---------------------------------------------------------------------------
# SessionWatcher
# ---------------------------------------------------------------------------


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
                logger.exception(
                    "Analyser failed for %s", event.file_path
                )


# ---------------------------------------------------------------------------
# Parallel span embedding
# ---------------------------------------------------------------------------


def _embed_worker(args: tuple[list[str], str]) -> list[list[float]]:
    """Worker function for multiprocessing-based parallel embedding.

    Each worker independently loads the sentence-transformers model and
    embeds its chunk of texts.  Returns a list of embedding vectors as
    plain Python lists (for pickle-safe IPC).

    Parameters
    ----------
    args:
        Tuple of (texts, model_name).
    """
    texts, model_name = args
    if not texts:
        return []

    import warnings

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    return [emb.tolist() for emb in embeddings]


def parallel_embed(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    n_workers: int | None = None,
) -> list[NDArray[np.float32]]:
    """Embed texts in parallel using multiprocessing.

    Splits *texts* into chunks (one per CPU core, up to a cap) and embeds
    each chunk in a separate process.  Each worker loads the
    sentence-transformers model independently.

    Falls back to single-process embedding if:
    - ``len(texts) < 100`` (overhead not worth it)
    - Multiprocessing fails for any reason

    Parameters
    ----------
    texts:
        List of text strings to embed.
    model_name:
        Sentence-transformers model identifier.
    n_workers:
        Number of worker processes.  Defaults to
        ``min(cpu_count(), 4)`` to avoid over-subscription.

    Returns
    -------
    list[NDArray[np.float32]]
        One embedding vector per input text, in the same order.
    """
    if not texts:
        return []

    # Determine worker count.
    if n_workers is None:
        cpu_count = os.cpu_count() or 1
        n_workers = min(cpu_count, 4)
    n_workers = max(1, n_workers)

    # Fall back to single-process for small inputs.
    if len(texts) < 100 or n_workers <= 1:
        return _embed_single_process(texts, model_name)

    # Split texts into chunks.
    chunk_size = math.ceil(len(texts) / n_workers)
    chunks: list[list[str]] = [
        texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)
    ]

    try:
        with multiprocessing.Pool(processes=min(n_workers, len(chunks))) as pool:
            results = pool.map(
                _embed_worker,
                [(chunk, model_name) for chunk in chunks],
            )

        # Merge results back in order.
        merged: list[NDArray[np.float32]] = []
        for chunk_result in results:
            for vec in chunk_result:
                merged.append(np.array(vec, dtype=np.float32))
        return merged

    except Exception:
        logger.warning(
            "Parallel embedding failed -- falling back to single process.",
            exc_info=True,
        )
        return _embed_single_process(texts, model_name)


def _embed_single_process(
    texts: list[str],
    model_name: str,
) -> list[NDArray[np.float32]]:
    """Single-process fallback for embedding."""
    import warnings

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return [np.array(emb, dtype=np.float32) for emb in embeddings]
