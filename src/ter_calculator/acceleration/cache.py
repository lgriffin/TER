"""Extracted acceleration responsibility."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .hashing import hash_file

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ter" / "analysis"
CACHE_VERSION = 2
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


@dataclass
class CacheStats:
    """Statistics about the analysis cache."""

    hit_count: int
    miss_count: int
    total_size_bytes: int
    entry_count: int
    oldest_entry_age_hours: float


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

    def _cache_hmac_key(self) -> bytes:
        """Derive a machine-local HMAC key from the cache directory path."""
        raw = f"ter-cache-hmac:{self.cache_dir}".encode()
        return hashlib.sha256(raw).digest()

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
            logger.debug(
                "Expired entry for key %s (%.1fh > %dh)", key[:16], age_hours, ttl_hours
            )
            pkl_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

        # Verify integrity before deserialising.
        try:
            pkl_bytes = pkl_path.read_bytes()
        except OSError:
            logger.warning("Failed to read cache file for key %s", key[:16])
            return None

        expected_hmac = meta.get("hmac")
        if not expected_hmac:
            logger.debug("No HMAC in metadata for key %s -- treating as miss", key[:16])
            pkl_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

        actual_hmac = hmac.new(
            self._cache_hmac_key(), pkl_bytes, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(actual_hmac, expected_hmac):
            logger.warning("HMAC mismatch for key %s -- rejecting", key[:16])
            pkl_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

        try:
            return pickle.loads(pkl_bytes)  # noqa: S301
        except (
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            ImportError,
            IndexError,
            OSError,
        ):
            logger.warning("Failed to unpickle key %s -- treating as miss", key[:16])
            pkl_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

    def _write(self, key: str, value: Any, ttl_hours: int) -> None:
        """Persist a value and its metadata sidecar to disk atomically."""
        import tempfile

        pkl_path, meta_path = self._key_paths(key)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)

        pkl_bytes = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        digest = hmac.new(self._cache_hmac_key(), pkl_bytes, hashlib.sha256).hexdigest()

        meta = {
            "key": key,
            "timestamp": time.time(),
            "ttl_hours": ttl_hours,
            "version": CACHE_VERSION,
            "hmac": digest,
        }
        meta_bytes = json.dumps(meta, indent=2).encode("utf-8")

        # Write both files atomically via temp-then-rename.
        fd, tmp = tempfile.mkstemp(dir=pkl_path.parent)
        try:
            with open(fd, "wb") as f:
                f.write(pkl_bytes)
            Path(tmp).replace(pkl_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

        fd2, tmp2 = tempfile.mkstemp(dir=meta_path.parent)
        try:
            with open(fd2, "wb") as f:
                f.write(meta_bytes)
            Path(tmp2).replace(meta_path)
        except BaseException:
            Path(tmp2).unlink(missing_ok=True)
            raise

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
