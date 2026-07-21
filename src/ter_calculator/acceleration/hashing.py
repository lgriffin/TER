"""Extracted acceleration responsibility."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING


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
