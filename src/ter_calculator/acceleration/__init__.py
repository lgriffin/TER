"""Value-stream acceleration utilities, split by responsibility.

The package preserves the public API of the former ``acceleration.py`` module.
"""

from __future__ import annotations

import multiprocessing  # noqa: F401 — accessed by tests via this module
import threading
import time  # noqa: F401 — accessed by tests via this module

from .hashing import (
    hash_file,
    DEFAULT_CACHE_DIR,
    CACHE_VERSION,
    DEFAULT_TTL_HOURS,
    DEFAULT_WATCH_INTERVAL,
    DEFAULT_WATCH_DIR,
    EMBEDDING_DIM,
)
from .cache import AnalysisCache, CacheStats
from .quick_analyser import QuickAnalyser
from .session_watcher import SessionWatcher, WatchEvent, WatchEventType
from . import parallel as _parallel

_embed_worker = _parallel._embed_worker
_embed_single_process = _parallel._embed_single_process

_parallel_embed_lock = threading.Lock()


def parallel_embed(texts, *, model_name="all-MiniLM-L6-v2", n_workers=None):
    """Compatibility facade that keeps monkeypatchable fallback hooks."""
    with _parallel_embed_lock:
        original = _parallel._embed_single_process
        _parallel._embed_single_process = globals()["_embed_single_process"]
        try:
            return _parallel.parallel_embed(
                texts,
                model_name=model_name,
                n_workers=n_workers,
            )
        finally:
            _parallel._embed_single_process = original


__all__ = [
    "AnalysisCache",
    "CacheStats",
    "QuickAnalyser",
    "SessionWatcher",
    "WatchEvent",
    "WatchEventType",
    "parallel_embed",
    "hash_file",
    "DEFAULT_CACHE_DIR",
    "CACHE_VERSION",
    "DEFAULT_TTL_HOURS",
    "DEFAULT_WATCH_INTERVAL",
    "DEFAULT_WATCH_DIR",
    "EMBEDDING_DIM",
]
