"""Extracted acceleration responsibility."""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
