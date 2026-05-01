"""Embedding cache and performance optimisations for span embedding.

Addresses the Performance — Span Embedding Bottleneck identified in the
architecture analysis. For a 100k-token session with 1,500-2,000 spans,
raw embedding takes 10-30s. This module provides:

- Batch embedding with span merging (reduce vector count)
- Disk-based embedding cache (re-analysis is near-instant)
- Lazy embedding filter (skip very short spans)
- GPU detection and configuration
- Optimal batch-size processing
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sentence_transformers import SentenceTransformer

    from ter_calculator.models import SpanPhase, TokenSpan

__all__ = [
    "MergedSpan",
    "SkippedSpanResult",
    "EmbeddingCache",
    "DeviceConfig",
    "merge_adjacent_spans",
    "filter_short_spans",
    "detect_device",
    "configure_model_device",
    "compute_batch_embeddings",
    "embed_spans",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MIN_TOKEN_COUNT = 10
"""Spans shorter than this are skipped by the lazy embedding filter."""

DEFAULT_BATCH_SIZE = 64
"""Default number of texts to embed in a single model.encode() call."""

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ter" / "embeddings"
"""Default location for the on-disk embedding cache."""

EMBEDDING_DIM = 384
"""Dimensionality of the all-MiniLM-L6-v2 embeddings."""


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MergedSpan:
    """A group of consecutive same-phase spans merged into one embedding unit.

    The merged text is the concatenation of the constituent spans' text,
    separated by a single space.  After embedding, the resulting vector is
    written back to every constituent :pyclass:`TokenSpan`.
    """

    text: str
    phase: str  # SpanPhase value name — avoids hard import
    start_position: int
    end_position: int
    total_token_count: int
    source_indices: tuple[int, ...]
    """Indices into the original span list so embeddings can be written back."""


@dataclass(frozen=True, slots=True)
class SkippedSpanResult:
    """Default low-confidence classification returned for spans too short to embed."""

    span_index: int
    default_confidence: float = 0.1
    default_label: str = "low_signal"
    reason: str = "below_min_token_count"


@dataclass(slots=True)
class DeviceConfig:
    """Detected compute device for sentence-transformers."""

    device: str  # "cuda", "mps", or "cpu"
    device_name: str  # human-readable description
    batch_size_hint: int  # suggested batch size for this device


# ---------------------------------------------------------------------------
# 1. Batch embedding with span merging
# ---------------------------------------------------------------------------


def merge_adjacent_spans(
    spans: list[TokenSpan],
) -> list[MergedSpan]:
    """Merge consecutive spans that share the same phase into larger chunks.

    This reduces the number of embedding calls — for example, five consecutive
    ``reasoning`` spans become a single embedding input, cutting the vector
    count by ~5x in the best case.

    Parameters
    ----------
    spans:
        Ordered list of :class:`TokenSpan` objects from session segmentation.

    Returns
    -------
    list[MergedSpan]:
        Merged spans ready for batch embedding.  Each records the indices of
        the original spans it was built from.
    """
    if not spans:
        return []

    merged: list[MergedSpan] = []
    group_texts: list[str] = [spans[0].text]
    group_indices: list[int] = [0]
    group_token_count: int = spans[0].token_count
    current_phase: str = spans[0].phase.value if hasattr(spans[0].phase, "value") else str(spans[0].phase)

    for i in range(1, len(spans)):
        span = spans[i]
        phase_val = span.phase.value if hasattr(span.phase, "value") else str(span.phase)

        if phase_val == current_phase:
            # Same phase — accumulate into the current group.
            group_texts.append(span.text)
            group_indices.append(i)
            group_token_count += span.token_count
        else:
            # Phase boundary — flush the current group.
            merged.append(
                MergedSpan(
                    text=" ".join(group_texts),
                    phase=current_phase,
                    start_position=spans[group_indices[0]].position,
                    end_position=spans[group_indices[-1]].position,
                    total_token_count=group_token_count,
                    source_indices=tuple(group_indices),
                )
            )
            # Start a new group.
            group_texts = [span.text]
            group_indices = [i]
            group_token_count = span.token_count
            current_phase = phase_val

    # Flush the final group.
    merged.append(
        MergedSpan(
            text=" ".join(group_texts),
            phase=current_phase,
            start_position=spans[group_indices[0]].position,
            end_position=spans[group_indices[-1]].position,
            total_token_count=group_token_count,
            source_indices=tuple(group_indices),
        )
    )
    return merged


# ---------------------------------------------------------------------------
# 2. Embedding cache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Disk-based cache for span embeddings, keyed by content hash.

    Embeddings are stored as numpy ``.npy`` files under
    ``~/.cache/ter/embeddings/<hash_prefix>/<hash>.npy``.  A companion
    ``index.json`` maps hashes to metadata for diagnostics.

    The cache uses SHA-256 of the span text to derive the key, ensuring
    that identical text always resolves to the same cached vector regardless
    of span position or session.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / "index.json"
        self._index: dict[str, dict] = self._load_index()

    # -- public API ----------------------------------------------------------

    @staticmethod
    def content_hash(text: str) -> str:
        """Return the SHA-256 hex digest for *text*."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> NDArray[np.float32] | None:
        """Look up a cached embedding by text content.

        Returns the embedding vector if found, ``None`` otherwise.
        """
        h = self.content_hash(text)
        npy_path = self._npy_path(h)
        if npy_path.exists():
            try:
                vec = np.load(npy_path)
                logger.debug("Cache HIT for hash %s…", h[:12])
                return vec
            except Exception:
                logger.warning("Corrupt cache entry %s — will re-embed.", h[:12])
                npy_path.unlink(missing_ok=True)
        return None

    def put(self, text: str, embedding: NDArray[np.float32]) -> None:
        """Store an embedding in the cache."""
        h = self.content_hash(text)
        npy_path = self._npy_path(h)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, embedding.astype(np.float32))
        self._index[h] = {"dim": embedding.shape[0], "text_len": len(text)}
        # Persist index periodically — caller should call flush() at the end.

    def get_many(self, texts: list[str]) -> dict[str, NDArray[np.float32] | None]:
        """Batch lookup.  Returns a dict mapping each text to its cached
        embedding (or ``None`` if not cached)."""
        return {t: self.get(t) for t in texts}

    def put_many(
        self,
        texts: list[str],
        embeddings: NDArray[np.float32],
    ) -> None:
        """Batch store.  ``embeddings`` must have shape ``(len(texts), dim)``."""
        for i, text in enumerate(texts):
            self.put(text, embeddings[i])

    def flush(self) -> None:
        """Persist the in-memory index to disk."""
        try:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f)
        except OSError:
            logger.warning("Failed to write embedding cache index.")

    def clear(self) -> None:
        """Remove all cached embeddings."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = {}

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._index)

    # -- internals -----------------------------------------------------------

    def _npy_path(self, hex_hash: str) -> Path:
        """Two-level fan-out: ``<first-2-chars>/<full-hash>.npy``."""
        return self.cache_dir / hex_hash[:2] / f"{hex_hash}.npy"

    def _load_index(self) -> dict[str, dict]:
        if self._index_path.exists():
            try:
                with open(self._index_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt cache index — starting fresh.")
        return {}


# ---------------------------------------------------------------------------
# 3. Lazy embedding filter
# ---------------------------------------------------------------------------


def filter_short_spans(
    spans: list[TokenSpan],
    min_token_count: int = DEFAULT_MIN_TOKEN_COUNT,
) -> tuple[list[TokenSpan], list[SkippedSpanResult]]:
    """Partition *spans* into those worth embedding and those to skip.

    Spans with ``token_count < min_token_count`` are skipped and receive a
    default low-confidence :class:`SkippedSpanResult` instead of an expensive
    embedding call.

    Parameters
    ----------
    spans:
        The full list of :class:`TokenSpan` objects.
    min_token_count:
        Minimum token count to be considered worth embedding.

    Returns
    -------
    (embeddable, skipped):
        ``embeddable`` — spans that should be embedded.
        ``skipped`` — :class:`SkippedSpanResult` for spans that were filtered out.
    """
    embeddable: list[TokenSpan] = []
    skipped: list[SkippedSpanResult] = []

    for idx, span in enumerate(spans):
        if span.token_count < min_token_count:
            skipped.append(SkippedSpanResult(span_index=idx))
            logger.debug(
                "Skipping span %d (%d tokens < %d minimum)",
                idx,
                span.token_count,
                min_token_count,
            )
        else:
            embeddable.append(span)

    logger.info(
        "Lazy filter: %d embeddable, %d skipped (min_token_count=%d)",
        len(embeddable),
        len(skipped),
        min_token_count,
    )
    return embeddable, skipped


# ---------------------------------------------------------------------------
# 4. GPU detection
# ---------------------------------------------------------------------------


def detect_device() -> DeviceConfig:
    """Detect the best available compute device for sentence-transformers.

    Checks for CUDA and Apple MPS in order of preference, falling back to
    CPU.  Returns a :class:`DeviceConfig` with a recommended batch size.
    """
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("CUDA device detected: %s", name)
            return DeviceConfig(
                device="cuda",
                device_name=name,
                batch_size_hint=256,
            )

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device detected.")
            return DeviceConfig(
                device="mps",
                device_name="Apple MPS",
                batch_size_hint=128,
            )
    except ImportError:
        logger.debug("PyTorch not installed — falling back to CPU.")

    logger.info("Using CPU for embeddings.")
    return DeviceConfig(
        device="cpu",
        device_name="CPU",
        batch_size_hint=DEFAULT_BATCH_SIZE,
    )


def configure_model_device(
    model: SentenceTransformer,
    device_config: DeviceConfig | None = None,
) -> SentenceTransformer:
    """Move a sentence-transformers model to the best available device.

    If *device_config* is ``None``, :func:`detect_device` is called
    automatically.

    Parameters
    ----------
    model:
        A loaded ``SentenceTransformer`` instance.
    device_config:
        Optional pre-detected device configuration.

    Returns
    -------
    SentenceTransformer:
        The same model, moved to the target device.
    """
    if device_config is None:
        device_config = detect_device()

    logger.info(
        "Configuring model for device=%s (%s)",
        device_config.device,
        device_config.device_name,
    )
    model.to(device_config.device)  # type: ignore[arg-type]
    return model


# ---------------------------------------------------------------------------
# 5. Batch processing
# ---------------------------------------------------------------------------


def compute_batch_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int | None = None,
    device_config: DeviceConfig | None = None,
    show_progress: bool = False,
) -> NDArray[np.float32]:
    """Embed a list of texts in optimal batches.

    Parameters
    ----------
    texts:
        Texts to embed.
    model:
        A loaded ``SentenceTransformer`` model.
    batch_size:
        Number of texts per batch.  If ``None``, uses the hint from
        *device_config* (or the default).
    device_config:
        Optional device config — used only for the batch-size hint when
        *batch_size* is ``None``.
    show_progress:
        Whether to show a progress bar (passed to ``model.encode``).

    Returns
    -------
    NDArray[np.float32]:
        Embedding matrix of shape ``(len(texts), embedding_dim)``.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    if batch_size is None:
        if device_config is not None:
            batch_size = device_config.batch_size_hint
        else:
            batch_size = DEFAULT_BATCH_SIZE

    logger.info(
        "Embedding %d texts in batches of %d",
        len(texts),
        batch_size,
    )

    embeddings: NDArray[np.float32] = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level convenience: embed_spans
# ---------------------------------------------------------------------------


def embed_spans(
    spans: list[TokenSpan],
    model: SentenceTransformer,
    *,
    cache: EmbeddingCache | None = None,
    min_token_count: int = DEFAULT_MIN_TOKEN_COUNT,
    merge: bool = True,
    batch_size: int | None = None,
    device_config: DeviceConfig | None = None,
    show_progress: bool = False,
) -> tuple[list[TokenSpan], list[SkippedSpanResult]]:
    """Full embedding pipeline: filter, merge, cache-check, embed, write back.

    This is the main entry point that combines all optimisations in this module.

    Steps
    -----
    1. Filter out very short spans (lazy embedding).
    2. Merge adjacent same-phase spans (batch merging).
    3. Look up cached embeddings for merged texts.
    4. Embed only the cache-miss texts in optimal batches.
    5. Store new embeddings in the cache.
    6. Write the embedding vector back to every constituent span.

    Parameters
    ----------
    spans:
        The full list of ``TokenSpan`` objects to embed.
    model:
        A loaded ``SentenceTransformer`` model.
    cache:
        Optional :class:`EmbeddingCache`.  A default cache at
        ``~/.cache/ter/embeddings/`` is created if ``None``.
    min_token_count:
        Minimum token count for the lazy filter.
    merge:
        Whether to merge adjacent same-phase spans.
    batch_size:
        Override for the embedding batch size.
    device_config:
        Optional pre-detected device configuration.
    show_progress:
        Whether to show a progress bar during embedding.

    Returns
    -------
    (spans, skipped):
        ``spans`` — the original list with ``.embedding`` populated.
        ``skipped`` — list of :class:`SkippedSpanResult` for filtered spans.
    """
    if cache is None:
        cache = EmbeddingCache()

    # Step 1 — lazy filter
    embeddable, skipped = filter_short_spans(spans, min_token_count)

    # Assign a zero-vector embedding to skipped spans so downstream code
    # always finds an embedding present.
    for sr in skipped:
        spans[sr.span_index].embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    if not embeddable:
        return spans, skipped

    # Step 2 — merge (optional)
    if merge:
        merged = merge_adjacent_spans(embeddable)
    else:
        # Treat each span as its own merged group.
        merged = [
            MergedSpan(
                text=s.text,
                phase=s.phase.value if hasattr(s.phase, "value") else str(s.phase),
                start_position=s.position,
                end_position=s.position,
                total_token_count=s.token_count,
                source_indices=(embeddable.index(s),),
            )
            for s in embeddable
        ]

    # Build a mapping from embeddable index back to original span index.
    embeddable_to_original: dict[int, int] = {}
    emb_idx = 0
    for orig_idx, span in enumerate(spans):
        if span.token_count >= min_token_count:
            embeddable_to_original[emb_idx] = orig_idx
            emb_idx += 1

    # Step 3 — cache lookup
    texts_to_embed: list[str] = []
    texts_to_embed_indices: list[int] = []  # index into merged list
    cached_embeddings: dict[int, NDArray[np.float32]] = {}

    for m_idx, mg in enumerate(merged):
        cached = cache.get(mg.text)
        if cached is not None:
            cached_embeddings[m_idx] = cached
        else:
            texts_to_embed.append(mg.text)
            texts_to_embed_indices.append(m_idx)

    logger.info(
        "Cache: %d hits, %d misses out of %d merged spans",
        len(cached_embeddings),
        len(texts_to_embed),
        len(merged),
    )

    # Step 4 — embed cache misses
    if texts_to_embed:
        new_embeddings = compute_batch_embeddings(
            texts_to_embed,
            model,
            batch_size=batch_size,
            device_config=device_config,
            show_progress=show_progress,
        )
        for i, m_idx in enumerate(texts_to_embed_indices):
            cached_embeddings[m_idx] = new_embeddings[i]

        # Step 5 — store in cache
        cache.put_many(texts_to_embed, new_embeddings)
        cache.flush()

    # Step 6 — write embeddings back to original spans
    for m_idx, mg in enumerate(merged):
        vec = cached_embeddings[m_idx]
        for src_idx in mg.source_indices:
            orig_idx = embeddable_to_original[src_idx]
            spans[orig_idx].embedding = vec

    return spans, skipped
