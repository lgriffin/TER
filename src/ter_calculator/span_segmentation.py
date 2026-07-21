"""Fine-grained, provenance-preserving text span segmentation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .embedding_cache import estimate_tokens

_PARAGRAPH_BOUNDARY_RE = re.compile(r"\n\s*\n+")
_HEADING_RE = re.compile(r"(?m)^(?:#{1,6}\s+.+|[-*_]{3,})\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+(?=[A-Z0-9#*`\[])")
_DISCOURSE_RE = re.compile(
    r"(?i)(?=\b(?:now|next|again|in summary|to summarize|let me reconsider|"
    r"as mentioned earlier|however|therefore|finally)\b)"
)


@dataclass(frozen=True)
class SegmentationConfig:
    """Controls fine segmentation without changing legacy block behavior."""

    enabled: bool = False
    min_tokens: int = 12
    max_tokens: int = 180

    def __post_init__(self) -> None:
        if self.min_tokens < 1:
            raise ValueError("min_tokens must be at least 1")
        if self.max_tokens < self.min_tokens:
            raise ValueError("max_tokens must be greater than or equal to min_tokens")


@dataclass(frozen=True)
class TextSegment:
    """A segment and its character offsets in the parent block."""

    text: str
    char_start: int
    char_end: int


def segment_text(text: str, config: SegmentationConfig) -> list[TextSegment]:
    """Split text into coherent units while retaining exact source offsets.

    Paragraphs, Markdown headings, sentence groups, and common discourse
    transitions are candidate boundaries. Tiny adjacent units are merged and
    oversized units are divided into sentence-sized groups.
    """
    if not text:
        return []
    if not config.enabled:
        return [TextSegment(text=text, char_start=0, char_end=len(text))]

    boundaries = {0, len(text)}
    for pattern in (_PARAGRAPH_BOUNDARY_RE, _HEADING_RE, _DISCOURSE_RE):
        for match in pattern.finditer(text):
            boundaries.add(match.start())
            boundaries.add(match.end())

    coarse = _segments_from_boundaries(text, sorted(boundaries))
    expanded: list[TextSegment] = []
    for segment in coarse:
        if estimate_tokens(segment.text) <= config.max_tokens:
            expanded.append(segment)
        else:
            expanded.extend(_split_oversized(text, segment, config.max_tokens))

    return _merge_small(text, expanded, config.min_tokens, config.max_tokens)


def _segments_from_boundaries(text: str, boundaries: list[int]) -> list[TextSegment]:
    result: list[TextSegment] = []
    for start, end in zip(boundaries, boundaries[1:]):
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            continue
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        actual_start = start + left
        actual_end = start + right
        result.append(
            TextSegment(
                text=text[actual_start:actual_end],
                char_start=actual_start,
                char_end=actual_end,
            )
        )
    return result


def _split_oversized(
    text: str, segment: TextSegment, max_tokens: int
) -> list[TextSegment]:
    local_boundaries = {0, len(segment.text)}
    for match in _SENTENCE_BOUNDARY_RE.finditer(segment.text):
        local_boundaries.add(match.start())
        local_boundaries.add(match.end())
    pieces = _segments_from_boundaries(segment.text, sorted(local_boundaries))
    if len(pieces) == 1:
        return _split_by_words(text, segment, max_tokens)

    result: list[TextSegment] = []
    current_start: int | None = None
    current_end: int | None = None
    for piece in pieces:
        global_start = segment.char_start + piece.char_start
        global_end = segment.char_start + piece.char_end
        candidate_start = global_start if current_start is None else current_start
        candidate = text[candidate_start:global_end].strip()
        if current_start is not None and estimate_tokens(candidate) > max_tokens:
            assert current_end is not None
            result.append(
                TextSegment(
                    text=text[current_start:current_end],
                    char_start=current_start,
                    char_end=current_end,
                )
            )
            current_start = global_start
        elif current_start is None:
            current_start = global_start
        current_end = global_end
    if current_start is not None and current_end is not None:
        result.append(
            TextSegment(
                text=text[current_start:current_end],
                char_start=current_start,
                char_end=current_end,
            )
        )
    return result


def _split_by_words(
    text: str, segment: TextSegment, max_tokens: int
) -> list[TextSegment]:
    matches = list(re.finditer(r"\S+", segment.text))
    if not matches:
        return []
    result: list[TextSegment] = []
    start_idx = 0
    while start_idx < len(matches):
        end_idx = start_idx + 1
        while end_idx < len(matches):
            local_start = matches[start_idx].start()
            local_end = matches[end_idx].end()
            if estimate_tokens(segment.text[local_start:local_end]) > max_tokens:
                break
            end_idx += 1
        chosen_end = max(start_idx + 1, end_idx)
        first = matches[start_idx]
        last = matches[chosen_end - 1]
        global_start = segment.char_start + first.start()
        global_end = segment.char_start + last.end()
        result.append(
            TextSegment(
                text=text[global_start:global_end],
                char_start=global_start,
                char_end=global_end,
            )
        )
        start_idx = chosen_end
    return result


def _merge_small(
    text: str, segments: list[TextSegment], min_tokens: int, max_tokens: int
) -> list[TextSegment]:
    if not segments:
        return []
    merged: list[TextSegment] = []
    for segment in segments:
        if merged and estimate_tokens(segment.text) < min_tokens:
            prior = merged[-1]
            candidate = text[prior.char_start : segment.char_end].strip()
            if estimate_tokens(candidate) <= max_tokens:
                merged[-1] = TextSegment(candidate, prior.char_start, segment.char_end)
                continue
        merged.append(segment)

    if len(merged) > 1 and estimate_tokens(merged[0].text) < min_tokens:
        first, second = merged[0], merged[1]
        candidate = text[first.char_start : second.char_end].strip()
        if estimate_tokens(candidate) <= max_tokens:
            merged[:2] = [TextSegment(candidate, first.char_start, second.char_end)]
    return merged
