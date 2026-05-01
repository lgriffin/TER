"""Extended waste pattern detectors.

Adds five new waste-detection strategies that go beyond the original three
(reasoning loops, duplicate tool calls, context restatement) shipped in
``waste.py``:

1. **Permission loops** -- tool denied, agent retries the same call.
2. **Error-retry spirals** -- tool errors, agent retries with tiny changes.
3. **Over-reading** -- same file read repeatedly without intervening writes.
4. **Abandoned approaches** -- editing work that is never finished.
5. **Verbose thinking** -- thinking blocks disproportionately large relative
   to the action they precede.

Each detector accepts ``list[ClassifiedSpan]`` and returns
``list[WastePattern]``.  ``detect_all_extended`` runs all five and returns
the combined results.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    pass  # all runtime imports are explicit below

from .models import ClassifiedSpan, SpanPhase, WastePattern

__all__ = [
    "ExtendedWasteType",
    "detect_abandoned_approaches",
    "detect_all_extended",
    "detect_error_retry_spirals",
    "detect_over_reading",
    "detect_permission_loops",
    "detect_verbose_thinking",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extended pattern-type enum
# ---------------------------------------------------------------------------


class ExtendedWasteType(Enum):
    """Waste pattern types introduced by this module."""

    PERMISSION_LOOP = "permission_loop"
    ERROR_RETRY_SPIRAL = "error_retry_spiral"
    OVER_READING = "over_reading"
    ABANDONED_APPROACH = "abandoned_approach"
    VERBOSE_THINKING = "verbose_thinking"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_PERMISSION_KEYWORDS: tuple[str, ...] = (
    "permission denied",
    "not allowed",
    "access denied",
    "eacces",
    "unauthorized",
)

_ERROR_KEYWORDS: tuple[str, ...] = (
    "error",
    "failed",
    "exception",
    "traceback",
)


def _parse_tool_name(span_text: str) -> str:
    """Extract the tool name from a ``tool_use`` span's text.

    By convention the loader serialises tool_use spans as
    ``"ToolName {\"param\":\"val\"}"`` -- the tool name is the first
    whitespace-delimited token.
    """
    return span_text.split(maxsplit=1)[0] if span_text else ""


def _parse_tool_params(span_text: str) -> str:
    """Return the serialised parameters portion of a tool_use span.

    Everything after the first whitespace token is treated as the JSON
    parameters string.  Returns an empty string when no parameters are
    present.
    """
    parts = span_text.split(maxsplit=1)
    return parts[1] if len(parts) > 1 else ""


def _extract_file_path_from_params(params_str: str) -> str | None:
    """Best-effort extraction of ``file_path`` from a JSON params string."""
    if not params_str:
        return None
    try:
        obj = json.loads(params_str)
        if isinstance(obj, dict):
            return obj.get("file_path") or obj.get("path")
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _text_contains_any(text: str, keywords: Sequence[str]) -> bool:
    """Case-insensitive check for any keyword in *text*."""
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _simple_cosine_similarity(a: str, b: str) -> float:
    """Cheap character-trigram cosine similarity between two strings.

    This avoids pulling in numpy / sentence-transformers for a rough
    similarity check on short tool-parameter strings.  Values range
    from 0.0 (no shared trigrams) to 1.0 (identical trigram sets).
    """
    if not a or not b:
        return 0.0 if (a != b) else 1.0

    def _trigrams(s: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in range(len(s) - 2):
            tri = s[i : i + 3]
            counts[tri] = counts.get(tri, 0) + 1
        return counts

    ta = _trigrams(a)
    tb = _trigrams(b)

    all_keys = set(ta) | set(tb)
    if not all_keys:
        return 1.0

    dot = sum(ta.get(k, 0) * tb.get(k, 0) for k in all_keys)
    mag_a = sum(v * v for v in ta.values()) ** 0.5
    mag_b = sum(v * v for v in tb.values()) ** 0.5

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# 1. Permission loop detection
# ---------------------------------------------------------------------------


def detect_permission_loops(
    classified_spans: list[ClassifiedSpan],
    *,
    min_retries: int = 2,
) -> list[WastePattern]:
    """Identify cycles where a tool call is denied and retried unchanged.

    A permission loop is recognised when:

    * A ``tool_use`` span is followed (possibly with intervening reasoning)
      by a ``tool_result`` span whose text contains a permission/denial
      keyword.
    * The agent subsequently issues another ``tool_use`` span with the
      **same tool name** and no meaningful parameter change.
    * This retry cycle occurs *min_retries* or more times.

    Parameters
    ----------
    classified_spans:
        Ordered list of classified spans for the session.
    min_retries:
        Minimum number of denied retries to flag (default 2).

    Returns
    -------
    list[WastePattern]
        Detected permission-loop waste patterns.
    """
    patterns: list[WastePattern] = []

    # Collect tool_use and tool_result spans in order.
    tool_use_spans: list[ClassifiedSpan] = []
    tool_result_spans: list[ClassifiedSpan] = []

    for cs in classified_spans:
        if cs.span.phase == SpanPhase.TOOL_USE:
            if cs.span.block_type == "tool_use":
                tool_use_spans.append(cs)
            elif cs.span.block_type == "tool_result":
                tool_result_spans.append(cs)

    # Build a quick lookup: position -> tool_result span.
    result_by_position: dict[int, ClassifiedSpan] = {
        cs.span.position: cs for cs in tool_result_spans
    }

    # Walk tool_use spans and look for denied-retry chains.
    # Group consecutive tool_use spans by tool name where the preceding
    # result was a denial.
    i = 0
    while i < len(tool_use_spans):
        chain_start = tool_use_spans[i]
        tool_name = _parse_tool_name(chain_start.span.text)
        chain: list[ClassifiedSpan] = [chain_start]

        j = i + 1
        while j < len(tool_use_spans):
            candidate = tool_use_spans[j]
            candidate_name = _parse_tool_name(candidate.span.text)

            if candidate_name != tool_name:
                break

            # Check if there is a denial result between the previous
            # tool_use in the chain and this candidate.
            prev_use = chain[-1]
            denial_found = False
            for pos in range(prev_use.span.position + 1, candidate.span.position):
                result_span = result_by_position.get(pos)
                if result_span is not None and _text_contains_any(
                    result_span.span.text, _PERMISSION_KEYWORDS
                ):
                    denial_found = True
                    break

            if not denial_found:
                break

            chain.append(candidate)
            j += 1

        # The first call is the original attempt; retries start at index 1.
        retries = len(chain) - 1
        if retries >= min_retries:
            wasted = sum(cs.span.token_count for cs in chain[1:])
            patterns.append(WastePattern(
                pattern_type=ExtendedWasteType.PERMISSION_LOOP.value,
                description=(
                    f"Permission loop: '{tool_name}' denied and retried "
                    f"{retries} time(s)"
                ),
                start_position=chain[0].span.position,
                end_position=chain[-1].span.position,
                spans_involved=len(chain),
                tokens_wasted=wasted,
                details={
                    "tool_name": tool_name,
                    "retries": retries,
                },
            ))

        i = j

    return patterns


# ---------------------------------------------------------------------------
# 2. Error-retry spiral detection
# ---------------------------------------------------------------------------


def detect_error_retry_spirals(
    classified_spans: list[ClassifiedSpan],
    *,
    similarity_threshold: float = 0.90,
    min_retries: int = 3,
) -> list[WastePattern]:
    """Detect tool calls that fail and are retried with minimal changes.

    An error-retry spiral is recognised when:

    * A ``tool_use`` span is followed by a ``tool_result`` containing an
      error indicator keyword.
    * The agent retries with a ``tool_use`` span whose serialised
      parameters have cosine similarity > *similarity_threshold* to the
      previous attempt.
    * The cycle repeats *min_retries* or more times.

    Parameters
    ----------
    classified_spans:
        Ordered list of classified spans.
    similarity_threshold:
        Minimum parameter similarity to count as "minimal change"
        (default 0.90).
    min_retries:
        Minimum retry count to flag (default 3).

    Returns
    -------
    list[WastePattern]
        Detected error-retry spiral patterns.
    """
    patterns: list[WastePattern] = []

    tool_use_spans: list[ClassifiedSpan] = [
        cs for cs in classified_spans
        if cs.span.phase == SpanPhase.TOOL_USE and cs.span.block_type == "tool_use"
    ]
    tool_result_spans: list[ClassifiedSpan] = [
        cs for cs in classified_spans
        if cs.span.phase == SpanPhase.TOOL_USE and cs.span.block_type == "tool_result"
    ]

    result_by_position: dict[int, ClassifiedSpan] = {
        cs.span.position: cs for cs in tool_result_spans
    }

    i = 0
    while i < len(tool_use_spans):
        chain: list[ClassifiedSpan] = [tool_use_spans[i]]
        chain_tool_name = _parse_tool_name(tool_use_spans[i].span.text)

        j = i + 1
        while j < len(tool_use_spans):
            prev_use = chain[-1]
            candidate = tool_use_spans[j]

            # Check for an error result between the two calls.
            error_found = False
            for pos in range(prev_use.span.position + 1, candidate.span.position):
                result_span = result_by_position.get(pos)
                if result_span is not None and _text_contains_any(
                    result_span.span.text, _ERROR_KEYWORDS
                ):
                    error_found = True
                    break

            if not error_found:
                break

            # Must be the same tool.
            candidate_name = _parse_tool_name(candidate.span.text)
            if candidate_name != chain_tool_name:
                break

            # Check parameter similarity.
            prev_params = _parse_tool_params(prev_use.span.text)
            cand_params = _parse_tool_params(candidate.span.text)
            sim = _simple_cosine_similarity(prev_params, cand_params)

            if sim < similarity_threshold:
                break

            chain.append(candidate)
            j += 1

        retries = len(chain) - 1
        if retries >= min_retries:
            wasted = sum(cs.span.token_count for cs in chain[1:])
            patterns.append(WastePattern(
                pattern_type=ExtendedWasteType.ERROR_RETRY_SPIRAL.value,
                description=(
                    f"Error-retry spiral: '{chain_tool_name}' failed and "
                    f"retried {retries} time(s) with minimal changes"
                ),
                start_position=chain[0].span.position,
                end_position=chain[-1].span.position,
                spans_involved=len(chain),
                tokens_wasted=wasted,
                details={
                    "tool_name": chain_tool_name,
                    "retries": retries,
                    "similarity_threshold": similarity_threshold,
                },
            ))

        i = j

    return patterns


# ---------------------------------------------------------------------------
# 3. Over-reading detection
# ---------------------------------------------------------------------------


def detect_over_reading(
    classified_spans: list[ClassifiedSpan],
    *,
    min_reads: int = 2,
) -> list[WastePattern]:
    """Flag files read multiple times without an intervening write/edit.

    Over-reading is recognised when:

    * A ``tool_use`` span whose tool name is ``Read`` or ``cat`` targets
      a specific file path.
    * The same file is read again later **without** an intervening
      ``Edit`` or ``Write`` to that file.
    * This occurs *min_reads* or more times for the same file.

    Parameters
    ----------
    classified_spans:
        Ordered list of classified spans.
    min_reads:
        Minimum redundant read count per file to flag (default 2).

    Returns
    -------
    list[WastePattern]
        Detected over-reading patterns.
    """
    patterns: list[WastePattern] = []

    # Collect ordered file operations: (position, op_type, file_path, span).
    # op_type is "read" or "write".
    _READ_TOOLS = {"Read", "cat"}
    _WRITE_TOOLS = {"Edit", "Write"}

    ops: list[tuple[int, str, str, ClassifiedSpan]] = []

    for cs in classified_spans:
        if cs.span.phase != SpanPhase.TOOL_USE or cs.span.block_type != "tool_use":
            continue

        tool_name = _parse_tool_name(cs.span.text)
        params_str = _parse_tool_params(cs.span.text)
        file_path = _extract_file_path_from_params(params_str)

        if file_path is None:
            continue

        if tool_name in _READ_TOOLS:
            ops.append((cs.span.position, "read", file_path, cs))
        elif tool_name in _WRITE_TOOLS:
            ops.append((cs.span.position, "write", file_path, cs))

    # Sort by position (should already be, but be safe).
    ops.sort(key=lambda o: o[0])

    # Track reads per file, resetting when a write is seen.
    # Key: file_path -> list of read spans since last write.
    read_tracker: dict[str, list[ClassifiedSpan]] = {}

    for _pos, op_type, file_path, cs in ops:
        if op_type == "write":
            # A write invalidates the read chain for this file.
            read_tracker.pop(file_path, None)
        elif op_type == "read":
            read_tracker.setdefault(file_path, []).append(cs)

    # Emit patterns for files with excessive reads.
    for file_path, read_spans in read_tracker.items():
        if len(read_spans) < min_reads + 1:
            # The first read is legitimate; we need min_reads *redundant*
            # reads, so total reads must be >= min_reads + 1.
            continue

        redundant = read_spans[1:]  # first read is necessary
        wasted = sum(cs.span.token_count for cs in redundant)
        short_name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        patterns.append(WastePattern(
            pattern_type=ExtendedWasteType.OVER_READING.value,
            description=(
                f"Over-reading: '{short_name}' read {len(read_spans)} "
                f"time(s) without intervening write"
            ),
            start_position=read_spans[0].span.position,
            end_position=read_spans[-1].span.position,
            spans_involved=len(read_spans),
            tokens_wasted=wasted,
            details={
                "file_path": file_path,
                "read_count": len(read_spans),
                "redundant_reads": len(redundant),
            },
        ))

    patterns.sort(key=lambda p: p.tokens_wasted, reverse=True)
    return patterns


# ---------------------------------------------------------------------------
# 4. Abandoned approach detection
# ---------------------------------------------------------------------------


def detect_abandoned_approaches(
    classified_spans: list[ClassifiedSpan],
) -> list[WastePattern]:
    """Identify editing work that is started but never completed.

    An abandoned approach is recognised when:

    * The agent starts editing/writing a file (``Edit`` or ``Write``).
    * It subsequently edits/reads a *different* file.
    * The original file is never touched again for the remainder of the
      session.

    The token span range of the abandoned work is reported.

    Parameters
    ----------
    classified_spans:
        Ordered list of classified spans.

    Returns
    -------
    list[WastePattern]
        Detected abandoned-approach patterns.
    """
    patterns: list[WastePattern] = []

    _WRITE_TOOLS = {"Edit", "Write"}

    # Collect all tool_use spans with file paths.
    file_ops: list[tuple[int, str, str, ClassifiedSpan]] = []
    # (position, tool_name, file_path, span)

    for cs in classified_spans:
        if cs.span.phase != SpanPhase.TOOL_USE or cs.span.block_type != "tool_use":
            continue

        tool_name = _parse_tool_name(cs.span.text)
        params_str = _parse_tool_params(cs.span.text)
        file_path = _extract_file_path_from_params(params_str)

        if file_path is None:
            continue

        file_ops.append((cs.span.position, tool_name, file_path, cs))

    file_ops.sort(key=lambda o: o[0])

    if not file_ops:
        return patterns

    # For each write/edit operation, check if the file is ever touched
    # again later.  If not, and the agent moves on to a *different* file,
    # flag it as abandoned.

    # Build a set of files touched at each position and after.
    # last_touch[file_path] = maximum position at which this file was touched.
    last_touch: dict[str, int] = {}
    for pos, _tool, fp, _cs in file_ops:
        if fp not in last_touch or pos > last_touch[fp]:
            last_touch[fp] = pos

    # Walk write operations and check for abandonment.
    seen_abandoned: set[str] = set()  # avoid duplicate reports per file

    for idx, (pos, tool_name, file_path, cs) in enumerate(file_ops):
        if tool_name not in _WRITE_TOOLS:
            continue

        # Is this the last touch of this file?
        if last_touch.get(file_path, pos) != pos:
            # File is touched again later -- not abandoned.
            continue

        # Check if the agent moves on to a different file after this.
        subsequent_ops = [
            (p, t, fp, s) for p, t, fp, s in file_ops
            if p > pos and fp != file_path
        ]

        if not subsequent_ops:
            # No subsequent work on other files -- not abandoned, just
            # the last operation in the session.
            continue

        # There is subsequent work on other files and this file was never
        # revisited.  Flag as abandoned.
        if file_path in seen_abandoned:
            continue
        seen_abandoned.add(file_path)

        # Gather all spans that worked on this file.
        file_spans = [s for _, _, fp, s in file_ops if fp == file_path]
        wasted = sum(s.span.token_count for s in file_spans)
        short_name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        first_pos = min(s.span.position for s in file_spans)
        last_pos = max(s.span.position for s in file_spans)

        patterns.append(WastePattern(
            pattern_type=ExtendedWasteType.ABANDONED_APPROACH.value,
            description=(
                f"Abandoned approach: '{short_name}' was edited but "
                f"never revisited"
            ),
            start_position=first_pos,
            end_position=last_pos,
            spans_involved=len(file_spans),
            tokens_wasted=wasted,
            details={
                "file_path": file_path,
                "edit_count": len(file_spans),
                "last_edit_position": last_pos,
            },
        ))

    patterns.sort(key=lambda p: p.tokens_wasted, reverse=True)
    return patterns


# ---------------------------------------------------------------------------
# 5. Verbose thinking detection
# ---------------------------------------------------------------------------


def detect_verbose_thinking(
    classified_spans: list[ClassifiedSpan],
    *,
    ratio_threshold: float = 10.0,
    min_thinking_tokens: int = 500,
) -> list[WastePattern]:
    """Flag thinking blocks disproportionately large for their outcome.

    A thinking block is flagged when:

    * Its token count exceeds *min_thinking_tokens*.
    * The ratio ``thinking_tokens / subsequent_action_tokens`` exceeds
      *ratio_threshold*.

    The "subsequent action" is the next non-reasoning span (tool_use or
    generation) that follows the thinking block.

    Parameters
    ----------
    classified_spans:
        Ordered list of classified spans.
    ratio_threshold:
        Maximum acceptable thinking-to-action token ratio (default 10.0).
    min_thinking_tokens:
        Minimum thinking block size to consider (default 500 tokens).

    Returns
    -------
    list[WastePattern]
        Detected verbose-thinking patterns.
    """
    patterns: list[WastePattern] = []

    for i, cs in enumerate(classified_spans):
        if cs.span.phase != SpanPhase.REASONING:
            continue

        thinking_tokens = cs.span.token_count
        if thinking_tokens < min_thinking_tokens:
            continue

        # Find the next non-reasoning span.
        action_span: ClassifiedSpan | None = None
        for j in range(i + 1, len(classified_spans)):
            if classified_spans[j].span.phase != SpanPhase.REASONING:
                action_span = classified_spans[j]
                break

        if action_span is None:
            # No subsequent action -- the thinking block is the last span.
            # Flag it: thinking with no action at all is pure waste.
            patterns.append(WastePattern(
                pattern_type=ExtendedWasteType.VERBOSE_THINKING.value,
                description=(
                    f"Verbose thinking: {thinking_tokens} tokens of "
                    f"reasoning with no subsequent action"
                ),
                start_position=cs.span.position,
                end_position=cs.span.position,
                spans_involved=1,
                tokens_wasted=thinking_tokens,
                details={
                    "thinking_tokens": thinking_tokens,
                    "action_tokens": 0,
                    "ratio": float("inf"),
                },
            ))
            continue

        action_tokens = action_span.span.token_count
        if action_tokens == 0:
            # Avoid division by zero; treat as infinitely verbose.
            ratio = float("inf")
        else:
            ratio = thinking_tokens / action_tokens

        if ratio <= ratio_threshold:
            continue

        # The "excess" tokens are the amount beyond what the threshold
        # would allow.
        excess = thinking_tokens - int(action_tokens * ratio_threshold)
        excess = max(0, excess)

        patterns.append(WastePattern(
            pattern_type=ExtendedWasteType.VERBOSE_THINKING.value,
            description=(
                f"Verbose thinking: {thinking_tokens} tokens of reasoning "
                f"for {action_tokens} tokens of action "
                f"(ratio {ratio:.1f}x, threshold {ratio_threshold:.1f}x)"
            ),
            start_position=cs.span.position,
            end_position=action_span.span.position,
            spans_involved=2,
            tokens_wasted=excess,
            details={
                "thinking_tokens": thinking_tokens,
                "action_tokens": action_tokens,
                "ratio": round(ratio, 2),
                "ratio_threshold": ratio_threshold,
            },
        ))

    return patterns


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------


def detect_all_extended(
    classified_spans: list[ClassifiedSpan],
    *,
    permission_min_retries: int = 2,
    error_similarity_threshold: float = 0.90,
    error_min_retries: int = 3,
    over_reading_min_reads: int = 2,
    verbose_ratio_threshold: float = 10.0,
    verbose_min_thinking_tokens: int = 500,
) -> list[WastePattern]:
    """Run all five extended waste detectors and return combined results.

    Parameters
    ----------
    classified_spans:
        Ordered list of classified spans for the session.
    permission_min_retries:
        Forwarded to :func:`detect_permission_loops`.
    error_similarity_threshold:
        Forwarded to :func:`detect_error_retry_spirals`.
    error_min_retries:
        Forwarded to :func:`detect_error_retry_spirals`.
    over_reading_min_reads:
        Forwarded to :func:`detect_over_reading`.
    verbose_ratio_threshold:
        Forwarded to :func:`detect_verbose_thinking`.
    verbose_min_thinking_tokens:
        Forwarded to :func:`detect_verbose_thinking`.

    Returns
    -------
    list[WastePattern]
        Combined waste patterns from all five detectors, sorted by
        ``start_position``.
    """
    all_patterns: list[WastePattern] = []

    all_patterns.extend(
        detect_permission_loops(
            classified_spans,
            min_retries=permission_min_retries,
        )
    )
    all_patterns.extend(
        detect_error_retry_spirals(
            classified_spans,
            similarity_threshold=error_similarity_threshold,
            min_retries=error_min_retries,
        )
    )
    all_patterns.extend(
        detect_over_reading(
            classified_spans,
            min_reads=over_reading_min_reads,
        )
    )
    all_patterns.extend(
        detect_abandoned_approaches(classified_spans)
    )
    all_patterns.extend(
        detect_verbose_thinking(
            classified_spans,
            ratio_threshold=verbose_ratio_threshold,
            min_thinking_tokens=verbose_min_thinking_tokens,
        )
    )

    all_patterns.sort(key=lambda p: p.start_position)
    return all_patterns
