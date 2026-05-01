"""Input validation for JSONL session data.

Validates JSONL lines, full sessions, and files before expensive analysis.
Produces health reports and completeness assessments so callers can decide
whether to proceed, warn the user, or abort early.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CompletenessAssessment",
    "ContentDistribution",
    "FileValidationResult",
    "HealthReport",
    "SessionValidationResult",
    "ValidationResult",
    "assess_completeness",
    "generate_health_report",
    "validate_jsonl_file",
    "validate_jsonl_line",
    "validate_session",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL_FIELDS = {"type", "uuid", "sessionId", "message"}

_REQUIRED_MESSAGE_FIELDS = {"role", "content"}

# Block types that carry text content.
_TEXT_BLOCK_TYPES = {"text", "thinking"}

# All recognised block types.
_KNOWN_BLOCK_TYPES = {"text", "tool_use", "tool_result", "thinking"}

# Non-message line types that are silently skipped during validation.
# These carry system or infrastructure data, not conversation messages.
_META_LINE_TYPES = {
    "attachment",
    "file-history-snapshot",
    "last-prompt",
    "permission-mode",
    "progress",
    "queue-operation",
    "summary",
    "system",
}

# Rough chars-per-token ratio for quick estimation.
_CHARS_PER_TOKEN = 4.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of validating a single JSONL line."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    line_number: int = 0


@dataclass(frozen=True, slots=True)
class SessionValidationResult:
    """Result of validating a fully parsed session."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    message_count: int = 0
    content_block_count: int = 0


@dataclass(frozen=True, slots=True)
class ContentDistribution:
    """Counts and percentages for each content block type."""

    text_count: int = 0
    tool_use_count: int = 0
    tool_result_count: int = 0
    thinking_count: int = 0
    other_count: int = 0

    @property
    def total(self) -> int:
        return (
            self.text_count
            + self.tool_use_count
            + self.tool_result_count
            + self.thinking_count
            + self.other_count
        )

    @property
    def text_pct(self) -> float:
        return self.text_count / self.total * 100 if self.total else 0.0

    @property
    def tool_use_pct(self) -> float:
        return self.tool_use_count / self.total * 100 if self.total else 0.0

    @property
    def tool_result_pct(self) -> float:
        return self.tool_result_count / self.total * 100 if self.total else 0.0

    @property
    def thinking_pct(self) -> float:
        return self.thinking_count / self.total * 100 if self.total else 0.0

    @property
    def other_pct(self) -> float:
        return self.other_count / self.total * 100 if self.total else 0.0


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Pre-analysis summary of a session's content."""

    user_message_count: int
    assistant_message_count: int
    estimated_total_tokens: int
    content_distribution: ContentDistribution
    reasoning_tokens: int
    tool_use_tokens: int
    generation_tokens: int
    parsing_warnings: list[str] = field(default_factory=list)
    estimated_analysis_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class CompletenessAssessment:
    """Assessment of whether a session appears complete."""

    is_complete: bool
    completeness_score: float  # 0.0 -- 1.0
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FileValidationResult:
    """Result of validating an entire JSONL file."""

    valid: bool
    total_lines: int
    valid_lines: int
    error_lines: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(value: str) -> datetime | None:
    """Try to parse an ISO-8601 timestamp string."""
    if not isinstance(value, str) or not value:
        return None
    try:
        # Python 3.11+ handles trailing "Z" via fromisoformat.
        cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None


def _estimate_tokens(text: str) -> int:
    """Quick character-based token estimation."""
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def _extract_block_text(block: dict[str, Any]) -> str:
    """Return the textual payload of a content block, if any."""
    btype = block.get("type", "")
    if btype in _TEXT_BLOCK_TYPES:
        return block.get("text", "") or block.get("thinking", "") or ""
    if btype == "tool_use":
        # Serialise the input dict for token estimation.
        tool_input = block.get("input")
        if isinstance(tool_input, dict):
            try:
                return json.dumps(tool_input)
            except (TypeError, ValueError):
                return ""
        return str(tool_input) if tool_input else ""
    if btype == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for sub in content:
                if isinstance(sub, dict):
                    parts.append(sub.get("text", "") or sub.get("output", "") or "")
                elif isinstance(sub, str):
                    parts.append(sub)
            return " ".join(parts)
        return ""
    return ""


# ---------------------------------------------------------------------------
# 1. JSONL line validation
# ---------------------------------------------------------------------------


def validate_jsonl_line(raw_line: str, *, line_number: int = 0) -> ValidationResult:
    """Validate a single JSONL line against the expected schema.

    Checks that the line is valid JSON and contains the required top-level
    fields (``type``, ``uuid``, ``sessionId``, ``message``), that
    ``message`` contains ``role`` and ``content``, and that each content
    block has the appropriate type-specific fields.

    Lines with meta types (``permission-mode``, ``file-history-snapshot``,
    ``attachment``, ``summary``) are treated as valid without further
    checks since they do not carry conversation messages.

    Parameters
    ----------
    raw_line:
        The raw text of a single JSONL line.
    line_number:
        The 1-based line number within the file (for error reporting).

    Returns
    -------
    ValidationResult
    """
    errors: list[str] = []
    warnings: list[str] = []

    # -- Parse JSON --------------------------------------------------------
    raw_line = raw_line.strip()
    if not raw_line:
        return ValidationResult(
            valid=True,
            errors=[],
            warnings=["Empty line"],
            line_number=line_number,
        )

    try:
        data: dict[str, Any] = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        return ValidationResult(
            valid=False,
            errors=[f"Line {line_number}: Invalid JSON: {exc}"],
            warnings=[],
            line_number=line_number,
        )

    if not isinstance(data, dict):
        return ValidationResult(
            valid=False,
            errors=[f"Line {line_number}: Expected a JSON object, got {type(data).__name__}"],
            warnings=[],
            line_number=line_number,
        )

    # -- Meta lines are valid but carry no message data --------------------
    line_type = data.get("type", "")
    if line_type in _META_LINE_TYPES:
        return ValidationResult(
            valid=True, errors=[], warnings=[], line_number=line_number
        )

    # -- Required top-level fields -----------------------------------------
    missing_top = _REQUIRED_TOP_LEVEL_FIELDS - data.keys()
    if missing_top:
        errors.append(
            f"Line {line_number}: Missing required top-level fields: "
            + ", ".join(sorted(missing_top))
        )

    # -- Message validation ------------------------------------------------
    message = data.get("message")
    if message is None:
        # Already reported above; nothing more to check.
        if errors:
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                line_number=line_number,
            )
        return ValidationResult(
            valid=True,
            errors=[],
            warnings=warnings,
            line_number=line_number,
        )

    if not isinstance(message, dict):
        errors.append(
            f"Line {line_number}: 'message' must be a dict, got {type(message).__name__}"
        )
        return ValidationResult(
            valid=False,
            errors=errors,
            warnings=warnings,
            line_number=line_number,
        )

    missing_msg = _REQUIRED_MESSAGE_FIELDS - message.keys()
    if missing_msg:
        errors.append(
            f"Line {line_number}: Missing required message fields: "
            + ", ".join(sorted(missing_msg))
        )

    role = message.get("role")
    if role is not None and role not in ("user", "assistant"):
        warnings.append(
            f"Line {line_number}: Unexpected role '{role}' (expected 'user' or 'assistant')"
        )

    # -- Content blocks ----------------------------------------------------
    content = message.get("content")
    if content is None:
        pass  # already reported as missing
    elif isinstance(content, str):
        # Plain-string content is valid (common for user messages).
        pass
    elif isinstance(content, list):
        for idx, block in enumerate(content):
            if not isinstance(block, dict):
                warnings.append(
                    f"Line {line_number}: content[{idx}] is not a dict"
                )
                continue

            btype = block.get("type")
            if btype is None:
                errors.append(
                    f"Line {line_number}: content[{idx}] missing 'type' field"
                )
                continue

            if btype not in _KNOWN_BLOCK_TYPES:
                warnings.append(
                    f"Line {line_number}: content[{idx}] has unknown block type '{btype}'"
                )
                continue

            # Type-specific checks.
            _validate_content_block(
                block, btype, idx, line_number, errors, warnings
            )
    else:
        errors.append(
            f"Line {line_number}: 'content' must be a string or list, "
            f"got {type(content).__name__}"
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        line_number=line_number,
    )


def _validate_content_block(
    block: dict[str, Any],
    btype: str,
    idx: int,
    line_number: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Check type-specific fields within a content block."""
    if btype == "text":
        if "text" not in block:
            errors.append(
                f"Line {line_number}: content[{idx}] (text) missing 'text' field"
            )
        elif not isinstance(block["text"], str):
            errors.append(
                f"Line {line_number}: content[{idx}] 'text' must be a string"
            )
    elif btype == "thinking":
        # Thinking blocks may carry an empty string (v2.1.72+), so we
        # only require the key to exist.
        if "thinking" not in block and "text" not in block:
            warnings.append(
                f"Line {line_number}: content[{idx}] (thinking) missing "
                "'thinking' or 'text' field"
            )
    elif btype == "tool_use":
        if "name" not in block:
            errors.append(
                f"Line {line_number}: content[{idx}] (tool_use) missing 'name' field"
            )
        if "id" not in block:
            errors.append(
                f"Line {line_number}: content[{idx}] (tool_use) missing 'id' field"
            )
    elif btype == "tool_result":
        if "tool_use_id" not in block:
            errors.append(
                f"Line {line_number}: content[{idx}] (tool_result) missing 'tool_use_id' field"
            )


# ---------------------------------------------------------------------------
# 2. Session validation
# ---------------------------------------------------------------------------


def validate_session(
    parsed_lines: list[dict[str, Any]],
) -> SessionValidationResult:
    """Validate a fully parsed session (list of decoded JSONL dicts).

    Checks:
    * At least one user message and one assistant message are present.
    * Timestamps are chronologically ordered (non-decreasing).
    * Every ``tool_result`` references an existing ``tool_use`` id.
    * Orphaned ``tool_use`` blocks without matching ``tool_result`` (warning).
    * Token counts (usage) are non-negative where present.

    Parameters
    ----------
    parsed_lines:
        A list of parsed JSON dicts, each representing one JSONL line from
        the session file.  Non-message lines are silently skipped.

    Returns
    -------
    SessionValidationResult
    """
    errors: list[str] = []
    warnings: list[str] = []
    message_count = 0
    content_block_count = 0
    has_user = False
    has_assistant = False

    timestamps: list[datetime] = []
    tool_use_ids: set[str] = set()
    tool_result_ids: set[str] = set()

    for entry in parsed_lines:
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type", "")
        if entry_type in _META_LINE_TYPES:
            continue

        message = entry.get("message")
        if not isinstance(message, dict):
            continue

        message_count += 1
        role = message.get("role", "")
        if role == "user":
            has_user = True
        elif role == "assistant":
            has_assistant = True

        # Timestamp ordering.
        ts_str = entry.get("timestamp")
        ts = _parse_timestamp(ts_str) if ts_str else None
        if ts is not None:
            timestamps.append(ts)

        # Token usage validation (assistant messages only).
        usage = message.get("usage")
        if usage is not None and isinstance(usage, dict):
            for ukey in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            ):
                val = usage.get(ukey)
                if val is not None and isinstance(val, (int, float)) and val < 0:
                    errors.append(
                        f"Negative token count: {ukey}={val} in message {entry.get('uuid', '?')}"
                    )

        # Content block inspection.
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                content_block_count += 1
                btype = block.get("type", "")
                if btype == "tool_use":
                    tid = block.get("id")
                    if tid:
                        tool_use_ids.add(tid)
                elif btype == "tool_result":
                    tid = block.get("tool_use_id")
                    if tid:
                        tool_result_ids.add(tid)
        elif isinstance(content, str):
            content_block_count += 1  # treat plain string as one block

    # -- Cross-message checks ----------------------------------------------

    if not has_user:
        errors.append("Session has no user messages")
    if not has_assistant:
        errors.append("Session has no assistant messages")

    # Chronological ordering.
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            errors.append(
                f"Timestamp out of order at position {i}: "
                f"{timestamps[i].isoformat()} < {timestamps[i - 1].isoformat()}"
            )
            break  # one error is enough to flag the issue

    # Tool use / tool result matching.
    unmatched_results = tool_result_ids - tool_use_ids
    if unmatched_results:
        errors.append(
            f"tool_result references non-existent tool_use ids: "
            + ", ".join(sorted(unmatched_results))
        )

    orphaned_uses = tool_use_ids - tool_result_ids
    if orphaned_uses:
        warnings.append(
            f"tool_use blocks without matching tool_result: "
            + ", ".join(sorted(orphaned_uses))
        )

    return SessionValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        message_count=message_count,
        content_block_count=content_block_count,
    )


# ---------------------------------------------------------------------------
# 3. Health report
# ---------------------------------------------------------------------------


def generate_health_report(
    parsed_lines: list[dict[str, Any]],
) -> HealthReport:
    """Generate a quick pre-analysis health summary of a session.

    Scans every parsed line to collect message counts, token estimates,
    content-type distribution, per-phase token estimates, and parsing
    warnings.  This is intended to be cheap enough to run before committing
    to the full TER analysis pipeline.

    Parameters
    ----------
    parsed_lines:
        A list of parsed JSON dicts from a JSONL session file.

    Returns
    -------
    HealthReport
    """
    user_count = 0
    assistant_count = 0
    parsing_warnings: list[str] = []

    text_count = 0
    tool_use_count = 0
    tool_result_count = 0
    thinking_count = 0
    other_count = 0

    reasoning_tokens = 0
    tool_use_tokens = 0
    generation_tokens = 0

    total_api_tokens = 0
    has_api_tokens = False

    span_count = 0

    for entry in parsed_lines:
        if not isinstance(entry, dict):
            parsing_warnings.append("Non-dict entry encountered in parsed lines")
            continue

        entry_type = entry.get("type", "")
        if entry_type in _META_LINE_TYPES:
            continue

        message = entry.get("message")
        if not isinstance(message, dict):
            continue

        role = message.get("role", "")
        if role == "user":
            user_count += 1
        elif role == "assistant":
            assistant_count += 1

        # Accumulate API-reported token counts.
        usage = message.get("usage")
        if isinstance(usage, dict):
            input_toks = usage.get("input_tokens", 0) or 0
            output_toks = usage.get("output_tokens", 0) or 0
            cache_create = usage.get("cache_creation_input_tokens", 0) or 0
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            msg_tokens = input_toks + output_toks + cache_create + cache_read
            if msg_tokens > 0:
                has_api_tokens = True
                total_api_tokens += msg_tokens

        content = message.get("content")
        if isinstance(content, str):
            text_count += 1
            span_count += 1
            generation_tokens += _estimate_tokens(content)
            continue

        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            btype = block.get("type", "")
            block_text = _extract_block_text(block)
            tokens = _estimate_tokens(block_text)
            span_count += 1

            if btype == "text":
                text_count += 1
                generation_tokens += tokens
            elif btype == "tool_use":
                tool_use_count += 1
                tool_use_tokens += tokens
            elif btype == "tool_result":
                tool_result_count += 1
                tool_use_tokens += tokens
            elif btype == "thinking":
                thinking_count += 1
                reasoning_tokens += tokens
            else:
                other_count += 1

    estimated_total = (
        total_api_tokens
        if has_api_tokens
        else reasoning_tokens + tool_use_tokens + generation_tokens
    )

    # Rough analysis time estimate: ~0.5ms per span for embedding + classify.
    estimated_analysis_seconds = span_count * 0.0005

    return HealthReport(
        user_message_count=user_count,
        assistant_message_count=assistant_count,
        estimated_total_tokens=estimated_total,
        content_distribution=ContentDistribution(
            text_count=text_count,
            tool_use_count=tool_use_count,
            tool_result_count=tool_result_count,
            thinking_count=thinking_count,
            other_count=other_count,
        ),
        reasoning_tokens=reasoning_tokens,
        tool_use_tokens=tool_use_tokens,
        generation_tokens=generation_tokens,
        parsing_warnings=parsing_warnings,
        estimated_analysis_seconds=estimated_analysis_seconds,
    )


# ---------------------------------------------------------------------------
# 4. Session completeness scoring
# ---------------------------------------------------------------------------


def assess_completeness(
    parsed_lines: list[dict[str, Any]],
) -> CompletenessAssessment:
    """Determine whether a session appears complete.

    Checks:
    * Whether the final assistant message has ``stop_reason == "end_turn"``.
    * Whether there are unresolved ``tool_use`` blocks (issued but never
      answered with a ``tool_result``).
    * Whether the session ends mid-tool-use (an assistant message ending
      with a ``tool_use`` block as the last action).

    A completeness score between 0 and 1 is computed as a weighted
    combination of the checks above.

    Parameters
    ----------
    parsed_lines:
        Parsed JSONL dicts from a session file.

    Returns
    -------
    CompletenessAssessment
    """
    issues: list[str] = []
    score = 1.0

    # Gather message-level data in order.
    last_assistant_msg: dict[str, Any] | None = None
    last_entry_type: str = ""
    tool_use_ids: set[str] = set()
    tool_result_ids: set[str] = set()

    for entry in parsed_lines:
        if not isinstance(entry, dict):
            continue

        entry_type = entry.get("type", "")
        if entry_type in _META_LINE_TYPES:
            continue

        message = entry.get("message")
        if not isinstance(message, dict):
            continue

        role = message.get("role", "")
        if role == "assistant":
            last_assistant_msg = message
            last_entry_type = entry_type

        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "tool_use":
                    tid = block.get("id")
                    if tid:
                        tool_use_ids.add(tid)
                elif btype == "tool_result":
                    tid = block.get("tool_use_id")
                    if tid:
                        tool_result_ids.add(tid)

    # -- Check 1: final stop_reason ---------------------------------------
    if last_assistant_msg is None:
        issues.append("No assistant messages found")
        score -= 0.5
    else:
        stop_reason = last_assistant_msg.get("stop_reason")
        if stop_reason != "end_turn":
            issues.append(
                f"Final assistant message has stop_reason='{stop_reason}' "
                "(expected 'end_turn')"
            )
            score -= 0.3

    # -- Check 2: unresolved tool_use blocks -------------------------------
    unresolved = tool_use_ids - tool_result_ids
    if unresolved:
        issues.append(
            f"{len(unresolved)} unresolved tool_use block(s) without matching tool_result"
        )
        # Scale penalty by proportion of unresolved tools.
        ratio = len(unresolved) / len(tool_use_ids) if tool_use_ids else 0
        score -= 0.2 * ratio

    # -- Check 3: session ends mid-tool-use --------------------------------
    if last_assistant_msg is not None:
        content = last_assistant_msg.get("content")
        if isinstance(content, list) and content:
            last_block = content[-1]
            if isinstance(last_block, dict) and last_block.get("type") == "tool_use":
                issues.append("Session ends mid-tool-use (last block is tool_use)")
                score -= 0.2

    score = max(0.0, min(1.0, score))
    return CompletenessAssessment(
        is_complete=len(issues) == 0,
        completeness_score=round(score, 4),
        issues=issues,
    )


# ---------------------------------------------------------------------------
# 5. Batch JSONL file validation
# ---------------------------------------------------------------------------


def validate_jsonl_file(
    path: str | Path,
) -> FileValidationResult:
    """Validate every line of a JSONL session file.

    Reads the file line by line, validates each with
    :func:`validate_jsonl_line`, and collects all errors and warnings.
    Does **not** fail on the first malformed line; instead, all problems
    are reported together with their line numbers.

    Parameters
    ----------
    path:
        Filesystem path to the JSONL file.

    Returns
    -------
    FileValidationResult

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"JSONL file not found: {filepath}")

    total_lines = 0
    valid_lines = 0
    error_lines: list[int] = []
    all_errors: list[str] = []
    all_warnings: list[str] = []

    with filepath.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            total_lines += 1
            result = validate_jsonl_line(raw_line, line_number=line_num)

            all_warnings.extend(result.warnings)

            if result.valid:
                valid_lines += 1
            else:
                error_lines.append(line_num)
                all_errors.extend(result.errors)

    return FileValidationResult(
        valid=len(all_errors) == 0,
        total_lines=total_lines,
        valid_lines=valid_lines,
        error_lines=error_lines,
        errors=all_errors,
        warnings=all_warnings,
    )
