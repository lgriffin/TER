"""JSONL session loading and span segmentation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import (
    ContentBlock,
    Message,
    Session,
    SpanPhase,
    TokenSpan,
    TokenUsage,
)


def load_session(path: str | Path) -> Session:
    """Load a Claude Code session from a JSONL file.

    Parses each line, constructs Message objects, deduplicates by requestId
    (keeps the entry with the highest output_tokens), and builds a Session.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Session file not found: {path}")
    if not path.suffix == ".jsonl":
        raise ValueError(f"Expected .jsonl file, got: {path.suffix}")

    raw_entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e

    if not raw_entries:
        raise ValueError(f"Session file is empty: {path}")

    # Deduplicate by requestId — keep entry with highest output_tokens.
    deduped = _deduplicate_entries(raw_entries)

    # Build messages.
    messages: list[Message] = []
    session_id = ""
    first_timestamp = None

    for entry in deduped:
        entry_type = entry.get("type", "")
        if entry_type not in ("user", "assistant"):
            continue

        msg_data = entry.get("message", {})
        uuid = entry.get("uuid", "")
        if not session_id:
            session_id = entry.get("sessionId", "")

        timestamp = _parse_timestamp(entry.get("timestamp"))
        if first_timestamp is None:
            first_timestamp = timestamp

        content_blocks = _parse_content_blocks(msg_data.get("content", []))
        usage = _parse_usage(msg_data.get("usage"))

        messages.append(Message(
            uuid=uuid,
            role=msg_data.get("role", entry_type),
            content_blocks=content_blocks,
            parent_uuid=entry.get("parentUuid"),
            timestamp=timestamp,
            request_id=entry.get("requestId"),
            usage=usage,
            stop_reason=msg_data.get("stop_reason"),
        ))

    # Extract user prompts.
    user_prompts = _extract_user_prompts(messages)

    # Compute total tokens from assistant message usage.
    total_tokens = sum(
        m.usage.output_tokens for m in messages
        if m.usage is not None
    )

    return Session(
        session_id=session_id or path.stem,
        file_path=str(path),
        messages=messages,
        timestamp=first_timestamp,
        total_tokens=total_tokens,
        user_prompts=user_prompts,
    )


def segment_spans(session: Session) -> list[TokenSpan]:
    """Extract TokenSpans from a Session's content blocks.

    Assigns phases based on block type:
    - thinking → reasoning
    - tool_use, tool_result → tool_use
    - text → generation

    Estimates token counts using character heuristic (len / 4).
    """
    spans: list[TokenSpan] = []
    position = 0

    for message in session.messages:
        for block in message.content_blocks:
            text = _get_block_text(block)
            if not text:
                continue

            phase = _block_type_to_phase(block.block_type)
            token_count = max(1, len(text) // 4)

            spans.append(TokenSpan(
                text=text,
                phase=phase,
                position=position,
                token_count=token_count,
                source_message_uuid=message.uuid,
                block_type=block.block_type,
            ))
            position += 1

    return spans


def _deduplicate_entries(entries: list[dict]) -> list[dict]:
    """Deduplicate entries by requestId, keeping highest output_tokens."""
    seen_request_ids: dict[str, int] = {}  # requestId -> index in result
    result: list[dict] = []

    for entry in entries:
        request_id = entry.get("requestId")
        if request_id is None:
            result.append(entry)
            continue

        usage = entry.get("message", {}).get("usage", {})
        output_tokens = usage.get("output_tokens", 0) if usage else 0

        if request_id in seen_request_ids:
            idx = seen_request_ids[request_id]
            existing_usage = result[idx].get("message", {}).get("usage", {})
            existing_output = existing_usage.get("output_tokens", 0) if existing_usage else 0
            if output_tokens > existing_output:
                result[idx] = entry
        else:
            seen_request_ids[request_id] = len(result)
            result.append(entry)

    return result


def _parse_content_blocks(content) -> list[ContentBlock]:
    """Parse content field into ContentBlock objects."""
    if isinstance(content, str):
        return [ContentBlock(block_type="text", text=content)]

    if not isinstance(content, list):
        return []

    blocks: list[ContentBlock] = []
    for item in content:
        if isinstance(item, str):
            blocks.append(ContentBlock(block_type="text", text=item))
            continue
        if not isinstance(item, dict):
            continue

        block_type = item.get("type", "text")
        text = item.get("text")
        if text is None:
            raw_content = item.get("content")
            if isinstance(raw_content, str):
                text = raw_content
            elif isinstance(raw_content, list):
                # tool_result content can be a list of {type, text} objects.
                text = " ".join(
                    c.get("text", "") for c in raw_content
                    if isinstance(c, dict) and c.get("text")
                )
            # Otherwise leave text as None.
        blocks.append(ContentBlock(
            block_type=block_type,
            text=text,
            tool_name=item.get("name"),
            tool_input=item.get("input"),
            tool_use_id=item.get("id") or item.get("tool_use_id"),
        ))

    return blocks


def _parse_usage(usage_data) -> TokenUsage | None:
    if not isinstance(usage_data, dict):
        return None
    return TokenUsage(
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
        cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
    )


def _parse_timestamp(ts_str) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _extract_user_prompts(messages: list[Message]) -> list[str]:
    """Extract text content from user messages."""
    prompts: list[str] = []
    for msg in messages:
        if msg.role != "user":
            continue
        for block in msg.content_blocks:
            if block.block_type == "text" and block.text:
                prompts.append(block.text)
    return prompts


def _get_block_text(block: ContentBlock) -> str:
    """Get displayable text from a content block."""
    if block.block_type in ("text", "thinking"):
        return block.text or ""
    if block.block_type == "tool_use":
        parts = [block.tool_name or "unknown_tool"]
        if block.tool_input:
            try:
                parts.append(json.dumps(block.tool_input, separators=(",", ":")))
            except (TypeError, ValueError):
                pass
        return " ".join(parts)
    if block.block_type == "tool_result":
        return block.text or ""
    return ""


def _block_type_to_phase(block_type: str) -> SpanPhase:
    """Map content block type to span phase."""
    if block_type == "thinking":
        return SpanPhase.REASONING
    if block_type in ("tool_use", "tool_result"):
        return SpanPhase.TOOL_USE
    return SpanPhase.GENERATION
