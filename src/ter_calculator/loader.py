"""JSONL session loading and span segmentation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .embedding_cache import estimate_tokens
from .jsonl_identity import content_block_fingerprint, entry_identity
from .span_segmentation import SegmentationConfig, segment_text
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

    Parses each line, merges sibling lines that share a requestId (Claude Code
    writes one line per content block), and builds a Session.
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
                entry = json.loads(line)
                if isinstance(entry, dict):
                    entry["_source_line"] = line_num
                    raw_entries.append(entry)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e

    if not raw_entries:
        raise ValueError(f"Session file is empty: {path}")

    # Deduplicate by requestId — keep entry with highest output_tokens.
    deduped, merge_warnings = _deduplicate_entries_with_warnings(raw_entries)

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

        content_blocks = _parse_content_blocks(
            msg_data.get("content", []),
            role=msg_data.get("role", entry_type),
        )
        usage = _parse_usage(msg_data.get("usage"))

        messages.append(
            Message(
                uuid=uuid,
                role=msg_data.get("role", entry_type),
                content_blocks=content_blocks,
                parent_uuid=entry.get("parentUuid"),
                timestamp=timestamp,
                request_id=entry.get("requestId"),
                usage=usage,
                stop_reason=msg_data.get("stop_reason"),
                source_lines=list(
                    entry.get("_source_lines", [entry.get("_source_line")])
                )
                if entry.get("_source_line") is not None
                else [],
                merge_warnings=list(entry.get("_merge_warnings", [])),
            )
        )

    # Extract user prompts.
    user_prompts = _extract_user_prompts(messages)

    # Compute total tokens from assistant message usage.
    total_tokens = sum(m.usage.output_tokens for m in messages if m.usage is not None)

    return Session(
        session_id=session_id or path.stem,
        file_path=str(path),
        messages=messages,
        timestamp=first_timestamp,
        total_tokens=total_tokens,
        user_prompts=user_prompts,
        merge_warnings=merge_warnings,
    )


def segment_spans(
    session: Session,
    config: SegmentationConfig | None = None,
) -> list[TokenSpan]:
    """Extract model-output TokenSpans from a Session's content blocks.

    User messages remain available on the Session for intent construction and
    input analysis, but they are excluded from TER scoring. This prevents long
    prompts and duplicated queue metadata from being counted as generated
    output.

    Assigns phases based on assistant block type:
    - thinking → reasoning
    - tool_use, tool_result → tool_use
    - text → generation

    Token counts use tiktoken cl100k_base (same BPE approximation as live mode).
    """
    spans: list[TokenSpan] = []
    position = 0
    segmentation = config or SegmentationConfig()

    for message in session.messages:
        # TER measures model-output efficiency. User prompts are inputs used to
        # construct intent and must never become scored generation spans.
        if message.role != "assistant":
            continue

        for block_index, block in enumerate(message.content_blocks):
            text = _get_block_text(block)
            if not text:
                continue

            phase = _block_type_to_phase(block.block_type)
            parent_block_id = f"{message.uuid}:{block_index}"
            should_split = block.block_type in {"thinking", "text"}
            pieces = (
                segment_text(text, segmentation)
                if should_split
                else segment_text(text, SegmentationConfig(enabled=False))
            )

            for segment_index, piece in enumerate(pieces):
                spans.append(
                    TokenSpan(
                        text=piece.text,
                        phase=phase,
                        position=position,
                        token_count=estimate_tokens(piece.text),
                        source_message_uuid=message.uuid,
                        block_type=block.block_type,
                        source_role=message.role,
                        tool_name=block.tool_name,
                        tool_input=block.tool_input,
                        parent_block_id=parent_block_id,
                        segment_index=segment_index,
                        char_start=piece.char_start,
                        char_end=piece.char_end,
                        source_line=block.source_line,
                        source_lines=list(block.source_lines),
                        content_fingerprint=block.content_fingerprint,
                        source_block_index=block.block_index,
                    )
                )
                position += 1

    return spans


def discover_subagents(parent_path: str | Path) -> list[Path]:
    """Discover subagent session files for a parent session.

    Given a parent session at ``{dir}/SESSION_ID.jsonl``, looks for
    subagent files at ``{dir}/SESSION_ID/subagents/*.jsonl``.
    """
    pp = Path(parent_path)
    subagent_dir = pp.parent / pp.stem / "subagents"
    if not subagent_dir.is_dir():
        return []
    return sorted(subagent_dir.glob("*.jsonl"))


def find_latest_session(project_path: str | Path | None = None) -> Path:
    """Find the most recent session file based on modification time.

    Args:
        project_path: Path to Claude Code project directory. If None, uses
                     ~/.claude/projects and finds the most recent project.

    Returns:
        Path to the latest .jsonl session file.

    Raises:
        FileNotFoundError: If no sessions found or project path doesn't exist.
    """
    if project_path is None:
        # Find most recent project in ~/.claude/projects
        home = Path.home()
        claude_dir = home / ".claude" / "projects"
        if not claude_dir.exists():
            raise FileNotFoundError(
                "No Claude Code projects found at ~/.claude/projects/"
            )

        # Find all sessions across all projects, sorted by modification time
        all_sessions = []
        for jsonl_file in claude_dir.rglob("*.jsonl"):
            if "subagents" not in jsonl_file.parts:
                all_sessions.append(jsonl_file)

        if not all_sessions:
            raise FileNotFoundError(f"No session files found in {claude_dir}")

        # Return the most recently modified
        return max(all_sessions, key=lambda p: p.stat().st_mtime)

    # Use specified project path
    project_dir = Path(project_path)
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_path}")

    # Find all sessions in this project, excluding subagents
    sessions = []
    for jsonl_file in project_dir.rglob("*.jsonl"):
        if "subagents" not in jsonl_file.parts:
            sessions.append(jsonl_file)

    if not sessions:
        raise FileNotFoundError(f"No session files found in {project_path}")

    # Return the most recently modified
    return max(sessions, key=lambda p: p.stat().st_mtime)


def _deduplicate_entries(entries: list[dict]) -> list[dict]:
    """Compatibility wrapper returning deterministically merged entries."""
    merged, _ = _deduplicate_entries_with_warnings(entries)
    return merged


def _deduplicate_entries_with_warnings(
    entries: list[dict],
) -> tuple[list[dict], list[str]]:
    """Merge sibling records while preserving order, conflicts, and provenance.

    Exact content-block fingerprints are emitted once. Distinct sibling blocks
    sharing an entry identity are retained in first-seen order. Missing request
    IDs fall back to UUID/message identity and ultimately source-line identity.
    """
    import copy

    seen: dict[str, int] = {}
    result: list[dict] = []
    warnings: list[str] = []

    for sequence, original in enumerate(entries, 1):
        entry = copy.deepcopy(original)
        source_line = int(entry.get("_source_line", sequence))
        identity = entry_identity(entry, source_line)
        role = entry.get("message", {}).get("role", entry.get("type", ""))
        content = entry.get("message", {}).get("content", [])
        blocks = content if isinstance(content, list) else [content]

        annotated_blocks: list[dict | str] = []
        for block_index, block in enumerate(blocks):
            if not isinstance(block, (dict, str)):
                continue
            if isinstance(block, dict):
                annotated_dict = copy.deepcopy(block)
                annotated_dict["_source_line"] = source_line
                annotated_dict["_source_lines"] = [source_line]
                annotated_dict["_source_block_index"] = block_index
                annotated_dict["_content_fingerprint"] = content_block_fingerprint(
                    role, block
                )
                annotated_blocks.append(annotated_dict)
            else:
                annotated_blocks.append(block)

        entry.setdefault("message", {})["content"] = annotated_blocks
        entry["_source_lines"] = [source_line]

        if identity not in seen:
            seen[identity] = len(result)
            result.append(entry)
            continue

        base = result[seen[identity]]
        base_msg = base.setdefault("message", {})
        base_content = base_msg.setdefault("content", [])
        if not isinstance(base_content, list):
            base_content = [base_content]
            base_msg["content"] = base_content

        existing: dict[str, dict] = {}
        for block in base_content:
            if isinstance(block, dict):
                fingerprint = block.get("_content_fingerprint")
                if fingerprint:
                    existing[fingerprint] = block

        added_distinct = False
        for block in annotated_blocks:
            if not isinstance(block, dict):
                if block not in base_content:
                    base_content.append(block)
                    added_distinct = True
                continue
            fingerprint = block.get("_content_fingerprint")
            if fingerprint in existing:
                previous = existing[fingerprint]
                lines = previous.setdefault("_source_lines", [])
                if source_line not in lines:
                    lines.append(source_line)
                continue
            base_content.append(block)
            if fingerprint:
                existing[fingerprint] = block
            added_distinct = True

        source_lines = base.setdefault("_source_lines", [])
        if source_line not in source_lines:
            source_lines.append(source_line)

        usage = entry.get("message", {}).get("usage")
        if not base_msg.get("usage") and usage:
            base_msg["usage"] = usage

        if added_distinct:
            warning = (
                f"Merged distinct sibling content for {identity} from source line "
                f"{source_line}; all non-duplicate blocks were retained."
            )
            base.setdefault("_merge_warnings", []).append(warning)
            warnings.append(warning)

    return result, warnings


def _parse_content_blocks(
    content,
    *,
    role: str = "",
) -> list[ContentBlock]:
    """Parse content into blocks with stable identity and source provenance."""
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
        if text is None and block_type == "thinking":
            text = item.get("thinking")
        if text is None:
            raw_content = item.get("content")
            if isinstance(raw_content, str):
                text = raw_content
            elif isinstance(raw_content, list):
                # tool_result content can be a list of {type, text} objects.
                text = " ".join(
                    c.get("text", "")
                    for c in raw_content
                    if isinstance(c, dict) and c.get("text")
                )
            # Otherwise leave text as None.
        fingerprint = item.get("_content_fingerprint") or content_block_fingerprint(
            role,
            {key: value for key, value in item.items() if not str(key).startswith("_")},
        )
        source_line = item.get("_source_line")
        source_lines = item.get("_source_lines", [])
        blocks.append(
            ContentBlock(
                block_type=block_type,
                text=text,
                tool_name=item.get("name"),
                tool_input=item.get("input"),
                tool_use_id=item.get("id") or item.get("tool_use_id"),
                source_line=source_line if isinstance(source_line, int) else None,
                source_lines=[line for line in source_lines if isinstance(line, int)],
                content_fingerprint=str(fingerprint),
                block_index=item.get("_source_block_index"),
            )
        )

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
