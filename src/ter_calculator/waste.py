"""Waste pattern detection.

Detects three categories of waste:
1. Reasoning loops — the agent rehashes the same reasoning multiple times
2. Duplicate tool calls — identical tool invocations within a window
3. Context restatement — response text that repeats what was already said
"""

from __future__ import annotations

import re

from .classifier import cosine_similarity
from .models import (
    ALIGNED_LABELS,
    ClassifiedSpan,
    ContentBlock,
    Session,
    SpanLabel,
    SpanPhase,
    WastePattern,
)


def detect_waste_patterns(
    classified_spans: list[ClassifiedSpan],
    restatement_threshold: float = 0.85,
    session: Session | None = None,
) -> list[WastePattern]:
    """Detect all waste patterns in classified spans and session data."""
    patterns: list[WastePattern] = []
    patterns.extend(detect_reasoning_loops(classified_spans))
    patterns.extend(detect_duplicate_tool_calls(classified_spans))
    patterns.extend(
        detect_context_restatement(classified_spans, restatement_threshold)
    )
    if session is not None:
        patterns.extend(detect_repetitive_reads(session))
        patterns.extend(detect_edit_fragmentation(session))
        patterns.extend(detect_bash_antipatterns(session))
        patterns.extend(detect_failed_tool_retries(session))
        patterns.extend(detect_repeated_commands(session))
    return patterns


def summarize_waste(
    classified_spans: list[ClassifiedSpan],
    waste_patterns: list[WastePattern],
) -> dict:
    """Produce a human-readable waste summary.

    Returns a dict with:
    - total_waste_tokens: total tokens classified as waste
    - waste_by_category: breakdown by waste type
    - waste_by_phase: breakdown by phase
    - top_patterns: the most impactful waste patterns
    - explanation: human-readable summary string
    """
    waste_spans = [
        cs for cs in classified_spans if cs.label not in ALIGNED_LABELS
    ]

    total_waste = sum(cs.span.token_count for cs in waste_spans)
    total_all = sum(cs.span.token_count for cs in classified_spans)

    # By category.
    by_category: dict[str, int] = {}
    for cs in waste_spans:
        cat = _label_to_category(cs.label)
        by_category[cat] = by_category.get(cat, 0) + cs.span.token_count

    # By phase.
    by_phase: dict[str, int] = {}
    for cs in waste_spans:
        phase = cs.span.phase.value
        by_phase[phase] = by_phase.get(phase, 0) + cs.span.token_count

    # Top patterns by tokens wasted.
    top_patterns = sorted(
        waste_patterns, key=lambda p: p.tokens_wasted, reverse=True
    )[:5]

    # Build explanation.
    explanation = _build_explanation(
        total_waste, total_all, by_category, by_phase, top_patterns
    )

    return {
        "total_waste_tokens": total_waste,
        "waste_by_category": by_category,
        "waste_by_phase": by_phase,
        "top_patterns": [
            {
                "type": p.pattern_type,
                "tokens_wasted": p.tokens_wasted,
                "description": p.description,
            }
            for p in top_patterns
        ],
        "explanation": explanation,
    }


def _label_to_category(label: SpanLabel) -> str:
    return {
        SpanLabel.REDUNDANT_REASONING: "Redundant Reasoning",
        SpanLabel.UNNECESSARY_TOOL_CALL: "Unnecessary Tool Calls",
        SpanLabel.OVER_EXPLANATION: "Over-Explanation",
    }.get(label, "Other")


def _build_explanation(
    total_waste: int,
    total_all: int,
    by_category: dict[str, int],
    by_phase: dict[str, int],
    top_patterns: list[WastePattern],
) -> str:
    if total_all == 0:
        return "No tokens to analyze."
    if total_waste == 0:
        return "No waste detected. All tokens contributed to the task."

    pct = total_waste / total_all * 100
    lines = [
        f"{total_waste:,} of {total_all:,} tokens ({pct:.1f}%) were "
        f"identified as waste.",
    ]

    if by_category:
        biggest = max(by_category, key=by_category.get)  # type: ignore[arg-type]
        lines.append(
            f"The largest waste category is {biggest} "
            f"({by_category[biggest]:,} tokens)."
        )

    if top_patterns:
        p = top_patterns[0]
        lines.append(
            f"The most impactful pattern: {p.description} "
            f"({p.tokens_wasted:,} tokens)."
        )

    return " ".join(lines)


# --- Pattern detectors ---


def detect_reasoning_loops(
    classified_spans: list[ClassifiedSpan],
    min_consecutive: int = 3,
) -> list[WastePattern]:
    """Detect 3+ consecutive redundant reasoning spans."""
    patterns: list[WastePattern] = []
    consecutive: list[ClassifiedSpan] = []

    for cs in classified_spans:
        if (cs.span.phase == SpanPhase.REASONING
                and cs.label == SpanLabel.REDUNDANT_REASONING):
            consecutive.append(cs)
        else:
            if len(consecutive) >= min_consecutive:
                patterns.append(_make_reasoning_loop_pattern(consecutive))
            consecutive = []

    if len(consecutive) >= min_consecutive:
        patterns.append(_make_reasoning_loop_pattern(consecutive))

    return patterns


def detect_duplicate_tool_calls(
    classified_spans: list[ClassifiedSpan],
    window_size: int = 5,
) -> list[WastePattern]:
    """Detect repeated tool calls with identical name+params within a window.

    Only considers actual tool_use blocks (not tool_result), since
    duplicate results just mean the system returned similar confirmations.
    """
    patterns: list[WastePattern] = []
    seen_sigs: set[str] = set()
    tool_spans = [
        cs for cs in classified_spans
        if cs.span.phase == SpanPhase.TOOL_USE
        and cs.span.block_type == "tool_use"
    ]

    for i, cs in enumerate(tool_spans):
        sig = _get_tool_signature(cs)
        if sig is None:
            continue

        window_start = max(0, i - window_size)
        for j in range(window_start, i):
            prev_sig = _get_tool_signature(tool_spans[j])
            if prev_sig == sig:
                # Deduplicate: only report once per unique signature.
                if sig not in seen_sigs:
                    patterns.append(WastePattern(
                        pattern_type="duplicate_tool_call",
                        description=f"Duplicate tool call: {sig[:60]}",
                        start_position=tool_spans[j].span.position,
                        end_position=cs.span.position,
                        spans_involved=2,
                        tokens_wasted=cs.span.token_count,
                        details={"signature": sig},
                    ))
                    seen_sigs.add(sig)
                break

    return patterns


def detect_context_restatement(
    classified_spans: list[ClassifiedSpan],
    similarity_threshold: float = 0.85,
) -> list[WastePattern]:
    """Detect generation spans that closely repeat prior generation spans."""
    patterns: list[WastePattern] = []
    prior_gen: list[ClassifiedSpan] = []

    for cs in classified_spans:
        if cs.span.phase != SpanPhase.GENERATION:
            continue
        if cs.span.embedding is None:
            prior_gen.append(cs)
            continue

        for prior in prior_gen:
            if prior.span.embedding is None:
                continue
            sim = cosine_similarity(cs.span.embedding, prior.span.embedding)
            if sim >= similarity_threshold:
                patterns.append(WastePattern(
                    pattern_type="context_restatement",
                    description=(
                        f"Response restates prior content "
                        f"(similarity: {sim:.2f})"
                    ),
                    start_position=cs.span.position,
                    end_position=cs.span.position,
                    spans_involved=1,
                    tokens_wasted=cs.span.token_count,
                    details={
                        "similarity": round(sim, 4),
                        "prior_position": prior.span.position,
                    },
                ))
                break

        prior_gen.append(cs)

    return patterns


def _make_reasoning_loop_pattern(spans: list[ClassifiedSpan]) -> WastePattern:
    total_tokens = sum(cs.span.token_count for cs in spans)
    return WastePattern(
        pattern_type="reasoning_loop",
        description=f"{len(spans)} consecutive redundant reasoning spans",
        start_position=spans[0].span.position,
        end_position=spans[-1].span.position,
        spans_involved=len(spans),
        tokens_wasted=total_tokens,
    )


def _get_tool_signature(cs: ClassifiedSpan) -> str | None:
    text = cs.span.text
    if not text:
        return None
    return text.strip()


# --- Session-level pattern detectors ---


def detect_repetitive_reads(
    session: Session,
    min_reads: int = 3,
) -> list[WastePattern]:
    """Detect files read multiple times, estimating wasted input tokens.

    The first read of a file is necessary; subsequent reads of the same
    file are redundant. The token cost is the tool_result content that
    gets re-injected into the context each time.
    """
    # Build tool_use_id -> file_path map for Read operations.
    tool_file_map: dict[str, str] = {}
    for msg in session.messages:
        if msg.role != "assistant":
            continue
        for block in msg.content_blocks:
            if (block.block_type == "tool_use"
                    and block.tool_name == "Read"
                    and block.tool_use_id
                    and block.tool_input):
                fp = block.tool_input.get("file_path", "")
                if fp:
                    tool_file_map[block.tool_use_id] = fp

    # Match tool_result blocks to file paths and measure token cost.
    file_reads: dict[str, list[int]] = {}  # file_path -> [token_count, ...]
    for msg in session.messages:
        if msg.role != "user":
            continue
        for block in msg.content_blocks:
            if block.block_type == "tool_result" and block.tool_use_id:
                fp = tool_file_map.get(block.tool_use_id)
                if fp is not None:
                    text = block.text or ""
                    tokens = max(1, len(text) // 4) if text else 0
                    file_reads.setdefault(fp, []).append(tokens)

    # Build patterns for files read min_reads+ times.
    patterns: list[WastePattern] = []
    for fp, token_list in file_reads.items():
        if len(token_list) < min_reads:
            continue
        redundant_tokens = sum(token_list[1:])
        short_name = fp.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        patterns.append(WastePattern(
            pattern_type="repetitive_read",
            description=(
                f"{short_name} read {len(token_list)}x "
                f"({redundant_tokens:,} redundant tokens)"
            ),
            start_position=0,
            end_position=0,
            spans_involved=len(token_list),
            tokens_wasted=redundant_tokens,
            details={
                "file_path": fp,
                "read_count": len(token_list),
                "per_read_tokens": token_list,
            },
        ))

    patterns.sort(key=lambda p: p.tokens_wasted, reverse=True)
    return patterns


def detect_edit_fragmentation(
    session: Session,
    min_consecutive: int = 3,
) -> list[WastePattern]:
    """Detect consecutive Edit/Write operations targeting the same file.

    Multiple sequential edits to one file could often be a single
    batched operation. The first edit is necessary; subsequent ones
    in the run are fragmentation waste.
    """
    # Extract ordered list of (tool_name, file_path, token_count).
    ops: list[tuple[str, str, int]] = []
    for msg in session.messages:
        if msg.role != "assistant":
            continue
        for block in msg.content_blocks:
            if (block.block_type == "tool_use"
                    and block.tool_name in ("Edit", "Write")
                    and block.tool_input):
                fp = block.tool_input.get("file_path", "")
                if fp:
                    text = block.text or ""
                    tokens = max(1, len(text) // 4) if text else 0
                    ops.append((block.tool_name, fp, tokens))

    # Find consecutive runs targeting the same file.
    patterns: list[WastePattern] = []
    current_file: str | None = None
    current_run: list[tuple[str, int]] = []

    def _flush_run():
        if current_file and len(current_run) >= min_consecutive:
            fragmented_tokens = sum(t for _, t in current_run[1:])
            short_name = current_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            patterns.append(WastePattern(
                pattern_type="edit_fragmentation",
                description=(
                    f"{len(current_run)} consecutive edits to {short_name} "
                    f"({fragmented_tokens:,} fragmented tokens)"
                ),
                start_position=0,
                end_position=0,
                spans_involved=len(current_run),
                tokens_wasted=fragmented_tokens,
                details={
                    "file_path": current_file,
                    "edit_count": len(current_run),
                    "operations": [name for name, _ in current_run],
                },
            ))

    for name, fp, tokens in ops:
        if fp == current_file:
            current_run.append((name, tokens))
        else:
            _flush_run()
            current_file = fp
            current_run = [(name, tokens)]

    _flush_run()

    patterns.sort(key=lambda p: p.tokens_wasted, reverse=True)
    return patterns


# Bash anti-pattern rules: (regex, recommended tool, description).
_BASH_ANTIPATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"(?:^|\|\s*)cat\s+"), "Read", "cat → Read"),
    (re.compile(r"(?:^|\|\s*)head\s+"), "Read", "head → Read"),
    (re.compile(r"(?:^|\|\s*)tail\s+"), "Read", "tail → Read"),
    (re.compile(r"(?:^|\|\s*)grep\s+"), "Grep", "grep → Grep"),
    (re.compile(r"(?:^|\|\s*)rg\s+"), "Grep", "rg → Grep"),
    (re.compile(r"^find\s+"), "Glob", "find → Glob"),
]


def _classify_bash_command(command: str) -> tuple[str, str] | None:
    """Return (recommended_tool, description) if the command is an anti-pattern."""
    cmd = command.strip()
    for pattern, tool, desc in _BASH_ANTIPATTERNS:
        if pattern.search(cmd):
            return tool, desc
    return None


def detect_bash_antipatterns(
    session: Session,
) -> list[WastePattern]:
    """Detect Bash commands that should use dedicated tools.

    Flags commands like `cat file`, `grep pattern`, `find .` that
    have dedicated tools (Read, Grep, Glob) which are faster and
    produce better-structured output.
    """
    # Collect anti-pattern instances and their tool_result token costs.
    tool_result_tokens: dict[str, int] = {}  # tool_use_id → tokens
    instances: list[tuple[str, str, str]] = []  # (tool, desc, tool_use_id)

    for msg in session.messages:
        if msg.role == "assistant":
            for block in msg.content_blocks:
                if (block.block_type == "tool_use"
                        and block.tool_name == "Bash"
                        and block.tool_input):
                    cmd = block.tool_input.get("command", "")
                    result = _classify_bash_command(cmd)
                    if result and block.tool_use_id:
                        instances.append((result[0], result[1], block.tool_use_id))
        elif msg.role == "user":
            for block in msg.content_blocks:
                if block.block_type == "tool_result" and block.tool_use_id:
                    text = block.text or ""
                    tool_result_tokens[block.tool_use_id] = max(1, len(text) // 4)

    if not instances:
        return []

    # Group by recommended tool.
    by_tool: dict[str, list[tuple[str, str]]] = {}  # tool → [(desc, tool_use_id)]
    for tool, desc, tuid in instances:
        by_tool.setdefault(tool, []).append((desc, tuid))

    patterns: list[WastePattern] = []
    total_tokens = sum(
        tool_result_tokens.get(tuid, 0)
        for _, _, tuid in instances
    )

    patterns.append(WastePattern(
        pattern_type="bash_antipattern",
        description=(
            f"{len(instances)} Bash calls should use dedicated tools "
            f"({total_tokens:,} tokens)"
        ),
        start_position=0,
        end_position=0,
        spans_involved=len(instances),
        tokens_wasted=total_tokens,
        details={
            "instance_count": len(instances),
            "by_tool": {
                tool: len(items) for tool, items in by_tool.items()
            },
        },
    ))

    return patterns


def detect_failed_tool_retries(
    session: Session,
) -> list[WastePattern]:
    """Detect tool calls that fail and are retried.

    The failed call + its error result consume tokens without
    contributing to progress.
    """
    # Build ordered list of (tool_use_id, tool_name, is_assistant).
    tool_calls: list[tuple[str, str]] = []  # (tool_use_id, tool_name)
    error_ids: set[str] = set()
    result_tokens: dict[str, int] = {}  # tool_use_id → result tokens

    for msg in session.messages:
        if msg.role == "assistant":
            for block in msg.content_blocks:
                if block.block_type == "tool_use" and block.tool_use_id:
                    tool_calls.append((block.tool_use_id, block.tool_name or ""))
        elif msg.role == "user":
            for block in msg.content_blocks:
                if block.block_type == "tool_result" and block.tool_use_id:
                    text = block.text or ""
                    result_tokens[block.tool_use_id] = max(1, len(text) // 4)
                    # Check for error indicators.
                    if _is_error_result(block):
                        error_ids.add(block.tool_use_id)

    if not error_ids:
        return []

    # Sum token cost of failed calls and their error results.
    total_tokens = sum(result_tokens.get(eid, 0) for eid in error_ids)

    return [WastePattern(
        pattern_type="failed_tool_retry",
        description=(
            f"{len(error_ids)} failed tool calls "
            f"({total_tokens:,} wasted tokens)"
        ),
        start_position=0,
        end_position=0,
        spans_involved=len(error_ids),
        tokens_wasted=total_tokens,
        details={
            "error_count": len(error_ids),
        },
    )]


def _is_error_result(block: ContentBlock) -> bool:
    """Check if a tool_result block indicates an error."""
    text = block.text or ""
    if "<tool_use_error>" in text:
        return True
    if text.startswith("Error:") or text.startswith("Exit code 1"):
        return True
    # Check for common error patterns in the content.
    error_markers = [
        "File does not exist",
        "command not found",
        "No such file or directory",
        "Permission denied",
        "File has not been read yet",
    ]
    return any(marker in text for marker in error_markers)


def detect_repeated_commands(
    session: Session,
    min_repeats: int = 3,
) -> list[WastePattern]:
    """Detect the same Bash command run multiple times.

    Normalizes commands to ignore minor differences in piped
    output (e.g. `| tail -30` vs `| tail -50`).
    """
    # Extract bash commands and their result token costs.
    commands: list[tuple[str, str]] = []  # (normalized_cmd, tool_use_id)
    result_tokens: dict[str, int] = {}

    for msg in session.messages:
        if msg.role == "assistant":
            for block in msg.content_blocks:
                if (block.block_type == "tool_use"
                        and block.tool_name == "Bash"
                        and block.tool_input
                        and block.tool_use_id):
                    cmd = block.tool_input.get("command", "")
                    norm = _normalize_bash_command(cmd)
                    if norm:
                        commands.append((norm, block.tool_use_id))
        elif msg.role == "user":
            for block in msg.content_blocks:
                if block.block_type == "tool_result" and block.tool_use_id:
                    text = block.text or ""
                    result_tokens[block.tool_use_id] = max(1, len(text) // 4)

    # Count occurrences of each normalized command.
    cmd_groups: dict[str, list[str]] = {}  # normalized → [tool_use_ids]
    for norm, tuid in commands:
        cmd_groups.setdefault(norm, []).append(tuid)

    patterns: list[WastePattern] = []
    for norm, tuids in cmd_groups.items():
        if len(tuids) < min_repeats:
            continue
        # First invocation is needed; subsequent are waste.
        redundant_tokens = sum(result_tokens.get(t, 0) for t in tuids[1:])
        short_cmd = norm[:60] + "..." if len(norm) > 60 else norm
        patterns.append(WastePattern(
            pattern_type="repeated_command",
            description=(
                f"'{short_cmd}' run {len(tuids)}x "
                f"({redundant_tokens:,} redundant tokens)"
            ),
            start_position=0,
            end_position=0,
            spans_involved=len(tuids),
            tokens_wasted=redundant_tokens,
            details={
                "command": norm,
                "run_count": len(tuids),
            },
        ))

    patterns.sort(key=lambda p: p.tokens_wasted, reverse=True)
    return patterns


def _normalize_bash_command(cmd: str) -> str:
    """Normalize a bash command for deduplication.

    Strips trailing pipe modifiers (| tail -N, | head -N, | grep)
    and whitespace so that near-identical commands match.
    """
    cmd = cmd.strip()
    # Remove trailing pipe to tail/head with varying line counts.
    cmd = re.sub(r'\s*\|\s*(tail|head)\s+-\d+\s*$', '', cmd)
    # Normalize whitespace.
    cmd = re.sub(r'\s+', ' ', cmd)
    return cmd
