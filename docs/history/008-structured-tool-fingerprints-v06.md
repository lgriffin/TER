# TER v06 — Structured Tool-Call Fingerprints

## Purpose

v06 replaces embedding-only duplicate detection for statically parsed tool calls with a deterministic structured fingerprint. Tool calls are compared using the tool name and normalized arguments rather than the rendered JSON string.

## Normalized evidence

- Tool name
- File or directory path
- Line range, offset, and limit
- Search query or pattern
- Shell command
- Remaining normalized arguments

Volatile metadata such as request IDs, timestamps, tool-use IDs, and descriptions is excluded from the fingerprint.

## Classification behavior

- Exact structured repeat: strong duplicate evidence.
- Same tool with a different file range: parameter novelty; not a duplicate.
- Same search tool with a different query: new work; not a duplicate.
- Same shell tool with a different command: new work; not a duplicate.
- Spans without structured tool metadata retain the previous semantic fallback.

Production intent thresholds and non-tool repetition thresholds are unchanged.

## Compatibility

`TokenSpan` gained optional `tool_name` and `tool_input` fields with defaults, so existing constructors and external consumers remain compatible. Static session segmentation now preserves these fields from `ContentBlock`.

## Validation assets

- `tests/unit/test_tool_fingerprints.py`
- `benchmarks/tool_call_adversarial.jsonl`

The synthetic benchmark is a regression fixture, not empirical validation data.
