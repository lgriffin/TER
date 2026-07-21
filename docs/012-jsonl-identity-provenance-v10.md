# TER v10.1 — Stable JSONL Identity and Provenance

TER v10.1 retains the v10 identity and provenance design and fixes a static typing regression. TER v10 strengthens session ingestion so duplicate, sibling, partial, and identifier-poor JSONL records are handled deterministically.

## Stable block fingerprints

Each content block receives a SHA-256 fingerprint derived from its normalized role and block content. Volatile fields such as timestamps, request IDs, UUIDs, and parent UUIDs are excluded. Text whitespace and dictionary ordering are normalized.

## Deterministic merge rules

1. Records sharing `requestId` and role are sibling candidates.
2. When `requestId` is absent, an explicit `messageId` or `message_id` may identify siblings.
3. UUID-only records remain independent for backward compatibility.
4. Exact block fingerprints are emitted once.
5. Duplicate blocks retain every contributing source line.
6. Distinct sibling blocks are retained in first-seen order.
7. Distinct sibling merges generate inspectable warnings.
8. Usage is backfilled when the first sibling lacks it.

## Provenance fields

`ContentBlock` and `TokenSpan` now expose:

- `source_line`
- `source_lines`
- `content_fingerprint`
- source block index

`Message` exposes source lines and merge warnings. `Session` exposes aggregate merge warnings.

## Compatibility

- Existing `ContentBlock`, `Message`, `Session`, and `TokenSpan` constructors remain valid because all new fields have defaults.
- `_deduplicate_entries()` remains available as a compatibility wrapper.
- Existing request-ID sibling behavior is preserved.
- Fine segmentation continues to propagate block provenance to every derived segment.

## Validation

The v10-specific loader, provenance, and pipeline regression suite passes 29 tests. Ruff passes across source and tests. The full local suite cannot complete in the restricted build environment because existing semantic tests require Hugging Face model access.
