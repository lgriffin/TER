from __future__ import annotations

import json

from ter_calculator.jsonl_identity import content_block_fingerprint
from ter_calculator.loader import (
    _deduplicate_entries_with_warnings,
    load_session,
    segment_spans,
)


def _entry(line: int, content: list[dict], *, request_id: str | None = "r1") -> dict:
    entry = {
        "type": "assistant",
        "uuid": "message-1",
        "sessionId": "session-1",
        "message": {"role": "assistant", "content": content},
        "_source_line": line,
    }
    if request_id is not None:
        entry["requestId"] = request_id
    return entry


def test_fingerprint_ignores_volatile_metadata_and_normalizes_text() -> None:
    first = {"type": "text", "text": "hello   world", "timestamp": "one"}
    second = {"timestamp": "two", "text": "hello world", "type": "text"}
    assert content_block_fingerprint("assistant", first) == content_block_fingerprint(
        "assistant", second
    )


def test_exact_duplicate_blocks_are_emitted_once_with_all_source_lines() -> None:
    merged, warnings = _deduplicate_entries_with_warnings(
        [
            _entry(1, [{"type": "text", "text": "same"}]),
            _entry(2, [{"type": "text", "text": "same"}]),
        ]
    )
    blocks = merged[0]["message"]["content"]
    assert len(blocks) == 1
    assert blocks[0]["_source_lines"] == [1, 2]
    assert warnings == []


def test_distinct_sibling_blocks_are_retained_in_source_order() -> None:
    merged, warnings = _deduplicate_entries_with_warnings(
        [
            _entry(4, [{"type": "thinking", "thinking": "plan"}]),
            _entry(7, [{"type": "text", "text": "answer"}]),
        ]
    )
    blocks = merged[0]["message"]["content"]
    assert [block["type"] for block in blocks] == ["thinking", "text"]
    assert merged[0]["_source_lines"] == [4, 7]
    assert len(warnings) == 1


def test_missing_request_id_keeps_uuid_only_records_distinct() -> None:
    merged, _ = _deduplicate_entries_with_warnings(
        [
            _entry(1, [{"type": "text", "text": "one"}], request_id=None),
            _entry(2, [{"type": "text", "text": "two"}], request_id=None),
        ]
    )
    assert len(merged) == 2


def test_explicit_message_id_merges_siblings_without_request_id() -> None:
    first = _entry(1, [{"type": "text", "text": "one"}], request_id=None)
    second = _entry(2, [{"type": "text", "text": "two"}], request_id=None)
    first["messageId"] = second["messageId"] = "shared-message"
    merged, _ = _deduplicate_entries_with_warnings([first, second])
    assert len(merged) == 1
    assert len(merged[0]["message"]["content"]) == 2


def test_load_session_and_spans_preserve_block_provenance(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    rows = [
        {
            key: value
            for key, value in _entry(1, [{"type": "text", "text": "same"}]).items()
            if key != "_source_line"
        },
        {
            key: value
            for key, value in _entry(2, [{"type": "text", "text": "same"}]).items()
            if key != "_source_line"
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    session = load_session(path)
    block = session.messages[0].content_blocks[0]
    assert block.source_line == 1
    assert block.source_lines == [1, 2]
    assert block.content_fingerprint

    span = segment_spans(session)[0]
    assert span.source_line == 1
    assert span.source_lines == [1, 2]
    assert span.content_fingerprint == block.content_fingerprint
    assert span.source_block_index == 0
