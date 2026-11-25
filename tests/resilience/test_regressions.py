"""The inputs the fuzzers found, pinned so they stay fixed (TER-TST-009,
TER-TST-013). Each case crashed a reader before this suite existed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.event_log import JsonlEventLog, event_to_record
from ter.adapters.driven.gare import GareRunSource
from ter.domain import EventKind

PROMPT = {
    "type": "user",
    "uuid": "u1",
    "sessionId": "s1",
    "timestamp": "2026-01-15T10:00:00.000Z",
    "message": {"role": "user", "content": [{"type": "text", "text": "Fix it"}]},
}
CALL = {
    "type": "assistant",
    "uuid": "a1",
    "sessionId": "s1",
    "requestId": "r1",
    "timestamp": "2026-01-15T10:00:01.000Z",
    "message": {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "Read",
                "input": {"file_path": "a"},
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    },
}


def _read(tmp_path: Path, *records: Any) -> Any:
    path = tmp_path / "s1.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return ClaudeCodeJsonlSource().read(path)


def _with(record: dict[str, Any], **message: Any) -> dict[str, Any]:
    return record | {"message": record["message"] | message}


@pytest.mark.req("TER-TST-009")
def test_a_record_repeated_with_its_uuid_gives_its_events_once(tmp_path: Path) -> None:
    trace = _read(tmp_path, PROMPT, PROMPT, CALL)
    assert [e.kind for e in trace.events] == [
        EventKind.PROMPT,
        EventKind.TOOL_REQUESTED,
    ]


RESULT = {
    "type": "user",
    "uuid": "u2",
    "sessionId": "s1",
    "timestamp": "2026-01-15T10:00:02.000Z",
    "message": {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
    },
}


def _texts(*texts: str, uuid: str = "a1") -> list[dict[str, Any]]:
    """One response written as Claude Code writes it: a line per block."""
    return [
        CALL
        | {"uuid": f"{uuid}-{i}" if i else uuid}
        | {"message": CALL["message"] | {"content": [{"type": "text", "text": t}]}}
        for i, t in enumerate(texts)
    ]


@pytest.mark.req("TER-TST-009")
def test_a_repeated_tool_result_completes_the_tool_once(tmp_path: Path) -> None:
    trace = _read(tmp_path, PROMPT, CALL, RESULT, RESULT)
    kinds = [e.kind for e in trace.events]
    assert kinds.count(EventKind.TOOL_COMPLETED) == 1


@pytest.mark.req("TER-TST-009")
def test_sibling_lines_of_one_response_keep_every_block(tmp_path: Path) -> None:
    trace = _read(tmp_path, PROMPT, *_texts("First", "Second"))
    texts = [e.text for e in trace.events if e.kind is not EventKind.PROMPT]
    assert texts == ["First", "Second"]
    assert len({e.id for e in trace.events}) == len(trace.events)
    assert [e.provenance.record_id for e in trace.events[1:]] == ["a1", "a1-1"]


@pytest.mark.req("TER-TST-009")
def test_different_records_reusing_a_uuid_keep_their_own_events(
    tmp_path: Path,
) -> None:
    second = _with(PROMPT, content=[{"type": "text", "text": "And the tests"}])
    trace = _read(tmp_path, PROMPT, second)
    assert [e.text for e in trace.events] == ["Fix it", "And the tests"]
    assert len({e.id for e in trace.events}) == 2


@pytest.mark.req("TER-TST-009")
@pytest.mark.parametrize(
    "broken",
    [
        PROMPT | {"message": None},
        PROMPT | {"message": "Fix it"},
        _with(PROMPT, role=None),
        _with(PROMPT, content=7),
        _with(PROMPT, content=[{"type": 3, "text": ["x"]}]),
        PROMPT | {"uuid": 5, "sessionId": [], "timestamp": 0},
        PROMPT | {"_source_lines": "x", "_merge_warnings": 1},
    ],
    ids=[
        "null-message",
        "text-message",
        "null-role",
        "number-content",
        "mistyped-block",
        "mistyped-ids",
        "loader-keys-in-file",
    ],
)
def test_a_record_of_the_wrong_shape_is_read_around(
    tmp_path: Path, broken: dict[str, Any]
) -> None:
    trace = _read(tmp_path, broken, CALL)
    assert EventKind.TOOL_REQUESTED in [e.kind for e in trace.events]


@pytest.mark.req("TER-TST-009")
def test_mistyped_tool_input_and_usage_are_dropped(tmp_path: Path) -> None:
    block = CALL["message"]["content"][0] | {"input": '{"file_path": "a"}'}
    call = _with(
        CALL, content=[block], usage={"input_tokens": "10", "output_tokens": 5}
    )
    [event] = _read(tmp_path, call).events
    assert event.tool is not None and dict(event.tool.arguments) == {}
    assert event.usage is not None and event.usage.input_tokens == 0
    assert event.usage.output_tokens == 5


@pytest.mark.req("TER-TST-013")
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("tool", "call_id"), ["t1"]),
        (("text",), {"a": 1}),
        (("sequence",), "3"),
        (("sequence",), True),
        (("provenance", "lines"), "3"),
        (("usage", "input_tokens"), 1.5),
        (("tool", "arguments"), [["a", 1]]),
    ],
    ids=lambda v: ".".join(v) if isinstance(v, tuple) else None,
)
def test_an_event_record_of_the_wrong_type_is_skipped(
    tmp_path: Path, path: tuple[str, ...], value: Any
) -> None:
    trace = _read(tmp_path, CALL)
    good = trace.events[0]
    record = event_to_record(good)
    target = record
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    log = JsonlEventLog(tmp_path / "log")
    log.append(good)
    with log.path_for(good.session_id).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    reader = JsonlEventLog(tmp_path / "log")
    assert reader.events(good.session_id) == (good,)
    assert reader.skipped == 1


GARE_RUN = Path(__file__).resolve().parents[1] / "fixtures" / "gare" / "failover-run"


@pytest.mark.req("TER-TST-009")
@pytest.mark.parametrize("field", ["candidates", "errors"])
def test_a_gare_list_of_the_wrong_type_is_read_around(
    tmp_path: Path, field: str
) -> None:
    explain = json.loads((GARE_RUN / "explain.json").read_text(encoding="utf-8"))
    if field == "errors":
        explain["errors"] = 7
    else:
        for row in explain["events"]:
            if isinstance(row.get("detail"), dict) and "candidates" in row["detail"]:
                row["detail"]["candidates"] = 7
    (tmp_path / "explain.json").write_text(json.dumps(explain), encoding="utf-8")
    usage = (GARE_RUN / "gare-ter-usage.jsonl").read_text(encoding="utf-8")
    (tmp_path / "gare-ter-usage.jsonl").write_text(usage, encoding="utf-8")
    assert GareRunSource().read(tmp_path).events


@pytest.mark.req("TER-TST-007")
def test_a_gare_usage_file_that_is_not_utf8_is_a_format_error(tmp_path: Path) -> None:
    """Found by the conformance kit: a .jsonl file of bytes that are not UTF-8
    raised UnicodeDecodeError instead of SessionFormatError."""
    from ter.domain import SessionFormatError

    path = tmp_path / "not-utf8.jsonl"
    path.write_bytes(b"\xff\xfe\x00\x00{}\n")
    with pytest.raises(SessionFormatError, match="is not UTF-8"):
        GareRunSource().read(path)


@pytest.mark.req("TER-TST-009")
def test_a_gare_usage_row_holding_a_line_separator_is_one_row(tmp_path: Path) -> None:
    """Rows split at newlines only: U+2028 inside a JSON string is data."""
    rows = (GARE_RUN / "gare-ter-usage.jsonl").read_text(encoding="utf-8")
    first, rest = rows.split("\n", 1)
    path = tmp_path / "gare-ter-usage.jsonl"

    def read(note: str) -> Any:
        row = json.loads(first)
        row["note"] = note
        text = json.dumps(row, ensure_ascii=False) + "\n" + rest
        path.write_text(text, encoding="utf-8")
        return GareRunSource().read(path)

    plain = read("ab")
    assert plain.coverage == read("a\u2028b").coverage
    assert len(plain.events) == len(read("a\u2028b").events)


@pytest.mark.req("TER-TST-009")
def test_a_gare_usage_file_separated_by_carriage_returns_keeps_its_rows(
    tmp_path: Path,
) -> None:
    """Review of #118: rows split at \\r alone (universal newlines) as well."""
    rows = (GARE_RUN / "gare-ter-usage.jsonl").read_text(encoding="utf-8")
    unix = tmp_path / "unix" / "gare-ter-usage.jsonl"
    old_mac = tmp_path / "mac" / "gare-ter-usage.jsonl"
    for path, text in ((unix, rows), (old_mac, rows.replace("\n", "\r"))):
        path.parent.mkdir()
        path.write_bytes(text.encode("utf-8"))
    expected = GareRunSource().read(unix)
    found = GareRunSource().read(old_mac)
    assert len(found.events) == len(expected.events) > 0
    assert found.coverage == expected.coverage
