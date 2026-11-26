"""The read-only MCP server (Stage U3, ADR 0008 decision 6, P210)."""

from __future__ import annotations

import io
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ter.adapters.driving.mcp_server import (
    PROTOCOL_VERSIONS,
    TOOLS,
    McpServer,
    McpServices,
    serve,
)
from ter.domain.maturity import Maturity

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
QUIET = 600


def _services(calls: list[tuple[str, Path]] | None = None) -> McpServices:
    seen = calls if calls is not None else []

    def tool(name: str) -> Any:
        def run(path: Path) -> dict[str, Any]:
            seen.append((name, path))
            if path.name == "broken.jsonl":
                raise ValueError("line 1 is not JSON")
            return {"tool": name, "session": path.name}

        return run

    return McpServices(tool("explain"), tool("a3"), tool("record"), version="9.9")


def _session(folder: Path, name: str, age: float) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    stamp = (NOW - timedelta(seconds=age)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def _server(
    roots: list[Path], calls: list[tuple[str, Path]] | None = None, **kw: Any
) -> McpServer:
    return McpServer(
        _services(calls), roots, quiet_seconds=QUIET, clock=lambda: NOW, **kw
    )


def _call(server: McpServer, name: str, arguments: Any = None, ident: int = 7) -> Any:
    params: dict[str, Any] = {"name": name}
    if arguments is not None:
        params["arguments"] = arguments
    response = server.handle(
        {"jsonrpc": "2.0", "id": ident, "method": "tools/call", "params": params}
    )
    assert response is not None and response["id"] == ident
    return response


def _result(response: Any) -> Any:
    assert "result" in response, response
    return response["result"]


# -- the protocol ------------------------------------------------------------


@pytest.mark.req("TER-INT-025")
def test_initialize_answers_the_version_the_client_asked_for_when_listed(
    tmp_path: Path,
) -> None:
    for version in PROTOCOL_VERSIONS:
        response = _server([tmp_path]).handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": version, "capabilities": {}},
            }
        )
        assert response is not None
        result = response["result"]
        assert result["protocolVersion"] == version
        assert result["capabilities"] == {"tools": {"listChanged": False}}
        assert result["serverInfo"]["name"] == "ter"
        assert result["serverInfo"]["version"] == "9.9"


@pytest.mark.req("TER-INT-025")
@pytest.mark.parametrize("asked", ["2099-01-01", "", None, 3])
def test_initialize_answers_the_newest_listed_version_for_any_other(
    tmp_path: Path, asked: object
) -> None:
    response = _server([tmp_path]).handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": asked},
        }
    )
    assert response is not None
    assert response["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0] == "2025-06-18"


@pytest.mark.req("TER-INT-026")
def test_ping_answers_an_empty_result(tmp_path: Path) -> None:
    assert _server([tmp_path]).handle(
        {"jsonrpc": "2.0", "id": "a", "method": "ping"}
    ) == {
        "jsonrpc": "2.0",
        "id": "a",
        "result": {},
    }


@pytest.mark.req("TER-INT-026")
def test_notifications_get_no_answer(tmp_path: Path) -> None:
    server = _server([tmp_path])
    assert (
        server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    )
    assert server.handle({"jsonrpc": "2.0", "method": "no/such"}) is None
    assert server.handle({"jsonrpc": "2.0", "method": "x", "params": [1]}) is None


@pytest.mark.req("TER-INT-026")
def test_a_response_from_the_client_gets_no_answer(tmp_path: Path) -> None:
    server = _server([tmp_path])
    assert server.handle({"jsonrpc": "2.0", "id": 3, "result": {}}) is None
    assert server.handle({"jsonrpc": "2.0", "id": 3, "error": {"code": 1}}) is None


@pytest.mark.req("TER-INT-026")
@pytest.mark.parametrize(
    ("message", "code", "ident"),
    [
        ([{"jsonrpc": "2.0", "id": 1, "method": "ping"}], -32600, None),
        ("ping", -32600, None),
        (None, -32600, None),
        ({"id": 1, "method": "ping"}, -32600, 1),
        ({"jsonrpc": "1.0", "id": 1, "method": "ping"}, -32600, 1),
        ({"jsonrpc": "2.0", "id": 1}, -32600, 1),
        ({"jsonrpc": "2.0", "id": 1, "method": 5}, -32600, 1),
        ({"jsonrpc": "2.0", "id": True, "method": "ping"}, -32600, None),
        ({"jsonrpc": "2.0", "id": 1.5, "method": "ping"}, -32600, None),
        ({"jsonrpc": "2.0", "id": [1], "method": "ping"}, -32600, None),
        ({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": [1]}, -32602, 1),
        ({"jsonrpc": "2.0", "id": 1, "method": "resources/list"}, -32601, 1),
        ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}}, -32602, 1),
        (
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "rm"},
            },
            -32602,
            1,
        ),
        (
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "list_sessions", "arguments": [1]},
            },
            -32602,
            1,
        ),
    ],
)
def test_a_malformed_message_gets_a_json_rpc_error(
    tmp_path: Path, message: object, code: int, ident: object
) -> None:
    response = _server([tmp_path]).handle(message)
    assert response is not None
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == ident
    assert response["error"]["code"] == code
    assert response["error"]["message"]
    assert "result" not in response


@pytest.mark.req("TER-INT-026")
def test_a_line_that_is_not_json_gets_a_parse_error(tmp_path: Path) -> None:
    response = _server([tmp_path]).handle_line("{not json")
    assert response == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "not JSON"},
    }


@pytest.mark.req("TER-INT-026")
def test_an_unexpected_failure_is_an_internal_error_and_the_server_carries_on(
    tmp_path: Path,
) -> None:
    _session(tmp_path, "s.jsonl", age=QUIET * 2)

    def boom(path: Path) -> dict[str, Any]:
        raise RuntimeError("bug")

    server = McpServer(
        McpServices(boom, boom, boom),
        [tmp_path],
        quiet_seconds=QUIET,
        clock=lambda: NOW,
    )
    response = _call(server, "explain_session", {"path": "s.jsonl"})
    assert response["error"]["code"] == -32603
    assert "RuntimeError: bug" in response["error"]["message"]
    assert _result(_call(server, "list_sessions"))["isError"] is False


@pytest.mark.req("TER-INT-026")
def test_serve_answers_line_by_line_and_keeps_serving_after_bad_lines(
    tmp_path: Path,
) -> None:
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        "",
        "garbage",
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    ]
    out = io.StringIO()
    assert serve(_server([tmp_path]), io.StringIO("\n".join(lines) + "\n"), out) == 0
    answers = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [a.get("id") for a in answers] == [1, None, 2]
    assert answers[1]["error"]["code"] == -32700
    assert [t["name"] for t in answers[2]["result"]["tools"]] == [
        "list_sessions",
        "explain_session",
        "session_a3",
        "session_record",
    ]


# -- the tools ---------------------------------------------------------------


@pytest.mark.req("TER-INT-024")
def test_every_tool_is_annotated_read_only_and_closed_world(tmp_path: Path) -> None:
    response = _server([tmp_path]).handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
    assert response is not None
    tools = response["result"]["tools"]
    assert len(tools) == len(TOOLS) == 4
    for tool in tools:
        assert tool["annotations"] == {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
        assert tool["inputSchema"]["type"] == "object"
        assert tool["inputSchema"]["additionalProperties"] is False
        assert tool["description"]


@pytest.mark.req("TER-INT-024")
@pytest.mark.parametrize(
    ("tool", "service"),
    [
        ("explain_session", "explain"),
        ("session_a3", "a3"),
        ("session_record", "record"),
    ],
)
def test_a_session_tool_returns_its_service_document_as_text_and_structure(
    tmp_path: Path, tool: str, service: str
) -> None:
    path = _session(tmp_path, "done.jsonl", age=QUIET)
    calls: list[tuple[str, Path]] = []
    result = _result(_call(_server([tmp_path], calls), tool, {"path": str(path)}))
    document = {"tool": service, "session": "done.jsonl"}
    assert result["isError"] is False
    assert result["structuredContent"] == document
    assert json.loads(result["content"][0]["text"]) == document
    assert result["content"][0]["type"] == "text"
    assert calls == [(service, path.resolve())]


@pytest.mark.req("TER-INT-024")
def test_the_tools_change_no_file(tmp_path: Path) -> None:
    _session(tmp_path, "a/one.jsonl", age=QUIET * 3)
    _session(tmp_path, "two.json", age=QUIET * 2)
    before = {
        p: (p.read_bytes(), p.stat().st_mtime)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    server = _server([tmp_path])
    _call(server, "list_sessions")
    for tool in ("explain_session", "session_a3", "session_record"):
        _call(server, tool, {"path": "a/one.jsonl"})
    after = {
        p: (p.read_bytes(), p.stat().st_mtime)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_a_relative_path_is_read_from_the_root_that_holds_it(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    _session(second, "s.jsonl", age=QUIET)
    calls: list[tuple[str, Path]] = []
    _result(_call(_server([first, second], calls), "session_a3", {"path": "s.jsonl"}))
    assert calls == [("a3", (second / "s.jsonl").resolve())]


def test_a_file_root_reads_its_own_file_by_name(tmp_path: Path) -> None:
    path = _session(tmp_path, "only.jsonl", age=QUIET)
    _session(tmp_path, "other.jsonl", age=QUIET)
    server = _server([path])
    assert (
        _result(_call(server, "session_record", {"path": "only.jsonl"}))["isError"]
        is False
    )
    refused = _result(_call(server, "session_record", {"path": "other.jsonl"}))
    assert refused["isError"] is True
    assert "outside the server's roots" in refused["content"][0]["text"]


@pytest.mark.parametrize("path", [None, "", "  ", 3, ["a"]])
def test_a_tool_without_a_path_is_refused(tmp_path: Path, path: object) -> None:
    arguments = {} if path is None else {"path": path}
    result = _result(_call(_server([tmp_path]), "explain_session", arguments))
    assert result == {
        "content": [{"type": "text", "text": "path must name a session file"}],
        "isError": True,
    }


def test_a_missing_session_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "folder").mkdir()
    for name in ("gone.jsonl", "folder"):
        result = _result(_call(_server([tmp_path]), "explain_session", {"path": name}))
        assert result["isError"] is True
        assert result["content"][0]["text"] == f"no session file {name}"


def test_a_session_its_reader_refuses_is_a_tool_error(tmp_path: Path) -> None:
    _session(tmp_path, "broken.jsonl", age=QUIET)
    result = _result(
        _call(_server([tmp_path]), "explain_session", {"path": "broken.jsonl"})
    )
    assert result["isError"] is True
    assert result["content"][0]["text"] == "cannot read the session: line 1 is not JSON"


# -- refusals ----------------------------------------------------------------


@pytest.mark.req("TER-INT-023")
@pytest.mark.parametrize("escape", ["../outside.jsonl", "link.jsonl"])
def test_a_path_outside_the_roots_is_refused_unread(
    tmp_path: Path, escape: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = _session(tmp_path, "outside.jsonl", age=QUIET * 2)
    (root / "link.jsonl").symlink_to(outside)
    calls: list[tuple[str, Path]] = []
    server = _server([root], calls)
    for path in (escape, str(outside)):
        result = _result(_call(server, "session_record", {"path": path}))
        assert result["isError"] is True
        assert result["content"][0]["text"] == f"{path} is outside the server's roots"
    assert calls == []


@pytest.mark.req("TER-INT-023")
def test_listing_names_only_sessions_under_the_roots(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _session(root, "in.jsonl", age=QUIET)
    _session(tmp_path, "out.jsonl", age=QUIET)
    listed = _result(_call(_server([root]), "list_sessions"))["structuredContent"]
    assert [Path(s["path"]).name for s in listed["sessions"]] == ["in.jsonl"]


@pytest.mark.req("TER-INT-022")
@pytest.mark.parametrize("tool", ["explain_session", "session_a3", "session_record"])
def test_a_session_changed_within_the_quiet_period_is_refused(
    tmp_path: Path, tool: str
) -> None:
    _session(tmp_path, "live.jsonl", age=QUIET - 1)
    calls: list[tuple[str, Path]] = []
    result = _result(_call(_server([tmp_path], calls), tool, {"path": "live.jsonl"}))
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert text.startswith("live.jsonl changed 599 s ago, so it may still be running")
    assert "below L5" in text
    assert calls == []


@pytest.mark.req("TER-INT-022")
def test_a_session_quiet_for_exactly_the_period_has_ended(tmp_path: Path) -> None:
    _session(tmp_path, "done.jsonl", age=QUIET)
    result = _result(
        _call(_server([tmp_path]), "session_record", {"path": "done.jsonl"})
    )
    assert result["isError"] is False


@pytest.mark.req("TER-INT-022")
def test_a_file_from_the_future_counts_as_running(tmp_path: Path) -> None:
    _session(tmp_path, "skewed.jsonl", age=-60)
    result = _result(
        _call(_server([tmp_path]), "session_record", {"path": "skewed.jsonl"})
    )
    assert result["isError"] is True
    assert "changed 0 s ago" in result["content"][0]["text"]


@pytest.mark.req("TER-INT-022")
def test_listing_counts_running_sessions_without_naming_them(tmp_path: Path) -> None:
    _session(tmp_path, "old.jsonl", age=QUIET * 10)
    _session(tmp_path, "sub/newer.json", age=QUIET + 5)
    _session(tmp_path, "live.jsonl", age=10)
    _session(tmp_path, "notes.txt", age=QUIET * 10)
    listed = _result(_call(_server([tmp_path, tmp_path / "sub"]), "list_sessions"))
    assert listed["isError"] is False
    document = listed["structuredContent"]
    assert document == {
        "sessions": [
            {
                "path": str((tmp_path / "sub/newer.json").resolve()),
                "quiet_seconds": QUIET + 5,
            },
            {
                "path": str((tmp_path / "old.jsonl").resolve()),
                "quiet_seconds": QUIET * 10,
            },
        ],
        "ended": 2,
        "running": 1,
        "not_listed": 0,
    }


def test_listing_names_at_most_two_hundred_sessions(tmp_path: Path) -> None:
    for n in range(203):
        _session(tmp_path, f"s{n:03}.jsonl", age=QUIET + n)
    document = _result(_call(_server([tmp_path]), "list_sessions"))["structuredContent"]
    assert len(document["sessions"]) == 200
    assert document["ended"] == 203
    assert document["not_listed"] == 3
    assert Path(document["sessions"][0]["path"]).name == "s000.jsonl"


@pytest.mark.req("TER-INT-022")
def test_at_the_corrective_ceiling_a_running_session_is_answered(
    tmp_path: Path,
) -> None:
    _session(tmp_path, "live.jsonl", age=1)
    server = _server([tmp_path], ceiling=Maturity.CORRECTIVE)
    result = _result(_call(server, "session_record", {"path": "live.jsonl"}))
    assert result["isError"] is False


# -- review of #118 ------------------------------------------------------------


@pytest.mark.req("TER-INT-026")
def test_a_line_nested_too_deep_for_the_decoder_gets_a_parse_error(
    tmp_path: Path,
) -> None:
    response = _server([tmp_path]).handle_line("[" * 100_000)
    assert response is not None and response["error"]["code"] == -32700


@pytest.mark.req("TER-INT-026")
def test_serve_answers_a_failure_of_its_handler_and_keeps_serving(
    tmp_path: Path,
) -> None:
    class Flaky(McpServer):
        def handle_line(self, line: str) -> dict[str, Any] | None:
            if "boom" in line:
                raise MemoryError("boom")
            return super().handle_line(line)

    server = Flaky(_services(), [tmp_path], quiet_seconds=QUIET, clock=lambda: NOW)
    ping = json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"})
    out = io.StringIO()
    serve(server, io.StringIO(f"boom\n{ping}\n"), out)
    answers = [json.loads(line) for line in out.getvalue().splitlines()]
    assert answers[0]["error"]["code"] == -32603
    assert answers[1] == {"jsonrpc": "2.0", "id": 3, "result": {}}


@pytest.mark.req("TER-INT-022")
def test_a_session_whose_age_cannot_be_read_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(tmp_path, "s.jsonl", age=QUIET * 2)
    server = _server([tmp_path])
    monkeypatch.setattr(McpServer, "_age", lambda self, path: None)
    result = _result(_call(server, "session_record", {"path": "s.jsonl"}))
    assert result["isError"] is True
    assert result["content"][0]["text"] == "cannot tell when s.jsonl last changed"


@pytest.mark.req("TER-INT-023")
def test_listing_leaves_out_a_link_that_leads_outside_the_roots(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _session(root, "in.jsonl", age=QUIET)
    outside = _session(tmp_path, "outside.jsonl", age=QUIET)
    (root / "link.jsonl").symlink_to(outside)
    listed = _result(_call(_server([root]), "list_sessions"))["structuredContent"]
    assert [Path(s["path"]).name for s in listed["sessions"]] == ["in.jsonl"]
    assert listed["ended"] == 1


@pytest.mark.req("TER-INT-023")
def test_a_session_swapped_for_a_link_while_it_is_read_is_refused(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    session = _session(root, "s.jsonl", age=QUIET * 2)
    outside = _session(tmp_path, "secret.jsonl", age=QUIET * 2)

    def swap(path: Path) -> dict[str, Any]:
        session.unlink()
        session.symlink_to(outside)
        return {"read": path.read_text(encoding="utf-8")}

    server = McpServer(
        McpServices(swap, swap, swap), [root], quiet_seconds=QUIET, clock=lambda: NOW
    )
    result = _result(_call(server, "session_record", {"path": "s.jsonl"}))
    assert result["isError"] is True
    assert (
        result["content"][0]["text"] == "s.jsonl changed while it was read; ask again"
    )


@pytest.mark.req("TER-INT-026")
def test_standard_input_is_read_as_utf8_and_a_bad_byte_is_a_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ter.adapters.driving.cli import _utf8_stdin

    ping = json.dumps({"jsonrpc": "2.0", "id": "é", "method": "ping"})
    raw = io.BytesIO(b"\xff\xfe{\n" + ping.encode("utf-8") + b"\n")
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(raw, encoding="latin-1"))
    out = io.StringIO()
    serve(_server([tmp_path]), _utf8_stdin(), out)
    answers = [json.loads(line) for line in out.getvalue().splitlines()]
    assert answers[0]["error"]["code"] == -32700
    assert answers[1] == {"jsonrpc": "2.0", "id": "é", "result": {}}
