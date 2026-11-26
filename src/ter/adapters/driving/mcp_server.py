"""``python -m ter mcp``: TER's read-only analysis as a Model Context Protocol
server over stdio (Stage U3, ADR 0008 decision 6, TER-INT-022 to TER-INT-026).

An MCP client (Claude Code, an IDE, another agent harness) starts the server
and asks it about sessions that have ended: the L2 findings, the A3 and the
TER Record, the same documents ``explain --json``, ``a3 --json`` and
``record export`` write. The protocol is spoken from the published
specification by pinned version, not through an SDK (ADR 0008 decision 1):
newline-delimited JSON-RPC 2.0 on stdin and stdout, with ``initialize``,
``ping``, ``tools/list`` and ``tools/call``.

Two refusals keep the server inside the L4 ceiling and the machine's
privacy:

- a session whose file changed within the quiet period is still running,
  and while the ceiling is below L5 its analysis would become input to the
  agent that is producing it (TER-INT-002), so every tool refuses it
  (TER-INT-022);
- a path outside the roots the server was started with is refused before it
  is read (TER-INT-023), so a client cannot point TER at any file the user
  can read.

Every tool only reads (TER-INT-024). A malformed message gets a JSON-RPC
error and the server keeps serving (TER-INT-026).
"""

from __future__ import annotations

import heapq
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from ...domain.maturity import Maturity

__all__ = [
    "PROTOCOL_VERSIONS",
    "SESSION_SUFFIXES",
    "McpServer",
    "McpServices",
    "ToolRefusal",
    "serve",
]

#: The MCP protocol versions this server speaks, newest first. A client that
#: asks for another one is answered with the newest (TER-INT-025).
PROTOCOL_VERSIONS: tuple[str, ...] = ("2025-06-18", "2025-03-26", "2024-11-05")
#: Files the server offers as sessions.
SESSION_SUFFIXES: tuple[str, ...] = (".jsonl", ".json")
#: How long a session's file must be unchanged before it counts as ended.
DEFAULT_QUIET_SECONDS = 600

_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603

#: Sessions listed before the rest are counted.
_LISTED = 200


class ToolRefusal(Exception):
    """A tool call the server will not answer, with the reason the client
    is shown."""


@dataclass(frozen=True)
class McpServices:
    """The read-only use cases behind the tools, wired by the composition
    root. Each takes a session path and returns the document the matching
    command writes; each may raise ``ValueError`` (a session it cannot
    read)."""

    explain: Callable[[Path], dict[str, Any]]
    a3: Callable[[Path], dict[str, Any]]
    record: Callable[[Path], dict[str, Any]]
    version: str = "0"


_PATH = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "the session file, as list_sessions names it",
        }
    },
    "required": ["path"],
    "additionalProperties": False,
}
_READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "list_sessions",
        "title": "List ended sessions",
        "description": (
            "Sessions under the server's roots whose files have been quiet "
            "long enough to count as ended, newest first. Running sessions "
            "are counted, not listed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "annotations": _READ_ONLY,
    },
    {
        "name": "explain_session",
        "title": "Explain a session",
        "description": (
            "The L2 Lean analysis of an ended session: waste findings with "
            "their evidence event ids and confidence rules, and the scorecard "
            "(as `python -m ter explain --json`)."
        ),
        "inputSchema": _PATH,
        "annotations": _READ_ONLY,
    },
    {
        "name": "session_a3",
        "title": "A3 of a session",
        "description": (
            "The one-page Lean A3 of an ended session as JSON: value stream, "
            "waste Pareto, root causes, countermeasures and follow-up "
            "measures (as `python -m ter a3 --json`)."
        ),
        "inputSchema": _PATH,
        "annotations": _READ_ONLY,
    },
    {
        "name": "session_record",
        "title": "TER Record of a session",
        "description": (
            "The content-free ter.record/0.1 of an ended session: counts, "
            "classes, cost and digests, no prompt or file content (as "
            "`python -m ter record export`)."
        ),
        "inputSchema": _PATH,
        "annotations": _READ_ONLY,
    },
)


@dataclass
class McpServer:
    """One MCP conversation: :meth:`handle` takes a parsed message and
    returns the response to write, or None for a notification."""

    services: McpServices
    roots: Sequence[Path]
    quiet_seconds: float = DEFAULT_QUIET_SECONDS
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    ceiling: Maturity = Maturity.ADVISORY

    def __post_init__(self) -> None:
        self.roots = tuple(Path(r).expanduser().resolve() for r in self.roots)

    # -- JSON-RPC ---------------------------------------------------------

    def handle_line(self, line: str) -> dict[str, Any] | None:
        """The response to one line of the stream, or None."""
        try:
            message = json.loads(line)
        except (ValueError, RecursionError):
            # Nesting too deep for the decoder is not JSON TER can read.
            return _error(None, _PARSE_ERROR, "not JSON")
        return self.handle(message)

    def handle(self, message: object) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(
                _id_of(message), _INVALID_REQUEST, "not a JSON-RPC 2.0 message"
            )
        method = message.get("method")
        has_id = "id" in message
        ident = message.get("id")
        if not isinstance(method, str):
            if has_id and ("result" in message or "error" in message):
                return None  # a response to a request this server never sends
            return _error(_id_of(message), _INVALID_REQUEST, "no method")
        if has_id and (isinstance(ident, bool) or not isinstance(ident, str | int)):
            return _error(None, _INVALID_REQUEST, "id must be a string or an integer")
        params = message.get("params", {})
        if not isinstance(params, dict):
            return (
                _error(ident, _INVALID_PARAMS, "params must be an object")
                if has_id
                else None
            )
        if not has_id:
            return None  # notifications (initialized, cancelled) need no answer
        try:
            result = self._dispatch(method, params)
        except _RpcError as exc:
            return _error(ident, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 - one bad request must not end the server
            return _error(ident, _INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        return {"jsonrpc": "2.0", "id": ident, "result": result}

    def _dispatch(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            asked = params.get("protocolVersion")
            version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {
                    "name": "ter",
                    "title": "TER",
                    "version": self.services.version,
                },
                "instructions": (
                    "TER explains ended coding-agent sessions as Lean analysis. "
                    "Call list_sessions, then explain_session, session_a3 or "
                    "session_record with a path it lists. Running sessions are "
                    "refused while TER's maturity ceiling is below L5."
                ),
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [dict(tool) for tool in TOOLS]}
        if method == "tools/call":
            return self._call(params)
        raise _RpcError(_METHOD_NOT_FOUND, f"no method {method}")

    # -- tools ------------------------------------------------------------

    def _call(self, params: Mapping[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise _RpcError(_INVALID_PARAMS, "arguments must be an object")
        tools = {
            "list_sessions": lambda: self._list(),
            "explain_session": lambda: self._read(self.services.explain, arguments),
            "session_a3": lambda: self._read(self.services.a3, arguments),
            "session_record": lambda: self._read(self.services.record, arguments),
        }
        if not isinstance(name, str) or name not in tools:
            raise _RpcError(_INVALID_PARAMS, f"no tool {name!r}")
        try:
            document = tools[name]()
        except ToolRefusal as exc:
            return _tool_error(str(exc))
        except (OSError, ValueError) as exc:
            # Session readers refuse bad input with a ValueError subclass.
            return _tool_error(f"cannot read the session: {exc}")
        return {
            "content": [
                {"type": "text", "text": json.dumps(document, ensure_ascii=False)}
            ],
            "structuredContent": document,
            "isError": False,
        }

    def _read(
        self,
        tool: Callable[[Path], dict[str, Any]],
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """``tool`` on the session ``arguments`` name, refused when the file
        checked is not the file still there once it was read."""
        path = self._session(arguments)
        before = _identity(path)
        document = tool(path)
        if before is None or _identity(path) != before:
            raise ToolRefusal(f"{path.name} changed while it was read; ask again")
        return document

    def _session(self, arguments: Mapping[str, Any]) -> Path:
        raw = arguments.get("path")
        if not isinstance(raw, str) or not raw.strip():
            raise ToolRefusal("path must name a session file")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            # Relative to the first root that holds it (a file root: its folder).
            bases = [root.parent if root.is_file() else root for root in self.roots]
            path = next(
                (b / path for b in bases if (b / path).exists()), bases[0] / path
            )
        try:
            path = path.resolve()
        except (OSError, RuntimeError) as exc:
            raise ToolRefusal(f"cannot resolve {raw}: {exc}") from None
        if not any(path.is_relative_to(root) for root in self.roots):
            raise ToolRefusal(f"{raw} is outside the server's roots")
        if not path.is_file():
            raise ToolRefusal(f"no session file {raw}")
        if self._age(path) is None:
            raise ToolRefusal(f"cannot tell when {raw} last changed")
        running = self._running(path)
        if running is not None:
            raise ToolRefusal(running)
        return path

    def _running(self, path: Path) -> str | None:
        """Why ``path`` counts as a running session, or None when it ended.

        While the ceiling is below L5, analysis must not reach the agent
        that is producing the session (TER-INT-022)."""
        if self.ceiling >= Maturity.CORRECTIVE:
            return None
        age = self._age(path)
        if age is None or age >= self.quiet_seconds:
            return None
        return (
            f"{path.name} changed {age:.0f} s ago, so it may still be running; "
            f"TER answers only for sessions quiet for {self.quiet_seconds:.0f} s "
            "while its maturity ceiling is below L5"
        )

    def _age(self, path: Path) -> float | None:
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        except OSError:
            return None
        return max(0.0, (self.clock() - modified).total_seconds())

    def _list(self) -> dict[str, Any]:
        counts = {"ended": 0, "running": 0}

        def ended() -> Iterable[tuple[float, str]]:
            for path in _session_files(self.roots):
                age = self._age(path)
                if age is None:
                    continue
                if self._running(path) is not None:
                    counts["running"] += 1
                else:
                    counts["ended"] += 1
                    yield age, str(path)

        # Only the newest sessions listed are held, however large the roots.
        newest = heapq.nsmallest(_LISTED, ended())
        return {
            "sessions": [
                {"path": path, "quiet_seconds": round(age)} for age, path in newest
            ],
            "ended": counts["ended"],
            "running": counts["running"],
            "not_listed": max(0, counts["ended"] - _LISTED),
        }


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _session_files(roots: Sequence[Path]) -> Iterable[Path]:
    """Every session file under ``roots``, walked as it goes, each resolved
    and kept only when it lies inside a root, as a tool call requires."""
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            candidates: Iterable[Path] = [root]
        else:
            candidates = (
                Path(folder) / name
                for folder, _, names in os.walk(root)
                for name in names
            )
        for candidate in candidates:
            if candidate.suffix not in SESSION_SUFFIXES:
                continue
            try:
                path = candidate.resolve()
            except (OSError, RuntimeError):
                continue
            if (
                path not in seen
                and any(path.is_relative_to(r) for r in roots)
                and path.is_file()
            ):
                seen.add(path)
                yield path


def _identity(path: Path) -> tuple[int, int] | None:
    """The file ``path`` names without following a link (device and inode),
    or None when it is a link or gone."""
    try:
        found = path.lstat()
    except OSError:
        return None
    if path.is_symlink():
        return None
    return (found.st_dev, found.st_ino)


def _id_of(message: object) -> str | int | None:
    if isinstance(message, dict):
        ident = message.get("id")
        if isinstance(ident, str | int) and not isinstance(ident, bool):
            return ident
    return None


def _error(ident: str | int | None, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


def _tool_error(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def serve(server: McpServer, stdin: IO[str], stdout: IO[str]) -> int:
    """Answer each line of ``stdin`` on ``stdout`` until it closes."""
    for line in stdin:
        if not line.strip():
            continue
        try:
            response = server.handle_line(line)
        except Exception as exc:  # noqa: BLE001 - one bad message never ends the server
            response = _error(None, _INTERNAL_ERROR, f"internal error: {exc}")
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()
    return 0
