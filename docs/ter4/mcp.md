# The MCP server

`python -m ter mcp` serves TER's analysis of **ended** sessions to any
[Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18)
client over stdio: Claude Code, an IDE, or another agent harness. It is
Stage U3 of ADR 0008 (point P210, TER-INT-022 to TER-INT-026).

```bash
python -m ter mcp --root ~/.claude/projects
```

To add it to Claude Code (user scope, so every project can ask about past
sessions):

```bash
claude mcp add ter --scope user -- python -m ter mcp --root ~/.claude/projects
```

## Tools

Every tool is read-only. Each is annotated `readOnlyHint`, `idempotentHint`
and closed-world, and changes no file (TER-INT-024). Each session tool
returns the same document as the matching command, as JSON text and as
`structuredContent`.

| Tool | Arguments | Returns | Same as |
|---|---|---|---|
| `list_sessions` | none | ended sessions under the roots, newest first (at most 200), with counts of ended, running and unlisted sessions | |
| `explain_session` | `path` | the L2 findings with evidence ids and confidence rules, and the scorecard | `python -m ter explain --json` |
| `session_a3` | `path` | the A3 as JSON | `python -m ter a3 --json -` |
| `session_record` | `path` | the content-free `ter.record/0.1` | `python -m ter record export` |

A `path` is absolute, or relative to the first root that holds it. Files
ending in `.jsonl` or `.json` are offered, so any session source TER reads
works: Claude Code transcripts, GARE run exports and OpenTelemetry span
exports. A GARE run given as a folder is not offered; name its files.

## What it refuses

- **A session that is still running** (TER-INT-022). A session counts as
  ended once its file has not changed for the quiet period: 600 seconds by
  default, set with `--quiet-seconds`. While TER's maturity ceiling is below
  L5 Corrective, advice may change no agent input (TER-INT-002). An agent
  that asks about its own running session would take TER's analysis as
  input, so every tool refuses a running session with a tool error.
  `list_sessions` counts running sessions without naming them.
- **A path outside the roots** (TER-INT-023). The person who starts the
  server chooses what a client may read with `--root`, which can be
  repeated. A path that resolves outside every root, including through a
  symlink, is refused before it is read, and is not listed. A session file
  that is replaced by a link, or by another file, while it is read is
  refused after the read ("changed while it was read"). That narrows the
  window but does not close it: the session readers open the file, and a
  Claude Code session's `subagents` folder, by name. Whoever can write
  inside a root can already change what the server reads, so keep roots to
  folders only you write to.
- **A session whose age cannot be read.** It is refused, because the quiet
  period cannot be checked.
- **A malformed message** (TER-INT-026). A line that is not JSON, a batch,
  or a request without a method gets a JSON-RPC error, and the server keeps
  serving. A failure inside a tool is an internal error for that request
  only.

## Protocol

The server speaks newline-delimited JSON-RPC 2.0 on stdin and stdout from
the published specification. It uses no SDK (ADR 0008 decision 1). It
supports these methods: `initialize`, `ping`, `tools/list` and
`tools/call`. Notifications get no answer.

Pinned protocol versions, newest first: `2025-06-18`, `2025-03-26`,
`2024-11-05`. A client that asks for one of these gets it. A client that
asks for any other version is answered with `2025-06-18`, and the client
decides whether to carry on (TER-INT-025). Supporting a new protocol
version means adding it to `PROTOCOL_VERSIONS` in
`src/ter/adapters/driving/mcp_server.py` together with tests. Never infer a
version.

Only the protocol goes to stdout. The one line for a person ("serving N
root(s)") goes to stderr.

## What not to claim

- The quiet period is a heuristic. A session that pauses for longer than
  the period and then resumes was read while it was paused. Set
  `--quiet-seconds` above the longest pause you expect.
- The server analyses sessions on demand, with TER offline scoring by
  default (`--ter`). A long session takes the same few seconds that
  `python -m ter explain` takes.
