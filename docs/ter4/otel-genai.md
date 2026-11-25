# OpenTelemetry GenAI spans

Any agent instrumented with the
[OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai)
can be read by TER as a session through the `otel-genai` session source
(`ter.adapters.driven.otel_genai`, Stage U2,
[ADR 0008](../decisions/0008-upstream-adapters.md)). TER reads the spans the
agent exported as OTLP JSON. It never imports an OpenTelemetry SDK, so no
extra install is needed.

## Exporting spans

TER reads two layouts:

- one OTLP JSON `ExportTraceServiceRequest` object (`{"resourceSpans": [...]}`)
  in a `.json` file;
- one such object per line, as the OpenTelemetry Collector's `file` exporter
  writes them, in a `.json` or `.jsonl` file.

A file is one session. Export one conversation per file: TER keys the session
by the `gen_ai.conversation.id` its spans carry, and refuses a file holding
several, naming them. Spans with no conversation id are one session only when
they belong to one trace.

```bash
python -m ter observe spans.json --timeline
python -m ter a3 spans.json --html a3.html
python -m ter record export spans.json --out r.json
```

Any command that takes a session takes a span export: TER tells it from a
Claude Code transcript by its `resourceSpans` key.

## Mapping

| Span (`gen_ai.operation.name`) | ter.event kind | Actor |
|---|---|---|
| inference (`chat`, `text_completion`, `generate_content`) | `intent.stated` when its input ends with a user message (content captured) | user |
| | `reasoning`, `response` and one `tool.requested` per tool call it output; the turn's usage on the first | assistant |
| `execute_tool` | `tool.requested` when no inference output named the call, then `tool.completed` | assistant, tool |
| `invoke_agent` under another agent span | `tool.requested` (`agent.handoff`), then `subagent.completed` | assistant, system |
| `invoke_agent` at the top, `create_agent` | documented metadata | |
| any other span, or one that is not a span | counted against coverage as `span:<operation>`, `span:not-gen_ai` or `span:invalid` | |

Tool names map to tool kinds through the same table as Claude Code's, so an
agent that names its tools `Read`, `Edit` or `Bash` gets the same detectors.
Other names are `other`. An `execute_tool` span with `gen_ai.transfer.mode`
(an agent called as a tool) is a handoff.

## Usage

The conventions count cached tokens inside `gen_ai.usage.input_tokens`. TER's
input tokens are the uncached ones, so TER takes
`gen_ai.usage.cache_read.input_tokens` and the cache write count out of it.
The write count is read under both names in use:
`gen_ai.usage.cache_creation.input_tokens` (the attribute registry) and
`gen_ai.usage.cache_write.input_tokens` (the GenAI repository).

The conventions record cache counts only when they are available and
non-zero, so a turn with no cache attribute is not a turn with no cache: it is
read with no cache figures reported, and its cost is marked an estimate, as a
GARE turn's is. The model is `gen_ai.response.model`, else
`gen_ai.request.model`.

## Content capture

Message content is opt-in and off by default: `gen_ai.input.messages`,
`gen_ai.output.messages`, `gen_ai.tool.call.arguments` and
`gen_ai.tool.call.result`. Without it:

- response text is empty, and the trace carries the `no-response-text`
  limit, as a GARE run does;
- TER cannot tell a prompt the developer added from a tool result returned to
  the model, so each trace (one request of the application) is one intent,
  and the trace carries the `intents-from-traces` limit.

Usage, cost, tool kinds and the observed shape are still read. Reports and
Records state each limit beside the figures it qualifies.

## Convention versions

A span names its convention version through the `schemaUrl` of its
instrumentation scope, such as `https://opentelemetry.io/schemas/1.45.0`. A
resource's `schemaUrl` versions the resource attributes, not the GenAI spans,
so TER does not read it as their convention version. TER reads 1.37.0
to 1.45.0, the versions that name the provider `gen_ai.provider.name` and the
token counts `input_tokens` and `output_tokens`. It refuses any other version,
naming it, so an export of an older convention (`gen_ai.system`,
`prompt_tokens`) or a newer one is never read by guess (TER-SRC-028).

The GenAI conventions moved to their own repository in 1.42.0, and that
repository publishes no schema URL yet. A file that names no version is read
under the latest known one and carries the `convention-unstated` limit.

## Checked against the transcript reader

Every golden session is committed as the span export an instrumented agent
would have written (`tests/fixtures/otel/`, written by
`tests/golden/otel_export.py`). Read that way it gives the same events, and
the same TER Record apart from its source name, session id and digests, as
its Claude Code transcript (TER-SRC-031). That is the direction document's
harness-neutral test, run on synthetic sessions. A real agent's export is the
next check.

## Requirements

TER-SRC-027 to TER-SRC-031, vision point P207.
